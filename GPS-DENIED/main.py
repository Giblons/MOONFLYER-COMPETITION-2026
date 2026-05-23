"""
Vision-Based GPS (VGPS) Core Orchestrator.
Spawns and coordinates:
1. CaptureThread: Continuous, low-latency, rectified nadir camera acquisition (60fps).
2. ProcessingThread: Dynamic boundary binarization, ORB feature extraction,
   LSH-FLANN matching against cached MBTiles, Homography, and WGS84 Affine projections.
3. TelemetryThread: Continuous 10-30Hz MAVLink stream to Pixhawk.
Establishes clean OS signal listeners for graceful multi-threaded shutdown.
"""

import time
import signal
import sys
import logging
import threading
import cv2
import numpy as np

import config
from camera import CaptureThread
from vision import preprocess_image, VGPSVisionPipeline
from geospatial import GeospatialDatabase, CoordinateTransformer
from telemetry import TelemetryThread

# Configure root logger for the entire suite
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VGPS.Main")


class ProcessingThread(threading.Thread):
    """
    Asynchronous Core Image Processing and Mathematical Pipeline Thread.
    Consumes rectified camera frames, resolves spatial correspondences against
    pre-loaded MBTiles descriptors, and pushes georeferenced coords to telemetry.
    """
    def __init__(self, capture_thread, telemetry_thread):
        super().__init__(name="ProcessingThread", daemon=True)
        self.capture_thread = capture_thread
        self.telemetry_thread = telemetry_thread
        
        # Initialize CV pipeline and GIS Tile database connection
        self.pipeline = VGPSVisionPipeline()
        self.db = GeospatialDatabase()
        
        self._running = False
        
        # Performance logging
        self.fps_calc = 0.0
        self.frame_count = 0
        
        # ORB Reference Tile Cache:
        # Caches keypoints/descriptors of the currently active 256x256 map tile to avoid
        # extracting features of the reference map tile on every frame (critical optimization for ARM Cortex-A53).
        self.cached_tile_coords = (None, None)
        self.cached_tile_kp = None
        self.cached_tile_desc = None
        
        # Track last known good latitude and longitude for database lookups
        self.last_lat = config.TAKEOFF_LAT
        self.last_lon = config.TAKEOFF_LON

    def _ensure_reference_tile_features(self, lat, lon):
        """
        Resolves Slippy map indices for coordinates. Fetches tile from DB and computes ORB features
        only if we cross into a new tile boundary, utilizing local memory cache.
        """
        # Fetch matching Web Mercator indices
        from geospatial import wgs84_to_slippy
        x_tile, y_tile = wgs84_to_slippy(lat, lon, config.MAP_ZOOM_LEVEL)
        
        # Return cache if tile indices remain unchanged
        if (x_tile, y_tile) == self.cached_tile_coords and self.cached_tile_desc is not None:
            return self.cached_tile_kp, self.cached_tile_desc, x_tile, y_tile
            
        logger.info(f"Crossed tile boundary! Loading reference tile X={x_tile}, Y={y_tile} at Zoom={config.MAP_ZOOM_LEVEL}...")
        
        # Fetch the BGR tile image from SQLite/MBTiles
        tile_img, fetched_x, fetched_y = self.db.fetch_tile_by_wgs84(lat, lon)
        
        if tile_img is None:
            logger.error(f"Failed to load tile X={x_tile}, Y={y_tile} from MBTiles cache.")
            return None, None, x_tile, y_tile
            
        # Apply standard binarization to reference tile to align boundary gradients with live preprocessor
        binary_tile = preprocess_image(tile_img, method='adaptive')
        
        # Extract features and descriptors for this reference tile
        kp_ref, desc_ref = self.pipeline.extract_features(binary_tile)
        
        logger.info(f"Extracted {len(kp_ref)} ORB descriptors for reference tile X={x_tile}, Y={y_tile}.")
        
        # Cache descriptors in RAM
        self.cached_tile_coords = (x_tile, y_tile)
        self.cached_tile_kp = kp_ref
        self.cached_tile_desc = desc_ref
        
        return kp_ref, desc_ref, x_tile, y_tile

    def run(self):
        """
        Core CV execution loop.
        Pulls newest rectified frames, matches features, solves homography,
        projects pixel displacements, and updates MAVLink telemetry targets.
        """
        self._running = True
        start_time = time.time()
        
        logger.info("Core Processing Thread started successfully.")
        
        while self._running:
            loop_start = time.time()
            
            # Retrieve latest frame from capture buffer
            frame = self.capture_thread.get_frame()
            if frame is None:
                # Idle briefly if capture buffer is empty (waiting on camera stream)
                time.sleep(0.005)
                continue
                
            # Step 1: Pre-load and cache descriptors for the active reference geospatial tile
            ref_kp, ref_desc, x_tile, y_tile = self._ensure_reference_tile_features(self.last_lat, self.last_lon)
            
            if ref_desc is None or len(ref_desc) == 0:
                logger.warning("No reference features available. Telemetry invalid.")
                self.telemetry_thread.invalidate_position()
                continue
                
            # Step 2: Apply dynamic binarization on the live rectified frame to highlight structural contours
            # (Gaussian adaptive thresholding handles complex outdoor cloud shadows beautifully)
            binary_live = preprocess_image(frame, method='adaptive')
            
            # Step 3: Extract rotationally invariant ORB descriptors on the binarized live frame
            live_kp, live_desc = self.pipeline.extract_features(binary_live)
            
            if len(live_desc) < 8:
                logger.warning("Insufficient ORB features detected in live camera frame. Telemetry invalid.")
                self.telemetry_thread.invalidate_position()
                continue
                
            # Step 4: Run FLANN LSH binary matching and filter outliers via Lowe's Ratio test
            good_matches = self.pipeline.match_features(live_desc, ref_desc)
            
            # Step 5: Solve perspective transformation using cv2.findHomography + RANSAC
            H, mask = self.pipeline.estimate_homography(live_kp, ref_kp, good_matches)
            
            if H is not None:
                # Step 6: Map live camera center pixel to tile coordinate space (px, py)
                px, py = self.pipeline.resolve_center_pixel(H, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
                
                if px is not None and py is not None:
                    # Validate that resolved pixel coordinates are within standard bounds of the reference tile
                    # (Allowing slight margins to enable continuous tracking across borders before reloading)
                    margin = 64.0
                    if (-margin <= px <= 256.0 + margin) and (-margin <= py <= 256.0 + margin):
                        # Step 7: Apply Affine World File equations to project tile pixels to absolute coordinates
                        transformer = CoordinateTransformer(x_tile, y_tile, config.MAP_ZOOM_LEVEL)
                        lat_est, lon_est = transformer.pixel_to_wgs84(px, py)
                        
                        # Update localized coordinates
                        self.last_lat = lat_est
                        self.last_lon = lon_est
                        
                        # Calculate a simulated heading/yaw based on the homography shear components
                        # H[0,1] and H[1,0] represent skew/rotation profiles
                        # In strict nadir cameras, this maps directly to UAV heading:
                        yaw_rad = np.arctan2(H[1, 0], H[0, 0])
                        yaw_deg = float(np.degrees(yaw_rad)) % 360.0
                        
                        # Push calculated WGS84 coordinate to telemetry thread
                        self.telemetry_thread.update_position(
                            lat=lat_est,
                            lon=lon_est,
                            alt=config.DEFAULT_ALTITUDE,
                            yaw=yaw_deg
                        )
                        
                        logger.debug(f"Position resolved! Lat: {lat_est:.6f}, Lon: {lon_est:.6f}, Yaw: {yaw_deg:.1f}deg")
                    else:
                        logger.warning(f"Resolved tile coordinates out of bounds: px={px:.1f}, py={py:.1f}")
                        self.telemetry_thread.invalidate_position()
                else:
                    logger.warning("Failed to project center pixel using Homography matrix.")
                    self.telemetry_thread.invalidate_position()
            else:
                # If RANSAC fails, tracking is lost. Invalidate telemetry to avoid Pixhawk EKF innovation spikes
                logger.warning("Terrain match tracking lost: unable to resolve Homography.")
                self.telemetry_thread.invalidate_position()
                
            # Perform processing diagnostics
            self.frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                self.fps_calc = self.frame_count / elapsed
                logger.info(f"Processing Thread Active: {self.fps_calc:.1f} FPS | Matches: {len(good_matches)} | Last Lat: {self.last_lat:.6f}")
                self.frame_count = 0
                start_time = time.time()
                
            # Relinquish thread slice slightly to optimize ARM thermal boundaries
            time.sleep(0.001)

    def stop(self):
        """
        Signals the thread to terminate.
        """
        self._running = False
        logger.info("Stop signal sent to Processing Thread.")


class VGPSOrchestrator:
    """
    Central manager. Launches Capture, Processing, and Telemetry threads,
    coordinates graceful shutdown sequences, and handles OS interrupts.
    """
    def __init__(self):
        self.capture_thread = None
        self.telemetry_thread = None
        self.processing_thread = None
        
        # Set up system termination signals
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)

    def start(self):
        """
        Initializes and starts all thread components.
        """
        logger.info("=========================================")
        logger.info("    INITIALIZING VISION-BASED GPS SYSTEM   ")
        logger.info("=========================================")
        
        # Step 1: Spawn capture thread (V4L2 camera loop / dev/video0)
        self.capture_thread = CaptureThread(
            camera_index=config.CAMERA_INDEX,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=config.CAMERA_FPS
        )
        
        # Step 2: Spawn telemetry thread (MAVLink serial loop / dev/ttyS0)
        self.telemetry_thread = TelemetryThread(
            port=config.SERIAL_PORT,
            baud=config.BAUD_RATE,
            hz=config.TELEMETRY_UPDATE_HZ
        )
        
        # Step 3: Spawn core processing thread
        self.processing_thread = ProcessingThread(
            capture_thread=self.capture_thread,
            telemetry_thread=self.telemetry_thread
        )
        
        # Launch thread loops
        logger.info("Launching component threads...")
        self.capture_thread.start()
        self.telemetry_thread.start()
        self.processing_thread.start()
        
        logger.info("All threads running. Press Ctrl+C to terminate.")

    def _handle_shutdown_signal(self, signum, frame):
        """
        Catches termination interrupts and executes structured shutdown sequence.
        """
        logger.warning(f"Shutdown signal ({signum}) caught! Commencing graceful termination sequence...")
        self.stop()
        sys.exit(0)

    def stop(self):
        """
        Terminates all running thread loops safely.
        """
        logger.info("Halting VGPS threads...")
        
        if self.processing_thread is not None:
            self.processing_thread.stop()
            self.processing_thread.join(timeout=2.0)
            
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread.join(timeout=2.0)
            
        if self.telemetry_thread is not None:
            self.telemetry_thread.stop()
            self.telemetry_thread.join(timeout=2.0)
            
        logger.info("VGPS shutdown completed successfully.")


if __name__ == "__main__":
    # Import threading here to ensure classes run correctly in thread pools
    import threading
    orchestrator = VGPSOrchestrator()
    orchestrator.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1.0)
    except (KeyboardInterrupt, SystemExit):
        orchestrator.stop()
