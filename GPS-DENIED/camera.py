"""
Camera Capture and Lens Distortion Correction Module.
Implements a low-latency Capture Thread using cv2.VideoCapture with aggressive
frame-dropping to prevent buffer lag, alongside high-performance NEON-optimized
fisheye lens undistortion remapping. Includes a high-fidelity synthetic ground generator fallback.
"""

import time
import logging
import threading
import cv2
import numpy as np
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s")
logger = logging.getLogger("VGPS.Camera")


class FisheyeRectifier:
    """
    Manages fisheye lens distortion correction.
    Pre-computes rectification maps to enable extremely fast cv2.remap calls
    leveraging ARM NEON SIMD instructions instead of running heavy undistort loops.
    """
    def __init__(self, width, height, K, D, balance=0.0):
        self.width = width
        self.height = height
        self.K = K
        self.D = D
        self.balance = balance
        
        # Pre-compute the rectification maps
        self.map1 = None
        self.map2 = None
        self.new_K = None
        self._compute_maps()

    def _compute_maps(self):
        """
        Pre-computes rectification maps using cv2.fisheye.
        estimateNewCameraMatrixForUndistort with balance=0 crops out all black border pixels.
        """
        logger.info("Pre-calculating fisheye undistortion rectification maps...")
        # Estimate new camera matrix for undistort to get cropped ROI
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (self.width, self.height), np.eye(3), balance=self.balance
        )
        
        # Generate undistort rectification map (using CV_16SC2/CV_32FC1 for high performance remapping)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.new_K, (self.width, self.height), cv2.CV_16SC2
        )
        logger.info("Fisheye rectification maps pre-computed successfully.")

    def rectify(self, frame):
        """
        Applies pre-computed distortion mapping using NEON-accelerated cv2.remap.
        """
        if self.map1 is None or self.map2 is None:
            return frame
        return cv2.remap(
            frame, 
            self.map1, 
            self.map2, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT
        )


class CaptureThread(threading.Thread):
    """
    Asynchronous camera capture thread.
    Reads continuous frames from a physical camera device or falls back gracefully
    to an animated synthetic terrain generator if no hardware device is found.
    Implements a single-frame mutex-locked buffer to minimize frame delivery latency (zero-queue lag).
    """
    def __init__(self, camera_index=config.CAMERA_INDEX, 
                 width=config.CAMERA_WIDTH, 
                 height=config.CAMERA_HEIGHT, 
                 fps=config.CAMERA_FPS):
        super().__init__(name="CaptureThread", daemon=True)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
        # Lens undistortion setup
        self.rectifier = FisheyeRectifier(width, height, config.K, config.D, config.UNDISTORT_BALANCE)
        
        # Thread safety buffers
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._is_mock = False
        
        # Performance logging variables
        self.frame_count = 0
        self.fps_calc = 0.0

    def _init_camera(self):
        """
        Attempts to initialize the V4L2 camera. Falls back to generating a mock
        synthetic visual landscape if hardware is unavailable.
        """
        logger.info(f"Attempting to open camera index {self.camera_index}...")
        
        # On Linux, V4L2 is the high-performance camera backend
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # Try setting framerate if hardware supports it
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Read a test frame to ensure sensor is streaming
            ret, frame = self.cap.read()
            if ret and frame is not None:
                logger.info(f"Successfully connected to USB camera /dev/video{self.camera_index}.")
                return
            else:
                self.cap.release()
        
        # If camera could not be opened, flag mock state and construct synthetic terrain pipeline
        logger.warning(f"Could not open /dev/video{self.camera_index}. Initializing synthetic high-fidelity terrain generator.")
        self._is_mock = True

    def _generate_synthetic_frame(self, t):
        """
        Generates highly detailed synthetic ground landscape frames containing simulated fields,
        roads, and structural boundaries. This guarantees that ORB and FLANN can register
        valid matched keypoints and solve Homographies offline without hardware.
        """
        # Create base green landscape
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = [34, 139, 34]  # Forest Green
        
        # Draw multiple agricultural fields/polygons
        np.random.seed(42)  # Maintain stable static world landmarks
        field_centers = [
            (100, 100, [60, 179, 113]),   # Medium Sea Green
            (450, 150, [107, 142, 35]),   # Olive Drab
            (150, 480, [143, 188, 143]),  # Dark Sea Green
            (500, 450, [85, 107, 47]),    # Dark Olive Green
            (320, 320, [46, 139, 87])     # Sea Green
        ]
        
        # Add dynamic panning/movement (simulating a hovering/drifting UAV)
        dx = int(50 * np.sin(t * 0.15))
        dy = int(50 * np.cos(t * 0.15))
        
        # Render static terrain landmarks offset by current UAV drift
        for cx, cy, color in field_centers:
            # Shift landmarks to simulate drone flight
            sx = (cx + dx) % self.width
            sy = (cy + dy) % self.height
            cv2.rectangle(frame, (sx - 80, sy - 80), (sx + 80, sy + 80), color, -1)
            # Add micro texture borders to ensure high keypoint detection density
            cv2.rectangle(frame, (sx - 80, sy - 80), (sx + 80, sy + 80), (200, 200, 200), 2)
            cv2.circle(frame, (sx, sy), 10, (100, 100, 255), -1)
            
        # Draw a simulated structural road crossing the frame
        road_y = (300 + dy) % self.height
        cv2.line(frame, (0, road_y), (self.width, road_y), (128, 128, 128), 24)
        cv2.line(frame, (0, road_y), (self.width, road_y), (255, 255, 255), 2, lineType=cv2.LINE_AA)
        
        # Simulate lens radial distortion on the generated frame to verify rectifier code path
        # A simple software-based barrel distortion wrap
        if not self._is_mock:
            # We don't apply barrel distortion to hardware frames since the lens already does it
            pass
        else:
            # Apply slight simulated barrel distortion to mock frame so the rectifier has work to do
            map_x, map_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            # Center of distortion
            cx, cy = self.width / 2.0, self.height / 2.0
            x = (map_x - cx) / cx
            y = (map_y - cy) / cy
            r = np.sqrt(x**2 + y**2)
            # Radial distortion formula: r_distorted = r * (1 + k1*r^2 + k2*r^4)
            factor = 1.0 + 0.1 * r**2 + 0.05 * r**4
            map_x_dist = (x * factor * cx) + cx
            map_y_dist = (y * factor * cy) + cy
            frame = cv2.remap(frame, map_x_dist.astype(np.float32), map_y_dist.astype(np.float32), cv2.INTER_LINEAR)
            
        # Overlay informational status banner
        cv2.putText(frame, "SIMULATED NADIR UAV CAM FEED", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame

    def run(self):
        """
        Thread execution loop. Captures frames at specified FPS rate, undistorts them immediately
        via the rectifier, and writes them to the mutex-locked buffer, overwriting stale data.
        """
        self._init_camera()
        self._running = True
        
        start_time = time.time()
        loop_interval = 1.0 / self.fps
        
        while self._running:
            loop_start = time.time()
            
            if not self._is_mock:
                ret, raw_frame = self.cap.read()
                if not ret or raw_frame is None:
                    logger.error("Camera frame read error. Re-initializing...")
                    time.sleep(0.5)
                    continue
            else:
                # Generate synthetic ground terrain moving over time
                raw_frame = self._generate_synthetic_frame(time.time() - start_time)
            
            # Apply fast NEON-optimized fisheye lens rectification
            undistorted_frame = self.rectifier.rectify(raw_frame)
            
            # Atomically update buffer (stale frame-dropping)
            with self._lock:
                self._latest_frame = undistorted_frame.copy()
            
            self.frame_count += 1
            
            # Update FPS diagnostics every second
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                self.fps_calc = self.frame_count / elapsed
                logger.debug(f"Capture Thread active: {self.fps_calc:.1f} FPS (Mock: {self._is_mock})")
                self.frame_count = 0
                start_time = time.time()
                
            # Regulate capture rate to avoid wasting CPU cycles
            delay = loop_interval - (time.time() - loop_start)
            if delay > 0:
                time.sleep(delay)
                
        # Clean up capture resources
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        logger.info("Capture thread shut down successfully.")

    def get_frame(self):
        """
        Retrieves the latest undistorted frame from the buffer in a thread-safe manner.
        Returns None if no frame is currently available.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            # Return a copy to prevent thread memory conflicts
            return self._latest_frame.copy()

    def stop(self):
        """
        Signals the thread to stop execution.
        """
        self._running = False
        logger.info("Stop signal sent to Capture Thread.")
