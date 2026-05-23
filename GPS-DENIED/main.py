import time
import sys
import argparse
from camera_capture import CameraCaptureThread
from preprocessing import Preprocessor
from database import DatabaseManager
from matching import Matcher
from transformation import CoordinateTransformer
from nmea_emitter import NMEAEmitterThread

def main():
    """
    Main Processing Engine (Core): Pulls the frame and executes the core computer vision,
    multi-database matching, and geospatial mathematical pipeline.
    """
    parser = argparse.ArgumentParser(description="Autonomous Vision-Based Navigation (VBN) System")
    parser.add_argument("--mbtiles", default="map.mbtiles", help="Path to MBTiles raster database")
    parser.add_argument("--geojson", default="map.geojson", help="Path to GeoJSON vector database")
    parser.add_argument("--port", default="/dev/ttyS0", help="Serial port for NMEA spoofing")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    args = parser.parse_args()

    print("[Core] Initializing VBN System...")

    # Initialize components
    # Using 1080p target for Rockchip RK3566
    preprocessor = Preprocessor(img_shape=(1920, 1080))
    db_manager = DatabaseManager(mbtiles_path=args.mbtiles, geojson_path=args.geojson)
    matcher = Matcher(min_confidence=0.7)
    transformer = CoordinateTransformer()

    # Initialize multi-threaded architecture
    camera_thread = CameraCaptureThread(camera_index=0, width=1920, height=1080, fps=30)
    nmea_thread = NMEAEmitterThread(port=args.port, baudrate=args.baud, hz=5)

    # Start threads
    camera_thread.start()
    nmea_thread.start()

    # Initial "last known good" coordinates (e.g. startup location or last GPS fix before jamming)
    # In a real system, this would be injected upon startup.
    last_lon, last_lat = -122.4194, 37.7749 # Example: San Francisco
    zoom_level = 18 # Typical operational zoom level

    print("[Core] Entering main processing loop.")
    try:
        while True:
            start_time = time.time()
            
            # Pull latest frame from mutex-locked buffer
            frame = camera_thread.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
                
            # 1. Advanced Preprocessing
            # Correct fisheye distortion and convert color spaces
            undistorted_frame, gray_frame, hsv_frame, binary_frame = preprocessor.preprocess_pipeline(frame)
            
            # 2. Multi-Tiered Database Integration & Expanding Radius Search
            match_found = False
            best_confidence = 0.0
            best_lon, best_lat = 0.0, 0.0
            
            # Yield tiles in expanding radius
            for xtile, ytile, ref_img in db_manager.expanding_radius_search(last_lon, last_lat, zoom_level, radius=1, max_radius=3):
                
                # 3. Multi-Modal Feature Matching & Confidence Scoring
                # ORB/FLANN Raster match
                H, confidence = matcher.process_match(gray_frame, ref_img)
                
                # Check Vector Match stub as well
                # vector_shapes = db_manager.query_vector_data(...)
                # v_H, v_conf = matcher.semantic_match(binary_frame, vector_shapes)

                if H is not None and confidence >= 0.7:
                    if confidence > best_confidence:
                        best_confidence = confidence
                        # Get top-left coordinates of the matched tile
                        tile_lon, tile_lat = db_manager.tile_to_lonlat(xtile, ytile, zoom_level)
                        
                        # 4. Coordinate Transformation (Homography to WGS84)
                        best_lon, best_lat = transformer.process_transformation(H, gray_frame.shape, tile_lon, tile_lat, zoom_level)
                        match_found = True
                        break # Stop searching if we found a good match

            if match_found:
                print(f"[Core] Fix Acquired: {best_lat:.6f}, {best_lon:.6f} (Conf: {best_confidence:.2f})")
                # Update NMEA thread
                nmea_thread.update_coordinates(best_lat, best_lon, confidence=best_confidence)
                # Update last known good
                last_lon, last_lat = best_lon, best_lat
            else:
                # No match found within max radius, lose fix
                print("[Core] Fix Lost. Expanding search radius on next tick...")
                nmea_thread.update_coordinates(last_lat, last_lon, confidence=0.0)
                
            # Keep engine loop manageable (e.g. max 10Hz)
            elapsed = time.time() - start_time
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)
                
    except KeyboardInterrupt:
        print("\n[Core] Shutting down VBN System...")
    finally:
        camera_thread.stop()
        nmea_thread.stop()
        db_manager.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
