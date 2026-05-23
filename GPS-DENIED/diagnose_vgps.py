"""
VGPS Offline Pipeline Diagnostic Utility.
Runs a full execution pass of the Vision-Based GPS system offline.
Generates synthetic terrain, undistorts it using the fisheye lens map,
applies both Otsu and Adaptive binarizations, queries the local SQLite/MBTiles cache,
computes ORB/FLANN LSH matches, solves the planar homography, and computes the WGS84 position.
"""

import time
import logging
import cv2
import numpy as np
import math

import config
from camera import FisheyeRectifier, CaptureThread
from vision import preprocess_image, VGPSVisionPipeline
from geospatial import GeospatialDatabase, CoordinateTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VGPS.Diagnostics")


def run_diagnostic():
    logger.info("=========================================")
    logger.info("  STARTING OFFLINE VGPS PIPELINE DIAGNOSTIC ")
    logger.info("=========================================")
    
    # 1. Initialize DB and Fetch Reference Tile
    logger.info("\n--- STEP 1: Tile Database Verification ---")
    db = GeospatialDatabase()
    tile_img, x_tile, y_tile = db.fetch_tile_by_wgs84(config.TAKEOFF_LAT, config.TAKEOFF_LON)
    
    if tile_img is not None:
        logger.info(f"SUCCESS: Loaded reference map tile at indices: X={x_tile}, Y={y_tile}")
        logger.info(f"Tile image dimensions: {tile_img.shape} (Channels={tile_img.shape[2]})")
    else:
        logger.error("FAILURE: Unable to query reference map tile from database.")
        return False
        
    # 2. Camera Rectifier and Fisheye undistortion
    logger.info("\n--- STEP 2: Fisheye Lens Calibration Map Verification ---")
    rectifier = FisheyeRectifier(
        config.CAMERA_WIDTH, 
        config.CAMERA_HEIGHT, 
        config.K, 
        config.D, 
        config.UNDISTORT_BALANCE
    )
    
    # Generate mock frame with barrel distortion from CaptureThread class
    capture_thread = CaptureThread()
    # Generate frame at time = 0.5s (slight motion offset)
    mock_raw_frame = capture_thread._generate_synthetic_frame(0.5)
    
    start_t = time.time()
    rectified_frame = rectifier.rectify(mock_raw_frame)
    rectify_elapsed = (time.time() - start_t) * 1000.0
    
    logger.info(f"SUCCESS: Mock camera frame successfully processed.")
    logger.info(f"Rectification execution speed: {rectify_elapsed:.2f} ms (Target < 5ms for 60FPS).")
    logger.info(f"Rectified frame dimensions: {rectified_frame.shape}")
    
    # 3. Dynamic Thresholding/Binarization Preprocessing
    logger.info("\n--- STEP 3: Dynamic Preprocessing (Binarization) ---")
    
    t_start = time.time()
    binary_adaptive = preprocess_image(rectified_frame, method='adaptive')
    adaptive_elapsed = (time.time() - t_start) * 1000.0
    
    t_start = time.time()
    binary_otsu = preprocess_image(rectified_frame, method='otsu')
    otsu_elapsed = (time.time() - t_start) * 1000.0
    
    logger.info(f"Adaptive Gaussian Binarization speed: {adaptive_elapsed:.2f} ms")
    logger.info(f"Otsu's Global Binarization speed: {otsu_elapsed:.2f} ms")
    
    # Check binary frame pixel distribution
    density_adaptive = (np.count_nonzero(binary_adaptive) / binary_adaptive.size) * 100.0
    density_otsu = (np.count_nonzero(binary_otsu) / binary_otsu.size) * 100.0
    logger.info(f"Adaptive thresholding structural density: {density_adaptive:.1f}%")
    logger.info(f"Otsu's thresholding structural density: {density_otsu:.1f}%")
    
    # 4. ORB and FLANN LSH Matching
    logger.info("\n--- STEP 4: ORB Feature Detection and FLANN-LSH Matching ---")
    pipeline = VGPSVisionPipeline()
    
    # Preprocess reference tile and extract keypoints
    binary_ref_tile = preprocess_image(tile_img, method='adaptive')
    
    kp_ref, desc_ref = pipeline.extract_features(binary_ref_tile)
    kp_live, desc_live = pipeline.extract_features(binary_adaptive)
    
    logger.info(f"Reference Tile Feature count: {len(kp_ref)}")
    logger.info(f"Live Frame Feature count: {len(kp_live)}")
    
    if len(desc_live) == 0 or len(desc_ref) == 0:
        logger.error("FAILURE: ORB failed to extract descriptors. Check image texture/binarization.")
        return False
        
    # Execute matching
    t_start = time.time()
    good_matches = pipeline.match_features(desc_live, desc_ref)
    match_elapsed = (time.time() - t_start) * 1000.0
    
    logger.info(f"LSH-FLANN Matcher executed in: {match_elapsed:.2f} ms")
    logger.info(f"Lowe's Ratio Test filtered matches count: {len(good_matches)}")
    
    # 5. Homography and Affine Coordinates
    logger.info("\n--- STEP 5: Planar Homography & GIS Transformation ---")
    H, mask = pipeline.estimate_homography(kp_live, kp_ref, good_matches)
    
    if H is None:
        logger.error("FAILURE: Homography matrix could not be resolved. Insufficient valid keypoint matches.")
        return False
        
    inliers = int(np.sum(mask)) if mask is not None else 0
    inlier_ratio = (inliers / len(good_matches)) * 100.0 if len(good_matches) > 0 else 0
    logger.info(f"Homography solved successfully.")
    logger.info(f"RANSAC Inliers: {inliers}/{len(good_matches)} ({inlier_ratio:.1f}%)")
    
    # Resolve center pixel projection
    px, py = pipeline.resolve_center_pixel(H, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    logger.info(f"Projected optical center pixel to tile coordinate space: px={px:.2f}, py={py:.2f}")
    
    # Transform pixels to WGS84 coordinates
    transformer = CoordinateTransformer(x_tile, y_tile, config.MAP_ZOOM_LEVEL)
    lat_est, lon_est = transformer.pixel_to_wgs84(px, py)
    
    logger.info("\n--- DIAGNOSTIC RESULTS ---")
    logger.info(f"Reference Takeoff Latitude:  {config.TAKEOFF_LAT:.6f}")
    logger.info(f"Reference Takeoff Longitude: {config.TAKEOFF_LON:.6f}")
    logger.info(f"Estimated Nadir Latitude:    {lat_est:.6f}")
    logger.info(f"Estimated Nadir Longitude:   {lon_est:.6f}")
    
    # Calculate geometric error distance (approximate in meters)
    lat_err_m = (lat_est - config.TAKEOFF_LAT) * 111139.0
    lon_err_m = (lon_est - config.TAKEOFF_LON) * 111139.0 * math.cos(math.radians(config.TAKEOFF_LAT))
    distance_err_m = math.sqrt(lat_err_m**2 + lon_err_m**2)
    logger.info(f"Drift Distance from Takeoff Coordinate: {distance_err_m:.2f} meters")
    
    logger.info("=========================================")
    logger.info("      ALL PIPELINE CHECKS SUCCESSFUL     ")
    logger.info("=========================================")
    return True


if __name__ == "__main__":
    run_diagnostic()
