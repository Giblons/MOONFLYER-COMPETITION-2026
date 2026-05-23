"""
Computer Vision Pipeline Module.
Implements dynamic binarization algorithms (Otsu Global & Adaptive Gaussian),
ORB keypoint extraction, FLANN-LSH binary descriptor matching, Lowe's Ratio filtering,
and planar Homography estimation using RANSAC. Optimized for high-throughput embedded processors.
"""

import logging
import cv2
import numpy as np
import config

logger = logging.getLogger("VGPS.Vision")


def preprocess_image(img, method='adaptive'):
    """
    Applies binarization to isolate topographical and structural boundaries (roads, fields, river beds).
    Optimized to enhance contrast and eliminate illumination artifacts (clouds, non-uniform solar angles).
    
    Parameters:
        img (np.ndarray): Input BGR or Grayscale image.
        method (str): 'adaptive' for Adaptive Gaussian, 'otsu' for Otsu Global thresholding.
        
    Returns:
        np.ndarray: Pristine binary image.
    """
    # Convert to grayscale if it is a color frame
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    # Gaussian blur to reduce high-frequency salt-and-pepper noise prior to binarization
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if method.lower() == 'otsu':
        try:
            # Otsu's Thresholding (cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # Ideal for bimodal histogram scenarios (high-contrast terrain shadows)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            return thresh
        except Exception as e:
            logger.warning(f"Otsu thresholding failed: {e}. Falling back to Adaptive Gaussian.")
            
    # Fallback / Primary default: Adaptive Gaussian Thresholding
    # Outstanding for non-uniform lighting/vignetting and rolling terrain shadows.
    # Uses ADAPTIVE_THRESH_GAUSSIAN_C with block_size=11, constant_C=2.
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    return thresh


class VGPSVisionPipeline:
    """
    Core CV pipeline executing keypoint detection, matching, and homography.
    Uses ORB + FLANN LSH to resolve matching under high-rotation, translation, and scale variance.
    """
    def __init__(self):
        # Initialize ORB detector with WTA_K=2 (which yields standard 256-bit binary descriptors)
        self.orb = cv2.ORB_create(
            nfeatures=config.ORB_MAX_FEATURES,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_FAST_SCORE,
            patchSize=31,
            fastThreshold=10        # Lowered from 20 — more sensitive to low-contrast terrain corners
        )
        
        # Initialize FLANN matcher utilizing Locality Sensitive Hashing (LSH) for binary ORB keys
        # Avoiding KD-Trees exceptions triggered by treating binary descriptors as continuous spaces.
        self.flann = cv2.FlannBasedMatcher(
            config.FLANN_INDEX_PARAMS,
            config.FLANN_SEARCH_PARAMS
        )
        
        logger.info("ORB detector and LSH-FLANN matcher initialized successfully.")

    def extract_features(self, binary_img):
        """
        Extracts keypoints and 256-bit binary ORB descriptors from preprocessed binary frames.
        
        Returns:
            keypoints (list of cv2.KeyPoint), descriptors (np.ndarray of uint8 binary strings)
        """
        kp, desc = self.orb.detectAndCompute(binary_img, None)
        if desc is None:
            desc = np.empty((0, 32), dtype=np.uint8)
        return kp, desc

    def match_features(self, desc_live, desc_ref):
        """
        Executes FLANN-LSH binary matching between live and reference descriptors.
        Filters outliers via Lowe's Ratio Test (knnMatch k=2) to reject false positives.
        
        Returns:
            list of cv2.DMatch: Good filtered matches.
        """
        # Require a minimal number of descriptors to prevent matcher exception
        if len(desc_live) < 2 or len(desc_ref) < 2:
            return []
            
        try:
            # KNN-matching with k=2
            raw_matches = self.flann.knnMatch(desc_live, desc_ref, k=2)
        except Exception as e:
            logger.error(f"FLANN matching threw exception: {e}")
            return []
            
        good_matches = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                # Lowe's ratio test: reject matches where second-best match is highly similar
                # Discards repetitive agricultural grids, dense structural facades, etc.
                if m.distance < config.LOWE_RATIO_THRESHOLD * n.distance:
                    good_matches.append(m)
                    
        return good_matches

    def estimate_homography(self, kp_live, kp_ref, good_matches):
        """
        Calculates Planar Perspective Transformation between live camera frame and reference map tile.
        Employs Random Sample Consensus (RANSAC) to dynamically isolate mapping outliers.
        
        Returns:
            H (np.ndarray, 3x3): Homography matrix or None if calculation fails or insufficient matches.
            mask (np.ndarray): RANSAC inlier mask.
        """
        # We need a minimum of 4 matches for planar projection, 
        # but require 8+ for high confidence and outlier resilience on UAVs.
        if len(good_matches) < 8:
            logger.warning(f"Insufficient matches for Homography estimation: {len(good_matches)} (Need >=8)")
            return None, None
            
        # Extract matching keypoint pixel locations
        src_pts = np.float32([kp_live[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Solve Homography using RANSAC with a pixel reprojection error tolerance of 5.0
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            logger.warning("RANSAC failed to converge on stable Homography matrix.")
            return None, None
            
        return H, mask

    def resolve_center_pixel(self, H, width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT):
        """
        Projects the central optical nadir pixel of the camera frame (cx, cy)
        into the coordinate frame of the reference map tile using perspective mapping.
        
        Returns:
            px_tile (float), py_tile (float): Pixel index on the target 256x256 reference tile.
        """
        if H is None:
            return None, None
            
        cx = width / 2.0
        cy = height / 2.0
        
        # Structure point array for cv2.perspectiveTransform (shape must be 1x1x2)
        center_pt = np.array([[[cx, cy]]], dtype=np.float32)
        
        try:
            # Transform point using homography, accounting for perspective division (scale w)
            transformed = cv2.perspectiveTransform(center_pt, H)
            px_tile, py_tile = transformed[0, 0]
            return float(px_tile), float(py_tile)
        except Exception as e:
            logger.error(f"Perspective transformation error: {e}")
            return None, None
