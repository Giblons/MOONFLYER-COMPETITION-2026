"""
compare_feature_extractors.py — Feature Extractor & Image Index Performance Lab

Compares:
  - Extractors: ORB vs SIFT vs AKAZE
  - Preprocessing: Standard Grayscale vs Excess Green Index (ExG)

Usage:
  python compare_feature_extractors.py
"""

import os
import sys
import time
import cv2
import numpy as np

# Absolute paths to drone videos
VIDEO_PATH = "/home/rizky/object-detection/down1.mp4"

def get_excess_green(frame):
    """Calculates the Excess Green Index (ExG): 2*G - R - B."""
    b, g, r = cv2.split(frame.astype(np.float32))
    exg = 2 * g - r - b
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return exg

def run_performance_test(img_gray, img_exg, name):
    """Evaluates ORB, SIFT, and AKAZE on both grayscale and ExG images."""
    results = {}
    
    # ----------------------------------------------------
    # ORB Setup
    # ----------------------------------------------------
    orb = cv2.ORB_create(nfeatures=1500)
    
    # ORB - Grayscale
    t0 = time.time()
    kp_orb_g, desc_orb_g = orb.detectAndCompute(img_gray, None)
    t_orb_g = (time.time() - t0) * 1000
    
    # ORB - ExG
    t0 = time.time()
    kp_orb_e, desc_orb_e = orb.detectAndCompute(img_exg, None)
    t_orb_e = (time.time() - t0) * 1000

    # ----------------------------------------------------
    # SIFT Setup
    # ----------------------------------------------------
    sift = cv2.SIFT_create(nfeatures=1500)
    
    # SIFT - Grayscale
    t0 = time.time()
    kp_sift_g, desc_sift_g = sift.detectAndCompute(img_gray, None)
    t_sift_g = (time.time() - t0) * 1000
    
    # SIFT - ExG
    t0 = time.time()
    kp_sift_e, desc_sift_e = sift.detectAndCompute(img_exg, None)
    t_sift_e = (time.time() - t0) * 1000

    # ----------------------------------------------------
    # AKAZE Setup
    # ----------------------------------------------------
    akaze = cv2.AKAZE_create()
    
    # AKAZE - Grayscale
    t0 = time.time()
    kp_akaze_g, desc_akaze_g = akaze.detectAndCompute(img_gray, None)
    t_akaze_g = (time.time() - t0) * 1000
    
    # AKAZE - ExG
    t0 = time.time()
    kp_akaze_e, desc_akaze_e = akaze.detectAndCompute(img_exg, None)
    t_akaze_e = (time.time() - t0) * 1000

    return {
        "ORB_Gray":    (len(kp_orb_g), t_orb_g, kp_orb_g),
        "ORB_ExG":     (len(kp_orb_e), t_orb_e, kp_orb_e),
        "SIFT_Gray":   (len(kp_sift_g), t_sift_g, kp_sift_g),
        "SIFT_ExG":    (len(kp_sift_e), t_sift_e, kp_sift_e),
        "AKAZE_Gray":  (len(kp_akaze_g), t_akaze_g, kp_akaze_g),
        "AKAZE_ExG":   (len(kp_akaze_e), t_akaze_e, kp_akaze_e)
    }

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Test video not found at: {VIDEO_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Could not read frame from test video.")
        sys.exit(1)

    # Resize to standard size for speed & matching consistency
    frame = cv2.resize(frame, (640, 640))
    
    # Generate Preprocessing Targets
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_exg = get_excess_green(frame)

    print("=========================================================================")
    print("        VGPS RICE FIELD EXTRACTOR & BINARIZATION ANALYSIS LAB            ")
    print("=========================================================================")
    print(f" Frame Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print("-------------------------------------------------------------------------")

    res = run_performance_test(img_gray, img_exg, "down1")

    # Display Performance Table
    print(f"{'Method & Input':<22} | {'Keypoints Detected':<18} | {'Execution Speed (ms)':<20}")
    print("-" * 68)
    for key, val in res.items():
        print(f"{key:<22} | {val[0]:<18,} | {val[1]:<20.2f}")
    print("=========================================================================")

    # Generate Visualization Panel Grid
    # Draw keypoints on cloned images
    vis_orb_g   = cv2.drawKeypoints(img_gray, res["ORB_Gray"][2][:250], None, color=(0, 255, 0))
    vis_orb_e   = cv2.drawKeypoints(img_exg, res["ORB_ExG"][2][:250], None, color=(0, 255, 0))
    
    vis_sift_g  = cv2.drawKeypoints(img_gray, res["SIFT_Gray"][2][:250], None, color=(0, 0, 255))
    vis_sift_e  = cv2.drawKeypoints(img_exg, res["SIFT_ExG"][2][:250], None, color=(0, 0, 255))
    
    vis_akaze_g = cv2.drawKeypoints(img_gray, res["AKAZE_Gray"][2][:250], None, color=(255, 0, 0))
    vis_akaze_e = cv2.drawKeypoints(img_exg, res["AKAZE_ExG"][2][:250], None, color=(255, 0, 0))

    # Add descriptive label overlays
    font = cv2.FONT_HERSHEY_SIMPLEX
    for img, text in [
        (vis_orb_g, "ORB + Grayscale"), (vis_orb_e, "ORB + Excess Green (ExG)"),
        (vis_sift_g, "SIFT + Grayscale"), (vis_sift_e, "SIFT + Excess Green (ExG)"),
        (vis_akaze_g, "AKAZE + Grayscale"), (vis_akaze_e, "AKAZE + Excess Green (ExG)")
    ]:
        cv2.rectangle(img, (5, 5), (310, 32), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 24), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Stack into a beautiful comparative workspace matrix
    row1 = np.hstack((vis_orb_g, vis_sift_g, vis_akaze_g))
    row2 = np.hstack((vis_orb_e, vis_sift_e, vis_akaze_e))
    dashboard = np.vstack((row1, row2))

    # Resize dashboard to fit normal screen dimensions comfortably
    screen_h, screen_w = dashboard.shape[:2]
    scale = 0.65
    dashboard_resized = cv2.resize(dashboard, (int(screen_w * scale), int(screen_h * scale)), interpolation=cv2.INTER_AREA)

    # Save to disk instead of imshow to support headless OpenCV environments
    export_path = "/home/rizky/object-detection/exports/feature_comparison_dashboard.jpg"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    cv2.imwrite(export_path, dashboard_resized)

    print(f"\nVisual analysis complete! Dashboard saved to: {export_path}")
    print("Open this image file to see the side-by-side comparison.")

if __name__ == "__main__":
    main()
