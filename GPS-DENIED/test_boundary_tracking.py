"""
test_boundary_tracking.py — Boundary Line Segment Detection (LSD) Lab

Demonstrates why point-feature extractors (ORB) struggle in agricultural environments due
to repetitive crop row "noise", and how Hough Line Boundary Tracking extracts clean, 
invariant geometric dividers (mud bunds, canals, paths) instead.

Compares:
  - Left Panel  : Standard ORB Point Features (noisy clustering on plants).
  - Right Panel : Line Segment Detection / Hough Dividers (ignores crops, extracts structure).

Usage:
  python test_boundary_tracking.py
"""

import os
import sys
import cv2
import numpy as np

VIDEO_PATH = "/home/rizky/object-detection/down1.mp4"

def get_dike_lines(frame):
    """Detects mud bunds, canals, and field boundary lines using Canny + Hough Transform."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth out individual crop leaf noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Edge detection optimized for soil-to-water boundary dividers
    edges = cv2.Canny(blurred, 30, 80, apertureSize=3)
    
    # Hough Line Transform to group edges into clean linear structures
    # Returns an array of lines [x1, y1, x2, y2]
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=60,
        minLineLength=80,
        maxLineGap=20
    )
    return lines, edges

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Test video not found at: {VIDEO_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = 15

    # Video Writer Setup
    export_path = "/home/rizky/object-detection/exports/boundary_tracking_sim.mp4"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(export_path, fourcc, fps, (1152, 576))

    # Initialize ORB for point feature comparison
    orb = cv2.ORB_create(nfeatures=500)

    print("=========================================================================")
    print("      VGPS FIELD BOUNDARY TRACKING & DIKE EXTRACTION LAB                 ")
    print("=========================================================================")
    print("  Left Panel  : Standard ORB Points (Clusters on crop rows, high noise)")
    print("  Right Panel : Hough Line Dividers (Extracts clean fields & canal edges)")
    print("=========================================================================")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Looping video.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 640))

        # ----------------------------------------------------
        # Panel 1: Standard ORB Keypoint Clustering
        # ----------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, _ = orb.detectAndCompute(gray, None)
        
        vis_orb = frame.copy()
        for k in kp:
            x, y = map(int, k.pt)
            # Draw point clusters in red
            cv2.circle(vis_orb, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)

        # ----------------------------------------------------
        # Panel 2: Hough Line Segment Detection (Dikes & Canals)
        # ----------------------------------------------------
        lines, edges = get_dike_lines(frame)
        
        vis_lines = frame.copy()
        
        # Draw detected dike boundaries in glowing neon green
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis_lines, (x1, y1), (x2, y2), (0, 255, 120), 2, cv2.LINE_AA)
                cv2.circle(vis_lines, (x1, y1), 4, (255, 0, 0), -1)
                cv2.circle(vis_lines, (x2, y2), 4, (255, 0, 0), -1)

        # Add HUD Label Overlays
        cv2.rectangle(vis_orb, (10, 10), (320, 42), (0, 0, 0), -1)
        cv2.putText(vis_orb, f"ORB KEYPOINTS: {len(kp)} CLUSTERS", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)
        
        line_count = len(lines) if lines is not None else 0
        cv2.rectangle(vis_lines, (10, 10), (380, 42), (0, 0, 0), -1)
        cv2.putText(vis_lines, f"HOUGH LINE DIKE SEGMENTS: {line_count} DETECTED", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1, cv2.LINE_AA)

        # Stack Side-by-Side
        dashboard = np.hstack((vis_orb, vis_lines))
        dashboard_resized = cv2.resize(dashboard, (1152, 576), interpolation=cv2.INTER_AREA)

        # Save frame to output .mp4
        out_video.write(dashboard_resized)

        frame_idx += 1
        # Limit frames for automated testing (headless loop)
        if frame_idx >= 300: # 20 seconds at 15fps
            break

    cap.release()
    out_video.release()
    print(f"\nSimulation complete! Video saved to: {export_path}")

if __name__ == "__main__":
    main()
