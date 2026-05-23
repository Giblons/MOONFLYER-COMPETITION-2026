"""
test_visual_odometry.py — Visual Odometry Trajectory Simulator

Demonstrates how Visual Odometry (using Lucas-Kanade Optical Flow) preserves the flight
trajectory during absolute terrain match dropouts (e.g. over a featureless crop sector).

Compares:
  - Trajectory A (Without VO): Freezes or drops out when absolute matching fails.
  - Trajectory B (With VO): Uses frame-to-frame optical flow to track displacement.

Usage:
  python test_visual_odometry.py
"""

import os
import sys
import cv2
import numpy as np

VIDEO_PATH = "/home/rizky/object-detection/down1.mp4"

# Lucas-Kanade optical flow parameters
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Test video not found at: {VIDEO_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read video frame.")
        sys.exit(1)

    # Initialize frame sizing
    prev_frame = cv2.resize(prev_frame, (640, 640))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Select initial keypoints to track using Good Features to Track
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

    # Coordinates for tracking
    # (Starting at virtual center coordinate x=200, y=320)
    pos_no_vo = np.array([200.0, 320.0])
    pos_with_vo = np.array([200.0, 320.0])

    # Trajectory points for drawing
    trail_no_vo = []
    trail_with_vo = []

    frame_idx = 0
    fps = 15

    # Video Writer Setup
    export_path = "/home/rizky/object-detection/exports/visual_odometry_sim.mp4"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(export_path, fourcc, fps, (1152, 576))

    print("=========================================================================")
    print("        VISUAL ODOMETRY TRAJECTORY FLIGHT TESTING SUITE                  ")
    print("=========================================================================")
    print("  White Line  : Trajectory WITH Visual Odometry (Continuous tracking)")
    print("  Orange Line : Trajectory WITHOUT Visual Odometry (Freezes on lock loss)")
    print("=========================================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Looping video.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 640))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_idx += 1

        # --- Simulate terrain match lock dropout (e.g. frames 50 to 120 are 'feature-poor') ---
        # When lock is dropped, absolute terrain matching cannot update the position.
        lock_lost = (50 <= frame_idx <= 120) or (180 <= frame_idx <= 240)

        # 1. Update position *without* VO (absolute matching only)
        # If lock is active, it tracks fake moving flight speed. If lost, it halts completely.
        if not lock_lost:
            pos_no_vo += np.array([1.5, -0.6]) # Simulated constant drift velocity
        
        trail_no_vo.append(tuple(map(int, pos_no_vo)))

        # 2. Update position *with* VO (Calculates LK Optical Flow)
        dx, dy = 0.0, 0.0
        if prev_pts is not None and len(prev_pts) > 0:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **LK_PARAMS)
            
            # Select good points
            good_new = next_pts[status == 1]
            good_old = prev_pts[status == 1]
            
            if len(good_new) > 0:
                # Calculate mean translation shift
                translation = good_new - good_old
                dx, dy = np.mean(translation, axis=0)
            
            # Update tracking keypoints
            prev_pts = good_new.reshape(-1, 1, 2)
            
            # Re-detect keypoints if tracked points fall below threshold
            if len(prev_pts) < 30:
                prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        else:
            prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

        # Integrate displacement. If absolute match is lost, fall back entirely to VO displacement!
        if not lock_lost:
            # Absolute map matching works
            pos_with_vo += np.array([1.5, -0.6])
        else:
            # Fallback to local optical flow displacement (scaled to flight coordinates)
            pos_with_vo += np.array([-dx * 0.8, -dy * 0.8])
        
        trail_with_vo.append(tuple(map(int, pos_with_vo)))

        # ----------------------------------------------------
        # Rendering
        # ----------------------------------------------------
        # Visual Canvas 1: Live Camera Tracking Overlay
        vis_tracking = frame.copy()
        if prev_pts is not None:
            for pt in prev_pts:
                x, y = map(int, pt[0])
                cv2.circle(vis_tracking, (x, y), 3, (0, 255, 0), -1)

        # Visual Canvas 2: Navigation Map Plotter
        vis_map = np.zeros((640, 640, 3), dtype=np.uint8)
        # Draw grid
        for i in range(0, 640, 40):
            cv2.line(vis_map, (i, 0), (i, 640), (20, 20, 25), 1)
            cv2.line(vis_map, (0, i), (640, i), (20, 20, 25), 1)

        # Draw trails
        # WITHOUT VO Trail
        for i in range(1, len(trail_no_vo)):
            cv2.line(vis_map, trail_no_vo[i-1], trail_no_vo[i], (0, 100, 255), 2, cv2.LINE_AA)
        
        # WITH VO Trail
        for i in range(1, len(trail_with_vo)):
            cv2.line(vis_map, trail_with_vo[i-1], trail_with_vo[i], (255, 255, 255), 2, cv2.LINE_AA)

        # Draw current positions
        cv2.circle(vis_map, tuple(map(int, pos_no_vo)), 6, (0, 100, 255), -1, cv2.LINE_AA)
        cv2.circle(vis_map, tuple(map(int, pos_with_vo)), 6, (255, 255, 255), -1, cv2.LINE_AA)

        # Interface Overlay & Text
        cv2.rectangle(vis_map, (15, 15), (620, 105), (10, 10, 15), -1)
        cv2.rectangle(vis_map, (15, 15), (620, 105), (50, 50, 60), 1)
        
        status_color = (0, 80, 255) if lock_lost else (0, 255, 120)
        status_text = "TERRAIN MATCH LOCK: LOST (VO FALLBACK ACTIVE)" if lock_lost else "TERRAIN MATCH LOCK: ACTIVE"
        
        cv2.putText(vis_map, status_text, (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)
        cv2.putText(vis_map, f"ORANGE PATH : WITHOUT VO (Accumulated position freeze)", (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv2.LINE_AA)
        cv2.putText(vis_map, f"WHITE PATH  : WITH VISUAL ODOMETRY (Continuous trajectory)", (25, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Label overlays for side-by-side
        cv2.putText(vis_tracking, "LIVE OPTICAL FLOW VECTOR FIELD", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)

        # Stack Side-by-Side
        dashboard = np.hstack((vis_tracking, vis_map))
        dashboard_resized = cv2.resize(dashboard, (1152, 576), interpolation=cv2.INTER_AREA)

        # Save frame to output .mp4
        out_video.write(dashboard_resized)

        # Update frame state
        prev_gray = gray.copy()
        
        # Limit frames for automated testing (headless loop)
        if frame_idx >= 300: # 20 seconds at 15fps
            break

    cap.release()
    out_video.release()
    print(f"\nSimulation complete! Video saved to: {export_path}")

if __name__ == "__main__":
    main()
