"""
test_vgps_sim.py — Advanced WGS84 VGPS Agricultural Flight Simulator & Navigation Engine

Upgraded with high-accuracy agricultural navigation methods:
  1. Preprocessing: Excess Green Index (ExG) to isolate green vegetation and highlight mud dikes.
  2. Extractor: AKAZE Feature Detection & Matching for robust scale/rotation organic texture locking.
  3. Continuous Tracking: Lucas-Kanade (LK) Optical Flow Visual Odometry (VO) as a seamless fallback
     when terrain matching locks are dropped (e.g. over repetitive/textureless fields).
  4. Mud Bund Boundary Tracking: Live Hough line dike detection overlay (Neon Green lines).
  5. Silent Headless Export: Writes full premium telemetry and side-by-side matches to
     `exports/tracked_vgps_simulation.mp4` without requiring GUI display drivers.

Usage:
  python test_vgps_sim.py --video down1    # Downward Nadir Video (Recommended)
  python test_vgps_sim.py --video down2
"""

import os
import sys
import time
import struct
import socket
import logging
import argparse
import cv2
import numpy as np

# Set up Argument Parser
parser = argparse.ArgumentParser(description="Advanced WGS84 VGPS Agricultural Drone Flight Simulator")
parser.add_argument(
    "--video",
    type=str,
    default="down1",
    choices=["down1", "down2", "front1", "front2"],
    help="Select the test video to play (default: down1)"
)
parser.add_argument(
    "--lat",
    type=float,
    default=None,
    help="Initial Latitude (overrides config.TAKEOFF_LAT)"
)
parser.add_argument(
    "--lon",
    type=float,
    default=None,
    help="Initial Longitude (overrides config.TAKEOFF_LON)"
)
args = parser.parse_args()

# Map options to video paths
video_map = {
    "down1": "/home/rizky/object-detection/down1.mp4",
    "down2": "/home/rizky/object-detection/down2.mp4",
    "front1": "/home/rizky/object-detection/front1.mp4",
    "front2": "/home/rizky/object-detection/front2.mp4"
}
video_path = video_map[args.video]

# 1. Programmatically configure settings BEFORE importing main pipeline modules
import config
MOCK_DB_PATH = "/home/rizky/GPS-DENIED/uav_vision_map(1).mbtiles"
REAL_DB_PATH = "/home/rizky/GPS-DENIED/uav_vision_map(1).mbtiles"

# Auto-detect real database or default to mock
using_mock = True
if os.path.exists(REAL_DB_PATH) and os.path.getsize(REAL_DB_PATH) > 1024 * 1024:
    config.DATABASE_PATH = REAL_DB_PATH
    using_mock = False
else:
    config.DATABASE_PATH = MOCK_DB_PATH

config.POSITION_UDP_PORT = 13131  # Match the Ground Handler port
config.POSITION_UDP_IP = "127.0.0.1"

from geospatial import GeospatialDatabase, CoordinateTransformer, wgs84_to_slippy

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VGPS.Engine")

# Ensure fastcrc is supported
try:
    from fastcrc import crc8 as fastcrc_crc8
    HAS_CRC = True
except ImportError:
    HAS_CRC = False

def get_crc8(data: bytes) -> int:
    return fastcrc_crc8.dvb_s2(data) if HAS_CRC else 0

def get_excess_green(frame):
    """Isolates crops and vegetation: ExG = 2*G - R - B."""
    b, g, r = cv2.split(frame.astype(np.float32))
    exg = 2 * g - r - b
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return exg

def get_dike_lines(frame):
    """Detects linear crop bund dividers and canals using Canny + Hough Transform."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 80, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=50,
        minLineLength=60, maxLineGap=15
    )
    return lines

class PositionUDPSender:
    """Sends simulation position telemetry to local data-handler."""
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.count = 0

    def send(self, lat: float, lon: float, alt: float, yaw: float, fix: bool) -> str:
        lat_i = int(lat * 1_000_000)
        lon_i = int(lon * 1_000_000)
        alt_h = int(alt * 100)
        yaw_h = int((yaw % 360.0) * 100)
        fix_b = 0x01 if fix else 0x00

        inner = struct.pack("!iihHB", lat_i, lon_i, alt_h, yaw_h, fix_b)
        pkt_len = len(inner)
        header = struct.pack("!BBB", 0x7E, pkt_len, config.VGPS_PACKET_TYPE)
        payload = header + inner
        crc = get_crc8(payload)
        data = payload + struct.pack("!B", crc)

        self.sock.sendto(data, (self.ip, self.port))
        self.count += 1
        return data.hex().upper()

def draw_premium_hud(frame, lat, lon, alt, yaw, matches_count, valid, pkt_count, pkt_hex, video_name, db_mode, vo_active):
    """Draws a premium sci-fi style HUD overlay on the visualization canvas."""
    h, w = frame.shape[:2]

    # Dark gradient top panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 105), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Status labels
    if valid:
        fix_color = (0, 255, 120)
        fix_text = "TERRAIN MATCH LOCK: ACTIVE (AKAZE+ORB DUAL HIGH-CONFIDENCE)"
    elif vo_active:
        fix_color = (255, 255, 255)
        fix_text = "TERRAIN LOCK LOSS: ACTIVE (LK OPTICAL FLOW FALLBACK)"
    else:
        fix_color = (0, 80, 255)
        fix_text = "TERRAIN MATCH: ACQUIRING SEARCH BLOCK..."
    
    cv2.putText(frame, fix_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, fix_color, 2, cv2.LINE_AA)
    
    # Coords info
    cv2.putText(frame, f"ESTIMATED LAT : {lat:.6f}", (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ESTIMATED LON : {lon:.6f}", (15, 52 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
    
    # Alt, Yaw, Packets
    stat_line = f"ALTITUDE: {alt:.1f} m  |  HEADING: {yaw:.1f} deg  |  INLIERS: {matches_count}  |  TX PKTS: {pkt_count}"
    cv2.putText(frame, stat_line, (15, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 185, 200), 1, cv2.LINE_AA)

    # Video Source and DB indicator (Top Right)
    cv2.putText(frame, f"SOURCE: {video_name}", (w - 240, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"DB MODE: {db_mode}", (w - 240, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ENGINE : AKAZE+ORB+LK VO", (w - 240, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1, cv2.LINE_AA)

    # Cross-hair at center of camera feed (left half of canvas)
    cx, cy = 320, 320 + 105 # Offset by top HUD panel height
    cv2.circle(frame, (cx, cy), 15, (0, 255, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (cx - 25, cy), (cx - 5, cy), (0, 255, 200), 1)
    cv2.line(frame, (cx + 5, cy), (cx + 25, cy), (0, 255, 200), 1)
    cv2.line(frame, (cx, cy - 25), (cx, cy - 5), (0, 255, 200), 1)
    cv2.line(frame, (cx, cy + 5), (cx, cy + 25), (0, 255, 200), 1)

    # Telemetry hex output banner at the bottom
    if pkt_hex:
        banner_h = 32
        ov_bottom = frame.copy()
        cv2.rectangle(ov_bottom, (0, h - banner_h), (w, h), (10, 10, 12), -1)
        cv2.addWeighted(ov_bottom, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, f"TX TELEMETRY PACKET: 0x{pkt_hex}", (15, h - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1, cv2.LINE_AA)

def main():
    # Make sure output directories exist
    os.makedirs("/home/rizky/GPS-DENIED/exports", exist_ok=True)
    export_path = f"/home/rizky/GPS-DENIED/exports/tracked_vgps_simulation.mp4"

    logger.info("=====================================================")
    logger.info("    INITIALIZING HYBRID WGS84 VGPS+VO FLIGHT ENGINE  ")
    logger.info("=====================================================")

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        sys.exit(1)

    db_mode = "MOCK SYNTHETIC" if using_mock else "REAL MBTILES"
    db_path_to_use = MOCK_DB_PATH if using_mock else REAL_DB_PATH
    logger.info(f"Source Video Path    : {video_path}")
    logger.info(f"Map GIS Database     : {db_path_to_use} ({db_mode})")

    # Initialize components
    db = GeospatialDatabase(db_path_to_use)
    udp_sender = PositionUDPSender(config.POSITION_UDP_IP, config.POSITION_UDP_PORT)

    # --- Dual Extractor: Tuned AKAZE (float) + ORB (binary) ---
    akaze = cv2.AKAZE_create(
        threshold=0.0005,               # Default 0.001 → more keypoints in low-contrast fields
        nOctaves=4,
        nOctaveLayers=4,
        diffusivity=cv2.KAZE_DIFF_PM_G2
    )
    orb = cv2.ORB_create(
        nfeatures=3000,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=10,               # Sensitive to low-contrast terrain corners
        scoreType=cv2.ORB_FAST_SCORE
    )
    # FLANN for AKAZE: float descriptors → KD-Tree (algorithm=1)
    flann_akaze = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )
    # FLANN for ORB: binary descriptors → LSH (algorithm=6)
    flann_orb = cv2.FlannBasedMatcher(
        dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
        dict(checks=50)
    )
    
    # Setup Video Writer
    frame_w = 640 + 640  # Live camera (640x640) + Reference tile (640x640 scaled)
    frame_h = 640 + 105 + 32  # 640 image + 105 HUD panel + 32 hex banner
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(export_path, fourcc, fps, (frame_w, frame_h))
    
    logger.info(f"Writing advanced tracking telemetry to: {export_path}")

    # Set initial coordinates (Use CLI arguments if provided, else fallback to config.py)
    lat = args.lat if args.lat is not None else config.TAKEOFF_LAT
    lon = args.lon if args.lon is not None else config.TAKEOFF_LON
    alt = config.DEFAULT_ALTITUDE
    yaw = 0.0

    logger.info(f"Initializing VGPS starting coordinates at LAT: {lat:.6f}, LON: {lon:.6f}")

    # Reference tile variables (AKAZE = _a suffix, ORB = _o suffix)
    cached_tile_coords = (None, None)
    ref_kp_a, ref_desc_a = None, None
    ref_kp_o, ref_desc_o = None, None
    x_tile, y_tile = None, None

    # Visual Odometry variables
    prev_gray = None
    prev_pts = None
    lk_params = dict(
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    frame_idx = 0
    try:
        while True:
            # 1. Fetch real video frame
            ret, raw_frame = cap.read()
            if not ret:
                logger.info("Video end reached. Completing flight simulation...")
                break

            # Resize the video frame to exactly 640x640 for the visual pipeline layout
            live_frame = cv2.resize(raw_frame, (640, 640))
            gray_live = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)

            # 2. Advanced Preprocessing: Extract Excess Green Index (ExG)
            exg_live = get_excess_green(live_frame)

            # 3. Get/cache reference tile features — AKAZE + ORB on ExG
            x_tile_curr, y_tile_curr = wgs84_to_slippy(lat, lon, config.MAP_ZOOM_LEVEL)
            if (x_tile_curr, y_tile_curr) != cached_tile_coords:
                logger.debug(f"Loading reference tile for X={x_tile_curr}, Y={y_tile_curr}")
                tile_img, _, _ = db.fetch_tile_by_wgs84(lat, lon, config.MAP_ZOOM_LEVEL)
                exg_tile = get_excess_green(tile_img)
                ref_kp_a, ref_desc_a = akaze.detectAndCompute(exg_tile, None)
                ref_kp_o, ref_desc_o = orb.detectAndCompute(exg_tile, None)
                cached_tile_coords = (x_tile_curr, y_tile_curr)
                x_tile, y_tile = x_tile_curr, y_tile_curr

            # 4. Extract live features — AKAZE + ORB (both on ExG)
            live_kp_a, live_desc_a = akaze.detectAndCompute(exg_live, None)
            live_kp_o, live_desc_o = orb.detectAndCompute(exg_live, None)

            # 5. Dual FLANN Matching — pool src/dst pixel coordinates from both extractors
            LOWE = 0.75
            src_pts_list, dst_pts_list = [], []
            n_akaze_matches = 0  # boundary index for visualization coloring

            # AKAZE matches (KD-Tree FLANN, float descriptors)
            if (live_desc_a is not None and ref_desc_a is not None
                    and len(live_desc_a) >= 2 and len(ref_desc_a) >= 2):
                try:
                    for m, n in flann_akaze.knnMatch(live_desc_a, ref_desc_a, k=2):
                        if m.distance < LOWE * n.distance:
                            src_pts_list.append(live_kp_a[m.queryIdx].pt)
                            dst_pts_list.append(ref_kp_a[m.trainIdx].pt)
                except Exception:
                    pass
            n_akaze_matches = len(src_pts_list)

            # ORB matches (LSH FLANN, binary descriptors)
            if (live_desc_o is not None and ref_desc_o is not None
                    and len(live_desc_o) >= 2 and len(ref_desc_o) >= 2):
                try:
                    for m, n in flann_orb.knnMatch(live_desc_o, ref_desc_o, k=2):
                        if m.distance < LOWE * n.distance:
                            src_pts_list.append(live_kp_o[m.queryIdx].pt)
                            dst_pts_list.append(ref_kp_o[m.trainIdx].pt)
                except Exception:
                    pass

            # 6. RANSAC Homography on pooled dual-extractor point cloud (4.0px tolerance)
            valid_absolute = False
            inliers_count = 0
            H, mask = None, None

            if len(src_pts_list) >= 8:
                src_pts = np.float32(src_pts_list).reshape(-1, 1, 2)
                dst_pts = np.float32(dst_pts_list).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

                if H is not None and mask is not None:
                    inliers_count = int(np.sum(mask))
                    if inliers_count >= 8:
                        pts_center = np.float32([[[320, 320]]])
                        proj_center = cv2.perspectiveTransform(pts_center, H)
                        px, py = proj_center[0][0]

                        margin = 64.0
                        if (-margin <= px <= 256.0 + margin) and (-margin <= py <= 256.0 + margin):
                            transformer = CoordinateTransformer(x_tile, y_tile, config.MAP_ZOOM_LEVEL)
                            lat, lon = transformer.pixel_to_wgs84(px, py)
                            yaw_rad = np.arctan2(H[1, 0], H[0, 0])
                            yaw = float(np.degrees(yaw_rad)) % 360.0
                            valid_absolute = True

            # 7. Lucas-Kanade Visual Odometry Fallback (denser + median-filtered)
            vo_triggered = False
            dx, dy = 0.0, 0.0

            if prev_gray is not None and prev_pts is not None and len(prev_pts) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_live, prev_pts, None, **lk_params)
                good_new = next_pts[status == 1]
                good_old = prev_pts[status == 1]

                if len(good_new) > 0:
                    translation = good_new - good_old
                    # Median filter rejects outlier motion vectors from vegetation noise
                    dx, dy = np.median(translation, axis=0)
                    prev_pts = good_new.reshape(-1, 1, 2)
                else:
                    prev_pts = None
            else:
                prev_pts = cv2.goodFeaturesToTrack(gray_live, maxCorners=200, qualityLevel=0.01, minDistance=7)

            # Re-seed with 200 corners when tracked points drop below 50
            if prev_pts is not None and len(prev_pts) < 50:
                prev_pts = cv2.goodFeaturesToTrack(gray_live, maxCorners=200, qualityLevel=0.01, minDistance=7)

            # Fuse: if terrain matching failed, apply VO displacement
            if not valid_absolute:
                if prev_gray is not None and (abs(dx) > 0.01 or abs(dy) > 0.01):
                    scale_factor = 0.59 / 111139.0
                    lat -= dy * scale_factor
                    lon += dx * scale_factor * np.cos(np.radians(lat))
                    vo_triggered = True

            # Save state for next iteration
            prev_gray = gray_live.copy()

            # 8. Send MAVLink Emulated UDP position packet (type 0x10)
            pkt_hex = udp_sender.send(lat, lon, alt, yaw, (valid_absolute or vo_triggered))

            # 9. Boundary tracking: Extract mud bunds & canals for visualization
            dike_lines = get_dike_lines(live_frame)

            # 10. Render side-by-side mapping visualization canvas
            tile_original, _, _ = db.fetch_tile_by_wgs84(lat, lon, config.MAP_ZOOM_LEVEL)
            tile_scaled = cv2.resize(tile_original, (640, 640), interpolation=cv2.INTER_LINEAR)

            # Draw detected boundaries on live frame (Neon Green)
            vis_live = live_frame.copy()
            if dike_lines is not None:
                for line in dike_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(vis_live, (x1, y1), (x2, y2), (0, 255, 120), 2, cv2.LINE_AA)

            # Combine live frame and reference tile side-by-side
            canvas = np.hstack((vis_live, tile_scaled))

            # Draw RANSAC inlier match lines (AKAZE=cyan, ORB=orange)
            if H is not None and mask is not None and len(src_pts_list) > 0:
                max_lines_drawn = 30
                drawn_count = 0
                for idx, (sp, dp) in enumerate(zip(src_pts_list, dst_pts_list)):
                    if idx < len(mask) and mask[idx][0] == 1:
                        pt_live = tuple(map(int, sp))
                        pt_ref_x = int(dp[0] * (640.0 / 256.0)) + 640
                        pt_ref_y = int(dp[1] * (640.0 / 256.0))
                        line_color = (255, 200, 0) if idx < n_akaze_matches else (0, 165, 255)
                        cv2.line(canvas, pt_live, (pt_ref_x, pt_ref_y), line_color, 1, cv2.LINE_AA)
                        cv2.circle(canvas, pt_live, 3, (0, 0, 255), -1)
                        cv2.circle(canvas, (pt_ref_x, pt_ref_y), 3, (255, 0, 0), -1)
                        drawn_count += 1
                        if drawn_count >= max_lines_drawn:
                            break

            # 11. Stack top HUD panel and bottom telemetry banner
            top_panel = np.zeros((105, frame_w, 3), dtype=np.uint8)
            bottom_panel = np.zeros((32, frame_w, 3), dtype=np.uint8)
            
            full_frame = np.vstack((top_panel, canvas, bottom_panel))
            
            # Apply premium HUD textures and labels on the full stacked canvas
            draw_premium_hud(
                full_frame, lat, lon, alt, yaw, inliers_count, valid_absolute,
                udp_sender.count, pkt_hex, os.path.basename(video_path), db_mode, vo_triggered
            )

            # 12. Save frame to output .mp4
            out_video.write(full_frame)

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Simulation interrupted.")
    finally:
        # Shutdown resources
        cap.release()
        out_video.release()
        
        # Clean up temporary mock database if it was created
        if using_mock and os.path.exists(MOCK_DB_PATH):
            os.remove(MOCK_DB_PATH)
            
        logger.info("=====================================================")
        logger.info("    SIMULATION COMPLETE — MP4 EXPORTED SUCCESSFUL    ")
        logger.info("=====================================================")

if __name__ == "__main__":
    main()
