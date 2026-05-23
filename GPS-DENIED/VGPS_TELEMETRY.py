"""
VGPS_TELEMETRY.py — Vision-Based GPS with Real-Time UDP Position Telemetry.

Runs the full terrain-matching pipeline on a RADXA ZERO 3W:
  Camera → ORB Feature Matching → MBTiles Homography → WGS84 Estimate
  → MAVLink VISION_POSITION_ESTIMATE (Pixhawk UART)
  → UDP position packet (ground station / companion logger)

UDP Packet Layout (Big-Endian):
  [0x7E][len][0x10][lat_i32][lon_i32][alt_i16][yaw_i16][fix][crc8]
    B    B    B       i          i       h         h      B    B

  - lat_i32 : latitude  in microdegrees  (int(lat * 1e6))
  - lon_i32 : longitude in microdegrees  (int(lon * 1e6))
  - alt_i16 : altitude  in centimetres   (int(alt * 100))
  - yaw_i16 : heading   in centidegrees  (int(yaw * 100))
  - fix     : 0x01 = valid terrain lock, 0x00 = tracking lost
  - crc8    : DVB-S2 of full payload (identical to DRONE_TRACKER.py)

Usage:
  python VGPS_TELEMETRY.py                         # live camera, MAVLink + UDP
  python VGPS_TELEMETRY.py --no-mavlink            # UDP only (no Pixhawk)
  python VGPS_TELEMETRY.py --no-preview            # headless / SBC mode
  python VGPS_TELEMETRY.py --ip 192.168.1.10       # send to GCS laptop
  python VGPS_TELEMETRY.py --port 14550            # custom UDP port
  python VGPS_TELEMETRY.py --source 1              # camera index 1
  python VGPS_TELEMETRY.py --zoom 17               # override map zoom level
"""

import argparse
import socket
import struct
import sys
import time
import signal
import logging
import threading

import cv2
import numpy as np

import config
from camera import CaptureThread
from vision import preprocess_image, VGPSVisionPipeline
from geospatial import GeospatialDatabase, CoordinateTransformer, wgs84_to_slippy

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VGPS")

# ---------------------------------------------------------------------------
# CRC8 helper (DVB-S2) — mirrors DRONE_TRACKER.py
# ---------------------------------------------------------------------------
try:
    from fastcrc import crc8 as fastcrc_crc8
    HAS_CRC = True
except ImportError:
    HAS_CRC = False
    logger.warning("fastcrc not found — UDP packets sent WITHOUT CRC. "
                   "Install: pip install fastcrc")


def _crc8(data: bytes) -> int:
    if HAS_CRC:
        return fastcrc_crc8.dvb_s2(data)
    return 0


# ---------------------------------------------------------------------------
# UDP Position Sender
# ---------------------------------------------------------------------------
class PositionUDPSender:
    """
    Sends real-time drone position over UDP using a compact binary protocol.

    Packet (15 bytes total, Big-Endian):
      0x7E  LEN  0x10  lat_i32  lon_i32  alt_i16  yaw_i16  fix  crc8
       B     B    B       i        i        h         h      B    B
    """

    HEADER = 0x7E
    # Payload after header+len+type: 4+4+2+2+1 = 13 bytes
    _PAYLOAD_FMT = "!BBBiihbB"   # will add crc separately
    _DATA_FMT    = "!BBBiihbB"

    def __init__(self, ip: str, port: int):
        self.ip      = ip
        self.port    = port
        self.sock    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.enabled = True
        self.count   = 0

    # lat/lon in degrees, alt in metres, yaw in degrees, fix bool
    def send(self, lat: float, lon: float, alt: float, yaw: float, fix: bool) -> str | None:
        if not self.enabled:
            return None

        lat_i = int(lat * 1_000_000)   # microdegrees
        lon_i = int(lon * 1_000_000)
        alt_h = int(alt * 100)          # centimetres  (int16 → max ±327 m, fine for UAV)
        yaw_h = int((yaw % 360.0) * 100)  # centidegrees 0–35999
        fix_b = 0x01 if fix else 0x00

        # Build payload (header + data, no crc yet)
        #   struct layout:  B B B i i h H B
        # yaw is 0-35999 so use unsigned short H not signed h
        inner = struct.pack("!iihHB", lat_i, lon_i, alt_h, yaw_h, fix_b)
        pkt_len = len(inner)  # 13 bytes
        header  = struct.pack("!BBB", self.HEADER, pkt_len, config.VGPS_PACKET_TYPE)
        payload = header + inner
        crc     = _crc8(payload)
        data    = payload + struct.pack("!B", crc)

        self.sock.sendto(data, (self.ip, self.port))
        self.count += 1
        hex_str = data.hex().upper()
        logger.info(
            f"[UDP #{self.count}] fix={'OK' if fix else '--'} "
            f"lat={lat:.6f} lon={lon:.6f} alt={alt:.1f}m yaw={yaw:.1f}° | "
            f"0x{hex_str}"
        )
        return hex_str

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled


# ---------------------------------------------------------------------------
# Position State (shared between processing and telemetry threads)
# ---------------------------------------------------------------------------
class PositionState:
    def __init__(self):
        self._lock = threading.Lock()
        self._lat   = config.TAKEOFF_LAT
        self._lon   = config.TAKEOFF_LON
        self._alt   = config.DEFAULT_ALTITUDE
        self._yaw   = 0.0
        self._valid = False

    def update(self, lat, lon, alt, yaw):
        with self._lock:
            self._lat   = lat
            self._lon   = lon
            self._alt   = alt
            self._yaw   = yaw
            self._valid = True

    def invalidate(self):
        with self._lock:
            self._valid = False

    def snapshot(self):
        with self._lock:
            return self._lat, self._lon, self._alt, self._yaw, self._valid


# ---------------------------------------------------------------------------
# MAVLink telemetry (optional)
# ---------------------------------------------------------------------------
def _connect_mavlink(port, baud):
    try:
        from pymavlink import mavutil
        conn = mavutil.mavlink_connection(
            port, baud=baud,
            source_system=config.MAVLINK_SYSTEM_ID,
            source_component=config.MAVLINK_COMPONENT_ID
        )
        logger.info(f"MAVLink connected on {port} @ {baud}")
        return conn
    except Exception as e:
        logger.warning(f"MAVLink unavailable ({e}) — running in UDP-only mode.")
        return None


def _send_mavlink(conn, lat, lon, alt, yaw):
    cov = [0.0] * 21
    cov[0] = cov[5] = 0.01
    cov[9] = 0.05
    cov[20] = 0.02
    try:
        conn.mav.vision_position_estimate_send(
            usec=int(time.time() * 1e6),
            x=lat, y=lon, z=alt,
            roll=0.0, pitch=0.0, yaw=yaw,
            covariance=cov,
            reset_counter=0
        )
    except Exception as e:
        logger.error(f"MAVLink TX error: {e}")


# ---------------------------------------------------------------------------
# Processing thread
# ---------------------------------------------------------------------------
class ProcessingThread(threading.Thread):
    def __init__(self, capture_thread: CaptureThread, state: PositionState,
                 zoom: int):
        super().__init__(name="ProcessingThread", daemon=True)
        self.capture = capture_thread
        self.state   = state
        self.zoom    = zoom

        self.pipeline = VGPSVisionPipeline()
        self.db       = GeospatialDatabase()

        self._running = False

        # Reference tile cache (avoids re-extracting ORB on every frame)
        self._cached_tile_xy   = (None, None)
        self._cached_tile_kp   = None
        self._cached_tile_desc = None

        self._last_lat = config.TAKEOFF_LAT
        self._last_lon = config.TAKEOFF_LON

    def _get_ref_features(self, lat, lon):
        x, y = wgs84_to_slippy(lat, lon, self.zoom)
        if (x, y) == self._cached_tile_xy and self._cached_tile_desc is not None:
            return self._cached_tile_kp, self._cached_tile_desc, x, y

        logger.info(f"Loading reference tile X={x} Y={y} Z={self.zoom}...")
        tile_img, fx, fy = self.db.fetch_tile_by_wgs84(lat, lon, self.zoom)
        if tile_img is None:
            logger.error(f"Tile X={x} Y={y} not in database.")
            return None, None, x, y

        binary = preprocess_image(tile_img, method='adaptive')
        kp, desc = self.pipeline.extract_features(binary)
        logger.info(f"Reference tile: {len(kp)} ORB keypoints.")

        self._cached_tile_xy   = (x, y)
        self._cached_tile_kp   = kp
        self._cached_tile_desc = desc
        return kp, desc, x, y

    def run(self):
        self._running = True
        logger.info("ProcessingThread started.")
        frame_count = 0
        t0 = time.time()

        while self._running:
            frame = self.capture.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            ref_kp, ref_desc, x_tile, y_tile = self._get_ref_features(
                self._last_lat, self._last_lon)

            if ref_desc is None or len(ref_desc) == 0:
                self.state.invalidate()
                time.sleep(0.01)
                continue

            binary_live = preprocess_image(frame, method='adaptive')
            live_kp, live_desc = self.pipeline.extract_features(binary_live)

            if len(live_desc) < 8:
                logger.debug("Too few live features — tracking lost.")
                self.state.invalidate()
                time.sleep(0.01)
                continue

            good_matches = self.pipeline.match_features(live_desc, ref_desc)
            H, _ = self.pipeline.estimate_homography(live_kp, ref_kp, good_matches)

            if H is not None:
                px, py = self.pipeline.resolve_center_pixel(
                    H, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

                if px is not None and py is not None:
                    margin = 64.0
                    if (-margin <= px <= 256.0 + margin) and (-margin <= py <= 256.0 + margin):
                        transformer = CoordinateTransformer(x_tile, y_tile, self.zoom)
                        lat_est, lon_est = transformer.pixel_to_wgs84(px, py)
                        self._last_lat = lat_est
                        self._last_lon = lon_est

                        yaw_rad = np.arctan2(H[1, 0], H[0, 0])
                        yaw_deg = float(np.degrees(yaw_rad)) % 360.0

                        self.state.update(lat_est, lon_est,
                                          config.DEFAULT_ALTITUDE, yaw_deg)
                        logger.debug(
                            f"Position: {lat_est:.6f}, {lon_est:.6f} "
                            f"yaw={yaw_deg:.1f}° matches={len(good_matches)}")
                    else:
                        self.state.invalidate()
                else:
                    self.state.invalidate()
            else:
                logger.debug(f"Homography failed ({len(good_matches)} matches).")
                self.state.invalidate()

            frame_count += 1
            elapsed = time.time() - t0
            if elapsed >= 5.0:
                logger.info(
                    f"Processing: {frame_count/elapsed:.1f} FPS | "
                    f"matches={len(good_matches)} | "
                    f"lat={self._last_lat:.6f} lon={self._last_lon:.6f}")
                frame_count = 0
                t0 = time.time()

            time.sleep(0.001)

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Telemetry broadcast thread
# ---------------------------------------------------------------------------
class BroadcastThread(threading.Thread):
    """Reads PositionState and fans out to UDP + optional MAVLink at fixed Hz."""

    def __init__(self, state: PositionState, udp: PositionUDPSender,
                 mav_conn, hz: float):
        super().__init__(name="BroadcastThread", daemon=True)
        self.state    = state
        self.udp      = udp
        self.mav_conn = mav_conn
        self.interval = 1.0 / hz
        self._running = False

    def run(self):
        self._running = True
        logger.info(
            f"BroadcastThread started @ {1/self.interval:.1f} Hz → "
            f"UDP {self.udp.ip}:{self.udp.port}"
        )
        while self._running:
            t0 = time.time()
            lat, lon, alt, yaw, valid = self.state.snapshot()

            # Always send UDP (fix flag tells receiver whether it's valid)
            self.udp.send(lat, lon, alt, yaw, valid)

            # MAVLink only when estimate is valid (avoid EKF corruption)
            if valid and self.mav_conn is not None:
                _send_mavlink(self.mav_conn, lat, lon, alt, yaw)

            elapsed = time.time() - t0
            delay = self.interval - elapsed
            if delay > 0:
                time.sleep(delay)

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Preview window (optional, for ground debugging)
# ---------------------------------------------------------------------------
def _draw_hud(frame, lat, lon, alt, yaw, valid, pkt_count, pkt_hex):
    h, w = frame.shape[:2]

    # Semi-transparent overlay strip at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    fix_color = (0, 255, 100) if valid else (0, 80, 255)
    fix_label = "TERRAIN LOCK" if valid else "TRACKING LOST"
    cv2.putText(frame, fix_label,      (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fix_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"LAT: {lat:.6f}",  (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"LON: {lon:.6f}",  (12, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"YAW: {yaw:.1f} deg   ALT: {alt:.1f} m   PKTS: {pkt_count}",
                (12, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 255), 1, cv2.LINE_AA)

    # TX banner at bottom
    if pkt_hex:
        banner_h = 28
        ov2 = frame.copy()
        cv2.rectangle(ov2, (0, h - banner_h), (w, h), (5, 5, 5), -1)
        cv2.addWeighted(ov2, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"TX: 0x{pkt_hex}",
                    (10, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    # Cross-hair at frame centre
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 200), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 200), 1)
    cv2.circle(frame, (cx, cy), 5, (0, 200, 255), 1)
    return frame


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run(args):
    # Connect MAVLink (optional)
    mav_conn = None
    if not args.no_mavlink:
        mav_conn = _connect_mavlink(config.SERIAL_PORT, config.BAUD_RATE)

    # UDP sender
    udp_ip   = args.ip   or config.POSITION_UDP_IP
    udp_port = args.port or config.POSITION_UDP_PORT
    udp = PositionUDPSender(udp_ip, udp_port)
    logger.info(f"UDP position telemetry → {udp_ip}:{udp_port}")

    # Shared position state
    state = PositionState()

    # Camera
    cam_idx = int(args.source) if str(args.source).isdigit() else 0
    capture = CaptureThread(
        camera_index=cam_idx,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        fps=config.CAMERA_FPS
    )

    # Processing
    zoom = args.zoom or config.MAP_ZOOM_LEVEL
    proc = ProcessingThread(capture, state, zoom)

    # Broadcast
    broadcast = BroadcastThread(state, udp, mav_conn, config.TELEMETRY_UPDATE_HZ)

    # OS signal handlers
    threads = [capture, proc, broadcast]
    def _shutdown(sig, _frame):
        logger.warning(f"Signal {sig} received — shutting down...")
        for t in threads:
            t.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Launch
    logger.info("=" * 52)
    logger.info("   VGPS TERRAIN MATCHING + UDP TELEMETRY ACTIVE   ")
    logger.info("=" * 52)
    capture.start()
    time.sleep(0.3)   # give camera time to open
    proc.start()
    broadcast.start()

    last_hex = None

    if args.no_preview:
        logger.info("Headless mode. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1.0)
        except (KeyboardInterrupt, SystemExit):
            pass
    else:
        win = "VGPS — Terrain Matching | ESC=Quit | S=Toggle UDP"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 800, 600)

        while True:
            frame = capture.get_frame()
            if frame is not None:
                lat, lon, alt, yaw, valid = state.snapshot()
                if udp.count > 0:
                    last_hex = None   # reset — will be filled by logger
                display = _draw_hud(
                    frame.copy(), lat, lon, alt, yaw, valid,
                    udp.count, last_hex)
                cv2.imshow(win, display)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('s'):
                enabled = udp.toggle()
                logger.info(f"UDP {'enabled' if enabled else 'disabled'}")

        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

    # Clean shutdown
    for t in threads:
        t.stop()
    for t in threads:
        t.join(timeout=2.0)

    logger.info(f"Shutdown complete. Total UDP packets sent: {udp.count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VGPS_TELEMETRY — Vision-Based GPS with UDP + MAVLink output"
    )
    parser.add_argument("--source",     default="0",
                        help="Camera index (default: 0)")
    parser.add_argument("--ip",         default=None,
                        help=f"UDP destination IP (default: {config.POSITION_UDP_IP})")
    parser.add_argument("--port",       type=int, default=None,
                        help=f"UDP destination port (default: {config.POSITION_UDP_PORT})")
    parser.add_argument("--zoom",       type=int, default=None,
                        help=f"MBTiles zoom level (default: {config.MAP_ZOOM_LEVEL})")
    parser.add_argument("--no-mavlink", action="store_true",
                        help="Disable MAVLink output (UDP only)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Headless mode — no OpenCV window (use on SBC)")
    args = parser.parse_args()
    run(args)
