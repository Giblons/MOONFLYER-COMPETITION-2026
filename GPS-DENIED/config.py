"""
Configuration parameters for the Vision-Based Global Positioning System (VGPS).
Contains hardware interface settings, camera intrinsic calibration matrices,
computer vision thresholds, and geospatial configuration parameters.
"""

import numpy as np

# --- Camera & Capture Configuration ---
CAMERA_INDEX = 0  # /dev/video0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640
CAMERA_FPS = 60

# Equidistant Fisheye Lens Calibration Coefficients
# K: 3x3 Intrinsic Camera Matrix
# D: 4-parameter Distortion Coefficients [k1, k2, k3, k4]
K = np.array([
    [380.0,   0.0, 320.0],
    [  0.0, 380.0, 320.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

D = np.array([-0.05, 0.015, -0.005, 0.0002], dtype=np.float64)

# Bounding box extraction balance (0.0 crops out all black pixels, analogous to alpha=0)
UNDISTORT_BALANCE = 0.0

# --- Computer Vision & Matcher Settings ---
ORB_MAX_FEATURES = 3000          # Increased from 2000 for denser match coverage
LOWE_RATIO_THRESHOLD = 0.75      # Tightened from 0.70 — rejects more ambiguous matches

# FLANN LSH Matcher Configuration
# algorithm=6 refers to FLANN_INDEX_LSH
FLANN_INDEX_LSH = 6
FLANN_INDEX_PARAMS = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=6,      # Number of hash tables
    key_size=12,         # Size of key in bits
    multi_probe_level=1  # Number of bits to shift
)
FLANN_SEARCH_PARAMS = dict(checks=50)

# --- Geospatial & MBTiles Settings ---
MAP_ZOOM_LEVEL = 18  # Web Mercator Zoom Level

# Primary offline satellite tile database (built by mbtiles_generator.py)
DATABASE_PATH = "/home/rizky/GPS-DENIED/uav_vision_map.mbtiles"

# Reference Initial/Takeoff Location (matches mbtiles_generator.py center coords)
TAKEOFF_LAT = 23.725018
TAKEOFF_LON = 120.374205
DEFAULT_ALTITUDE = 10.0  # Default height above ground in metres (use rangefinder if available)

# --- Telemetry & MAVLink Settings ---
# Pixhawk UART on Radxa Zero 3W: usually /dev/ttyS0, /dev/ttyS3, or /dev/ttyAML1
SERIAL_PORT = "/dev/ttyS0"
BAUD_RATE = 115200
MAVLINK_SYSTEM_ID = 1
MAVLINK_COMPONENT_ID = 1
TELEMETRY_UPDATE_HZ = 15.0  # 10-30 Hz keeps EKF3 happy

# --- UDP Position Telemetry (VGPS_TELEMETRY.py) ---
# Change POSITION_UDP_IP to your GCS/laptop IP for remote reception
POSITION_UDP_IP   = "127.0.0.1"
POSITION_UDP_PORT = 13131
VGPS_PACKET_TYPE  = 0x10  # Distinct from DRONE_TRACKER (0x02)
