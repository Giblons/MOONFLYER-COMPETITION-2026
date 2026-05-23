#!/usr/bin/env bash
# =============================================================================
# run_vgps.sh — VGPS Terrain Matching Launcher for Radxa Zero 3W
#
# Usage:
#   ./run_vgps.sh                          # live cam, MAVLink + UDP loopback
#   ./run_vgps.sh --ip 192.168.1.10        # send UDP to GCS laptop
#   ./run_vgps.sh --no-mavlink             # UDP only (no Pixhawk connected)
#   ./run_vgps.sh --no-preview             # force headless (auto on SBC)
#   ./run_vgps.sh --source 1              # use /dev/video1
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
MAIN="$SCRIPT_DIR/VGPS_TELEMETRY.py"
LOG="$SCRIPT_DIR/vgps.log"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
    echo "[ERROR] Virtual environment not found at $VENV"
    echo "        Run: python3 -m venv .venv && source .venv/bin/activate"
    echo "        Then: pip install opencv-python-headless pymavlink fastcrc"
    exit 1
fi

if [ ! -f "$MAIN" ]; then
    echo "[ERROR] VGPS_TELEMETRY.py not found at $MAIN"
    exit 1
fi

DB="$SCRIPT_DIR/uav_vision_map.mbtiles"
if [ ! -f "$DB" ]; then
    echo "[WARN] Offline map database not found: $DB"
    echo "       The pipeline will use its mock tile fallback until"
    echo "       mbtiles_generator.py finishes downloading."
fi

# ---------------------------------------------------------------------------
# ARM / Radxa Zero 3W performance tuning
# ---------------------------------------------------------------------------
# Limit OpenCV to 3 threads (leave 1 core for OS + camera driver)
export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0
export OMP_NUM_THREADS=3

# Prefer V4L2 backend on Linux SBCs (avoids GStreamer init overhead)
export OPENCV_VIDEOIO_BACKEND=v4l2

# Disable Python's GIL write-check warnings on ARM
export PYTHONWARNINGS="ignore"

# ---------------------------------------------------------------------------
# Auto-detect headless mode (no display connected → force --no-preview)
# ---------------------------------------------------------------------------
EXTRA_ARGS="$@"
if [ -z "$DISPLAY" ] && [[ "$@" != *"--no-preview"* ]]; then
    echo "[INFO] No DISPLAY detected — enabling headless mode automatically."
    EXTRA_ARGS="--no-preview $@"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "============================================="
echo "  VGPS Terrain Matching — Radxa Zero 3W"
echo "============================================="
echo "  Venv    : $VENV"
echo "  Script  : $MAIN"
echo "  Log     : $LOG"
echo "  Args    : $EXTRA_ARGS"
echo "  Time    : $(date)"
echo "============================================="
echo ""

"$VENV/bin/python" "$MAIN" $EXTRA_ARGS 2>&1 | tee "$LOG"
