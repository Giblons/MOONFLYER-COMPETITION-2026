#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Unified Launcher for VGPS and DRONE_TRACKER on Radxa Zero 3W
#
# Starts both services in the background and monitors their execution.
# =============================================================================

# Define paths
VGPS_DIR="/home/rizky/GPS-DENIED"
TRACKER_DIR="/home/rizky/object-detection"

# --- CAMERA CONFIGURATION ---
# IMPORTANT: A single camera index (e.g. /dev/video0) cannot be shared by both scripts at the same time.
# Usually, a drone will have:
#   - Downward-facing camera (Index 0) for VGPS terrain matching
#   - Forward-facing / gimbal camera (Index 1) for Object Tracking
VGPS_CAM_INDEX="0"
TRACKER_CAM_INDEX="1"

# --- UDP / TELEMETRY CONFIGURATION ---
# Set GCS Laptop / receiver IP here (or keep localhost 127.0.0.1 for testing)
GCS_IP="127.0.0.1"

echo "========================================================="
echo "   STARTING UNIFIED UAV FLIGHT SOFTWARE DAEMONS"
echo "========================================================="
echo "  VGPS (Nadir Camera Index: $VGPS_CAM_INDEX) → sending UDP to $GCS_IP:15151"
echo "  Tracker (Forward Camera Index: $TRACKER_CAM_INDEX) → sending UDP to $GCS_IP:13131"
echo "========================================================="

# Create log directories if they don't exist
mkdir -p "$VGPS_DIR"
mkdir -p "$TRACKER_DIR"

# Clean up any stale logs
echo "--- Starting new log session ---" > "$VGPS_DIR/vgps.log"
echo "--- Starting new log session ---" > "$TRACKER_DIR/tracker.log"

# Clean exit handler
cleanup() {
    echo ""
    echo "⚠️ Shutting down both software daemons..."
    kill "$VGPS_PID" 2>/dev/null || true
    kill "$TRACKER_PID" 2>/dev/null || true
    wait "$VGPS_PID" 2>/dev/null || true
    wait "$TRACKER_PID" 2>/dev/null || true
    echo "✅ Shutdown complete."
    exit 0
}

trap cleanup SIGINT SIGTERM

# 1. Launch VGPS (Terrain Matching)
echo "🚀 Launching VGPS Terrain Matching..."
cd "$VGPS_DIR"
./run_vgps.sh --no-preview --source "$VGPS_CAM_INDEX" --ip "$GCS_IP" >> "$VGPS_DIR/vgps.log" 2>&1 &
VGPS_PID=$!

# Give VGPS a moment to initialize the camera
sleep 3

# 2. Launch Drone Tracker (YOLO Detection)
echo "🚀 Launching YOLO Target Tracker (Auto-Lock active)..."
cd "$TRACKER_DIR"
./run_tracker.sh --no-preview --auto-lock --source "$TRACKER_CAM_INDEX" --ip "$GCS_IP" >> "$TRACKER_DIR/tracker.log" 2>&1 &
TRACKER_PID=$!

echo "========================================================="
echo "  Both processes running!"
echo "  - VGPS PID: $VGPS_PID"
echo "  - Tracker PID: $TRACKER_PID"
echo "  Monitoring active. Press Ctrl+C to terminate both."
echo "========================================================="

# Monitor both PIDs. If one dies, restart it or exit.
while true; do
    if ! kill -0 "$VGPS_PID" 2>/dev/null; then
        echo "❌ [ALERT] VGPS daemon died! Restarting..."
        cd "$VGPS_DIR"
        ./run_vgps.sh --no-preview --source "$VGPS_CAM_INDEX" --ip "$GCS_IP" >> "$VGPS_DIR/vgps.log" 2>&1 &
        VGPS_PID=$!
    fi
    
    if ! kill -0 "$TRACKER_PID" 2>/dev/null; then
        echo "❌ [ALERT] Tracker daemon died! Restarting..."
        cd "$TRACKER_DIR"
        ./run_tracker.sh --no-preview --auto-lock --source "$TRACKER_CAM_INDEX" --ip "$GCS_IP" >> "$TRACKER_DIR/tracker.log" 2>&1 &
        TRACKER_PID=$!
    fi
    sleep 2
done
