#!/usr/bin/env bash
# =============================================================================
# run_data_handler.sh — Data Handler Daemon Launcher
#
# Runs data-handler.py using the GPS-DENIED python virtual environment.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="/home/rizky/GPS-DENIED/.venv"
MAIN="$SCRIPT_DIR/data-handler.py"
LOG="$SCRIPT_DIR/data_handler.log"

# Clean up stale logs
echo "--- Starting new log session ---" > "$LOG"

if [ ! -d "$VENV" ]; then
    echo "[ERROR] Shared virtual environment not found at $VENV"
    exit 1
fi

echo "============================================="
echo "  Ground Data Handler — Radxa/Companion PC"
echo "============================================="
echo "  Venv    : $VENV"
echo "  Script  : $MAIN"
echo "  Log     : $LOG"
echo "  Time    : $(date)"
echo "============================================="
echo ""

"$VENV/bin/python" "$MAIN" 2>&1 | tee "$LOG"
