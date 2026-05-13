#!/usr/bin/env bash
# Kill processes that block teleop: camera pipelines and RTDE connections.
# Run this before demo_real_robot.py if you hit "Device or resource busy" or
# "RTDE input registers already in use".

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

killed_any=0

kill_pattern() {
    local label="$1"; local pattern="$2"
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}[kill]${NC} $label (pids: $pids)"
        kill $pids 2>/dev/null || true
        killed_any=1
    fi
}

echo "── Stopping known teleop blockers ──────────────────────────────────────"

# Camera-holding scripts
kill_pattern "debug_depth_cameras"    "debug_depth_cameras"
kill_pattern "demo_real_robot"        "demo_real_robot"
kill_pattern "eval_real_robot"        "eval_real_robot"

# RealSense subprocesses (spawn'd by MultiRealsense)
kill_pattern "SingleRealsense worker" "single_realsense"

# RTDE / robot controller subprocesses
kill_pattern "RTDEOSCController"      "rtde_interpolation_controller"
kill_pattern "RTDEControlInterface"   "RTDEControl"

# Catch-all: any Python process still holding a /dev/video* node
VIDEO_PIDS=$(fuser /dev/video* 2>/dev/null | tr ' ' '\n' | sort -u | grep -v '^$' || true)
if [ -n "$VIDEO_PIDS" ]; then
    echo -e "${YELLOW}[kill]${NC} processes holding /dev/video* ($VIDEO_PIDS)"
    kill $VIDEO_PIDS 2>/dev/null || true
    killed_any=1
fi

# Give processes a moment to exit cleanly, then SIGKILL stragglers
if [ "$killed_any" -eq 1 ]; then
    sleep 1
    # SIGKILL anything that didn't die
    for pattern in debug_depth_cameras demo_real_robot eval_real_robot \
                   single_realsense rtde_interpolation_controller; do
        stragglers=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [ -n "$stragglers" ]; then
            echo -e "${RED}[force-kill]${NC} $pattern still alive — sending SIGKILL"
            kill -9 $stragglers 2>/dev/null || true
        fi
    done
    VIDEO_PIDS=$(fuser /dev/video* 2>/dev/null | tr ' ' '\n' | sort -u | grep -v '^$' || true)
    if [ -n "$VIDEO_PIDS" ]; then
        echo -e "${RED}[force-kill]${NC} /dev/video* still held — sending SIGKILL to $VIDEO_PIDS"
        kill -9 $VIDEO_PIDS 2>/dev/null || true
    fi
    sleep 0.5
fi

echo "── Verification ────────────────────────────────────────────────────────"

# Check cameras are free
STILL_HELD=$(fuser /dev/video* 2>/dev/null | tr ' ' '\n' | sort -u | grep -v '^$' || true)
if [ -n "$STILL_HELD" ]; then
    echo -e "${RED}[WARN]${NC} /dev/video* still held by: $STILL_HELD"
else
    echo -e "${GREEN}[OK]${NC}  cameras free"
fi

# Check no blockers remain
REMAINING=$(pgrep -f "demo_real_robot\|debug_depth_cameras\|eval_real_robot\|rtde_interpolation_controller" 2>/dev/null || true)
if [ -n "$REMAINING" ]; then
    echo -e "${RED}[WARN]${NC} blocker processes still running: $REMAINING"
else
    echo -e "${GREEN}[OK]${NC}  no blocker processes"
fi

if [ "$killed_any" -eq 0 ]; then
    echo -e "${GREEN}[OK]${NC}  nothing to kill — already clear"
fi

echo "────────────────────────────────────────────────────────────────────────"
echo "Ready to run demo_real_robot.py"
