#!/usr/bin/env bash
# Check RealSense camera connection status.
# Prints which of the three expected cameras are connected, their firmware,
# and whether any process is currently holding their video devices.

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

EXPECTED_CAMERAS=(
    "215122255213:D455:front"
    "832112070487:D435:side"
    "746112060198:D415:wrist"
)

echo "── RealSense camera check ──────────────────────────────────────────────"

# Query connected devices via Python/pyrealsense2
# Write to a temp file — conda run doesn't forward heredoc stdin reliably
_PY=$(mktemp /tmp/check_cameras_XXXXXX.py)
cat > "$_PY" <<'EOF'
import pyrealsense2 as rs
ctx = rs.context()
for d in ctx.devices:
    print("|".join([
        d.get_info(rs.camera_info.serial_number),
        d.get_info(rs.camera_info.name),
        d.get_info(rs.camera_info.firmware_version),
        d.get_info(rs.camera_info.product_line),
    ]))
EOF
DEVICE_INFO=$(/home/yandabao/miniforge3/envs/robodiff_real/bin/python3 "$_PY" 2>/dev/null)
rm -f "$_PY"

all_ok=1
for entry in "${EXPECTED_CAMERAS[@]}"; do
    serial="${entry%%:*}"; rest="${entry#*:}"; model="${rest%%:*}"; role="${rest#*:}"
    label="$(printf '%-6s' "$role") ($model, $serial)"

    match=$(echo "$DEVICE_INFO" | grep "^$serial|" || true)
    if [ -z "$match" ]; then
        echo -e "  ${RED}[MISSING]${NC}  $label"
        all_ok=0
    else
        fw=$(echo "$match" | cut -d'|' -f3)
        echo -e "  ${GREEN}[OK]${NC}       $label  fw=$fw"
    fi
done

echo ""
echo "── /dev/video* device lock check ───────────────────────────────────────"
HELD=$(fuser /dev/video* 2>/dev/null | tr ' ' '\n' | sort -u | grep -v '^$' || true)
if [ -n "$HELD" ]; then
    echo -e "  ${YELLOW}[BUSY]${NC} the following PIDs are holding camera devices:"
    for pid in $HELD; do
        cmd=$(ps -p "$pid" -o comm= 2>/dev/null || echo "?")
        args=$(ps -p "$pid" -o args= 2>/dev/null | cut -c1-80 || echo "?")
        echo -e "         pid=$pid  cmd=$cmd  ($args)"
    done
    echo -e "  Run ${CYAN}bash kill_teleop_blockers.sh${NC} to free them."
    all_ok=0
else
    echo -e "  ${GREEN}[OK]${NC} no processes holding /dev/video*"
fi

echo ""
echo "── Summary ─────────────────────────────────────────────────────────────"
if [ "$all_ok" -eq 1 ]; then
    echo -e "  ${GREEN}All cameras connected and free — ready for teleop.${NC}"
else
    echo -e "  ${RED}One or more issues found — see above.${NC}"
fi
echo "────────────────────────────────────────────────────────────────────────"
