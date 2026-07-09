"""Drive the Robotiq 2F-85 through open/close cycles and log its measured position.

Purpose: measure the REAL gripper's open/close SPEED so it can be matched against the
sim playback (``scripts_v2/tools/sim2real/playback_actions.py`` in UWLab). The arm is left
stationary -- this only commands the gripper -- so make sure the arm is already in a safe
pose before running.

What it does:
  * connects to the Robotiq URCap socket (robot_ip:63352) and activates the gripper,
  * runs ``--cycles`` cycles of ``--close_s`` closed then ``--open_s`` open (default 2s/2s x3),
  * commands the close and open transitions with independent SPE speeds
    (``--close_speed`` / ``--open_speed``, each 0-255; both default to ``--speed``) so open and
    close can be tuned separately -- the eval controller hardcodes 128 for both,
  * polls the measured POS register (gPO, actual finger position) at ``--poll_hz`` and logs it.

Two outputs:
  * ``gripper_speed_real_spe<speed>_for<force>.npz`` -- the high-rate real trajectory
    (t, pos_norm, cmd, motor_current) for plotting/measuring the real slew speed.
  * ``gripper_cycle_actions.npz`` -- a (T, 7) action array at ``--frequency`` (arm zeros,
    gripper -1=close / +1=open) to replay in sim for a like-for-like comparison:
        python scripts_v2/tools/sim2real/playback_actions.py --headless --video \
            --actions <path>/gripper_cycle_actions.npz --reset_type ObjectAnywhereEEAnywhere --idx 0

Usage:
    python scripts/sim2real/gripper_speed_test.py --robot_ip 192.168.1.10 --speed 128 --force 128 -o pc_debug/gripper_test
"""

import argparse
import os
import sys
import time

import numpy as np

# Make the repo importable when run directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diffusion_policy.real_world.robotiq_gripper import RobotiqGripper


def main():
    p = argparse.ArgumentParser(description="Robotiq 2F-85 open/close speed test (gripper only, arm stationary).")
    p.add_argument("--robot_ip", default="192.168.1.10", help="Robot/gripper IP (URCap socket host).")
    p.add_argument("--gripper_port", type=int, default=63352, help="Robotiq URCap socket port.")
    p.add_argument("--speed", type=int, default=128, help="Shared default gripper SPE (0-255); overridden per-direction below.")
    p.add_argument("--close_speed", type=int, default=None, help="SPE for the close move (0-255). Defaults to --speed.")
    p.add_argument("--open_speed", type=int, default=None, help="SPE for the open move (0-255). Defaults to --speed.")
    p.add_argument("--force", type=int, default=128, help="Gripper FOR register (0-255).")
    p.add_argument("--cycles", type=int, default=3, help="Number of close/open cycles.")
    p.add_argument("--close_s", type=float, default=2.0, help="Seconds held closed per cycle.")
    p.add_argument("--open_s", type=float, default=2.0, help="Seconds held open per cycle.")
    p.add_argument("--poll_hz", type=float, default=60.0, help="Rate at which the measured POS is logged.")
    p.add_argument("--frequency", type=float, default=10.0, help="Control rate for the sim-playback action npz (match eval).")
    p.add_argument("-o", "--out", default="pc_debug/gripper_test", help="Output directory.")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    close_speed = args.close_speed if args.close_speed is not None else args.speed
    open_speed = args.open_speed if args.open_speed is not None else args.speed

    # ---- connect + activate ----
    g = RobotiqGripper()
    print(f"Connecting to gripper at {args.robot_ip}:{args.gripper_port} ...")
    g.connect(args.robot_ip, args.gripper_port)
    g.activate()
    p_open, p_close = g.get_open_position(), g.get_closed_position()
    span = float(p_close - p_open) or 1.0
    norm = lambda raw: (float(raw) - p_open) / span  # 0=open, 1=closed
    print(f"Activated. open={p_open} closed={p_close}  close_speed={close_speed} open_speed={open_speed} force={args.force}")

    # Start fully open so the first cycle begins with a clean close transition.
    g.move(p_open, open_speed, args.force)
    time.sleep(1.5)

    # ---- run cycles, logging measured POS at poll_hz ----
    period = 1.0 / args.poll_hz
    cycle_s = args.close_s + args.open_s
    total_s = args.cycles * cycle_s
    log_t, log_pos, log_cmd, log_cur = [], [], [], []
    last_close = None
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        if t >= total_s:
            break
        # schedule: each cycle is [close_s closed] then [open_s open]
        want_close = (t % cycle_s) < args.close_s
        if want_close != last_close:
            if want_close:
                g.move(p_close, close_speed, args.force)
            else:
                g.move(p_open, open_speed, args.force)
            last_close = want_close
        log_t.append(t)
        log_pos.append(norm(g.get_current_position()))
        log_cmd.append(-1.0 if want_close else 1.0)  # eval convention: <0 = close
        log_cur.append(float(g.get_motor_current()))
        # keep a steady poll rate
        sleep = period - ((time.monotonic() - t0) - t)
        if sleep > 0:
            time.sleep(sleep)

    g.move(p_open, open_speed, args.force)  # leave it open

    log_t = np.asarray(log_t, np.float64)
    log_pos = np.asarray(log_pos, np.float32)
    log_cmd = np.asarray(log_cmd, np.float32)
    log_cur = np.asarray(log_cur, np.float32)

    # ---- quick speed readout: time from a close command to reaching 90% closed ----
    def first_cross(after_t, thresh, closing):
        m = (log_t >= after_t) & ((log_pos >= thresh) if closing else (log_pos <= thresh))
        return float(log_t[m][0] - after_t) if m.any() else float("nan")

    close_rise = first_cross(0.0, 0.9, True)
    open_fall = first_cross(args.close_s, 0.1, False)
    print(f"\nReal gripper:  close (SPE={close_speed}) 0->90% in ~{close_rise*1000:.0f} ms | "
          f"open (SPE={open_speed}) 100->10% in ~{open_fall*1000:.0f} ms")

    # ---- save the real trajectory ----
    real_path = os.path.join(args.out, f"gripper_speed_real_c{close_speed}o{open_speed}_for{args.force}.npz")
    np.savez(real_path, t=log_t, pos_norm=log_pos, cmd=log_cmd, motor_current=log_cur,
             close_speed=np.int32(close_speed), open_speed=np.int32(open_speed), force=np.int32(args.force),
             close_s=np.float32(args.close_s), open_s=np.float32(args.open_s), cycles=np.int32(args.cycles))

    # ---- build the sim-playback action npz at --frequency (arm zeros, gripper +-1) ----
    n_close = int(round(args.close_s * args.frequency))
    n_open = int(round(args.open_s * args.frequency))
    grip = np.tile(np.concatenate([-np.ones(n_close), np.ones(n_open)]).astype(np.float32), args.cycles)
    action = np.zeros((grip.shape[0], 7), np.float32)
    action[:, 6] = grip
    act_path = os.path.join(args.out, "gripper_cycle_actions.npz")
    np.savez(act_path, action=action, frequency=np.float32(args.frequency),
             close_speed=np.int32(close_speed), open_speed=np.int32(open_speed), force=np.int32(args.force))

    print(f"\nSaved real trajectory -> {real_path}  ({len(log_t)} samples)")
    print(f"Saved sim actions     -> {act_path}  (action {action.shape}, {args.cycles}x {args.close_s}s/{args.open_s}s @ {args.frequency}Hz)")
    print("\nReplay in sim (UWLab repo):")
    print(f"  python scripts_v2/tools/sim2real/playback_actions.py --headless --video \\")
    print(f"      --actions {os.path.abspath(act_path)} --reset_type ObjectAnywhereEEAnywhere --idx 0")


if __name__ == "__main__":
    main()
