"""Overlay real vs sim gripper open/close traces to compare slew speed.

Feeds:
  * REAL  -- ``gripper_speed_real_c<..>o<..>_for<..>.npz`` from ``gripper_speed_test.py``
             (keys: t, pos_norm[0=open,1=closed], cmd[<0=close], close_speed, open_speed, ...).
  * SIM   -- the ``playback_*.npz`` written by ``scripts_v2/tools/sim2real/playback_actions.py``
             when replaying ``gripper_cycle_actions.npz`` (keys: gripper_joint_pos[T+1,6] in rad,
             actions[T,7], real/frequency). finger_joint (col 0) is normalized by --closed_rad.

Both are put on the same 0..1 axis (0=open, 1=closed) and time axis (s), aligned at t=0 =
first close command, so the slopes are directly comparable. Pass any number of --real / --sim
files to compare tunings side by side.

Usage:
    python scripts/sim2real/compare_gripper_speed.py \
        --real pc_debug/gripper_test/gripper_speed_real_c128o128_for128.npz \
        --sim  playback_gripper_cycle_actions_idx0.npz \
        -o pc_debug/gripper_test/compare.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLOSED_RAD_DEFAULT = 0.785398  # sim finger_joint value at fully closed (Robotiq binary close cmd)


def edges_from_cmd(t, cmd):
    """Return [(t_edge, closing_bool)] from a +/-1 command signal (<0 = close)."""
    s = np.sign(np.asarray(cmd, float))
    s[s == 0] = 1.0  # treat 0 (warmup) as 'open'
    out = [(float(t[0]), s[0] < 0)]
    for i in np.where(np.diff(s) != 0)[0]:
        out.append((float(t[i + 1]), s[i + 1] < 0))
    return out


def slew_after(t, pos, t0, closing):
    """Time from t0 until pos crosses 90% (close) / 10% (open); nan if never."""
    m = t >= t0
    tt, pp = t[m], pos[m]
    hit = (pp >= 0.9) if closing else (pp <= 0.1)
    return float(tt[hit][0] - t0) if hit.any() else np.nan


def median_slews(t, pos, edges):
    c = [slew_after(t, pos, t0, True) for t0, cl in edges if cl]
    o = [slew_after(t, pos, t0, False) for t0, cl in edges if not cl]
    return np.nanmedian(c) if c else np.nan, np.nanmedian(o) if o else np.nan


def load_real(path):
    d = np.load(path)
    t, pos, cmd = d["t"], d["pos_norm"].astype(float), d["cmd"]
    cs = int(d["close_speed"]) if "close_speed" in d.files else -1
    os_ = int(d["open_speed"]) if "open_speed" in d.files else -1
    label = f"real c{cs}/o{os_}" if cs >= 0 else f"real {os.path.basename(path)}"
    return t, pos, edges_from_cmd(t, cmd), label


def load_sim(path, closed_rad):
    d = np.load(path)
    if "gripper_joint_pos" not in d.files:
        raise KeyError(f"{path} has no 'gripper_joint_pos' (keys: {d.files}). Is it a playback_actions output?")
    finger = d["gripper_joint_pos"][:, 0].astype(float) / closed_rad  # -> 0..1
    freq = float(d["real/frequency"]) if "real/frequency" in d.files else 10.0
    t = np.arange(finger.shape[0]) / freq
    act = d["actions"]  # (T,7); obs is T+1
    tc = np.arange(act.shape[0]) / freq
    return t, finger, edges_from_cmd(tc, act[:, 6]), f"sim {os.path.basename(path)}", freq


def main():
    p = argparse.ArgumentParser(description="Overlay real vs sim gripper open/close speed.")
    p.add_argument("--real", nargs="*", default=[], help="Real gripper_speed_real_*.npz file(s).")
    p.add_argument("--sim", nargs="*", default=[], help="Sim playback_*.npz file(s).")
    p.add_argument("--closed_rad", type=float, default=CLOSED_RAD_DEFAULT, help="Sim finger_joint value when closed.")
    p.add_argument("-o", "--out", default=None, help="Output png (default: alongside the first input).")
    args = p.parse_args()
    if not args.real and not args.sim:
        p.error("pass at least one --real and/or --sim file")

    fig, ax = plt.subplots(figsize=(12, 6))
    rows = []  # (label, close_ms, open_ms)
    shade_edges = None

    for i, path in enumerate(args.real):
        t, pos, edges, label = load_real(path)
        ax.plot(t, pos, lw=1.8, color=f"C{i}", label=label)
        rows.append((label, *median_slews(t, pos, edges)))
        shade_edges = shade_edges or edges

    for j, path in enumerate(args.sim):
        t, pos, edges, label, _ = load_sim(path, args.closed_rad)
        ax.plot(t, pos, lw=1.8, ls="--", color=f"C{len(args.real)+j}", label=label)
        rows.append((label, *median_slews(t, pos, edges)))
        shade_edges = shade_edges or edges

    # shade close phases using the first available command schedule
    if shade_edges:
        tmax = ax.get_xlim()[1]
        for k, (t0, closing) in enumerate(shade_edges):
            t1 = shade_edges[k + 1][0] if k + 1 < len(shade_edges) else tmax
            if closing:
                ax.axvspan(t0, t1, color="C3", alpha=0.05)

    ax.axhline(0.9, ls=":", c="C3", lw=1); ax.axhline(0.1, ls=":", c="C0", lw=1)
    ax.set_xlabel("time (s)  (aligned at first close command)")
    ax.set_ylabel("normalized position (0=open, 1=closed)")
    ax.set_ylim(-0.05, 1.15); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Gripper open/close: real vs sim  (shaded = close command)")
    fig.tight_layout()

    out = args.out or (os.path.splitext(args.real[0] if args.real else args.sim[0])[0] + "_compare.png")
    fig.savefig(out, dpi=120); plt.close(fig)

    print(f"saved {out}\n")
    print(f"  {'trace':<34}{'close 0->90%':>14}{'open 100->10%':>15}")
    for label, c, o in rows:
        cs = f"{c*1000:.0f} ms" if np.isfinite(c) else "  --"
        os_ = f"{o*1000:.0f} ms" if np.isfinite(o) else "  --"
        print(f"  {label:<34}{cs:>14}{os_:>15}")


if __name__ == "__main__":
    main()
