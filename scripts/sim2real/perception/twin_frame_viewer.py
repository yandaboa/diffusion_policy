"""Viewer for the IsaacLab digital-twin frame stream with per-camera peg overlays.

`live_digital_twin.py --stream_frames PATH` writes an .npz per step: RGB frames
{overview, front, side, wrist} plus a JSON `meta` (per-view projection matrix, world-frame
per-camera peg estimates, per-side-cam seen flag + visible tag ids). This composes them:

  * overview (left): the solid sim peg = fused CENTROID (full opacity), plus the 3 per-camera
    estimates as low-opacity color-coded wireframe boxes (front=red, side=green, wrist=blue).
  * each side tile (right): the centroid peg (from the render) + only THAT camera's own
    estimate box (low opacity), with text green/red for seen/not-seen and the tag ids it sees.

Run in the `foundstereo` env (cv2 + a display):
    python scripts/sim2real/perception/twin_frame_viewer.py --scale 0.9
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Fused base-frame peg pose published by peg_fusion_viz --publish (repo/Log/peg_twin_state.json).
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
_STATE_DEFAULT = os.path.join(_REPO, "Log", "peg_twin_state.json")

CAM_ORDER = ["front", "side", "wrist"]
_BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7),
              (0, 2), (1, 3), (4, 6), (5, 7),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def _box_corners(T, extent):
    e = np.asarray(extent, float) / 2.0
    c = np.array([[sx * e[0], sy * e[1], sz * e[2]]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    T = np.asarray(T, float)
    return (T[:3, :3] @ c.T).T + T[:3, 3]


def _draw_box(img, T_world_peg, P, color_bgr, extent, alpha=0.4, thick=2):
    """Project an 8-corner peg box with 3x4 world->pixel P and alpha-blend the wireframe."""
    if T_world_peg is None or P is None:
        return
    P = np.asarray(P, float)
    W = _box_corners(T_world_peg, extent)
    proj = P @ np.hstack([W, np.ones((8, 1))]).T          # 3x8
    z = proj[2]
    uv = (proj[:2] / np.where(np.abs(z) < 1e-6, 1e-6, z)).T
    overlay = img.copy()
    drew = False
    for i, j in _BOX_EDGES:
        if z[i] > 0 and z[j] > 0:
            cv2.line(overlay, (int(round(uv[i, 0])), int(round(uv[i, 1]))),
                     (int(round(uv[j, 0])), int(round(uv[j, 1]))), color_bgr, thick, cv2.LINE_AA)
            drew = True
    if drew:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_estimates(img_bgr, view, pegs, extent, colors):
    P = view.get("P")
    for nm in view.get("draw", []):
        col = colors.get(nm, [255, 255, 255])            # RGB -> BGR
        _draw_box(img_bgr, pegs.get(nm), P, (col[2], col[1], col[0]), extent)


def _text(img, lines_colors, org=(6, 20), scale=0.55, dy=22):
    x, y = org
    for txt, col in lines_colors:
        for c, th in (((0, 0, 0), 4), (col, 1)):
            cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, c, th, cv2.LINE_AA)
        y += dy


def _load(path):
    with open(path, "rb") as f:
        obj = np.load(f, allow_pickle=False)
        if isinstance(obj, np.lib.npyio.NpzFile):
            return {k: obj[k] for k in obj.files}
        return {"overview": np.asarray(obj)}


def _load_front(path):
    """Load the real front-camera RGB dumped by peg_fusion_viz --front-frame; return BGR or None."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            rgb = np.load(f, allow_pickle=False)
        return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _load_peg_pose(path, stale_s=1.0):
    """Read the FUSED base-frame peg pose (T_base_peg) published by peg_fusion_viz --publish.

    Returns dict(xyz, quat_wxyz, rpy_deg, ncams, tags, stamp, age_s, fresh) or None.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        d = json.load(open(path))
        T = d.get("T_base_peg")
        if T is None:
            return None
        T = np.asarray(T, dtype=float)
        rot = R.from_matrix(T[:3, :3])
        q = rot.as_quat()          # [x, y, z, w]
        stamp = d.get("stamp")
        age = (time.time() - stamp) if stamp is not None else float("inf")
        tags = sorted({int(t) for c in (d.get("cams") or {}).values()
                       for t in (c.get("tags") or [])})
        return {"xyz": T[:3, 3],
                "quat_wxyz": [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
                "rpy_deg": rot.as_euler("xyz", degrees=True),
                "ncams": int(d.get("ncams", 0) or 0), "tags": tags,
                "stamp": stamp, "age_s": age, "fresh": age <= stale_s}
    except Exception:
        return None


def _draw_peg_pose(panel, peg):
    """Overlay the fused base-frame peg pose (xyz + rpy + quat) at the bottom-left of a panel.

    Colour-coded by confidence: green (fresh, >=2 cams) / amber (fresh, 1 cam -- single tag is
    flip-ambiguous) / red (stale)."""
    if peg is None:
        return
    fresh, nc = peg["fresh"], peg["ncams"]
    col = (0, 220, 0) if (fresh and nc >= 2) else (0, 215, 255) if fresh else (0, 0, 255)  # BGR
    x, y, z = peg["xyz"]
    r, p, yw = peg["rpy_deg"]
    qw, qx, qy, qz = peg["quat_wxyz"]
    lines = [
        (f"PEG base xyz [{x:+.3f} {y:+.3f} {z:+.3f}] m   ncams={nc} tags={peg['tags']}"
         + ("" if fresh else f"  STALE {peg['age_s']:.1f}s"), col),
        (f"rpy [{r:+.0f} {p:+.0f} {yw:+.0f}] deg   quat_wxyz [{qw:+.3f} {qx:+.3f} {qy:+.3f} {qz:+.3f}]", col),
    ]
    _text(panel, lines, org=(6, panel.shape[0] - 34), scale=0.5, dy=22)


def _compose(arrs, real_front=None, peg=None):
    meta = None
    if "meta" in arrs:
        try:
            meta = json.loads(str(arrs["meta"]))
        except Exception:
            meta = None
    pegs = meta["pegs"] if meta else {}
    extent = meta["extent"] if meta else [0.03, 0.03, 0.06]
    colors = meta["colors"] if meta else {}
    views = meta["views"] if meta else {}

    overview = cv2.cvtColor(arrs["overview"], cv2.COLOR_RGB2BGR) if "overview" in arrs \
        else np.zeros((540, 960, 3), np.uint8)
    if meta and "overview" in views:
        _draw_estimates(overview, views["overview"], pegs, extent, colors)
    OH, OW = overview.shape[:2]

    def cam_img(name, size=None):
        """BGR sim camera render with peg overlays; optionally resized."""
        img = cv2.cvtColor(arrs[name], cv2.COLOR_RGB2BGR)
        if meta:
            _draw_estimates(img, views.get(name, {}), pegs, extent, colors)
        return cv2.resize(img, size) if size is not None else img

    def cam_labels(name):
        v = views.get(name, {})
        seen, tags = bool(v.get("seen", False)), v.get("tags", [])
        return [(f"{name}   {'SEEN' if seen else 'NO TAG'}",
                 (0, 200, 0) if seen else (0, 0, 255)),
                ("ids: " + (",".join(map(str, tags)) if tags else "-"), (235, 235, 235))]

    present = [n for n in CAM_ORDER if n in arrs]

    # Big side-by-side comparison: SIM front cam (left) + REAL front cam (right). The sim
    # scene/overview is demoted to a small tile. Requires both the sim front render and the
    # real front stream; otherwise fall back to the sim overview as the single big panel.
    if real_front is not None and "front" in arrs:
        left = cam_img("front", (OW, OH))
        cv2.rectangle(left, (0, 0), (OW - 1, OH - 1), (60, 60, 60), 1)
        _text(left, [("SIM front cam", (235, 235, 235)), cam_labels("front")[1]])
        right = cv2.resize(real_front, (OW, OH))
        cv2.rectangle(right, (0, 0), (OW - 1, OH - 1), (60, 60, 60), 1)
        _text(right, [("REAL front cam", (235, 235, 235))])
        bigs = [left, right]
        # tiles: sim overview first, then the remaining sim cams (side, wrist)
        tiles = [("overview", None)] + [("cam", n) for n in present if n != "front"]
    else:
        _text(overview, [("SIM overview", (235, 235, 235))])
        bigs = [overview]
        if real_front is not None:                    # no sim front render -> still show real one
            rp = cv2.resize(real_front, (OW, OH))
            cv2.rectangle(rp, (0, 0), (OW - 1, OH - 1), (60, 60, 60), 1)
            _text(rp, [("REAL front cam", (235, 235, 235))])
            bigs.append(rp)
        tiles = [("cam", n) for n in present]

    _draw_peg_pose(bigs[0], peg)          # fused base-frame peg pose on the big panel

    if not tiles:
        return bigs[0] if len(bigs) == 1 else np.hstack(bigs)
    tw, th = max(200, int(OW * 0.33)), OH // len(tiles)
    col = []
    for kind, n in tiles:
        if kind == "overview":
            t = cv2.resize(overview, (tw, th))
            cv2.rectangle(t, (0, 0), (tw - 1, th - 1), (60, 60, 60), 1)
            _text(t, [("SIM overview", (235, 235, 235))])
        else:
            t = cam_img(n, (tw, th))
            cv2.rectangle(t, (0, 0), (tw - 1, th - 1), (60, 60, 60), 1)
            _text(t, cam_labels(n))
        col.append(t)
    col = np.vstack(col)
    H = col.shape[0]
    bigs = [cv2.resize(b, (OW, H)) if b.shape[0] != H else b for b in bigs]
    return np.hstack(bigs + [col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="/dev/shm/twin_frame.npy",
                    help="Frame .npz written by live_digital_twin.py --stream_frames.")
    ap.add_argument("--scale", type=float, default=1.0, help="Final window scale factor.")
    ap.add_argument("--hz", type=float, default=30.0, help="Max refresh rate.")
    ap.add_argument("--real-front", default="/dev/shm/real_front_frame.npy",
                    help="Real front-camera RGB dumped by peg_fusion_viz --front-frame; shown at "
                         "overview size to the right of the sim overview. Set '' to disable.")
    ap.add_argument("--state-file", default=_STATE_DEFAULT,
                    help="Fused peg pose JSON from peg_fusion_viz --publish; its base-frame "
                         "T_base_peg (xyz + rpy + quat_wxyz) is overlaid on the big panel. "
                         "Set '' to disable.")
    ap.add_argument("--peg-stale", type=float, default=1.0,
                    help="Show the peg pose in red (stale) if its publish stamp is older than this (s).")
    args = ap.parse_args()

    print(f"[viewer] reading {args.frame}  (ESC to quit)")
    if args.state_file:
        print(f"[viewer] peg pose <- {args.state_file}")
    win = "digital twin (overview + per-camera estimates)"
    period = 1.0 / max(args.hz, 1.0)
    last_mtime = None
    last_peg_stamp = object()   # sentinel; recompose when the peg publish stamp changes too
    while True:
        t0 = time.time()
        img = None
        try:
            peg = _load_peg_pose(args.state_file, args.peg_stale) if args.state_file else None
            peg_stamp = peg["stamp"] if peg else None
            if os.path.exists(args.frame):
                m = os.path.getmtime(args.frame)
                if m != last_mtime or peg_stamp != last_peg_stamp:
                    img = _compose(_load(args.frame), _load_front(args.real_front), peg)
                    last_mtime, last_peg_stamp = m, peg_stamp
        except Exception:
            img = None
        if img is not None:
            if args.scale != 1.0:
                img = cv2.resize(img, None, fx=args.scale, fy=args.scale)
            cv2.imshow(win, img)
        elif last_mtime is None:
            blank = np.zeros((240, 480, 3), np.uint8)
            cv2.putText(blank, "waiting for twin frames...", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, blank)
        if cv2.waitKey(1) == 27:
            break
        sleep = period - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
