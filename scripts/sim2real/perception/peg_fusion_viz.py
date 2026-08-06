"""Live multi-camera peg-pose fusion in the robot BASE frame.

Each camera independently estimates the peg pose from its AprilTags (reusing
apriltag_peg_pose.estimate_peg_pose), then we lift every estimate into the shared
robot-base frame and draw them together with their centroid:

    front (Orbbec)  T_base_peg = T_base_cam_front @ T_cam_peg           [fixed extrinsic]
    side  (D435)    T_base_peg = T_base_cam_side  @ T_cam_peg           [fixed extrinsic]
    wrist (D415)    T_base_peg = FK(joints) @ T_wrist3link_cam @ T_cam_peg   [FK * fixed]

The static extrinsics come from the hand-aligned per-serial files; the wrist offset
from wrist_cam_in_ee.npy; the wrist's live base pose from calibrated FK on the robot's
current joints (rtde_receive, read-only). All land in the same REP-103 sim base frame,
so the three estimates are directly comparable -- their spread IS the cross-camera
accuracy check.

Open3D scene (base frame): world axes at origin; one peg wireframe box per camera
(front=red, side=green, wrist=blue); the fused centroid as a white box + RGB axis triad;
a small axis triad at each camera's pose. Console prints per-camera base-frame position,
the centroid, and the agreement (max pairwise distance + rotation spread).

Run in the `foundstereo` env:
    python scripts/sim2real/perception/peg_fusion_viz.py           # all 3 cameras
    python scripts/sim2real/perception/peg_fusion_viz.py --no-wrist   # skip robot/FK
"""
import argparse
from collections import deque
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))          # perception.*
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)                              # diffusion_policy.*

import apriltag_peg_pose as apt
import tag_body as tb
from async_peg_pipeline import AsyncCameraPoseWorker
from perception.orbbec import gather_orbbec_cameras
from perception.realsense import gather_realsense_cameras

CALIB_DIR = os.path.join(_HERE, "calibrations")
DEFAULT_STATE_FILE = os.path.join(_REPO, "Log", "peg_twin_state.json")
SER_FRONT, SER_SIDE, SER_WRIST = "215122255213", "832112070487", "746112060198"  # front=D455, side=D435, wrist=D415
# per-role color resolution defaults: D455 front caps at 1280x800, but the D435 side and
# D415 wrist do 1920x1080@30 -> bigger tag pixels for easier tracking.
DEFAULT_RES = {"front": (1280, 720), "side": (1920, 1080), "wrist": (1920, 1080)}
SER_ROLE = {SER_FRONT: "front", SER_SIDE: "side", SER_WRIST: "wrist"}
COLOR = {"front": (1.0, 0.15, 0.15), "side": (0.15, 0.85, 0.15), "wrist": (0.25, 0.5, 1.0)}

_BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7),   # z
              (0, 2), (1, 3), (4, 6), (5, 7),   # y
              (0, 4), (1, 5), (2, 6), (3, 7)]   # x


def _load_static_extrinsic(serial):
    """T_base_cam (cam->base) from a hand-aligned per-serial file."""
    p = os.path.join(CALIB_DIR, f"most_recent_hand_aligned_extrinsic_{serial}.json")
    if not os.path.exists(p):
        raise SystemExit(f"missing extrinsic {p} -- align this camera first.")
    return np.array(json.load(open(p))["T_total_cam_simbase"], dtype=np.float64)


def _load_wrist_offset():
    for p in (os.path.join(CALIB_DIR, "wrist_cam_in_ee.npy"),
              os.path.join(_REPO, "..", "UWLab-patrick-private", "wrist_cam_in_ee.npy")):
        if os.path.exists(p):
            return np.load(p).astype(np.float64), p
    raise SystemExit("missing wrist_cam_in_ee.npy -- run get_wrist_cam_in_ee.py first.")


def _box_points(T, extent):
    e = np.asarray(extent, float) / 2.0
    c = np.array([[sx * e[0], sy * e[1], sz * e[2]]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return (T[:3, :3] @ c.T).T + T[:3, 3]


def _triad_points(T, size):
    o = T[:3, 3]
    return np.array([o, o + T[:3, 0] * size, o + T[:3, 1] * size, o + T[:3, 2] * size])


def _set_box(ls, T, extent, color):
    if T is None:
        ls.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        ls.lines = o3d.utility.Vector2iVector(np.empty((0, 2), int))
        return
    ls.points = o3d.utility.Vector3dVector(_box_points(T, extent))
    ls.lines = o3d.utility.Vector2iVector(np.array(_BOX_EDGES))
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(_BOX_EDGES), 1)))


def _set_triad(ls, T, size):
    if T is None:
        ls.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        ls.lines = o3d.utility.Vector2iVector(np.empty((0, 2), int))
        return
    ls.points = o3d.utility.Vector3dVector(_triad_points(T, size))
    ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 2], [0, 3]]))
    ls.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]))


class _HoleAccumulator:
    """Running estimate of the STATIC peg-hole pose in the base frame.

    The hole does not move, so every distinct measurement -- across cameras AND across time --
    is evidence about the same quantity. We keep a bounded window of per-camera base-frame
    estimates and take their confidence-weighted geodesic median. Averaging ~100 independent
    observations drives the tag noise down by ~10x versus any single frame, which is what buys
    accuracy against the sim's 2.5 mm success gate.

    A camera's estimate is only admitted once per underlying capture: the async worker carries
    its latest aux result forward on every publish tick, so we de-duplicate on capture_stamp.
    """

    def __init__(self, window=120, min_tags=2):
        self.window = int(window)
        self.min_tags = int(min_tags)
        self._buf = {}            # name -> deque[(T_base_hole, conf)]
        self._last_stamp = {}     # name -> capture_stamp of the last admitted sample
        self._cache = None
        self.n_admitted = 0

    def push(self, name, capture_stamp, T_base_hole, conf, n_tags):
        if T_base_hole is None or n_tags < self.min_tags:
            return False
        if self._last_stamp.get(name) == capture_stamp:
            return False                       # same capture, already counted
        self._last_stamp[name] = capture_stamp
        self._buf.setdefault(name, deque(maxlen=self.window)).append(
            (np.asarray(T_base_hole, dtype=np.float64), max(float(conf), 0.0)))
        self._cache = None
        self.n_admitted += 1
        return True

    def estimate(self):
        """(T_base_hole | None, n_samples, spread_mm, spread_deg, cams)."""
        if self._cache is not None:
            return self._cache
        samples = [s for buf in self._buf.values() for s in buf]
        if not samples:
            return None, 0, 0.0, 0.0, []
        Ts = [s[0] for s in samples]
        w = [s[1] for s in samples]
        T = tb.robust_mean(Ts, w)
        pos_mm, rot_deg = tb.spread(Ts, T)
        self._cache = (T, len(Ts), pos_mm, rot_deg, sorted(self._buf))
        return self._cache

    def reset(self):
        self._buf.clear()
        self._last_stamp.clear()
        self._cache = None
        self.n_admitted = 0


def _write_state(path, joints, centroid, cam_info, capture_stamp=None, hole=None):
    """Atomically write the live twin state for the IsaacLab subscriber.

    joints: list[6] (rad, UR controller order) or None; centroid: 4x4 (peg->base) or None;
    cam_info: {name: {"T": 4x4|None, "seen": bool, "tags": [ids]}} per-camera estimates.
    capture_stamp: wall-clock time.time() of the OLDEST camera frame that produced this fused
        pose (from RealSenseCamera.read_camera's ``read_time``); lets the consumer measure true
        capture->policy latency, not just publish->read. ``stamp`` below is the publish time.
    hole: {"T": 4x4|None, "n": samples, "spread_mm": .., "spread_deg": .., "cams": [..],
        "tags": [ids]} -- the accumulated STATIC peg-hole pose, or None when hole tracking is off.
    """
    import time

    def _mat(m):
        return np.asarray(m).tolist() if m is not None else None

    state = {
        "stamp": time.time(),
        "capture_stamp": capture_stamp,
        "joints": list(joints) if joints is not None else None,
        "T_base_peg": _mat(centroid),          # centroid (back-compat key)
        "ncams": sum(1 for c in cam_info.values() if c["T"] is not None),
        "cams": {
            n: {
                "T": _mat(c["T"]),
                "seen": bool(c["seen"]),
                "tags": list(c["tags"]),
                "capture_stamp": c.get("capture_stamp"),
                "detect_ms": c.get("detect_ms"),
                "used_roi": c.get("used_roi"),
            }
            for n, c in cam_info.items()
        },
    }
    if hole is not None:
        state["T_base_hole"] = _mat(hole.get("T"))
        state["hole"] = {k: v for k, v in hole.items() if k != "T"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)          # atomic swap so the reader never sees a half-written file


def _write_front_frame(path, rgb):
    """Atomically dump the real front-camera RGB to a .npy for the twin viewer to overlay
    beside the sim overview (the viewer can't open the camera itself -- it's busy here)."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, np.ascontiguousarray(rgb))
    os.replace(tmp, path)


def _draw_tag(vis, det, color):
    c = np.asarray(det["corners"], dtype=np.float64).reshape(-1, 2).astype(np.int32)
    cv2.polylines(vis, [c], True, color, 2)
    cx, cy = c.mean(axis=0).astype(int)
    cv2.putText(vis, str(int(det["id"])), (cx - 8, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _draw_tag_view(latest_frames, results, fresh, tile_w=640):
    """Horizontal strip of per-camera views with the detected AprilTags drawn.

    Peg-face tags green, peg-hole tags orange, ROI blue. Detections come from each camera's
    latest RESULT, which may lag its latest FRAME by a frame or two -- fine for monitoring.
    The hole is scanned at a low rate and its last result carried forward, so its outlines
    are only drawn while reasonably recent.
    """
    tiles = []
    now = time.time()
    for name in ("front", "side", "wrist"):
        frame = latest_frames.get(name)
        if frame is None or frame.get("rgb") is None:
            continue
        vis = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
        result = results.get(name)
        peg_ids, hole_ids = [], []
        if result is not None:
            if result.get("used_roi") and result.get("roi") is not None:
                x0, y0, x1, y1 = [int(v) for v in result["roi"]]
                cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 128, 0), 1)
            for det in (result.get("dbg") or []):
                _draw_tag(vis, det, (0, 255, 0))
                peg_ids.append(int(det["id"]))
            aux = result.get("aux") or {}
            if now - (aux.get("capture_stamp") or 0.0) < 3.0:
                for det in (aux.get("dbg") or []):
                    _draw_tag(vis, det, (0, 165, 255))
                    hole_ids.append(int(det["id"]))
        ok = name in fresh
        header = (f"{name}  {'FRESH' if ok else 'stale'}"
                  f"  peg={peg_ids if peg_ids else '--'}  hole={hole_ids if hole_ids else '--'}")
        scale = tile_w / vis.shape[1]
        vis = cv2.resize(vis, (tile_w, int(round(vis.shape[0] * scale))))
        cv2.putText(vis, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if ok else (0, 0, 255), 2)
        tiles.append(vis)
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
             for t in tiles]
    return np.hstack(tiles)


_REC_IDX = {"front": 0, "side": 1, "wrist": 2}   # cam_idx matching RealEnv get_obs_rgb (3-cam order)


class _EpisodeVideoRecorder:
    """Save per-camera videos to ``<video_dir>/<episode>/<cam_idx>.mp4`` while an external
    record-control JSON (written by eval_real_robot.py --save_video) says recording -- the same
    layout RealEnv-with-cameras produces.

    Encoding runs in a BACKGROUND THREAD so it never throttles the fusion/pose loop: ``step()``
    just swaps immutable latest-frame references; the thread polls the control file, manages the
    per-episode writers, and emits real ``fps`` frames on WALL-CLOCK time -- holding the latest
    frame to fill elapsed time (and dropping if it ever runs faster). 1 s real -> ``fps`` frames
    -> real-time playback. Unique content is still limited by how fast the fusion loop feeds new
    frames (i.e. its own rate); the thread just decouples the ENCODE cost from that loop."""

    def __init__(self, ctrl_path, fps):
        self.ctrl_path = ctrl_path
        self.fps = int(fps)
        self._latest = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # writer state -- owned exclusively by the background thread
        self.episode_id = None
        self.dir = None
        self.writers = {}
        self.t0 = None
        self.frames_written = {}
        self._thread = threading.Thread(target=self._run, name="peg-video-rec", daemon=True)
        self._thread.start()

    def step(self, rgb_by_role):
        """Publish immutable latest-frame references to the encoder thread."""
        with self._lock:
            self._latest = {r: v for r, v in rgb_by_role.items() if v is not None}

    def _read_ctrl(self):
        try:
            d = json.load(open(self.ctrl_path))
            return bool(d.get("recording")), d.get("episode_id"), d.get("video_dir")
        except Exception:
            return False, None, None

    def _close_writers(self):
        for w in self.writers.values():
            try:
                w.close()
            except Exception:
                pass
        self.writers = {}
        self.episode_id = None
        self.dir = None
        self.t0 = None
        self.frames_written = {}

    def _run(self):
        import imageio
        while not self._stop.is_set():
            t_tick = time.monotonic()
            want, ep, vdir = self._read_ctrl()
            if (not want) or ep != self.episode_id:
                self._close_writers()
            if want and self.episode_id is None and ep is not None and vdir:
                self.dir = os.path.join(vdir, str(int(ep)))
                os.makedirs(self.dir, exist_ok=True)
                self.episode_id = ep
                self.t0 = time.monotonic()
            if self.episode_id is not None:
                with self._lock:
                    latest = self._latest
                target = int(round((time.monotonic() - self.t0) * self.fps))
                for role, rgb in latest.items():
                    idx = _REC_IDX.get(role)
                    if idx is None or rgb is None:
                        continue
                    if role not in self.writers:
                        path = os.path.join(self.dir, f"{idx}.mp4")
                        self.writers[role] = imageio.get_writer(
                            path, fps=self.fps, codec="libx264",
                            output_params=["-crf", "23", "-preset", "veryfast"])
                        self.frames_written[role] = 0
                        print(f"[record] episode {self.episode_id}: {role} (cam {idx}) -> {path} @ {self.fps}fps")
                    n = max(0, min(target - self.frames_written[role], self.fps))
                    for _ in range(n):
                        self.writers[role].append_data(rgb)
                    self.frames_written[role] += n
            # tick at ~fps; interruptible sleep so close() is snappy
            self._stop.wait(max(0.0, 1.0 / self.fps - (time.monotonic() - t_tick)))

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._close_writers()


def _save_refined_front(serial, T_new, before_mm, after_mm):
    """Overwrite the front per-serial hand-aligned file with the corrected extrinsic (backs up)."""
    import shutil
    import time as _t
    p = os.path.join(CALIB_DIR, f"most_recent_hand_aligned_extrinsic_{serial}.json")
    d = json.load(open(p))
    bak = p.replace(".json", f"_prerefine_{int(_t.time())}.json")
    shutil.copy(p, bak)
    q = R.from_matrix(T_new[:3, :3]).as_quat()      # xyzw
    d["T_total_cam_simbase"] = T_new.tolist()
    d["camera_base_pos"] = T_new[:3, 3].tolist()
    d["camera_base_quat_wxyz"] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
    d["refined"] = {"method": "peg_cross_camera", "before_mm": float(before_mm),
                    "after_mm": float(after_mm)}
    json.dump(d, open(p, "w"), indent=2)
    print(f"[refine] backed up original -> {bak}")
    print(f"[refine] saved refined front extrinsic -> {p}")


def _run_refine(cams, detector, T_peg_tag, T_base_cam, rtde_r, fk, T_wrist3_cam, args):
    """Solve a rigid correction C so the front camera agrees with the reference cameras.

    front error is rigid: T_base_cam_front is off by a fixed C, which propagates as a left-
    multiply on every T_base_peg it produces. So with the peg co-visible, per frame
    C_i = T_base_peg_ref @ inv(T_base_peg_front); we chordal-average C_i and apply
    T_base_cam_front' = C @ T_base_cam_front.
    """
    front = "front"
    refs = [r.strip() for r in args.ref_cams.split(",") if r.strip() in cams and r.strip() != front]
    if front not in cams:
        raise SystemExit("[refine] front camera not connected; cannot refine.")
    if not refs:
        raise SystemExit(f"[refine] no reference cameras among {sorted(cams)} (need side/wrist).")
    print(f"\n[refine] correcting front ({SER_FRONT}) to match {refs}.")
    print(f"[refine] Move the peg SLOWLY through the shared workspace so front + {refs} all see it, "
          f"covering varied positions/orientations.")
    print(f"[refine] collecting {args.refine_frames} co-visible frames (Ctrl-C to stop early)\n")

    def estimate(name, joints):
        """Return (T_base_peg | None, n_tags)."""
        cam, K, dist = cams[name]
        gray = cv2.cvtColor(cam.read_camera()["rgb"], cv2.COLOR_RGB2GRAY)
        T_cam_peg, dbg = apt.estimate_peg_pose(gray, detector, K, dist, T_peg_tag)
        nt = len(dbg)
        if T_cam_peg is None:
            return None, nt
        if name == "wrist":
            if joints is None or T_wrist3_cam is None:
                return None, nt
            return fk(joints)[0] @ T_wrist3_cam @ T_cam_peg, nt
        return T_base_cam[name] @ T_cam_peg, nt

    def _delta(A, B):
        return (float(np.linalg.norm(A[:3, 3] - B[:3, 3])),
                float(np.degrees((R.from_matrix(A[:3, :3]) * R.from_matrix(B[:3, :3]).inv()).magnitude())))

    pairs = []          # (T_front, T_ref) -- each captured while the peg is STATIONARY
    last_pose = None
    try:
        while len(pairs) < args.refine_frames:
            joints = list(rtde_r.getActualQ()) if rtde_r is not None else None
            Tf1, nf1 = estimate(front, joints)                  # front read BEFORE the refs
            ref_est = {r: estimate(r, joints) for r in refs}
            Tf2, nf2 = estimate(front, joints)                  # front read AFTER the refs
            if Tf1 is None or Tf2 is None:
                continue
            # 2-TAG gate: a single planar tag is flip-ambiguous (20-35deg random swings) and is
            # exactly what corrupted earlier refines. Require the front to see >=2 tags in BOTH
            # bracketing reads so we only calibrate on its stable, non-flipping estimates.
            if nf1 < 2 or nf2 < 2:
                continue
            # STATIC gate: front must not move between the two reads bracketing the refs,
            # else front and refs saw DIFFERENT peg poses (sequential-read motion lag) and the
            # pair is corrupted -- this was the cause of the correction making things worse.
            mp, mr = _delta(Tf1, Tf2)
            if mp > 0.003 or mr > 1.5:
                continue
            ref_Ts = [T for (T, nt) in ref_est.values() if T is not None and nt >= 2]
            if not ref_Ts:
                continue
            if len(ref_Ts) > 1:                                 # skip if the refs disagree a lot
                spread = max(np.linalg.norm(ref_Ts[a][:3, 3] - ref_Ts[b][:3, 3])
                             for a in range(len(ref_Ts)) for b in range(a + 1, len(ref_Ts)))
                if spread > 0.03:
                    continue
            Tf = Tf1
            Tr = apt._mean_transform(ref_Ts) if len(ref_Ts) > 1 else ref_Ts[0]
            # DIVERSITY gate: require a NEW peg pose vs the last sample (so C isn't fit to one spot)
            if last_pose is not None:
                dp, dr = _delta(Tf, last_pose)
                if dp < 0.02 and dr < 10.0:
                    continue
            last_pose = Tf
            pairs.append((Tf, Tr))
            print(f"\r[refine] {len(pairs)}/{args.refine_frames}", end="", flush=True)
    except KeyboardInterrupt:
        print()

    if len(pairs) < 5:
        print(f"\n[refine] only {len(pairs)} frames; too few to solve -- not saving.")
        return

    # rigid correction C (T_base_cam_front' = C @ T_base_cam_front). Solve rotation and
    # translation separately: translation uses the SINGLE averaged rotation so per-frame
    # rotation noise isn't amplified by the peg's ~0.5 m distance from base.
    Rc = R.from_matrix(np.array([Tr[:3, :3] @ Tf[:3, :3].T for Tf, Tr in pairs])).mean().as_matrix()
    tc = np.mean([Tr[:3, 3] - Rc @ Tf[:3, 3] for Tf, Tr in pairs], axis=0)
    C = np.eye(4)
    C[:3, :3], C[:3, 3] = Rc, tc

    def _perr(Ts):  # mean pos err (mm)
        return float(np.mean([np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in Ts]) * 1000)

    def _rerr(Ts):  # mean rot err (deg)
        return float(np.mean([np.degrees((R.from_matrix(a[:3, :3]) * R.from_matrix(b[:3, :3]).inv()
                                          ).magnitude()) for a, b in Ts]))

    before = [(Tf, Tr) for Tf, Tr in pairs]
    after = [(C @ Tf, Tr) for Tf, Tr in pairs]
    pb, rb, pa, ra = _perr(before), _rerr(before), _perr(after), _rerr(after)
    cdeg = float(np.degrees(R.from_matrix(C[:3, :3]).magnitude()))
    print(f"\n[refine] frames={len(pairs)}")
    print(f"[refine] front-vs-ref  BEFORE: pos {pb:5.1f}mm  rot {rb:4.1f}deg")
    print(f"[refine] front-vs-ref  AFTER : pos {pa:5.1f}mm  rot {ra:4.1f}deg")
    print(f"[refine] correction C: |pos|={np.linalg.norm(C[:3, 3]) * 1000:.1f}mm  |rot|={cdeg:.2f}deg")
    cmm = float(np.linalg.norm(C[:3, 3]) * 1000)
    if pa > pb + 1.0 or ra > rb + 1.0:
        print("[refine] WARNING: correction did NOT reduce BOTH pos and rot -- not saving. "
              "The residual is likely single-tag flip noise, not a rigid offset. Re-pose so the "
              "FRONT clearly sees >=2 tags throughout.")
        return
    if cmm > 60.0 or cdeg > 15.0:
        print(f"[refine] WARNING: correction implausibly large (|pos|={cmm:.0f}mm |rot|={cdeg:.0f}deg) "
              f"-- a true extrinsic bias is ~cm/deg-scale; this is fitting flips. NOT saving.")
        return
    _save_refined_front(SER_FRONT, C @ T_base_cam[front], pb, pa)


def _robust_mean_R(Rs, w, iters=8, eps_deg=0.5):
    """IRLS weighted geodesic MEDIAN on SO(3): start at the weighted chordal mean, then
    reweight by 1/angle each pass so a flipped (outlier) camera is driven out of the fit."""
    eps = np.radians(eps_deg)
    Rm = Rs.mean(weights=w)
    for _ in range(iters):
        ang = (Rs * Rm.inv()).magnitude()
        Rm = Rs.mean(weights=w / np.maximum(ang, eps))
    return Rm


def _robust_mean_t(ts, w, iters=8, eps=1e-3):
    """IRLS weighted geometric median of positions (L1) -- robust to one bad camera."""
    tm = (w[:, None] * ts).sum(0) / w.sum()
    for _ in range(iters):
        d = np.linalg.norm(ts - tm, axis=1)
        ww = w / np.maximum(d, eps)
        tm = (ww[:, None] * ts).sum(0) / ww.sum()
    return tm


def _fuse(ests, weights=None, mode="median"):
    """ests: {name: T_base_peg}; weights: {name: conf>=0}. Return
    (T_centroid, max_pair_mm, rot_spread_deg).

    mode="median" (default): per-camera-confidence-WEIGHTED geodesic median -- high-conf
    cameras pull harder and a single flipped camera is rejected as an IRLS outlier.
    mode="mean": OLD behavior -- plain equal-weight chordal mean (ignores weights). Use to
    A/B whether the weighting/median is the source of jitter.
    """
    names = list(ests)
    Ts = [ests[n] for n in names]
    ps = np.array([T[:3, 3] for T in Ts])
    Rs = R.from_matrix(np.array([T[:3, :3] for T in Ts]))
    if weights is None or mode == "mean":
        w = np.ones(len(Ts))
    else:
        w = np.array([max(float(weights.get(n, 0.0)), 0.0) for n in names])
    if not np.any(w > 0):
        w = np.ones(len(Ts))
    w = w / w.sum()
    if len(Ts) < 2:
        Rc, pc = Rs[0], ps[0]
    elif mode == "mean":
        Rc, pc = Rs.mean(), ps.mean(0)
    elif len(Ts) == 2:
        # a robust MEDIAN needs >=3 (a majority to reject against); with 2 it's degenerate
        # and collapses toward the higher-weight camera -> jitter. Use a weighted MEAN.
        Rc, pc = Rs.mean(weights=w), (w[:, None] * ps).sum(0) / w.sum()
    else:
        Rc, pc = _robust_mean_R(Rs, w), _robust_mean_t(ps, w)
    Tc = np.eye(4)
    Tc[:3, :3], Tc[:3, 3] = Rc.as_matrix(), pc
    max_pair = rot_spread = 0.0
    if len(Ts) >= 2:
        max_pair = max(np.linalg.norm(ps[i] - ps[j])
                       for i in range(len(ps)) for j in range(i + 1, len(ps))) * 1000.0
        rot_spread = float(np.degrees((Rs * Rc.inv()).magnitude().max()))
    return Tc, max_pair, rot_spread


def _open_cameras(use_wrist, res_by_role, color_fps=30):
    """Return {name: (camera, K, dist)} for the connected, configured cameras.

    `res_by_role` is {role: (w,h)}; each RealSense opens at its role's resolution
    (front D455 -> 720p, side D435 / wrist D415 -> 1080p by default) so bigger tag
    pixels make detection/pose easier on every camera, not just the front.
    """
    want = {SER_FRONT: "front", SER_SIDE: "side"}
    if use_wrist:
        want[SER_WRIST] = "wrist"
    # per-serial resolution dict for gather_realsense_cameras
    serial_res = {ser: res_by_role[role] for ser, role in SER_ROLE.items() if role in res_by_role}
    cams = []
    try:                                   # Orbbec is optional (front is now a RealSense D455)
        cams += list(gather_orbbec_cameras(rgb=True, depth=False, align=None))
    except Exception:
        pass
    cams += list(gather_realsense_cameras(rgb=True, depth=False, align=None,
                                          color_wh=serial_res, color_fps=color_fps))
    out = {}
    for c in cams:
        name = want.get(str(c._serial_number))
        if name is None:
            try:
                c.disable_camera()
            except Exception:
                pass
            continue
        intr = c.calibration["intrinsics"]["rgb"]
        out[name] = (c, np.asarray(intr["cameraMatrix"], float), np.asarray(intr["distCoeffs"], float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_ip", default="192.168.1.10", help="UR IP for live joints / wrist FK.")
    ap.add_argument("--no-wrist", action="store_true", help="Skip the wrist camera / robot FK.")
    ap.add_argument("--publish", nargs="?", const=DEFAULT_STATE_FILE, default=None, metavar="PATH",
                    help="Write live {joints, T_base_peg} to a JSON state file each loop (for the "
                         f"IsaacLab digital twin). PATH optional (default {DEFAULT_STATE_FILE}).")
    ap.add_argument("--tag-view", action="store_true",
                    help="Show a live camera strip with the detected AprilTags drawn "
                         "(peg=green, hole=orange, ROI=blue). Independent of --headless "
                         "(which only suppresses the Open3D scene); needs a display.")
    ap.add_argument("--headless", action="store_true",
                    help="No Open3D window; just estimate + publish (run alongside the sim GUI).")
    ap.add_argument("--refine-front", action="store_true",
                    help="Collect co-visible peg frames and solve a rigid correction so the front "
                         "camera agrees with the reference cameras; overwrites the front extrinsic (backs up).")
    ap.add_argument("--refine-frames", type=int, default=20, help="frames to collect for --refine-front")
    ap.add_argument("--ref-cams", default="side,wrist",
                    help="reference cameras front is corrected to match (comma-separated)")
    ap.add_argument("--color-res", default=None,
                    help="Override color WxH for ALL cameras (e.g. 1280x720). Default: per-role "
                         "(front 1280x720, side/wrist 1920x1080). A per-role flag below wins over this.")
    ap.add_argument("--front-res", default=None, help="front (D455) color WxH; caps ~1280x800.")
    ap.add_argument("--side-res", default=None, help="side (D435) color WxH; up to 1920x1080.")
    ap.add_argument("--wrist-res", default=None, help="wrist (D415) color WxH; up to 1920x1080.")
    ap.add_argument("--color-fps", type=int, default=30,
                    help="RealSense color fps (30 at these resolutions).")
    ap.add_argument("--pose-mode", choices=["joint", "per-tag"], default="joint",
                    help="per-camera pose: 'joint' (one PnP over all a cam's tags) or "
                         "'per-tag' (OLD: solve each tag, frontal-weighted average).")
    ap.add_argument("--merge-mode", choices=["median", "mean"], default="median",
                    help="cross-camera merge: 'median' (conf-weighted robust) or "
                         "'mean' (OLD: plain equal-weight chordal mean).")
    ap.add_argument("--no-prior", action="store_true",
                    help="disable the single-tag temporal branch-selection prior.")
    ap.add_argument("--prior-hold", type=int, default=15,
                    help="frames to keep a camera's flip-disambiguation prior alive through tag "
                         "dropouts before clearing it (prevents re-acquisition flips). 0 = clear immediately.")
    ap.add_argument("--allow-single-tag", action="store_true",
                    help="let 1-tag cameras vote in the centroid even when another camera sees >=2 "
                         "tags (default: exclude them -- a single planar tag is flip-ambiguous).")
    ap.add_argument("--fuse-exclude", default="",
                    help="comma-separated cameras to keep OUT of the fused centroid entirely (still "
                         "estimated/drawn/published). e.g. 'front' -- the D455 sees 1 tag most frames "
                         "and is extrinsically biased; excluding it gives side/wrist-only accuracy.")
    ap.add_argument("--front-frame", nargs="?", const="/dev/shm/real_front_frame.npy",
                    default=None, metavar="PATH",
                    help="Dump the real front-camera RGB each loop (for twin_frame_viewer to show "
                         "beside the sim overview). PATH optional (default /dev/shm/real_front_frame.npy).")
    ap.add_argument("--record-control", default=None, metavar="PATH",
                    help="Poll this record-control JSON (written by eval_real_robot.py --save_video) "
                         "and save per-camera videos to <video_dir>/<episode>/<cam_idx>.mp4 while it "
                         "says recording -- the same layout RealEnv-with-cameras uses "
                         "(front=0, side=1, wrist=2).")
    ap.add_argument("--record-fps", type=int, default=30,
                    help="Real-time output FPS for the recorded videos (default 30). Frames are "
                         "held/dropped to match wall-clock, so playback is real-time regardless "
                         "of the fusion loop rate.")
    ap.add_argument("--publish-hz", type=float, default=30.0,
                    help="Bounded fusion/publish loop rate. Camera capture and detection remain "
                         "fully asynchronous (default 30 Hz).")
    ap.add_argument("--result-max-age-ms", type=float, default=200.0,
                    help="Do not fuse a camera result older than this at publish time (default 200ms).")
    ap.add_argument("--fusion-skew-ms", type=float, default=80.0,
                    help="Only merge camera results captured within this much time of the newest "
                         "fresh result, avoiding motion smear (default 80ms).")
    ap.add_argument("--wrist-fusion-skew-ms", type=float, default=200.0,
                    help="Wrist-specific capture skew allowance. Wrist full-frame detection is "
                         "slower, so a valid fresh wrist result remains eligible even when fixed "
                         "cameras have newer results (default 200ms).")
    ap.add_argument("--hole-tags", nargs="?", const=tb.DEFAULT_HOLE_TAGS, default="__auto__",
                    metavar="PATH",
                    help="Track the STATIC peg-hole from its own AprilTags and publish "
                         "T_base_hole alongside the peg. PATH is the hole_tags.json written by "
                         f"calibrate_hole_tags.py (default {tb.DEFAULT_HOLE_TAGS}, used "
                         "automatically when it exists). Use --no-hole-tags to disable.")
    ap.add_argument("--no-hole-tags", action="store_true",
                    help="Disable peg-hole tag tracking even if hole_tags.json exists.")
    ap.add_argument("--hole-scan-interval", type=float, default=0.5,
                    help="Seconds between full-frame peg-hole scans per camera (default 0.5). "
                         "The hole is static, so this is deliberately slow -- its cost is "
                         "amortized and never competes with peg tracking.")
    ap.add_argument("--hole-window", type=int, default=120,
                    help="Peg-hole observations retained per camera for the running robust "
                         "average (default 120).")
    ap.add_argument("--hole-min-tags", type=int, default=2,
                    help="Require >= this many hole tags per camera estimate (default 2; a "
                         "single planar tag has a 2-fold flip ambiguity).")
    ap.add_argument("--no-roi", action="store_true",
                    help="Disable pose-projected tracking ROIs and always detect full-frame.")
    ap.add_argument("--roi-padding-px", type=float, default=128.0,
                    help="Fixed pixel margin around the projected peg ROI (default 128).")
    ap.add_argument("--roi-padding-fraction", type=float, default=0.6,
                    help="Additional ROI margin as a fraction of projected peg span (default 0.6).")
    ap.add_argument("--full-scan-interval", type=float, default=0.25,
                    help="Minimum seconds between expensive full-frame reacquisition scans for an "
                         "unseen camera (default 0.25).")
    args = ap.parse_args()
    if args.publish_hz <= 0:
        ap.error("--publish-hz must be > 0")
    if args.result_max_age_ms <= 0:
        ap.error("--result-max-age-ms must be > 0")
    if args.fusion_skew_ms < 0:
        ap.error("--fusion-skew-ms must be >= 0")
    if args.wrist_fusion_skew_ms < 0:
        ap.error("--wrist-fusion-skew-ms must be >= 0")
    if args.roi_padding_px < 0 or args.roi_padding_fraction < 0:
        ap.error("ROI padding values must be >= 0")
    if args.full_scan_interval <= 0:
        ap.error("--full-scan-interval must be > 0")

    def _parse_res(s):
        return tuple(int(x) for x in s.lower().split("x"))

    # resolve per-role resolution: per-role flag > global --color-res > DEFAULT_RES
    res_by_role = {}
    for role in ("front", "side", "wrist"):
        override = getattr(args, f"{role}_res")
        if override:
            res_by_role[role] = _parse_res(override)
        elif args.color_res:
            res_by_role[role] = _parse_res(args.color_res)
        else:
            res_by_role[role] = DEFAULT_RES[role]
    pose_mode = args.pose_mode.replace("-", "_")     # "per-tag" -> "per_tag" for estimate_peg_pose_ex
    fuse_exclude = {n.strip() for n in args.fuse_exclude.split(",") if n.strip()}
    print(f"[fusion] pose={args.pose_mode}  merge={args.merge_mode}  prior={'off' if args.no_prior else 'on'}"
          + (f"  fuse_exclude={sorted(fuse_exclude)}" if fuse_exclude else ""))
    headless = args.headless

    T_base_cam = {"front": _load_static_extrinsic(SER_FRONT),
                  "side": _load_static_extrinsic(SER_SIDE)}
    use_wrist = not args.no_wrist
    T_wrist3_cam = None
    rtde_r = None
    fk = None
    # Connect the robot if the wrist camera OR publishing needs live joints.
    if use_wrist or args.publish is not None:
        try:
            import rtde_receive
            from diffusion_policy.real_world.ur5e_kinematics import forward_kinematics_calibrated
            rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
            fk = forward_kinematics_calibrated
            print(f"[fusion] connected to robot {args.robot_ip} (live joints)")
        except Exception as e:
            print(f"[fusion] no robot ({e}); wrist + joint publishing disabled")
            rtde_r = None
    if use_wrist and rtde_r is not None:
        T_wrist3_cam, wp = _load_wrist_offset()
        print(f"[fusion] wrist offset from {wp}")
    else:
        use_wrist = False

    cams = _open_cameras(use_wrist, res_by_role, color_fps=args.color_fps)
    print(f"[fusion] color res: " + "  ".join(f"{r}={res_by_role[r][0]}x{res_by_role[r][1]}"
                                              for r in ("front", "side", "wrist")))
    if not cams:
        raise SystemExit("no configured cameras connected.")
    print(f"[fusion] cameras: {sorted(cams)}")
    if args.publish is not None:
        print(f"[fusion] publishing state -> {args.publish}")
    recorder = None
    if args.record_control is not None:
        recorder = _EpisodeVideoRecorder(args.record_control, args.record_fps)
        print(f"[fusion] recording camera videos per record-control {args.record_control} "
              f"@ {recorder.fps}fps (wall-clock, real-time) -> <video_dir>/<episode>/<cam_idx>.mp4")

    T_peg_tag = {f["id"]: apt.build_T_peg_tag(apt.PEG_DIMS_M, f) for f in apt.FACES}

    # ---- static peg-hole body (optional) ----
    hole_body = None
    hole_acc = None
    hole_path = None if args.no_hole_tags else args.hole_tags
    if hole_path == "__auto__":
        hole_path = tb.DEFAULT_HOLE_TAGS if os.path.exists(tb.DEFAULT_HOLE_TAGS) else None
    if hole_path:
        if not os.path.exists(hole_path):
            raise SystemExit(f"missing {hole_path} -- run calibrate_hole_tags.py first "
                             "(or pass --no-hole-tags).")
        hole_body = tb.TagBody.load(hole_path)
        hole_acc = _HoleAccumulator(window=args.hole_window, min_tags=args.hole_min_tags)
        print(f"[hole] {hole_body} from {hole_path}  "
              f"(scan every {args.hole_scan_interval:g}s, window {args.hole_window}, "
              f"min_tags {args.hole_min_tags})")
    else:
        print("[hole] peg-hole tag tracking OFF (no hole_tags.json); consumers fall back to "
              "their configured fixed hole pose.")

    if args.refine_front:
        detector = apt.make_detector()
        _run_refine(cams, detector, T_peg_tag, T_base_cam, rtde_r, fk, T_wrist3_cam, args)
        for (c, _, _) in cams.values():
            try:
                c.disable_camera()
            except Exception:
                pass
        return

    # --- Open3D scene (skipped when headless) ---
    vis = cam_box = cam_triad = centroid_box = centroid_triad = hole_box = hole_triad = None
    if not headless:
        vis = o3d.visualization.Visualizer()
        vis.create_window("peg fusion (base frame) -- ESC to quit", width=1280, height=860)
        vis.get_render_option().line_width = 4.0
        vis.get_render_option().background_color = np.array([0.05, 0.05, 0.06])
        world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        vis.add_geometry(world)
        cam_box = {n: o3d.geometry.LineSet() for n in ("front", "side", "wrist")}
        cam_triad = {n: o3d.geometry.LineSet() for n in ("front", "side", "wrist")}
        centroid_box = o3d.geometry.LineSet()
        centroid_triad = o3d.geometry.LineSet()
        hole_box = o3d.geometry.LineSet()
        hole_triad = o3d.geometry.LineSet()
        for n in ("front", "side", "wrist"):
            vis.add_geometry(cam_box[n]); vis.add_geometry(cam_triad[n])
        vis.add_geometry(centroid_box); vis.add_geometry(centroid_triad)
        vis.add_geometry(hole_box); vis.add_geometry(hole_triad)

    workers = {
        name: AsyncCameraPoseWorker(
            name=name,
            camera=cam,
            K=K,
            dist=dist,
            T_peg_tag=T_peg_tag,
            pose_mode=pose_mode,
            use_prior=not args.no_prior,
            prior_hold=args.prior_hold,
            use_roi=not args.no_roi,
            roi_padding_px=args.roi_padding_px,
            roi_padding_fraction=args.roi_padding_fraction,
            full_scan_interval_s=args.full_scan_interval,
            aux_body=hole_body,
            aux_interval_s=args.hole_scan_interval,
        ).start()
        for name, (cam, K, dist) in cams.items()
    }
    print(f"[fusion] async camera workers started; publish={args.publish_hz:g}Hz"
          f"  roi={'off' if args.no_roi else 'on'}"
          f"  max_age={args.result_max_age_ms:g}ms"
          f"  max_skew={args.fusion_skew_ms:g}ms"
          f"  wrist_skew={args.wrist_fusion_skew_ms:g}ms")

    first = True
    status_last = -float("inf")
    error_last = -float("inf")
    joint_history = deque(maxlen=max(32, int(args.publish_hz * 3)))
    period = 1.0 / args.publish_hz
    next_tick = time.monotonic()

    def joints_near(stamp, fallback):
        """Nearest sampled robot configuration to a camera capture (primarily for wrist FK)."""
        if not joint_history:
            return fallback
        return min(joint_history, key=lambda item: abs(item[0] - stamp))[1]

    def joints_quiet(stamp, window=0.15, tol=0.01):
        """Was the arm essentially still around ``stamp``?

        The hole accumulator averages over MANY seconds, so a wrist sample whose FK is
        mistimed against a moving arm would be a systematic (not zero-mean) error. Admit wrist
        hole observations only while the arm is quiet, where FK-vs-capture skew cannot bite.
        """
        near = [q for t, q in joint_history if abs(t - stamp) <= window]
        if len(near) < 2:
            return False
        arr = np.asarray(near, dtype=np.float64)
        return float(np.max(arr.max(0) - arr.min(0))) < tol

    print("\nlegend: front=RED  side=GREEN  wrist=BLUE  centroid=WHITE box + RGB axes\n")
    try:
        while True:
            # Bounded-rate publisher: it snapshots whatever each independent detector has ready;
            # a slow/full-frame/occluded camera never delays the other cameras or this deadline.
            now_mono = time.monotonic()
            if now_mono < next_tick:
                time.sleep(next_tick - now_mono)
            elif now_mono - next_tick > period:
                next_tick = now_mono
            next_tick += period

            joints = list(rtde_r.getActualQ()) if rtde_r is not None else None
            tick_wall = time.time()
            if joints is not None:
                joint_history.append((tick_wall, joints))

            results = {name: worker.latest_result() for name, worker in workers.items()}
            fresh = {}
            for name, result in results.items():
                if result is None or result["T_cam_peg"] is None:
                    continue
                age_s = tick_wall - result["capture_stamp"]
                if -0.02 <= age_s <= args.result_max_age_ms * 1e-3:
                    fresh[name] = result
            newest_capture = max(
                (r["capture_stamp"] for r in fresh.values()), default=None)

            ests = {}
            weights = {}       # name -> per-camera confidence (fusion weight)
            cam_info = {}      # name -> {"T": T_base_peg|None, "seen": bool, "tags": [ids]}
            capture_by_name = {}
            ntags = {}
            for name in cams:
                result = results[name]
                is_fresh = name in fresh
                tags = result["tags"] if result is not None and is_fresh else []
                ntags[name] = len(tags)
                T_base_peg = None
                T_bc = None
                if is_fresh:
                    T_cam_peg = result["T_cam_peg"]
                    if name == "wrist":
                        q_capture = joints_near(result["capture_stamp"], joints)
                        if q_capture is not None and T_wrist3_cam is not None:
                            T_bc = fk(q_capture)[0] @ T_wrist3_cam
                    else:
                        T_bc = T_base_cam[name]
                    if T_bc is not None:
                        T_base_peg = T_bc @ T_cam_peg

                skew_limit_ms = (
                    args.wrist_fusion_skew_ms if name == "wrist"
                    else args.fusion_skew_ms
                )
                coherent = (
                    T_base_peg is not None
                    and newest_capture is not None
                    and newest_capture - result["capture_stamp"] <= skew_limit_ms * 1e-3
                )
                cam_info[name] = {
                    "T": T_base_peg,
                    "seen": T_base_peg is not None,
                    "tags": tags,
                    "capture_stamp": result["capture_stamp"] if result is not None else None,
                    "detect_ms": result["detect_ms"] if result is not None else None,
                    "used_roi": result["used_roi"] if result is not None else None,
                }
                if not headless:
                    if T_bc is None:
                        if name == "wrist" and joints is not None and T_wrist3_cam is not None:
                            T_bc = fk(joints)[0] @ T_wrist3_cam
                        elif name != "wrist":
                            T_bc = T_base_cam[name]
                    _set_triad(cam_triad[name], T_bc, 0.05)
                    _set_box(cam_box[name], T_base_peg, apt.PEG_DIMS_M, COLOR[name])
                if coherent:
                    ests[name] = T_base_peg
                    weights[name] = result["info"]["conf"]
                    capture_by_name[name] = result["capture_stamp"]

            # ---- static peg-hole: accumulate every distinct observation, all cameras, over time.
            # No freshness/skew gate -- the hole does not move, so an older sample is just as
            # valid as a new one, and more samples is strictly better.
            hole_state = None
            if hole_acc is not None:
                for name in cams:
                    result = results.get(name)
                    aux = result.get("aux") if result is not None else None
                    if aux is None:
                        continue
                    if name == "wrist":
                        if not joints_quiet(aux["capture_stamp"]):
                            continue          # FK/capture skew would bias a long-running average
                        q_aux = joints_near(aux["capture_stamp"], joints)
                        if q_aux is None or T_wrist3_cam is None:
                            continue
                        T_bc_aux = fk(q_aux)[0] @ T_wrist3_cam
                    else:
                        T_bc_aux = T_base_cam[name]
                    hole_acc.push(name, aux["capture_stamp"], T_bc_aux @ aux["T_cam_aux"],
                                  aux["conf"], aux["n_tags"])
                T_hole, n_hole, hole_pos_mm, hole_rot_deg, hole_cams = hole_acc.estimate()
                hole_state = {
                    "T": T_hole, "n": int(n_hole), "spread_mm": float(hole_pos_mm),
                    "spread_deg": float(hole_rot_deg), "cams": hole_cams,
                    "tags": sorted({t for name in cams
                                    for t in ((results.get(name) or {}).get("aux") or {}).get("tags", [])}),
                }

            # A single planar tag is fundamentally ambiguous -> when ANY camera sees >=2 tags,
            # drop the 1-tag cameras from the CENTROID (weight 0) so a flipping single-tag cam
            # cannot pollute the fused pose. They're still reported for visibility diagnostics.
            if not args.allow_single_tag and ests:
                best = max(ntags.get(n, 0) for n in ests)
                if best >= 2:
                    for name in ests:
                        if ntags.get(name, 0) < 2:
                            weights[name] = 0.0
            for name in fuse_exclude:
                if name in weights:
                    weights[name] = 0.0

            Tc = None
            max_pair = rot_spread = 0.0
            if ests:
                Tc, max_pair, rot_spread = _fuse(ests, weights, mode=args.merge_mode)

            now_mono = time.monotonic()
            if now_mono - status_last >= 0.2:
                status_last = now_mono
                wsum = sum(weights.get(n, 0.0) for n in ests) or 1.0
                parts = []
                for name in ("front", "side", "wrist"):
                    result = results.get(name)
                    if name in ests:
                        p = ests[name][:3, 3]
                        wf = 100.0 * weights.get(name, 0.0) / wsum
                        pose = f"[{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}]{wf:3.0f}%"
                    else:
                        pose = "--"
                    if result is None:
                        perf = "wait"
                    else:
                        mode = "R" if result["used_roi"] else ("F*" if result["fallback_full"] else "F")
                        perf = f"{result['detect_ms']:.0f}{mode}"
                    parts.append(f"{name}={pose}/{perf}")
                if Tc is not None:
                    agree = (f"pair={max_pair:4.1f}mm rot={rot_spread:3.1f}deg"
                             if len(ests) >= 2 else "1cam")
                    center = f"c=[{Tc[0,3]:+.3f} {Tc[1,3]:+.3f} {Tc[2,3]:+.3f}] {agree}"
                else:
                    center = "no coherent fresh pose"
                if hole_state is not None:
                    if hole_state["T"] is not None:
                        hp = hole_state["T"][:3, 3]
                        center += (f" | hole=[{hp[0]:+.3f} {hp[1]:+.3f} {hp[2]:+.3f}]"
                                   f" n={hole_state['n']} sp={hole_state['spread_mm']:.1f}mm")
                    else:
                        center += " | hole=--"
                print(f"\r{'  '.join(parts)} | {center}" + " " * 8, end="", flush=True)

            if now_mono - error_last >= 2.0:
                errors = [f"{name}: {worker.error}" for name, worker in workers.items()
                          if worker.error]
                if errors:
                    error_last = now_mono
                    print("\n[camera][WARN] " + " | ".join(errors))

            if args.publish is not None:
                # Timestamp only actual nonzero-weight contributors, rather than a blind/excluded
                # camera that happened to be read first.
                contributors = [n for n in ests if weights.get(n, 0.0) > 0.0]
                if not contributors:
                    contributors = list(ests)
                capture_stamp = min(
                    (capture_by_name[n] for n in contributors), default=None)
                _write_state(args.publish, joints, Tc, cam_info, capture_stamp=capture_stamp,
                             hole=hole_state)

            latest_frames = {name: worker.latest_frame() for name, worker in workers.items()}
            front_frame = latest_frames.get("front")
            if args.front_frame is not None and front_frame is not None:
                _write_front_frame(args.front_frame, front_frame["rgb"])
            if recorder is not None:
                recorder.step({name: frame["rgb"] for name, frame in latest_frames.items()
                               if frame is not None})
            if args.tag_view:
                canvas = _draw_tag_view(latest_frames, results, fresh)
                if canvas is not None:
                    cv2.imshow("apriltag view", canvas)
                    cv2.waitKey(1)

            if not headless:
                _set_box(centroid_box, Tc, apt.PEG_DIMS_M, (1, 1, 1))
                _set_triad(centroid_triad, Tc, 0.09)
                T_hole_vis = None
                if hole_state is not None and hole_state["T"] is not None:
                    # the hole's USD origin is not its bbox centre -- shift the wireframe so it
                    # sits on the physical block instead of around the frame origin
                    T_hole_vis = hole_state["T"].copy()
                    T_hole_vis[:3, 3] += T_hole_vis[:3, 2] * tb.HOLE_BBOX_CENTER_Z
                _set_box(hole_box, T_hole_vis, hole_body.dims_m if hole_body else tb.HOLE_DIMS_M,
                         (1.0, 0.8, 0.2))
                _set_triad(hole_triad, hole_state["T"] if hole_state else None, 0.07)
                for g in (*cam_box.values(), *cam_triad.values(), centroid_box, centroid_triad,
                          hole_box, hole_triad):
                    vis.update_geometry(g)
                if first:
                    vis.reset_view_point(True)
                    first = False
                if not vis.poll_events():
                    break
                vis.update_renderer()
    except KeyboardInterrupt:
        pass
    finally:
        for worker in workers.values():
            worker.stop()
        if recorder is not None:
            recorder.close()
        if vis is not None:
            vis.destroy_window()
        for (cam, _, _) in cams.values():
            try:
                cam.disable_camera()
            except Exception:
                pass


if __name__ == "__main__":
    main()
