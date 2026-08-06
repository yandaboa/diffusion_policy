"""Estimate the camera-relative pose of a rectangular peg from AprilTags on its faces.

The peg is a cuboid with one AprilTag glued on each of its 6 faces. Every tag has a
fixed, known transform to the peg *center* (pure geometry). So each visible tag votes
for the body pose independently; we fuse the votes (weighted by how frontal each tag
is) into one robust ``T_cam_peg``.

Pipeline per frame:
  1. detect AprilTags  (cv2.aruco, DICT_APRILTAG_36h11)
  2. per tag: solvePnP(4 corners, tag_size, K, dist, IPPE_SQUARE) -> T_cam_tag_i
  3. per tag: T_cam_peg_i = T_cam_tag_i @ inv(T_peg_tag_i)     (T_peg_tag_i is a constant)
  4. fuse: weighted rotation/translation mean + outlier rejection

Tag frame convention (matches multi_camera_wrapper._estimatePoseSingleMarkers):
    +X right, +Y up, +Z out of the tag toward the viewer.

Usage (run inside the `foundstereo` conda env):
    # 1. Read off which printed tag ID is on which face -- hold each face to the camera:
    python apriltag_peg_pose.py --identify

    # 2. Fill in TAG_SIZE_M, PEG_DIMS_M, and FACES below, then estimate live:
    python apriltag_peg_pose.py --run
"""
import argparse
import itertools
import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perception.multi_camera_wrapper import MultiCameraWrapper

# ----------------------------------------------------------------------------------
# ---- CONFIGURE THESE for your physical peg ----------------------------------------
# ----------------------------------------------------------------------------------
ARUCO_DICT = cv2.aruco.DICT_APRILTAG_16h5    # tag16h5 family (IDs 0-29)
TAG_SIZE_M = 0.025                            # 25mm black-edge length
PEG_DIMS_M = (0.030, 0.030, 0.060)            # peg.usd bbox: 30x30x60mm, long axis +Z

# Body frame == the peg.usd frame: origin at the geometric center, +Z is the long axis,
# +X/+Y span the 30mm square cross-section. Recovered T_cam_peg is in THIS frame, so it
# lines up with the sim asset.
#
# For each face give:
#   id      : the AprilTag ID printed on that face (from --identify)
#   normal  : outward face normal in body coords, one of the six unit axes
#   up      : which body direction the TOP EDGE of the printed tag points to (tag +Y).
#             MUST be perpendicular to `normal`. This is the in-plane orientation you
#             set when gluing the tag on -- the one thing you can't get from CAD.
#
# Convention below: on the 4 long side faces the tag's top edge points toward the peg
# tip (+Z). On the 2 square end faces the tag's top edge points toward +Y. GLUE THE
# TAGS TO MATCH, or fix `up` empirically (see the disagreement warning at runtime).
FACES = [
    {"id": 0, "normal": (0, 1, 0), "up": (0.0151, 0.0, 0.9999)},
    {"id": 1, "normal": (-1, 0, 0), "up": (0.0, -0.0124, 0.9999)},
    {"id": 2, "normal": (0, 0, 1), "up": (0.0, 1.0, 0.0)},
    {"id": 3, "normal": (1, 0, 0), "up": (0.0, 0.0037, -1.0)},
    {"id": 4, "normal": (0, -1, 0), "up": (0.0324, 0.0, 0.9995)},
    {"id": 5, "normal": (0, 0, -1), "up": (0.0164, 0.9999, 0.0)},
]
# ----------------------------------------------------------------------------------

# tag-frame corner model, ordered to match cv2.aruco corner order: TL, TR, BR, BL.
_TAG_OBJP = np.array([
    [-0.5,  0.5, 0], [0.5,  0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0],
], dtype=np.float32)


def _hat(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def build_T_peg_tag(dims, face):
    """Constant transform from peg-center frame to a face tag's frame.

    Tag +Z = outward normal; tag +Y = `up`; tag +X = up x normal (right-handed).
    Translation = center of the face = normal * (half-extent along that axis).
    """
    dims = np.asarray(dims, float)
    n = _hat(face["normal"])
    y = _hat(face["up"])
    if abs(float(n @ y)) > 1e-6:
        raise ValueError(f"face id={face['id']}: `up` must be perpendicular to `normal`")
    z = n                       # tag looks outward
    x = _hat(np.cross(y, z))    # right-handed: x = y x z
    Rpt = np.column_stack([x, y, z])          # tag axes in body coords
    axis = int(np.argmax(np.abs(n)))          # which body axis this face faces
    t = n * (dims[axis] / 2.0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rpt, t
    return T


def _Rz4(k):
    """4x4 in-plane rotation by k*90 deg about the tag's own +Z axis."""
    a = k * np.pi / 2.0
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def make_detector(aruco_dict=None):
    """Detector for ``aruco_dict`` (a cv2.aruco.DICT_* constant); defaults to the peg's family."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT if aruco_dict is None else aruco_dict)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG  # subpixel -> better PnP

    # --- binarization / "contrast" threshold (the adaptive-threshold stage) ----------
    # Each pixel is compared to the mean of a WinSize x WinSize neighborhood minus
    # `adaptiveThreshConstant`. LOWER constant = more lenient = picks up low-contrast /
    # dim / glare-washed tags (but more noise). Sweeping several window sizes helps when
    # the tag's apparent size (and so its ideal window) varies with distance.
    p.adaptiveThreshConstant = 3        # default 7; try 3-5 for faint/low-contrast tags
    p.adaptiveThreshWinSizeMin = 3      # default 3
    p.adaptiveThreshWinSizeMax = 23     # default 23; raise for big/close tags
    p.adaptiveThreshWinSizeStep = 10    # default 10; smaller step = finer sweep (slower)

    # How much bit-error to tolerate when decoding the marker (0..1). HIGHER = more
    # forgiving of a partially-degraded tag, at the cost of more false IDs.
    p.errorCorrectionRate = 0.6         # default 0.6

    return cv2.aruco.ArucoDetector(d, p)


def _poly_area_px(c):
    """Pixel area of a 4-corner marker (shoelace)."""
    p = np.asarray(c, float).reshape(4, 2)
    return 0.5 * abs(float(p[:, 0] @ np.roll(p[:, 1], -1) - p[:, 1] @ np.roll(p[:, 0], -1)))


def _compose_T(rvec, tvec):
    Rm, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rm, np.asarray(tvec).ravel()
    return T


def _reproj_rms(objp, imgp, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    d = proj.reshape(-1, 2) - np.asarray(imgp, float).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _estimate_per_tag(tags, objp_tag, K, dist, T_peg_tag, verbose=False):
    """OLD path: solve each tag independently (IPPE) then frontal-weighted average with a
    robust reject pass (fuse_votes). Kept for A/B comparison against the joint solve."""
    votes = []
    objp = objp_tag.astype(np.float32)
    for i, cc in tags:
        ok, rvec, tvec = cv2.solvePnP(objp, cc.astype(np.float32), K, dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        T_cam_tag = _compose_T(rvec, tvec)
        tag_z_cam = T_cam_tag[:3, 2]
        to_cam = -_hat(np.asarray(tvec).ravel())
        frontal = max(0.0, float(tag_z_cam @ to_cam))
        votes.append((frontal, T_cam_tag @ _inv(T_peg_tag[i])))
    if not votes:
        return None
    return fuse_votes(votes, verbose=verbose)


def estimate_peg_pose(gray, detector, K, dist, T_peg_tag, verbose=False, mode="joint",
                      tag_size=None):
    """Back-compat: return (T_cam_peg [4x4] or None, list of per-tag debug dicts)."""
    T, dbg, _ = estimate_peg_pose_ex(gray, detector, K, dist, T_peg_tag, verbose=verbose,
                                     mode=mode, tag_size=tag_size)
    return T, dbg


def estimate_peg_pose_ex(gray, detector, K, dist, T_peg_tag, prior=None, verbose=False,
                         mode="joint", tag_size=None):
    """Estimate T_cam_peg via a JOINT multi-tag PnP, plus a per-camera confidence weight.

    Unlike the old per-tag-then-average path, when >=2 tags are visible we solve ONE PnP
    over ALL their corners against the rigid peg model. Tags on different faces are
    non-coplanar, which removes the single-tag IPPE flip ambiguity almost entirely.

    A single visible tag is a planar (IPPE) solve with an inherent 2-fold ambiguity: we
    take BOTH solutions from solvePnPGeneric and pick the branch closest to `prior` (the
    previous T_cam_peg) when given, else the lower-reprojection one; the two solutions'
    error ratio becomes an ambiguity confidence.

    ``T_peg_tag`` is any rigid tag body ({tag_id: 4x4 body->tag}); ``tag_size`` overrides the
    module-level ``TAG_SIZE_M`` so a second body (e.g. the peg-hole) can carry its own tag size.

    Returns (T_cam_peg | None, dbg, info) where info = {n_tags, reproj_rms(px),
    size_px(mean tag edge px), ambiguity(e2/e1 or None), conf(>=0 fusion weight)}.
    """
    tag_size = TAG_SIZE_M if tag_size is None else float(tag_size)
    corners, ids, _ = detector.detectMarkers(gray)
    dbg = []
    info = {"n_tags": 0, "reproj_rms": None, "size_px": 0.0, "ambiguity": None, "conf": 0.0}
    if ids is None:
        return None, dbg, info

    objp_tag = (_TAG_OBJP * tag_size).astype(np.float64)        # (4,3) corners in tag frame
    tags = []
    for c, i in zip(corners, ids.flatten()):
        i = int(i)
        if i not in T_peg_tag:
            continue
        cc = np.asarray(c, float).reshape(4, 2)
        tags.append((i, cc))
        dbg.append({"id": i, "corners": c, "area_px": _poly_area_px(cc)})
    if not tags:
        return None, dbg, info

    n = len(tags)
    areas = np.array([_poly_area_px(cc) for _, cc in tags])
    size_px = float(np.sqrt(max(areas.mean(), 0.0)))            # ~ mean tag edge length (px)
    info.update(n_tags=n, size_px=size_px)

    T_cam_peg = None
    if mode == "per_tag":
        # ---- OLD path: per-tag IPPE + frontal-weighted robust average (for A/B) ----
        T_cam_peg = _estimate_per_tag(tags, objp_tag, K, dist, T_peg_tag, verbose=verbose)
    elif n >= 2:
        # ---- JOINT solve: every corner lifted into the peg body frame, one PnP ----
        OBJ, IMG = [], []
        for i, cc in tags:
            Tpt = T_peg_tag[i]                                 # peg-frame pt = Tpt @ tag-frame pt
            OBJ.append((Tpt[:3, :3] @ objp_tag.T).T + Tpt[:3, 3])
            IMG.append(cc)
        OBJ = np.concatenate(OBJ).astype(np.float32)
        IMG = np.concatenate(IMG).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(OBJ, IMG, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            rvec, tvec = cv2.solvePnPRefineLM(OBJ, IMG, K, dist, rvec, tvec)
            T_cam_peg = _compose_T(rvec, tvec)                 # peg->cam == T_cam_peg
            info["reproj_rms"] = _reproj_rms(OBJ, IMG, rvec, tvec, K, dist)
    else:
        # ---- single tag: IPPE, resolve the 2-fold planar ambiguity ----
        i, cc = tags[0]
        objp = objp_tag.astype(np.float32)
        imgp = cc.astype(np.float32)
        _, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        cand = [(float(np.ravel(errs[k])[0]), _compose_T(rvecs[k], tvecs[k]) @ _inv(T_peg_tag[i]))
                for k in range(len(rvecs))]
        cand.sort(key=lambda z: z[0])
        if len(cand) >= 2 and cand[0][0] > 1e-9:
            info["ambiguity"] = float(cand[1][0] / cand[0][0])
        if prior is not None and len(cand) >= 2:
            def _gd(A, B):
                return float((R.from_matrix(A[:3, :3]) * R.from_matrix(B[:3, :3]).inv()).magnitude())
            T_cam_peg = min(cand, key=lambda z: _gd(z[1], prior))[1]
        else:
            T_cam_peg = cand[0][1]
        info["reproj_rms"] = float(cand[0][0])

    if T_cam_peg is None:
        return None, dbg, info

    # ---- per-camera confidence ~ rotational information (weights the cross-camera merge) ----
    # angular sigma ~ pixel_noise / tag_size_px, improved by more corners -> info ~
    # (size_px * sqrt(#corners) / reproj_rms)^2. Reproj floored so it can't blow up.
    rms = info["reproj_rms"] if info["reproj_rms"] else 1.0
    conf = (size_px * size_px) * (4.0 * n) / (rms * rms + 0.09)
    if n == 1:
        # one planar tag: the out-of-plane (flip) axis is barely constrained -> cut hard,
        # and zero it if the two IPPE branches are nearly tied (ambiguous).
        amb = info["ambiguity"]
        ambf = float(np.clip((amb - 1.0) / 2.0, 0.0, 1.0)) if amb is not None else 0.3
        conf *= 0.25 * ambf
    info["conf"] = float(max(conf, 0.0))
    return T_cam_peg, dbg, info


def fuse_votes(votes, ang_thresh_deg=15.0, pos_thresh_m=0.02, verbose=False):
    """Weighted mean of body-pose votes with one robust outlier-rejection pass."""
    w = np.array([max(v[0], 1e-3) for v in votes])
    Rs = R.from_matrix(np.array([v[1][:3, :3] for v in votes]))
    ts = np.array([v[1][:3, 3] for v in votes])

    def weighted(Rs, ts, w):
        Rm = Rs.mean(weights=w)
        tm = (w[:, None] * ts).sum(0) / w.sum()
        return Rm, tm

    Rm, tm = weighted(Rs, ts, w)
    if len(votes) >= 3:
        ang = np.degrees((Rs * Rm.inv()).magnitude())
        pos = np.linalg.norm(ts - tm, axis=1)
        keep = (ang < ang_thresh_deg) & (pos < pos_thresh_m)
        if keep.sum() >= 2 and keep.sum() < len(votes):
            if verbose:
                print(f"  [fuse] rejected {int((~keep).sum())}/{len(votes)} outlier tag(s)")
            Rm, tm = weighted(Rs[keep], ts[keep], w[keep])
    elif len(votes) == 2 and verbose:
        ang = np.degrees((Rs[0] * Rs[1].inv()).magnitude())
        if ang > ang_thresh_deg:
            print(f"  [fuse] WARNING: 2 tags disagree by {ang:.1f} deg "
                  f"-- check the `up` of one of the co-visible faces")

    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rm.as_matrix(), tm
    return T


def _detect_tag_poses(gray, detector, K, dist, valid_ids, tag_size=None):
    """Return list of (id, T_cam_tag) for every configured tag seen in the frame."""
    corners, ids, _ = detector.detectMarkers(gray)
    out = []
    if ids is None:
        return out
    objp = (_TAG_OBJP * (TAG_SIZE_M if tag_size is None else float(tag_size))).astype(np.float32)
    for c, i in zip(corners, ids.flatten()):
        i = int(i)
        if i not in valid_ids:
            continue
        ok, rvec, tvec = cv2.solvePnP(objp, c.reshape(4, 2), K, dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        Rct, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rct, tvec.ravel()
        out.append((i, T))
    return out


def _vote_spread(quats, ts):
    """(rotational spread deg, translational spread m) of a set of body-pose votes."""
    q = np.array(quats)
    q[np.einsum("ij,j->i", q, q[0]) < 0] *= -1     # hemisphere-align to the first
    qm = q.mean(0)
    qm /= np.linalg.norm(qm)
    ang = np.degrees(2.0 * np.arccos(np.clip(np.abs(q @ qm), 0, 1)))
    t = np.array(ts)
    return float(ang.mean()), float(np.linalg.norm(t - t.mean(0), axis=1).mean())


CANON_NORMALS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _mean_transform(Ts):
    """Chordal mean of a list of 4x4 rigid transforms (quat mean + centroid)."""
    q = np.array([R.from_matrix(T[:3, :3]).as_quat() for T in Ts])
    q[np.einsum("ij,j->i", q, q[0]) < 0] *= -1
    qm = q.mean(0)
    qm /= np.linalg.norm(qm)
    T = np.eye(4)
    T[:3, :3] = R.from_quat(qm).as_matrix()
    T[:3, 3] = np.mean([t[:3, 3] for t in Ts], axis=0)
    return T


def _relative_transforms(frames):
    """{(i,j): (T_tagi_tagj, count)} averaged over every frame where i,j co-occur."""
    from collections import defaultdict
    acc = defaultdict(list)
    for fr in frames:
        for i, Ti in fr:
            for j, Tj in fr:
                if i != j:
                    acc[(i, j)].append(_inv(Ti) @ Tj)
    return {k: (_mean_transform(v), len(v)) for k, v in acc.items()}


def _face_candidates(dims):
    """The 24 valid tag placements: (normal, T_peg_tag) for 6 faces x 4 in-plane spins."""
    cands = []
    for normal in CANON_NORMALS:
        up0 = (0, 0, 1) if abs(normal[2]) < 0.5 else (0, 1, 0)
        base = build_T_peg_tag(dims, {"id": 0, "normal": normal, "up": up0})
        for k in range(4):
            cands.append((normal, base @ _Rz4(k)))
    return cands


def _snap_face(T_est, cands, free_up=False):
    """Snap an estimated T_peg_tag to a valid face placement. Returns
    (normal_tuple, up_tuple, residual_deg).

    Discrete mode (default): nearest of the 24 canonical placements (6 normals x 4
    90-deg spins) -- assumes each tag's top edge is aligned to a face edge; residual is
    the full-orientation snap error.

    ``free_up=True``: snap ONLY the normal to the nearest of 6 axes, but keep the tag's
    *continuous* in-plane orientation for ``up`` (its measured +Y projected into the
    face plane). This models a tag glued at an arbitrary in-plane twist exactly; the
    residual then reports just the normal TILT (how far off-flat the mount is)."""
    Rest = R.from_matrix(T_est[:3, :3])
    best = None
    for normal, T in cands:
        d = np.degrees((Rest * R.from_matrix(T[:3, :3]).inv()).magnitude())
        if best is None or d < best[2]:
            up = tuple(int(v) for v in T[:3, 1].round())
            best = (tuple(normal), up, float(d))
    if not free_up:
        return best

    # Continuous up: snap the normal by the measured outward axis (tag +Z), then take
    # the measured in-plane orientation instead of forcing a 90-deg multiple.
    Rm = T_est[:3, :3]
    z_est = Rm[:, 2]
    normal = max(CANON_NORMALS, key=lambda nn: float(z_est @ _hat(nn)))
    n = _hat(normal)
    tilt = float(np.degrees(np.arccos(np.clip(z_est @ n, -1.0, 1.0))))
    up = Rm[:, 1] - (Rm[:, 1] @ n) * n           # project tag +Y into the face plane
    nrm = np.linalg.norm(up)
    if nrm < 1e-6:
        return best                               # degenerate (+Y ~parallel n): fall back
    up_t = tuple(round(float(v), 4) for v in up / nrm)
    return (tuple(int(v) for v in normal), up_t, tilt)


def _solve_faces_auto(frames, dims, ref_id, ref_face, free_up=False):
    """Recover every tag's (normal, up) from co-visibility, anchored on one known tag.

    Builds each tag's transform relative to `ref_id` by chaining averaged pairwise
    relative poses (BFS over the co-visibility graph), then snaps each to the nearest
    valid cube-face placement. With ``free_up`` the in-plane angle is kept continuous
    (see ``_snap_face``). Returns ({id: (normal, up, residual_deg) | None}, ids).
    """
    from collections import deque
    ids = sorted({i for fr in frames for i, _ in fr})
    rel = _relative_transforms(frames)

    T_peg = {ref_id: build_T_peg_tag(dims, ref_face)}
    dq = deque([ref_id])
    while dq:
        i = dq.popleft()
        nbrs = sorted((j for j in ids if j not in T_peg and (i, j) in rel),
                      key=lambda j: -rel[(i, j)][1])
        for j in nbrs:
            if j not in T_peg:
                T_peg[j] = T_peg[i] @ rel[(i, j)][0]
                dq.append(j)

    cands = _face_candidates(dims)
    out = {i: (_snap_face(T_peg[i], cands, free_up=free_up) if i in T_peg else None)
           for i in ids}
    return out, ids


def run_calibrate_up(n_frames=40, min_tags=2, ref_id=None, free_up=False):
    """Recover the FULL per-face geometry (normal AND up) from multi-tag frames.

    Glue the tags however you like; you don't need the `normal`s in FACES to be right.
    Show the peg from many angles so every pair of adjacent faces is co-visible at some
    point. This chains the tags' relative poses, anchors them on ONE tag you trust
    (`--ref-id`, default = the first observed tag that's already in FACES, using its
    current FACES line as the anchor), and prints a ready-to-paste FACES block.

    Because the peg is symmetric, the anchor is what ties the recovered frame to your
    sim axes: pick the one tag whose face+orientation you are 100% sure of.
    """
    cam, K, dist = _get_cam()
    detector = make_detector()
    valid = {f["id"] for f in FACES}
    print(f"Collecting {n_frames} frames with >= {min_tags} co-visible tags. "
          f"Slowly rotate the peg so every pair of faces is seen together. ESC to stop early.\n")

    frames = []
    while len(frames) < n_frames:
        rgb = cam.read_camera()["rgb"]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        obs = _detect_tag_poses(gray, detector, K, dist, valid)
        if len(obs) >= min_tags:
            frames.append(obs)
            print(f"  frame {len(frames)}/{n_frames}  tags={sorted(i for i, _ in obs)}")
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(vis, f"{len(frames)}/{n_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("calibrate-up (ESC to stop)", vis)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

    observed = sorted({i for fr in frames for i, _ in fr})
    if len(frames) < 3 or len(observed) < 2:
        print("Not enough multi-tag data. Need several frames covering >= 2 faces.")
        return
    print(f"\nCollected {len(frames)} frames covering faces {observed}. Solving...")

    # Anchor tag: the one you trust. Default to the first observed id present in FACES,
    # using its current FACES line as the ground-truth placement.
    faces_by_id = {f["id"]: f for f in FACES}
    if ref_id is None:
        ref_id = next((i for i in observed if i in faces_by_id), observed[0])
    if ref_id not in observed:
        print(f"ref-id {ref_id} was never seen; pick one of {observed}.")
        return
    ref_face = faces_by_id.get(ref_id, {"id": ref_id, "normal": (1, 0, 0), "up": (0, 0, 1)})
    print(f"anchoring on tag {ref_id} -> normal={tuple(ref_face['normal'])} "
          f"up={tuple(ref_face['up'])}\n")

    if free_up:
        print("free-up mode: recovering CONTINUOUS in-plane angle (snap residual = "
              "normal tilt only)\n")
    solved, ids = _solve_faces_auto(frames, PEG_DIMS_M, ref_id, ref_face, free_up=free_up)

    # Verify: rebuild T_peg_tag from the solution and measure residual disagreement.
    T_peg_tag = {i: build_T_peg_tag(PEG_DIMS_M, {"id": i, "normal": v[0], "up": v[1]})
                 for i, v in solved.items() if v is not None}
    tot, rot_tot, trans_tot, wsum = 0.0, 0.0, 0.0, 0
    for fr in frames:
        qs, ts = [], []
        for i, Tct in fr:
            if i in T_peg_tag:
                Tcp = Tct @ _inv(T_peg_tag[i])
                qs.append(R.from_matrix(Tcp[:3, :3]).as_quat())
                ts.append(Tcp[:3, 3])
        if len(qs) >= 2:
            rot, trans = _vote_spread(qs, ts)
            tot += (rot + 1000.0 * trans) * len(qs)
            rot_tot += rot * len(qs)
            trans_tot += trans * len(qs)
            wsum += len(qs)
    score = tot / wsum if wsum else float("nan")
    rot_mean = rot_tot / wsum if wsum else float("nan")
    trans_mean = trans_tot / wsum if wsum else float("nan")

    # Report.
    used = [v[0] for v in solved.values() if v is not None]
    dup = {n for n in used if used.count(n) > 1}
    print(f"post-fit disagreement: {score:.2f} (deg + mm-equiv); "
          f"< ~2 is great. High snap residual or duplicate normals => a tag physically "
          f"moved / was mis-detected.")
    print(f"  breakdown: rotation {rot_mean:.2f} deg | translation "
          f"{1000.0 * trans_mean:.2f} mm\n")
    print("Paste this into FACES:\n")
    print("FACES = [")
    for i in sorted(solved):
        v = solved[i]
        if v is None:
            src = faces_by_id.get(i)
            up_t = tuple(src["up"]) if src else (0, 0, 1)
            nm = tuple(src["normal"]) if src else (1, 0, 0)
            print(f'    {{"id": {i}, "normal": {nm}, "up": {up_t}}},'
                  f'   # DISCONNECTED from ref -- get it co-visible & rerun')
            continue
        normal, up, res = v
        flags = []
        thresh = 10 if free_up else 20
        if res > thresh:
            why = "tilted/loose mount?" if free_up else "tag glued off-square?"
            kind = "normal tilt" if free_up else "snap residual"
            flags.append(f"{kind} {res:.0f} deg -- susp: {why}")
        if normal in dup:
            flags.append("DUPLICATE normal -- two tags on one face?")
        note = ("   # " + "; ".join(flags)) if flags else ""
        print(f'    {{"id": {i}, "normal": {normal}, "up": {up}}},{note}')
    print("]")


def _get_cam():
    try:
        mcw = MultiCameraWrapper(rgb=True, depth=False, ir=False,
                                 high_res_rgb=False, align=None, type="orbbec")
    except AssertionError:
        print("no Orbbec found, falling back to RealSense")
        mcw = MultiCameraWrapper(rgb=True, depth=False, ir=False,
                                 high_res_rgb=False, align=None, type="realsense")
    cam = mcw._all_cameras[0]
    intr = cam.calibration["intrinsics"]["rgb"]
    return cam, intr["cameraMatrix"], intr["distCoeffs"]


def run_identify():
    """Live view: print detected IDs so you can map ID -> face."""
    cam, _, _ = _get_cam()
    detector = make_detector()
    print("Hold each face to the camera; note which ID appears. Ctrl-C to stop.\n")
    seen = set()
    while True:
        rgb = cam.read_camera()["rgb"]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            now = sorted(int(i) for i in ids.flatten())
            for i in now:
                if i not in seen:
                    seen.add(i)
                    print(f"  detected new tag id={i}  (all seen: {sorted(seen)})")
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.imshow("identify (ESC to quit)", vis)
        else:
            cv2.imshow("identify (ESC to quit)", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def _put_lines(img, lines, org=(10, 28), color=(0, 255, 0), scale=0.7):
    """Draw text lines with a black outline for readability."""
    x, y = org
    for ln in lines:
        for c, th in ((( 0, 0, 0), 4), (color, 1)):   # outline, then fill
            cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, c, th,
                        cv2.LINE_AA)
        y += int(34 * scale)


def _annotate(vis, T, dbg, K, dist):
    """Draw markers, the peg-center axes, and the pose/orientation readout."""
    for d in dbg:
        cv2.aruco.drawDetectedMarkers(vis, [d["corners"]], np.array([[d["id"]]]))
    if T is not None:
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        cv2.drawFrameAxes(vis, K, dist, rvec, T[:3, 3], 0.5 * min(PEG_DIMS_M), 2)
        t = T[:3, 3]
        rpy = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
        ids = sorted(d["id"] for d in dbg)
        _put_lines(vis, [
            f"center [m]  x={t[0]:+.4f}  y={t[1]:+.4f}  z={t[2]:+.4f}",
            f"rpy [deg]   r={rpy[0]:+6.1f}  p={rpy[1]:+6.1f}  y={rpy[2]:+6.1f}",
            f"tags {ids}",
        ])
    else:
        _put_lines(vis, ["no configured tag visible"], color=(0, 0, 255))
    return vis


def run_estimate(show=True, video=None):
    cam, K, dist = _get_cam()
    detector = make_detector()
    T_peg_tag = {f["id"]: build_T_peg_tag(PEG_DIMS_M, f) for f in FACES}
    print(f"Configured {len(T_peg_tag)} faces: ids {sorted(T_peg_tag)}")
    print("Estimating T_cam_peg. Ctrl-C / ESC to stop.\n")
    if video:
        print(f"recording annotated video -> {video}")
    writer = None
    try:
        while True:
            rgb = cam.read_camera()["rgb"]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            T, dbg = estimate_peg_pose(gray, detector, K, dist, T_peg_tag, verbose=True)
            if T is not None:
                t = T[:3, 3]
                rpy = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
                ids = sorted(d["id"] for d in dbg)
                print(f"tags {ids}  center xyz=[{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m  "
                      f"rpy=[{rpy[0]:+6.1f} {rpy[1]:+6.1f} {rpy[2]:+6.1f}] deg")
            else:
                print("no configured tag visible")

            if show or video:
                vis = _annotate(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), T, dbg, K, dist)
                if video:
                    if writer is None:
                        h, w = vis.shape[:2]
                        writer = cv2.VideoWriter(
                            video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError(f"could not open VideoWriter for {video}")
                    writer.write(vis)
                if show:
                    cv2.imshow("peg pose (ESC to quit)", vis)
                    if cv2.waitKey(1) == 27:
                        break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
            print(f"\nsaved video: {video}")
        cv2.destroyAllWindows()


def _feed_tile(rgb, label, dbg, T, K, dist, seen, target_h=360):
    """One camera's feed as a fixed-height BGR tile: peg markers + center axes drawn,
    border green if a peg tag is visible else red, label + camera-frame pose readout.

    `dbg`/`T` come from ``estimate_peg_pose`` on THIS camera's intrinsics, so the pose
    is camera-relative (no extrinsics / fusion). Markers and axes are drawn on the
    full-res frame (where K is valid) before the tile is resized down.
    """
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for d in dbg:
        cv2.aruco.drawDetectedMarkers(vis, [d["corners"]], np.array([[d["id"]]]))
    if T is not None:
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        cv2.drawFrameAxes(vis, K, dist, rvec, T[:3, 3], 0.5 * min(PEG_DIMS_M), 2)

    h, w = vis.shape[:2]
    vis = cv2.resize(vis, (int(w * target_h / h), target_h))
    color = (0, 200, 0) if seen else (0, 0, 255)          # BGR: green / red
    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, target_h - 1), color, 10)

    lines = [f"{label}   {seen if seen else 'NO TAG'}"]
    if T is not None:
        t = T[:3, 3]
        rpy = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
        lines.append(f"xyz {t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f} m")
        lines.append(f"rpy {rpy[0]:+.0f} {rpy[1]:+.0f} {rpy[2]:+.0f} deg")
    _put_lines(vis, lines, org=(16, 30), color=color, scale=0.6)
    return vis


def run_coverage(camera_types=("orbbec", "realsense"), hz=30.0, video=None, show=True):
    """Open every connected camera across the given backends and print, at ~hz, which
    peg tags each one sees.

    No pose, no extrinsics -- just a live visibility check to see whether at least one
    camera keeps eyes on the peg through a motion. Logs a permanent line on every
    BLIND<->reacquired transition, plus a live one-line status. Labels are
    ``<backend-prefix><last-4-of-serial>`` (e.g. ``or:1234``, ``rs:5678``) so the
    Orbbec and the two RealSenses are told apart.
    """
    wrappers, cams = [], []       # cams: list of (label, camera, K, dist)
    for t in camera_types:
        try:
            mcw = MultiCameraWrapper(rgb=True, depth=False, ir=False,
                                     high_res_rgb=False, align=None, type=t)
        except Exception as e:
            print(f"[coverage] skipping {t}: {e}")
            continue
        wrappers.append(mcw)
        pre = {"orbbec": "or", "realsense": "rs"}.get(t, t[:2])
        for c in mcw._all_cameras:
            intr = c.calibration["intrinsics"]["rgb"]
            cams.append((f"{pre}:{c._serial_number[-4:]}", c,
                         intr["cameraMatrix"], intr["distCoeffs"]))

    if not cams:
        print("[coverage] no cameras found on any requested backend.")
        return

    detector = make_detector()
    peg_ids = {f["id"] for f in FACES}
    T_peg_tag = {f["id"]: build_T_peg_tag(PEG_DIMS_M, f) for f in FACES}
    print(f"{len(cams)} camera(s): {[lbl for lbl, *_ in cams]}")
    print(f"peg tags {sorted(peg_ids)}. Coverage ~{hz:.0f} Hz. Ctrl-C to stop.")
    if video:
        print(f"recording tiled coverage video -> {video}")
    print()

    dt = 1.0 / hz
    prev_blind = None
    t_prev = time.time()
    ema_hz = hz
    writer = None
    pending = []                 # (t, composite) buffered until fps is measured
    WARMUP_N, WARMUP_S = 45, 1.5

    # visibility / blind-episode accounting (time-weighted)
    total_time = vis_time = 0.0
    last_tick = None
    blind_episodes, longest_blind, blind_start = 0, 0.0, None

    def open_writer(fps, comp_shape):
        h, w = comp_shape[:2]
        vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"),
                             float(np.clip(fps, 1.0, 120.0)), (w, h))
        if not vw.isOpened():
            raise RuntimeError(f"could not open VideoWriter for {video}")
        return vw

    try:
        while True:
            t0 = time.time()
            per_cam = []      # (label, seen_ids, rgb, dbg, T, K, dist)
            for lbl, c, K, dist in cams:
                rgb = c.read_camera()["rgb"]
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                T, dbg = estimate_peg_pose(gray, detector, K, dist, T_peg_tag)
                seen = sorted(d["id"] for d in dbg)
                per_cam.append((lbl, seen, rgb, dbg, T, K, dist))

            any_visible = any(s for _, s, *_ in per_cam)
            now = time.time()

            # time-weighted visibility + blind-episode stats
            if last_tick is not None:
                d = now - last_tick
                total_time += d
                if any_visible:
                    vis_time += d
            last_tick = now
            blind = not any_visible
            if blind != prev_blind:
                stamp = time.strftime("%H:%M:%S")
                if blind:
                    blind_episodes += 1
                    blind_start = now
                    print(f"\n[{stamp}] >>> BLIND: no camera sees the peg")
                else:
                    if blind_start is not None:
                        longest_blind = max(longest_blind, now - blind_start)
                    who = "  ".join(f"{lbl}:{s}" for lbl, s, *_ in per_cam if s)
                    print(f"\n[{stamp}] <<< reacquired  {who}")
                prev_blind = blind

            ema_hz = 0.9 * ema_hz + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            pct = 100.0 * vis_time / total_time if total_time > 0 else 100.0
            status = "  ".join(f"{lbl}:{s if s else '--'}" for lbl, s, *_ in per_cam)
            flag = "VISIBLE" if any_visible else "BLIND  "
            print(f"\r[{ema_hz:4.1f}Hz] {flag} vis={pct:5.1f}% | {status}        ",
                  end="", flush=True)

            if video or show:
                tiles = [_feed_tile(rgb, lbl, dbg, T, K, dist, seen)
                         for lbl, seen, rgb, dbg, T, K, dist in per_cam]
                comp = np.hstack(tiles)
                banner = (0, 200, 0) if any_visible else (0, 0, 255)
                _put_lines(comp, [f"{flag.strip()}   vis={pct:.1f}%   {ema_hz:4.1f}Hz"],
                           org=(16, comp.shape[0] - 18), color=banner, scale=0.8)
                if video:
                    if writer is None:
                        # buffer until we've measured the true capture rate, then
                        # open the writer at that fps -> real-time playback.
                        pending.append((now, comp))
                        span = now - pending[0][0]
                        if len(pending) >= WARMUP_N or (span >= WARMUP_S and len(pending) >= 2):
                            fps = (len(pending) - 1) / max(span, 1e-6)
                            writer = open_writer(fps, comp.shape)
                            for _, cf in pending:
                                writer.write(cf)
                            pending.clear()
                            print(f"\n[coverage] recording at measured {fps:.1f} fps")
                    else:
                        writer.write(comp)
                if show:
                    cv2.imshow("coverage (ESC to quit)", comp)
                    if cv2.waitKey(1) == 27:
                        break

            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        # flush any warmup buffer if we stopped before the writer opened
        if video and writer is None and len(pending) >= 2:
            span = pending[-1][0] - pending[0][0]
            fps = (len(pending) - 1) / max(span, 1e-6)
            writer = open_writer(fps, pending[0][1].shape)
            for _, cf in pending:
                writer.write(cf)
        if writer is not None:
            writer.release()
            print(f"\nsaved video: {video}")
        if prev_blind and blind_start is not None:            # close an open blind run
            longest_blind = max(longest_blind, time.time() - blind_start)
        if total_time > 0:
            pct = 100.0 * vis_time / total_time
            print(f"\n=== coverage summary ===")
            print(f"  duration            {total_time:6.1f} s")
            print(f"  >=1 tag visible     {pct:5.1f} %  ({vis_time:.1f} s)")
            print(f"  fully blind         {100 - pct:5.1f} %  ({total_time - vis_time:.1f} s)")
            print(f"  blind episodes      {blind_episodes}  (longest {longest_blind:.2f} s)")
        cv2.destroyAllWindows()
        for mcw in wrappers:
            try:
                mcw.disable_cameras()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--identify", action="store_true", help="map tag IDs to faces")
    g.add_argument("--coverage", action="store_true",
                   help="print, at --hz, which peg tags each connected camera sees")
    g.add_argument("--calibrate-up", action="store_true",
                   help="solve each face's in-plane `up` from multi-tag frames")
    g.add_argument("--run", action="store_true", help="live peg-center pose")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--video", nargs="?", const="__auto__", default=None,
                    metavar="PATH",
                    help="record annotated --run video; PATH optional (auto-named .mp4)")
    ap.add_argument("--frames", type=int, default=300, help="frames for --calibrate-up")
    ap.add_argument("--ref-id", type=int, default=None,
                    help="anchor tag for --calibrate-up (the one face you're sure of)")
    ap.add_argument("--free-up", action="store_true",
                    help="--calibrate-up: recover a CONTINUOUS in-plane angle instead of "
                         "snapping to 90 deg (use when tags are glued off-square)")
    ap.add_argument("--camera-type", nargs="+", default=["orbbec", "realsense"],
                    choices=["realsense", "orbbec"],
                    help="backend(s) for --coverage (default: both -> all 3 cameras)")
    ap.add_argument("--hz", type=float, default=30.0, help="print rate for --coverage")
    args = ap.parse_args()
    coverage_video = args.video
    if coverage_video == "__auto__":
        coverage_video = f"coverage_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    if args.identify:
        run_identify()
    elif args.coverage:
        run_coverage(camera_types=args.camera_type, hz=args.hz,
                     video=coverage_video, show=not args.no_show)
    elif args.calibrate_up:
        run_calibrate_up(n_frames=args.frames, ref_id=args.ref_id, free_up=args.free_up)
    else:
        video = args.video
        if video == "__auto__":
            video = f"peg_pose_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        run_estimate(show=not args.no_show, video=video)


if __name__ == "__main__":
    main()
