"""Calibrate the peg-hole's AprilTags by anchoring them to the peg tracker.

WHY NOT MEASURE THE BLOCK: what the policy actually consumes is
``in_receptive = inv(T_base_hole) @ T_base_peg``. If a residual extrinsic error C
left-multiplies both terms it CANCELS -- but only if the hole was localized through the same
tag+extrinsic path as the peg. So we do not measure the block with calipers; we derive the
hole frame FROM the peg tracker, and the systematic bias cancels where it matters.

WHY THE GRASP IS INVOLVED: seated, the peg is almost entirely inside the block. Its four side
tags are centred on its faces, which puts them BELOW the block's top face -- only ~4mm of each
25mm tag clears it, so they cannot be detected. The peg tracker is blind exactly where we need
it. The grasp is rigid though, so we measure the peg in free space (all tags visible) and let
forward kinematics carry its pose into the hole.

PROCEDURE (3 phases, all prompted)
  A. Grip the peg, hold it up in clear view. Move it slowly between a few poses, pausing at
     each -- samples are taken only while the arm is still, and multiple arm configurations
     average down FK error rather than baking one pose's in. Yields ``T_ee_peg``.
  B. Without changing the grip, insert the peg until it bottoms out (free-drive is ideal --
     let the bore self-align it) and hold still. Then:
         T_base_peg  = FK(q) @ T_ee_peg
         T_base_hole = T_base_peg @ Trans(0, 0, -ASSEMBLED_OFFSET_Z)      [14.837 mm]
         T_hole_tag  = inv(T_base_hole) @ T_base_cam @ T_cam_tag          [per hole tag]
     It then verifies by re-localizing from the block's tags ALONE against this truth.
  C. Lift the peg back out, still gripped. The grasp is re-measured; if the peg shifted in the
     gripper during seating, that error went straight into the hole frame, so the run ABORTS
     without saving rather than writing a quietly-wrong calibration.

After this the peg is never needed again -- ``peg_fusion_viz --hole-tags`` recovers
``T_base_hole`` from the block's own tags alone.

Run in the `foundstereo` env, with the peg_fusion_viz worker STOPPED (this owns the cameras):

    # which IDs are on the block? (just look at it)
    python scripts/sim2real/perception/calibrate_hole_tags.py --identify

    # the calibration itself:
    python scripts/sim2real/perception/calibrate_hole_tags.py \\
        --robot_ip 192.168.1.10 --ids 6,7,8,9 --hole-tag-size 0.026
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import apriltag_peg_pose as apt
import peg_fusion_viz as pfv
import tag_body as tb

_TAG_OBJP = apt._TAG_OBJP


def _detect_tags(gray, detector, K, dist, tag_size):
    """[(id, [T_cam_tag branch0, branch1], area_px, ambiguity)] for every tag in the frame.

    Both IPPE branches are kept: a single planar tag is 2-fold ambiguous and the branch is
    resolved later with a geometric outwardness test (see ``_pick_branch``), which is far more
    reliable than the reprojection ratio alone at these tag sizes.
    """
    corners, ids, _ = detector.detectMarkers(gray)
    out = []
    if ids is None:
        return out
    objp = (_TAG_OBJP * float(tag_size)).astype(np.float32)
    for c, i in zip(corners, ids.flatten()):
        cc = np.asarray(c, dtype=np.float32).reshape(4, 2)
        try:
            _, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                objp, cc, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            continue
        if not len(rvecs):
            continue
        branches, e = [], []
        for k in range(len(rvecs)):
            branches.append(apt._compose_T(rvecs[k], tvecs[k]))
            e.append(float(np.ravel(errs[k])[0]))
        order = np.argsort(e)
        branches = [branches[k] for k in order]
        e = [e[k] for k in order]
        amb = (e[1] / e[0]) if len(e) >= 2 and e[0] > 1e-9 else None
        out.append((int(i), branches, apt._poly_area_px(cc), amb))
    return out


def _pick_branch(branches, T_base_cam, T_base_hole):
    """Choose the IPPE branch whose tag normal points OUTWARD from the hole origin.

    Tags are stuck on the outside of the block, so the tag's +Z (out of its face, toward the
    viewer) must have a positive component along the direction from the hole origin to the tag
    centre. That test is geometric and unambiguous; the reprojection ratio only breaks ties.

    Returns (T_hole_tag, outwardness, T_cam_tag) -- the raw camera-frame branch comes back too
    so the tag-size fit can rescale it analytically.
    """
    T_hole_base = tb.inv(T_base_hole)
    best = None
    for rank, T_cam_tag in enumerate(branches):
        T_hole_tag = T_hole_base @ T_base_cam @ T_cam_tag
        p = T_hole_tag[:3, 3]
        n = T_hole_tag[:3, 2]
        r = float(np.linalg.norm(p))
        outward = float(n @ (p / r)) if r > 1e-4 else -1.0
        # prefer outward; among equally outward branches prefer the lower-reprojection one
        score = (outward > 0.1, outward, -rank)
        if best is None or score > best[0]:
            best = (score, T_hole_tag, outward, T_cam_tag)
    return best[1], best[2], best[3]


def _fit_tag_size(tag_samples, T_base_hole, s0, radius=tb.HOLE_DIMS_M[0] / 2.0):
    """Measure the hole tags' true edge length from the data, as a check on --hole-tag-size.

    A wrong tag size is the easiest setup error to make and it is SILENT: IPPE translation
    scales exactly linearly with the assumed size, so the tags land off the block along each
    camera ray and the hole pose is biased with no obvious symptom.

    Constraint used: a tag glued on one of the four SIDE faces sits at a known radial distance
    from the hole's own +Z axis (half the block width, 34.925 mm). Along a camera ray,
    ``p(s) = s*(M_R @ t_cam) + M_t`` is affine in the assumed size, so ``|p_xy(s)| = radius``
    is a quadratic in s -- solved in closed form, no search and no re-detection.

    Deliberately uses only the tag's TRANSLATION, which is identical for both IPPE branches.
    An earlier version keyed off the tag normal, which needs the branch resolved, which needs
    the scale to already be right -- circular, and it silently returned wrong sizes.

    Only side-mounted tags satisfy this; a top-face tag shows up as scatter, which the caller
    reports as inconclusive rather than acting on.

    Returns (best_size_m, max_per_tag_disagreement_mm, mean_radial_error_mm_at_s0).
    """
    T_hb = tb.inv(T_base_hole)
    ests, resid0 = [], []
    for i, samples in tag_samples.items():
        roots, rad0 = [], []
        for _, name, T_bc, branches, area, amb in samples:
            M = T_hb @ T_bc                                        # hole <- cam
            A = (M[:3, :3] @ branches[0][:3, 3])[:2]               # both branches share t_cam
            b = M[:3, 3][:2]
            rad0.append(abs(float(np.linalg.norm(A + b)) - radius) * 1000.0)
            aa = float(A @ A)
            if aa < 1e-12:
                continue
            disc = float((A @ b) ** 2 - aa * (float(b @ b) - radius ** 2))
            if disc < 0:                                           # ray misses the block
                continue
            r = np.sqrt(disc)
            cand = [(-float(A @ b) - r) / aa, (-float(A @ b) + r) / aa]
            cand = [c for c in cand if c > 1e-6]
            if cand:
                roots.append(min(cand) * s0)     # near crossing = the face the camera can see
        if roots:
            ests.append(float(np.median(roots)))
            resid0.append(float(np.mean(rad0)))
    if not ests:
        return None, None, None
    s_best = float(np.median(ests))
    return (s_best,
            float(np.max([abs(e - s_best) for e in ests]) * 1000.0),
            float(np.mean(resid0)) if resid0 else None)


def _yaw_branch(T_base_hole, nominal_R, verbose=True):
    """Resolve the square peg's 4-fold seating ambiguity about the hole's own +Z.

    A square peg in a square hole seats in any of 4 yaws, so the peg only pins the hole frame
    up to k*90 deg. Pick the branch closest to the nominal hole orientation (identity =
    axis-aligned with the robot base, matching eval_real_robot.DEFAULT_HOLE_POSE).
    """
    best = None
    for k in range(4):
        Rz = np.eye(4)
        Rz[:3, :3] = R.from_euler("z", 90.0 * k, degrees=True).as_matrix()
        cand = T_base_hole @ Rz
        d = float((R.from_matrix(cand[:3, :3]) * R.from_matrix(nominal_R).inv()).magnitude())
        if best is None or d < best[0]:
            best = (d, k, cand)
    if verbose:
        print(f"[yaw] seating is 4-fold ambiguous; picked k={best[1]} (+{90 * best[1]}deg about "
              f"hole +Z), {np.degrees(best[0]):.1f}deg from nominal.")
    return best[2], best[1]


def _open_rig(args):
    """Cameras + their base-frame extrinsic providers (static, or FK for the wrist)."""
    res_by_role = dict(pfv.DEFAULT_RES)
    T_base_cam = {"front": pfv._load_static_extrinsic(pfv.SER_FRONT),
                  "side": pfv._load_static_extrinsic(pfv.SER_SIDE)}
    use_wrist = not args.no_wrist
    rtde_r = fk = T_wrist3_cam = None
    if use_wrist:
        try:
            import rtde_receive
            from diffusion_policy.real_world.ur5e_kinematics import forward_kinematics_calibrated
            rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
            fk = forward_kinematics_calibrated
            T_wrist3_cam, wp = pfv._load_wrist_offset()
            print(f"[rig] wrist offset from {wp}")
        except Exception as e:
            print(f"[rig] no robot/wrist ({e}); continuing with front+side only")
            use_wrist = False
    cams = pfv._open_cameras(use_wrist, res_by_role, color_fps=30)
    if not cams:
        raise SystemExit("no configured cameras connected.")
    print(f"[rig] cameras: {sorted(cams)}")

    def get_joints():
        return None if rtde_r is None else np.asarray(rtde_r.getActualQ(), dtype=np.float64)

    def base_from_cam(name, q=None):
        """T_base_cam for this camera right now (the arm must be still during calibration)."""
        if name == "wrist":
            if rtde_r is None or T_wrist3_cam is None:
                return None
            return fk(get_joints() if q is None else q)[0] @ T_wrist3_cam
        return T_base_cam.get(name)

    return cams, base_from_cam, get_joints, fk


def _quiet(hist, window=0.4, tol=0.002):
    """Was the arm still over the last ``window`` seconds? FK only carries the grasped peg's
    pose faithfully when the arm is not moving relative to the camera exposures."""
    if not hist:
        return False
    now = hist[-1][0]
    near = [q for t, q in hist if now - t <= window and q is not None]
    if len(near) < 4:
        return False
    a = np.asarray(near, dtype=np.float64)
    return float(np.max(a.max(0) - a.min(0))) < tol


def _measure_grasp(cams, base_from_cam, get_joints, fk, args, peg_det, T_peg_tag, n, label):
    """T_ee_peg -- the peg's pose in the wrist_3 frame while it is GRASPED and in free space.

    This is the trick that makes the whole calibration possible. Fully seated, the peg's four
    side tags are buried in the block (only ~4mm of each 25mm tag clears the top face), so the
    peg tracker cannot see it where we need it. But the grasp is rigid: measure T_ee_peg out in
    the open where every tag is visible, and FK carries the peg's pose into the hole for us.

    Samples only while the arm is STILL, and across whatever poses you move it through --
    different arm configurations average down FK error instead of baking one pose's in.
    """
    from collections import deque
    hist = deque(maxlen=64)
    samples, confs, npose = [], [], 0
    last_q = None
    print(f"  [{label}] hold the peg in the gripper, in clear view. Move it slowly between "
          f"poses and PAUSE; samples are taken only while the arm is still.")
    while len(samples) < n:
        q = get_joints()
        hist.append((time.time(), q))
        if q is None or not _quiet(hist):
            time.sleep(0.01)
            continue
        ests, w = {}, {}
        for name, (cam, K, dist) in cams.items():
            gray = cv2.cvtColor(cam.read_camera()["rgb"], cv2.COLOR_RGB2GRAY)
            T_bc = base_from_cam(name, q)
            if T_bc is None:
                continue
            T_cam_peg, dbg, info = apt.estimate_peg_pose_ex(gray, peg_det[name], K, dist, T_peg_tag)
            if T_cam_peg is not None and len(dbg) >= args.min_peg_tags:
                ests[name] = T_bc @ T_cam_peg
                w[name] = info["conf"]
        if not ests:
            continue
        T_base_peg, _, _ = pfv._fuse(ests, w, mode="median")
        samples.append(tb.inv(fk(q)[0]) @ T_base_peg)
        confs.append(sum(w.values()))
        if last_q is None or float(np.max(np.abs(q - last_q))) > 0.05:
            npose += 1
            last_q = q
        print(f"\r  [{label}] {len(samples)}/{n} samples, ~{npose} arm poses   ", end="", flush=True)
    print()
    T = tb.robust_mean(samples, confs)
    p_mm, r_deg = tb.spread(samples, T)
    return T, p_mm, r_deg, npose


def run_identify(args):
    """Print every tag ID visible in the hole's dictionary, per camera, with its size on screen."""
    cams, _ = _open_rig(args)
    dict_id = getattr(cv2.aruco, args.hole_dict)
    det = apt.make_detector(dict_id)
    peg_ids = {f["id"] for f in apt.FACES}
    same_family = args.hole_dict == "DICT_APRILTAG_16h5"
    print(f"\nWatching {args.hole_dict}. Peg IDs (excluded from suggestions): {sorted(peg_ids)}"
          if same_family else f"\nWatching {args.hole_dict} (peg uses a different family).")
    print("Point the cameras at the peg-hole. Ctrl-C to stop.\n")
    seen = defaultdict(set)
    try:
        while True:
            for name, (cam, K, dist) in cams.items():
                rgb = cam.read_camera()["rgb"]
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                for i, _, area, _ in _detect_tags(gray, det, K, dist, args.hole_tag_size):
                    if i not in seen[name]:
                        seen[name].add(i)
                        flag = "  <-- this is a PEG id" if same_family and i in peg_ids else ""
                        print(f"  {name}: new id={i}  ~{np.sqrt(area):.0f}px edge{flag}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        allseen = sorted(set().union(*seen.values())) if seen else []
        cand = [i for i in allseen if not (same_family and i in peg_ids)]
        print(f"\nall ids seen: {allseen}")
        print(f"hole-tag candidates: {cand}")
        print(f"  -> rerun without --identify (add --ids {','.join(map(str, cand))} to pin them)")
        _close(cams)


def _close(cams):
    for (cam, _, _) in cams.values():
        try:
            cam.disable_camera()
        except Exception:
            pass


def _collect(cams, base_from_cam, args, hole_det, T_peg_tag, peg_det, n_frames, label,
             T_ee_peg=None, get_joints=None, fk=None):
    """Sweep n_frames: peg pose + every raw hole-tag observation, per frame.

    Nothing moves during collection, so this loop is deliberately SYNCHRONOUS -- no async
    workers, no cross-camera time skew to correct for.

    With ``T_ee_peg`` the peg pose comes from FK on the grasped peg (the seated peg's own tags
    are buried in the block); otherwise it is estimated from the peg's tags directly.
    """
    from collections import deque
    peg_samples = []                              # [T_base_peg]
    tag_samples = defaultdict(list)               # id -> [(frame_idx, cam, T_bc, branches, area, amb)]
    skipped = 0
    hist = deque(maxlen=64)
    f = 0
    while f < n_frames:
        f += 1
        q = get_joints() if get_joints is not None else None
        if T_ee_peg is not None:
            hist.append((time.time(), q))
            if q is None or not _quiet(hist):
                skipped += 1
                f -= 1                              # a moving arm costs no budget, just wait
                time.sleep(0.01)
                continue
        ests, weights, holes = {}, {}, {}
        for name, (cam, K, dist) in cams.items():
            rgb = cam.read_camera()["rgb"]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            T_bc = base_from_cam(name, q)
            if T_bc is None:
                continue
            if T_ee_peg is None:
                T_cam_peg, dbg, info = apt.estimate_peg_pose_ex(
                    gray, peg_det[name], K, dist, T_peg_tag)
                if T_cam_peg is not None and len(dbg) >= args.min_peg_tags:
                    ests[name] = T_bc @ T_cam_peg
                    weights[name] = info["conf"]
            holes[name] = (T_bc, _detect_tags(gray, hole_det[name], K, dist, args.hole_tag_size))
        if T_ee_peg is not None:
            ests = {"fk": fk(q)[0] @ T_ee_peg}      # grasp carried in by forward kinematics
            weights = {"fk": 1.0}
        if not ests:
            skipped += 1
        else:
            T_base_peg, _, _ = pfv._fuse(ests, weights, mode="median")
            peg_samples.append(T_base_peg)
            idx = len(peg_samples) - 1
            for name, (T_bc, tags) in holes.items():
                for i, branches, area, amb in tags:
                    if args.ids and i not in args.ids:
                        continue
                    tag_samples[i].append((idx, name, T_bc, branches, area, amb))
        if f % 10 == 0:
            print(f"\r  [{label}] frame {f + 1}/{n_frames}  peg_cams={len(ests)}  "
                  f"tags={sorted(tag_samples)}  skipped={skipped}   ", end="", flush=True)
    print()
    return peg_samples, tag_samples, skipped


def run_calibrate(args):
    cams, base_from_cam, get_joints, fk = _open_rig(args)
    peg_det = {n: apt.make_detector() for n in cams}
    dict_id = getattr(cv2.aruco, args.hole_dict)
    hole_det = ({n: peg_det[n] for n in cams} if args.hole_dict == "DICT_APRILTAG_16h5"
                else {n: apt.make_detector(dict_id) for n in cams})
    T_peg_tag = {f["id"]: apt.build_T_peg_tag(apt.PEG_DIMS_M, f) for f in apt.FACES}
    peg_ids = set(T_peg_tag)
    if args.hole_dict == "DICT_APRILTAG_16h5" and args.ids and (set(args.ids) & peg_ids):
        raise SystemExit(f"--ids {sorted(set(args.ids) & peg_ids)} collide with peg tag IDs "
                         f"{sorted(peg_ids)} in the same dictionary.")
    if args.grasp_carry and get_joints is None:
        raise SystemExit("--grasp-carry needs the robot (FK). Fix the RTDE connection, or pass "
                         "--no-grasp-carry if the seated peg's own tags are visible.")

    T_ee_peg = None
    try:
        if args.grasp_carry:
            # ---- phase A: measure the grasp out in the open, where every peg tag is visible --
            print("\n=== PHASE A: grasp transform ===")
            print(">>> Close the gripper firmly on the peg and hold it up in clear view.")
            input(">>> Press ENTER when ready...")
            T_ee_peg, gp_mm, gr_deg, nposes = _measure_grasp(
                cams, base_from_cam, get_joints, fk, args, peg_det, T_peg_tag,
                args.grasp_frames, "grasp")
            print(f"  T_ee_peg over {args.grasp_frames} samples / ~{nposes} arm poses: "
                  f"spread {gp_mm:.2f}mm / {gr_deg:.2f}deg")
            print(f"  peg in wrist frame: {np.round(T_ee_peg[:3, 3] * 1000, 2)} mm")
            if gp_mm > 3.0:
                print("  [WARN] the grasp transform is not settling. The peg may be slipping, or "
                      "the arm was moving while sampling.")
            print("\n=== PHASE B: seat it ===")
            print(">>> WITHOUT changing the grip, insert the peg into the hole until it bottoms")
            print(">>> out. Free-drive is ideal -- let the bore self-align it. Then let go of the")
            print(">>> arm and keep it still.")
            input(">>> Press ENTER when the peg is seated and the arm is still...")
        else:
            print("\n>>> Seat the peg FULLY in the hole, release the gripper, move the arm clear.")
            print(f">>> Collecting {args.frames} frames in {args.settle:.0f}s...\n")
            time.sleep(args.settle)

        peg_samples, tag_samples, skipped = _collect(
            cams, base_from_cam, args, hole_det, T_peg_tag, peg_det, args.frames, "collect",
            T_ee_peg=T_ee_peg, get_joints=get_joints, fk=fk)
        if len(peg_samples) < max(5, args.frames // 10):
            raise SystemExit(
                f"only {len(peg_samples)}/{args.frames} frames had a usable peg pose. "
                + ("Was the arm still, and is the robot connected?" if args.grasp_carry else
                   "The seated peg's side tags are BURIED in the block (only ~4mm of each 25mm "
                   "tag clears the top face) -- use --grasp-carry instead of --no-grasp-carry."))
        if args.hole_dict == "DICT_APRILTAG_16h5":
            for i in list(tag_samples):
                if i in peg_ids:
                    del tag_samples[i]
        if not tag_samples:
            raise SystemExit("no hole tags detected. Run --identify first, check --hole-dict / "
                             "--hole-tag-size, and make sure the block's tags are in view.")

        # ---- hole frame, pinned to the seated peg by the sim's assembled offset -------------
        T_base_peg = tb.robust_mean(peg_samples)
        pos_sp, rot_sp = tb.spread(peg_samples, T_base_peg)
        print(f"\n[peg]  {len(peg_samples)} usable frames  |  static spread "
              f"{pos_sp:.2f}mm / {rot_sp:.2f}deg  (this is your tracker's noise floor)")
        T_base_hole = T_base_peg @ tb.inv(tb.T_hole_peg_seated())
        nominal_R = R.from_rotvec(np.array([0.0, 0.0, np.radians(args.nominal_yaw_deg)])).as_matrix()
        T_base_hole, k_yaw = _yaw_branch(T_base_hole, nominal_R)

        # ---- is --hole-tag-size actually right? (silent-failure guard) ----------------------
        s_fit, s_scatter_mm, resid_mm = _fit_tag_size(tag_samples, T_base_hole, args.hole_tag_size)
        if s_fit is not None:
            err_mm = abs(s_fit - args.hole_tag_size) * 1000.0
            print(f"\n[tag-size] you passed {args.hole_tag_size * 1000:.1f}mm; the block geometry "
                  f"implies {s_fit * 1000:.1f}mm (tags disagree by <={s_scatter_mm:.1f}mm). "
                  f"Tags sit {resid_mm:.1f}mm off the block faces at your size.")
            if err_mm > 1.5 and s_scatter_mm < 3.0:
                print(f"[tag-size][WARN] that is a {err_mm:.1f}mm mismatch and the tags AGREE on "
                      f"it -- your --hole-tag-size is probably wrong, which biases the hole pose "
                      f"silently. Measure the black square with calipers and rerun with\n"
                      f"        --hole-tag-size {s_fit:.4f}")
            elif s_scatter_mm >= 3.0:
                print("[tag-size] tags disagree too much for this check to be conclusive "
                      "(a tag may not be flat on a face, or the block is not the CAD block).")

        # ---- each tag's constant pose in that frame -----------------------------------------
        tags, report = {}, []
        for i in sorted(tag_samples):
            obs, w, outs, cams_seen = [], [], [], set()
            for _, name, T_bc, branches, area, amb in tag_samples[i]:
                T_hole_tag, outward, _ = _pick_branch(branches, T_bc, T_base_hole)
                obs.append(T_hole_tag)
                w.append(area)               # bigger on-screen tag -> better corner geometry
                outs.append(outward)
                cams_seen.add(name)
            if len(obs) < args.min_tag_obs:
                print(f"  [tag {i}] only {len(obs)} observations (< --min-tag-obs "
                      f"{args.min_tag_obs}) -- DROPPED")
                continue
            T = tb.robust_mean(obs, w)
            p_mm, r_deg = tb.spread(obs, T)
            tags[i] = T
            report.append({"id": i, "n": len(obs), "cams": sorted(cams_seen),
                           "spread_mm": p_mm, "spread_deg": r_deg,
                           "outward": float(np.mean(outs))})
            warn = ""
            if np.mean(outs) < 0.3:
                warn = "  <-- WEAK outward test; branch may be flipped, check the pose in --verify"
            print(f"  [tag {i}] n={len(obs):4d} cams={sorted(cams_seen)}  "
                  f"pos={T[:3, 3] * 1000} mm  spread={p_mm:.2f}mm/{r_deg:.2f}deg{warn}")

        if not tags:
            raise SystemExit("every candidate tag was dropped; nothing to save.")

        body = tb.TagBody(name="peg_hole", tags=tags, tag_size_m=args.hole_tag_size,
                          aruco_dict=args.hole_dict, dims_m=tb.HOLE_DIMS_M,
                          calib={"method": "peg_anchored_assembled_offset",
                                 "assembled_offset_z": tb.ASSEMBLED_OFFSET_Z,
                                 "yaw_branch_k": int(k_yaw),
                                 "nominal_yaw_deg": float(args.nominal_yaw_deg),
                                 "frames": int(len(peg_samples)),
                                 "peg_spread_mm": float(pos_sp),
                                 "peg_spread_deg": float(rot_sp),
                                 "T_base_hole_at_calib": T_base_hole.tolist(),
                                 "tags": report,
                                 "stamp": time.time()})

        # ---- verify: re-localize from the tags ALONE and compare to the peg-derived truth ----
        # Runs while the peg is STILL SEATED, so the peg-derived hole frame is a live ground
        # truth. This is the number that actually tells you the calibration is good.
        if args.verify_frames > 0:
            print(f"\n[verify] re-localizing from the block's tags alone ({args.verify_frames} "
                  f"frames), peg still seated...")
            vp, vt, _ = _collect(cams, base_from_cam, args, hole_det, T_peg_tag, peg_det,
                                 args.verify_frames, "verify",
                                 T_ee_peg=T_ee_peg, get_joints=get_joints, fk=fk)
            est = []
            vdet = {n: body.make_detector() for n in cams}
            for _ in range(args.verify_frames // 4 + 1):
                ests, weights = {}, {}
                for name, (cam, K, dist) in cams.items():
                    rgb = cam.read_camera()["rgb"]
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    T_bc = base_from_cam(name)
                    T_cam_hole, dbg, info = body.estimate(gray, vdet[name], K, dist)
                    if T_cam_hole is not None and T_bc is not None:
                        ests[name] = T_bc @ T_cam_hole
                        weights[name] = info["conf"]
                if ests:
                    est.append(pfv._fuse(ests, weights, mode="median")[0])
            if est and vp:
                truth = tb.robust_mean(vp) @ tb.inv(tb.T_hole_peg_seated())
                Rz = np.eye(4)
                Rz[:3, :3] = R.from_euler("z", 90.0 * k_yaw, degrees=True).as_matrix()
                truth = truth @ Rz
                T_est = tb.robust_mean(est)
                dp = np.linalg.norm(T_est[:3, 3] - truth[:3, 3]) * 1000.0
                dr = np.degrees((R.from_matrix(T_est[:3, :3])
                                 * R.from_matrix(truth[:3, :3]).inv()).magnitude())
                jit_mm, jit_deg = tb.spread(est, T_est)
                ok = (dp < tb.HOLE_SUCCESS_POS_M * 1000.0
                      and np.radians(dr) < tb.HOLE_SUCCESS_ORI_RAD)
                print(f"[verify] tags-only vs peg-derived:  {dp:.2f} mm  {dr:.2f} deg   "
                      f"(sim success gate: {tb.HOLE_SUCCESS_POS_M * 1000:.1f} mm / "
                      f"{np.degrees(tb.HOLE_SUCCESS_ORI_RAD):.1f} deg)  -> "
                      f"{'OK' if ok else 'TOO LOOSE'}")
                print(f"[verify] tags-only frame-to-frame jitter: {jit_mm:.2f} mm / {jit_deg:.2f} deg")
                body.calib["verify"] = {"pos_err_mm": float(dp), "rot_err_deg": float(dr),
                                        "jitter_mm": float(jit_mm), "jitter_deg": float(jit_deg),
                                        "passed": bool(ok)}
                if not ok:
                    print("[verify][WARN] the tags-only estimate disagrees with the seated peg by "
                          "more than the sim's success threshold. Likely causes: a flipped "
                          "single-tag branch (see WEAK outward warnings), a wrong "
                          "--hole-tag-size, or a tag that moved during collection.")
            else:
                print("[verify][WARN] could not verify (no tags-only or no peg pose).")

        # ---- phase C: did the grasp slip while seating? if it did, everything above is wrong --
        # Must come AFTER verify -- it takes the peg back out of the hole.
        if args.grasp_carry:
            print("\n=== PHASE C: grasp re-check ===")
            print(">>> Lift the peg straight back out, still gripped, and hold it in clear view.")
            input(">>> Press ENTER when ready...")
            T_ee_peg2, sp_mm, sp_deg, _ = _measure_grasp(
                cams, base_from_cam, get_joints, fk, args, peg_det, T_peg_tag,
                max(20, args.grasp_frames // 3), "recheck")
            d = tb.inv(T_ee_peg) @ T_ee_peg2
            slip_mm = float(np.linalg.norm(d[:3, 3]) * 1000.0)
            slip_deg = float(np.degrees(np.linalg.norm(R.from_matrix(d[:3, :3]).as_rotvec())))
            body.calib["grasp"] = {"T_ee_peg": T_ee_peg.tolist(), "slip_mm": slip_mm,
                                   "slip_deg": slip_deg, "spread_mm": float(gp_mm)}
            print(f"  grasp moved {slip_mm:.2f} mm / {slip_deg:.2f} deg during seating "
                  f"(re-measure spread {sp_mm:.2f}mm)")
            if slip_mm > args.max_slip_mm:
                raise SystemExit(
                    f"GRASP SLIPPED by {slip_mm:.2f} mm (> --max-slip-mm {args.max_slip_mm}). The "
                    f"peg moved in the gripper while you seated it, so the hole frame derived from "
                    f"FK is wrong by roughly that much. NOTHING WAS SAVED -- grip harder (or nearer "
                    f"the peg's centre), and rerun.")
            print(f"  grasp held (<= --max-slip-mm {args.max_slip_mm}) -- the FK-carried hole "
                  f"frame is trustworthy.")

        out = args.out or tb.DEFAULT_HOLE_TAGS
        body.save(out)
        print(f"\nsaved {body} -> {out}")
        print(f"\nHole pose at calibration time (base frame):\n"
              f"{np.array2string(T_base_hole, precision=4, suppress_small=True)}")
        print("\nNext: the fusion worker will now publish T_base_hole from these tags --\n"
              f"    python {os.path.join('scripts', 'sim2real', 'perception', 'peg_fusion_viz.py')} "
              f"--publish --headless --robot_ip {args.robot_ip}\n"
              "    python eval_real_robot.py -i <policy.pt> -o <out> --robot_ip "
              f"{args.robot_ip}   # --hole_from_tags is on by default")
    finally:
        _close(cams)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot_ip", default="192.168.1.10")
    ap.add_argument("--no-wrist", action="store_true",
                    help="skip the wrist camera / robot FK (front+side only)")
    ap.add_argument("--identify", action="store_true",
                    help="just print the tag IDs visible on the block, then exit")
    ap.add_argument("--grasp-carry", dest="grasp_carry", action="store_true", default=True,
                    help="Carry the peg's pose into the hole through the GRASP + forward "
                         "kinematics (default, and required with the current peg tagging): fully "
                         "seated, only ~4mm of each 25mm side tag clears the block, so the peg "
                         "tracker cannot see the peg where it matters. Measure the grasp in free "
                         "space, then let FK carry it in.")
    ap.add_argument("--no-grasp-carry", dest="grasp_carry", action="store_false",
                    help="Read the SEATED peg's own tags instead. Only works if >=2 peg tags "
                         "clear the block when seated (they do not, with tags centred on the "
                         "peg's faces).")
    ap.add_argument("--grasp-frames", type=int, default=60,
                    help="still-arm samples for the grasp transform (default 60)")
    ap.add_argument("--max-slip-mm", type=float, default=1.0,
                    help="abort if the peg moved this far in the gripper during seating "
                         "(default 1.0mm) -- that error goes straight into the hole frame")
    ap.add_argument("--frames", type=int, default=150, help="frames to collect (default 150)")
    ap.add_argument("--verify-frames", type=int, default=40,
                    help="frames for the tags-only verification pass (0 to skip)")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds to wait before collecting, so you can step clear")
    ap.add_argument("--hole-dict", default="DICT_APRILTAG_16h5",
                    help="ArUco/AprilTag family on the peg-hole. Use a DIFFERENT family from the "
                         "peg (e.g. DICT_APRILTAG_36h11) if you are printing new tags -- it makes "
                         "peg/hole ID collisions structurally impossible and decodes far more "
                         "reliably. Default matches the peg's family.")
    ap.add_argument("--hole-tag-size", type=float, default=0.026,
                    help="black-edge length of the HOLE tags in metres (default 26mm, the size of "
                         "the currently printed tags; the block's faces are 70x40mm so go as big "
                         "as fits -- precision scales with it)")
    ap.add_argument("--ids", default="", type=lambda s: [int(x) for x in s.replace(" ", "").split(",") if x],
                    help="comma-separated hole tag IDs (default: every non-peg ID seen)")
    ap.add_argument("--min-peg-tags", type=int, default=2,
                    help="require >= this many peg tags per camera (2+ kills the planar flip)")
    ap.add_argument("--min-tag-obs", type=int, default=20,
                    help="drop a hole tag with fewer than this many observations")
    ap.add_argument("--nominal-yaw-deg", type=float, default=0.0,
                    help="nominal hole yaw about base +Z used to resolve the square peg's 4-fold "
                         "seating ambiguity (0 = axis-aligned, matching DEFAULT_HOLE_POSE)")
    ap.add_argument("--out", default=None, help=f"output JSON (default {tb.DEFAULT_HOLE_TAGS})")
    args = ap.parse_args()
    if args.identify:
        run_identify(args)
    else:
        run_calibrate(args)


if __name__ == "__main__":
    main()
