"""
Camera-drift check: reset the robot to a saved pose, re-capture the Orbbec cloud,
and compare it against the one stored at capture time.

Workflow (the inverse of `orbbec_live_crop.py --save_dir`):
  1. Load a saved dataset dir: `cloud_crop.ply` + `robot_state.json` (the latter
     holds the UR5e joint angles AND the crop/plane parameters used).
  2. moveJ the robot back to those exact joint angles, so the arm occupies the
     same volume it did when the reference cloud was taken.
  3. Capture one Orbbec frame and run the IDENTICAL crop + floor removal (reusing
     `crop_cloud` from orbbec_live_crop, parameterized from the saved JSON).
  4. Compare the new cloud against the reference:
       * direct nearest-neighbor distances (no alignment) -- how far apart the two
         clouds sit as-is;
       * rigid ICP (new -> reference) -- the best-fit transform; its translation /
         rotation magnitude is what the camera appears to have MOVED by.
     A static scene + unmoved camera => tiny direct distances and a near-identity
     ICP transform. A moved camera => ICP recovers a non-identity rigid transform
     with still-low residual. A *changed scene* => high residual even after ICP.

ASSUMPTION: the physical scene (objects in the crop) is unchanged between the two
captures -- otherwise the difference reflects the scene, not the camera. Keep the
same objects in place; the robot is reset for exactly this reason.

Runs in the `foundstereo` conda env (pyorbbecsdk2 + open3d + ur_rtde).

Usage:
  (foundstereo)$ python orbbec_compare_capture.py --dataset_dir tmp/capture0
  # skip the robot move (arm already in place), just re-capture + compare:
  (foundstereo)$ python orbbec_compare_capture.py --dataset_dir tmp/capture0 --no_move
  # non-interactive (no confirmation prompt before the robot moves):
  (foundstereo)$ python orbbec_compare_capture.py --dataset_dir tmp/capture0 --yes --vis
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from orbbec_camera import (
    open_camera, warmup_autoexposure, capture_aligned,
    make_pointcloud_filter, color_intrinsics, orbbec_pointcloud,
)
from orbbec_live_crop import (
    plane_frame, rotate_in_plane, project_pixel, crop_cloud, parse_xyz,
)

# crop params that orbbec_live_crop's save mode does NOT write to robot_state.json;
# fall back to its defaults (overridable on the CLI).
CROP_DEFAULTS = dict(plane_slope_m=0.001, seed_win=12, voxel_m=0.0)


# ----------------------------------------------------------------------------
# dataset loading -> crop config
# ----------------------------------------------------------------------------
def load_dataset(dataset_dir):
    """Return (reference o3d cloud, state dict) from a saved capture directory."""
    state_path = os.path.join(dataset_dir, "robot_state.json")
    with open(state_path) as f:
        state = json.load(f)
    ply_path = os.path.join(dataset_dir, state.get("cloud_file", "cloud_crop.ply"))
    ref = o3d.io.read_point_cloud(ply_path)
    if ref.is_empty():
        raise RuntimeError(f"reference cloud {ply_path} is empty / unreadable")
    return ref, state


def crop_config_from_state(state, K, overrides):
    """Rebuild everything crop_cloud() needs from the saved JSON: the plane frame,
    up_sign, center, seed pixels, and an args-like namespace of crop knobs."""
    p0 = np.array(state["p0_mm"], float)
    p1 = np.array(state["p1_mm"], float)
    p2 = np.array(state["p2_mm"], float)
    x_axis, y_axis, z_axis = plane_frame(p0, p1, p2)
    if state.get("rotate_deg"):
        x_axis, y_axis = rotate_in_plane(x_axis, y_axis, z_axis, state["rotate_deg"])
    up_sign = -1.0 if state.get("flip_normal") else 1.0
    center_m = p1 / 1000.0
    seed_px0 = [project_pixel(p0, K), project_pixel(p1, K), project_pixel(p2, K)]

    xb = state["x_bounds_m"]
    yb = state["y_bounds_m"]
    cfg = SimpleNamespace(
        x_min=xb[0], x_max=xb[1], y_min=yb[0], y_max=yb[1],
        slope_floor=bool(state.get("slope_floor", False)),
        floor_height_m=state.get("floor_height_m", 0.02),
        plane_slope_m=overrides.get("plane_slope_m", CROP_DEFAULTS["plane_slope_m"]),
        seed_win=overrides.get("seed_win", CROP_DEFAULTS["seed_win"]),
        voxel_m=overrides.get("voxel_m", CROP_DEFAULTS["voxel_m"]),
    )
    return cfg, (x_axis, y_axis, z_axis, up_sign, center_m, seed_px0)


# ----------------------------------------------------------------------------
# robot reset
# ----------------------------------------------------------------------------
def reset_robot(robot_ip, joints_rad, speed, accel, assume_yes):
    """moveJ the UR5e to `joints_rad`. Prompts for confirmation unless assume_yes.
    Returns True if the move ran, False if skipped/failed."""
    try:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
    except Exception as e:
        print(f"[compare] ur_rtde not importable ({e}); cannot move robot")
        return False

    q_target = np.asarray(joints_rad, dtype=float)
    try:
        rtde_r = RTDEReceiveInterface(robot_ip)
        q_now = np.array(rtde_r.getActualQ(), dtype=float)
        rtde_r.disconnect()
    except Exception as e:
        print(f"[compare] could not read current joints at {robot_ip} ({e})")
        q_now = None

    print("[compare] target joints (deg):", np.round(np.rad2deg(q_target), 2).tolist())
    if q_now is not None:
        dmax = np.max(np.abs(np.rad2deg(q_target - q_now)))
        print("[compare] current joints (deg):", np.round(np.rad2deg(q_now), 2).tolist())
        print(f"[compare] max per-joint move: {dmax:.2f} deg")

    if not assume_yes:
        ans = input("[compare] move the robot to the saved pose? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("[compare] robot move declined; continuing without reset")
            return False

    try:
        rtde_c = RTDEControlInterface(robot_ip)
        print(f"[compare] moveJ (speed={speed}, accel={accel})...")
        ok = rtde_c.moveJ(q_target.tolist(), speed, accel)   # blocks until reached
        rtde_c.stopScript()
        rtde_c.disconnect()
    except Exception as e:
        print(f"[compare] moveJ failed ({e})")
        return False
    if not ok:
        print("[compare] moveJ returned False (motion not completed)")
        return False
    print("[compare] robot at saved pose")
    return True


# ----------------------------------------------------------------------------
# cloud comparison
# ----------------------------------------------------------------------------
def nn_distance_stats(src, dst):
    """Per-point nearest-neighbor distance from src to dst (meters). Returns dict
    of mm stats."""
    if src.is_empty() or dst.is_empty():
        return None
    d = np.asarray(src.compute_point_cloud_distance(dst)) * 1000.0   # m -> mm
    return {
        "mean_mm": float(d.mean()), "median_mm": float(np.median(d)),
        "rms_mm": float(np.sqrt((d ** 2).mean())),
        "p95_mm": float(np.percentile(d, 95)), "max_mm": float(d.max()),
        "n": int(d.size),
    }


def rotation_angle_deg(R):
    """Geodesic angle of a rotation matrix (deg)."""
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def compare_clouds(new, ref, icp_max_corr_m, icp_voxel_m):
    """Compare new vs ref: direct NN distances + rigid ICP (new -> ref)."""
    direct_new_to_ref = nn_distance_stats(new, ref)
    direct_ref_to_new = nn_distance_stats(ref, new)

    a, b = new, ref
    if icp_voxel_m > 0:                      # downsample only for the ICP solve
        a = new.voxel_down_sample(icp_voxel_m)
        b = ref.voxel_down_sample(icp_voxel_m)
    icp = o3d.pipelines.registration.registration_icp(
        a, b, icp_max_corr_m, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    T = icp.transformation
    trans_mm = float(np.linalg.norm(T[:3, 3]) * 1000.0)
    rot_deg = rotation_angle_deg(T[:3, :3])
    return {
        "direct_new_to_ref": direct_new_to_ref,
        "direct_ref_to_new": direct_ref_to_new,
        "icp_translation_mm": trans_mm,
        "icp_rotation_deg": rot_deg,
        "icp_fitness": float(icp.fitness),
        "icp_inlier_rmse_mm": float(icp.inlier_rmse * 1000.0),
        "icp_transformation": T.tolist(),
    }


def verdict(cmp, move_thresh_mm, rot_thresh_deg, rmse_thresh_mm):
    """Heuristic interpretation of the comparison metrics."""
    moved = (cmp["icp_translation_mm"] > move_thresh_mm
             or cmp["icp_rotation_deg"] > rot_thresh_deg)
    well_aligned = cmp["icp_inlier_rmse_mm"] <= rmse_thresh_mm and cmp["icp_fitness"] >= 0.5
    if moved and well_aligned:
        return ("CAMERA MOVED", "ICP recovered a significant rigid transform with a "
                "clean fit -- consistent with the camera shifting.")
    if not moved and well_aligned:
        return ("camera steady", "near-identity ICP with a clean fit -- camera looks "
                "unchanged.")
    if moved and not well_aligned:
        return ("INCONCLUSIVE", "large transform but poor fit -- the scene likely "
                "changed (or low overlap), so this isn't a clean camera-move signal.")
    return ("INCONCLUSIVE", "small transform but poor fit -- scene change / noise / "
            "low overlap; re-check the scene is identical.")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True,
                    help="dir with cloud_crop.ply + robot_state.json from orbbec_live_crop")
    ap.add_argument("--out_dir", default=None,
                    help="where to write captured_crop.ply + compare_report.json "
                         "(default: alongside the dataset, in <dataset_dir>/compare)")
    # robot reset
    ap.add_argument("--no_move", action="store_true",
                    help="skip the robot reset (arm already positioned)")
    ap.add_argument("--yes", action="store_true",
                    help="don't prompt for confirmation before moving the robot")
    ap.add_argument("--robot_ip", default=None,
                    help="UR5e IP (default: the one stored in robot_state.json)")
    ap.add_argument("--speed", type=float, default=0.5, help="moveJ joint speed (rad/s)")
    ap.add_argument("--accel", type=float, default=0.5, help="moveJ joint accel (rad/s^2)")
    ap.add_argument("--settle_s", type=float, default=0.5,
                    help="pause after the move before capturing (let things settle)")
    # camera
    ap.add_argument("--serial", default=None)
    ap.add_argument("--color_w", type=int, default=1280)
    ap.add_argument("--color_h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--exposure", type=int, default=None)
    ap.add_argument("--warmup_seconds", type=float, default=3.0)
    # crop knobs not stored in the dataset (fall back to orbbec_live_crop defaults)
    ap.add_argument("--plane_slope_m", type=float, default=None)
    ap.add_argument("--seed_win", type=int, default=None)
    ap.add_argument("--voxel_m", type=float, default=None)
    # comparison
    ap.add_argument("--icp_max_corr_m", type=float, default=0.05,
                    help="ICP max correspondence distance (m)")
    ap.add_argument("--icp_voxel_m", type=float, default=0.005,
                    help="voxel size for the ICP solve only (m); 0 = full density")
    ap.add_argument("--move_thresh_mm", type=float, default=5.0,
                    help="ICP translation above this flags a camera move")
    ap.add_argument("--rot_thresh_deg", type=float, default=1.0,
                    help="ICP rotation above this flags a camera move")
    ap.add_argument("--rmse_thresh_mm", type=float, default=8.0,
                    help="ICP inlier RMSE below this counts as a clean fit")
    ap.add_argument("--vis", action="store_true",
                    help="show reference (gray) vs new (red) and the ICP-aligned new (green)")
    args = ap.parse_args()

    ref, state = load_dataset(args.dataset_dir)
    print(f"[compare] reference cloud: {len(ref.points)} pts from {args.dataset_dir}")

    rstate = state.get("robot_state")
    # --- reset the robot ---
    if args.no_move:
        print("[compare] --no_move: skipping robot reset")
    elif not rstate or "joints_rad" not in rstate:
        print("[compare] no joints in robot_state.json; cannot reset (use --no_move)")
        return 1
    else:
        robot_ip = args.robot_ip or rstate.get("robot_ip", "192.168.1.10")
        reset_robot(robot_ip, rstate["joints_rad"], args.speed, args.accel, args.yes)
        if args.settle_s > 0:
            time.sleep(args.settle_s)

    overrides = {k: v for k, v in (("plane_slope_m", args.plane_slope_m),
                                   ("seed_win", args.seed_win),
                                   ("voxel_m", args.voxel_m)) if v is not None}

    # --- capture + crop a fresh cloud with the SAME pipeline ---
    pipe, align = open_camera(args.serial, args.color_w, args.color_h, args.fps,
                              exposure=args.exposure)
    try:
        warmup_autoexposure(pipe, align, max_seconds=args.warmup_seconds)
        K, cam = color_intrinsics(pipe)
        pcf = make_pointcloud_filter(cam)
        cfg, frame = crop_config_from_state(state, K, overrides)
        x_axis, y_axis, z_axis, up_sign, center_m, seed_px0 = frame
        _, fs = capture_aligned(pipe, align)
        grid = orbbec_pointcloud(pcf, fs)
        new, n_keep = crop_cloud(grid, x_axis, y_axis, z_axis, up_sign,
                                 center_m, seed_px0, cfg)
    finally:
        pipe.stop()
    print(f"[compare] new cloud: {n_keep} pts")
    if new.is_empty():
        print("[compare] new cloud is empty; nothing to compare")
        return 1

    # --- compare ---
    cmp = compare_clouds(new, ref, args.icp_max_corr_m, args.icp_voxel_m)
    label, why = verdict(cmp, args.move_thresh_mm, args.rot_thresh_deg,
                         args.rmse_thresh_mm)

    d = cmp["direct_new_to_ref"]
    print("\n================= camera-drift report =================")
    print(f"  reference pts : {len(ref.points)}")
    print(f"  new pts       : {len(new.points)}")
    print(f"  direct NN (new->ref): median {d['median_mm']:.2f} mm | "
          f"mean {d['mean_mm']:.2f} | p95 {d['p95_mm']:.2f}")
    print(f"  ICP transform : {cmp['icp_translation_mm']:.2f} mm, "
          f"{cmp['icp_rotation_deg']:.3f} deg")
    print(f"  ICP fit       : fitness {cmp['icp_fitness']:.3f}, "
          f"inlier RMSE {cmp['icp_inlier_rmse_mm']:.2f} mm")
    print(f"  VERDICT       : {label} -- {why}")
    print("=======================================================\n")

    # --- save outputs ---
    out_dir = args.out_dir or os.path.join(args.dataset_dir, "compare")
    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(out_dir, "captured_crop.ply"), new)
    report = {
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "moved_robot": (not args.no_move),
        "verdict": label, "verdict_reason": why,
        "thresholds": {"move_mm": args.move_thresh_mm, "rot_deg": args.rot_thresh_deg,
                       "rmse_mm": args.rmse_thresh_mm},
        "n_ref": len(ref.points), "n_new": len(new.points),
        "metrics": cmp, "timestamp": time.time(),
    }
    json.dump(report, open(os.path.join(out_dir, "compare_report.json"), "w"), indent=2)
    print(f"[compare] wrote captured_crop.ply + compare_report.json -> {out_dir}")

    if args.vis:
        ref_v = o3d.geometry.PointCloud(ref); ref_v.paint_uniform_color([0.6, 0.6, 0.6])
        new_v = o3d.geometry.PointCloud(new); new_v.paint_uniform_color([0.9, 0.1, 0.1])
        aligned = o3d.geometry.PointCloud(new)
        aligned.transform(np.array(cmp["icp_transformation"]))
        aligned.paint_uniform_color([0.1, 0.8, 0.1])
        print("[compare] viewer: gray=reference, red=new (as-is), green=new ICP-aligned")
        o3d.visualization.draw_geometries([ref_v, new_v, aligned],
                                          window_name="camera-drift compare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
