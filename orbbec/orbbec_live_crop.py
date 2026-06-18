"""
Live Orbbec Femto Bolt point-cloud viewer, cropped to a square on a plane.

Streams RGB-D from the Bolt, turns each frame into the camera's dense point
cloud, and keeps ONLY the points inside an INFINITE rectangular prism: a window
in the plane through three picked points (p0, p1, p2), extruded without bound
along the plane normal. So anything that projects into that window -- at any
height above or below the plane -- is kept. The window is independently bounded
on each in-plane axis: [--x_min, --x_max] from p1 along x_axis (default
-0.425..+0.5 m) and [--y_min, --y_max] along y_axis (default -0.5..+0.4 m).

The plane frame (all unit vectors, right-handed):
    x_axis = normalize(p1 - p0)              # in-plane
    z_axis = normalize((p1-p0) x (p2-p0))    # plane normal (prism axis; unbounded)
    y_axis = z_axis x x_axis                 # in-plane, completes the frame
A camera point P is kept when, with d = P - p1,
    x_min <= d.x_axis <= x_max  and  y_min <= d.y_axis <= y_max  (d.z_axis free).
So the window's edges run along x_axis / y_axis. Pick p0,p1 along a physical edge
(e.g. the table edge) if you want the crop axis-aligned to something real.

Floor removal: by DEFAULT this is a one-sided height crop -- keep only points
more than --floor_height_m ABOVE the plane, dropping the floor slab AND everything
below the plane (w <= 0). Pass --slope_floor to instead use a slope-aware region
grow: each frame we project the 3 reference points into the color image and grow
the floor outward from them across the per-pixel height field (height = signed
distance from the plane). A neighbor pixel joins the floor only if BOTH (a) its
height differs from the ADJACENT floor pixel by < --plane_slope_m (absorbs gentle
slopes, blocks sharp object edges), and (b) its absolute height stays within
--floor_height_m of the plane. The grown floor and everything under the plane are
dropped; objects rising off the plane survive. Which side is "up" comes from the
p0/p1/p2 order (the normal's sign); use --flip_normal if it's the wrong side.

The picked points are in the Femto Bolt color/camera frame, in MILLIMETERS
(that's what the interactive picker / DepthPoints PLY report). Defaults below are
the three points you picked; override with --p0/--p1/--p2 "x,y,z" (mm).

Runs in the `foundstereo` conda env (pyorbbecsdk2 + open3d). Camera I/O comes
from the shared `orbbec_camera` core module.

Usage:
  (foundstereo)$ python orbbec_live_crop.py
  (foundstereo)$ python orbbec_live_crop.py --x_min -0.425 --x_max 0.5 --y_min -0.5 --y_max 0.4
  (foundstereo)$ python orbbec_live_crop.py --p1 -108.2,139.2,689 --voxel_m 0.004
  # one-shot: save a single cropped cloud + the robot's joint state, no viewer
  (foundstereo)$ python orbbec_live_crop.py --save_dir tmp/capture0 --robot_ip 192.168.1.10
Close the viewer window (or Ctrl-C) to stop.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import cv2
import open3d as o3d

from orbbec_camera import (
    open_camera, warmup_autoexposure, capture_aligned,
    make_pointcloud_filter, color_intrinsics, orbbec_pointcloud,
)

# Defaults = the three points picked on the plane (camera frame, millimeters).
DEFAULT_P0 = "-526.550,48.431,881.000"
DEFAULT_P1 = "-115.248,120.786,734.000"
DEFAULT_P2 = "101.551,7.410,1080.000"


def parse_xyz(s):
    """'x,y,z' -> float ndarray (3,)."""
    parts = s.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"expected 3 numbers 'x,y,z', got {s!r}")
    return np.array([float(p) for p in parts], dtype=float)


def plane_frame(p0, p1, p2):
    """Right-handed orthonormal frame on the plane through p0,p1,p2.
    Returns (x_axis, y_axis, z_axis); x along p0->p1, z = plane normal."""
    e1 = p1 - p0
    e2 = p2 - p0
    normal = np.cross(e1, e2)
    if np.linalg.norm(normal) < 1e-9:
        raise ValueError("the 3 points are collinear; they define no plane")
    z_axis = normal / np.linalg.norm(normal)
    x_axis = e1 / np.linalg.norm(e1)
    y_axis = np.cross(z_axis, x_axis)            # in-plane, unit, ⟂ x_axis
    return x_axis, y_axis, z_axis


def rotate_in_plane(x_axis, y_axis, z_axis, deg):
    """Spin the in-plane (x,y) axes about z_axis by `deg`. Positive = clockwise
    as seen looking along -z_axis (i.e. with the normal pointing toward you)."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    x_rot = c * x_axis - s * y_axis
    y_rot = s * x_axis + c * y_axis
    return x_rot, y_rot


def crop_rect_lineset(center, x_axis, y_axis, x_min, x_max, y_min, y_max):
    """Outline of the in-plane crop rectangle (at w=0), for visual reference.
    Spans [x_min, x_max] along x_axis and [y_min, y_max] along y_axis."""
    corners = np.array([
        center + ux * x_axis + uy * y_axis
        for ux, uy in ((x_min, y_min), (x_max, y_min),
                       (x_max, y_max), (x_min, y_max))
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(np.array(edges)))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.7, 0.0], (len(edges), 1)))   # amber
    return ls


def project_pixel(P_mm, K):
    """Project a 3D camera-frame point (mm) to its (u,v) color pixel via K."""
    X, Y, Z = P_mm
    u = K[0, 0] * X / Z + K[0, 2]
    v = K[1, 1] * Y / Z + K[1, 2]
    return int(round(u)), int(round(v))


def nudge_seed(u, v, valid_img, crop_img, h_img, win):
    """Move a seed pixel to the nearest valid, in-crop pixel closest to the plane
    (|height| min) within a +-win window. Returns (u,v) or None if none qualify."""
    H, W = h_img.shape
    best, best_abs = None, np.inf
    for dv in range(-win, win + 1):
        vv = v + dv
        if vv < 0 or vv >= H:
            continue
        for du in range(-win, win + 1):
            uu = u + du
            if uu < 0 or uu >= W:
                continue
            if valid_img[vv, uu] and crop_img[vv, uu]:
                a = abs(float(h_img[vv, uu]))
                if a < best_abs:
                    best_abs, best = a, (uu, vv)
    return best


def grow_floor_mask(h_img, valid_img, crop_img, seeds_uv, slope_tol, height_cap):
    """Region-grow the floor from seed pixels across the height image `h_img`
    (signed distance from the plane, meters). A pixel joins the floor when its
    height differs from the ADJACENT floor pixel by < slope_tol -- so a gradual
    slope is absorbed step by step, but an object's sharp height step blocks it.
    A pixel whose |height| exceeds height_cap is a barrier too, so the grown
    floor can never drift more than that far off the plane (slope + height).
    Confined to crop & valid pixels. Returns a (H,W) bool floor mask."""
    H, W = h_img.shape
    work = h_img.astype(np.float32).copy()
    work[~valid_img] = 1e6                        # holes act as impassable walls
    # floodFill mask is 2px larger; nonzero entries are barriers it won't cross.
    ff = np.zeros((H + 2, W + 2), np.uint8)
    barrier = (~crop_img) | (~valid_img)
    if height_cap > 0:
        barrier = barrier | (np.abs(h_img) > height_cap)   # cap how far off-plane
    ff[1:-1, 1:-1][barrier] = 2                    # confine flood to crop & valid & band
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8)  # 4-conn, floating range, fill=1
    for (u, v) in seeds_uv:
        if u is None:
            continue
        if 0 <= u < W and 0 <= v < H and ff[v + 1, u + 1] == 0:
            cv2.floodFill(work, ff, (u, v), 0, slope_tol, slope_tol, flags)
    return ff[1:-1, 1:-1] == 1


def read_robot_state(robot_ip):
    """Read the UR5e's current 6-DOF joint angles (radians) + TCP pose over RTDE,
    so a sim/replay can reset to the exact pose this cloud was captured at. Best-
    effort: returns None (with a warning) if ur_rtde is missing or the robot is
    unreachable."""
    try:
        from rtde_receive import RTDEReceiveInterface
    except Exception as e:
        print(f"[live-crop] ur_rtde not importable ({e}); skipping robot state")
        return None
    try:
        rtde_r = RTDEReceiveInterface(robot_ip)
        q = np.array(rtde_r.getActualQ(), dtype=float)          # 6 joints, radians
        tcp = np.array(rtde_r.getActualTCPPose(), dtype=float)  # x,y,z,rx,ry,rz
        rtde_r.disconnect()
    except Exception as e:
        print(f"[live-crop] could not read robot at {robot_ip} ({e}); "
              f"skipping robot state")
        return None
    print(f"[live-crop] robot joints (rad): {np.round(q, 4).tolist()}")
    return {
        "robot_ip": robot_ip,
        "joints_rad": q.tolist(),
        "joints_deg": np.rad2deg(q).tolist(),
        "tcp_pose": tcp.tolist(),       # meters + axis-angle (rad), UR base frame
        "timestamp": time.time(),
    }


def crop_cloud(grid, x_axis, y_axis, z_axis, up_sign, center_m, seed_px0, args):
    """Apply the in-plane crop + floor removal to one (H,W,6) Orbbec grid and
    return (open3d PointCloud in meters, n_kept)."""
    flat = grid.reshape(-1, 6)
    xyz = flat[:, :3] / 1000.0                        # mm -> m
    rgb = (flat[:, 3:6] / 255.0).clip(0.0, 1.0)
    valid = flat[:, 2] > 0.0                          # drop invalid depth (z==0)

    d = xyz - center_m
    u = d @ x_axis
    v = d @ y_axis
    w = (d @ z_axis) * up_sign                         # signed height; + = above plane
    in_crop = ((u >= args.x_min) & (u <= args.x_max)
               & (v >= args.y_min) & (v <= args.y_max))
    keep = valid & in_crop

    if args.slope_floor:
        # slope-aware region grow from the 3 reference points (opt-in)
        H, W = grid.shape[0], grid.shape[1]
        h_img = w.reshape(H, W)
        valid_img = valid.reshape(H, W)
        crop_img = in_crop.reshape(H, W)
        # nudge each projected reference point onto a real on-plane seed pixel
        seeds = [nudge_seed(su, sv, valid_img, crop_img, h_img, args.seed_win)
                 for (su, sv) in seed_px0]
        floor = grow_floor_mask(h_img, valid_img, crop_img, seeds,
                                args.plane_slope_m, args.floor_height_m)
        keep &= ~floor.reshape(-1)
        keep &= w > 0.0                  # also drop everything under the plane
    elif args.floor_height_m > 0:
        # default: one-sided crop -- keep only points ABOVE the floor band,
        # dropping the floor slab AND everything below the plane (w <= 0)
        keep &= w > args.floor_height_m

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[keep])
    pcd.colors = o3d.utility.Vector3dVector(rgb[keep])
    if args.voxel_m > 0 and len(pcd.points):
        pcd = pcd.voxel_down_sample(args.voxel_m)
    return pcd, int(keep.sum())


def plane_axes_frame(center, x_axis, y_axis, z_axis, size):
    """A coordinate triad at p1 oriented to the plane frame (x=red,y=green,n=blue)."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = center
    frame.transform(T)
    return frame


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0", default=DEFAULT_P0, help="plane point 0 'x,y,z' (mm)")
    ap.add_argument("--p1", default=DEFAULT_P1, help="plane point 1 = crop center 'x,y,z' (mm)")
    ap.add_argument("--p2", default=DEFAULT_P2, help="plane point 2 'x,y,z' (mm)")
    ap.add_argument("--x_min", type=float, default=-0.425,
                    help="lower crop bound along x_axis, signed offset from p1 (m)")
    ap.add_argument("--x_max", type=float, default=0.5,
                    help="upper crop bound along x_axis, signed offset from p1 (m)")
    ap.add_argument("--y_min", type=float, default=-0.425,
                    help="lower crop bound along y_axis, signed offset from p1 (m)")
    ap.add_argument("--y_max", type=float, default=0.45,
                    help="upper crop bound along y_axis, signed offset from p1 (m)")
    ap.add_argument("--rotate_deg", type=float, default=25.0,
                    help="rotate the in-plane crop axes about the plane normal "
                         "(deg; + = clockwise looking along the normal). Negate if "
                         "it spins the wrong way for your camera view.")
    ap.add_argument("--floor_height_m", type=float, default=0.034,
                    help="height floor crop (ON by default): keep only points more "
                         "than this far ABOVE the plane, dropping the floor slab and "
                         "everything below it; 0 disables. With --slope_floor this "
                         "instead caps how far the grow may climb.")
    ap.add_argument("--flip_normal", action="store_true",
                    help="flip which side of the plane counts as 'up' (the normal's "
                         "sign comes from p0/p1/p2 order); use if the floor crop keeps "
                         "the wrong side / the cloud comes up empty")
    ap.add_argument("--slope_floor", action="store_true",
                    help="opt-in: replace the plain height band with a slope-aware "
                         "region grow from the 3 reference points (still bounded by "
                         "--floor_height_m), so gentle slopes are absorbed but objects "
                         "rising off the plane are kept")
    ap.add_argument("--plane_slope_m", type=float, default=0.001,
                    help="(--slope_floor) max height delta (m) between ADJACENT pixels "
                         "to still count as the same surface; bigger absorbs steeper "
                         "slopes but risks climbing onto objects")
    ap.add_argument("--seed_win", type=int, default=12,
                    help="(--slope_floor) search radius (px) around each projected "
                         "reference point for a valid seed pixel on the plane")
    ap.add_argument("--voxel_m", type=float, default=0.0,
                    help="voxel-downsample size (m); 0 disables (keep full density)")
    # one-shot save (instead of the live viewer)
    ap.add_argument("--save_dir", default=None,
                    help="capture ONE frame, save the cropped cloud + robot joint "
                         "state into this dir, and exit (no live viewer)")
    ap.add_argument("--robot_ip", default="192.168.1.10",
                    help="UR5e IP to read joint state from when --save_dir is set")
    ap.add_argument("--read_robot", type=int, default=1,
                    help="read + save robot joint state alongside the cloud (0 to skip)")
    # camera
    ap.add_argument("--serial", default=None, help="Orbbec serial (default: first device)")
    ap.add_argument("--color_w", type=int, default=1280)
    ap.add_argument("--color_h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--exposure", type=int, default=None,
                    help="fix color exposure (disables AE); default None = auto")
    ap.add_argument("--warmup_seconds", type=float, default=3.0)
    ap.add_argument("--show_box", type=int, default=1,
                    help="draw the crop-box wireframe + plane axes (1/0)")
    ap.add_argument("--point_size", type=float, default=1.5)
    args = ap.parse_args()

    p0, p1, p2 = parse_xyz(args.p0), parse_xyz(args.p1), parse_xyz(args.p2)
    x_axis, y_axis, z_axis = plane_frame(p0, p1, p2)
    if args.rotate_deg:
        x_axis, y_axis = rotate_in_plane(x_axis, y_axis, z_axis, args.rotate_deg)
    center_m = p1 / 1000.0                         # p1, mm -> m
    up_sign = -1.0 if args.flip_normal else 1.0    # which way along z counts as 'up'
    print(f"[live-crop] plane frame:\n  x={x_axis}\n  y={y_axis}\n  n={z_axis}"
          f"  (up = {'-' if args.flip_normal else '+'}normal)")
    print(f"[live-crop] crop: x in [{args.x_min:+.3f}, {args.x_max:+.3f}]m, "
          f"y in [{args.y_min:+.3f}, {args.y_max:+.3f}]m from p1, INFINITE along the "
          f"plane normal; p1={center_m} m")

    pipe, align = open_camera(args.serial, args.color_w, args.color_h, args.fps,
                              exposure=args.exposure)
    vis = None
    try:
        warmup_autoexposure(pipe, align, max_seconds=args.warmup_seconds)
        K, cam = color_intrinsics(pipe)
        pcf = make_pointcloud_filter(cam)
        # where the 3 plane points land in the color image -> floor seeds
        seed_px0 = [project_pixel(p0, K), project_pixel(p1, K), project_pixel(p2, K)]

        # --- one-shot save mode: grab a single frame, write cloud + robot state ---
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            _, fs = capture_aligned(pipe, align)
            grid = orbbec_pointcloud(pcf, fs)
            pcd, n_keep = crop_cloud(grid, x_axis, y_axis, z_axis, up_sign,
                                     center_m, seed_px0, args)
            ply_path = os.path.join(args.save_dir, "cloud_crop.ply")
            o3d.io.write_point_cloud(ply_path, pcd)
            robot_state = read_robot_state(args.robot_ip) if args.read_robot else None
            state_path = os.path.join(args.save_dir, "robot_state.json")
            json.dump({
                "robot_state": robot_state,
                "n_points": n_keep,
                "cloud_file": "cloud_crop.ply",
                "frame": "Femto Bolt color/camera (meters)",
                "p0_mm": p0.tolist(), "p1_mm": p1.tolist(), "p2_mm": p2.tolist(),
                "x_bounds_m": [args.x_min, args.x_max],
                "y_bounds_m": [args.y_min, args.y_max],
                "rotate_deg": args.rotate_deg, "flip_normal": bool(args.flip_normal),
                "floor_height_m": args.floor_height_m, "slope_floor": bool(args.slope_floor),
            }, open(state_path, "w"), indent=2)
            print(f"[live-crop] saved {n_keep}-pt cloud -> {ply_path}")
            print(f"[live-crop] saved robot state    -> {state_path}")
            return 0

        vis = o3d.visualization.Visualizer()
        vis.create_window("Orbbec live crop", width=1280, height=720)
        vis.get_render_option().point_size = args.point_size
        vis.get_render_option().background_color = np.array([0.05, 0.05, 0.05])

        # static reference geometry (added once; frames the initial view)
        if args.show_box:
            vis.add_geometry(crop_rect_lineset(center_m, x_axis, y_axis,
                                               args.x_min, args.x_max,
                                               args.y_min, args.y_max))
            vis.add_geometry(plane_axes_frame(center_m, x_axis, y_axis, z_axis,
                                              size=0.25))

        pcd = o3d.geometry.PointCloud()
        pcd_added = False
        frames = 0
        while True:
            color, fs = capture_aligned(pipe, align)
            grid = orbbec_pointcloud(pcf, fs)                 # (H,W,6) xyz(mm)+rgb
            cur, n_keep = crop_cloud(grid, x_axis, y_axis, z_axis, up_sign,
                                     center_m, seed_px0, args)
            # copy into the persistent geometry so update_geometry tracks it
            pcd.points = cur.points
            pcd.colors = cur.colors

            if not pcd_added:
                vis.add_geometry(pcd)                         # first frame: auto-frames
                pcd_added = True
            else:
                vis.update_geometry(pcd)
            if not vis.poll_events():                         # window closed
                break
            vis.update_renderer()

            frames += 1
            if frames % 30 == 0:
                print(f"[live-crop] frame {frames}: {n_keep} cropped pts")
    except KeyboardInterrupt:
        print("\n[live-crop] interrupted")
    finally:
        if vis is not None:
            vis.destroy_window()
        pipe.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
