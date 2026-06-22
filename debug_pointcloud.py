"""Offline / live de-risk tool for the PointNet cloud pipeline (see POINTCLOUD_EVAL.md §8).

Captures (or loads) a depth frame, optionally segments it with SAM2, builds the EE-frame
segmented cloud exactly as the policy will see it, prints per-class budget realization +
AABB, optionally renders it in Open3D colored by seg label, and saves the cloud. NO robot
motion, NO policy stepping -- purely to validate frame / scale / labels.

Depth sources (--depth-source)
  realsense  hardware depth (aligned to color); SAM2 runs on color.
  ffs        Fast-FoundationStereo on the IR stereo pair (left-IR frame); SAM2 on left-IR.
  file       depth/K/extrinsic/joints (+ optional label_map) from --from-file.
  (--demo uses a synthetic scene; no hardware, no SAM2.)

Segmentation (--seg) prompts SAM2 interactively (one window per class: robot/peg/hole)
to produce the 4th channel. Without --seg the cloud is geometry-only (label 0.0).

Examples
  python debug_pointcloud.py --demo --save /tmp/demo
  python debug_pointcloud.py --live --depth-source ffs --serial 215122255213 \
      --joints 0,-1.57,1.57,-1.57,-1.57,0 --seg --ffs-mock     # plumbing before FFS weights
  python debug_pointcloud.py --live --depth-source realsense --joints ... --seg
"""

from __future__ import annotations

import pathlib

import click
import numpy as np

from diffusion_policy.real_world import pointcloud_builder as B

LABEL_COLORS = {0.0: (0.6, 0.6, 0.6), -1.0: (0.85, 0.15, 0.15), 1.0: (0.15, 0.75, 0.2)}
LABEL_NAMES = {0.0: "robot", -1.0: "peg", 1.0: "hole"}


def _load_array(path):
    if isinstance(path, str) and "," in path:
        return np.array([float(x) for x in path.split(",")], np.float64)
    return np.load(path)


def _ee_pose_from_joints(arm_joints):
    from diffusion_policy.real_world.ur5e_kinematics import get_ee_pose
    pos, quat = get_ee_pose(np.asarray(arm_joints, np.float64))
    return np.asarray(pos), np.asarray(quat)


def _synthetic_frame():
    K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
    H, W = 480, 640
    depth = np.zeros((H, W), np.float32)
    lab = np.full((H, W), np.nan, np.float32)
    rng = np.random.default_rng(0)
    for (label, (r0, r1, c0, c1), z) in [
        (0.0, (80, 220, 80, 320), 0.7), (-1.0, (250, 320, 260, 360), 0.6),
        (1.0, (260, 360, 380, 520), 0.65),
    ]:
        ys, xs = np.mgrid[r0:r1, c0:c1]
        depth[ys, xs] = (z + rng.normal(0, 0.005, ys.shape)) * 1000.0
        lab[ys, xs] = label
    T = np.eye(4); T[:3, 3] = [0.5, 0.0, 0.3]
    return depth, lab, K, T, np.array([0., -1.57, 1.57, -1.57, -1.57, 0.]), 1000.0


def _capture_realsense(serial, resolution):
    """One (color RGB, depth, K, units-per-metre) from a front RealSense (hardware depth)."""
    from multiprocessing.managers import SharedMemoryManager
    from diffusion_policy.real_world.single_realsense import SingleRealsense
    shm = SharedMemoryManager(); shm.start()
    cam = SingleRealsense(shm, serial, resolution=tuple(resolution),
                          enable_color=True, enable_depth=True)
    cam.start(wait=True); cam.start_wait()
    out = cam.get()
    for _ in range(30):
        out = cam.get()
    K = cam.get_intrinsics()
    ds_rs = cam.get_depth_scale()
    color = out["color"][..., ::-1].copy()  # BGR->RGB
    depth = out["depth"]
    cam.stop(wait=True); shm.shutdown()
    return color, depth, K, 1.0 / ds_rs


def _capture_ffs(serial, resolution, mock):
    """Stereo IR -> FoundationStereo metric depth (left-IR frame). seg image = left IR (RGB)."""
    from diffusion_policy.real_world.realsense_stereo import capture_stereo
    from diffusion_policy.real_world.ffs_depth_client import FFSDepthClient
    f = capture_stereo(serial, resolution=tuple(resolution), want_color=False)
    with FFSDepthClient(mock=mock) as ffs:
        depth = ffs.infer(f.left, f.right, f.K_ir, f.baseline_m)  # metres
    seg_img = np.repeat(f.left[..., None], 3, axis=2)  # IR->3ch for SAM2
    return seg_img, depth, f.K_ir, 1.0  # depth already metric -> units-per-metre = 1


def _segment(seg_img, sam2_ckpt, sam2_cfg, erode):
    from diffusion_policy.real_world.pointcloud_segmenter import (
        PointCloudSegmenter, pick_prompts_interactive)
    prompts = pick_prompts_interactive(seg_img)
    if not prompts:
        click.echo("  [seg] no prompts given -> geometry-only")
        return None
    seg = PointCloudSegmenter(sam2_ckpt, sam2_cfg)
    return seg.label_map(seg_img, prompts, erode=erode)


def _print_stats(stats, cloud):
    click.echo(f"  raw valid depth px : {stats.n_raw_valid}")
    click.echo(f"  after crop         : {stats.n_after_crop}")
    for lab in B.DEFAULT_BUDGET:
        if lab not in stats.per_class_realized:
            continue
        avail = stats.per_class_available.get(lab, 0)
        got = stats.per_class_realized.get(lab, 0)
        flag = "  <-- SHORT (padded)" if lab in stats.short_classes else ""
        click.echo(f"  {LABEL_NAMES.get(lab, lab):>5}: avail {avail:5d}  emitted {got:5d}{flag}")
    bmin, bmax = stats.bbox_min, stats.bbox_max
    if bmin is not None and bmax is not None:
        click.echo(f"  EE-frame AABB min  : {np.round(bmin, 3)}")
        click.echo(f"  EE-frame AABB max  : {np.round(bmax, 3)}")
    click.echo(f"  cloud shape        : {cloud.shape}")


def _visualize(cloud, geometry_only):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    if geometry_only:
        z = cloud[:, 2]; t = (z - z.min()) / (np.ptp(z) + 1e-9)
        colors = np.stack([t, 0.4 * np.ones_like(t), 1 - t], axis=1)
    else:
        colors = np.array([LABEL_COLORS.get(float(l), (1, 1, 0)) for l in cloud[:, 3]])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    click.echo("  [open3d] close the window to continue...")
    o3d.visualization.draw_geometries([pcd, axes], window_name="PointNet cloud (EE frame)")


@click.command()
@click.option("--demo", is_flag=True, help="Synthetic scene, no hardware.")
@click.option("--from-file", "from_file", type=click.Path(exists=True), default=None,
              help="Dir with depth.npy,K.npy,extrinsic.npy,joints.npy[,label_map.npy] or an .npz.")
@click.option("--live", is_flag=True, help="Capture one frame from a RealSense.")
@click.option("--serial", default="215122255213", help="Front RealSense serial (live).")
@click.option("--resolution", nargs=2, type=int, default=(1280, 720))
@click.option("--depth-source", type=click.Choice(["realsense", "ffs", "file"]), default="realsense")
@click.option("--ffs-mock", is_flag=True, help="FFS ramp depth (test stereo+transform sans weights).")
@click.option("--extrinsic", type=click.Path(exists=True),
              default="calib/front_cam_to_base_simapprox.npy",
              help="(4,4) camera->base npy. Defaults to the sim-approx D455 extrinsic.")
@click.option("--joints", default=None, help="6 comma-sep joint angles (rad) for live EE pose.")
@click.option("--depth-scale", type=float, default=1000.0, help="depth units per metre (file mode).")
@click.option("--crop", nargs=6, type=float, default=None,
              help="EE-frame AABB: xmin ymin zmin xmax ymax zmax (metres).")
@click.option("--seg", is_flag=True, help="Run SAM2 to produce robot/peg/hole labels (live).")
@click.option("--sam2-ckpt", default="orbbec/weights/sam2/sam2.1_hiera_base_plus.pt")
@click.option("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
@click.option("--erode", type=int, default=3, help="erode each SAM2 mask by N px.")
@click.option("--no-vis", is_flag=True, help="Skip Open3D window (headless).")
@click.option("--save", "save_path", default=None, help="Save cloud to <path>.npy (+ .ply).")
def main(demo, from_file, live, serial, resolution, depth_source, ffs_mock, extrinsic,
         joints, depth_scale, crop, seg, sam2_ckpt, sam2_cfg, erode, no_vis, save_path):
    label_map, seg_img = None, None

    if demo:
        depth, label_map, K, T_cam_base, arm_joints, depth_scale = _synthetic_frame()
        ee_pos, ee_quat = _ee_pose_from_joints(arm_joints)
        mode = "demo"

    elif from_file:
        p = pathlib.Path(from_file)
        d = np.load(p) if p.suffix == ".npz" else {f.stem: np.load(f) for f in p.glob("*.npy")}
        depth, K, T_cam_base = d["depth"], d["K"], d["extrinsic"]
        ee_pos, ee_quat = _ee_pose_from_joints(d["joints"])
        label_map = d["label_map"] if "label_map" in d else None
        mode = "file"

    elif live:
        assert joints is not None, "--joints required for --live (EE pose via FK)"
        T_cam_base = np.load(extrinsic)
        ee_pos, ee_quat = _ee_pose_from_joints(_load_array(joints))
        if depth_source == "ffs":
            seg_img, depth, K, depth_scale = _capture_ffs(serial, resolution, ffs_mock)
        else:
            seg_img, depth, K, depth_scale = _capture_realsense(serial, resolution)
        if seg:
            label_map = _segment(seg_img, sam2_ckpt, sam2_cfg, erode)
        mode = f"live:{depth_source}"
    else:
        raise click.UsageError("Pick one of --demo / --from-file / --live.")

    crop_lo, crop_hi = (crop[:3], crop[3:]) if crop else (None, None)
    cloud, stats = B.build_cloud(
        depth, K, T_cam_base, ee_pos, ee_quat, label_map=label_map,
        depth_scale=depth_scale, crop_lo=crop_lo, crop_hi=crop_hi)

    click.echo(f"[debug_pointcloud] mode={mode} seg={'on' if label_map is not None else 'off'}")
    _print_stats(stats, cloud)

    if save_path:
        sp = pathlib.Path(save_path)
        np.save(sp.with_suffix(".npy"), cloud)
        click.echo(f"  saved {sp.with_suffix('.npy')}")
        try:
            import open3d as o3d
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(cloud[:, :3])
            o3d.io.write_point_cloud(str(sp.with_suffix(".ply")), pc)
            click.echo(f"  saved {sp.with_suffix('.ply')}")
        except Exception as e:
            click.echo(f"  (ply skipped: {e})")

    if not no_vis:
        try:
            _visualize(cloud, geometry_only=label_map is None)
        except Exception as e:
            click.echo(f"  [vis skipped: {e}]")


if __name__ == "__main__":
    main()
