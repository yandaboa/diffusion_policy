"""Offline / live de-risk tool for the PointNet cloud pipeline (see POINTCLOUD_EVAL.md §8).

Captures (or loads) a depth frame, optionally segments it with SAM2, builds the EE-frame
segmented cloud exactly as the policy will see it, prints per-class budget realization +
AABB, optionally renders it in Open3D colored by seg label, and saves the cloud. NO robot
motion, NO policy stepping -- purely to validate frame / scale / labels.

Depth sources (--depth-source)
  realsense  hardware depth (aligned to color); SAM2 runs on color.
  ffs        Fast-FoundationStereo on the IR stereo pair (left-IR frame); SAM2 runs on the
             color image, masks reverse-warped into the left-IR/depth frame (SAM2 >> on RGB).
  file       depth/K/extrinsic/joints (+ optional label_map) from --from-file.
  (--demo uses a synthetic scene; no hardware, no SAM2.)

Segmentation (--seg) prompts SAM2 interactively (one window per class: robot/peg/hole)
to produce the 4th channel. Without --seg the cloud is geometry-only (label 0.0).

Real-time (--video) holds the camera open, prompts SAM2 ONCE on the first frame, then
*tracks* every subsequent frame with the SAM2 streaming predictor (no re-clicking, so the
masks follow the moving arm) and rebuilds the segmented cloud per frame. It saves an
annotated MP4 (color + per-class mask overlay + a per-stage latency / FPS HUD) so you can
eyeball both the segmentation+masking quality and whether it keeps up with the control
loop. Needs the SAM2 streaming fork (see online_segmentation.md / StreamingSegmenter).

Examples
  python debug_pointcloud.py --demo --save /tmp/demo
  python debug_pointcloud.py --live --depth-source ffs --serial 215122255213 \
      --joints 0,-1.57,1.57,-1.57,-1.57,0 --seg --ffs-mock     # plumbing before FFS weights
  python debug_pointcloud.py --live --depth-source realsense --joints ... --seg
  python debug_pointcloud.py --video --depth-source ffs --joints ... \
      --duration 20 --video-out /tmp/seg_stream.mp4            # real-time seg+masking test
"""

from __future__ import annotations

import contextlib
import pathlib
import time

import click
import numpy as np

from diffusion_policy.real_world import pointcloud_builder as B

LABEL_COLORS = {0.0: (0.6, 0.6, 0.6), -1.0: (0.85, 0.15, 0.15), 1.0: (0.15, 0.75, 0.2)}
LABEL_NAMES = {0.0: "robot", -1.0: "peg", 1.0: "hole"}
# Per-class overlay colors for the saved real-time video (BGR, for cv2).
SEG_OVERLAY_BGR = {"robot": (200, 200, 200), "peg": (60, 60, 230), "hole": (70, 200, 70)}


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
    """Stereo IR -> FoundationStereo metric depth (left-IR frame).

    SAM2 runs far better on the COLOR image than on raw IR, so the seg image is the color
    frame and we return a ``warp`` dict (K_ir/K_color/T_ir_color) so the color-frame masks can
    be reverse-warped into the IR/depth frame. Returns (seg_img, depth, K_ir, units/m, warp).
    """
    from diffusion_policy.real_world.realsense_stereo import capture_stereo
    from diffusion_policy.real_world.ffs_depth_client import FFSDepthClient
    f = capture_stereo(serial, resolution=tuple(resolution), want_color=True)
    with FFSDepthClient(mock=mock) as ffs:
        depth = ffs.infer(f.left, f.right, f.K_ir, f.baseline_m)  # metres
    warp = {"K_ir": f.K_ir, "K_color": f.K_color, "T_ir_color": f.T_ir_color}
    return f.color, depth, f.K_ir, 1.0, warp  # depth already metric -> units-per-metre = 1


@contextlib.contextmanager
def _open_realsense_stream(serial, resolution):
    """Persistent hardware-depth RealSense stream; yields ``grab() -> (color RGB, depth, K, units/m)``.

    The streaming sibling of ``_capture_realsense`` (which starts/stops the camera per call):
    keep one ``SingleRealsense`` open so the --video hot loop pulls frames at sensor rate.
    """
    from multiprocessing.managers import SharedMemoryManager
    from diffusion_policy.real_world.single_realsense import SingleRealsense
    shm = SharedMemoryManager(); shm.start()
    cam = SingleRealsense(shm, serial, resolution=tuple(resolution),
                          enable_color=True, enable_depth=True)
    cam.start(wait=True); cam.start_wait()
    try:
        for _ in range(30):  # let auto-exposure settle before the first grab
            cam.get()
        K = cam.get_intrinsics()
        units_per_m = 1.0 / cam.get_depth_scale()

        def grab():
            out = cam.get()
            color = out["color"][..., ::-1].copy()  # BGR->RGB
            return color, out["depth"], K, units_per_m

        yield grab
    finally:
        cam.stop(wait=True); shm.shutdown()


def _draw_video_frame(color_rgb, masks, stats, timings, frame_idx, fps_avg):
    """Annotated BGR frame: per-class mask overlay + a latency / FPS / point-count HUD."""
    import cv2
    bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    for name, m in masks.items():
        col = np.array(SEG_OVERLAY_BGR.get(name, (0, 255, 255)), np.float32)
        bgr[m] = (0.5 * bgr[m] + 0.5 * col).astype(np.uint8)

    total_ms = sum(timings.values())
    lines = [f"frame {frame_idx:04d}   {fps_avg:4.1f} FPS avg   {1000.0 / max(total_ms, 1e-6):4.1f} inst"]
    lines.append("  ".join(f"{k} {v:5.1f}ms" for k, v in timings.items()) + f"   tot {total_ms:5.1f}ms")
    for name in ("robot", "peg", "hole"):
        lab = B.SEG_LABELS[name]
        if lab not in stats.per_class_realized:
            continue
        avail = stats.per_class_available.get(lab, 0)
        got = stats.per_class_realized.get(lab, 0)
        flag = " SHORT" if lab in stats.short_classes else ""
        lines.append(f"{name:>5}: avail {avail:5d}  emitted {got:4d}{flag}")

    pad, lh, scale = 8, 22, 0.55
    box_h = lh * len(lines) + pad
    text_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for t in lines)
    box_w = min(bgr.shape[1], text_w + 2 * pad)
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    bgr = cv2.addWeighted(overlay, 0.45, bgr, 0.55, 0)
    for i, text in enumerate(lines):
        cv2.putText(bgr, text, (pad, pad + lh * i + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (60, 255, 60), 1, cv2.LINE_AA)
    return bgr


def _run_video(depth_source, serial, resolution, ffs_mock, T_cam_base, ee_pos, ee_quat,
               sam2_ckpt, sam2_cfg, erode, crop_lo, crop_hi, video_out, video_fps,
               duration, max_frames):
    """Hold the camera open, prompt SAM2 once, track + rebuild the cloud per frame, save an MP4."""
    import cv2
    from diffusion_policy.real_world.pointcloud_segmenter import (
        StreamingSegmenter, compose_label_map, pick_prompts_interactive,
        warp_masks_to_depth_frame)

    with contextlib.ExitStack() as stack:
        # --- persistent frame source: returns (color RGB, depth, K, units/m, warp, (t_grab, t_depth)) ---
        if depth_source == "ffs":
            from diffusion_policy.real_world.realsense_stereo import open_stereo_stream
            from diffusion_policy.real_world.ffs_depth_client import FFSDepthClient
            grab_stereo = stack.enter_context(
                open_stereo_stream(serial, tuple(resolution), want_color=True))
            ffs = stack.enter_context(FFSDepthClient(mock=ffs_mock))

            def grab():
                t0 = time.perf_counter()
                f = grab_stereo()
                t1 = time.perf_counter()
                depth = ffs.infer(f.left, f.right, f.K_ir, f.baseline_m)  # metres
                t2 = time.perf_counter()
                warp = {"K_ir": f.K_ir, "K_color": f.K_color, "T_ir_color": f.T_ir_color}
                return f.color, depth, f.K_ir, 1.0, warp, (1e3 * (t1 - t0), 1e3 * (t2 - t1))
        else:
            grab_rs = stack.enter_context(_open_realsense_stream(serial, resolution))

            def grab():
                t0 = time.perf_counter()
                color, depth, K, ups = grab_rs()
                return color, depth, K, ups, None, (1e3 * (time.perf_counter() - t0), 0.0)

        # --- frame 0: prompt SAM2 once and seed the tracker ---
        color, depth, K, depth_scale, warp, (t_grab, t_depth) = grab()
        prompts = pick_prompts_interactive(color)
        if not prompts:
            raise click.UsageError("--video needs at least one class clicked on the first frame")
        click.echo(f"  [seg] streaming classes: {list(prompts)} (tracking, no re-clicking)")
        seg = StreamingSegmenter(sam2_ckpt, sam2_cfg)
        t0 = time.perf_counter()
        masks = seg.start(color, prompts)
        t_track = 1e3 * (time.perf_counter() - t0)

        writer, n, t_start, cum_ms, fps_avg = None, 0, time.perf_counter(), 0.0, 0.0
        try:
            while True:
                if n > 0:  # frame 0 already grabbed + seeded above
                    color, depth, K, depth_scale, warp, (t_grab, t_depth) = grab()
                    t0 = time.perf_counter()
                    masks = seg.track(color)
                    t_track = 1e3 * (time.perf_counter() - t0)

                # masks (color frame) -> label map aligned to depth -> cloud
                t0 = time.perf_counter()
                if warp is None:
                    label_map = compose_label_map(masks, depth.shape, erode=erode)
                else:
                    masks_depth = warp_masks_to_depth_frame(
                        masks, depth, warp["K_ir"], warp["K_color"], warp["T_ir_color"])
                    label_map = compose_label_map(masks_depth, depth.shape, erode=erode)
                _, stats = B.build_cloud(
                    depth, K, T_cam_base, ee_pos, ee_quat, label_map=label_map,
                    depth_scale=depth_scale, crop_lo=crop_lo, crop_hi=crop_hi)
                t_build = 1e3 * (time.perf_counter() - t0)

                # fps_avg from accumulated per-frame work (consistent from frame 0; excludes the
                # one-time interactive prompt). duration check uses wall clock since loop start.
                timings = {"grab": t_grab, "depth": t_depth, "track": t_track, "build": t_build}
                cum_ms += sum(timings.values())
                fps_avg = 1000.0 * (n + 1) / max(cum_ms, 1e-6)
                frame = _draw_video_frame(color, masks, stats, timings, n, fps_avg)

                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        video_out, cv2.VideoWriter_fourcc(*"mp4v"), float(video_fps), (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"could not open VideoWriter for {video_out}")
                writer.write(frame)

                n += 1
                if n % 10 == 0:
                    click.echo(f"  frame {n:04d}  {fps_avg:.1f} FPS avg  "
                               f"(grab {t_grab:.0f} depth {t_depth:.0f} track {t_track:.0f} "
                               f"build {t_build:.0f} ms)")
                if (max_frames and n >= max_frames) or \
                        (duration and time.perf_counter() - t_start >= duration):
                    break
        finally:
            if writer is not None:
                writer.release()

    click.echo(f"[debug_pointcloud] wrote {n} frames @ {fps_avg:.1f} FPS avg -> {video_out}")


def _segment(seg_img, sam2_ckpt, sam2_cfg, erode, depth=None, warp=None):
    """Prompt SAM2 on ``seg_img`` -> (H, W) label map aligned to ``depth``.

    If ``warp`` is given (FFS path: seg_img is color, depth is in the IR frame), the color-frame
    masks are reverse-warped into the IR/depth frame before composing. Otherwise seg_img already
    shares the depth frame (RealSense color-aligned) and masks compose directly.
    """
    from diffusion_policy.real_world.pointcloud_segmenter import (
        PointCloudSegmenter, compose_label_map, pick_prompts_interactive,
        warp_masks_to_depth_frame)
    prompts = pick_prompts_interactive(seg_img)
    if not prompts:
        click.echo("  [seg] no prompts given -> geometry-only")
        return None
    seg = PointCloudSegmenter(sam2_ckpt, sam2_cfg)
    if warp is None:
        return seg.label_map(seg_img, prompts, erode=erode)
    assert depth is not None, "warp requires the depth map to reproject masks into"
    masks = warp_masks_to_depth_frame(
        seg.masks(seg_img, prompts), depth, warp["K_ir"], warp["K_color"], warp["T_ir_color"])
    return compose_label_map(masks, depth.shape, erode=erode)


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
@click.option("--video", is_flag=True,
              help="Real-time test: hold camera open, prompt SAM2 once, track + rebuild per frame, save MP4.")
@click.option("--video-out", default="/tmp/seg_stream.mp4", help="Output MP4 path (--video).")
@click.option("--video-fps", type=float, default=15.0, help="Playback fps of the saved MP4 (--video).")
@click.option("--duration", type=float, default=20.0, help="Seconds to stream (--video); 0 = no limit.")
@click.option("--max-frames", type=int, default=0, help="Stop after N frames (--video); 0 = no limit.")
def main(demo, from_file, live, serial, resolution, depth_source, ffs_mock, extrinsic,
         joints, depth_scale, crop, seg, sam2_ckpt, sam2_cfg, erode, no_vis, save_path,
         video, video_out, video_fps, duration, max_frames):
    label_map, seg_img = None, None

    if video:
        if depth_source not in ("realsense", "ffs"):
            raise click.UsageError("--video needs --depth-source realsense or ffs")
        assert joints is not None, "--joints required for --video (EE pose via FK)"
        if not (duration or max_frames):
            raise click.UsageError("--video needs --duration or --max-frames (else it never stops)")
        T_cam_base = np.load(extrinsic)
        ee_pos, ee_quat = _ee_pose_from_joints(_load_array(joints))
        crop_lo, crop_hi = (crop[:3], crop[3:]) if crop else (None, None)
        _run_video(depth_source, serial, resolution, ffs_mock, T_cam_base, ee_pos, ee_quat,
                   sam2_ckpt, sam2_cfg, erode, crop_lo, crop_hi, video_out, video_fps,
                   duration, max_frames)
        return

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
        warp = None
        if depth_source == "ffs":
            seg_img, depth, K, depth_scale, warp = _capture_ffs(serial, resolution, ffs_mock)
        else:
            seg_img, depth, K, depth_scale = _capture_realsense(serial, resolution)
        if seg:
            label_map = _segment(seg_img, sam2_ckpt, sam2_cfg, erode, depth=depth, warp=warp)
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
