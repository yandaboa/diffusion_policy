"""Real-robot eval for the BC PointNet (segmented point-cloud) policy.

Point-cloud sibling of ``eval_real_robot.py`` / ``eval_real_robot_depth.py``. Instead of RGB/
depth images, the policy observation is the EE-frame segmented cloud (1024, 4) = xyz + seg
label, produced live by ``RealEnv.get_obs_pc`` (front-D455 stereo -> FFS metric depth -> SAM2
streaming masks -> ``build_cloud_torch``; the same pipeline as ``debug_pointcloud.py --video``).
The front camera is owned by the point-cloud pipeline, so keep it OUT of the env's
MultiRealsense (the side cam is used for recording/vis + keyboard focus).

Inference matches the other eval scripts: ``PointNetPolicy.predict_from_state`` returns the
DENORMALIZED action (the checkpoint's action_mean/std -- the "scaling built into" the policy);
the caller then applies the RelCartesian OSC ``CARTESIAN_SCALE`` and ``apply_delta_pose`` onto
the current EE pose, exactly as in eval_real_robot.py. Single-cloud model -> one inference per
control cycle (no action horizon).

Usage:
(foundstereo)$ python eval_real_robot_pc.py -i <ckpt.ckpt> -o <save_dir> --robot_ip <ip> \
    --front_serial 215122255213 --depth_source ffs \
    --extrinsic scripts/sim2real/perception/calibrations/most_recent_hand_aligned_extrinsic.json

The --extrinsic accepts the hand-aligned camera->base .json (default, written by
pc_overlay_align.py) or a legacy (4,4) .npy.

You click the robot/peg/hole once on the first frame (one window per class), then the SAM2
tracker follows them. Controls (click the OpenCV window first):
  'c' start is implicit (policy runs immediately); 's' stop, 'r' reset+restart, 'g' open
  gripper macro, 'q'/Ctrl-C exit. On auto-termination (EE z>--z_terminate or timeout) you
  label the episode 's'=success / 'f'=fail (-> <save_dir>/eval_results.json).

Gripper: the raw policy gripper channel is passed through by sign (>0 -> open, <0 -> close),
matching eval_real_robot.py -- no thresholding.

⚠ One value to confirm on the real machine (see POINTCLOUD_EVAL.md):
  * --cartesian_scale must be the DATA-COLLECTION OSC scale (eval cfg uses
    0.01,0.01,0.002,0.02,0.02,0.2). If demos used a different scale, pass it explicitly.
"""

# %%
import time
import json
import pathlib
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import torch

from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.pointnet_policy import PointNetPolicy
from diffusion_policy.real_world.pointcloud_builder import SEG_LABELS
from diffusion_policy.real_world.ur5e_kinematics import (
    get_ee_pose, quat_to_axis_angle, apply_delta_pose)

# Per-class overlay colors for the vis/video (BGR).
SEG_OVERLAY_BGR = {"robot": (200, 200, 200), "peg": (60, 60, 230), "hole": (70, 200, 70)}


def _overlay(color_rgb, masks):
    """Front color frame (RGB) + per-class SAM2 masks (device bool tensors) -> BGR for cv2."""
    bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    for name, m in masks.items():
        mm = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
        col = np.array(SEG_OVERLAY_BGR.get(name, (0, 255, 255)), np.float32)
        bgr[mm] = (0.5 * bgr[mm] + 0.5 * col).astype(np.uint8)
    return bgr


def _depth_overlay(color_rgb, depth, depth_scale, K, warp, vis_range, alpha=0.6):
    """Depth points splatted onto the RGB frame, colored by distance -- a coverage sanity check.

    RealSense (warp=None): depth is color-aligned, painted per-pixel. FFS (warp given): depth lives
    in the left-IR frame, so each valid depth pixel is back-projected and forward-warped into the
    color frame with the SAME K_ir/K_color/T_ir_color used for the masks, then splatted (painter's
    order, near wins). Prints the % of color pixels that received a depth sample.
    """
    bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    z = np.asarray(depth, np.float32) / float(depth_scale)   # -> metres
    valid = np.isfinite(z) & (z > 0)

    if warp is None:
        if z.shape != (H, W):
            z = cv2.resize(z, (W, H), interpolation=cv2.INTER_NEAREST)
            valid = np.isfinite(z) & (z > 0)
        vu = np.argwhere(valid)
        vc, uc, zz = vu[:, 0], vu[:, 1], z[valid]
    else:
        hD, wD = z.shape
        vv, uu = np.mgrid[0:hD, 0:wD].astype(np.float32)
        Kd = np.asarray(K, np.float32)
        x = (uu - Kd[0, 2]) / Kd[0, 0] * z
        y = (vv - Kd[1, 2]) / Kd[1, 1] * z
        pts = np.stack([x, y, z], -1)
        T = np.asarray(warp["T_ir_color"], np.float32)
        Kc = np.asarray(warp["K_color"], np.float32)
        pcam = pts @ T[:3, :3].T + T[:3, 3]
        zc = pcam[..., 2]
        safe = valid & (zc > 0)
        zc_safe = np.where(safe, zc, 1.0)
        uc = np.round(pcam[..., 0] / zc_safe * Kc[0, 0] + Kc[0, 2]).astype(np.int32)
        vc = np.round(pcam[..., 1] / zc_safe * Kc[1, 1] + Kc[1, 2]).astype(np.int32)
        inb = safe & (uc >= 0) & (uc < W) & (vc >= 0) & (vc < H)
        uc, vc, zz = uc[inb], vc[inb], zc[inb]

    out = bgr.copy()
    if len(zz):
        lo, hi = vis_range
        t = np.clip((zz - lo) / max(hi - lo, 1e-6), 0, 1)
        cols = cv2.applyColorMap((t * 255).astype(np.uint8).reshape(-1, 1),
                                 cv2.COLORMAP_TURBO).reshape(-1, 3)
        order = np.argsort(-zz)                       # far first -> near overwrites
        layer = np.zeros_like(bgr)
        hit = np.zeros((H, W), np.uint8)
        layer[vc[order], uc[order]] = cols[order]
        hit[vc, uc] = 1
        if warp is not None:                          # tiny dilation so sparse splats read
            k = np.ones((2, 2), np.uint8)
            layer, hit = cv2.dilate(layer, k), cv2.dilate(hit, k)
        m = hit.astype(bool)
        out[m] = (alpha * layer[m] + (1 - alpha) * bgr[m]).astype(np.uint8)
        cov = 100.0 * int(m.sum()) / (H * W)
    else:
        cov = 0.0
    cv2.putText(out, f"depth coverage {cov:4.1f}%   range [{vis_range[0]:.1f},{vis_range[1]:.1f}] m",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# Per-class colors for the perception-test PLY sanity artifact (RGB; background gray).
SEG_PLY_RGB = {"robot": (200, 200, 200), "peg": (230, 60, 60), "hole": (70, 200, 70)}


def _write_ply(path, pts, labels, colors=None):
    """Write a binary little-endian colored PLY (per-class colors unless real RGB is given)."""
    n = pts.shape[0]
    if colors is not None:
        rgb = np.ascontiguousarray(colors[:, :3]).astype(np.uint8)
    else:
        rgb = np.full((n, 3), 120, np.uint8)  # background / unlabeled -> gray
        for name, val in SEG_LABELS.items():
            rgb[labels == val] = np.array(SEG_PLY_RGB.get(name, (255, 255, 0)), np.uint8)
    vtx = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vtx['x'], vtx['y'], vtx['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    vtx['red'], vtx['green'], vtx['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vtx.tobytes())


def _save_perception_cloud(out_dir, data, depth_source, resolution, extrinsic=None, frame='base'):
    """Persist one un-downsampled segmented cloud for sim-vs-real comparison.

    frame='base' (default) applies `extrinsic` (camera->base (4,4)) so the points land in
    the robot base frame -- the frame the corrected hand-aligned extrinsic maps into, and
    the one the sim cloud lives in. frame='camera' keeps the raw camera-frame points.
    """
    pts, labs, cols = data['points'], data['labels'], data['colors']
    if frame == 'base':
        if extrinsic is None:
            raise ValueError("frame='base' needs an extrinsic (camera->base).")
        pts = (pts.astype(np.float64) @ np.asarray(extrinsic)[:3, :3].T
               + np.asarray(extrinsic)[:3, 3]).astype(np.float32)
    npz_path = out_dir / 'perception_test.npz'
    save_kwargs = dict(
        points=pts, labels=labs,
        K=np.asarray(data['K'], np.float64),
        depth_scale=np.float32(data['depth_scale']),
        depth_source=str(depth_source),
        resolution=np.asarray(resolution, np.int64),
        frame=str(frame),
        extrinsic=np.asarray(extrinsic, np.float64) if extrinsic is not None else np.full((4, 4), np.nan),
        seg_label_names=np.array(list(SEG_LABELS.keys())),
        seg_label_values=np.array(list(SEG_LABELS.values()), np.float32),
        timestamp=np.float64(data['timestamp']),
    )
    if cols is not None:
        save_kwargs['colors'] = cols
    np.savez(npz_path, **save_kwargs)

    import imageio
    imageio.imwrite(str(out_dir / 'perception_test_color.png'), data['color'])
    _write_ply(out_dir / 'perception_test.ply', pts, labs, cols)

    finite = np.isfinite(labs)
    print(f"[perception_test] saved {pts.shape[0]} pts "
          f"({int(finite.sum())} segmented) in {frame.upper()} frame -> {npz_path}")
    for name, val in SEG_LABELS.items():
        print(f"    {name:6s} (label {val:+.0f}): {int((labs == val).sum())} pts")
    print(f"    {'bg':6s} (NaN)      : {int((~finite).sum())} pts")
    print(f"    color frame -> {out_dir / 'perception_test_color.png'}")
    print(f"    colored PLY -> {out_dir / 'perception_test.ply'}")


def load_extrinsic(path):
    """Load a (4,4) camera->base extrinsic from a hand-aligned .json or a legacy .npy.

    The hand-aligned JSON (written by pc_overlay_align.py, e.g.
    perception/calibrations/most_recent_hand_aligned_extrinsic.json) stores the full
    camera->sim-base transform under 'T_total_cam_simbase'. A raw ChArUco calibration
    JSON ('extrinsics_raw', list-or-dict) is also accepted as a fallback. Anything else
    is treated as a .npy holding a (4,4) array.
    """
    if str(path).endswith('.json'):
        d = json.load(open(path))
        if isinstance(d, list):              # ChArUco calib is a list of cameras
            d = d[0]
        if 'T_total_cam_simbase' in d:       # hand-aligned (preferred)
            T = np.asarray(d['T_total_cam_simbase'], dtype=np.float64)
        elif 'extrinsics_raw' in d:          # raw ChArUco calib fallback
            T = np.asarray(d['extrinsics_raw'], dtype=np.float64)
        else:
            raise ValueError(
                f"{path}: no 'T_total_cam_simbase' or 'extrinsics_raw' key in JSON")
    else:
        T = np.asarray(np.load(path), dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{path}: extrinsic must be (4,4), got {T.shape}")
    return T


@click.command()
@click.option('--input', '-i', default=None,
              help='BC PointNet policy: eager Lightning .ckpt, or JIT .pt from convert_bc_to_jit.py. '
                   'Required unless --perception_test (which loads no policy).')
@click.option('--policy_format', type=click.Choice(['auto', 'jit', 'eager']), default='auto',
              help="Policy artifact format. 'auto' detects JIT via the <path>.meta.json sidecar.")
@click.option('--output', '-o', required=True, help='Directory to save recording / results.')
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.1.10")
@click.option('--front_serial', default='215122255213', help='Front D455 serial (point-cloud cam).')
@click.option('--extrinsic',
              default='scripts/sim2real/perception/calibrations/most_recent_hand_aligned_extrinsic.json',
              help="(4,4) camera->base extrinsic. Hand-aligned .json (pc_overlay_align.py, "
                   "reads 'T_total_cam_simbase') or a legacy (4,4) .npy. For --depth_source orbbec "
                   "this is the Orbbec color frame -> base transform.")
@click.option('--depth_source', type=click.Choice(['ffs', 'realsense', 'orbbec']), default='orbbec',
              help="Cloud source. 'ffs'/'realsense' use the front D455 (stereo+FoundationStereo / "
                   "hardware depth) with the D455 color for SAM2. 'orbbec' uses the Femto Bolt's "
                   "RGB for SAM2 AND its PointCloudFilter cloud for geometry (one camera for both).")
@click.option('--ffs_mock', is_flag=True, default=False, help='FFS ramp depth (plumbing test).')
@click.option('--input_res', default='1280x720', type=str,
              help='Front capture resolution WxH. Orbbec defaults to 1280x960 (4:3 native) when '
                   'left at the D455 default.')
@click.option('--sam2_ckpt', default='orbbec/weights/sam2/sam2.1_hiera_base_plus.pt')
@click.option('--sam2_cfg', default='configs/sam2.1/sam2.1_hiera_b+.yaml')
@click.option('--erode', type=int, default=3, help='Erode each SAM2 mask by N px.')
@click.option('--flying_pixel_mm', default=5.0, type=float,
              help='Orbbec-only flying-pixel filter: drop points whose 3x3 local depth range '
                   'exceeds this many mm (ToF mixed pixels streaking between fg and bg). '
                   '0 disables. No effect for --depth_source ffs/realsense.')
@click.option('--crop', nargs=6, type=float, default=None,
              help='EE-frame AABB xmin ymin zmin xmax ymax zmax (m); default no crop.')
@click.option('--point_config', type=click.Choice(['default', 'peg_hole']), default='default',
              help="TEMPORARY point-cloud composition selector (until the ckpt config drives it). "
                   "'default': robot/peg/hole budget 512/256/256, 4D points (xyz + seg label). "
                   "'peg_hole': 50/50 peg/hole, NO robot, and the label channel is dropped so the "
                   "policy sees 3D (xyz) points -- for the occobject-only policy (order-agnostic "
                   "encoder). The exact split is num_points/2 each, from the checkpoint's num_points.")
@click.option('--perception_test', is_flag=True, default=False,
              help='Perception/PC quality probe: load NO policy, stand up only real_env + the '
                   'point-cloud pipeline, save ONE full-resolution cloud in the CAMERA frame (no '
                   'downsampling / crop / budget), then exit. Full raw cloud by default; add '
                   '--perception_segment to also run SAM2 and store per-point labels.')
@click.option('--perception_warmup', default=30, type=int,
              help='With --perception_test, discard this many frames before saving the cloud '
                   '(lets exposure/depth settle); 0 saves the very first grab.')
@click.option('--perception_segment', is_flag=True, default=False,
              help='With --perception_test, also run SAM2 (interactive clicks) and store per-point '
                   'seg labels. Default OFF: save the FULL raw cloud (geometry+color, no clicks).')
@click.option('--perception_frame', type=click.Choice(['base', 'camera']), default='base',
              help='Frame the perception_test cloud is saved in. base (default): apply --extrinsic '
                   'to put points in the robot base frame. camera: leave points in the camera frame.')
@click.option('--perception_close_gripper', is_flag=True, default=False,
              help='With --perception_test, keep commanding the gripper CLOSED (holding the current '
                   'arm pose, so the arm does not move) throughout the warmup + grab, so the probed '
                   'cloud matches the eval scene (e.g. peg held in a closed gripper).')
@click.option('--init_joints', '-j', is_flag=True, default=False)
@click.option('--frequency', '-f', default=10, type=float, help='Control frequency (Hz).')
@click.option('--max_duration', '-md', default=60, type=float, help='Max episode seconds.')
@click.option('--z_terminate', default=0.4, type=float,
              help='Auto-terminate when EE z (base frame, m) exceeds this.')
@click.option('--action_noise', default=0.0, type=float,
              help='Std of Gaussian noise on the raw arm action (pre-scale).')
@click.option('--cartesian_scale', default='0.01,0.01,0.002,0.02,0.02,0.2', type=str,
              help='6 comma-sep RelCartesian OSC scales applied to the raw arm delta.')
@click.option('--sample_actions/--mean_actions', default=False,
              help='Sample actions from the policy Gaussian head instead of taking the mean '
                   '(stochastic "expert sampling"). Needs an eager .ckpt with predict_std=True; '
                   'JIT exports discard the std head and will error.')
@click.option('--sample_temperature', default=1.0, type=float,
              help='Std scale when --sample_actions (0=mean, 1=trained std, >1 = more exploration).')
@click.option('--gripper_proprio/--no_gripper_proprio', default=False,
              help='Include the 6 Robotiq mimic gripper joints in proprio (18-d vs 12-d). '
                   'Default off: proprio = [arm_joint_pos(6), ee_pose(6)] for *_no_gripper models.')
@click.option('--save_video', is_flag=True, default=False,
              help='Save the front color + seg overlay as an MP4 (+ a depth-coverage MP4).')
@click.option('--depth_vis_range', default='0.2,1.2', type=str,
              help='min,max metres for the depth-coverage overlay colormap.')
@click.option('--device', default='cuda', type=str, help='Torch device for policy + cloud build.')
def main(input, policy_format, output, robot_ip, front_serial, extrinsic, depth_source, ffs_mock,
         input_res, sam2_ckpt, sam2_cfg, erode, flying_pixel_mm, crop, point_config, perception_test, perception_warmup,
         perception_segment, perception_frame, perception_close_gripper, init_joints, frequency, max_duration, z_terminate, action_noise,
         cartesian_scale,
         sample_actions, sample_temperature, gripper_proprio,
         save_video, depth_vis_range, device):
    # Orbbec's native 4:3 color is 1280x960; default to it (instead of the D455's 16:9 1280x720)
    # so the SAM2 RGB + PointCloudFilter cloud match the 4:3 sim camera. Honor an explicit override.
    if depth_source == 'orbbec' and input_res.lower() == '1280x720':
        input_res = '1280x960'
        print(f"[orbbec] using 4:3 native color {input_res} (override --input_res to change)")
    capture_w, capture_h = (int(x) for x in input_res.lower().split('x'))
    capture_resolution = (capture_w, capture_h)
    CARTESIAN_SCALE = np.array([float(x) for x in cartesian_scale.split(',')], np.float64)
    assert CARTESIAN_SCALE.shape == (6,), "--cartesian_scale must have 6 values"
    crop_lo, crop_hi = (crop[:3], crop[3:]) if crop else (None, None)
    depth_vis_range = tuple(float(x) for x in depth_vis_range.split(','))
    assert len(depth_vis_range) == 2, "--depth_vis_range must be 'min,max'"
    out_dir = pathlib.Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- episode success/fail labels (-> eval_results.json) --------------------------------
    episode_results = []

    def prompt_success_fail(episode_id, ee_z):
        print(f"Episode {episode_id} terminated (EE z={ee_z:.3f}). 's'=SUCCESS, 'f'=FAIL ...")
        result = None
        while result is None:
            canvas = np.zeros((200, 700, 3), dtype=np.uint8)
            cv2.putText(canvas, f"EPISODE {episode_id} TERMINATED (z={ee_z:.3f})", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, "'s' = SUCCESS", (15, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(canvas, "'f' = FAIL", (15, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Policy Control', canvas)
            key = cv2.waitKey(50) & 0xFF
            if key == ord('s'):
                result = True
            elif key == ord('f'):
                result = False
        print(f"  -> {'SUCCESS' if result else 'FAIL'}")
        episode_results.append({'episode': int(episode_id),
                                'result': 'success' if result else 'fail',
                                'z': float(ee_z), 't': time.time()})
        n_succ = sum(r['result'] == 'success' for r in episode_results)
        n_tot = len(episode_results)
        with open(out_dir / 'eval_results.json', 'w') as f:
            json.dump({'n_episodes': n_tot, 'n_success': n_succ,
                       'success_rate': n_succ / n_tot, 'episodes': episode_results}, f, indent=2)
        print(f"  [{n_succ}/{n_tot} success] saved to {out_dir / 'eval_results.json'}")
        return result

    # ---- load policy (skipped entirely for the perception/PC quality probe) ---------------
    if not torch.cuda.is_available() and device.startswith('cuda'):
        device = 'cpu'
    policy = None
    if perception_test:
        print("[perception_test] policy load skipped -- only real_env + PC pipeline will run.")
    else:
        if input is None:
            raise click.UsageError("--input/-i is required unless --perception_test is set.")
        jit = {'auto': None, 'jit': True, 'eager': False}[policy_format]
        print(f"Loading BC PointNet from {input} on {device} (format={policy_format})")
        policy = PointNetPolicy(input, device=device, jit=jit)
        print(f"  jit={policy.jit} point_dim={policy.point_dim} num_points={policy.num_points} "
              f"proprio_dim={policy.proprio_dim} action_dim={policy.action_dim}")
        assert policy.action_dim == 7, \
            f"expected 7-d action (6 OSC dpose + gripper), got {policy.action_dim}"

        # --gripper_proprio is authoritative over the (often absent) proprio_dim metadata: it picks
        # the 18-d (arm+gripper mimic joints+ee) vs 12-d (arm+ee, *_no_gripper) layout that
        # predict_from_state assembles. Sync policy.proprio_dim so predict()'s assert matches.
        policy.include_gripper_joints = gripper_proprio
        expected_proprio_dim = 18 if gripper_proprio else 12
        if policy.proprio_dim != expected_proprio_dim:
            print(f"[gripper_proprio={gripper_proprio}] WARNING: checkpoint proprio_dim="
                  f"{policy.proprio_dim} disagrees; overriding -> {expected_proprio_dim}. "
                  f"Make sure this matches how the model was trained.")
            policy.proprio_dim = expected_proprio_dim
        print(f"Proprio: {expected_proprio_dim}-d "
              f"({'arm+gripper joints+ee' if gripper_proprio else 'arm+ee (no gripper joints)'})")

        # Stochastic "expert sampling": draw actions from the policy Gaussian head instead of the
        # mean. Needs a predict_std checkpoint -- an eager .ckpt, or a JIT re-exported to return
        # (mean, std). A mean-only JIT has no std head; predict() raises a clear error on the first
        # (warmup) inference below, so a bad combo fails before the robot moves rather than mid-rollout.
        policy.sample = sample_actions
        policy.sample_temperature = sample_temperature
        if sample_actions:
            print(f"Action sampling: ON (temperature={sample_temperature}) -- sampling from policy "
                  f"Gaussian head ({'JIT (mean,std)' if policy.jit else 'eager predict_std'})")
        else:
            print("Action sampling: OFF (deterministic mean action)")
        print(f"Cartesian OSC scale: {CARTESIAN_SCALE}")

    # ---- point-cloud composition (TEMPORARY --point_config; eventually driven by ckpt cfg) ----
    # 'default' -> budget=None (robot/peg/hole DEFAULT_BUDGET), 4D points (xyz + seg label).
    # 'peg_hole' -> 50/50 peg/hole (no robot); prompt only peg/hole so SAM2 skips the robot, and
    # feed 3D points (drop the label channel). NOTE: we can't trust policy.point_dim here -- JIT
    # .pt checkpoints without point_dim in their .meta.json default it to 4, so drive the channel
    # count off --point_config instead. build_cloud always emits [x, y, z, label], so slicing the
    # leading ``point_channels`` columns gives xyz (3) or xyz+label (4).
    pc_budget = None
    prompt_classes = None
    point_channels = 4
    if point_config == 'peg_hole':
        from diffusion_policy.real_world.pointcloud_builder import SEG_LABELS
        n_pts = int(policy.num_points) if (policy is not None and policy.num_points) else 1024
        n_peg = n_pts // 2
        pc_budget = {SEG_LABELS['peg']: n_peg, SEG_LABELS['hole']: n_pts - n_peg}
        prompt_classes = ('peg', 'hole')
        point_channels = 3
        print(f"[point_config=peg_hole] budget peg/hole = {n_peg}/{n_pts - n_peg} (no robot); "
              f"feeding 3D points (xyz, label dropped).")
    # --point_config is authoritative over the (often absent) JIT point_dim metadata, so sync the
    # policy's expected channel count to it -- otherwise predict()'s point_dim assert rejects the
    # sliced cloud (JIT .pt without point_dim in .meta.json defaults point_dim to 4).
    if policy is not None and policy.point_dim != point_channels:
        print(f"[point_config] overriding policy.point_dim {policy.point_dim} -> {point_channels}")
        policy.point_dim = point_channels

    # ---- pick the recording/vis cameras (front cam is owned by the PC pipeline) -----------
    import pyrealsense2 as _rs
    connected = {d.get_info(_rs.camera_info.serial_number)
                 for d in _rs.context().devices
                 if d.get_info(_rs.camera_info.name).lower() != 'platform camera'}
    _all_cameras = [
        ('832112070487', json.load(open("diffusion_policy/real_world/realsense_config/435_side.json"))),
    ]
    active = [(s, c) for s, c in _all_cameras if s in connected and s != front_serial]
    camera_serial_numbers = [s for s, _ in active]
    configs = [c for _, c in active]
    print(f"Recording/vis cameras (excl. front {front_serial}): {camera_serial_numbers}")

    dt = 1 / frequency
    video_writer = None
    depth_video_writer = None

    with SharedMemoryManager() as shm_manager:
        with RealEnv(
                output_dir=output,
                robot_ip=robot_ip,
                frequency=frequency,
                n_obs_steps=1,
                obs_float32=True,
                init_joints=init_joints,
                enable_multi_cam_vis=len(camera_serial_numbers) > 0,
                record_raw_video=True,
                rolling_action_buffer=True,
                action_mode='cartesian',
                arm_action_dim=6,
                camera_serial_numbers=camera_serial_numbers,
                camera_configs=configs,
                video_capture_resolution=capture_resolution,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager) as env:

            cv2.setNumThreads(1)
            print("Waiting for hardware...")
            time.sleep(5.0)

            # Stand up the point-cloud pipeline (opens the front cam). Segmentation (SAM2 prompts)
            # runs for the policy obs path, and for the perception probe only if --perception_segment.
            do_segment = (not perception_test) or perception_segment
            extrinsic_mat = load_extrinsic(extrinsic)
            print(f"[eval] camera->base extrinsic loaded from {extrinsic}\n{extrinsic_mat.round(4)}")
            env.setup_pointcloud(
                front_serial=front_serial, extrinsic=extrinsic_mat, depth_source=depth_source,
                resolution=capture_resolution, sam2_ckpt=sam2_ckpt, sam2_cfg=sam2_cfg,
                erode=erode, crop_lo=crop_lo, crop_hi=crop_hi, budget=pc_budget,
                prompt_classes=prompt_classes, ffs_mock=ffs_mock,
                segment=do_segment,
                flying_pixel_mm=flying_pixel_mm if flying_pixel_mm > 0 else None,
                device=device)

            # ---- perception/PC quality probe: grab ONE camera-frame cloud, save, exit ----
            if perception_test:
                # Optionally hold the gripper CLOSED during the probe. The gripper state is coupled
                # to a control command, so we target the CURRENT joints (which the controller already
                # holds by default) with close_gripper=True -- closes the gripper without moving the arm.
                def _hold_gripper_closed():
                    if not perception_close_gripper:
                        return
                    curr_q = np.asarray(env.get_robot_state()['ActualQ'], np.float64)[:6]
                    env.robot.joint_torque_control(target_joints=curr_q, close_gripper=True)

                if perception_close_gripper:
                    print("[perception_test] holding gripper CLOSED (arm pose unchanged).")
                kind = "segmented" if perception_segment else "FULL (un-segmented)"
                for i in range(perception_warmup):
                    _hold_gripper_closed()
                    env.grab_segmented_cloud_camera_frame()  # settle exposure/depth
                    print(f"[perception_test] warmup frame {i + 1}/{perception_warmup}")
                _hold_gripper_closed()
                print(f"[perception_test] grabbing one full-resolution {kind} cloud...")
                data = env.grab_segmented_cloud_camera_frame()
                _save_perception_cloud(out_dir, data, depth_source, capture_resolution,
                                       extrinsic=extrinsic_mat, frame=perception_frame)
                print("[perception_test] done.")
                return

            print("Warming up policy inference...")
            obs = env.get_obs_pc()
            _ = policy.predict_from_state(
                obs['point_cloud'][:, :point_channels], obs['arm_joint_pos'],
                float(obs['gripper_pos']))
            print('Ready!')
            time.sleep(1.0)

            # Per-episode overlay videos, mirroring RealEnv's out_dir/videos/<episode_id>/ layout
            # (start_episode already created that folder). Opened at each episode start and
            # finalized at each end, so successive episodes/runs don't clobber a single file.
            def open_video_writers():
                nonlocal video_writer, depth_video_writer
                if not save_video:
                    return
                import imageio
                episode_id = env.replay_buffer.n_episodes
                ep_dir = out_dir / 'videos' / str(episode_id)
                ep_dir.mkdir(parents=True, exist_ok=True)
                full_path = ep_dir / 'policy_pc_full.mp4'
                depth_path = ep_dir / 'policy_pc_depth.mp4'
                video_writer = imageio.get_writer(str(full_path), format='FFMPEG', fps=int(frequency),
                                                  codec='libx264', output_params=['-crf', '23'])
                depth_video_writer = imageio.get_writer(str(depth_path), format='FFMPEG',
                                                        fps=int(frequency), codec='libx264',
                                                        output_params=['-crf', '23'])
                print(f"[ep {episode_id}] overlay video -> {full_path}")
                print(f"[ep {episode_id}] depth-coverage video -> {depth_path}")

            def close_video_writers():
                nonlocal video_writer, depth_video_writer
                if video_writer is not None:
                    video_writer.close()
                if depth_video_writer is not None:
                    depth_video_writer.close()
                video_writer = depth_video_writer = None

            gripper_open_steps_remaining = 0
            GRIPPER_OPEN_DURATION = 5
            STUCK_WINDOW_S = 2.0
            STUCK_JOINT_THRESHOLD_RAD = 0.002
            STUCK_GRIPPER_OPEN_STEPS = int(frequency)
            stuck_buffer = []

            while True:
                try:
                    print('Resetting robot to initial position...')
                    env.robot.reset_to_initial_position()
                    time.sleep(5.0)
                    print('Reset complete.')

                    gripper_open_steps_remaining = 0
                    stuck_buffer.clear()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    open_video_writers()
                    precise_wait(eval_t_start, time_func=time.time)
                    print("Started!")
                    iter_idx = 0

                    while True:
                        t_cycle_end = t_start + (iter_idx + 1) * dt

                        # ---- observation: segmented EE-frame cloud + robot state ----
                        obs = env.get_obs_pc()
                        obs_timestamp = obs['timestamp']
                        points = obs['point_cloud']           # (num_points, 4) = xyz + seg label
                        arm_jp = obs['arm_joint_pos']         # (6,)
                        grip_raw = float(obs['gripper_pos'])

                        # ---- inference: denormalized 7-d action (scaling baked into policy) ----
                        # Feed only the channels the policy expects: 4D (xyz+label) by default, or
                        # 3D (xyz) for --point_config peg_hole (label dropped). See point_channels.
                        raw_action = policy.predict_from_state(
                            points[:, :point_channels], arm_jp, grip_raw)
                        raw_arm = raw_action[:6].astype(np.float64)
                        if action_noise > 0:
                            raw_arm = raw_arm + np.random.randn(6) * action_noise
                        raw_gripper = float(raw_action[6])

                        # gripper open macro (overrides policy gripper)
                        if gripper_open_steps_remaining > 0:
                            gripper_val = 1.0  # >0 = open
                            gripper_open_steps_remaining -= 1
                        else:
                            # Pass the raw policy gripper channel through by sign, matching
                            # eval_real_robot.py (no thresholding). exec_actions convention:
                            # gripper col > 0 -> open, < 0 -> closed.
                            gripper_val = raw_gripper
                        close = gripper_val < 0
                        gripper_cmd = np.array([[gripper_val]], np.float32)

                        # ---- RelCartesian OSC: scale delta -> absolute EE target ----
                        scaled_delta = raw_arm * CARTESIAN_SCALE
                        obs_pos, obs_quat = get_ee_pose(arm_jp)
                        tgt_pos, tgt_quat = apply_delta_pose(obs_pos, obs_quat, scaled_delta)
                        tgt_aa = quat_to_axis_angle(tgt_quat)
                        abs_target = np.concatenate([tgt_pos, tgt_aa])[None]      # (1, 6)
                        target_actions = np.concatenate([abs_target, gripper_cmd], axis=1)  # (1,7)
                        # raw (pre-scale) arm + gripper, stored so last_arm_action obs is raw
                        raw_actions = np.concatenate([raw_arm[None], gripper_cmd], axis=1)

                        # ---- stuck detection: open gripper if the arm hasn't moved ----
                        if gripper_open_steps_remaining == 0:
                            t_now = time.monotonic()
                            stuck_buffer.append((t_now, arm_jp.copy()))
                            while stuck_buffer and (t_now - stuck_buffer[0][0]) > STUCK_WINDOW_S:
                                stuck_buffer.pop(0)
                            if len(stuck_buffer) >= STUCK_WINDOW_S * frequency:
                                jps = np.array([b[1] for b in stuck_buffer])
                                if np.max(jps.max(0) - jps.min(0)) < STUCK_JOINT_THRESHOLD_RAD:
                                    gripper_open_steps_remaining = STUCK_GRIPPER_OPEN_STEPS
                                    stuck_buffer.clear()
                                    print("[Stuck] no movement 2s -> opening gripper")

                        # ---- timing + execute ----
                        action_timestamp = float(obs_timestamp + dt)
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        if action_timestamp <= curr_time + action_exec_latency:
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + next_step_idx * dt
                        
                        print(raw_actions)
                        env.exec_actions(
                            actions=target_actions,
                            timestamps=np.array([action_timestamp]),
                            obs_actions=raw_actions)

                        # ---- vis / video (front color + seg overlay + HUD) ----
                        episode_id = env.replay_buffer.n_episodes
                        vis = _overlay(obs['color'], obs['masks'])
                        st = obs['cloud_stats']
                        hud = [f"ep {episode_id}  t {time.monotonic() - t_start:4.1f}s  "
                               f"z {float(obs_pos[2]):.3f}  grip {'C' if close else 'O'}"]
                        for nm in ('robot', 'peg', 'hole'):
                            lab = SEG_LABELS[nm]
                            if lab in st.per_class_realized:
                                hud.append(f"{nm}:{st.per_class_available.get(lab, 0)}"
                                           f"/{st.per_class_realized.get(lab, 0)}")
                        cv2.putText(vis, "  ".join(hud), (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (60, 255, 60), 1, cv2.LINE_AA)
                        cv2.imshow('Policy Control', vis)
                        if video_writer is not None:
                            video_writer.append_data(vis[..., ::-1])  # BGR->RGB

                        # ---- second window: depth points splatted on RGB (coverage check) ----
                        dvis = _depth_overlay(obs['color'], obs['depth'], obs['depth_scale'],
                                              obs['K'], obs['warp'], depth_vis_range)
                        cv2.imshow('Depth Coverage', dvis)
                        if depth_video_writer is not None:
                            depth_video_writer.append_data(dvis[..., ::-1])  # BGR->RGB

                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('g'):
                            gripper_open_steps_remaining = GRIPPER_OPEN_DURATION
                            print(f"[Gripper macro] opening for {GRIPPER_OPEN_DURATION} steps")
                        elif key_stroke == ord('q'):
                            raise KeyboardInterrupt
                        elif key_stroke == ord('s'):
                            env.end_episode(); close_video_writers(); print('Stopped.'); break
                        elif key_stroke == ord('r'):
                            print('Resetting for new trajectory...')
                            env.end_episode(); close_video_writers()
                            env.robot.reset_to_initial_position(); time.sleep(5.0)
                            stuck_buffer.clear()
                            start_delay = 1.0
                            eval_t_start = time.time() + start_delay
                            t_start = time.monotonic() + start_delay
                            env.start_episode(eval_t_start)
                            open_video_writers()
                            precise_wait(eval_t_start, time_func=time.time)
                            iter_idx = 0
                            print('Reset complete! New trajectory.')
                            continue

                        # ---- auto-termination ----
                        ee_z = float(obs_pos[2])
                        terminate = False
                        if ee_z > z_terminate:
                            terminate = True; print(f'Terminated: EE z={ee_z:.3f} > {z_terminate}')
                        elif time.monotonic() - t_start > max_duration:
                            terminate = True; print('Terminated by timeout!')
                        if terminate:
                            env.end_episode()
                            close_video_writers()
                            prompt_success_fail(episode_id, ee_z)
                            print('Episode terminated; restarting.')
                            break

                        precise_wait(t_cycle_end)
                        iter_idx += 1

                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.end_episode(); close_video_writers()
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback; traceback.print_exc()
                    env.end_episode(); close_video_writers()
                    break

            close_video_writers()  # safety net if we broke out mid-episode
            print("Stopped.")


# %%
if __name__ == '__main__':
    main()
