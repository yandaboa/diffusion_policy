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
    --front_serial 215122255213 --extrinsic calib/front_cam_to_base_simapprox.npy \
    --depth_source ffs

You click the robot/peg/hole once on the first frame (one window per class), then the SAM2
tracker follows them. Controls (click the OpenCV window first):
  'c' start is implicit (policy runs immediately); 's' stop, 'r' reset+restart, 'g' open
  gripper macro, 'q'/Ctrl-C exit. On auto-termination (EE z>--z_terminate or timeout) you
  label the episode 's'=success / 'f'=fail (-> <save_dir>/eval_results.json).

If the checkpoint carries a ``pc_signature`` (UWLab pc_signature.py; JIT ``.meta.json`` or
Lightning hparams), the perception pipeline is configured FROM it: which classes to
SAM2-prompt, the exact per-class point budget, seg channel on/off (3-ch models get an
xyz-only cloud), and the proprio layout (incl. arm-6-only, no gripper joints). Checkpoints
without one fall back to the legacy defaults (DEFAULT_BUDGET, robot/peg/hole, 4-ch cloud).

⚠ Two values to confirm on the real machine (see POINTCLOUD_EVAL.md):
  * --cartesian_scale must be the DATA-COLLECTION OSC scale (eval cfg uses
    0.01,0.01,0.002,0.02,0.02,0.2). If demos used a different scale, pass it explicitly.
  * --gripper_threshold polarity: binary channel >threshold -> close (see --invert_gripper).
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
from diffusion_policy.real_world.pointnet_policy import (
    PointNetPolicy, perception_from_signature)
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


@click.command()
@click.option('--input', '-i', required=True,
              help='BC PointNet policy: eager Lightning .ckpt, or JIT .pt from convert_bc_to_jit.py.')
@click.option('--policy_format', type=click.Choice(['auto', 'jit', 'eager']), default='auto',
              help="Policy artifact format. 'auto' detects JIT via the <path>.meta.json sidecar.")
@click.option('--output', '-o', required=True, help='Directory to save recording / results.')
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.1.10")
@click.option('--front_serial', default='215122255213', help='Front D455 serial (point-cloud cam).')
@click.option('--extrinsic', default='calib/front_cam_to_base_simapprox.npy',
              help='(4,4) front-camera->base npy.')
@click.option('--depth_source', type=click.Choice(['ffs', 'realsense']), default='ffs',
              help="Front depth: 'ffs' (stereo+FoundationStereo) or 'realsense' (hardware).")
@click.option('--ffs_mock', is_flag=True, default=False, help='FFS ramp depth (plumbing test).')
@click.option('--input_res', default='1280x720', type=str, help='Front capture resolution WxH.')
@click.option('--sam2_ckpt', default='orbbec/weights/sam2/sam2.1_hiera_base_plus.pt')
@click.option('--sam2_cfg', default='configs/sam2.1/sam2.1_hiera_b+.yaml')
@click.option('--erode', type=int, default=3, help='Erode each SAM2 mask by N px.')
@click.option('--crop', nargs=6, type=float, default=None,
              help='EE-frame AABB xmin ymin zmin xmax ymax zmax (m); default no crop.')
@click.option('--init_joints', '-j', is_flag=True, default=False)
@click.option('--frequency', '-f', default=10, type=float, help='Control frequency (Hz).')
@click.option('--max_duration', '-md', default=60, type=float, help='Max episode seconds.')
@click.option('--z_terminate', default=0.4, type=float,
              help='Auto-terminate when EE z (base frame, m) exceeds this.')
@click.option('--action_noise', default=0.0, type=float,
              help='Std of Gaussian noise on the raw arm action (pre-scale).')
@click.option('--cartesian_scale', default='0.01,0.01,0.002,0.02,0.02,0.2', type=str,
              help='6 comma-sep RelCartesian OSC scales applied to the raw arm delta.')
@click.option('--gripper_threshold', default=0.5, type=float,
              help='Binary gripper: raw channel > this -> close.')
@click.option('--invert_gripper', is_flag=True, default=False,
              help='Flip gripper polarity (raw channel > threshold -> open).')
@click.option('--save_video', is_flag=True, default=False,
              help='Save the front color + seg overlay as an MP4.')
@click.option('--device', default='cuda', type=str, help='Torch device for policy + cloud build.')
def main(input, policy_format, output, robot_ip, front_serial, extrinsic, depth_source, ffs_mock,
         input_res, sam2_ckpt, sam2_cfg, erode, crop, init_joints, frequency, max_duration,
         z_terminate, action_noise, cartesian_scale, gripper_threshold, invert_gripper,
         save_video, device):
    capture_w, capture_h = (int(x) for x in input_res.lower().split('x'))
    capture_resolution = (capture_w, capture_h)
    CARTESIAN_SCALE = np.array([float(x) for x in cartesian_scale.split(',')], np.float64)
    assert CARTESIAN_SCALE.shape == (6,), "--cartesian_scale must have 6 values"
    crop_lo, crop_hi = (crop[:3], crop[3:]) if crop else (None, None)
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

    # ---- load policy ----------------------------------------------------------------------
    if not torch.cuda.is_available() and device.startswith('cuda'):
        device = 'cpu'
    jit = {'auto': None, 'jit': True, 'eager': False}[policy_format]
    print(f"Loading BC PointNet from {input} on {device} (format={policy_format})")
    policy = PointNetPolicy(input, device=device, jit=jit)
    print(f"  jit={policy.jit} point_dim={policy.point_dim} num_points={policy.num_points} "
          f"proprio_dim={policy.proprio_dim} action_dim={policy.action_dim}")
    assert policy.action_dim == 7, \
        f"expected 7-d action (6 OSC dpose + gripper), got {policy.action_dim}"
    print(f"Cartesian OSC scale: {CARTESIAN_SCALE}")

    # Perception config from the checkpoint's PC observation signature (per-class budgets,
    # which classes to SAM2-prompt, seg-label remap). None -> legacy defaults.
    budget, prompt_classes, label_remap = perception_from_signature(policy.pc_signature, policy)

    def policy_cloud(raw_cloud):
        """Builder cloud (N, 4) -> the exact per-point layout the policy trained on:
        slice off the seg channel for 3-ch (xyz-only) models, remap seg label values if the
        trained convention differs from SEG_LABELS."""
        if policy.point_dim == 3:
            return raw_cloud[:, :3]
        if label_remap:
            lab = raw_cloud[:, 3].copy()
            for src, dst in label_remap.items():
                raw_cloud[:, 3][lab == src] = dst
        return raw_cloud

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

            # Stand up the point-cloud pipeline (opens the front cam, prompts SAM2 once).
            env.setup_pointcloud(
                front_serial=front_serial, extrinsic=extrinsic, depth_source=depth_source,
                resolution=capture_resolution, sam2_ckpt=sam2_ckpt, sam2_cfg=sam2_cfg,
                erode=erode, crop_lo=crop_lo, crop_hi=crop_hi, ffs_mock=ffs_mock, device=device,
                budget=budget, prompt_classes=prompt_classes)

            print("Warming up policy inference...")
            obs = env.get_obs_pc()
            _ = policy.predict_from_state(
                policy_cloud(obs['point_cloud']), obs['arm_joint_pos'], float(obs['gripper_pos']))
            print('Ready!')
            time.sleep(1.0)

            if save_video:
                import imageio
                vid_path = out_dir / 'policy_pc_full.mp4'
                video_writer = imageio.get_writer(str(vid_path), fps=int(frequency),
                                                  codec='libx264', output_params=['-crf', '23'])
                print(f"Saving overlay video to {vid_path}")

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
                    precise_wait(eval_t_start, time_func=time.time)
                    print("Started!")
                    iter_idx = 0

                    while True:
                        t_cycle_end = t_start + (iter_idx + 1) * dt

                        # ---- observation: segmented EE-frame cloud + robot state ----
                        obs = env.get_obs_pc()
                        obs_timestamp = obs['timestamp']
                        points = policy_cloud(obs['point_cloud'])  # (num_points, point_dim)
                        arm_jp = obs['arm_joint_pos']         # (6,)
                        grip_raw = float(obs['gripper_pos'])

                        # ---- inference: denormalized 7-d action (scaling baked into policy) ----
                        raw_action = policy.predict_from_state(points, arm_jp, grip_raw)
                        raw_arm = raw_action[:6].astype(np.float64)
                        if action_noise > 0:
                            raw_arm = raw_arm + np.random.randn(6) * action_noise
                        raw_gripper = float(raw_action[6])

                        # gripper open macro (overrides policy gripper)
                        if gripper_open_steps_remaining > 0:
                            close = False
                            gripper_open_steps_remaining -= 1
                        else:
                            close = raw_gripper > gripper_threshold
                            if invert_gripper:
                                close = not close
                        # exec_actions convention: gripper col < 0 -> closed
                        gripper_cmd = np.array([[-1.0 if close else 1.0]], np.float32)

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

                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('g'):
                            gripper_open_steps_remaining = GRIPPER_OPEN_DURATION
                            print(f"[Gripper macro] opening for {GRIPPER_OPEN_DURATION} steps")
                        elif key_stroke == ord('q'):
                            raise KeyboardInterrupt
                        elif key_stroke == ord('s'):
                            env.end_episode(); print('Stopped.'); break
                        elif key_stroke == ord('r'):
                            print('Resetting for new trajectory...')
                            env.end_episode()
                            env.robot.reset_to_initial_position(); time.sleep(5.0)
                            stuck_buffer.clear()
                            start_delay = 1.0
                            eval_t_start = time.time() + start_delay
                            t_start = time.monotonic() + start_delay
                            env.start_episode(eval_t_start)
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
                            prompt_success_fail(episode_id, ee_z)
                            print('Episode terminated; restarting.')
                            break

                        precise_wait(t_cycle_end)
                        iter_idx += 1

                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.end_episode()
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback; traceback.print_exc()
                    env.end_episode()
                    break

            if video_writer is not None:
                video_writer.close()
            print("Stopped.")


# %%
if __name__ == '__main__':
    main()
