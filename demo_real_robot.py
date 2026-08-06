"""
Mello teleop for UR5e: move the arm with the Mello device and optionally record demos.

  python demo_real_robot.py -o <output_dir> --robot_ip <ur5e_ip>

See README_ur5e.md for UR5e and Mello setup. If the robot stalls,
tune gains with --osc_kp_pos and --osc_kp_rot. Keys: C=start record, S=stop, Q=quit,
Backspace=drop last episode. Use --debug for fixed joint positions (no Mello).

Cameras are configured via the CAMERA_SPECS list below (serials + configs).
DA3 depth inference is off by default; pass --enable_da3 to turn it on.
"""

# %%
import time
from contextlib import nullcontext
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.real_world.mello_teleop import MelloTeleopInterface, DummyMelloTeleopInterface
from diffusion_policy.real_world.da3_depth_client import DA3DepthClient

# ── Depth constants matching depth_dagger_cfg.py ────────────────────────────
_DEPTH_CLIP = (0.01, 2.0)   # metres
_DEPTH_IMG_H, _DEPTH_IMG_W = 224, 224

# ── Camera configuration ────────────────────────────────────────────────────
# Single source of truth for which cameras are used and in what order. RealEnv
# names cameras by index: 3 cameras → front/side/wrist, 2 → side/wrist. To change
# the camera setup, edit this list (order = camera index in RealEnv). Each entry:
#   name   → produces obs key "<name>_rgb"
#   serial → RealSense device serial number
#   config → RealSense advanced-mode JSON in _CONFIG_DIR
_CONFIG_DIR = "diffusion_policy/real_world/realsense_config/"
CAMERA_SPECS = [
    # {'name': 'front', 'serial': '215122255213', 'config': '455_front.json'},
    {'name': 'side',  'serial': '832112070487', 'config': '435_side.json'},
    # {'name': 'wrist', 'serial': '746112060198', 'config': '415_wrist.json'},
]



def _process_depth(depth_u16: np.ndarray, depth_scale: float) -> np.ndarray:
    """uint16 depth frame → float32 [0,1] at 224×224, matching sim preprocessing."""
    d_min, d_max = _DEPTH_CLIP
    depth_m = depth_u16.astype(np.float32) * depth_scale
    depth_m[depth_m == 0.0] = d_max          # no-return pixels → max range
    np.clip(depth_m, d_min, d_max, out=depth_m)
    depth_norm = (depth_m - d_min) / (d_max - d_min)
    return cv2.resize(depth_norm, (_DEPTH_IMG_W, _DEPTH_IMG_H), interpolation=cv2.INTER_LINEAR)


def _depth_to_bgr(depth_norm: np.ndarray) -> np.ndarray:
    """float32 [0,1] → BGR uint8 via TURBO colormap."""
    return cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _metric_to_bgr(depth_m: np.ndarray, d_max: float = 3.0) -> np.ndarray:
    """float32 metres → BGR uint8 via TURBO colormap, clipped to [0, d_max]."""
    norm = np.clip(depth_m / d_max, 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--mello_port', '-mp', default='/dev/serial/by-id/usb-M5Stack_Technology_Co.__Ltd_M5Stack_UiFlow_2.0_24587ce945900000-if00', help="Mello device serial port")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec.")
@click.option('--debug', is_flag=True, help="Use dummy Mello interface with fixed joint positions for testing.")
@click.option('--osc_kp_pos', default=1000.0, type=float, help="OSC position stiffness (default 1000)")
@click.option('--osc_kp_rot', default=50.0, type=float, help="OSC rotation stiffness (default 50)")
@click.option('--enable_da3', is_flag=True, default=False, help="Enable DA3 depth model inference and fused-depth visualization (off by default).")
def main(output, robot_ip, mello_port, vis_camera_idx, init_joints, frequency, command_latency, debug, osc_kp_pos, osc_kp_rot, enable_da3):

    # Build camera serials/configs from the single CAMERA_SPECS source of truth.
    camera_serial_numbers = [spec['serial'] for spec in CAMERA_SPECS]
    configs = [json.load(open(_CONFIG_DIR + spec['config'])) for spec in CAMERA_SPECS]
    # Locate side/wrist cameras by name (used for depth viz + DA3 fusion).
    _name_to_idx = {spec['name']: i for i, spec in enumerate(CAMERA_SPECS)}
    side_idx  = _name_to_idx.get('side')
    wrist_idx = _name_to_idx.get('wrist')
    has_depth_pair = side_idx is not None and wrist_idx is not None

    dt = 1/frequency
    with SharedMemoryManager() as shm_manager:
        MelloInterface = DummyMelloTeleopInterface if debug else MelloTeleopInterface
        mello_kwargs = {} if debug else {'port': mello_port}
        da3_ctx = DA3DepthClient(device=0) if enable_da3 else nullcontext()
        with da3_ctx as da3_client, \
             KeystrokeCounter() as key_counter, \
             MelloInterface(**mello_kwargs) as mello, \
            RealEnv(
                output_dir=output,
                robot_ip=robot_ip,
                obs_image_resolution=(640,480),
                camera_serial_numbers=camera_serial_numbers,
                camera_configs=configs,
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=False,
                enable_depth=True,
                record_raw_video=True,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager,
                osc_kp_pos=osc_kp_pos,
                osc_kp_rot=osc_kp_rot,
            ) as env:
            cv2.setNumThreads(1)

            print('Waiting for environment to be ready (gripper calibration)...')
            t_ready_deadline = time.monotonic() + 60
            while not env.is_ready:
                if time.monotonic() > t_ready_deadline:
                    raise RuntimeError('Environment not ready after 60 s')
                time.sleep(0.1)

            kd_pos = 2 * np.sqrt(osc_kp_pos) * 1.0
            kd_rot = 2 * np.sqrt(osc_kp_rot) * 1.0
            print(f'OSC: Kp_pos={osc_kp_pos}, Kp_rot={osc_kp_rot}, Kd_pos={kd_pos:.1f}, Kd_rot={kd_rot:.1f}')
            print('Ready!')

            # ── Depth setup (side|wrist) — only if both cameras are present ──
            _side_depth_scale = _wrist_depth_scale = None
            _SIDE_FOCAL_PX = _WRIST_FOCAL_PX = None
            _depth_writer = _fused_writer = _fused_raw_writer = None
            _depth_video_path = _fused_video_path = _fused_raw_video_path = None
            if has_depth_pair:
                _side_serial  = CAMERA_SPECS[side_idx]['serial']
                _wrist_serial = CAMERA_SPECS[wrist_idx]['serial']
                # Depth scales and intrinsics — read once after cameras are ready
                _side_depth_scale  = env.realsense.cameras[_side_serial].get_depth_scale()
                _wrist_depth_scale = env.realsense.cameras[_wrist_serial].get_depth_scale()
                _side_K  = env.realsense.cameras[_side_serial].get_intrinsics()
                _wrist_K = env.realsense.cameras[_wrist_serial].get_intrinsics()
                _SIDE_FOCAL_PX  = float(_side_K[0, 0] + _side_K[1, 1]) / 2.0
                _WRIST_FOCAL_PX = float(_wrist_K[0, 0] + _wrist_K[1, 1]) / 2.0
                print(f'Depth scales — side: {_side_depth_scale:.5f}, wrist: {_wrist_depth_scale:.5f}')
                print(f'Focal lengths — side: {_SIDE_FOCAL_PX:.1f} px, wrist: {_WRIST_FOCAL_PX:.1f} px')
                cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Depth', _DEPTH_IMG_W * 2, _DEPTH_IMG_H)

                # Depth video writer — saves side|wrist panel to output dir
                _depth_video_path = output + '/depth_preview.mp4'
                _depth_writer = cv2.VideoWriter(
                    _depth_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    frequency,
                    (_DEPTH_IMG_W * 2, _DEPTH_IMG_H),
                )
                print(f'Depth video → {_depth_video_path}')

            # ── DA3 setup — only when explicitly enabled via --enable_da3 ────
            if enable_da3:
                if not has_depth_pair:
                    raise RuntimeError('--enable_da3 requires both "side" and "wrist" cameras in CAMERA_SPECS')
                print('Waiting for DA3 model to be ready (loading in background)...')
                da3_client.wait_ready(timeout=120.0)
                print('DA3 model ready.')

                # DA3-fused depth video writer (separate file, same dimensions)
                _fused_video_path = output + '/da3_fused_depth.mp4'
                _fused_writer = cv2.VideoWriter(
                    _fused_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    frequency,
                    (_DEPTH_IMG_W * 2, _DEPTH_IMG_H),
                )
                print(f'DA3 fused depth video → {_fused_video_path}')

                # DA3-fused depth video without colour overlay (pure depth colourmap)
                _fused_raw_video_path = output + '/da3_fused_depth_raw.mp4'
                _fused_raw_writer = cv2.VideoWriter(
                    _fused_raw_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    frequency,
                    (_DEPTH_IMG_W * 2, _DEPTH_IMG_H),
                )
                print(f'DA3 fused depth (raw) video → {_fused_raw_video_path}')
                cv2.namedWindow('DA3 Fused Depth', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('DA3 Fused Depth', _DEPTH_IMG_W * 2, _DEPTH_IMG_H)
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            # inner_finger_knuckle_joint: angle = (gripper_pos / 255) * pi/4
            _GRIPPER_JOINT_SCALE = np.pi / 4 / 255.0
            finger_log = []  # list of (timestamp, knuckle_angle_rad)
            finger_display = 0.0
            while not stop:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                obs = env.get_obs()

                # Fire off DA3 inference now; collect results after visualization
                # so the ~43 ms model forward pass overlaps with other loop work.
                if enable_da3:
                    da3_client.submit(obs['side_rgb'][-1], obs['wrist_rgb'][-1])

                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                stage = key_counter[Key.space]

                # visualize
                _cam_keys = [spec['name'] + '_rgb' for spec in CAMERA_SPECS]
                vis_img = obs[_cam_keys[vis_camera_idx]][-1,:,:,::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                if debug:
                    text += ' (DEBUG)'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )
                cv2.putText(
                    vis_img,
                    f'Knuckle: {finger_display:.4f} rad',
                    (10, 65),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    thickness=2,
                    color=(0, 255, 255)
                )
                cv2.imshow('default', vis_img)

                # ── Depth visualisation (side | wrist) with colour overlay ──
                if has_depth_pair and env.last_realsense_data is not None:
                    side_raw  = env.last_realsense_data[side_idx].get('depth')
                    wrist_raw = env.last_realsense_data[wrist_idx].get('depth')
                    if side_raw is not None and wrist_raw is not None:
                        side_vis  = _depth_to_bgr(_process_depth(side_raw[-1],  _side_depth_scale))
                        wrist_vis = _depth_to_bgr(_process_depth(wrist_raw[-1], _wrist_depth_scale))
                        # Blend colour image over depth colourmap (50/50)
                        # obs images are RGB uint8; convert to BGR and resize to 224×224
                        side_color  = cv2.resize(obs['side_rgb'][-1][:, :, ::-1],  (_DEPTH_IMG_W, _DEPTH_IMG_H))
                        wrist_color = cv2.resize(obs['wrist_rgb'][-1][:, :, ::-1], (_DEPTH_IMG_W, _DEPTH_IMG_H))
                        side_vis  = cv2.addWeighted(side_vis,  0.5, side_color,  0.5, 0)
                        wrist_vis = cv2.addWeighted(wrist_vis, 0.5, wrist_color, 0.5, 0)
                        cv2.putText(side_vis,  'SIDE',  (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                        cv2.putText(wrist_vis, 'WRIST', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                        depth_panel = np.concatenate([side_vis, wrist_vis], axis=1)
                        cv2.imshow('Depth', depth_panel)
                        _depth_writer.write(depth_panel)

                # ── DA3 fused depth (collect result submitted above) ─────────
                if enable_da3 and env.last_realsense_data is not None:
                    _da3_side_raw, _da3_wrist_raw = da3_client.collect()
                    _da3_side_m  = DA3DepthClient.to_metric(_da3_side_raw,  _SIDE_FOCAL_PX)
                    _da3_wrist_m = DA3DepthClient.to_metric(_da3_wrist_raw, _WRIST_FOCAL_PX)
                    _rs_side  = env.last_realsense_data[side_idx].get('depth')
                    _rs_wrist = env.last_realsense_data[wrist_idx].get('depth')
                    if _rs_side is not None and _rs_wrist is not None:
                        _fused_side  = DA3DepthClient.fuse_with_realsense(
                            _da3_side_m,  _rs_side[-1],  _side_depth_scale)
                        _fused_wrist = DA3DepthClient.fuse_with_realsense(
                            _da3_wrist_m, _rs_wrist[-1], _wrist_depth_scale)
                        _fsv = cv2.resize(
                            _metric_to_bgr(_fused_side),  (_DEPTH_IMG_W, _DEPTH_IMG_H))
                        _fwv = cv2.resize(
                            _metric_to_bgr(_fused_wrist), (_DEPTH_IMG_W, _DEPTH_IMG_H))
                        # Raw video: depth colourmap only, written before RGB blend
                        _fused_raw_panel = np.concatenate([_fsv, _fwv], axis=1)
                        _fused_raw_writer.write(_fused_raw_panel)
                        # Overlay video: blend RGB over depth colourmap
                        _sc  = cv2.resize(obs['side_rgb'][-1][:, :, ::-1],
                                          (_DEPTH_IMG_W, _DEPTH_IMG_H))
                        _wc  = cv2.resize(obs['wrist_rgb'][-1][:, :, ::-1],
                                          (_DEPTH_IMG_W, _DEPTH_IMG_H))
                        _fsv = cv2.addWeighted(_fsv, 0.5, _sc, 0.5, 0)
                        _fwv = cv2.addWeighted(_fwv, 0.5, _wc, 0.5, 0)
                        cv2.putText(_fsv,  'SIDE (fused)',  (5, 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                        cv2.putText(_fwv, 'WRIST (fused)', (5, 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                        _fused_panel = np.concatenate([_fsv, _fwv], axis=1)
                        cv2.imshow('DA3 Fused Depth', _fused_panel)
                        _fused_writer.write(_fused_panel)

                cv2.pollKey()

                precise_wait(t_sample)

                mello_values = mello.get_latest_values()
                mello_joints = mello_values[:6]
                gripper_command = mello_values[6]
                unified_action = np.concatenate([mello_joints, [gripper_command]])

                # inner_finger_knuckle_joint from real gripper POS register
                gripper_pos_raw = float(obs['gripper_pos'][-1])
                finger_display = gripper_pos_raw * _GRIPPER_JOINT_SCALE
                finger_log.append((time.monotonic() - t_start, finger_display))

                env.exec_actions(
                    actions=[unified_action], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])
                precise_wait(t_cycle_end)
                iter_idx += 1

            if _depth_writer is not None:
                _depth_writer.release()
                print(f'Depth video saved → {_depth_video_path}')
            if _fused_writer is not None:
                _fused_writer.release()
                print(f'DA3 fused depth video saved → {_fused_video_path}')
            if _fused_raw_writer is not None:
                _fused_raw_writer.release()
                print(f'DA3 fused depth (raw) video saved → {_fused_raw_video_path}')

            # Plot inner_finger_knuckle_joint after session ends
            if finger_log:
                timestamps = np.array([t for t, _ in finger_log])
                angles = np.array([a for _, a in finger_log])
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(timestamps, angles, linewidth=0.8)
                ax.set_ylabel('inner_finger_knuckle_joint (rad)')
                ax.set_xlabel('Time (s)')
                ax.grid(True)
                fig.suptitle('Gripper inner_finger_knuckle_joint over session')
                plt.tight_layout()
                plt.show()

# %%
if __name__ == '__main__':
    main()
