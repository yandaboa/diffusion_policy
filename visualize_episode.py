"""
Visualize a recorded episode: camera views + sensor graph synced to video.

Usage:
    python visualize_episode.py <data_dir> <episode_number> [-o output.mp4] [--signal tcp_force|gripper_current]
"""

import argparse
import os
import numpy as np
import cv2
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGNAL_CONFIGS = {
    'tcp_force': {
        'zarr_key': 'tcp_force',
        'labels': ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'],
        'colors': ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#1abc9c'],
        'ylabel': 'Force (N) / Torque (Nm)',
    },
    'gripper_current': {
        'zarr_key': 'gripper_current',
        'labels': ['current (0-255)'],
        'colors': ['#f39c12'],
        'ylabel': 'Motor current (~10 mA/unit)',
    },
    'gripper_pos': {
        'zarr_key': 'gripper_pos',
        'labels': ['inner_finger_knuckle_joint (rad)'],
        'colors': ['#e67e22'],
        'ylabel': 'Joint angle (rad)',
        'transform': lambda x: x * (3.14159265 / 4 / 255.0),
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing videos/ and replay_buffer.zarr/')
    parser.add_argument('episode', type=int, help='Episode number')
    parser.add_argument('-o', '--output', default=None, help='Output path (default: <data_dir>/episode_<N>_<signal>_viz.mp4)')
    parser.add_argument('--signal', default='tcp_force', choices=list(SIGNAL_CONFIGS),  # noqa: E501
                        help='Which signal to plot (default: tcp_force)')
    args = parser.parse_args()

    ep = args.episode
    data_dir = args.data_dir
    cfg = SIGNAL_CONFIGS[args.signal]
    out_path = args.output or os.path.join(data_dir, f'episode_{ep}_{args.signal}_viz.mp4')

    # --- Load zarr ---
    root = zarr.open(os.path.join(data_dir, 'replay_buffer.zarr'), 'r')
    episode_ends = root['meta/episode_ends'][:]
    start = 0 if ep == 0 else int(episode_ends[ep - 1])
    end = int(episode_ends[ep])
    signal_data = root[f'data/{cfg["zarr_key"]}'][start:end]  # (N,) or (N, D)
    if signal_data.ndim == 1:
        signal_data = signal_data[:, np.newaxis]
    if 'transform' in cfg:
        signal_data = cfg['transform'](signal_data)
    timestamps = root['data/timestamp'][start:end]

    N = len(signal_data)
    zarr_hz = 1.0 / float(np.mean(np.diff(timestamps)))

    # --- Load videos ---
    vid_dir = os.path.join(data_dir, 'videos', str(ep))
    vid_files = sorted(f for f in os.listdir(vid_dir) if f.endswith('.mp4'))
    caps = [cv2.VideoCapture(os.path.join(vid_dir, f)) for f in vid_files]

    vid_fps = caps[0].get(cv2.CAP_PROP_FPS)
    n_vid_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    ratio = vid_fps / zarr_hz

    ret, frame0 = caps[0].read()
    caps[0].set(cv2.CAP_PROP_POS_FRAMES, 0)
    cam_h, cam_w = frame0.shape[:2]

    print(f"Episode {ep}: {n_vid_frames} video frames @ {vid_fps:.0f} fps, "
          f"{N} zarr frames @ {zarr_hz:.1f} Hz ({len(vid_files)} cameras)")
    print(f"Signal: {args.signal}  |  Video/zarr ratio: {ratio:.2f}x  |  Duration: {n_vid_frames/vid_fps:.1f}s")

    # --- Build plot (rendered once, cursor drawn per-frame) ---
    total_cam_w = cam_w * len(vid_files)
    plot_h = 280
    plot_dpi = 100

    fig, ax = plt.subplots(figsize=(total_cam_w / plot_dpi, plot_h / plot_dpi), dpi=plot_dpi)
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#2a2a2a')

    t = np.arange(N)
    for i, (label, color) in enumerate(zip(cfg['labels'], cfg['colors'])):
        ax.plot(t, signal_data[:, i], color=color, lw=0.8, label=label, alpha=0.9)

    ax.set_xlim(0, N - 1)
    ax.set_ylabel(cfg['ylabel'], color='#cccccc', fontsize=8)
    ax.set_xlabel(f'Zarr frame  (1 step = {1000/zarr_hz:.0f} ms)', color='#cccccc', fontsize=8)
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
    ax.grid(True, alpha=0.2, color='#888888')
    ax.legend(loc='upper right', fontsize=7, ncol=len(cfg['labels']),
              facecolor='#333333', edgecolor='#555555', labelcolor='white')
    fig.tight_layout(pad=0.6)

    # Render static background once
    fig.canvas.draw()
    plot_bg = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).copy()
    plot_bg = plot_bg.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plot_bg_bgr = cv2.cvtColor(plot_bg, cv2.COLOR_RGB2BGR)

    # Get axes bounding box in pixel coords (matplotlib: origin bottom-left)
    bbox = ax.get_window_extent()
    img_h = plot_bg.shape[0]
    ax_x0 = int(bbox.x0)
    ax_x1 = int(bbox.x1)
    ax_y0_img = img_h - int(bbox.y1)  # flip to image coords
    ax_y1_img = img_h - int(bbox.y0)

    plt.close(fig)

    # --- Video writer ---
    out_h = cam_h + plot_h
    out_w = total_cam_w
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, vid_fps, (out_w, out_h))

    print(f"Writing {out_path}  ({out_w}x{out_h} @ {vid_fps:.0f} fps)...")

    for frame_idx in range(n_vid_frames):
        zarr_idx = min(int(round(frame_idx / ratio)), N - 1)

        # Camera row
        cam_frames = []
        for cap in caps:
            ret, frame = cap.read()
            cam_frames.append(frame if ret else np.zeros((cam_h, cam_w, 3), dtype=np.uint8))
        cam_row = np.concatenate(cam_frames, axis=1)

        # Force plot with cursor: copy background, draw vertical line
        plot_frame = plot_bg_bgr.copy()
        x_norm = zarr_idx / max(N - 1, 1)
        x_px = ax_x0 + int(x_norm * (ax_x1 - ax_x0))
        cv2.line(plot_frame, (x_px, ax_y0_img), (x_px, ax_y1_img), (255, 255, 255), 1)

        # Resize plot strip to exact output width (handles any rounding)
        if plot_frame.shape[1] != out_w:
            plot_frame = cv2.resize(plot_frame, (out_w, plot_h))

        out_frame = np.vstack([cam_row, plot_frame])
        writer.write(out_frame)

        if frame_idx % 300 == 0:
            pct = frame_idx / n_vid_frames * 100
            print(f"  {frame_idx}/{n_vid_frames}  ({pct:.0f}%)")

    writer.release()
    for cap in caps:
        cap.release()

    print(f"Done → {out_path}")


if __name__ == '__main__':
    main()
