"""
Plot a signal from a recorded episode and save the data as numpy.

Usage:
    python plot_episode_signal.py <data_dir> <episode_number> [--signal tcp_force|gripper_current]
"""

import argparse
import os
import numpy as np
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
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing replay_buffer.zarr/')
    parser.add_argument('episode', type=int, help='Episode number')
    parser.add_argument('--signal', default='tcp_force', choices=list(SIGNAL_CONFIGS),
                        help='Which signal to plot (default: tcp_force)')
    args = parser.parse_args()

    ep = args.episode
    data_dir = args.data_dir
    cfg = SIGNAL_CONFIGS[args.signal]
    stem = f'episode_{ep}_{args.signal}'

    # --- Load zarr ---
    root = zarr.open(os.path.join(data_dir, 'replay_buffer.zarr'), 'r')
    episode_ends = root['meta/episode_ends'][:]
    start = 0 if ep == 0 else int(episode_ends[ep - 1])
    end = int(episode_ends[ep])

    signal_data = root[f'data/{cfg["zarr_key"]}'][start:end]
    timestamps = root['data/timestamp'][start:end]

    if signal_data.ndim == 1:
        signal_data = signal_data[:, np.newaxis]

    N = len(signal_data)
    zarr_hz = 1.0 / float(np.mean(np.diff(timestamps)))
    t_sec = (timestamps - timestamps[0])

    # --- Save numpy ---
    npy_path = os.path.join(data_dir, f'{stem}.npz')
    np.savez(npy_path, signal=signal_data, timestamps=timestamps, t_sec=t_sec)
    print(f"Saved data → {npy_path}  (keys: signal {signal_data.shape}, timestamps, t_sec)")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, (label, color) in enumerate(zip(cfg['labels'], cfg['colors'])):
        ax.plot(t_sec, signal_data[:, i], color=color, lw=1.0, label=label, alpha=0.9)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(cfg['ylabel'])
    ax.set_title(f'Episode {ep} — {args.signal}  ({N} samples @ {zarr_hz:.1f} Hz)')
    ax.legend(loc='upper right', ncol=len(cfg['labels']))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(data_dir, f'{stem}.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot  → {plot_path}")


if __name__ == '__main__':
    main()
