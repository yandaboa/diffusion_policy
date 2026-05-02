"""Extract actions for a specified episode from tactile/replay_buffer.zarr and save as .npy."""

import argparse
import numpy as np
import zarr

def extract_episode_actions(zarr_path: str, episode_idx: int, output_path=None):
    z = zarr.open(zarr_path, "r")
    episode_ends = z["meta/episode_ends"][:]
    n_episodes = len(episode_ends)

    if episode_idx < 0 or episode_idx >= n_episodes:
        raise ValueError(f"episode_idx {episode_idx} out of range [0, {n_episodes - 1}]")

    start = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
    end = episode_ends[episode_idx]

    actions = z["data/action"][start:end]

    if output_path is None:
        output_path = f"episode_{episode_idx}_actions.npy"

    np.save(output_path, actions)
    print(f"Episode {episode_idx}: {actions.shape} actions saved to {output_path}")
    return actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=int, help="Episode index (0-based)")
    parser.add_argument("--zarr", default="tactile/replay_buffer.zarr", help="Path to replay_buffer.zarr")
    parser.add_argument("--out", default=None, help="Output .npy path (default: episode_N_actions.npy)")
    args = parser.parse_args()

    extract_episode_actions(args.zarr, args.episode, args.out)
