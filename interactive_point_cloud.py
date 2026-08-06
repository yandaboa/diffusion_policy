#!/usr/bin/env python3
"""Open an interactive window to inspect a point cloud and read off 3D positions.

Usage:
    python interactive_point_cloud.py [path/to/cloud.ply]

Controls (Open3D point-picking window):
    - Shift + Left click    : pick a point; its index and XYZ are printed to the terminal
    - Shift + Right click    : undo the most recent pick
    - Mouse drag / scroll    : rotate / zoom / pan
    - Press 'Q' or close window : quit; a summary of all picked points is printed

Notes:
    The picked coordinates are in the point cloud's own coordinate frame (for an
    Orbbec DepthPoints_*.ply this is the depth-camera frame, units in millimeters).
"""

import argparse
import sys

import numpy as np
import open3d as o3d

DEFAULT_PLY = "/opt/OrbbecSDK_v2.8.6/tools/output/pointcloud/1781412939118000/DepthPoints_1781412939118000.ply"


def pick_points(pcd: o3d.geometry.PointCloud) -> list[int]:
    """Show the editing visualizer and return indices of picked points."""
    print("\n" + "=" * 70)
    print("Shift + Left click  : pick a point (index + XYZ printed below)")
    print("Shift + Right click : undo last pick")
    print("Press 'Q' or close the window when finished")
    print("=" * 70 + "\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Interactive Point Cloud — Shift+Click to pick")
    vis.add_geometry(pcd)
    vis.run()  # blocks until the window is closed
    vis.destroy_window()
    return vis.get_picked_points()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ply", nargs="?", default=DEFAULT_PLY,
                        help="Path to the .ply point cloud (default: %(default)s)")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply)
    if pcd.is_empty():
        print(f"ERROR: no points loaded from {args.ply}", file=sys.stderr)
        return 1

    pts = np.asarray(pcd.points)
    print(f"Loaded {len(pts)} points from {args.ply}")
    print(f"  bounds min: {pts.min(axis=0)}")
    print(f"  bounds max: {pts.max(axis=0)}")

    picked = pick_points(pcd)

    if not picked:
        print("\nNo points were picked.")
        return 0

    print("\n" + "=" * 70)
    print(f"Picked {len(picked)} point(s):")
    print("=" * 70)
    for n, idx in enumerate(picked):
        x, y, z = pts[idx]
        print(f"  [{n}] index={idx:>8}  x={x:10.3f}  y={y:10.3f}  z={z:10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
