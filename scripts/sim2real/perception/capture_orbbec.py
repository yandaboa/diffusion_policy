"""Grab a single RGB (and optional depth) frame from the connected Orbbec over USB.

Usage:
    python capture_orbbec.py                       # -> orbbec_capture.png
    python capture_orbbec.py -o board.png          # custom path
    python capture_orbbec.py --depth               # also dump depth as .npy + colormap
"""
import argparse
import cv2
import numpy as np

from orbbec import gather_orbbec_cameras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="orbbec_capture.png")
    ap.add_argument("--depth", action="store_true", help="also save depth (mm) as .npy + viz")
    args = ap.parse_args()

    cams = gather_orbbec_cameras(rgb=True, depth=args.depth, align="rgb" if args.depth else None)
    if not cams:
        raise SystemExit("No Orbbec camera found on USB.")
    cam = cams[0]
    print(f"[orbbec] using device serial {cam._serial_number}")

    frames = cam.read_camera()
    rgb = frames["rgb"]  # HxWx3 RGB
    cv2.imwrite(args.out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"[orbbec] saved RGB {rgb.shape[1]}x{rgb.shape[0]} -> {args.out}")

    if args.depth:
        depth = frames["depth"]  # HxW uint16 mm
        npy = args.out.rsplit(".", 1)[0] + "_depth.npy"
        np.save(npy, depth)
        viz = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=255.0 / max(1, depth.max())), cv2.COLORMAP_JET)
        viz_path = args.out.rsplit(".", 1)[0] + "_depth.png"
        cv2.imwrite(viz_path, viz)
        valid = depth[depth > 0]
        rng = f"{valid.min()}..{valid.max()} mm" if valid.size else "no valid depth"
        print(f"[orbbec] saved depth -> {npy} (+ {viz_path}); range {rng}")

    cam.disable_camera()


if __name__ == "__main__":
    main()
