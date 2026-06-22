"""
Robot-arm segmentation for the captured D455 point clouds (POST-PROCESS step).

Runs in the `foundstereo` conda env (torch 2.6). Takes the frames written by
`depth_recorder.py` (left_NNNNN.png + depth_NNNNN.npy + meta.json/K) and, using
SAM2, keeps ONLY the points that belong to the robot arm:

  1. click the arm once on the first left-IR frame (or pass --point x,y),
  2. SAM2's video predictor propagates that mask through every frame,
  3. each frame's depth is masked to the arm and back-projected to a cloud.

This sidesteps the cam->base extrinsic entirely: we segment in the image, so the
arm-only cloud falls out without knowing where the camera sits -- which makes the
later sim<->real ICP calibration much cleaner (no table/background to fight).

Outputs (into <pc_dir>/arm/ by default):
  cloud_arm_NNNNN.ply   arm-only point cloud (left-IR frame, meters)
  mask_NNNNN.png        binary arm mask  (QA)
  overlay_NNNNN.png     mask drawn on the IR frame (QA)

Usage:
  (foundstereo)$ python segment_pointclouds.py --pc_dir tmp/depth_playback_001/pointclouds
  # headless / scripted: give the click directly
  (foundstereo)$ python segment_pointclouds.py --pc_dir <dir> --point 430,240
  # refine with extra clicks: positive ';'-separated, negatives via --neg
  (foundstereo)$ python segment_pointclouds.py --pc_dir <dir> --point 430,240;500,300 --neg 100,100
"""
import os, sys, glob, json, argparse

REPO = os.path.dirname(os.path.realpath(__file__))
FFS = os.path.join(REPO, 'Fast-FoundationStereo')
sys.path.append(FFS)

import numpy as np
import cv2
import torch

from Utils import depth2xyzmap, toOpen3dCloud, o3d
from sam2.build_sam import build_sam2_video_predictor


def parse_points(s):
    """'430,240;500,300' -> [[430,240],[500,300]]; '' -> []"""
    if not s:
        return []
    out = []
    for pair in s.split(';'):
        pair = pair.strip()
        if not pair:
            continue
        x, y = pair.split(',')
        out.append([float(x), float(y)])
    return out


def pick_points_interactive(img_gray):
    """Show the first frame; LEFT-click = arm (positive), RIGHT-click = background
    (negative). Close the window when done. Returns (pos_list, neg_list)."""
    import matplotlib
    import matplotlib.pyplot as plt
    print('[segment] LEFT-click the arm (green), RIGHT-click background (red), '
          'then close the window...')
    pos, neg = [], []
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(img_gray, cmap='gray')
    ax.set_title('LEFT-click = arm (green)  |  RIGHT-click = background (red)  |  '
                 'close window when done')

    def onclick(event):
        if event.inaxes != ax or event.xdata is None:
            return
        if event.button == 1:        # left -> positive
            pos.append([float(event.xdata), float(event.ydata)])
            ax.plot(event.xdata, event.ydata, 'o', color='lime', ms=8, mec='black')
        elif event.button == 3:      # right -> negative
            neg.append([float(event.xdata), float(event.ydata)])
            ax.plot(event.xdata, event.ydata, 'x', color='red', ms=10, mew=3)
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()                       # blocks until the window is closed
    fig.canvas.mpl_disconnect(cid)
    if not pos:
        raise RuntimeError('no positive (arm) point clicked; pass --point x,y instead')
    return pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pc_dir', required=True,
                    help='pointclouds dir from depth_recorder.py (left/depth/meta)')
    ap.add_argument('--out_subdir', default='arm')
    ap.add_argument('--checkpoint',
                    default=f'{REPO}/weights/sam2/sam2.1_hiera_base_plus.pt')
    ap.add_argument('--model_cfg', default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    ap.add_argument('--point', default='', help='positive click(s) "x,y" or "x1,y1;x2,y2"')
    ap.add_argument('--neg', default='', help='negative click(s), same format')
    ap.add_argument('--zmax', type=float, default=None, help='override max depth (m)')
    ap.add_argument('--save_ply', type=int, default=1)
    ap.add_argument('--save_qa', type=int, default=1, help='write mask + overlay pngs')
    args = ap.parse_args()

    torch.autograd.set_grad_enabled(False)

    # --- intrinsics / config from the recorder ---
    meta = json.load(open(os.path.join(args.pc_dir, 'meta.json')))
    K = np.array([[meta['fx'], 0, meta['cx']],
                  [0, meta['fy'], meta['cy']],
                  [0, 0, 1]], np.float32)
    zmax = args.zmax if args.zmax is not None else meta.get('zmax', 6.0)

    left_files = sorted(glob.glob(os.path.join(args.pc_dir, 'left_*.png')))
    if not left_files:
        raise RuntimeError(f'no left_*.png in {args.pc_dir}')
    ids = [int(os.path.basename(f)[5:10]) for f in left_files]
    print(f'[segment] {len(left_files)} frames, K fx={meta["fx"]:.1f} zmax={zmax}')

    out_dir = os.path.join(args.pc_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # --- SAM2 needs a dir of RGB jpgs named <idx>.jpg ---
    frame_dir = os.path.join(out_dir, '_sam2_frames')
    os.makedirs(frame_dir, exist_ok=True)
    for i, f in enumerate(left_files):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(frame_dir, f'{i:05d}.jpg'),
                    cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    H, W = cv2.imread(left_files[0], cv2.IMREAD_GRAYSCALE).shape[:2]

    # --- prompts ---
    pos = parse_points(args.point)
    neg = parse_points(args.neg)
    if not pos:
        pos, neg_click = pick_points_interactive(cv2.imread(left_files[0], cv2.IMREAD_GRAYSCALE))
        neg = neg + neg_click
    points = np.array(pos + neg, np.float32)
    labels = np.array([1] * len(pos) + [0] * len(neg), np.int32)
    print(f'[segment] prompts: {len(pos)} positive, {len(neg)} negative @ {pos}')

    # --- SAM2 video predictor ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=device)
    autocast = torch.autocast(device, dtype=torch.bfloat16) if device == 'cuda' \
        else torch.no_grad()
    with autocast:
        state = predictor.init_state(video_path=frame_dir,
                                     offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1,
                                        points=points, labels=labels)
        # propagate -> dict frame_idx -> bool mask (H,W)
        masks = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            masks[frame_idx] = (mask_logits[0, 0] > 0.0).cpu().numpy()

    # --- mask depth + back-project per frame ---
    n_written = 0
    for i, (f, idx) in enumerate(zip(left_files, ids)):
        if i not in masks:
            continue
        mask = masks[i]
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        depth = np.load(os.path.join(args.pc_dir, f'depth_{idx:05d}.npy')).astype(np.float32)

        depth_arm = np.where(mask, depth, 0.0)
        xyz = depth2xyzmap(depth_arm, K)
        pts = xyz.reshape(-1, 3)
        cols = np.tile(gray[..., None], (1, 1, 3)).reshape(-1, 3)
        keep = (pts[:, 2] > 0.1) & (pts[:, 2] <= zmax)
        pts, cols = pts[keep], cols[keep]

        if args.save_ply:
            pcd = toOpen3dCloud(pts, cols)
            o3d.io.write_point_cloud(os.path.join(out_dir, f'cloud_arm_{idx:05d}.ply'), pcd)
        if args.save_qa:
            cv2.imwrite(os.path.join(out_dir, f'mask_{idx:05d}.png'),
                        (mask * 255).astype(np.uint8))
            overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f'overlay_{idx:05d}.png'), overlay)
        n_written += 1
        print(f'  frame {idx:05d}: {int(mask.sum())} arm px -> {len(pts)} pts')

    json.dump({
        'pc_dir': args.pc_dir, 'n_frames': n_written,
        'checkpoint': args.checkpoint, 'model_cfg': args.model_cfg,
        'pos_points': pos, 'neg_points': neg, 'zmax': zmax,
        'note': 'arm-only clouds in the D455 LEFT-IR frame (meters); '
                'masked in-image via SAM2 so no cam->base extrinsic needed',
    }, open(os.path.join(out_dir, 'segment_meta.json'), 'w'), indent=2)
    print(f'[segment] wrote {n_written} arm clouds to {out_dir}')


if __name__ == '__main__':
    main()
