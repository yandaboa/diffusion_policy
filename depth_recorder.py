"""
Continuous D455 -> Fast-FoundationStereo point-cloud recorder.

Runs in the `foundstereo` conda env (torch 2.6 + xformers). Owns the D455
exclusively, captures the rectified left/right IR pair, runs Fast-FoundationStereo
to get depth, back-projects to a point cloud, and dumps every frame to disk with a
wall-clock timestamp so it can be aligned afterwards with the robot joint trajectory.

Coordinated by `action_playback_depth.py` via two sentinel files:
  --ready_file : created by THIS script once the model is warmed up and streaming
  --stop_file  : watched by THIS script; capture stops when it appears

Standalone smoke test (no robot needed):
  conda run -n foundstereo python depth_recorder.py --out_dir tmp/pc_test --max_seconds 5
"""
import os, sys, time, json, argparse

REPO = os.path.dirname(os.path.realpath(__file__))
FFS = os.path.join(REPO, 'Fast-FoundationStereo')
sys.path.append(FFS)

import numpy as np
import cv2
import torch
import pyrealsense2 as rs

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_seed, depth2xyzmap, toOpen3dCloud, o3d


def load_model(model_dir, valid_iters, max_disp):
    model = torch.load(model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()
    return model


def infer_disparity(model, left_gray, right_gray, valid_iters):
    l = np.tile(left_gray[..., None], (1, 1, 3))
    r = np.tile(right_gray[..., None], (1, 1, 3))
    H, W = l.shape[:2]
    lt = torch.as_tensor(l).cuda().float()[None].permute(0, 3, 1, 2)
    rt = torch.as_tensor(r).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(lt.shape, divis_by=32, force_square=False)
    lt, rt = padder.pad(lt, rt)
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(lt, rt, iters=valid_iters, test_mode=True,
                             optimize_build_volume='pytorch1')
    return padder.unpad(disp.float()).cpu().numpy().reshape(H, W).clip(0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--serial', default='215122255213', help='D455 serial')
    ap.add_argument('--model_dir',
                    default=f'{FFS}/weights/23-36-37/model_best_bp2_serialize.pth')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ready_file', default=None)
    ap.add_argument('--stop_file', default=None)
    ap.add_argument('--max_seconds', type=float, default=60.0,
                    help='hard cap on recording duration (safety)')
    ap.add_argument('--width', type=int, default=848)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--valid_iters', type=int, default=8)
    ap.add_argument('--max_disp', type=int, default=192)
    ap.add_argument('--zmax', type=float, default=6.0, help='max depth to keep (m)')
    ap.add_argument('--emitter', type=int, default=0,
                    help='IR dot projector: 0=off (clean for the model), 1=on')
    ap.add_argument('--save_ply', type=int, default=1)
    args = ap.parse_args()

    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- D455 ---
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
    cfg.enable_stream(rs.stream.infrared, 2, args.width, args.height, rs.format.y8, args.fps)
    profile = pipe.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, float(args.emitter))

    ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    intr = ir1.get_intrinsics()
    baseline = abs(ir1.get_extrinsics_to(ir2).translation[0])
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], np.float32)

    json.dump({
        'serial': args.serial, 'width': args.width, 'height': args.height,
        'fx': intr.fx, 'fy': intr.fy, 'cx': intr.ppx, 'cy': intr.ppy,
        'baseline_m': baseline, 'valid_iters': args.valid_iters,
        'max_disp': args.max_disp, 'zmax': args.zmax, 'emitter': args.emitter,
        'note': 'point clouds are in the D455 LEFT-IR camera frame, meters',
    }, open(f'{args.out_dir}/meta.json', 'w'), indent=2)
    np.savetxt(f'{args.out_dir}/K.txt', [K.reshape(-1), [baseline] + [0] * 8], fmt='%.6f')

    model = load_model(args.model_dir, args.valid_iters, args.max_disp)

    # warm up: discard frames for auto-exposure + compile the model once
    for _ in range(10):
        frames = pipe.wait_for_frames()
    left = np.asanyarray(frames.get_infrared_frame(1).get_data())
    right = np.asanyarray(frames.get_infrared_frame(2).get_data())
    _ = infer_disparity(model, left, right, args.valid_iters)
    print('[recorder] warmed up, streaming', flush=True)
    if args.ready_file:
        open(args.ready_file, 'w').close()

    manifest = open(f'{args.out_dir}/manifest.csv', 'w')
    manifest.write('idx,timestamp,n_points\n')

    t0 = time.time()
    idx = 0
    try:
        while True:
            if args.stop_file and os.path.exists(args.stop_file):
                print('[recorder] stop file seen', flush=True)
                break
            if time.time() - t0 > args.max_seconds:
                print('[recorder] max_seconds reached', flush=True)
                break

            frames = pipe.wait_for_frames()
            ts = time.time()
            left = np.asanyarray(frames.get_infrared_frame(1).get_data())
            right = np.asanyarray(frames.get_infrared_frame(2).get_data())

            disp = infer_disparity(model, left, right, args.valid_iters)
            depth = np.zeros_like(disp)
            valid = disp > 0
            depth[valid] = K[0, 0] * baseline / disp[valid]

            xyz = depth2xyzmap(depth, K)                  # (H,W,3), left-IR frame
            color = np.tile(left[..., None], (1, 1, 3))   # gray IR as point color
            pts = xyz.reshape(-1, 3)
            cols = color.reshape(-1, 3)
            keep = (pts[:, 2] > 0.1) & (pts[:, 2] <= args.zmax)
            pts, cols = pts[keep], cols[keep]

            np.save(f'{args.out_dir}/depth_{idx:05d}.npy', depth.astype(np.float16))
            cv2.imwrite(f'{args.out_dir}/left_{idx:05d}.png', left)
            if args.save_ply:
                pcd = toOpen3dCloud(pts, cols)
                o3d.io.write_point_cloud(f'{args.out_dir}/cloud_{idx:05d}.ply', pcd)
            manifest.write(f'{idx},{ts:.6f},{len(pts)}\n')
            manifest.flush()
            idx += 1
    finally:
        manifest.close()
        pipe.stop()
        open(f'{args.out_dir}/recorder_done', 'w').close()
        print(f'[recorder] done, captured {idx} frames', flush=True)


if __name__ == '__main__':
    main()
