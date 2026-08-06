# diffusion_policy — repo orientation

Sim2real manipulation on a UR5e + Robotiq 2F-85 (peg-in-hole). Policies are trained in IsaacLab
(`~/UWLab-patrick-private`) and evaluated here on the real robot.

## Eval entry points

| Script | Policy | Docs |
|---|---|---|
| `eval_real_robot.py` | **STATE** (215-dim low-dim, TorchScript) | **[STATE_POLICY.md](STATE_POLICY.md)** |
| `eval_real_robot_pc.py` | point cloud | `POINTCLOUD_EVAL.md` |
| `eval_real_robot_depth.py` | depth | `EVAL_DEPTH.md` |
| `eval_real_robot_rgb.py` | RGB DAgger student | — |
| `eval_real_robot_tactile.py` | history-conditioned tactile / in-context | — |

`eval_real_robot.py` was the diffusion-policy eval historically; it is now the state-policy eval,
and the old one moved to `eval_real_robot_tactile.py`. The root `README.md` is the upstream
diffusion-policy README and its `eval_real_robot.py` examples are stale.

## Environments

- **`foundstereo`** — all perception tooling (owns `pyorbbecsdk`): `peg_fusion_viz.py`,
  `calibrate_hole_tags.py`, `pc_overlay_align.py`, `twin_frame_viewer.py`, `eval_real_robot_pc.py`.
- **`robodiff_real`** — the original diffusion-policy env. `eval_real_robot.py` runs here or in
  `foundstereo`.
- **`env_uwlab`** — IsaacLab, for anything under `~/UWLab-patrick-private/scripts_v2/`. Use
  `CUDA_VISIBLE_DEVICES=1` (GPU0 is usually occupied).

## ⚠️ Unresolved merge conflicts on `omnireset`

Commit `9a8868d` committed conflict markers into tracked files. `eval_real_robot.py`,
`real_world/real_env.py`, `real_world/rtde_interpolation_controller.py`,
`real_world/pointnet_policy.py`, `scripts/sim2real/0_camera_calibrate.py`, and
`scripts/sim2real/ALIGN_SCENE.MD` do not parse. The `HEAD` side is the current
(state-policy / newer) code; the `e45600ad` side is the older RGB-eval code.

```bash
grep -rn "^<<<<<<< \|^>>>>>>> " --include="*.py" --include="*.md" --include="*.MD" . \
  | grep -v -e Depth-Anything -e FoundationPose -e segment-anything -e Fast-Found
```

## Camera serials

front D455 `215122255213` · side D435 `832112070487` · wrist D415 `746112060198`.
Extrinsics live in `scripts/sim2real/perception/calibrations/` as
`most_recent_hand_aligned_extrinsic_<serial>.json` — always read the **per-serial** file; the
unkeyed `most_recent_hand_aligned_extrinsic.json` is overwritten by whichever camera was aligned
last. See `CAMERA_EXTRINSICS.md` and STATE_POLICY.md §3.
