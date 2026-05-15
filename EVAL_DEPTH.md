# Eval a depth-DAgger student on the real UR5e

End-to-end recipe for going from a converged `StudentTeacherVision` checkpoint
trained on `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-...`
to a running policy on the real robot.

Two steps, two conda envs.

## 1. Export the JIT (one time per checkpoint)

Done inside the `isaac-sim` Docker container in the `patlab` env, because the
exporter has to instantiate the rsl_rl runner + `StudentTeacherVision` to read
its weights.

```bash
cd /mnt/storage/lti
bash isaac-start.sh                 # shells into the isaac-sim container
source activate_patlab.sh           # patlab env

cd /mnt/storage/lti/UWLab-patrick-private
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-PCTeacher-Lean-FullSysidDR-v0 \
  --num_envs 1 \
  --headless \
  --checkpoint <abs-path-to-model_*.pt>
```

The exporter (`uwlab_rl/rsl_rl/exporter.py:export_vision_student_as_jit`) runs
*before* the play loop. As soon as you see

```
[export_vision_student_as_jit] wrote <ckpt-dir>/exported/depth_policy.pt
[export_vision_student_as_jit] wrote sidecar <ckpt-dir>/exported/depth_policy_meta.txt
```

you can `Ctrl-C` — sim rollout is unnecessary for export.

The JIT bundles `student_obs_normalizer` + `depth_encoder` + `student` MLP into
one TorchScript module. Forward signature:

```
forward(proprio: (B, num_proprio) float32,
        images:  List[(B, 1, 224, 224) float32 in [0,1]]) -> (B, num_actions)
```

The sidecar `.txt` carries `num_proprio`, `num_actions`, `image_h/w/channels`,
`vision_groups` so the eval script can validate shapes at startup.

## 2. Run real-robot eval

Outside Docker, in the `robodiff_real` env (Py3.9 / PyTorch 1.12). No rsl_rl /
IsaacLab deps needed — the JIT is self-contained.

```bash
conda activate robodiff_real

cd /mnt/storage/lti/diffusion_policy
python eval_real_robot_depth.py \
  -i <ckpt-dir>/exported/depth_policy.pt \
  -o /tmp/depth_eval_$(date +%s) \
  --robot_ip <ur5e-ip> \
  --save_video
```

### Keys (focus the OpenCV window first)
- **C** — hand control to the policy
- **S** — stop episode, return to human control
- **R** — reset robot to initial joints + start new trajectory
- **G** — force gripper open for ~5 steps
- **Q** — quit

### Flags worth knowing
- `--no_da3_fusion` — feed raw RealSense u16 depth (no Depth-Anything-3 fusion).
  Matches sim more closely; use if the DA3 worker fails to start.
- `-j` / `--init_joints` — drive the arm to the standard start config before
  the first episode.
- `--frequency 10` (default) — control rate. Must match the sim env's 10 Hz.
- `--save_video` — also writes `policy_depth_input.mp4` (the side|wrist depth
  panels the policy actually sees) alongside the RGB videos.
- `--collect_sysid <path.pt>` — log the on-policy joint trajectory + OSC
  targets for later 500 Hz replay sysid.

## Obs layout (what the JIT expects)

Per-frame (built by `_build_proprio_tensor` in `eval_real_robot_depth.py`):

| field | dim | source |
|---|---|---|
| `prev_action` | 7 | last raw policy output (6 OSC delta + 1 gripper) |
| `joint_pos`   | 12 | 6 from RTDE `getActualQ` + 6 reconstructed gripper joints |
| `end_effector_pose` | 6 | calibrated FK on the 6 arm joints (axis-angle) |

History length 5, terms concatenated with `flatten_history_dim=True` →
`5*7 + 5*12 + 5*6 = 125` proprio dims. Layout:
`[prev_action @ T-4..T-0 flat, joint_pos @ T-4..T-0 flat, ee_pose @ T-4..T-0 flat]`.

The 6 reconstructed gripper joints come from the master `finger_joint` angle
(`gripper_pos_register * π/4 / 255`) multiplied by the per-column mimic ratios
`[+1, +1, -1, +1, -1, -1]` — derived from the closed-gripper reset state in
`/mnt/storage/lti/UWLab/reset_states_dataset_small/Resets/Peg/resets_ObjectAnywhereEEAnywhere.pt`.

Depth images: each (1, 224, 224) float32 in [0,1], clipped at `(0.01, 2.0)` m,
no-return pixels mapped to d_max — same `process_image` math as the sim env's
`DEPTH_CLIP`.

## Common failures

- **`proprio dim mismatch: built X but JIT expects Y`** — the eval script's
  proprio reconstruction disagrees with what the policy was trained on. Check
  `NUM_JOINTS`, `HISTORY_LEN`, and `GRIPPER_MIMIC_RATIOS` against the cfg.
- **DA3 timeout** — the DA3 worker takes ~26 s to load on RTX 4090. The script
  waits up to 120 s; if it still times out, run with `--no_da3_fusion`.
- **`Depth frames missing from realsense buffer`** — RealEnv was started with
  `enable_depth=False` somewhere upstream. Verify the `RealEnv(...)` kwargs in
  `eval_real_robot_depth.py:main` were not overridden.
- **Robot doesn't move at all** — check the policy is in scope (window
  focused, "C" pressed). The "stuck detection" auto-opens the gripper after
  2 s of no movement; if it triggers immediately the proprio is probably wrong
  (policy outputs near-zero deltas because obs is OOD).

## File map

| File | Purpose |
|---|---|
| `UWLab-patrick-private/source/uwlab_rl/uwlab_rl/rsl_rl/exporter.py` | `export_vision_student_as_jit` + scripted wrapper |
| `UWLab-patrick-private/scripts/reinforcement_learning/rsl_rl/play.py` | Auto-dispatches to vision exporter for `StudentTeacherVision` |
| `diffusion_policy/eval_real_robot_depth.py` | Real-robot eval loop (this file's commands) |
| `diffusion_policy/eval_real_robot.py` | Untouched — ASTEROID / diffusion_policy ckpts |
| `diffusion_policy/diffusion_policy/real_world/da3_depth_client.py` | DA3METRIC subprocess + RealSense fusion (reused) |
| `diffusion_policy/demo_real_robot.py` | Reference for depth viz + DA3 fusion patterns |
