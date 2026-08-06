# PointNet Point-Cloud Policy — Real-Robot Eval (RealSense + Fast-FoundationStereo)

Design doc for deploying a `PointNet` / `ResidualPointNet` behavior-cloning policy
(trained in `UWLab-patrick-private/scripts/imitation_learning/point_cloud/`) on the
real UR5e, using a **single front RealSense + Fast-FoundationStereo (FFS)** depth
pipeline and **SAM2** segmentation.

Status: **proposal / spec — no code written yet.** This doc is the contract the
implementation must satisfy. The spec below is **locked to the target checkpoint
`pnocc_xl_residual_big_ee`** (read off its `hyper_parameters` + the `omnireset`
training config); there are no remaining `⚠ CONFIRM` items.

**Target checkpoint:**
`/mnt/storage/lti/UWLab-patrick-private/logs/pc_bc/pnocc_xl_residual_big_ee/residual-point-net-045-0.0230.ckpt`
(`residual_point_net`, encoder `[1024,1024,1024]`, action head `[2048]×5`,
`point_dim=4`, `proprio_dim=18`, `action_dim=7`, `num_points=1024`, `predict_std=false`).

**Implementation progress (offline-verified, no hardware yet):**

- ✅ Vendored model: `diffusion_policy/model/point_cloud/{point_net,residual_point_net,flatten_mlp,bc_utils}.py` + `__init__.py` — loads the real checkpoint, forward → finite `(1,7)`.
- ✅ `diffusion_policy/real_world/pointnet_policy.py` — `PointNetPolicy` + 18-d proprio reconstruction (`build_joint_pos`/`build_ee_pose`/`build_proprio`). Parity with raw `bc_actions` math = exact (max|Δ|=0).
- ✅ `diffusion_policy/real_world/pointcloud_builder.py` — backproject → base → EE frame → crop → 512/256/256 budget sample. Unit-tested (transform math, backprojection, budgeting, end-to-end into policy).
- ✅ `debug_pointcloud.py` — `--demo`/`--from-file`/`--live`, depth sources `realsense|ffs|file`, `--seg` (SAM2), `--ffs-mock`.
- ✅ `real_world/realsense_stereo.py` — stereo-IR grabber (emitter off, IR1+IR2 + baseline + left-IR K + color).
- ✅ `real_world/ffs_depth_client.py` + `ffs_worker.py` — DA3-style shm client/worker. **Handshake verified end-to-end in `--mock`.** `_load_model`/`_infer_depth` now match the `**Fast-FoundationStereo` submodule** (`scripts/run_realsense_d455.py`): serialized full-model `torch.load`, `InputPadder(divis_by=32)`, `forward(test_mode=True, optimize_build_volume='pytorch1')`, `depth = fx·baseline/disp`. Defaults resolve to the submodule + `weights/23-36-37/model_best_bp2_serialize.pth`.
- ✅ `real_world/pointcloud_segmenter.py` — SAM2 (mirrors `orbbec_segment_pointclouds.py`); 3-class label-map composition unit-tested.
- ✅ `Fast-FoundationStereo` submodule initialized at repo root (`git@github.com:yandaboa/Fast-FoundationStereo.git`, branch `real`).
- ⬜ Needs real machine: `foundstereo` env + set `FFS_PYTHON`; **download FFS weights** (gitignored — Drive link in submodule readme → `Fast-FoundationStereo/weights/23-36-37/`); SAM2 weights at `orbbec/weights/sam2/`. Then `eval_real_robot_pointcloud.py` (robot loop).

---

## 1. Goal

Reproduce, on the real robot, the exact point-cloud observation the policy saw in
sim, feed it through the trained model each control cycle, and apply the decoded
action via `RealEnv`. The model does **no point normalization**, so the real cloud
must match the sim cloud in **frame, units, point count, per-class composition, and
segmentation labels**. Everything below exists to honor that.

---

## 2. The deployment target (sim cloud spec)

Extracted from the `omnireset` task in `UWLab-patrick-private`
(`.../manipulation/omnireset/mdp/observations.py`,
`.../config/ur5e_robotiq_2f85/{pc_obs_cfg,sim2real_pc_cfg}.py`):


| Property           | Value                                                                                                                                                                                                                       | Source of truth                                                 |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Frame              | **wrist_3_link / EE frame** (`ref_cfg = robot @ wrist_3_link`)                                                                                                                                                              | `Ur5eRobotiq2f85BCPointNetSegEvalCfg`, line 344                 |
| Units              | metres, no scaling                                                                                                                                                                                                          | —                                                               |
| Point count        | **1024** total                                                                                                                                                                                                              | `DataCollectObsCfg.scene_pc.num_points`, line 197               |
| Per-class budget   | robot / peg / hole = **512 / 256 / 256** (`class_ratios=(0.5,0.25,0.25)`)                                                                                                                                                   | line 202                                                        |
| 4th channel        | seg label, raw float: **0.0 = robot, −1.0 = insertive (peg), +1.0 = receptive (hole)**                                                                                                                                      | `segmentation_labels`, lines 336–340                            |
| RGB                | none                                                                                                                                                                                                                        | —                                                               |
| Cloud source       | mesh-sampled from robot/peg/hole **only** (no table/background), then projected to **front-camera optical frame** + occlusion-culled (frustum + HPR + dropout)                                                              | `OccludedScenePointCloud`                                       |
| Proprio (**18-d**) | `joint_pos(12: 6 arm + 6 Robotiq mimic, rad)` + `end_effector_pose(6: xyz + axis-angle, EE-in-**base**-frame)`, **in that declaration order**. **No prev_actions** — trainer allowlist is `{joint_pos, end_effector_pose}`. | `bc_utils._BC_PROPRIO_TERMS`; `DataCollectObsCfg` lines 215–224 |
| Action (7-d)       | RelCartesian OSC Δpose(6: xyz + axis-angle) + binary gripper(1)                                                                                                                                                             | `actions.py`                                                    |


Normalization stats (`proprio_mean/std`, `action_mean/std`) are **baked into the
checkpoint `state_dict`** and loaded by `bc_utils.load_bc_pointnet()`. The sim-side
inference path is `bc_utils.bc_actions()`:

```python
pc = scene_pc.reshape(N, num_points, point_dim)          # 4th channel = seg label
proprio = cat([joint_pos, end_effector_pose], -1)        # 18-d, declaration order
proprio = (proprio - proprio_mean) / proprio_std         # z-score
out = model(pc, proprio)                                 # ResidualPointNet forward
action = out * action_std + action_mean                  # denormalize → env action units
```

`pointnet_policy.py` re-implements exactly this on real tensors (no Isaac obs dict).
There is already a matching **sim eval config**, `Ur5eRobotiq2f85BCPointNetSegEvalCfg`
(`sim2real_pc_cfg.py:324`), driven by `play.py --bc_checkpoint`; our real loop is its
hardware twin. **We vendor** `bc_utils.py`, `point_net.py`, `residual_point_net.py`,
`flatten_mlp.py` into `diffusion_policy/` (self-contained: torch + torchvision only) so
the eval machine needs no `UWLab-patrick-private` checkout on `sys.path`.

---

## 3. The three sim2real seams (why this is non-trivial)

1. **Only robot + peg + hole points exist — no background.** Sim mesh-samples
  exactly those assets. So on real, background removal is mandatory and every kept
   point must be classified into `{robot, peg, hole}`. SAM2 does this.
2. **Single front-camera occluded view, not a fused dense cloud.** The sim2real
  training variant projects to one front camera and culls. So we use **one front
   camera only** — no multi-cam fusion (it would hand the model a denser cloud than
   training).
3. **Per-class point budget is fixed.** The model always saw ~285/87/140 points.
  The real sampler must reproduce that composition, not just "512 points total."

**Decision (user):** all three classes — including the robot — are **camera-segmented
via SAM2** (single code path). Tradeoff accepted: robot points are noisier than the
sim mesh-sampled robot and may shift the label distribution vs. the FK+URDF
alternative. If deployment shows the robot channel is the failure mode, revisit
generating robot points synthetically from FK + URDF meshes (exact sim match).

---

## 4. Architecture / data flow

```
                 ┌─────────────────────────────────────────────────────────┐
                 │  front RealSense D4xx                                     │
                 │   ├─ IR left / IR right (rectified) + K + baseline        │
                 │   └─ RGB (for SAM2)                                       │
                 └───────────────┬───────────────────────┬─────────────────┘
                                 │ IR pair               │ RGB
                                 ▼                       ▼
                    ffs_depth_client ──► FFS worker   SAM2 segmenter
                    (shared mem, own conda env)       masks: robot/peg/hole
                                 │ metric depth          │ per-pixel class
                                 ▼                       │
                    pcd_utils.depth_to_points            │
                    (backproject + extrinsics → base)    │
                                 │ points (H·W, 3)        │
                                 ▼                       ▼
                    ┌────────────── pointcloud_builder ──────────────┐
                    │ 1. assign each point its SAM2 class label      │
                    │ 2. drop unlabeled (background) points          │
                    │ 3. (optional) EE-frame re-expression           │
                    │ 4. per-class budget sample → 512 pts           │
                    │ 5. emit (1, 512, 4) float32 tensor             │
                    └───────────────────────┬────────────────────────┘
                                             ▼
                    pointnet_policy.predict(points, proprio)
                    (load_bc_pointnet; z-score proprio; forward; denorm action)
                                             │ OSC Δpose(6) + gripper(1), real units
                                             ▼
                    RealEnv / RTDEInterpolationController  (one step / cycle)
```

Proprio (19-d) is built each cycle from robot state (`joint_pos`, FK EE pose via
`ur5e_kinematics.get_ee_pose`, and the previous action), then z-scored with the
checkpoint stats inside `pointnet_policy`.

---

## 5. Files to create

All paths relative to `/mnt/storage/lti/diffusion_policy`.

### Reused as-is (no new code)

- `scripts/sim2real/perception/pcd_utils.py` — `depth_to_points`, `crop_points`,
`read_calibration_file` (extrinsics JSON: serial → intrinsics + base pos/ori).
- `diffusion_policy/real_world/real_env.py`, `rtde_interpolation_controller.py`,
`ur5e_kinematics.py`, `spacemouse_shared_memory.py`, `multi_realsense.py`.

### Vendored (copied into `diffusion_policy/`, no upstream `sys.path`)

- `diffusion_policy/model/point_cloud/{point_net,residual_point_net,flatten_mlp,bc_utils}.py`
— verbatim copies from `UWLab-patrick-private/.../point_cloud/`. Self-contained
(torch + torchvision). Keep a header comment noting the source + checkpoint format so
they can be re-synced if the upstream model changes.

### New files


| File                                                  | Responsibility                                                                                                                                                                                                                                      | Key signature(s)                                                                |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `diffusion_policy/real_world/ffs_depth_client.py`     | Shared-mem client managing the FFS subprocess (clone of `da3_depth_client.py` structure).                                                                                                                                                           | `infer(left_ir, right_ir, K, baseline) -> depth_m`; `submit/collect` async API  |
| `diffusion_policy/real_world/ffs_worker.py`           | Runs in the FFS conda env; loads Fast-FoundationStereo, disparity→metric depth, writes back over shared mem.                                                                                                                                        | state-byte protocol matching `da3_worker.py`                                    |
| `diffusion_policy/real_world/pointcloud_segmenter.py` | SAM2 over the RGB frame → per-pixel class in `{robot:0.0, peg:−1.0, hole:+1.0, bg:NaN}`. Maps masks→points by pixel index.                                                                                                                          | `segment(rgb) -> label_map (H,W)`; prompt/track config for the 3 classes        |
| `diffusion_policy/real_world/pointcloud_builder.py`   | **The sim-convention encoder.** depth+labels → backproject → drop bg → frame transform → per-class budget sample → `(1,512,4)`. Single place encoding frame/budget/labels. Depth-source-agnostic (FFS default; RealSense-hw-depth / DA3 swappable). | `build(depth, label_map, K, extrinsic, ee_pose) -> torch.FloatTensor (1,512,4)` |
| `diffusion_policy/real_world/pointnet_policy.py`      | Wraps `load_bc_pointnet`: holds model + norm stats; builds 19-d proprio; z-scores; forward; denorms action.                                                                                                                                         | `predict(points, proprio) -> action (7,)` in real units                         |
| `eval_real_robot_pointcloud.py`                       | Top-level loop forked from `eval_real_robot_depth.py` scaffold (RealEnv, `_KeyReader`, SpaceMouse fallback, success/fail labeling). Obs = cloud + proprio; one model step/cycle.                                                                    | CLI: `-i <ckpt> -o <save_dir> --robot_ip ...`                                   |
| `debug_pointcloud.py`                                 | **Offline de-risk tool, no robot.** Capture → FFS → SAM2 → builder → Open3D render colored by seg label, against the workspace box and base/EE axes. Validates frame/scale/labels/budget before the robot is ever moved.                            | CLI viewer                                                                      |
| `configs/pointcloud_eval.yaml`                        | Runtime config: front-camera serial, depth source (`ffs`/`realsense`/`da3`), frame (`base`/`ee`), `num_points`, per-class budget, seg label values, extrinsics-calib path, SAM2 prompts, FFS env/model path.                                        | —                                                                               |


---

## 6. Key implementation notes

- **CRITICAL — two different frames in one obs:** the **cloud is in the EE
(wrist_3_link) frame**, but the **proprio `end_effector_pose` is EE-in-base-frame**.
Don't conflate them. `pointcloud_builder` transforms points into the EE frame via
`quat_apply(quat_inv(q_ee), p_base - t_ee)` (same math as sim `observations.py`),
where `(t_ee, q_ee)` is live FK of wrist_3_link in base. The proprio term is the
*raw* base-frame EE pose (xyz + axis-angle), NOT re-expressed.
- **Proprio (18-d), exact order `[joint_pos(12), end_effector_pose(6)]`:**
  - `joint_pos` = 6 UR5e arm joints (rad) **+ 6 Robotiq mimic joints**. Real hardware
  reports one gripper position; reconstruct the 6 mimic-joint columns from it using
  the sign pattern already in `eval_real_robot_depth.py::GRIPPER_MIMIC_RATIOS`
  (`master·{+1,+1,−1,+1,−1,−1}`, master = `pos·π/4`). Column order must match Isaac's
  `robot.data.joint_pos`.
  - `end_effector_pose` = wrist_3_link in **base** frame, xyz + axis-angle, from
  `ur5e_kinematics.get_ee_pose` + `quat_to_axis_angle`.
  - z-score the full 18-vector with `proprio_mean/std` (do **not** normalize per-term).
- **Per-class budget sampling:** after labeling, select 512 / 256 / 256 (robot/peg/
hole). If a class is under-budget on real (e.g. peg occluded), **⚠ DECIDE** pad
policy (repeat-sample vs. zero-fill vs. accept short) — sim always hits budget, so
this is a real gap and a likely failure mode. `debug_pointcloud.py` should print the
realized per-class counts every frame.
- **SAM2 label mapping:** SAM2 yields instance masks; map instances → `{robot 0.0, peg −1.0, hole +1.0}` (fixed prompts / first-frame click + tracking, reusing
`orbbec/orbbec_segment_pointclouds.py`). Background → dropped.
- **Action decode chain:** `a = action * action_std + action_mean` → split into arm
Δpose(6) + binary gripper(1). The arm 6-vec is a *pre-scale* RelCartesian OSC delta;
apply the **same `scale_xyz_axisangle` used during data collection** (⚠ confirm
collection vs. eval scale — eval cfg uses `(0.01,0.01,0.002,0.02,0.02,0.2)`), then
`apply_delta_pose` onto the current EE pose and command the controller. Gripper:
threshold the binary channel → open/close. Single-step model: run every cycle, no
action horizon.
- **Predict-std checkpoints:** N/A here (`predict_std=false`); if ever true, take the
mean head only at eval (`out[0]`).

---

## 7. Calibration / external deps required

- **Extrinsics:** front camera → robot base, in the JSON schema `read_calibration_file`
expects (`camera_serial_number`, `intrinsics`, `camera_base_pos`, `camera_base_ori`).
See `CAMERA_EXTRINSICS.md`. The base-frame correctness of the whole cloud rides on this.
- **FFS:** Fast-FoundationStereo checkout + its conda env (external; referenced today
only via `sys.path` from `orbbec/orbbec_segment_pointclouds.py`). Needs rectified IR
L/R + intrinsics + baseline from the RealSense.
- **SAM2:** model weights + env (same one `orbbec_segment_pointclouds.py` uses).

---

## 8. Bring-up sequence (de-risk order) — run on the real machine

`debug_pointcloud.py` is an **inspection tool only** — it opens the real camera, builds
the exact EE-frame cloud the policy will see, prints per-class counts + the EE-frame
AABB, and renders it in Open3D. **No robot motion**, so the arm can be parked. Run the
three steps in order; each adds one risky piece so a failure is easy to localize.

**One-time setup on the real machine:**

- FFS conda env → `export FFS_PYTHON=/path/to/envs/foundstereo/bin/python3`
- FFS weights (gitignored — Drive link in `Fast-FoundationStereo/readme.md`) →
`Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth`
- SAM2 weights → `orbbec/weights/sam2/sam2.1_hiera_base_plus.pt`

Replace `<SERIAL>` with the front D455 serial and `--joints` with the **actual 6 arm
angles (rad)** at capture time (FK gives the EE pose the cloud is expressed in).

```bash
cd /mnt/storage/lti/diffusion_policy
export PYTHONPATH=/mnt/storage/lti/diffusion_policy
```

**Step 1 — plumbing (no FFS weights, no SAM2).** Ramp ("mock") depth, so a wrong cloud
points at a capture / extrinsic / joints bug rather than the depth model:

SERIAL front - 215122255213

```bash
python debug_pointcloud.py --live --depth-source ffs --ffs-mock \
    --resolution 848 480 --serial <SERIAL> \
    --joints 0,-1.57,1.57,-1.57,-1.57,0
```

**Step 2 — real FFS depth** (after weights + `FFS_PYTHON`). Drop `--ffs-mock`; eyeball
the printed AABB — extent should look like real workspace decimetres, not mm or 1000×:

```bash
export FFS_PYTHON=/path/to/envs/foundstereo/bin/python3
python debug_pointcloud.py --live --depth-source ffs \
    --resolution 848 480 --serial <SERIAL> \
    --joints 0,-1.57,1.57,-1.57,-1.57,0
```

**Step 3 — add SAM2 segmentation** (3-class labels + budget). A click window opens per
class — click **robot, then peg, then hole**. Add `--crop xmin ymin zmin xmax ymax zmax`
(EE frame, m) once the workspace box is known, and `--save` to keep the cloud:

```bash
python debug_pointcloud.py --live --depth-source ffs --seg \
    --resolution 848 480 --serial <SERIAL> \
    --joints 0,-1.57,1.57,-1.57,-1.57,0 \
    --crop -0.3 -0.3 -0.3 0.3 0.3 0.3 \
    --save /tmp/cloud_check
```

Notes:

- Over SSH without X-forwarding, add `--no-vis` and inspect the saved `.ply` instead of
the Open3D window.
- Default extrinsic is the sim-approx (`calib/front_cam_to_base_simapprox.npy`) — expect
roughly right, not exact, until a real calibration replaces it.
- FFS depth is in the **left-IR frame** while the sim extrinsic targets the color optical
center (~1–2 cm offset) — fine for a "looks correct" check.
- What to confirm before the robot loop: cloud is registered (sits where the EE is),
scaled (AABB plausible), labeled (peg/hole masks land on the right points), and the
printed per-class counts hit 512/256/256 without large **SHORT (padded)** counts.

**After the cloud looks right** (the gate), proceed to the still-unwritten robot loop:
4. **Policy dry-run:** `pointnet_policy.predict` on a saved cloud; sanity-check action
   magnitudes after denorm (no NaNs, plausible deltas).
5. **Robot, human-gated:** `eval_real_robot_pointcloud.py` with SpaceMouse fallback;
   hand control to the policy only after the cloud viewer looks right.

---

## 9. Status — resolved vs. remaining

**Resolved (locked for `pnocc_xl_residual_big_ee`):**

- Checkpoint, architecture, and all shapes — see header + §2.
- Frame = EE (wrist_3_link); cloud = 1024 pts; budget = 512/256/256; labels
`{0.0,−1.0,+1.0}`; proprio = 18-d `[joint_pos(12), end_effector_pose(6)]`,
no prev_actions; action = OSC Δpose(6) + binary gripper(1).
- Model import = **vendor** the 4 files into `diffusion_policy/model/point_cloud/`.

**Remaining implementation decisions (not blockers — choose at build time):**

1. **Under-budget pad policy** when SAM2 yields fewer than 512/256/256 points for a
  class (occlusion). Options: repeat-sample within class / zero-pad / accept short.
   Recommend repeat-sample (keeps N fixed without injecting origin points).
2. **OSC action scale** used at *data collection* time — confirm whether demos used
  the collection scale or the eval scale `(0.01,0.01,0.002,0.02,0.02,0.2)`; the real
   action decode must use the collection scale to match what the model learned.
3. **SAM2 prompting** for robot vs. peg vs. hole — fixed text prompts, first-frame
  clicks + video tracking, or geometric priors. Affects `pointcloud_segmenter.py`
   only; settle during the §8 step-3 bring-up.

The spec is complete enough to write `pointcloud_builder.py`, `pointnet_policy.py`,
and `debug_pointcloud.py` against concrete numbers now; items 1–3 are local choices.

Command:
```
python eval_real_robot_pc.py -i /home/yandabao/diffusion_policy/pc_policies/big_no_gripper.pt  -o tmp/pc_eval_$(date +%s)         --robot_ip 192.168.1.10 -j --save_video -j
```