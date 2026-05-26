# HUGSIM Methods and Run Guide

## 1. Core Execution Model

Most runs in this repo eventually go through:
- [scripts/run_single_scene.slurm](scripts/run_single_scene.slurm)
- [scripts/run_single_nuscenes_scene.sh](scripts/run_single_nuscenes_scene.sh)
- [closed_loop.py](closed_loop.py)

The main environment variables used across runs are:
- `PLANNER_NAME`: high-level planner family, usually `rap_vlm`, `drivor_vlm`, or `rule_based`
- `PLANNER_PATH`: planner config YAML under `configs/planners/`
- `SCENARIO_PATH`: scenario YAML under `configs/benchmark/...`
- `BASE_PATH`: sim/base config under `configs/sim/`
- `SIM_CUDA`: simulator GPU, usually `inherit` under Slurm
- `AD_CUDA`: planner-side CUDA selection, usually `inherit`
- `RAP_DEVICE_OVERRIDE`, `DRIVOR_DEVICE_OVERRIDE`: planner model device override
- `PLANNER_VLM_DEVICE_OVERRIDE`: VLM worker device override

`scripts/run_single_nuscenes_scene.sh` resolves the dataset type from the scenario YAML and maps `PLANNER_NAME` to the **primary autonomous-driving backend**:
- `rap` or `rap_vlm` -> primary backend `ad=rap`
- `drivor` or `drivor_vlm` -> primary backend `ad=drivor`
- `rule_based` or `rule_based_vlm` -> primary backend `ad=rule_based`

Current operational planner categories are:
- **solo learned planner**
  - RAP-only
  - DrivoR-only
- **standalone `rule_based`**
- **rule-based Method 1 / Choice A**
  - merged candidate pool
- **rule-based Method 2 / Choice B**
  - planner-gated selection

Camera-setup terminology in current configs:
- **1-camera / front-only**
  - the VLM sees only the front camera
  - this shows up in config names like `*_intervention_1cam_*`
  - these are mainly earlier or comparison variants
- **4-camera / multiview**
  - the intervention or planner-gate VLM sees multiple cameras
  - this shows up in config names like `*_intervention_4cam_*`
  - this is the main current setup for the active rule-merge and rule-gate methods

Important distinction:
- `camera_mode: front_only` usually means the base scoring/default VLM view is front camera only
- `intervention_camera_mode: multiview` means the intervention VLM stage uses 4-camera context
- `planner_gate_camera_mode: front_only` means the current Choice B planner gate still uses front-camera context unless the config says otherwise

So in the active current configs:
- **solo RAP / solo DrivoR baseline variants** can exist in either 1-camera or 4-camera forms depending on config
- **Choice A** currently uses the 4-camera intervention configs as the main path
- **Choice B** currently uses the 4-camera intervention setup plus a planner gate that is front-only by default
- **standalone `rule_based`** does not rely on a multiview VLM stage in its default config

That gives three current rule-based participation modes:
- **standalone `rule_based`**
  - primary backend is `rule_based`
- **Choice A / `*_rule_merge*`**
  - primary backend is still RAP or DrivoR
  - rule-based trajectories are added into the same candidate pool
  - the VLM selects from the combined learned + rule-based pool
- **Choice B / `*_rule_gate*`**
  - primary backend is still RAP or DrivoR
  - learned and rule-based candidate families are built separately
  - a planner-gating VLM chooses which family to use before lower-level selection

## 2. Current Method Catalog

### Baseline learned planners

These are the current learned-planner baselines without the rule-based merge variants.

| Method | What it is | Representative config |
| --- | --- | --- |
| RAP baseline | RAP planner with VLM support available but no current rule-based merge logic | `configs/planners/rap_vlm_0428.yaml` |
| DrivoR baseline | DrivoR planner with VLM support available but no current rule-based merge logic | `configs/planners/drivor_vlm_0428.yaml` |

### Intervention variants

These are the current intervention-focused learned-planner variants. They use multiview intervention and front-only scoring.

| Method | What it is | Representative config |
| --- | --- | --- |
| RAP intervention | RAP with VLM intervention enabled | `configs/planners/rap_vlm_intervention_4cam_0428.yaml` |
| DrivoR intervention | DrivoR with VLM intervention enabled | `configs/planners/drivor_vlm_intervention_4cam_0428.yaml` |

### Curated instruction-following demos

These are narrow demo-task variants on top of the current NuScenes-backed HUGSIM setup. They are intentionally hand-curated and should be treated as presentation/demo paths, not as a benchmark.

Current supported demo tasks:
- `stop_at_target`
- `park_at_target`

Current demo scenarios:
- `configs/benchmark/nuscenes_demo/scene-0013-stop-demo.yaml`
- `configs/benchmark/nuscenes_demo/scene-0411-park-demo.yaml`

Current demo behavior:
- the scenario `task` block overrides the coarse `command` text for the VLM selector path
- `front.mp4` is task-annotated in demo mode and is expected to show a visible `STOP` or `PARK` marker
- demo runs also write `demo_summary.json`
- `stop_at_target` now terminates the rollout when the ego both:
  - reaches the target tolerance, and
  - is below the configured stop speed threshold
- `park_at_target` now uses a simple park-approach brake override and terminates the rollout when the ego both:
  - reaches the parking target tolerance, and
  - is below the configured park speed threshold

Current known-good demo outputs:
- RAP stop demo:
  - `/bigdata/aidan/outputs/benchmark/out/05_23_26/demo/rap/qwen3-vl-8b-instruct_nusc_rap_intervention_vlm_4_cam/scene-0013_stop_demo`
  - expected completion signal: `task_completion_reason = "stop_reached"`
- DrivoR park demo:
  - `/bigdata/aidan/outputs/benchmark/out/05_23_26/demo/drivor/qwen3-vl-8b-instruct_nusc_drivor_vlm/scene-0411_park_demo`
  - expected completion signal: `task_completion_reason = "park_reached"`

### Standalone rule-based planner

| Method | What it is | Config |
| --- | --- | --- |
| `rule_based` | HUGSIM adapter around the external Rule-Planner repo; runs without VLM selection by default | `configs/planners/rule_based_local_aidan.yaml` |

Important current behavior:
- `include_privileged_pipe` is enabled automatically for `rule_based`
- the adapter/transport path is working end-to-end
- the planner runs on CPU by default

### Choice A: mixed trajectory selection

Choice A is the direct merged-candidate design:
- learned planner generates its candidate trajectories
- rule-based planner generates additional candidate trajectories
- both sets are merged into one candidate pool
- the VLM directly selects from that mixed pool

Current active configs:

| Method | What it is | Config |
| --- | --- | --- |
| `drivor_impl_a` | DrivoR + rule-based candidate merge (Choice A) | `configs/planners/drivor_vlm_intervention_4cam_rule_merge_0522.yaml` |
| `rap_impl_a` | RAP + rule-based candidate merge (Choice A) | `configs/planners/rap_vlm_intervention_4cam_rule_merge_0522.yaml` |

Single-scene non-benchmark configs:
- `configs/planners/drivor_vlm_intervention_4cam_rule_merge.yaml`
- `configs/planners/rap_vlm_intervention_4cam_rule_merge.yaml`

Current Choice A candidate behavior:
- `candidate_limit = 10`
- `rule_based_merge.topk = 3`
- `include_default_candidates = false`

So the VLM usually sees roughly:
- `3` rule-based candidates
- `6-7` learned-planner candidates
- sometimes `1` carry-forward candidate

### Choice B: planner-gated selection

Choice B is the planner-gating design:
- learned and rule-based candidates are built separately
- a planner-gating VLM first chooses `learned` or `rule_based`
- then the lower-level selector runs only on the chosen planner family

Current active configs:

| Method | What it is | Config |
| --- | --- | --- |
| `drivor_impl_b` | DrivoR + planner gating (Choice B) | `configs/planners/drivor_vlm_intervention_4cam_rule_gate_0522.yaml` |
| `rap_impl_b` | RAP + planner gating (Choice B) | `configs/planners/rap_vlm_intervention_4cam_rule_gate_0522.yaml` |

Single-scene non-benchmark configs:
- `configs/planners/drivor_vlm_intervention_4cam_rule_gate.yaml`
- `configs/planners/rap_vlm_intervention_4cam_rule_gate.yaml`

Current Choice B behavior:
- planner gate is enabled with `planner_gate_enabled: true`
- the planner gate is a VLM call
- the lower-level selector is still another VLM stage
- so current Choice B is a **two-stage VLM flow**, not a single-call low-token switcher

## 3. Common Run Patterns

### Single-scene baseline / learned-planner run

Use this pattern for a single scene with an explicit planner config.

```bash
REPO_ROOT=/bigdata/aidan/HUGSIM \
PLANNER_NAME=drivor_vlm \
PLANNER_PATH=configs/planners/drivor_vlm_0428.yaml \
SCENARIO_PATH=configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml \
BASE_PATH=configs/sim/nuscenes_base_local.yaml \
SIM_CUDA=inherit \
AD_CUDA=inherit \
sbatch --partition=gpu02 scripts/run_single_scene.slurm
```

Swap `PLANNER_NAME` / `PLANNER_PATH` for RAP:

```bash
REPO_ROOT=/bigdata/aidan/HUGSIM \
PLANNER_NAME=rap_vlm \
PLANNER_PATH=configs/planners/rap_vlm_0428.yaml \
SCENARIO_PATH=configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml \
BASE_PATH=configs/sim/nuscenes_base_local.yaml \
RAP_DEVICE_OVERRIDE=cuda:0 \
PLANNER_VLM_DEVICE_OVERRIDE=cuda:1 \
SIM_CUDA=inherit \
AD_CUDA=inherit \
sbatch --partition=gpu02 --gres=gpu:2 scripts/run_single_scene.slurm
```

### Single-scene standalone `rule_based`

```bash
REPO_ROOT=/bigdata/aidan/HUGSIM \
PLANNER_NAME=rule_based \
PLANNER_PATH=configs/planners/rule_based_local_aidan.yaml \
SCENARIO_PATH=configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml \
BASE_PATH=configs/sim/nuscenes_base_local.yaml \
INCLUDE_PRIVILEGED_PIPE=true \
AD_CUDA=-1 \
SIM_CUDA=inherit \
sbatch --partition=gpu02 scripts/run_single_scene.slurm
```

### Single-scene Choice A

DrivoR Choice A:

```bash
REPO_ROOT=/bigdata/aidan/HUGSIM \
PLANNER_NAME=drivor_vlm \
PLANNER_PATH=configs/planners/drivor_vlm_intervention_4cam_rule_merge.yaml \
SCENARIO_PATH=configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml \
BASE_PATH=configs/sim/nuscenes_base_local.yaml \
SIM_CUDA=inherit \
AD_CUDA=inherit \
sbatch --partition=gpu02 scripts/run_single_scene.slurm
```

RAP Choice A:

```bash
REPO_ROOT=/bigdata/aidan/HUGSIM \
PLANNER_NAME=rap_vlm \
PLANNER_PATH=configs/planners/rap_vlm_intervention_4cam_rule_merge.yaml \
SCENARIO_PATH=configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml \
BASE_PATH=configs/sim/nuscenes_base_local.yaml \
RAP_DEVICE_OVERRIDE=cuda:0 \
PLANNER_VLM_DEVICE_OVERRIDE=cuda:1 \
SIM_CUDA=inherit \
AD_CUDA=inherit \
sbatch --partition=gpu02 --gres=gpu:2 scripts/run_single_scene.slurm
```

### Single-scene Choice B

DrivoR Choice B:

```bash
REPO_ROOT=/bigdata/aidan/HUGSIM \
PLANNER_NAME=drivor_vlm \
PLANNER_PATH=configs/planners/drivor_vlm_intervention_4cam_rule_gate.yaml \
SCENARIO_PATH=configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml \
BASE_PATH=configs/sim/nuscenes_base_local.yaml \
SIM_CUDA=inherit \
AD_CUDA=inherit \
sbatch --partition=gpu02 scripts/run_single_scene.slurm
```

RAP Choice B:

```bash
REPO_ROOT=/bigdata/aidan/HUGSIM \
PLANNER_NAME=rap_vlm \
PLANNER_PATH=configs/planners/rap_vlm_intervention_4cam_rule_gate.yaml \
SCENARIO_PATH=configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml \
BASE_PATH=configs/sim/nuscenes_base_local.yaml \
RAP_DEVICE_OVERRIDE=cuda:0 \
PLANNER_VLM_DEVICE_OVERRIDE=cuda:1 \
SIM_CUDA=inherit \
AD_CUDA=inherit \
sbatch --partition=gpu02 --gres=gpu:2 scripts/run_single_scene.slurm
```

## 4. Dataset and Benchmark Setup

### Dataset inference

`scripts/run_single_nuscenes_scene.sh` infers `data_type` from the scenario YAML and maps it to:
- base config
- camera config

Current supported dataset types in that script:
- `nuscenes`
- `waymo`
- `kitti360`

### Shared dataset layout on this server

The current cluster layout follows the same split for all supported datasets:
- benchmark scenario YAMLs live separately from processed HUGSIM scene assets
- `cfg.base.model_base` should point at the processed HUGSIM scene root
- the scenario YAML's `scene_name` is what gets resolved under that processed root

Current shared roots:

| Dataset | Scenario YAML root | Processed HUGSIM scene root | Local base config |
| --- | --- | --- | --- |
| `nuscenes` | `configs/benchmark/nuscenes_all_variants` and related repo-local scenario dirs | `/bigdata/datasets/HUGSIM/scenes/nuscenes` | `configs/sim/nuscenes_base_local.yaml` and benchmark-specific local variants |
| `waymo` | `/bigdata/datasets/HUGSIM/waymo` | `/bigdata/datasets/HUGSIM/scenes/waymo` | `configs/sim/waymo_base_local.yaml` |
| `kitti360` | `/bigdata/datasets/HUGSIM/kitti360` | `/bigdata/datasets/HUGSIM/scenes/kitti360` | `configs/sim/kitti360_base_local.yaml` |

Important current behavior for `waymo` and `kitti360`:
- some processed scenes are nested one level deeper inside the processed root
- [closed_loop.py](closed_loop.py) now resolves scene folders recursively under `cfg.base.model_base`
- scene-local processed `cfg.yaml` files can still contain stale absolute paths from older machines; runtime now preserves the resolved local `cfg.model_path`
- `sim/utils/sim_utils.py` now tolerates processed camera metadata that does not already expose nuScenes-style names like `CAM_FRONT`
- `load_HD_map` is currently false in the shared Waymo/KITTI scenario YAMLs used for these runs

### Current 3-scene smoke-test launcher

Use this script for small dataset verification runs on the shared Waymo/KITTI assets:
- `scripts/submit_dataset_3easy.sh`

Current fixed scene-set files:
- `configs/benchmark/scene_sets/waymo_3easy.txt`
- `configs/benchmark/scene_sets/kitti360_3easy.txt`

Supported planner families in that launcher:

| `PLANNER_NAME` | Default planner config |
| --- | --- |
| `rap_vlm` | `configs/planners/rap_vlm_intervention_4cam_0428.yaml` |
| `drivor_vlm` | `configs/planners/drivor_vlm_intervention_4cam_0428.yaml` |
| `rule_based` | `configs/planners/rule_based_local_aidan.yaml` |

Example commands:

```bash
bash scripts/submit_dataset_3easy.sh waymo drivor_vlm
```

```bash
bash scripts/submit_dataset_3easy.sh kitti360 drivor_vlm
```

```bash
bash scripts/submit_dataset_3easy.sh waymo rule_based
```

```bash
bash scripts/submit_dataset_3easy.sh kitti360 rule_based
```

If no planner config is passed explicitly, the submit wrapper chooses the default config from `PLANNER_NAME`.

### Verified dataset support status

The following planner families have already completed end-to-end 3-scene runs on shared `waymo` and `kitti360` HUGSIM assets:
- `rap_vlm`
- `drivor_vlm`
- `rule_based`

That means the current support path is operational for:
- dataset inference from scenario YAML
- base/camera config selection
- processed scene resolution under `/bigdata/datasets/HUGSIM/scenes/...`
- camera metadata handling for Waymo/KITTI processed scenes
- planner launch and evaluation output generation

### Main benchmark scenario set

Current full benchmark scenario directory:
- `configs/benchmark/nuscenes_all_variants`

Current full benchmark base config:
- `configs/sim/nuscenes_base_local_0522_full.yaml`

This file is a fully explicit local base config. It is not just an inheritance stub. That matters because the benchmark path loads it directly.

### Output layout

General convention:
- `/bigdata/aidan/outputs/benchmark/out/<date>/<method>/<scene>/`

Each completed scene usually contains:
- `eval.json`
- `output.txt`
- `front.mp4`
- `video.mp4`
- planner-specific logs such as `rap_client.log`, `drivor_client.log`, or `rule_based_client.log`

### Full 4-way benchmark

Submit wrapper:
- `scripts/submit_0522_full_4way_benchmark.sh`

Slurm runner:
- `scripts/run_0522_full_4way_benchmark.slurm`

Default benchmark methods:
- `drivor_impl_a`
- `drivor_impl_b`
- `rap_impl_a`
- `rap_impl_b`

Default GPU layout:
- `drivor_impl_a`: 1 GPU
- `drivor_impl_b`: 1 GPU
- `rap_impl_a`: 2 GPUs
- `rap_impl_b`: 2 GPUs