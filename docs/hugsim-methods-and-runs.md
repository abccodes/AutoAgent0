# HUGSIM Methods And Runs

This document holds operational HUGSIM run details that used to live in
`AGENTS.md`. Keep `AGENTS.md` short; put method/run specifics here or in the
more focused docs linked below.

## Core Execution

Canonical run entrypoints:
- `scripts/baselines/common/run_single_scene.slurm`
- `scripts/baselines/common/run_single_nuscenes_scene.sh`
- `closed_loop.py`

Root-level launch files such as `scripts/run_single_scene.slurm`,
`scripts/run_single_nuscenes_scene.sh`, and `scripts/run_scene_array_item.slurm`
are compatibility shims. Prefer `scripts/baselines/...` for new automation.

Common environment variables:
- `PLANNER_NAME`: high-level planner family, usually `rap_vlm`, `drivor_vlm`, or
  `rule_based`.
- `PLANNER_PATH`: planner config YAML under `configs/planners/`.
- `SCENARIO_PATH`: scenario YAML under `configs/benchmark/...`.
- `BASE_PATH`: sim/base config under `configs/sim/`.
- `SIM_CUDA`, `AD_CUDA`: simulator and planner CUDA selection; usually
  `inherit` under Slurm.
- `RAP_DEVICE_OVERRIDE`, `DRIVOR_DEVICE_OVERRIDE`,
  `PLANNER_VLM_DEVICE_OVERRIDE`: explicit planner/VLM device overrides.
- `BASELINE_ID`, `BASELINE_SUITE`, `BASELINE_RUN_TYPE`,
  `BASELINE_RUN_VARIANT`: canonical baseline launcher metadata.
- `BENCHMARK_OUTPUT_ROOT_OVERRIDE`: explicit per-run output root.

Dataset inference happens in the run wrapper by reading the scenario YAML and
mapping dataset type to base/camera config.

## Current Method Families

Canonical baseline IDs:
- `rap_vlm`
- `drivor_vlm`
- `rap_intervention_4cam`
- `drivor_intervention_4cam`
- `rule_based`
- `rap_impl_a`
- `drivor_impl_a`
- `rap_impl_b`
- `drivor_impl_b`
- `rap_autoagent0`
- `drivor_autoagent0`

Method meanings:
- `rap_vlm`, `drivor_vlm`: learned base-policy baselines.
- `rap_intervention_4cam`, `drivor_intervention_4cam`: learned planner plus
  VLM intervention/scoring.
- `rule_based`: standalone external Rule-Planner adapter.
- `rap_impl_a`, `drivor_impl_a`: Method A / Choice A merged learned +
  rule-based candidate pool.
- `rap_impl_b`, `drivor_impl_b`: Method B / Choice B VLM planner gate between
  learned and rule-based families.
- `rap_autoagent0`, `drivor_autoagent0`: active bounded AutoAgent0
  recovery-loop prototypes.

AutoAgent0 architecture and design details live in:
- `docs/autoagent0-architecture.md`
- `docs/autoagent0-design.md`

Baseline registry/output-management details live in:
- `docs/baseline-management.md`
- `configs/baselines/registry.yaml`
- `configs/baselines/validated_runs.yaml`

Curated demo-task details live in:
- `docs/curated-demo-tasks.md`

## Common Single-Scene Runs

DrivoR base policy:

```bash
REPO_ROOT=/bigdata/aidan/HUGSIM \
PLANNER_NAME=drivor_vlm \
PLANNER_PATH=configs/planners/basepolicy/drivor_vlm_0428.yaml \
SCENARIO_PATH=configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml \
BASE_PATH=configs/sim/nuscenes_base_local.yaml \
SIM_CUDA=inherit \
AD_CUDA=inherit \
sbatch --partition=gpu02 scripts/run_single_scene.slurm
```

RAP base policy:

```bash
REPO_ROOT=/bigdata/aidan/HUGSIM \
PLANNER_NAME=rap_vlm \
PLANNER_PATH=configs/planners/basepolicy/rap_vlm_0428.yaml \
SCENARIO_PATH=configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml \
BASE_PATH=configs/sim/nuscenes_base_local.yaml \
RAP_DEVICE_OVERRIDE=cuda:0 \
PLANNER_VLM_DEVICE_OVERRIDE=cuda:1 \
SIM_CUDA=inherit \
AD_CUDA=inherit \
sbatch --partition=gpu02 --gres=gpu:2 scripts/run_single_scene.slurm
```

Standalone rule-based:

```bash
REPO_ROOT=/bigdata/aidan/HUGSIM \
PLANNER_NAME=rule_based \
PLANNER_PATH=configs/planners/rule_based/rule_based_local_aidan.yaml \
SCENARIO_PATH=configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml \
BASE_PATH=configs/sim/nuscenes_base_local.yaml \
INCLUDE_PRIVILEGED_PIPE=true \
AD_CUDA=-1 \
SIM_CUDA=inherit \
sbatch --partition=gpu02 scripts/run_single_scene.slurm
```

DrivoR AutoAgent0 recovery loop:

```bash
REPO_ROOT=/bigdata/aidan/HUGSIM \
PLANNER_NAME=drivor_vlm \
PLANNER_PATH=configs/planners/autoagent0/drivor_autoagent0_recovery_loop_0605.yaml \
SCENARIO_PATH=configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml \
BASE_PATH=configs/sim/nuscenes_base_local.yaml \
SIM_CUDA=inherit \
AD_CUDA=inherit \
sbatch --partition=gpu02 scripts/run_single_scene.slurm
```

RAP AutoAgent0 recovery loop:

```bash
REPO_ROOT=/bigdata/aidan/HUGSIM \
PLANNER_NAME=rap_vlm \
PLANNER_PATH=configs/planners/autoagent0/rap_autoagent0_recovery_loop_0605.yaml \
SCENARIO_PATH=configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml \
BASE_PATH=configs/sim/nuscenes_base_local.yaml \
RAP_DEVICE_OVERRIDE=cuda:0 \
PLANNER_VLM_DEVICE_OVERRIDE=cuda:1 \
SIM_CUDA=inherit \
AD_CUDA=inherit \
sbatch --partition=gpu02 --gres=gpu:2 scripts/run_single_scene.slurm
```

Swap `PLANNER_PATH` to the relevant config under these method-family
directories for intervention, Method A, and Method B runs:
- `configs/planners/vlm_intervention/`
- `configs/planners/choice_a_rule_merge/`
- `configs/planners/choice_b_rule_gate/`

## Smoke And Benchmark Runs

One-scene all-method smoke suite:

```bash
bash scripts/baselines/smoke/submit_method_smoke.sh
```

Switch smoke dataset:

```bash
DATASET=waymo bash scripts/baselines/smoke/submit_method_smoke.sh
DATASET=kitti360 bash scripts/baselines/smoke/submit_method_smoke.sh
```

Dry-run:

```bash
DRY_RUN=1 bash scripts/baselines/smoke/submit_method_smoke.sh
```

Shared Waymo/KITTI three-scene smoke:

```bash
bash scripts/baselines/smoke/submit_dataset_3easy.sh waymo drivor_intervention_4cam debug
bash scripts/baselines/smoke/submit_dataset_3easy.sh kitti360 rap_intervention_4cam debug
```

Full NuScenes four-way Method A/B benchmark:

```bash
bash scripts/baselines/full/submit_nuscenes_full_baselines.sh
```

Default full-benchmark methods:
- `drivor_impl_a`
- `drivor_impl_b`
- `rap_impl_a`
- `rap_impl_b`

## Dataset And Output Layout

Shared dataset roots:

| Dataset | Scenario YAML root | Processed HUGSIM scene root | Base config |
| --- | --- | --- | --- |
| `nuscenes` | `configs/benchmark/nuscenes_all_variants` | `/bigdata/datasets/HUGSIM/scenes/nuscenes` | `configs/sim/nuscenes_base_local.yaml` |
| `waymo` | `/bigdata/datasets/HUGSIM/waymo` | `/bigdata/datasets/HUGSIM/scenes/waymo` | `configs/sim/waymo_base_local.yaml` |
| `kitti360` | `/bigdata/datasets/HUGSIM/kitti360` | `/bigdata/datasets/HUGSIM/scenes/kitti360` | `configs/sim/kitti360_base_local.yaml` |

Canonical output convention:
- `/bigdata/aidan/outputs/benchmark/out/baselines/<baseline_id>/<dataset>/<suite>/<run_variant>/<scene>/`

Debug/smoke output convention:
- `/bigdata/aidan/outputs/benchmark/out/debug/<baseline_id>/<dataset>/current/<scene>/`

Archive convention:
- `/bigdata/aidan/outputs/benchmark/out/archive/<baseline_id>/<dataset>/<suite>/<archive_reason>/<run_variant>/<scene>/`

Completed scenes usually contain:
- `eval.json`
- `output.txt`
- `front.mp4`
- `video.mp4`
- planner logs such as `rap_client.log`, `drivor_client.log`, or
  `rule_based_client.log`

## Historical And Archived Runs

Do not recreate historical date-based top-level output buckets. Old date-based
roots were moved under:
- `/bigdata/aidan/outputs/benchmark/out/archive/date_roots/`

Small one-off experimental roots live under:
- `/bigdata/aidan/outputs/benchmark/out/archive/legacy_experiments/`

Historical note for the `05_26_26` shared Waymo/KITTI extended runs:
- treat those outputs as base-policy baselines, not successful intervention
  baselines;
- the intervention gate was invoked, but structured outputs failed and the
  system fell back on every frame;
- preserve them as historical base-policy runs unless a newer validated run
  replaces them.

Duplicate policy:
- keep the newest validated-correct run for a given semantic baseline;
- archive older duplicates under `out/archive/.../replaced_duplicate/...`;
- keep useful uncertain outputs under `out/archive/.../historical_unverified/...`;
- delete only when a directory is clearly useless and known invalid.
