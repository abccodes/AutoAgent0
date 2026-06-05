# Rules

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.


# HUGSIM Methods and Run Guide

## 1. Core Execution Model

Most runs in this repo eventually go through:
- [scripts/baselines/common/run_single_scene.slurm](scripts/baselines/common/run_single_scene.slurm)
- [scripts/baselines/common/run_single_nuscenes_scene.sh](scripts/baselines/common/run_single_nuscenes_scene.sh)
- [closed_loop.py](closed_loop.py)

Legacy compatibility wrappers still exist at:
- `scripts/run_single_scene.slurm`
- `scripts/run_single_nuscenes_scene.sh`
- `scripts/run_scene_array_item.slurm`

Those root-level files should be treated as shims. The canonical entrypoints now live under `scripts/baselines/...`.

The main environment variables used across runs are:
- `PLANNER_NAME`: high-level planner family, usually `rap_vlm`, `drivor_vlm`, or `rule_based`
- `PLANNER_PATH`: planner config YAML under `configs/planners/`
- `SCENARIO_PATH`: scenario YAML under `configs/benchmark/...`
- `BASE_PATH`: sim/base config under `configs/sim/`
- `SIM_CUDA`: simulator GPU, usually `inherit` under Slurm
- `AD_CUDA`: planner-side CUDA selection, usually `inherit`
- `RAP_DEVICE_OVERRIDE`, `DRIVOR_DEVICE_OVERRIDE`: planner model device override
- `PLANNER_VLM_DEVICE_OVERRIDE`: VLM worker device override
- `BASELINE_ID`: canonical baseline key from `configs/baselines/registry.yaml`
- `BASELINE_SUITE`: benchmark bucket such as `full` or `3easy`
- `BASELINE_RUN_TYPE`: one of `canonical` or `debug`
- `BASELINE_RUN_VARIANT`: optional canonical variant label
- `BENCHMARK_OUTPUT_ROOT_OVERRIDE`: explicit per-run output root used by canonical launchers

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
  - the intervention gate judges the learned base policy first
  - only intervention-triggered frames send the combined learned + rule-based pool to the VLM scorer
- **Choice B / `*_rule_gate*`**
  - primary backend is still RAP or DrivoR
  - learned and rule-based candidate families are built separately
  - a planner-gating VLM runs every frame and chooses which family to trust
  - after the gate decision, the system takes the chosen family's top/default trajectory directly

## 2. Current Method Catalog

### AutoAgent0 architecture layer

AutoAgent0 preparation code now lives under:
- `autoagent0/core/`
- `autoagent0/experts/`
- `autoagent0/adapters/hugsim/`
- `autoagent0/prompts/`

This is currently an agentic boundary around the existing HUGSIM pipeline.
Most existing method paths remain behavior-preserving. The opt-in
`rap_autoagent0` / `drivor_autoagent0` methods add the first active
VLM-critic-driven recovery loop. HUGSIM remains the evaluation backend,
`planners/` remain the RAP/DrivoR/rule-based backend adapters, and existing
configs remain the source of truth for current baselines.

Current shared-code ownership:
- `autoagent0/core/candidates.py` owns candidate summaries, candidate-row
  formatting, path-length helpers, and planner-gate candidate filtering
- `autoagent0/adapters/hugsim/context.py` owns route/task/camera/ego context
  helpers used by HUGSIM-facing VLM paths
- `autoagent0/prompts/orchestrator.py` owns the current scoring,
  intervention, and planner-gate prompt builders for legacy methods
- `autoagent0/prompts/critic.py` owns the active `*_autoagent0`
  single-candidate critique prompt
- `autoagent0/prompts/planner.py` owns the active `*_autoagent0`
  revised-candidate final selection prompt
- `autoagent0/prompts/designer.py` owns the design-change prompt boundary for
  future dynamic designer requests
- `autoagent0/core/orchestrator.py` owns VLM decision coercion, score parsing,
  and selected-candidate reasoning helpers
- `autoagent0/experts/rule_based.py` wraps the existing Rule-Planner provider
  without moving Edmund's external implementation into this repo

`planners/common/vlm_selector.py` remains the active runtime integration point
and compatibility facade. Do not add new shared helper logic directly to that
file unless it is HUGSIM/VLM-selector-specific; prefer adding reusable code
under `autoagent0/` and re-exporting it through the facade only when needed for
existing imports.

Current method mapping:
- solo VLM intervention -> learned-intervention agent flow
- Choice A / `*_rule_merge*` -> rule-merge Designer + Orchestrator flow
- Choice B / `*_rule_gate*` -> policy-gate Orchestrator flow
- standalone `rule_based` -> rule-based expert baseline
- `*_autoagent0` -> opt-in agentic recovery-loop prototype

The phase-1 deterministic verifier object is passive and always accepts on
behavior-preserving paths. It is exposed through debug-only `agent_trace` fields
and must not alter selected trajectories, fallbacks, metrics, or launcher
behavior. In the active `*_autoagent0` prototype, critique/rejection is handled
by the dedicated AutoAgent0 VLM Critic prompt rather than by the passive
verifier.

The active `rap_autoagent0` / `drivor_autoagent0` prototype is separate from
Method A/B. It first critiques one learned default trajectory with the
AutoAgent0 VLM Critic. Only if that critique requests redesign does it ask for
expanded learned + rule-based candidates, use the AutoAgent0 Planner prompt to
select a revised candidate, critique the revised selection once more, and then
execute the revised selection once the configured one-redesign limit is reached.

Frame-level VLM debug JSON now includes `agent_trace` where the current VLM
paths already write debug artifacts. This trace is diagnostic only and records:
- designer candidate counts by source
- orchestrator decision type
- selected source or selected planner family
- passive verifier status on behavior-preserving paths
- critique/redesign/fallback phases on active `*_autoagent0` paths

See `docs/autoagent0-architecture.md` for the short architecture guide.

### Canonical baseline registry

Canonical baseline definitions now live in:
- `configs/baselines/registry.yaml`
- `configs/baselines/validated_runs.yaml`
- `docs/baseline-management.md`

The registry defines:
- stable `baseline_id` values
- canonical planner config paths
- canonical dataset/suite support
- canonical output roots
- historical roots that should eventually migrate into canonical or archive trees

The validated-runs manifest is the source of truth for:
- which current outputs are known-good and trusted
- which historical outputs are valid base-policy baselines versus valid intervention baselines
- which important baselines are still pending rerun under current semantics

The short operational guide for maintaining this structure lives in:
- `docs/baseline-management.md`

Current main baseline IDs:
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

### Baseline learned planners

These are the current learned-planner baselines without the rule-based merge variants.

| Method | What it is | Representative config |
| --- | --- | --- |
| RAP baseline | RAP planner with VLM support available but no current rule-based merge logic | `configs/planners/basepolicy/rap_vlm_0428.yaml` |
| DrivoR baseline | DrivoR planner with VLM support available but no current rule-based merge logic | `configs/planners/basepolicy/drivor_vlm_0428.yaml` |

### Intervention variants

These are the current intervention-focused learned-planner variants. They use multiview intervention and front-only scoring.

| Method | What it is | Representative config |
| --- | --- | --- |
| RAP intervention | RAP with VLM intervention enabled | `configs/planners/vlm_intervention/rap_vlm_intervention_4cam_0428.yaml` |
| DrivoR intervention | DrivoR with VLM intervention enabled | `configs/planners/vlm_intervention/drivor_vlm_intervention_4cam_0428.yaml` |

### Curated instruction-following demos

These are narrow demo-task variants on top of the current NuScenes-backed HUGSIM setup. They are intentionally hand-curated and should be treated as presentation/demo paths, not as a benchmark.

Current supported demo tasks:
- `stop_at_target`
- `park_at_target`

Current demo scenarios:
- `configs/benchmark/nuscenes_demo/scene-0038-stop-demo.yaml`
- `configs/benchmark/nuscenes_demo/scene-0411-park-demo.yaml`

Current demo behavior:
- the scenario `task` block overrides the coarse `command` text for the VLM selector path
- `front.mp4` is task-annotated in demo mode and is expected to show a visible `STOP` or `PARK` marker
- demo runs also write `demo_summary.json`
- `stop_at_target` now terminates the rollout when the ego both:
  - reaches the target tolerance, and
  - is below the configured stop speed threshold
- `park_at_target` is currently a forward pull-over / parking-like stop task, not a reverse parking maneuver
- `park_at_target` uses a simple park-approach brake override and terminates the rollout when the ego both:
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
| `rule_based` | HUGSIM adapter around the external Rule-Planner repo; runs without VLM selection by default | `configs/planners/rule_based/rule_based_local_aidan.yaml` |

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
| `drivor_impl_a` | DrivoR + rule-based candidate merge (Choice A) | `configs/planners/choice_a_rule_merge/drivor_vlm_intervention_4cam_rule_merge_0522.yaml` |
| `rap_impl_a` | RAP + rule-based candidate merge (Choice A) | `configs/planners/choice_a_rule_merge/rap_vlm_intervention_4cam_rule_merge_0522.yaml` |

Current Choice A candidate behavior:
- `candidate_limit = 10`
- `rule_based_merge.topk = 3`
- `include_default_candidates = false`

Choice A runtime behavior:
- the intervention gate sees the learned base-policy default only
- if `should_intervene = false`, the run keeps the learned base policy and skips the scorer
- if `should_intervene = true`, the scorer usually sees roughly:
  - `3` rule-based candidates
  - `6-7` learned-planner candidates
  - sometimes `1` carry-forward candidate

### Choice B: planner-gated selection

Choice B is the planner-gating design:
- learned and rule-based candidates are built separately
- a planner-gating VLM runs every frame and chooses `learned` or `rule_based`
- if the gate chooses `learned`, the system uses the learned planner's top/default trajectory directly
- if the gate chooses `rule_based`, the system uses the rule-based planner's top/default scored trajectory directly
- if the gate fails, the system falls back to the learned base policy

Current active configs:

| Method | What it is | Config |
| --- | --- | --- |
| `drivor_impl_b` | DrivoR + planner gating (Choice B) | `configs/planners/choice_b_rule_gate/drivor_vlm_intervention_4cam_rule_gate_0522.yaml` |
| `rap_impl_b` | RAP + planner gating (Choice B) | `configs/planners/choice_b_rule_gate/rap_vlm_intervention_4cam_rule_gate_0522.yaml` |

### Canonical script and config layout

Active planner configs are now grouped by method family:
- `configs/planners/basepolicy/`
- `configs/planners/vlm_intervention/`
- `configs/planners/rule_based/`
- `configs/planners/choice_a_rule_merge/`
- `configs/planners/choice_b_rule_gate/`
- `configs/planners/archive/`

Root-level `configs/planners/*.yaml` files are currently retained for compatibility and historical reference. New launcher work should prefer the method-family directories above.

Canonical launcher and helper paths are now:
- `scripts/baselines/common/`
- `scripts/baselines/smoke/`
- `scripts/baselines/full/`
- `scripts/baselines/migration/`
- `scripts/debug/`
- `scripts/archive/`

Root-level `scripts/*.sh` and `scripts/*.slurm` entrypoints are being preserved as compatibility wrappers where practical. New automation should prefer the canonical `scripts/baselines/...` paths.

Current Choice B behavior:
- planner gate is enabled with `planner_gate_enabled: true`
- the planner gate is a VLM call
- the planner gate sees both learned and rule-based candidate families
- the lower-level scorer is not used in Choice B anymore
- so current Choice B is an **always-on single-gate planner router**

### AutoAgent0 recovery-loop prototype

The active AutoAgent0 recovery-loop methods are:

| Method | What it is | Config |
| --- | --- | --- |
| `rap_autoagent0` | RAP default trajectory plus one bounded VLM-critic-triggered redesign pass | `configs/planners/autoagent0/rap_autoagent0_recovery_loop_0605.yaml` |
| `drivor_autoagent0` | DrivoR default trajectory plus one bounded VLM-critic-triggered redesign pass | `configs/planners/autoagent0/drivor_autoagent0_recovery_loop_0605.yaml` |

These methods are separate from Choice A/B. They do not always merge or always
gate learned and rule-based policies. Their current runtime flow is:
- request one learned default/top trajectory
- critique that one trajectory with the AutoAgent0 VLM Critic prompt
- execute the default trajectory if critique accepts it
- if critique requests redesign, build an expanded learned + rule-based pool
- use the AutoAgent0 Planner prompt to select one revised candidate
- critique the revised candidate once
- execute the revised candidate if accepted
- if the final critique still rejects, current runtime either falls back or
  executes the VLM Planner-selected revised candidate depending on the
  configured redesign limit behavior

Current limits:
- `redesign_candidate_budget = 10`
- `max_redesign_attempts = 3` is present in config, but repeated redesign
  iterations are not implemented yet; the current runtime performs one expanded
  redesign pass and uses this value only for final-rejection behavior
- no deterministic map/TTC/collision verifier is active yet
- no memory module is active yet
- no new rule-based scorer is active yet

Relevant config block:

```yaml
autoagent0:
  enabled: true
  mode: recovery_loop
  redesign_candidate_budget: 10
  max_redesign_attempts: 3
  fallback_mode: hold
```

## 3. Common Run Patterns

### Single-scene baseline / learned-planner run

Use this pattern for a single scene with an explicit planner config.

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

Swap `PLANNER_NAME` / `PLANNER_PATH` for RAP:

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

### Single-scene standalone `rule_based`

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

### Single-scene Choice A

DrivoR Choice A:

```bash
REPO_ROOT=/bigdata/aidan/HUGSIM \
PLANNER_NAME=drivor_vlm \
PLANNER_PATH=configs/planners/choice_a_rule_merge/drivor_vlm_intervention_4cam_rule_merge_0522.yaml \
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
PLANNER_PATH=configs/planners/choice_a_rule_merge/rap_vlm_intervention_4cam_rule_merge_0522.yaml \
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
PLANNER_PATH=configs/planners/choice_b_rule_gate/drivor_vlm_intervention_4cam_rule_gate_0522.yaml \
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
PLANNER_PATH=configs/planners/choice_b_rule_gate/rap_vlm_intervention_4cam_rule_gate_0522.yaml \
SCENARIO_PATH=configs/benchmark/nuscenes_all_variants/scene-0038-easy-00.yaml \
BASE_PATH=configs/sim/nuscenes_base_local.yaml \
RAP_DEVICE_OVERRIDE=cuda:0 \
PLANNER_VLM_DEVICE_OVERRIDE=cuda:1 \
SIM_CUDA=inherit \
AD_CUDA=inherit \
sbatch --partition=gpu02 --gres=gpu:2 scripts/run_single_scene.slurm
```

### Single-scene AutoAgent0 recovery loop

DrivoR AutoAgent0:

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

RAP AutoAgent0:

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
- `scripts/baselines/smoke/submit_dataset_3easy.sh`

Legacy shim:
- `scripts/submit_dataset_3easy.sh`

Current fixed scene-set files:
- `configs/benchmark/scene_sets/waymo_3easy.txt`
- `configs/benchmark/scene_sets/kitti360_3easy.txt`

Supported planner families in that launcher:

| `PLANNER_NAME` | Default planner config |
| --- | --- |
| `rap_vlm` | `configs/planners/vlm_intervention/rap_vlm_intervention_4cam_0428.yaml` |
| `drivor_vlm` | `configs/planners/vlm_intervention/drivor_vlm_intervention_4cam_0428.yaml` |
| `rule_based` | `configs/planners/rule_based/rule_based_local_aidan.yaml` |

The canonical smoke launcher is baseline-driven. Preferred current usage is:

```bash
bash scripts/baselines/smoke/submit_dataset_3easy.sh waymo drivor_intervention_4cam debug
```

```bash
bash scripts/baselines/smoke/submit_dataset_3easy.sh kitti360 rap_intervention_4cam debug
```

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

### One-scene all-method smoke suite

Use this after large refactors to verify that every canonical method still runs
on one easy scene:
- `scripts/baselines/smoke/submit_method_smoke.sh`
- `scripts/baselines/smoke/run_method_smoke.slurm`
- `scripts/baselines/smoke/check_method_smoke.py`

Default behavior:
- dataset: `nuscenes`
- scene: `configs/benchmark/nuscenes_all_variants/scene-0010-easy-00.yaml`
- suite: `full`
- GPU allocation: `6`
- methods: all 9 canonical baseline IDs:
  - `rap_vlm`
  - `drivor_vlm`
  - `rap_intervention_4cam`
  - `drivor_intervention_4cam`
  - `rule_based`
  - `rap_impl_a`
  - `drivor_impl_a`
  - `rap_impl_b`
  - `drivor_impl_b`

Run the default NuScenes smoke suite:

```bash
bash scripts/baselines/smoke/submit_method_smoke.sh
```

Switch to one easy Waymo or KITTI-360 scene while still running all 9 methods:

```bash
DATASET=waymo bash scripts/baselines/smoke/submit_method_smoke.sh
```

```bash
DATASET=kitti360 bash scripts/baselines/smoke/submit_method_smoke.sh
```

Useful dry-runs:

```bash
DRY_RUN=1 bash scripts/baselines/smoke/submit_method_smoke.sh
```

```bash
DATASET=waymo DRY_RUN=1 bash scripts/baselines/smoke/submit_method_smoke.sh
```

The smoke suite writes fresh debug outputs under:
- `/bigdata/aidan/outputs/benchmark/out/debug/<baseline_id>/<dataset>/<SMOKE_RUN_ID>/<scene>/`

The checker validates `eval.json`, `output.txt`, VLM debug artifacts when
expected, and `agent_trace` for Choice A/Choice B frame debug outputs.

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

Canonical convention:
- `/bigdata/aidan/outputs/benchmark/out/baselines/<baseline_id>/<dataset>/<suite>/<run_variant>/<scene>/`

Archive convention:
- `/bigdata/aidan/outputs/benchmark/out/archive/<baseline_id>/<dataset>/<suite>/<archive_reason>/<run_variant>/<scene>/`

Debug/smoke convention:
- `/bigdata/aidan/outputs/benchmark/out/debug/<baseline_id>/<dataset>/current/<scene>/`

Historical date-based output buckets should not be recreated.
The old date-based top-level roots have been moved under:
- `/bigdata/aidan/outputs/benchmark/out/archive/date_roots/`

Small one-off experimental roots now live under:
- `/bigdata/aidan/outputs/benchmark/out/archive/legacy_experiments/`

Each completed scene usually contains:
- `eval.json`
- `output.txt`
- `front.mp4`
- `video.mp4`
- planner-specific logs such as `rap_client.log`, `drivor_client.log`, or `rule_based_client.log`

Historical note for the `05_26_26` shared Waymo/KITTI extended runs:
- the original top-level output roots should be interpreted as **base-policy baselines**
- the intervention gate was invoked, but it failed to produce valid structured outputs and fell back on every frame
- so those are not valid successful intervention-plus-scorer baselines, even though their original run names used `*_extended`
- they should be preserved as historical **base-policy** baselines under the canonical baselines tree:
  - `/bigdata/aidan/outputs/benchmark/out/baselines/drivor_vlm/waymo/extended/drivor-base-policy-legacy`
  - `/bigdata/aidan/outputs/benchmark/out/baselines/drivor_vlm/kitti360/extended/drivor-base-policy-legacy`
  - `/bigdata/aidan/outputs/benchmark/out/baselines/rap_vlm/waymo/extended/rap-base-policy-legacy`
  - `/bigdata/aidan/outputs/benchmark/out/baselines/rap_vlm/kitti360/extended/rap-base-policy-legacy`

Duplicate policy for future migration work:
- keep the newest validated-correct run for an exact semantic baseline as the canonical baseline tree
- move older duplicates into `out/archive/.../replaced_duplicate/...`
- keep useful but uncertain historical outputs in `out/archive/.../historical_unverified/...`
- only delete a directory when it is clearly useless and already known invalid

### Full 4-way benchmark

Submit wrapper:
- `scripts/baselines/full/submit_nuscenes_full_baselines.sh`

Slurm runner:
- `scripts/baselines/full/run_nuscenes_full_baselines.slurm`

Legacy shims:
- `scripts/submit_0522_full_4way_benchmark.sh`
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




git commit -m "Canonicalize baseline outputs and launcher layout"
  git add \
    AGENTS.md \
    docs/baseline-management.md \
    configs/baselines/registry.yaml \
    configs/baselines/validated_runs.yaml \
    configs/planners/basepolicy/rap_vlm_0428.yaml \
    configs/planners/basepolicy/drivor_vlm_0428.yaml \
    configs/planners/vlm_intervention/rap_vlm_intervention_4cam_0428.yaml \
    configs/planners/vlm_intervention/drivor_vlm_intervention_4cam_0428.yaml \
    configs/planners/rule_based/rule_based_local_aidan.yaml \
    configs/planners/choice_a_rule_merge/rap_vlm_intervention_4cam_rule_merge_0522.yaml \
    configs/planners/choice_a_rule_merge/drivor_vlm_intervention_4cam_rule_merge_0522.yaml \
    configs/planners/choice_b_rule_gate/rap_vlm_intervention_4cam_rule_gate_0522.yaml \
    configs/planners/choice_b_rule_gate/drivor_vlm_intervention_4cam_rule_gate_0522.yaml \
    scripts/resolve_output_path.py \
    scripts/run_scene_array_item.slurm \
    scripts/run_single_nuscenes_scene.sh \
    scripts/run_single_scene.slurm \
    scripts/submit_dataset_3easy.sh \
    scripts/submit_0522_full_4way_benchmark.sh \
    scripts/run_0522_full_4way_benchmark.slurm \
    scripts/baselines/common/baseline_registry.py \
    scripts/baselines/common/resolve_output_path.py \
    scripts/baselines/common/run_scene_array_item.slurm \
    scripts/baselines/common/print_baseline_inventory.py \
    scripts/baselines/smoke/submit_dataset_3easy.sh \
    scripts/baselines/full/submit_nuscenes_full_baselines.sh \
    scripts/baselines/full/run_nuscenes_full_baselines.slurm \
    scripts/baselines/migration/plan_baseline_migration.py \
    scripts/baselines/migration/copy_validated_baselines.py
