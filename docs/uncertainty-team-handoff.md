# Uncertainty benchmarking handoff

## Current state

- HUGSIM uncertainty work is maintained on
  `aidan/uncertainty-offline-passive`.
- AgenticDriving's earlier core uncertainty work is published on
  `origin/aidan/dev` through commit `93c6adb`.
- The 94-route historical HUGSIM corpus has been reanalyzed with grouped
  cross-validation and event-level metrics.
- The historical fixed quadrant is a diagnostic reference only. Its frame
  precision is 0.238 at 7.0% coverage, and its event recall is 16/49.
- Passive observation mode, raw proposal telemetry, post-step execution
  outcomes, and exact evaluator NC provenance are implemented and tested.
- The restored-asset passive smoke run completed as Slurm job `33506`. All 73
  frames recorded 64 raw DrivoR proposals, full-distribution uncertainty,
  `affects_candidate_allocation: false`, argmax-matching baseline proposals,
  execution outcomes, and evaluator NC provenance. The route completed with
  PDMS `1.0000`, RC `0.9971`, and HDScore `0.9971`.

## Current blockers

The originally configured shared HUGSIM assets were absent on `bolei-gpu02`:

- `/bigdata/datasets/HUGSIM/scenes/nuscenes`
- specifically a processed directory such as `scene-0010` containing
  `cfg.yaml`, `scene.pth`, `ground_param.pkl`, and `meta_data.json`

Slurm job `33038` attempted the one-scene passive smoke run on 2026-08-13 and
failed before simulator/planner startup with this missing-root error. No partial
benchmark artifact was produced. The common scene launcher now performs this
asset check before importing simulator or CUDA dependencies.

For the smoke rerun, the official `XDimLab/HUGSIM` release of
`scene-0010.zip` was restored under
`/bigdata/aidan/HUGSIM-assets/scenes/nuscenes/scene-0010` and verified against
the release LFS SHA-256. The dedicated base config is
`configs/sim/nuscenes_base_local_aidan_assets.yaml`. This restores only the
static easy-route dependency; broader scenarios still require their released
scene archives and any 3DRealCar models referenced by `plan_list`.

The first restored-asset rerun, Slurm job `33502`, passed scene preflight but
timed out after 900 seconds while preloading the unrelated Qwen3-VL-8B worker
from shared storage. The telemetry smoke config
`configs/planners/autoagent0/calibration/drivor_uncertainty_observe_smoke.yaml`
disables VLM selection/intervention only; the full passive collection config
remains unchanged.

Slurm job `33504` exposed a second measurement issue: passive telemetry was
present, but intra-learned uncertainty used the VLM candidate shortlist rather
than the planner's full proposal distribution. Job `33506` is the corrected
reference smoke run. Do not use `33504` to calibrate uncertainty thresholds.

The current AgenticDriving CARLA/Fail2Drive checkout also references paths that
are absent on this machine:

- `/data2/jiageng/fail2drive`
- `/data2/agenticdriving/f2d_carla`
- `/data2/jiageng/DrivoR`

## Decisions needed from the team

1. Provide the current shared paths or mount instructions for the HUGSIM
   NuScenes assets and the AgenticDriving CARLA/Fail2Drive stack.
2. Confirm that evaluator plan-risk (NC/DAC/TTC), simulator collision, route
   completion, and timeout should remain separately reported outcomes.
3. Confirm the canonical AgenticDriving planner, route subset, random seeds,
   controller, and commit used for uncertainty A/B runs.
4. Confirm whether `coverage_rescue` remains the uncertainty intervention
   target in AgenticDriving or should be supplemented by future safety risk.
5. Select the held-out scene grouping key used for calibration and final
   reporting so variants of one base scene never cross folds.

## Run sequence after assets are restored

1. Run one passive `scene-0010-easy-00` smoke route and validate every recorded
   frame has raw proposals, `affects_candidate_allocation: false`, and an
   `execution_outcome`.
2. Run a small passive diagnostic set containing object-overlap,
   background-only, detected, missed, and false-alert cases.
3. Collect the full passive corpus and select any threshold on grouped training
   scenes only.
4. Freeze the policy and run paired baseline-versus-active routes with the same
   planner checkpoint, scenario, seed, controller, and simulator configuration.
5. Report PDMS components and route outcomes alongside uncertainty event
   precision/recall, intervention rate, false interventions per minute, and
   warning lead time.

Do not compare the new active policy against the old `calibration-20260626`
routes as if they were a paired causal A/B. That corpus remains valid for
feature screening, but its active policy, telemetry, and terminal-outcome
contract differ from the new implementation.
