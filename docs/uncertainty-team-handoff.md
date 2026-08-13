# Uncertainty benchmarking handoff

## Current state

- HUGSIM uncertainty work is published on
  `aidan/uncertainty-offline-passive` through commit `15768e5`.
- AgenticDriving's earlier core uncertainty work is published on
  `origin/aidan/dev` through commit `93c6adb`.
- The 94-route historical HUGSIM corpus has been reanalyzed with grouped
  cross-validation and event-level metrics.
- The historical fixed quadrant is a diagnostic reference only. Its frame
  precision is 0.238 at 7.0% coverage, and its event recall is 16/49.
- Passive observation mode, raw proposal telemetry, post-step execution
  outcomes, and exact evaluator NC provenance are implemented and tested.

## Current blockers

Live HUGSIM collection cannot start because the configured shared assets are
absent on `bolei-gpu02`:

- `/bigdata/datasets/HUGSIM/scenes/nuscenes`
- specifically a processed directory such as `scene-0010` containing
  `cfg.yaml`, `scene.pth`, `ground_param.pkl`, and `meta_data.json`

Slurm job `33038` attempted the one-scene passive smoke run on 2026-08-13 and
failed before simulator/planner startup with this missing-root error. No partial
benchmark artifact was produced. The common scene launcher now performs this
asset check before importing simulator or CUDA dependencies.

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
