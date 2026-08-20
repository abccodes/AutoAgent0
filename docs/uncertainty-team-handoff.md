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
- The five-route passive diagnostic completed as Slurm job `33508`, and its
  matched uncertainty-off run completed as job `33509`. Both jobs used the
  same DrivoR checkpoint, controller, scenarios, simulator configuration, and
  one worker per GPU.
- All 277 passive frames recorded 64 learned proposals, `policy_mode=observe`,
  `affects_candidate_allocation=false`, and a baseline proposal matching the
  DrivoR argmax. All selected sources were `drivor_argmax`; every frame has a
  post-step execution outcome.

## Five-route diagnostic (2026-08-19)

| route | uncertainty off | passive observe | PDMS off -> observe | RC off -> observe |
|---|---|---|---:|---:|
| scene-0013 easy | route complete | route complete | 1.0000 -> 0.8516 | 0.9940 -> 0.9913 |
| scene-0383 easy | route departure | route departure | 0.9894 -> 1.0000 | 0.5618 -> 0.5618 |
| scene-0383 extreme | collision | collision | 1.0000 -> 1.0000 | 0.2001 -> 0.2001 |
| scene-0383 hard | route departure | route departure | 1.0000 -> 1.0000 | 0.5618 -> 0.5482 |
| scene-0383 medium | collision | collision | 0.4257 -> 0.2890 | 0.1888 -> 0.1888 |

Mean PDMS was `0.8830` off and `0.8281` in passive observe mode. Mean RC was
`0.5013` off and `0.4980` in observe mode. Terminal outcome categories matched
on every route, but trajectories diverged after the first two frames even
though passive selection always matched DrivoR argmax. This single-repeat
comparison therefore does not identify an uncertainty effect: planner/runtime
nondeterminism and passive-monitor latency must be bounded with repeated
off/off and observe/observe controls before interpreting the score difference.

The passive corpus contains four future plan-risk events across only two base
scene groups. At a 20-frame horizon, grouped two-fold results were:

| predictor | AUROC | precision | recall | coverage |
|---|---:|---:|---:|---:|
| legacy logistic features | 0.435 | 0.158 | 0.214 | 0.304 |
| recoverable feature set | 0.461 | 0.000 | 0.000 | 0.080 |
| raw-proposal feature set | 0.485 | 0.000 | 0.000 | 0.064 |
| trained low-disagreement quadrant | n/a | 0.296 | 0.375 | 0.284 |

The trained quadrant detected 3/4 events, but with only two held-out groups
this is a diagnostic, not an estimate of generalization. The current fixed
thresholds sent 244/277 frames to `lean_rule_based` and 33/277 to
`rule_based_fallback`; they do not provide a useful calibrated partition on
this corpus. An ungrouped silhouette AUC of `0.689` is also not sufficient
evidence because variants of scene-0383 are strongly correlated.

## Current blockers

The originally configured shared HUGSIM assets were absent on `bolei-gpu02`:

- `/bigdata/datasets/HUGSIM/scenes/nuscenes`
- specifically a processed directory such as `scene-0010` containing
  `cfg.yaml`, `scene.pth`, `ground_param.pkl`, and `meta_data.json`

Slurm job `33038` attempted the one-scene passive smoke run on 2026-08-13 and
failed before simulator/planner startup with this missing-root error. No partial
benchmark artifact was produced. The common scene launcher now performs this
asset check before importing simulator or CUDA dependencies.

For the smoke and diagnostic reruns, the official `XDimLab/HUGSIM` releases of
`scene-0010.zip`, `scene-0013.zip`, `scene-0383.zip`, and the five referenced
3DRealCar models were restored under `/bigdata/aidan/HUGSIM-assets` and
verified. The dedicated base config is
`configs/sim/nuscenes_base_local_aidan_assets.yaml`. This restores only the
three listed NuScenes scenes; other routes still require their released scene
archives and any 3DRealCar models referenced by `plan_list`.

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

1. Run repeated off/off and observe/observe controls on the restored five-route
   diagnostic, then quantify within-condition variance and monitor overhead.
2. Restore enough independent scenes for at least five meaningful held-out
   scene groups; scene difficulty variants must remain in the same fold.
3. Collect the expanded passive corpus and select any threshold on grouped training
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
