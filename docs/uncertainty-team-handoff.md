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
- Repeated five-route controls completed as jobs `33519`-`33522` with seed 17
  and deterministic PyTorch settings. The live closed loop still diverges at
  frame 2 within both conditions, so single paired-route score differences are
  not causal estimates.
- Expanded passive collection completed across 38 easy/medium routes from 19
  independent NuScenes scenes. Jobs `33523` and `33525` produced 1,911 valid
  frames; `33525` filled routes blocked by a missing shared unicycle checkpoint.
- Every expanded-corpus frame records 64 learned proposals, uses the full
  64-member distribution for uncertainty, has `policy_mode=observe`, has
  `affects_candidate_allocation=false`, selects the DrivoR argmax, and includes
  a post-step execution outcome.
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

## Repeated seeded controls (2026-08-19)

Two uncertainty-off and two passive-observe repeats used the same five routes,
DrivoR checkpoint, controller, GPU assignment, seed 17, deterministic PyTorch
algorithms, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

| metric | off mean | passive mean | passive - off |
|---|---:|---:|---:|
| NC | 0.9096 | 0.9198 | +0.0102 |
| TTC | 0.8388 | 0.8399 | +0.0011 |
| PDMS | 0.8571 | 0.8595 | +0.0024 |
| RC | 0.5001 | 0.4963 | -0.0037 |
| HDScore | 0.4687 | 0.4681 | -0.0006 |

Terminal outcome categories agreed within each condition on all five routes.
However, every off/off and observe/observe route pair diverged at frame 2. Mean
selected-plan distance across a repeated route ranged from `0.119` to `1.845 m`
for off and `0.080` to `2.783 m` for observe. Seeding Python, NumPy, Torch CPU,
and Torch CUDA therefore does not make the HUGSIM renderer/simulator/planner
closed loop deterministic.

Passive monitoring averaged `2.550 s/frame`, compared with `2.286 s/frame` when
uncertainty was off, an approximately 11.5% wall-clock overhead. The mean score
deltas above are small relative to route-level and within-condition variation;
they do not show that passive uncertainty changes driving performance.

The complete control report is under
`uncertainty-repeatability-seed17-controls` in the benchmark output root.

## Expanded passive benchmark (2026-08-19)

The 38-route corpus covers 19 independent scene groups and 1,911 frames. Route
outcomes were 20 completions, 17 collisions, and one route departure. Mean
route metrics were NC `0.6814`, DAC `1.0000`, TTC `0.6071`, comfort `0.9920`,
PDMS `0.6260`, RC `0.5974`, and HDScore `0.5224`. These describe the passive
corpus; there is no matched 38-route uncertainty-off corpus from which to infer
a performance effect.

At a 20-frame prediction horizon, grouped five-fold cross-validation keeps both
difficulty variants of each base scene in the same fold. The predictive cohort
contains 1,614 currently-safe frames with a 14.5% future plan-risk rate.

| predictor | precision | recall | coverage | lift | OOF AUROC |
|---|---:|---:|---:|---:|---:|
| trained low-disagreement quadrant | 0.629 | 0.479 | 0.110 | 4.34x | n/a |
| legacy logistic features | 0.383 | 0.329 | 0.125 | 2.64x | 0.674 |
| recoverable feature set | 0.365 | 0.316 | 0.126 | 2.51x | 0.657 |
| raw-proposal feature set | 0.271 | 0.372 | 0.199 | 1.87x | 0.647 |

The trained quadrant detects 17/25 future-risk events (0.680 event recall), but
episode precision is `0.447`, false alerts are `2.64/min`, and median first
warning lead is only two frames. Fold precision and recall also vary materially
across scene families. The legacy fixed thresholds do not fire the intended
quadrant on this corpus, while the ungrouped calibration grid's 80%-recall
solution routes 100% of frames and is unusable.

**Activation decision:** keep uncertainty in passive mode. The expanded corpus
validates telemetry and risk enrichment, but not a sufficiently selective,
timely intervention policy. Do not apply `recommended_config.diff` or enable
active uncertainty from the ungrouped calibration report. A future active
canary must freeze a conservative policy selected with grouped training data
and evaluate repeated live runs because the simulator is not deterministic.

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

## Environment status and blockers

The originally configured shared HUGSIM assets were absent on `bolei-gpu02`:

- `/bigdata/datasets/HUGSIM/scenes/nuscenes`
- specifically a processed directory such as `scene-0010` containing
  `cfg.yaml`, `scene.pth`, `ground_param.pkl`, and `meta_data.json`

Slurm job `33038` attempted the one-scene passive smoke run on 2026-08-13 and
failed before simulator/planner startup with this missing-root error. No partial
benchmark artifact was produced. The common scene launcher now performs this
asset check before importing simulator or CUDA dependencies.

The official `XDimLab/HUGSIM` releases for all 19 easy/medium NuScenes scenes
and the referenced 3DRealCar models are now restored under
`/bigdata/aidan/HUGSIM-assets`. All 16 newly downloaded ZIPs passed archive and
Hugging Face LFS SHA256 verification. The shared unicycle checkpoint used by
the easy routes is installed in all 19 scene directories with the same hash.
The dedicated base config is
`configs/sim/nuscenes_base_local_aidan_assets.yaml`.

The common launcher now preflights the scene export, dynamic model files, and
scenario-referenced unicycle checkpoint. The calibration launcher also retains
all independent workers but returns failure when any route fails, preventing an
incomplete corpus from being reported as a successful Slurm job.

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

1. Provide the current shared paths or mount instructions for the
   AgenticDriving CARLA/Fail2Drive stack.
2. Confirm that evaluator plan-risk (NC/DAC/TTC), simulator collision, route
   completion, and timeout should remain separately reported outcomes.
3. Confirm the canonical AgenticDriving planner, route subset, random seeds,
   controller, and commit used for uncertainty A/B runs.
4. Confirm whether `coverage_rescue` remains the uncertainty intervention
   target in AgenticDriving or should be supplemented by future safety risk.
5. Confirm that base NuScenes scene ID is the canonical grouping key so
   difficulty variants never cross calibration folds.

## Next experiment sequence

1. Freeze a conservative candidate policy using grouped training scenes only;
   do not use the ungrouped 100%-coverage calibration recommendation.
2. Reserve untouched scene groups, or collect additional independent scenes,
   for a final passive shadow test of alert rate and lead time.
3. If the shadow gate is met, run a bounded active canary with braking/hold as
   the only initial intervention and retain verifier authority.
4. Use multiple baseline and active repeats per route, the same checkpoint,
   seed, controller, and GPU assignment, and report distributions rather than
   relying on paired equality.
5. Report PDMS components and route outcomes alongside uncertainty event
   precision/recall, intervention rate, false interventions per minute, and
   warning lead time.

Do not compare the new active policy against the old `calibration-20260626`
routes as if they were a paired causal A/B. That corpus remains valid for
feature screening, but its active policy, telemetry, and terminal-outcome
contract differ from the new implementation.
