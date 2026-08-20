# Agentic Uncertainty Technical Integration and Benchmark Plan

## Scope

This document is the working implementation and evaluation plan. The shorter
[uncertainty proposal](agentic-uncertainty-integration-proposal.md) is intended
for project and research-direction review.

## Technical objective

The immediate objective is a calibrated pre-selection coverage monitor for
AgenticDriving. It estimates when the normal PDMS candidate budget may miss an
admissible trajectory and, in active mode, evaluates more existing candidates.
It does not generate trajectories, trigger recovery, or bypass the verifier.

The implementation is partially complete. Recovery-stage uncertainty is
already merged in AgenticDriving, and a separate top-level CARLA monitor is
implemented locally. The remaining work is to port that monitor onto current
`main`, reconcile the duplicate policies, collect CARLA data, calibrate an
artifact, and run the active comparison.

## Relationship to the monitor baseline

The [closed-loop monitor baseline](closed-loop-monitor-baselines.md) wraps a
frozen planner with a fixed minimum-risk fallback. This proposal instead uses
the same signal family to decide when the full agentic harness should spend
additional evaluation or recovery effort.

| Property | Monitor baseline | This proposal |
|---|---|---|
| System | Frozen base planner | Full AgenticDriving harness |
| Response | Slowdown or stop | Expand PDMS coverage; later evaluate tool routing |
| Recovery | Fixed fallback | Existing verified recovery system |
| Execution authority | Monitor fallback rule | Existing final verifier |

## Source-of-truth implementation status

Status was verified on 2026-08-20 against AgenticDriving `main` at
`613120c`.

### Merged recovery-stage implementation

Current `main` contains:

- `agenticdriving/scorer/uncertainty.py` with intra-learned spread,
  cross-family disagreement, mode count, and danger-quadrant routing;
- `agenticdriving/scorer/selection_strategies/recovery.py`, which computes
  those signals after recovery selection begins and changes the learned versus
  rule-based redesign budget;
- `agenticdriving/calibration/` and `scripts/calibrate_uncertainty.py`;
- legacy `AGENTICDRIVING_UNCERTAINTY_*` configuration fields.

This code is real and merged, but it cannot warn during a healthy normal loop.
It also represents unavailable cross-family evidence as zero and uses
HUGSIM-derived routing thresholds, so it should not be treated as the final
CARLA monitor. The legacy uncertainty field defaults on inside its configuration
even though AgenticDriving itself defaults off; the top-level CARLA branch
changes that legacy default to off.

### Unmerged pre-selection CARLA implementation

Local commit `24f8c68` in `/bigdata/aidan/AgenticDriving-uncertainty` adds:

- `agenticdriving/uncertainty/monitor.py` with `off`, `observe`, and
  `active` modes;
- measurement after planner inference and before candidate scoring;
- an observe-only shadow scorer that preserves the baseline winner;
- calibrated hysteresis that expands the two PDMS candidate budgets;
- planner/perception/schema validation for calibration artifacts;
- config generation, grouped calibration, paired A/B analysis, documentation,
  and five focused unit tests.

The five tests pass in the RAP environment. No CARLA uncertainty run outputs or
calibrated artifact are present. The commit is based on `504d123` and must be
ported across eight newer `main` commits.

### Evidence boundary

| Claim | Status |
|---|---|
| HUGSIM passive telemetry works | Validated |
| HUGSIM signals enrich for future plan risk | Supported by grouped development results |
| Top-level CARLA monitor code works in focused unit tests | Validated |
| Observe mode preserves selection in a real CARLA run | Not yet tested |
| CARLA artifact predicts coverage rescue on held-out routes | Not yet tested |
| Active expansion improves route-level performance | Not yet tested |
| Uncertainty should trigger semantic reasoning or recovery | Future hypothesis |

## System role

The verifier and uncertainty monitor answer different questions:

- **Verifier:** Is this exact trajectory admissible now?
- **Uncertainty monitor:** Does the proposal distribution and recent history
  suggest that the normal planner/verifier loop needs more scrutiny?

```text
Planner proposals and scores
            |
            v
   pre-selection uncertainty
            |
     +------+------+
     |             |
     v             v
 normal PDMS    expanded PDMS
 coverage       coverage when active
     |             |
     +------+------+
            v
    selected trajectory
            |
            v
       final verifier
            |
       execute/recover
```

The merged implementation calculates uncertainty inside the recovery selector,
after the normal verifier or another independent trigger has already entered
recovery. Commit `24f8c68` adds the missing pre-selection placement: it
measures the original learned proposals before candidate scoring and can expand
the scoring budget without changing the verifier or recovery system.

During the port, the legacy recovery-routing switch should default off or be
explicitly deprecated. Otherwise two different uncertainty policies, targets,
and schemas can affect one episode and make the benchmark uninterpretable.

## Uncertainty method

### Inputs and timing

At planning tick `t`, the learned planner emits:

```text
C_t = {(tau_i, q_i)} for i = 1, ..., M
```

`tau_i` is a candidate trajectory in the common ego-local frame and `q_i` is
its native score. The CARLA monitor receives the original distribution before
candidate scoring. For bounded runtime, distribution features may use the
highest-scored `feature_candidate_limit` members, currently 32; this is logged
as a warning and does not truncate the planner or selector inputs.

`observe` mode is passive with respect to selection but runs an expanded
shadow scorer for the coverage-rescue label. `active` mode may change only the
candidate-scoring budget.

### 1. Intra-planner trajectory disagreement

The merged recovery implementation applies softmax directly to native scores.
The pre-selection CARLA monitor first standardizes scores to reduce sensitivity
to planner-specific score scale, then applies softmax:

```text
z_i = (q_i - mean(q)) / std(q)
w_i = exp(z_i) / sum_j exp(z_j)
```

For horizon step `h`, compute the weighted mean waypoint and weighted RMS
deviation:

```text
mu_h       = sum_i w_i * tau_i[h]
rms_h      = sqrt(sum_i w_i * ||tau_i[h] - mu_h||^2)
U_intra(t) = mean_h rms_h
```

`U_intra` measures concentration in trajectory space. High spread can be
healthy multimodality; low spread can be either an easy scene or confident
failure. It is therefore interpreted with alternative-family, score, and
temporal features rather than as a standalone safety score.

The report records candidate count, horizon, normalized entropy, effective
candidate count, and feature availability. In the top-level monitor,
insufficient proposals produce an explicit unavailable value rather than zero.

### 2. Alternative-family disagreement

The HUGSIM implementation compares the highest-scored learned trajectory with
the nearest independent Rule-Planner trajectory:

```text
U_cross(t) = min_r mean_h ||tau_learned[h] - tau_rule_r[h]||
```

The merged recovery code applies the same form to the learned default and its
generated rule-based candidate rows. The pre-selection CARLA branch generates
the existing recovery primitive pool once, caches it for later recovery use,
and records the nearest primitive distance as `primitive_distance_m`.

This is not equivalent to disagreement between the end-to-end and
target-region PDMS branch winners, because those winners can originate from the
same learned proposal distribution. That branch-winner feature remains a
possible later ablation, not part of commit `24f8c68`. Missing primitive
evidence is explicitly unavailable.

### 3. Proposal mode count

Flatten trajectories over the common horizon and run deterministic K-means for
candidate values of `k`. Select the `k` with sufficient silhouette
improvement over one mode. Both implementations test up to three modes by
default. The merged recovery implementation retains per-`k` silhouette
metadata; the top-level report currently emits only the selected mode count.

Mode count remains a calibration feature. In HUGSIM, unconditional mode-count
fallback rules routed too many frames and reduced precision.

### 4. Score, geometry, and temporal context

Log these complementary features:

- normalized score entropy and effective candidate count;
- normalized top-score margin;
- pairwise and endpoint proposal dispersion;
- change in the selected trajectory from the previous planning tick;
- nearest-primitive distance;
- planner identity, perception source, proposal count, and horizon.

Verifier outcome is attached after selection as a label and debug field. It is
not used as a pre-selection model feature, which avoids target leakage.

Expanded logistic feature sets did not outperform the simple HUGSIM rule, but
the features should be retained for CARLA calibration rather than assumed to
transfer.

### 5. Calibration targets

The strongest HUGSIM rule identified a low-disagreement region:

```text
consensus_risk = (U_intra <= T_intra) AND (U_cross <= T_cross)
```

This does **not** mean the model is uncertain in the usual high-variance sense.
It indicates confident agreement that has historically preceded a future
safety failure, potentially reflecting shared blind spots or mode collapse.

The HUGSIM classifier should remain a benchmark rather than a transferred
policy. CARLA calibration should compare:

1. Low-disagreement consensus risk.
2. High-disagreement rules.
3. Entropy and score-margin baselines.
4. A grouped logistic model.
5. Temporally persistent variants of each method.

The implemented CARLA script currently fits a balanced logistic model to a
different operational label:

```text
coverage_rescue =
    baseline winner fails the verifier threshold
    AND observe-mode expanded-budget winner passes
```

It produces leave-one-route-group-out probabilities and refuses to create an
artifact with fewer than 20 positive examples. Thresholds and models must be
trained only on training groups. Artifacts are bound to their schema, planner,
and perception source.

Future plan risk and coverage rescue answer different questions and must remain
separate labels. The first supports anticipation claims; the second directly
tests whether spending more PDMS compute can find an admissible candidate.

### 6. Coverage hysteresis

The implemented active policy uses two streak counters:

```text
BASELINE --N high-risk ticks--> EXPANDED
EXPANDED --M low-risk ticks--> BASELINE
```

The defaults are two high-risk ticks to activate and three low-risk ticks to
release. `observe` mode never enters `EXPANDED` for execution; it only runs
the expanded scorer as a shadow comparison. Cooldowns, simulator-time windows,
and route budgets are reasonable future additions but are not implemented in
the current branch.

### 7. Coverage response and authority

| Mode/state | Implemented response |
|---|---|---|
| `off` | Original candidate scoring |
| `observe` | Original scoring and selection plus expanded shadow scoring |
| `active`, baseline state | Original candidate budget |
| `active`, expanded state | Multiply existing end-to-end and target-region PDMS budgets by the configured factor |
| Selected plan rejected | Existing deterministic recovery path |

The implemented monitor cannot invoke semantic review or recovery. It cannot
bypass the verifier, declare a trajectory safe, or generate arbitrary
waypoints. Semantic or recovery triggering is a later experiment and should be
described separately from the coverage policy.

### Runtime contract

```text
FrameUncertaintyReport:
  schema_version, mode, planner_name, perception_source
  candidate_count, horizon_steps, horizon_seconds
  features, feature_availability
  risk_probability, risk_tier, policy_action
  baseline_candidate_budget, applied_candidate_budget
  estimator_latency_ms, warnings
```

After final verification, debug telemetry adds whether expanded shadow scoring
would have produced a `coverage_rescue`. In `observe` mode, applied and
baseline execution budgets remain equal, and selected output must match
`off` on every comparable frame.

## HUGSIM findings

### Repeated controls

Two monitor-off and two passive-observe repeats used the same five routes,
checkpoint, controller, seed, deterministic Torch settings, and GPU assignment.

| Metric | Off mean | Passive mean | Passive - off |
|---|---:|---:|---:|
| NC | 0.9096 | 0.9198 | +0.0102 |
| TTC | 0.8388 | 0.8399 | +0.0011 |
| PDMS | 0.8571 | 0.8595 | +0.0024 |
| RC | 0.5001 | 0.4963 | -0.0037 |
| HDScore | 0.4687 | 0.4681 | -0.0006 |
| Wall time per frame | 2.286 s | 2.550 s | +11.5% |

Every repeated route pair diverged by planning frame 2 despite deterministic
controls. The small score differences therefore do not establish a passive
performance effect. The tested implementation added 11.5% wall-time overhead.

### Expanded passive corpus

| Property | Result |
|---|---:|
| Routes | 38 |
| Independent NuScenes scenes | 19 |
| Frames | 1,911 |
| Route completions | 20 |
| Collisions | 17 |
| Route departures | 1 |
| Mean route PDMS | 0.6260 |
| Mean route RC | 0.5974 |
| Mean route HDScore | 0.5224 |

All frames satisfied the passive telemetry contract: 64 proposals were used,
the baseline DrivoR argmax was preserved, candidate allocation was unchanged,
and post-step outcomes were recorded.

### Grouped prediction results

The primary cohort contained 1,614 currently safe frames with a 14.5% future
plan-risk rate. Difficulty variants of each base scene remained in the same one
of five cross-validation folds.

| Predictor | Precision | Recall | Coverage | Lift | OOF AUROC |
|---|---:|---:|---:|---:|---:|
| Trained low-disagreement quadrant | 0.629 | 0.479 | 0.110 | 4.34x | n/a |
| Legacy logistic features | 0.383 | 0.329 | 0.125 | 2.64x | 0.674 |
| Recoverable feature set | 0.365 | 0.316 | 0.126 | 2.51x | 0.657 |
| Raw-proposal feature set | 0.271 | 0.372 | 0.199 | 1.87x | 0.647 |

| Event metric | Result |
|---|---:|
| Future-risk events detected | 17 / 25 |
| Event recall | 0.680 |
| Episode precision | 0.447 |
| False alerts | 2.64 / simulator minute |
| Median first-warning lead | 2 planning frames |

The mechanism enriches for future risk, but its precision, alert rate, and lead
time are not sufficient for direct intervention. Results validate passive
prediction and telemetry, not route-level benefit from an active policy.

## AgenticDriving integration plan

| Deliverable | Status |
|---|---|
| HUGSIM passive implementation and grouped analysis | Complete |
| Recovery-stage uncertainty in AgenticDriving | Merged in `main` |
| Pre-selection CARLA monitor, scripts, and focused tests | Implemented in local `24f8c68` |
| Port onto current AgenticDriving `main` | Not started |
| CARLA off/observe smoke routes | Not run |
| Passive Fail2Drive calibration corpus | Not collected |
| Planner/perception-specific CARLA artifact | Not fitted |
| Active candidate-coverage A/B | Not run |
| Semantic/recovery event experiment | Deferred |

### Phase 1: port and reconcile existing code

- Rebase or reapply `24f8c68` onto current AgenticDriving `main`.
- Resolve conflicts with the eight newer orchestrator and scoring commits.
- Retain pre-selection measurement, shadow scoring, configuration plumbing,
  scripts, tests, and documentation.
- Disable or deprecate legacy recovery-stage uncertainty routing by default;
  keep it available only as an explicit comparison if needed.
- Ensure primitive generation is cached once per tick and reused by recovery.
- Run the five focused tests plus the affected orchestrator/scoring suite.

**Exit:** one clearly owned uncertainty policy on current `main`, with all
relevant tests passing.

### Phase 2: CARLA off/observe smoke validation

- Run identical small route sets with `off` and `observe`.
- Confirm the normal candidate budget and selected trajectory are unchanged.
- Confirm the expanded scorer is shadow-only and `coverage_rescue` is emitted
  after verifier evaluation.
- Check feature availability, planner/perception identifiers, schema version,
  primitive generation, prediction JSON output, and estimator latency.
- Test each intended planner adapter before large collection.

**Exit:** complete CARLA telemetry, selection invariance on every comparable
frame, no control-path regression, and acceptable measurement overhead.

### Phase 3: passive CARLA/Fail2Drive calibration

- Collect paired ID and tail routes without changing behavior.
- Group route pairs, scene variants, and repetitions in the same fold.
- Use `coverage_rescue` as the implemented policy label and retain future
  verifier rejection, collision, drivable-area failure, route departure,
  deadlock, semantic takeover, and timeout as separate research labels.
- Calibrate planner families separately and reserve unseen routes as a lockbox.
- Require at least 20 positive coverage-rescue frames before fitting, then
  inspect prevalence and grouped out-of-fold quality before freezing an
  artifact.

**Exit:** a frozen event policy meets team-approved thresholds on unseen groups.

### Phase 4: active candidate-coverage A/B

Compare `off` with `active` using the frozen artifact. Active mode may
only expand the existing PDMS candidate batches after hysteresis. The final
winner follows the normal selection and verifier path.

**Exit:** acceptable intervention frequency and compute cost, with improved
held-out route utility or safety and no material regression in completion.

### Phase 5: optional agentic event experiment

Only after the coverage policy succeeds should uncertainty be allowed to
request asynchronous semantic review. Direct recovery should require an
independent confirmation signal and retain the existing verifier boundary.

**Exit:** a better compute/route tradeoff than periodic semantic review and no
increase in unnecessary recovery or timeout.

## Benchmark plan

### Experiment arms

| Arm | Behavior | Purpose |
|---|---|---|
| A. Current AgenticDriving | Uncertainty off | Route-level baseline |
| B. Passive observation | Log only | Invariance, latency, calibration |
| C. Active coverage | Expand PDMS candidate batches | Test implemented policy |
| D. Agentic event canary | Trigger semantic review or confirmed recovery | Future orchestration test |

Use the same checkpoint, controller, perception source, route, seed, and GPU
assignment across arms. Run multiple repetitions and report distributions.
Keep paired ID/tail routes and their repetitions in one statistical group.
Freeze code, configuration, calibration artifact, and route manifest before the
final comparison.

### Metrics

**Route performance:** Fail2Drive utility, route completion, collision and
route-departure rates, deadlock/timeout, PDMS and its component terms, and route
loss caused by unnecessary intervention.

**Coverage-monitor performance:** coverage-rescue prevalence, grouped AUROC and
average precision, precision/recall at the frozen threshold, active expansion
frequency and duration, expanded candidates scored per kilometer, rescues
actually selected, and estimator/PDMS/total latency.

**Future-risk analysis:** frame precision/recall/coverage/lift, event recall,
episode precision, false alerts per simulator minute, and warning lead. These
metrics are secondary until a CARLA future-risk policy is explicitly defined.

**Later agentic outcomes:** VLM calls per kilometer, recoveries requested,
admitted, rejected, completed, and aborted, unnecessary recovery rate, and time
to safe handoff. These do not apply to the initial coverage-expansion A/B.

### Proposed activation gate

Before collection, require complete telemetry and passive selection invariance
on every comparable smoke-test frame. Before fitting, require at least 20
coverage-rescue positives and enough independent route groups for out-of-group
evaluation. Before active testing, freeze the code, route manifest, artifact,
threshold, hysteresis, and expansion factor.

The active policy must have bounded intervention and latency and no credible
held-out route-utility regression. Exact precision, intervention-rate, and
compute limits should be declared after passive prevalence is measured, rather
than copied from HUGSIM. The earlier HUGSIM targets remain relevant only to a
later future-risk event experiment.

## Required ablations

1. Score/dispersion features only.
2. Primitive-distance feature only.
3. Mode count only.
4. HUGSIM low-disagreement rule without transferred thresholds.
5. Grouped logistic artifact versus simple threshold rules.
6. Frame-only activation versus hysteresis.
7. Baseline candidate budget versus each expansion factor.
8. End-to-end expansion versus target-region expansion versus both.
9. Legacy recovery routing off versus on as an explicit compatibility check.
10. Planner-specific versus shared calibration.
11. Ground-truth oracle versus sensor-only perception, if both are reported.

Semantic-review and recovery-trigger ablations belong to Phase 5 and should not
be mixed into the initial candidate-coverage claim.

## Key limitations and safeguards

| Issue | Treatment |
|---|---|
| HUGSIM and CARLA use different planners, branches, and dynamics | Fit a CARLA artifact; never import HUGSIM thresholds |
| Low disagreement also occurs in easy scenes | Grouped calibration and temporal confirmation |
| Missing alternative branch | Explicit unavailable state; never zero-fill |
| Planner/checkpoint or sensor change | Invalidate and recalibrate artifact |
| Duplicate merged and top-level policies | One default owner; legacy path disabled or explicit ablation |
| High expansion rate | Hysteresis and a frozen intervention threshold |
| Expanded compute causes latency | Measure estimator, PDMS, and total loop latency |
| Unsafe active winner | Preserve the exact final verifier |
| Simulator nondeterminism | Repeated runs and distributional reporting |
| Scene leakage | Grouped folds and locked route manifests |

HUGSIM future NC is a counterfactual plan check rather than a guaranteed
physical collision, and all 19 scene groups participated in cross-validation.
The current corpus is therefore development evidence, not an untouched final
test set.

## Immediate next step

Port `24f8c68` onto AgenticDriving `613120c`, with the legacy
recovery-routing policy disabled by default. Run the focused and affected
integration tests, then execute a small matched `off`/`observe` CARLA smoke
set. Do not fit an artifact or enable active expansion until those runs prove
selection invariance and produce complete coverage-rescue telemetry.
