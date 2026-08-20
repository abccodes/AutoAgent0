# Uncertainty as an Event Signal for AgenticDriving

## Proposal

The current proposal treats uncertainty as a passive, anticipatory event signal
for the AgenticDriving orchestrator. It identifies planner states that may need
additional scoring, semantic review, or recovery consideration, without
replacing the final verifier or authorizing execution.

The HUGSIM work validates the signal computation, telemetry pipeline, and
offline risk prediction. The next step is an observe-only port to the current
AgenticDriving loop, followed by planner-specific calibration in
CARLA/Fail2Drive. HUGSIM thresholds should not be transferred directly.

## Relationship to the monitor baseline

The [closed-loop monitor baseline](closed-loop-monitor-baselines.md) wraps a
frozen planner with a fixed minimum-risk fallback. This proposal instead uses
the same signal family to decide when the full agentic harness should spend
additional evaluation or recovery effort.

| Property | Monitor baseline | This proposal |
|---|---|---|
| System | Frozen base planner | Full AgenticDriving harness |
| Response | Slowdown or stop | Invoke another tool or increase scrutiny |
| Recovery | Fixed fallback | Existing verified recovery system |
| Execution authority | Monitor fallback rule | Existing final verifier |

## System role

The verifier and uncertainty monitor answer different questions:

- **Verifier:** Is this exact trajectory admissible now?
- **Uncertainty monitor:** Does the proposal distribution and recent history
  suggest that the normal planner/verifier loop needs more scrutiny?

```text
Planner proposals and scores
            |
            +-----------------------+
            |                       |
            v                       v
  normal branch scoring     uncertainty observation
            |                       |
            v                       v
    selected trajectory       temporal event filter
            |                       |
            v                       |
      final verifier ----------------+
            |                       |
            +-----------+-----------+
                        v
                   orchestrator
                        |
        +---------------+----------------+
        |               |                |
  execute normal   request more     enter recovery on
                   evaluation       independent trigger
        |               |                |
        +---------------+----------------+
                        v
             exact proposal verifier
                        |
                        v
                      control
```

In the reviewed AgenticDriving implementation, uncertainty is calculated
inside the recovery selector, after the normal verifier has rejected a plan.
That placement is useful as recovery context but is too late for anticipation.
The raw uncertainty observation should instead be computed in regular Loop 1
from the original proposal distribution. The existing recovery system,
semantic supervisor, memory, and final verifier remain unchanged.

## Uncertainty method

### Inputs and timing

At planning tick `t`, the learned planner emits:

```text
C_t = {(tau_i, q_i)} for i = 1, ..., M
```

`tau_i` is a candidate trajectory in the common ego-local frame and `q_i` is
its native score. Features are computed from the original distribution before
top-k truncation, VLM selection, recovery generation, or uncertainty-based
candidate allocation.

The observation is passive. It is combined with verifier context and a
temporal event filter only after normal candidate generation is complete.

### 1. Intra-planner trajectory disagreement

Convert planner scores to softmax weights:

```text
w_i = exp(q_i / temperature) / sum_j exp(q_j / temperature)
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
failure. It is therefore interpreted with cross-branch, score, temporal, and
verifier features rather than as a standalone safety score.

Required metadata includes candidate count, shared horizon, temperature,
weighting mode, and effective candidate count. Insufficient proposals produce
an explicit unavailable value, never a value of zero.

### 2. Cross-branch disagreement

The HUGSIM implementation compares the highest-scored learned trajectory with
the nearest independent Rule-Planner trajectory:

```text
U_cross(t) = min_r mean_h ||tau_learned[h] - tau_rule_r[h]||
```

AgenticDriving should log two distinct variants:

- `cross_branch_m`: distance between the end-to-end winner and the
  target-region/PDMS branch winner in the normal loop;
- `cross_recovery_m`: distance to verified rule or primitive candidates when
  recovery candidates already exist.

These values are not interchangeable. The normal AgenticDriving branches can
share a learned proposal set, while the HUGSIM Rule-Planner is independent.
Missing recovery candidates are recorded as unavailable, not as agreement.

### 3. Proposal mode count

Flatten trajectories over the common horizon and run deterministic K-means for
candidate values of `k`. Select the smallest `k` whose silhouette score
meaningfully improves on a single mode. The current implementation tests up to
three modes and records every silhouette score.

Mode count remains diagnostic during the passive port. In HUGSIM, unconditional
mode-count fallback rules routed too many frames and reduced precision.

### 4. Score, geometry, and temporal context

Log these complementary features:

- normalized score entropy and effective candidate count;
- normalized top-score margin;
- pairwise and endpoint proposal dispersion;
- change in the selected trajectory from the previous planning tick;
- changes in intra-planner and cross-branch disagreement;
- verifier gates and soft-term margins;
- planner identity, proposal count, route instruction, and recovery state.

Expanded logistic feature sets did not outperform the simple HUGSIM rule, but
the features should be retained for CARLA calibration rather than assumed to
transfer.

### 5. Risk classifier

The strongest HUGSIM rule identified a low-disagreement region:

```text
consensus_risk = (U_intra <= T_intra) AND (U_cross <= T_cross)
```

This does **not** mean the model is uncertain in the usual high-variance sense.
It indicates confident agreement that has historically preceded a future
safety failure, potentially reflecting shared blind spots or mode collapse.

CARLA calibration should compare:

1. Low-disagreement consensus risk.
2. High-disagreement rules.
3. Entropy and score-margin baselines.
4. A grouped logistic model.
5. Temporally persistent variants of each method.

Thresholds and models must be trained only on training scene groups. Planner
families should be calibrated separately unless scale equivalence is shown.

### 6. Temporal event filter

Frame predictions feed a deterministic state machine:

```text
NORMAL -> WATCH -> CONFIRMED -> COOLDOWN -> NORMAL
```

- `NORMAL`: no alert evidence.
- `WATCH`: one alert or rising risk; behavior is unchanged.
- `CONFIRMED`: persistence or accumulated evidence permits a configured tool
  request.
- `COOLDOWN`: suppress repeated calls while telemetry continues.

The filter supports consecutive-frame requirements, sliding simulator-time
windows, separate trigger/release thresholds, healthy-frame release,
freshness limits, post-recovery grace periods, and route/episode budgets. All
timing uses simulator time because planning and control frequencies can differ.

### 7. Orchestrator response and authority

| Verifier | Monitor | Response |
|---|---|---|
| Pass | Normal | Execute admitted plan |
| Pass | One-frame alert | Log and remain in normal loop |
| Pass | Confirmed | Broaden scoring or request asynchronous semantic review |
| Pass | Confirmed + semantic confirmation | Enter existing recovery path |
| Reject | Any state | Enter deterministic recovery and attach uncertainty evidence |
| Reject | No admitted recovery | Controlled braking or hold |

Uncertainty can record observations, emit confirmed events, change a bounded
evaluation budget, and provide structured context to the Critic and Designer.
It cannot bypass the verifier, declare a trajectory safe, generate arbitrary
waypoints, or execute a VLM suggestion. Every final trajectory or primitive is
materialized and admitted by the existing verifier.

### Runtime contract

```text
UncertaintyObservation:
  schema_version, planner_family, policy_mode, timestamp_sec
  affects_candidate_allocation, raw_candidate_count
  intra_disagreement_m, cross_branch_m
  cross_recovery_m or unavailable_reason
  mode_count, score_entropy, effective_candidate_count
  score_margin_normalized, temporal_selected_change_m
  thresholds_or_model_version, raw_feature_availability

PlannerRiskEvent:
  state, confirmed, reason_codes
  first_evidence_timestamp_sec, confirmation_timestamp_sec
  evidence_frames, requested_tool
  cooldown_until_sec, budget_remaining
```

In `observe` mode, `affects_candidate_allocation=false`, and the selected
trajectory must match the monitor-off path on every comparable frame.

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

### Phase 1: observe-only port

- Integrate against current AgenticDriving `main`.
- Move raw uncertainty computation into normal Loop 1.
- Preserve proposal scoring, verifier, recovery, semantic takeover, and memory.
- Add `off`, `observe`, and `active` modes, defaulting to `observe`.
- Prove that observation receives the original proposal set and cannot change
  selection or bypass verifier admission.
- Measure latency at the actual replanning rate.

**Exit:** complete telemetry, invariant selection, bounded overhead, and
explicit missing-feature handling.

### Phase 2: passive CARLA/Fail2Drive calibration

- Collect paired ID and tail routes without changing behavior.
- Group route pairs, scene variants, and repetitions in the same fold.
- Keep separate labels for future verifier rejection, collision, drivable-area
  failure, route departure, deadlock, semantic takeover, and timeout.
- Calibrate planner families separately and reserve unseen routes as a lockbox.

**Exit:** a frozen event policy meets team-approved thresholds on unseen groups.

### Phase 3: uncertainty-gated evaluation

A confirmed event may request asynchronous semantic review, score additional
candidates, run a normally skipped branch, or retain extra diagnostics. It does
not change actuation.

**Exit:** fewer reasoning/tool calls than periodic evaluation, acceptable false
calls, and no degradation in route utility.

### Phase 4: bounded recovery canary

Only after Phase 3 succeeds should a confirmed uncertainty event plus an
independent confirmation enter the existing recovery state. Use repeated runs
and conservative response budgets; stop if unnecessary recovery, timeout, or
route utility degrades beyond a predeclared limit.

**Exit:** improved held-out route utility or safety outcomes, not only improved
frame prediction.

## Benchmark plan

### Experiment arms

| Arm | Behavior | Purpose |
|---|---|---|
| A. Current AgenticDriving | Uncertainty off | Route-level baseline |
| B. Passive observation | Log only | Invariance, latency, calibration |
| C. Gated evaluation | Trigger extra scoring/semantic tool | Test compute routing |
| D. Recovery canary | Confirmed event may request recovery | Test active route benefit |

Use the same checkpoint, controller, perception source, route, seed, and GPU
assignment across arms. Run multiple repetitions and report distributions.
Keep paired ID/tail routes and their repetitions in one statistical group.
Freeze code, configuration, calibration artifact, and route manifest before the
final comparison.

### Metrics

**Route performance:** Fail2Drive utility, route completion, collision and
route-departure rates, deadlock/timeout, PDMS and its component terms, and route
loss caused by unnecessary intervention.

**Monitor performance:** frame precision/recall/coverage/lift, AUROC and average
precision, event recall, episode precision, false alerts per simulator minute,
and warning lead in seconds and planning frames.

**Agentic cost and outcome:** VLM and rescoring calls per kilometer, recoveries
requested/admitted/rejected/completed/aborted, unnecessary recovery rate,
handoff time, and component/total latency.

### Proposed activation gate

These are discussion values, not established safety requirements:

| Requirement | Proposed value |
|---|---:|
| Event recall | >= 0.70 |
| Episode precision | >= 0.70-0.80 |
| False alerts | <= 0.5 / simulator minute |
| Median warning lead | >= 4-5 planning frames |
| Passive selection invariance | Every comparable frame |
| Route utility | No credible observe-mode degradation |

The current HUGSIM results approach the event-recall target but miss the
precision, false-alert, and lead-time targets.

## Required ablations

1. Intra disagreement only.
2. Cross-branch disagreement only.
3. Mode count only.
4. Score entropy and margin baseline.
5. Low-disagreement consensus-risk rule.
6. Temporal confirmation versus frame-only classification.
7. Uncertainty-gated versus periodic semantic checks.
8. Recovery context versus uncertainty-based candidate allocation.
9. Ground-truth oracle versus sensor-only perception.
10. Planner-specific versus shared calibration.

## Key limitations and safeguards

| Issue | Treatment |
|---|---|
| HUGSIM and CARLA use different planners, branches, and dynamics | Recalibrate in AgenticDriving |
| Low disagreement also occurs in easy scenes | Grouped calibration and temporal confirmation |
| Missing alternative branch | Explicit unavailable state; never zero-fill |
| Planner/checkpoint or sensor change | Invalidate and recalibrate artifact |
| High alert rate | Hysteresis, cooldown, and route/episode budgets |
| VLM overinterprets uncertainty | Provide it as advisory evidence only |
| Unsafe active proposal | Materialize and verify the exact trajectory |
| Simulator nondeterminism | Repeated runs and distributional reporting |
| Scene leakage | Grouped folds and locked route manifests |

HUGSIM future NC is a counterfactual plan check rather than a guaranteed
physical collision, and all 19 scene groups participated in cross-validation.
The current corpus is therefore development evidence, not an untouched final
test set.

## Next step

Build the Loop-1 observe-only port and validate selection invariance, telemetry,
latency, and a small passive Fail2Drive run. Keep active recovery disabled until
the passive CARLA data meets an agreed activation gate.
