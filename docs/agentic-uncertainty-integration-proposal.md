# Uncertainty as an Event Signal for AgenticDriving

## Proposal status

This document proposes how the uncertainty work developed and evaluated in the
HUGSIM AutoAgent0 stack should be used in the current AgenticDriving system. It
is intended for technical review before implementation on AgenticDriving
`main`.

The current idea is to treat uncertainty as a passive, anticipatory event signal that
advises the orchestrator when to spend additional reasoning and verification effort.

The completed HUGSIM work validates the signal computation, telemetry contract,
and offline risk enrichment. It does not yet validate HUGSIM thresholds in
CARLA/Fail2Drive or justify direct uncertainty-triggered vehicle intervention.

## Relationship to the traditional monitor baseline

The separate [closed-loop monitor baseline proposal](closed-loop-monitor-baselines.md)
places a conventional score-based monitor directly around a frozen
SparseDriveV2 policy and gives it a minimum-risk fallback. That experiment asks
whether a simple runtime monitor can improve the base policy without the
agentic harness. This proposal asks how can the same family of proposal
signals improve the full AgenticDriving orchestrator?

| Property | Traditional uncertainty baseline | This proposal |
|---|---|---|
| System under test | Frozen base planner | Full AgenticDriving harness |
| Trigger input | Planner score proxy | Proposal, branch, temporal, and verifier context |
| Immediate response | Slowdown or stop | Invoke another tool or increase scrutiny |
| Recovery selection | Fixed minimum-risk fallback | Existing verified recovery system |
| VLM, memory, self-refinement | Excluded | Available after a confirmed event |
| Execution authority | Monitor fallback rule | Existing final verifier |

Both should be evaluated. They should not share an implementation path in the
primary baseline comparison because doing so would mix a conventional monitor
with the method being evaluated.

## Terminology

The implemented quantities are trajectory-distribution proxies, not a formal
Bayesian estimate of epistemic uncertainty. The observed danger region also
contains low disagreement, which is better interpreted as possible confident
model collapse or shared blind spots.

Recommended terms are:

- `uncertainty_observation`: raw per-frame signal decomposition;
- `planner_risk_event`: temporally confirmed monitor event;
- `consensus_risk`: risk associated with low within-planner and low
  cross-branch disagreement;
- `policy_mode`: `off`, `observe`, or `active`;
- `execution_authority`: retained exclusively by the final verifier.

## Detailed signal mechanism

Let the learned planner emit trajectories and native scores

```text
C_t = {(tau_i, q_i)} for i = 1, ..., M
```

where each `tau_i` is represented in a common ego-local trajectory frame. All
signals must be calculated from the original proposal distribution before
top-k truncation, VLM selection, recovery generation, or candidate-budget
allocation.

### 1. Intra-planner trajectory disagreement

Convert native scores into softmax weights:

```text
w_i = exp(q_i / temperature) / sum_j exp(q_j / temperature)
```

At each horizon step, compute the weighted mean waypoint and weighted RMS
deviation. The frame signal is the mean RMS deviation over the shared horizon:

```text
mu_h       = sum_i w_i * tau_i[h]
rms_h      = sqrt(sum_i w_i * ||tau_i[h] - mu_h||^2)
U_intra(t) = mean_h rms_h
```

This measures how concentrated the scored proposal distribution is in
trajectory space. It is not sufficient by itself bc high spread can represent
healthy multimodality, while low spread can represent either an easy scene or
confident failure.

Required metadata includes member count, horizon, weighting mode, score
temperature, and effective member count. Frames with too few proposals must be
marked unavailable rather than silently treated as certain.

### 2. Cross-branch disagreement

The HUGSIM implementation measured the mean waypoint distance from the best
learned trajectory to its nearest independent Rule-Planner trajectory:

```text
U_cross(t) = min_r mean_h ||tau_learned[h] - tau_rule_r[h]||
```

The current AgenticDriving regular loop has a cheaper and more system-aligned
signal available every tick: disagreement between the end-to-end branch winner
and the target-region/PDMS branch winner. This should be recorded separately as
`cross_branch_m` because both trajectories may originate from the learned
proposal set.

When recovery candidates are already being generated, a second
`cross_recovery_m` signal may compare the learned winner with verified
primitive or rule-based candidates. It must not be conflated with
`cross_branch_m`, and its absence in the normal loop is not zero disagreement.

This distinction matters because HUGSIM thresholds learned from an independent
Rule-Planner reference are not valid for AgenticDriving branch disagreement.

### 3. Proposal mode count

Flatten each trajectory across the common horizon, run deterministic K-means
for candidate `k` values, and select the smallest `k` whose silhouette score
meaningfully improves on a single mode. Current work evaluates up to three
modes and records all silhouette scores.

Mode count should remain a diagnostic feature during the AgenticDriving passive
phase. In HUGSIM, unconditional mode-count fallback rules routed too many frames
and reduced precision.

### 4. Score and temporal features

Retain inexpensive complementary signals:

- normalized score entropy;
- effective candidate count;
- normalized top-score margin;
- raw proposal pairwise dispersion and endpoint dispersion;
- selected-trajectory change from the previous planning frame;
- changes in `U_intra` and branch disagreement;
- current verifier gates and soft-term margins when available;
- planner identity, proposal count, route instruction, and recovery state.

The current offline experiment found that the expanded logistic feature sets
did not outperform the simple low-disagreement quadrant. These features should
still be logged in CARLA, but they should not be assumed to improve the model
without grouped evidence.

### 5. Empirical danger framing

Historical data originally motivated a low-intra and low-cross danger quadrant:

```text
consensus_risk = (U_intra <= T_intra) AND (U_cross <= T_cross)
```

The interpretation is not "the model is uncertain." It is "the learned and
alternative planners agree, but that consensus has historically preceded a
future safety failure." This can expose shared blind spots that a disagreement
threshold misses.

The exact classifier is not frozen. AgenticDriving calibration should compare:

1. low-disagreement quadrant;
2. high-disagreement rules;
3. score-based entropy/margin baseline;
4. grouped logistic model;
5. temporal persistence versions of the above.

Threshold selection must occur on training scene groups only.

## Proposed AgenticDriving placement

### Current placement problem

In the reviewed implementation, the regular loop selects and verifies a plan.
Only after verifier rejection does the orchestrator generate recovery
proposals and dispatch the recovery selector. The uncertainty calculation lives
inside that recovery selector.

### Proposed placement

Compute the raw observation in regular Loop 1, after the planner and candidate
scoring branches are available but before verifier admission is resolved. Then
combine the observation with the verifier result and temporal monitor state.

```text
Planner proposals and native scores
            |
            +----------------------+
            |                      |
            v                      v
  normal branch scoring     uncertainty observation
            |                      |
            v                      v
    selected trajectory      temporal event filter
            |                      |
            v                      |
     final verifier ---------------+
            |                      |
            +----------+-----------+
                       v
                  orchestrator
                       |
        +--------------+-------------------+
        |              |                   |
   execute normal   request more       enter recovery
                    evaluation         on hard trigger
        |              |                   |
        +--------------+-------------------+
                       v
             exact final proposal verifier
                       |
                       v
                    control
```

The uncertainty monitor does not sit inside the verifier. The verifier answers
whether one exact trajectory is currently admissible. The uncertainty monitor
answers whether the planner distribution and recent history justify spending
additional compute or gathering additional evidence.

## Authority and response policy

### Decision table

| Verifier | Monitor state | Orchestrator response |
|---|---|---|
| Pass | Normal | Execute the admitted plan |
| Pass | One-frame alert | Log, remain in normal loop |
| Pass | Confirmed alert | Broaden scoring or request asynchronous semantic evaluation |
| Pass | Confirmed alert plus semantic confirmation | Enter existing recovery proposal path |
| Reject | Any monitor state | Enter deterministic recovery; attach uncertainty evidence |
| Reject | Confirmed alert | Expand diagnosis/search budget within configured limits |
| Reject | No admitted recovery | Controlled braking or hold |

### Safety boundary

Uncertainty may have:

- observation authority: record and summarize planner-distribution evidence;
- event authority: request a configured tool invocation after confirmation;
- proposal influence: alter which candidates are considered during recovery;
- prompt influence: provide structured evidence to the Critic and Designer.

Uncertainty must not have:

- execution authority;
- permission to bypass the final verifier;
- permission to generate arbitrary waypoints;
- permission to label a trajectory safe;
- permission to directly execute a VLM suggestion.

Every revised trajectory or primitive remains materialized in the common
trajectory representation and passes the same final verifier.

## Temporal event filter

The passive HUGSIM policy produced too many isolated alerts for direct active
use. Add a small deterministic state machine around the frame classifier:

```text
NORMAL -> WATCH -> CONFIRMED -> COOLDOWN -> NORMAL
```

- `NORMAL`: no alert evidence.
- `WATCH`: one alert or a rising risk score; behavior unchanged.
- `CONFIRMED`: persistence or accumulated evidence crosses a configured gate;
  the orchestrator may invoke an additional tool.
- `COOLDOWN`: suppress repeated tool calls while continuing telemetry.

Candidate controls include:

- required consecutive alert frames;
- alert accumulation in a sliding simulator-time window;
- separate trigger and release thresholds;
- healthy-frame release requirement;
- result freshness limit;
- per-route and per-episode tool-call budgets;
- post-recovery grace period;
- reset rules for planner changes and route boundaries.

All timing should use simulator time, not wall time or callback count. The
planner and control loops may run at different frequencies.

## Interaction with agentic components

### Orchestrator

The orchestrator owns the monitor state and response policy. A confirmed event
is one event among verifier rejection, semantic takeover, stuck detection,
recovery completion, and execution-guard failure. It does not automatically
mean recovery.

### Critic

The Critic receives a compact structured summary:

- current and recent uncertainty values;
- why the event was confirmed;
- proposal/branch identities;
- verifier gate and soft-term decomposition;
- route instruction and target;
- recent failed or ineffective actions.

The Critic diagnoses whether the alert appears safety-relevant, semantic,
route-related, or spurious. The Critic does not recompute uncertainty and its
self-reported confidence is not a substitute for the monitor.

### Designer and planner

The Designer receives the diagnosed problem and bounded candidate-generation
options. Uncertainty may change the candidate budget or ensure that alternative
branches are represented, but should not eliminate all learned candidates based
on one noisy frame.

### Semantic supervisor

A strong initial active use is to trigger the existing asynchronous semantic
supervisor. This consumes additional compute without immediately changing
actuation. A fresh, high-confidence, confirmed semantic result can then use the
existing soft-takeover path.

### Verifier

The verifier remains deterministic and proposal-specific. It should log enough
term decomposition to distinguish monitor warnings that preceded a later
verifier rejection from monitor warnings that never became safety-relevant.

### Recovery memory

Record the monitor event, tool invoked, verifier outcome, selected primitive,
execution consequence, and later route progress. This enables analysis of which
uncertainty patterns were actionable and prevents repeated ineffective
responses during one recovery episode.

## Proposed runtime interface

A planner-independent observation should resemble:

```text
UncertaintyObservation:
  schema_version
  planner_family
  policy_mode
  timestamp_sec
  affects_candidate_allocation
  raw_candidate_count
  intra_disagreement_m
  cross_branch_m
  cross_recovery_m or unavailable_reason
  mode_count
  score_entropy
  effective_candidate_count
  score_margin_normalized
  temporal_selected_change_m
  thresholds_or_model_version
  raw_feature_availability
```

The temporal filter emits:

```text
PlannerRiskEvent:
  state
  confirmed
  reason_codes
  first_evidence_timestamp_sec
  confirmation_timestamp_sec
  evidence_frames
  requested_tool
  cooldown_until_sec
  budget_remaining
```

The `observe` contract requires `affects_candidate_allocation=false` and exact
equality between the baseline selection and the selection made with telemetry
enabled.

## Completed HUGSIM evidence

### Repeated controls

Two uncertainty-off and two passive-observe repeats ran the same five routes,
planner checkpoint, controller, seed, deterministic Torch settings, and GPU
assignment.

| Metric | Off mean | Passive mean | Passive - off |
|---|---:|---:|---:|
| NC | 0.9096 | 0.9198 | +0.0102 |
| TTC | 0.8388 | 0.8399 | +0.0011 |
| PDMS | 0.8571 | 0.8595 | +0.0024 |
| RC | 0.5001 | 0.4963 | -0.0037 |
| HDScore | 0.4687 | 0.4681 | -0.0006 |
| Wall time per frame | 2.286 s | 2.550 s | +11.5% |

Terminal outcome categories agreed within each condition on all five routes.
Nevertheless, every repeated route pair diverged at planning frame 2. Mean
selected-plan distance across a repeated route ranged from 0.119 to 1.845 m for
off and 0.080 to 2.783 m for passive observation.

The small cross-condition score changes are comparable to route and
within-condition variation. They do not establish a passive uncertainty effect
on driving performance. The overhead estimate is meaningful for the tested
DrivoR/HUGSIM stack but must be remeasured in AgenticDriving.

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

All 1,911 frames satisfied the passive telemetry contract:

- `policy_mode=observe`;
- `affects_candidate_allocation=false`;
- 64 learned proposals and scores recorded;
- 64 members used for intra-disagreement and mode count;
- baseline proposal index matched the learned-score argmax;
- executed selection remained the DrivoR argmax;
- post-step execution outcome was available.

### Grouped prediction results

The primary predictive cohort excluded frames whose current plan already
failed NC/DAC or whose executed action recorded collision. It contained 1,614
currently-safe frames with a 14.5% future plan-risk rate. Both difficulty
variants of a base scene stayed in one of five scene-grouped folds.

| Predictor | Precision | Recall | Coverage | Lift | OOF AUROC |
|---|---:|---:|---:|---:|---:|
| Trained low-disagreement quadrant | 0.629 | 0.479 | 0.110 | 4.34x | n/a |
| Legacy logistic features | 0.383 | 0.329 | 0.125 | 2.64x | 0.674 |
| Recoverable feature set | 0.365 | 0.316 | 0.126 | 2.51x | 0.657 |
| Raw-proposal feature set | 0.271 | 0.372 | 0.199 | 1.87x | 0.647 |

At the event level, the trained quadrant detected 17 of 25 future-risk events,
for 0.680 event recall. Episode precision was 0.447, false alerts were 2.64 per
minute, and median first-warning lead was two frames.

The old fixed thresholds did not fire the intended low-disagreement quadrant
on this expanded corpus. An ungrouped calibration grid achieved its recall goal
only by routing every frame, which is unusable. Neither result should be copied
into AgenticDriving configuration.

## Interpretation limits

- HUGSIM future NC is a counterfactual multi-step plan check, not necessarily a
  physical collision at the current frame.
- HUGSIM static-background collision geometry can be more conservative than the
  simulator's physical contact geometry.
- The HUGSIM experiment used DrivoR proposals and an independent Rule-Planner
  reference. AgenticDriving may use SparseDriveV2 and branch winners from one
  shared proposal distribution.
- HUGSIM thresholds are scale-dependent because trajectory horizon, planner
  score distribution, replan frequency, and coordinate conventions affect the
  features.
- All 19 HUGSIM scene groups participated in cross-validation. They are not an
  untouched final lockbox.
- The current results validate prediction and telemetry, not route-level benefit
  from an active policy.

## AgenticDriving implementation plan

### Phase 0: integrate against current main

- Start from the latest AgenticDriving `main`, not the older uncertainty
  development checkout.
- Preserve the current candidate-scoring, PDMS verifier, recovery state,
  semantic takeover, and memory interfaces.
- Add tests that prove uncertainty receives the original planner proposals and
  cannot bypass verifier admission.

Exit condition: the port is a small, reviewable change on current architecture,
not a replacement of the recovery system.

### Phase 1: top-level passive observation

- Move raw uncertainty calculation out of the recovery selector and into normal
  Loop 1.
- Add `off`, `observe`, and `active` policy modes; default to `observe` during
  development.
- Record raw proposal, branch, temporal, and verifier-context features.
- Keep selection and control byte-for-byte or numerically equivalent to the
  monitor-off path.
- Measure latency at the actual AgenticDriving replan frequency.

Exit condition: complete telemetry, no selection changes, bounded overhead, and
no missing-feature values silently interpreted as zero.

### Phase 2: CARLA/Fail2Drive passive calibration

- Collect paired in-distribution and tail routes without changing behavior.
- Calibrate each planner family separately unless feature-scale equivalence is
  demonstrated.
- Keep route pairs, scene variants, and repetitions in the same grouped fold.
- Retain separate labels for future verifier rejection, collision, drivable-area
  failure, route departure, stuck/deadlock, semantic takeover, and timeout.
- Select thresholds or models using training folds only.
- Reserve additional routes or scene groups as a lockbox.

Exit condition: a frozen event policy meets the team-approved passive gate on
unseen groups.

### Phase 3: uncertainty-gated additional evaluation

The first active response should not change actuation. A confirmed event may:

- request an asynchronous semantic check;
- score more candidates or run the normally skipped branch;
- retain extra verifier diagnostics;
- attach context if an independent verifier failure later starts recovery.

Exit condition: lower reasoning cost than periodic semantic evaluation, with no
degradation in route utility and an acceptable false tool-call rate.

### Phase 4: bounded recovery canary

Only after Phase 3 succeeds, allow a confirmed uncertainty event plus an
independent confirmation signal to enter the existing recovery proposal state.
Start with the most conservative bounded responses and keep final verifier
authority.

Use repeated runs because deterministic equality is not available. Stop the
canary if unnecessary recovery, route timeout, or degraded route utility exceeds
the predeclared limit.

Exit condition: active improvement on held-out route-level utility and safety
outcomes, not merely higher frame-level classifier metrics.

## Benchmark design

### Experiment arms

| Arm | Uncertainty behavior | Purpose |
|---|---|---|
| A. Current AgenticDriving | Off | Route-level baseline |
| B. Passive observation | Log only | Invariance, latency, calibration |
| C. Gated evaluation | Trigger extra scoring/semantic tool | Test agentic compute routing |
| D. Bounded recovery canary | Confirmed events may request recovery | Test active route benefit |

The traditional standalone uncertainty, RSS, and MPC monitor arms remain as
defined in the baseline proposal and should not reuse C or D.

### Repetition and grouping

- Use the same checkpoint, controller, perception source, scenario, seed, and
  GPU assignment across arms.
- Run multiple repetitions per route and report distributions because the
  closed-loop system is not assumed deterministic.
- Keep paired ID/tail routes and all repetitions in one statistical group.
- Freeze code, configuration, threshold/model artifact, and dataset manifest
  before the final run.

### Route-level outcomes

- Fail2Drive driving score or canonical route utility;
- route completion and success rate;
- collision rate and collision type;
- route departure;
- stuck/deadlock and timeout;
- PDMS and its NC, DAC, TTC, progress, comfort, and lane terms;
- intervention-induced route loss or excessive stopping.

### Monitor and agentic outcomes

- frame precision, recall, coverage, lift, AUROC, and average precision;
- event recall and episode precision;
- false alerts per simulator minute;
- first-warning lead in seconds and planning frames;
- VLM calls and candidate rescoring calls per kilometer;
- confirmed events that later caused verifier rejection;
- recoveries requested, admitted, rejected, completed, and aborted;
- unnecessary recovery rate;
- time from event to safe handoff;
- monitor, critic, designer, verifier, and total loop latency.

### Proposed review gate

These values are proposed starting points for team discussion, not established
safety requirements:

- event recall at least 0.70;
- episode precision at least 0.70 to 0.80;
- false alert episodes no more than 0.5 per simulator minute;
- median warning lead at least four to five planning frames;
- passive selection invariance on every comparable frame;
- bounded monitor overhead at the target replan rate;
- no statistically credible degradation in route utility in observe mode.

The current HUGSIM result meets the event-recall neighborhood but misses the
precision, false-alert, and warning-lead proposals.

## Ablations needed for a defensible paper claim

1. Intra disagreement only.
2. Cross-branch disagreement only.
3. Mode count only.
4. Score entropy and margin baseline.
5. Low-disagreement consensus-risk quadrant.
6. Temporal confirmation versus frame-only classification.
7. Uncertainty-gated semantic checks versus periodic semantic checks.
8. Uncertainty as recovery context versus uncertainty changing candidate
   allocation.
9. Ground-truth perception oracle versus sensor-only perception.
10. Planner-specific versus shared calibration.

The primary claim should be about improved event-driven orchestration only if
Arm C or D improves the relevant compute/route tradeoff on held-out routes.

## Failure modes and safeguards

| Risk | Required safeguard |
|---|---|
| Low disagreement on ordinary easy scenes | Grouped calibration and temporal confirmation |
| High false-alert rate | Hysteresis, cooldown, budgets, passive activation gate |
| Missing alternative branch | Explicit unavailable state, never zero-fill |
| Planner or checkpoint change | Invalidate calibration artifact |
| Sensor shift changes feature scale | Sensor-only recalibration and monitoring |
| VLM treats uncertainty as proof | Prompt states it is advisory evidence |
| Repeated recovery oscillation | Existing episode memory, cooldown, and action history |
| Unsafe active suggestion | Materialize and verify exact trajectory |
| Simulator nondeterminism | Repeated runs and distributional reporting |
| Metric leakage across scene variants | Grouped folds and locked manifests |

## Proposed paper framing

A supportable framing is:

> The uncertainty layer is a planner-distribution runtime monitor that produces
> sparse, auditable events for the high-level orchestrator. It estimates when
> the normal planner/verifier loop may require additional semantic or recovery
> tools. It does not authorize actuation; all generated actions remain subject
> to the common verifier.

Avoid claiming that the current proxy is calibrated epistemic uncertainty or
that HUGSIM prediction results demonstrate AgenticDriving route improvement.
Those claims require planner-specific CARLA calibration and held-out active
experiments.
