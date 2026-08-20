# Uncertainty

## Purpose

How the uncertainty work can contribute to the current AgenticDriving system.
It is intended to confirm that the role and research direction make sense
before we complete the AgenticDriving integration.

## Recommendation

Treat uncertainty as an anticipatory event signal for the high-level
orchestrator. It should identify planner states that may need more evaluation,
semantic reasoning, or recovery consideration. It should not replace the final
trajectory verifier or directly authorize vehicle control.

The HUGSIM work shows that we can compute this signal passively, preserve the
planner's original decision, and enrich for future plan risk. It does not yet
show that the signal improves closed-loop driving or that its thresholds
transfer to CARLA/Fail2Drive. I propose an observe-only AgenticDriving
integration before any active intervention.

## Where it fits in agentic system

A trajectory verifier evaluates one selected trajectory. The uncertainty
mechanism instead evaluates the planner's full proposal distribution and its
agreement with alternative planning branches. These are complementary signals:

- the verifier asks whether the current trajectory is admissible;
- uncertainty asks whether the system should gather more evidence before
  trusting the normal planning path.

```text
Planner proposals and scores
            |
       +----+----------------+
       |                     |
       v                     v
 normal selection     uncertainty signal
       |                     |
       v                     v
 final verifier       temporal confirmation
       |                     |
       +----------+----------+
                  v
             orchestrator
                  |
       execute / evaluate more / recover
                  |
                  v
       exact final trajectory verifier
```

## Method

### Planner-distribution signal

At planning tick `t`, the learned planner produces candidate trajectories
`tau_i` and native scores `q_i`:

```text
C_t = {(tau_i, q_i)} for i = 1, ..., M
```

All uncertainty features are computed from this original proposal distribution
before top-k truncation, recovery generation, VLM selection, or
uncertainty-based candidate allocation.

First, planner scores are converted into softmax weights:

```text
w_i = exp(q_i / temperature) / sum_j exp(q_j / temperature)
```

We then measure how much the weighted proposals spread around their mean at
each future horizon step:

```text
mu_h       = sum_i w_i * tau_i[h]
rms_h      = sqrt(sum_i w_i * ||tau_i[h] - mu_h||^2)
U_intra(t) = mean_h rms_h
```

`U_intra` describes concentration in trajectory space. It is not a safety
score by itself: high spread can represent legitimate multimodality, while low
spread can represent either an easy scene or confident failure.

### Cross-planner and contextual evidence

HUGSIM also compares the best learned trajectory with the nearest independent
Rule-Planner trajectory:

```text
U_cross(t) = min_r mean_h ||tau_learned[h] - tau_rule_r[h]||
```

In AgenticDriving, we propose logging disagreement between the end-to-end and
target-region/PDMS branch winners during normal driving. When recovery
candidates exist, disagreement with verified primitive or rule candidates can
be recorded separately. These measures require new AgenticDriving calibration
because they do not have the same distribution as the HUGSIM Rule-Planner
signal.

We also retain proposal mode count, score entropy, top-score margin, proposal
dispersion, selected-trajectory change, and verifier margins. These provide
context for distinguishing useful multimodality, ordinary agreement, and
potential planner collapse.

### Consensus risk and temporal confirmation

The strongest HUGSIM result came from a low-disagreement region:

```text
consensus_risk = (U_intra <= T_intra) AND (U_cross <= T_cross)
```

This result is better interpreted as **consensus risk** than conventional
high-variance uncertainty: the learned and alternative planners agree, but
similar agreement has historically preceded a future safety failure. It may
capture shared blind spots or confident mode collapse.

A frame-level signal should not immediately change behavior. Alerts pass
through a deterministic temporal filter:

```text
NORMAL -> WATCH -> CONFIRMED -> COOLDOWN -> NORMAL
```

Only persistent or accumulated evidence becomes a confirmed event. Cooldowns,
hysteresis, freshness checks, and route-level budgets prevent repeated tool
calls. Timing uses simulator time rather than wall time.

### Agentic response

| Verifier | Uncertainty event | Proposed response |
|---|---|---|
| Pass | None | Execute the admitted trajectory |
| Pass | Unconfirmed | Log only |
| Pass | Confirmed | Request more scoring or asynchronous semantic review |
| Pass | Confirmed plus independent confirmation | Consider existing recovery path |
| Reject | Any state | Use existing deterministic recovery |
| No admitted recovery | Any state | Controlled braking or hold |

Uncertainty may request additional evaluation and provide structured evidence
to the Critic, Designer, and recovery memory. It cannot bypass verification,
declare a trajectory safe, generate arbitrary waypoints, or execute a VLM
suggestion.

## Evidence to date

### Passive HUGSIM corpus

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

All 1,911 frames retained the full 64-proposal DrivoR distribution, preserved
the original DrivoR argmax, left candidate allocation unchanged, and recorded
post-step outcomes.

### Passive control comparison

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

Repeated routes diverged by planning frame 2 despite deterministic controls.
The small score differences therefore do not establish a passive performance
effect. The tested implementation added 11.5% wall-time overhead.

### Risk prediction

The primary grouped cohort contained 1,614 currently safe frames with a 14.5%
future plan-risk rate. Variants of each base scene remained in the same
cross-validation fold.

| Predictor | Precision | Recall | Coverage | Lift | OOF AUROC |
|---|---:|---:|---:|---:|---:|
| Low-disagreement quadrant | 0.629 | 0.479 | 0.110 | 4.34x | n/a |
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

The signal identifies a higher-risk subset, but the present precision, alert
rate, and warning lead are not sufficient for direct intervention. These
results support passive integration and further calibration, not an active
route-performance claim.

## Next steps

The first AgenticDriving version should compute uncertainty in the normal
planning loop and run in observe-only mode. It should log the original proposal
distribution, branch disagreement, temporal state, and verifier context while
proving that selection and control remain unchanged.

We would then collect passive CARLA/Fail2Drive data, calibrate each planner
family on grouped ID/tail routes, and evaluate on unseen route groups. If the
signal achieves acceptable event recall, precision, warning lead, and runtime
cost, its first active use should be additional scoring or asynchronous
semantic review. Direct recovery should be considered only later, with
independent confirmation and the existing final verifier.

The benchmark should compare the current system, passive observation,
uncertainty-gated evaluation, and a later bounded recovery canary. Route
utility, PDMS, completion, collisions, departures, deadlock, false alerts,
warning lead, tool calls, and latency should be reported over repeated runs.
