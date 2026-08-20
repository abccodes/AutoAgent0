# Uncertainty

## Purpose

This document summarizes how the uncertainty work fits into AgenticDriving,
what is already implemented, what has been validated, and what remains.

## Recommendation

Use uncertainty near term as a pre-selection coverage monitor: when the
planner's proposal distribution resembles cases where the normal candidate
budget misses an admissible trajectory, evaluate more existing candidates with
PDMS. This is a bounded response that does not generate a new trajectory or
bypass the final verifier.

Longer term, the same monitor could become an event signal for the high-level
orchestrator, requesting semantic review or recovery consideration. That
broader role should be evaluated only after the coverage policy is calibrated
and tested in CARLA/Fail2Drive.

## Current status

| Work | Repository status | What it does | Validation |
|---|---|---|---|
| Recovery-stage uncertainty | Merged in current AgenticDriving `main` | Computes learned spread, rule-based disagreement, and mode count after recovery begins; changes the redesign candidate mix | Code and calibration tooling are merged; it is not an anticipatory signal |
| Pre-selection CARLA monitor | Implemented locally in commit `24f8c68`; not merged | Runs before candidate scoring with `off`, `observe`, and `active` modes; active mode expands PDMS candidate coverage | Five focused unit tests pass; no CARLA routes or calibrated artifact yet |
| HUGSIM passive monitor | Completed in the HUGSIM/DrivoR stack | Records proposal-distribution signals without changing the selected trajectory | 1,911 frames across 38 routes; grouped offline analysis completed |

## Where it fits

A verifier evaluates the exact trajectory selected for execution. The
uncertainty monitor evaluates whether the normal candidate-scoring budget is
likely to be sufficient.

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

In `observe` mode, expanded scoring runs only as a shadow calculation and the
normal winner is preserved. In `active` mode, a calibrated and temporally
confirmed event increases the existing PDMS candidate budget. Every resulting
trajectory still passes through the normal final verifier.

This top-level monitor should replace or explicitly supersede the old
uncertainty-based routing inside recovery. Running both policies without clear
ownership would make the evaluation difficult to interpret.

## Method

### Proposal-distribution features

At planning tick `t`, the learned planner emits candidate trajectories
`tau_i` and native scores `q_i`:

```text
C_t = {(tau_i, q_i)} for i = 1, ..., M
```

The CARLA monitor receives this original proposal set before candidate scoring.
For bounded runtime, it can compute distribution features over the highest
scored `N` candidates, currently 32, while recording that truncation. This
feature cap does not alter the planner's candidates or selected trajectory.

Scores are standardized to reduce planner-scale sensitivity and converted to
weights:

```text
z_i = (q_i - mean(q)) / std(q)
w_i = exp(z_i) / sum_j exp(z_j)
```

The weighted spatial spread is:

```text
mu_h          = sum_i w_i * tau_i[h]
rms_h         = sqrt(sum_i w_i * ||tau_i[h] - mu_h||^2)
dispersion(t) = mean_h rms_h
```

The monitor also records normalized dispersion, pairwise and endpoint
distance, normalized score entropy, effective candidate count, top-score
margin, proposal mode count, change from the previous selected trajectory, and
distance from the best learned trajectory to the nearest recovery primitive.
Unavailable features are represented explicitly rather than as zero.

### Calibration target

The HUGSIM analysis found that future risk was enriched when learned and
rule-based trajectories both had low disagreement:

```text
consensus_risk = (U_intra <= T_intra) AND (U_cross <= T_cross)
```

This is better interpreted as confident consensus risk or a shared blind spot,
not conventional high-variance uncertainty. It is useful evidence, but HUGSIM
thresholds do not transfer directly to AgenticDriving.

The implemented CARLA path instead trains a planner- and perception-specific
logistic artifact against a measurable coverage target:

```text
coverage_rescue =
    normal PDMS winner fails admission
    AND shadow expanded-budget winner passes
```

Calibration uses grouped out-of-fold predictions so paired ID/tail routes and
route repetitions do not leak between training and evaluation. Active mode
requires an artifact whose schema, planner, and perception source match the
runtime configuration.

### Temporal policy and authority

A single high-risk frame does not expand coverage. The implementation requires
a configured number of consecutive high-risk ticks to activate expansion and
a configured number of low-risk ticks to release it.

| Mode | Behavior |
|---|---|
| `off` | Original scoring path; no monitor |
| `observe` | Log features and shadow expanded scoring; execute the original winner |
| `active` | Apply calibrated hysteresis and expand existing PDMS coverage |

The monitor can change only the number of existing candidates evaluated. It
cannot generate arbitrary waypoints, declare a trajectory safe, invoke recovery
directly, or bypass the verifier.

## HUGSIM evidence

### Passive corpus

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

All frames retained the 64-proposal DrivoR distribution, preserved the DrivoR
argmax, left candidate allocation unchanged, and recorded post-step outcomes.

### Passive control comparison

| Metric | Off mean | Passive mean | Passive - off |
|---|---:|---:|---:|
| NC | 0.9096 | 0.9198 | +0.0102 |
| TTC | 0.8388 | 0.8399 | +0.0011 |
| PDMS | 0.8571 | 0.8595 | +0.0024 |
| RC | 0.5001 | 0.4963 | -0.0037 |
| HDScore | 0.4687 | 0.4681 | -0.0006 |
| Wall time per frame | 2.286 s | 2.550 s | +11.5% |

Repeated routes diverged by planning frame 2 despite deterministic controls.
The small score differences do not establish a passive performance effect. The
tested HUGSIM implementation added 11.5% wall-time overhead.

### Grouped prediction

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

The HUGSIM signal enriches for future risk, but its precision, false-alert rate,
and warning lead are not sufficient for direct intervention. These results
justify continued passive work, not an AgenticDriving route-benefit claim.

## Next steps

First, port `24f8c68` onto current AgenticDriving `main`, reconcile it with
the merged recovery-stage implementation, and rerun unit and integration tests.
Then run small `off` and `observe` CARLA smoke tests to verify selection
invariance, telemetry completeness, and runtime overhead.

Next, collect a larger passive Fail2Drive corpus, keeping paired routes and
repetitions in the same statistical group. Calibrate the coverage-rescue
artifact separately for each planner/perception stack and reserve unseen routes
for evaluation.

Finally, compare `off` against `active` candidate-coverage expansion using
repeated routes. Report route utility, PDMS, completion, collisions, departures,
timeouts, coverage rescues, intervention frequency, and latency. Only after
that policy demonstrates route-level value should uncertainty be evaluated as
a trigger for semantic review or recovery.
