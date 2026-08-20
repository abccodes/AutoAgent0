# Closed-Loop Monitor Baselines for SparseDriveV2

## Summary

Propose evaluating three traditional closed-loop runtime monitors around a
frozen SparseDriveV2 driving policy:

1. a **score-based uncertainty monitor** with a minimum-risk fallback;
2. an **RSS safety-envelope monitor** with a braking fallback; and
3. an **MPC predictive safety filter** that minimally modifies an unsafe
   SparseDriveV2 plan.

All methods would begin with the same SparseDriveV2 checkpoint and native plan.
The monitor would sit between the selected trajectory and the existing
trajectory-following controller. This keeps the study separate from AutoAgent0
and lets us ask whether conventional uncertainty-, rule-, or model-based
monitoring can address the same long-tail failures as the full agentic system.

Our working interpretation of a *closed-loop monitor* is an **active runtime
wrapper**: it observes the policy and environment repeatedly and may replace an
unsafe action.

The proposed comparison is:

| Method | Shared base policy | Runtime intervention |
|---|---|---|
| SparseDriveV2 | SparseDriveV2 | None |
| + Uncertainty monitor | SparseDriveV2 | Minimum-risk slowdown/stop |
| + RSS monitor | SparseDriveV2 | Bounded braking trajectory |
| + MPC safety filter | SparseDriveV2 | Closest feasible trajectory or emergency stop |
| + AutoAgent0 | SparseDriveV2 | Verified agentic recovery |

## Feedback. Some questions before implementation. See later sections for implementation details, these are questions around main design based on details:

### 1. Monitor intervention semantics

**Question:** Should each monitor run after every SparseDriveV2 planning update
and be allowed to replace the native trajectory before it reaches the controller?
Under this active interpretation, uncertainty and RSS would switch to a fallback,
MPC would return a corrected feasible trajectory, and each monitor would return
control to SparseDriveV2 after its trigger clears. The alternative is an
alarm-only monitor that logs risk but still executes the native trajectory. We
recommend active intervention because alarm-only monitoring cannot change DS,
SR, or collision rate.

### 2. Common fallback

**Question:** Should fallback behavior be shared to isolate monitor quality, or
is method-specific fallback part of the intended comparison?

### 3. Evaluation scope and table format

**Question:** Is the full three-repetition Fail2Drive evaluation required?

### 4. AutoAgent0 comparison row

**Question:** Which AutoAgent0 configuration should appear in the primary table?
The current C4 ablation configuration uses ground-truth perception and would not
be a controlled sensor-only comparison without a rerun.

## Where the monitors fit

### AutoAgent0 wrapper

```text
SparseDriveV2 candidates
        |
        v
regular proposal construction and scoring
        |
        v
final verifier ---------------------- rejected/stuck/semantic trigger
        |                                           |
     accepted                                       v
        |                              diagnosis and recovery proposal
        |                                           |
        |                              verification/refinement/memory
        |                                           |
        +--------------------------> executed trajectory or maneuver
```

The conventional baselines would **not** reuse AutoAgent0's VLM, PDMS
arbitration, recovery library, self-refinement, or memory. Each monitor should
receive the native SparseDriveV2 plan directly and independently decide whether
to pass it through or apply its own limited intervention:

```text
SparseDriveV2 native plan -> baseline monitor -> existing controller
```

This isolates the benefit of the monitoring principle rather than combining a
traditional monitor with AutoAgent0.

## Proposed baseline 1: score-based uncertainty monitor

### Method

Use SparseDriveV2's trajectory candidate scores to estimate whether the planner
has a clear, stable preference. Candidate signals include:

- normalized entropy of the candidate-score distribution;
- the score margin between the top two candidates; and
- geometric dispersion among high-scoring candidate trajectories.

The initial implementation would combine the entropy and score margin into a
calibrated risk signal; trajectory dispersion can be retained as an ablation or
diagnostic. If risk remains above a trigger threshold for several planning
updates, the monitor replaces the native plan with a bounded minimum-risk
slowdown or stop. A lower release threshold and a short healthy-frame
requirement prevent rapid switching between driving and braking.

The thresholds should be calibrated on held-out Bench2Drive data. A reasonable 
calibration objective is to keep the intervention rate low on nominal routes
while detecting as many impending closed-loop failures as
possible.

### Interpretation and limitation

This is a lightweight uncertainty baseline because it uses outputs already
available from SparseDriveV2 and requires neither retraining nor repeated neural
network inference. SparseDriveV2 scores are not guaranteed to be calibrated
probabilities, however. The paper should therefore call the signal a
**score-based predictive uncertainty proxy**, not a formally calibrated measure
of epistemic uncertainty.

Monte Carlo dropout or an ensemble would more closely measure model uncertainty,
but would require more compute. The proposed baseline therefore uses the score-based proxy and
does not add repeated model inference.

### Original literature

- Michelmore, Kwiatkowska, and Gal, [*Evaluating Uncertainty Quantification in
  End-to-End Autonomous Driving Control*](https://arxiv.org/abs/1811.06817).
- Kochenderfer et al., [*Algorithms for Validation*, Chapter 12: Runtime
  Monitoring](https://algorithmsbook.com/validation/files/val.pdf).

## Proposed baseline 2: RSS safety-envelope monitor

### Method

Use BEVFormer detections to estimate the position and motion of nearby actors.
At every monitor update, evaluate the planned ego trajectory against
Responsibility-Sensitive Safety (RSS)-style longitudinal and lateral safe
distances. The calculation should use explicit response time, braking,
acceleration, and lateral-clearance assumptions.

If the native SparseDriveV2 trajectory enters an unsafe envelope, the monitor
replaces it with a bounded braking trajectory. It returns control only after the
scene remains safe for a short confirmation window. All decisions and individual
rule violations should be logged so that an intervention can be explained after
the run.

### Interpretation and limitation

This baseline represents deterministic, property-based monitoring. It should be
named an **RSS safety-envelope monitor** unless the implementation reproduces all
assumptions and formal definitions of the complete RSS framework. Detection and
velocity-estimation errors can cause both missed hazards and unnecessary stops,
which is why the main comparison should use the same BEVFormer perception source
as sensor-based AutoAgent0.

### Original literature

- Shalev-Shwartz, Shammah, and Shashua, [*On a Formal Model of Safe and Scalable
  Self-driving Cars*](https://arxiv.org/abs/1708.06374).
- Kochenderfer et al., [*Algorithms for Validation*, Chapter 12: Runtime
  Monitoring](https://algorithmsbook.com/validation/files/val.pdf).

## Proposed baseline 3: MPC predictive safety filter

### Method

Treat the native SparseDriveV2 trajectory as the nominal reference. Over a short
receding horizon, predict ego motion with a kinematic vehicle model and nearby
actor occupancy with BEVFormer state estimates. First test the nominal trajectory
against collision, drivable-area, control, and dynamic-feasibility constraints.

If the nominal plan is feasible, pass it through unchanged. Otherwise, solve for
a nearby feasible braking or avoidance trajectory that minimizes deviation from
the SparseDriveV2 reference, control effort, and discomfort. If the optimizer
cannot produce a valid solution within its runtime budget, execute the common
minimum-risk stop.

Conceptually, the safety filter solves

```text
minimize    deviation from SparseDriveV2 + control effort + discomfort
subject to vehicle dynamics, control bounds, and safety constraints.
```

The proposed implementation uses bounded trajectory/control samples followed by
short-horizon optimization, with its adaptation and limitations described
precisely. It does not attempt to reproduce an entire fallback-safe MPC research
stack inside the current simulator integration.

### Interpretation and limitation

The recommended name is **MPC predictive safety filter**, emphasizing that the
method supervises a learned policy rather than replacing SparseDriveV2 with a
complete model-based driving planner. Solver latency, infeasibility, inaccurate
actor prediction, and overly conservative constraints must all be measured.

### Original literature

- Wabersich and Zeilinger, [*A Predictive Safety Filter for Learning-Based
  Control of Constrained Nonlinear Dynamical Systems*](https://arxiv.org/abs/1812.05506).
- Sinha et al., [*Closing the Loop on Runtime Monitors with Fallback-Safe
  MPC*](https://arxiv.org/abs/2309.08603).
- Kochenderfer et al., [*Algorithms for Validation*, Chapter 12: Runtime
  Monitoring](https://algorithmsbook.com/validation/files/val.pdf).

## Fair comparison and evaluation

### Proposed paper table

| Method | DS &uarr; | RC &uarr; | SR &uarr; | Collision rate &darr; |
|---|---:|---:|---:|---:|
| SparseDriveV2 | -- | -- | -- | -- |
| + Uncertainty monitor | -- | -- | -- | -- |
| + RSS monitor | -- | -- | -- | -- |
| + MPC safety filter | -- | -- | -- | -- |
| + AutoAgent0 | -- | -- | -- | -- |

The following diagnostic metrics should also be retained, even if they are moved
to supplementary material:

- monitor intervention rate and intervention duration;
- interventions on nominal versus shifted routes;
- collisions or failures preceded by an alarm;
- unnecessary interventions and routes lost to excessive stopping;
- return-to-policy rate after fallback;
- monitor latency; and
- MPC timeout, solver-failure, and infeasibility rates.

These diagnostics distinguish genuine hazard mitigation from a monitor that
appears safe only because it frequently stops the vehicle.

### Evaluation scale and staging

The full comparison described in the current experiment draft contains five
methods, 200 paired Fail2Drive routes, and three repetitions.
