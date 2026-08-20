# Closed-Loop Monitor Baselines for SparseDriveV2

> **Status:** Draft for feedback; no implementation or experiments have started.
>
> **Scope note:** This document concerns the current SparseDriveV2/Fail2Drive
> system in `jiagengliu02/AgenticDriving`. It is stored in the AutoAgent0
> repository as a discussion document, but the proposed monitors are separate
> comparison systems rather than additions to the older HUGSIM implementation
> in this repository.

## Summary

We propose evaluating three traditional closed-loop runtime monitors around a
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
unsafe action. A passive alarm-only monitor would be useful diagnostically, but
it would execute the same actions as unmodified SparseDriveV2 and therefore
should not change route-level driving score, success rate, or collision rate.
This interpretation needs confirmation before implementation.

The proposed comparison is:

| Method | Shared base policy | Runtime intervention |
|---|---|---|
| SparseDriveV2 | SparseDriveV2 | None |
| + Uncertainty monitor | SparseDriveV2 | Minimum-risk slowdown/stop |
| + RSS monitor | SparseDriveV2 | Bounded braking trajectory |
| + MPC safety filter | SparseDriveV2 | Closest feasible trajectory or emergency stop |
| + AutoAgent0 | SparseDriveV2 | Verified agentic recovery |

## Where the monitors fit

### SparseDriveV2 base policy

At each planning update, SparseDriveV2 consumes the camera observations, ego
state, and route/navigation input. It produces a ranked collection of future ego
trajectories. The native policy selects its highest-scoring trajectory, and the
existing trajectory controller converts that plan into steering, throttle, and
brake commands.

```text
CARLA sensors + ego state + route command
                    |
                    v
              SparseDriveV2
                    |
                    v
       native selected future trajectory
                    |
                    v
             PID/controller
                    |
                    v
              vehicle control
```

SparseDriveV2 is a strong nominal driver, but its learned score is not itself a
safety guarantee. On out-of-distribution or long-tail scenes it can confidently
select a trajectory that collides, becomes blocked, violates the route intent,
or repeatedly makes no progress.

### AutoAgent0 wrapper

The current AutoAgent0 system is more than a single monitor. Its regular loop
constructs and scores alternative SparseDriveV2 proposals, checks the selected
proposal with an external verifier, and runs the accepted plan. When the plan is
rejected or execution becomes unhealthy, its recovery loop can diagnose the
failure, choose a bounded maneuver, revise the proposal using verifier feedback,
and use episode history to avoid repeating unsuccessful actions.

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

The conventional baselines should **not** reuse AutoAgent0's VLM, PDMS
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

The thresholds should be calibrated on held-out Bench2Drive data, never on
Fail2Drive. A reasonable calibration objective is to keep the intervention rate
low on nominal routes while detecting as many impending closed-loop failures as
possible.

### Interpretation and limitation

This is a lightweight uncertainty baseline because it uses outputs already
available from SparseDriveV2 and requires neither retraining nor repeated neural
network inference. SparseDriveV2 scores are not guaranteed to be calibrated
probabilities, however. The paper should therefore call the signal a
**score-based predictive uncertainty proxy**, not a formally calibrated measure
of epistemic uncertainty.

Monte Carlo dropout or an ensemble would more closely measure model uncertainty,
but would require repeated inference, architectural support, and substantially
more runtime. That is not recommended for the first baseline unless a more
faithful uncertainty reproduction is required.

### Initial effort estimate

Approximately **2--4 working days** for integration, logging, fallback behavior,
and unit/smoke tests, followed by calibration and scenario evaluation.

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

### Initial effort estimate

Approximately **3--6 working days**, with the largest uncertainty coming from
the consistency of BEVFormer actor velocities and coordinate conversion.

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

A practical first version can use bounded trajectory/control samples followed by
short-horizon optimization, provided the method and limitations are described
precisely. This is substantially more feasible than reproducing an entire
fallback-safe MPC research stack inside the current simulator integration.

### Interpretation and limitation

The recommended name is **MPC predictive safety filter**, emphasizing that the
method supervises a learned policy rather than replacing SparseDriveV2 with a
complete model-based driving planner. Solver latency, infeasibility, inaccurate
actor prediction, and overly conservative constraints must all be measured.

### Initial effort estimate

Approximately **5--10 working days**. This is the highest-risk baseline because
real-time solver behavior and coordinate/dynamics validation can require
substantial debugging.

### Original literature

- Wabersich and Zeilinger, [*A Predictive Safety Filter for Learning-Based
  Control of Constrained Nonlinear Dynamical Systems*](https://arxiv.org/abs/1812.05506).
- Sinha et al., [*Closing the Loop on Runtime Monitors with Fallback-Safe
  MPC*](https://arxiv.org/abs/2309.08603).
- Kochenderfer et al., [*Algorithms for Validation*, Chapter 12: Runtime
  Monitoring](https://algorithmsbook.com/validation/files/val.pdf).

## Fair comparison and evaluation

### Recommended experimental contract

- Freeze the same SparseDriveV2 checkpoint for every row.
- Do not train, fine-tune, or calibrate on Fail2Drive routes.
- Share routes, seeds, repetition indices, controller settings, and simulator
  settings across all methods.
- Use BEVFormer rather than simulator ground truth in the reported sensor-only
  monitor comparison.
- If useful, report ground-truth perception only as a clearly labeled oracle
  diagnostic.
- Do not expose AutoAgent0's proposal scoring, verifier, VLM, recovery skills, or
  memory to the three traditional baselines.
- Use one common, bounded minimum-risk braking implementation wherever a monitor
  requires an emergency fallback. The MPC filter may additionally generate a
  feasible corrective trajectory.
- Establish analytical RSS parameters and calibrate learned thresholds/MPC
  weights on held-out Bench2Drive routes before freezing the configurations.

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
methods, 200 paired Fail2Drive routes, and three repetitions:

```text
5 methods x 200 routes x 3 repetitions = 3,000 route episodes
```

At an average of 5--10 minutes per episode, this is approximately **250--500
simulator-hours** before parallelization, retries, initialization failures, and
result validation. This is an order-of-magnitude planning estimate, not a
schedule commitment.

Recommended execution stages after the design is approved:

1. Unit-test monitor calculations and coordinate transformations.
2. Validate each intervention in small, controlled CARLA scenarios.
3. Run the existing ten-route long-tail stress set.
4. Run one complete Fail2Drive pass and inspect intervention traces.
5. Freeze all configurations and run the final three repetitions.

## Feedback requested before implementation

The recommendations below are deliberately explicit so that implementation does
not begin with different assumptions across team members.

### 1. Active versus passive monitoring

- [ ] **Recommended:** Treat each monitor as an active runtime wrapper that can
  replace the native SparseDriveV2 plan.
- [ ] Alternative: Log alarms only and evaluate detection metrics without
  expecting route-level driving improvement.

**Question:** Does “closed-loop monitor” in the requested comparison mean active
intervention? A passive monitor would leave SparseDriveV2 driving behavior
unchanged.

### 2. Baseline families

- [ ] **Recommended:** Approve score-based uncertainty, RSS safety envelope, and
  MPC predictive safety filtering as the three complementary baselines.
- [ ] Revise one or more baseline families.

**Question:** Are these the intended three categories, or should a different
traditional monitor replace one of them?

### 3. MPC fidelity

- [ ] **Recommended:** Build a practical short-horizon predictive safety filter
  adapted to the existing SparseDriveV2 trajectory interface.
- [ ] Reproduce a specific fallback-safe MPC paper more faithfully, accepting a
  larger engineering scope.

**Question:** Is the practical safety-filter adaptation sufficient, or is exact
reproduction of a particular MPC method expected?

### 4. Perception and oracle use

- [ ] **Recommended:** Use BEVFormer for every reported sensor-based monitor and
  reserve simulator ground truth for separately labeled diagnostics.
- [ ] Permit privileged simulator actors in the main traditional-monitor rows.

**Question:** Must all primary comparison rows be sensor-only? This determines
whether the comparison measures monitor design or an advantage from privileged
perception.

### 5. Uncertainty definition and calibration

- [ ] **Recommended:** Use SparseDriveV2 candidate-score entropy and margin,
  calibrated on held-out Bench2Drive routes.
- [ ] Require Monte Carlo dropout or an ensemble as the uncertainty baseline.

**Question:** Is a lightweight score-based uncertainty proxy acceptable? Should
the calibration target a specific nominal intervention or false-alarm rate?

### 6. Common fallback

- [ ] **Recommended:** Use one bounded minimum-risk slowdown/stop for uncertainty,
  RSS, and MPC solver failure.
- [ ] Allow each monitor to use a separate fallback implementation.

**Question:** Should fallback behavior be shared to isolate monitor quality, or
is method-specific fallback part of the intended comparison?

### 7. Evaluation scope and table format

- [ ] **Recommended:** Prototype on the ten-route stress set, then report the full
  200-route Fail2Drive comparison over three repetitions.
- [ ] Use a smaller Fail2Drive subset for the monitor comparison.

**Question:** Is the full three-repetition Fail2Drive evaluation required? Please
also provide the referenced Table 7 if its formatting or metrics differ from the
proposed table above.

### 8. AutoAgent0 comparison row

- [ ] **Recommended:** Rerun the final AutoAgent0 comparison with BEVFormer so all
  primary rows are sensor-only.
- [ ] Retain a ground-truth AutoAgent0 row but label it as an oracle diagnostic.

**Question:** Which AutoAgent0 configuration should appear in the primary table?
The current C4 ablation configuration uses ground-truth perception and would not
be a controlled sensor-only comparison without a rerun.

## Proposed approval statement

If the recommendations above match the intended study, the working experimental
specification is:

> We interpret a closed-loop monitor as an active runtime wrapper around a frozen
> SparseDriveV2 policy. We will compare score-based uncertainty with a
> minimum-risk fallback, an RSS safety-envelope monitor, and an MPC predictive
> safety filter. Primary results will use sensor-based perception, no Fail2Drive
> training or calibration, and shared routes, seeds, controller settings, and
> fallback behavior. AutoAgent0 will remain a separate system and will not supply
> components to the traditional baselines.
