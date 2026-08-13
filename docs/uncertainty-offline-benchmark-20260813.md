# HUGSIM uncertainty offline benchmark (2026-08-13)

## Scope

This benchmark reuses the 94 completed DrivoR AutoAgent0 NuScenes routes from
`calibration-20260626` (7,775 frames, 19 base scene identities). It evaluates
whether saved uncertainty signals predict future evaluator NC failure. Scene
variants remain grouped during five-fold cross-validation, and thresholds are
selected on training folds only.

The predictive cohort excludes 573 frames whose saved plans already fail NC,
DAC, or collision. All 573 current failures are evaluator NC failures; none has
a saved simulator collision flag and none is a DAC failure.

An implementation audit showed that this is not evidence that the evaluator
invented 573 physical collisions. Evaluator NC rolls the saved multi-step plan
through static and object geometry, whereas `frame["collision"]` is sampled
before that frame's action is executed. The returned post-step state was never
attached to the frame, and a terminal collision therefore exited the loop
without being saved. The historical target is future evaluator plan-risk, not
confirmed future physical contact.

## Main results

The fixed legacy low-spread/low-cross gate (`intra <= 0.20` and
`cross <= 2.40`) produced:

- frame precision 0.238, recall 0.171, and 2.44x lift at 7.0% coverage;
- 16/49 merged failure episodes detected (event recall 0.327);
- 5.9% false-alert frame burden and 2.50 false-alert episodes per minute;
- median first-warning lead of 18.5 recorded frames at the 20-frame horizon.

Fifteen of the sixteen detected events came from three scene families. Adding
the legacy mode-count fallback lowered predictive precision from 0.238 to
0.142. A logistic model using all features recoverable from the historical
files achieved AUROC 0.506, versus 0.529 for the legacy-only feature model.
Neither is suitable for active deployment.

## Case review

Video inspection found plausible clearance hazards in some detected events,
including parked vehicles and construction cones. Other evaluator NC/TTC
failures occurred on visually open pavement while `frame["collision"]` remained
false, sometimes without recorded object boxes. The historical target should
therefore be described as an evaluator NC event, not a confirmed collision.

## Label audit

- All 7,775 historical frames lack a saved post-step execution outcome.
- No route reached the 401-frame runner timeout; 30/94 ended below 90% route
  completion, but the saved artifacts cannot distinguish collision from route
  departure for those endings.
- 545/573 NC-failing frames also fail evaluator TTC; 28 fail NC alone.
- Replaying the evaluator's object-box geometry finds planned object-overlap
  evidence in 360/573 NC-failing frames. The other 213 have no object overlap
  and therefore must be static-background NC failures. For the 360 overlap
  cases, the old artifacts cannot establish whether static geometry failed at
  an earlier trajectory step.

The runner now records the post-step reward, terminal flags, collision subtype,
route status, and termination reason under `execution_outcome`. New evaluator
outputs also retain `nc_failure_type` and `nc_failure_step`. These additions do
not alter NC, DAC, TTC, PDMS, control, or uncertainty routing.

There is also a static-geometry difference in the existing benchmark: physical
HUGSIM background collision uses an opacity-filtered point set, while evaluator
NC uses the broader exported `scene.ply`. This helps explain why some
background-only NC events are not visually obvious. We retain the official NC
calculation for comparability and report it as evaluator plan-risk rather than
silently redefining the benchmark.

## Decision

- Do not use mode count for active routing.
- Do not deploy the fitted historical logistic models.
- Keep the fixed quadrant only as a legacy reference.
- Collect the next corpus with `uncertainty_policy_mode: observe` and raw
  proposal telemetry before another active closed-loop A/B test.

## Reproduction

```bash
python scripts/benchmark_uncertainty_offline.py \
  --runs /path/to/calibration-20260626 \
  --horizon-steps 20 \
  --folds 5 \
  --seed 17 \
  --min-coverage 0.05 \
  --max-coverage 0.15 \
  --event-horizons 5 10 20 \
  --frame-dt-sec 0.25 \
  --event-merge-gap-steps 5 \
  --out-dir /path/to/offline-report
```

The analyzer writes the Markdown/JSON summary, enriched corpus, out-of-fold
predictions, horizon sweep, scene breakdown, and event/frame review tables.
