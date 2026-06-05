# AutoAgent0 Architecture

AutoAgent0 is the agentic driving layer being added inside this HUGSIM fork. It
now has two runtime uses:
- a behavior-preserving structure around the existing HUGSIM planner/VLM paths,
- an opt-in recovery-loop prototype for `rap_autoagent0` and
  `drivor_autoagent0`.

The existing baseline methods keep their current semantics. The recovery-loop
prototype is a separate method family and should not be treated as a replacement
for Method A/B ablation baselines.

## Current Boundaries

- `autoagent0/core/` defines the reusable agent contracts: scene context,
  trajectory candidates, orchestrator decisions, verifier results, memory, and
  debug traces.
- `autoagent0/experts/` names RAP, DrivoR, and rule-based planners as expert
  backends. The actual model/client code remains in `planners/`.
- `autoagent0/adapters/hugsim/` contains HUGSIM conversion, runtime, video,
  demo-task, output/result, and visualization helpers. HUGSIM is the current
  evaluation backend, not the long-term core abstraction.
- `autoagent0/prompts/` contains legacy prompt builders for current
  intervention/scoring/planner-gate paths plus role-specific Planner, Designer,
  and Critic prompt builders for the active `*_autoagent0` path.
- `autoagent0/vlm/` contains shared VLM backend, parsing, and debug utilities
  used by the current selector paths.

The first shared-code migration is intentionally narrow:
- `autoagent0/core/candidates.py` owns candidate summarization, candidate-row
  formatting, path-length helpers, and planner-gate candidate filtering.
- `autoagent0/core/payloads.py` owns normalized plan payload construction used
  by RAP, DrivoR, and rule-based client adapters.
- `autoagent0/core/planner_flow.py` owns shared planner-flow extraction helpers
  for method A/B and base-policy paths.
- `autoagent0/core/runtime.py` owns the SceneSmith-style runtime facade. It
  keeps the behavior-preserving Method A/B/base-policy flow and also exposes
  the opt-in one-redesign AutoAgent0 recovery loop.
- `autoagent0/core/config.py` owns the `AUTOAGENT0_*` config/env bridge used by
  RAP and DrivoR clients.
- `autoagent0/adapters/hugsim/context.py` owns current route/task/camera/ego
  context extraction helpers.
- `autoagent0/adapters/hugsim/defaults.py`, `geometry.py`, `io.py`, and
  `overlays.py` own HUGSIM default-trajectory, geometry, filesystem/image, and
  overlay helpers shared across planner clients.
- `autoagent0/adapters/hugsim/runtime.py` owns open-FIFO message IO and planner
  process liveness checks for the HUGSIM runner.
- `autoagent0/adapters/hugsim/video.py`, `demo_tasks.py`, and `results.py` own
  front/video rendering, curated demo task completion/overrides, and eval
  performance summary/output naming helpers.
- `autoagent0/prompts/orchestrator.py` owns current scoring, intervention, and
  planner-gate prompt builders for legacy methods.
- `autoagent0/prompts/critic.py` owns the active `*_autoagent0`
  single-candidate critique prompt.
- `autoagent0/prompts/planner.py` owns the active `*_autoagent0`
  revised-candidate final selection prompt.
- `autoagent0/prompts/designer.py` owns the design-change prompt boundary for
  future dynamic designer requests.
- `autoagent0/core/orchestrator.py` owns current VLM output coercion and
  selection helpers.
- `autoagent0/vlm/backends.py`, `parsing.py`, and `debug.py` own shared VLM
  model-call, JSON parsing, and debug artifact helpers.
- `autoagent0/experts/rule_based.py` is the normalized wrapper around the
  existing Rule-Planner provider.

`planners/common/vlm_selector.py` remains the active runtime integration point
and compatibility facade. Existing code should continue importing from it where
needed, but new shared helper edits should go into `autoagent0/` first and be
re-exported through the facade only when required for compatibility.

`closed_loop.py` remains the active HUGSIM rollout entrypoint. It now delegates
runtime pipe IO, video writing, demo task handling, and result/performance
summary construction to `autoagent0/adapters/hugsim/`, but it still owns the
HUGSIM environment loop and scene loading.

`planners/rap/client.py`, `planners/drivor/client.py`, and
`planners/rule_based/client.py` remain the active planner backend adapters.
They still own planner-specific launch/config/model integration; shared payload,
image, geometry, and default-trajectory helpers are imported from `autoagent0/`.

## Mapping From Existing Methods

- Solo VLM intervention maps to an AutoAgent0 learned-intervention flow.
- Choice A / Method A maps to an agentic rule-merge flow: learned and rule-based
  trajectories are merged, then the existing intervention/scorer path selects.
- Choice B / Method B maps to an agentic policy-gate flow: learned and
  rule-based candidates are separate, and the VLM gate chooses the planner
  family.
- Standalone `rule_based` remains an expert baseline.
- `rap_autoagent0` and `drivor_autoagent0` map to the active AutoAgent0
  recovery-loop prototype.

Existing baseline IDs and configs remain the source of truth for current runs:
`rap_vlm`, `drivor_vlm`, `rap_intervention_4cam`,
`drivor_intervention_4cam`, `rule_based`, `rap_impl_a`, `drivor_impl_a`,
`rap_impl_b`, `drivor_impl_b`, `rap_autoagent0`, and `drivor_autoagent0`.

## Active Recovery Loop

The `*_autoagent0` methods implement a bounded agentic loop:

1. Generate the learned planner's default/top trajectory.
2. Call `request_critique` using the AutoAgent0 VLM Critic prompt on that
   single trajectory.
3. If the critique accepts it, execute the default trajectory immediately.
4. If the critique requests intervention, call `request_design_change` and build
   an expanded candidate pool from learned candidates plus existing
   Rule-Planner candidates.
5. Use the AutoAgent0 Planner prompt to select one revised candidate from that
   pool.
6. Critique the revised candidate once with the AutoAgent0 VLM Critic prompt.
7. Execute the revised candidate if accepted. If the final critique still
   rejects, the current runtime falls back or executes the VLM Planner-selected
   revised candidate depending on the configured redesign-limit behavior.

This first prototype is intentionally bounded. It does not keep adding more
trajectories after the revised candidate is rejected. The current config exposes
`max_redesign_attempts = 3`, but repeated redesign iterations are not
implemented yet; the current loop performs one expanded redesign pass and uses
that value only for final-rejection behavior:

```yaml
autoagent0:
  enabled: true
  mode: recovery_loop
  redesign_candidate_budget: 10
  max_redesign_attempts: 3
  fallback_mode: hold
```

The expanded pool is capped by `redesign_candidate_budget`. With the current
configs this is 10 total candidates, composed from learned candidate rows and
available rule-based candidate rows.

## Phase-1 Verifier

The passive verifier object still exists for the behavior-preserving path. For
the active `*_autoagent0` prototype, critique/rejection is driven by the
dedicated AutoAgent0 VLM Critic prompt rather than by a deterministic
rule-based verifier. This is a temporary visual Critic implementation.

Future phases will add:
- off-road/map checks
- collision and TTC checks from symbolic state
- active `brake_and_hold` fallback
- structured rejection feedback to the orchestrator
- recovery trajectory generation
- memory for reusable recovery patterns

## Debug Trace

Frame-level VLM debug JSON now includes `agent_trace`. This records:
- designer candidate counts by source
- orchestrator decision type
- selected source or planner family
- critique phase and redesign request for `*_autoagent0`
- passive verifier acceptance for behavior-preserving paths
- previous verifier feedback, currently empty unless later extensions populate it

The trace is diagnostic only. It is not used by `eval.json` scoring and should
not affect plan payloads or selected trajectories.

## Verification when making large changes

After the phase-1 refactor, the one-scene all-method smoke suite passed on
NuScenes scene `scene-0010-easy-00` for all nine canonical methods:
`rap_vlm`, `drivor_vlm`, `rap_intervention_4cam`,
`drivor_intervention_4cam`, `rule_based`, `rap_impl_a`, `drivor_impl_a`,
`rap_impl_b`, and `drivor_impl_b`.

The `rap_autoagent0` and `drivor_autoagent0` methods were added after that smoke
run and should be verified with one-scene debug runs before adding them to the
default all-method smoke suite.

Use this command after future large refactors:

```bash
bash scripts/baselines/smoke/submit_method_smoke.sh
```
