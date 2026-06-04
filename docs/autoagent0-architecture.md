# AutoAgent0 Architecture

AutoAgent0 is the agentic driving layer being added inside this HUGSIM fork. In
the first phase it is a behavior-preserving structure around the existing
HUGSIM planner/VLM paths. It does not change planner inference, candidate
ranking, VLM prompts, scoring, or fallback behavior.

## Current Boundaries

- `autoagent0/core/` defines the reusable agent contracts: scene context,
  trajectory candidates, orchestrator decisions, verifier results, memory, and
  debug traces.
- `autoagent0/experts/` names RAP, DrivoR, and rule-based planners as expert
  backends. The actual model/client code remains in `planners/`.
- `autoagent0/adapters/hugsim/` contains HUGSIM conversion helpers. HUGSIM is
  the current evaluation backend, not the long-term core abstraction.
- `autoagent0/prompts/` contains prompt builders used by the current
  intervention scorer and planner gate.

The first shared-code migration is intentionally narrow:
- `autoagent0/core/candidates.py` owns candidate summarization, candidate-row
  formatting, path-length helpers, and planner-gate candidate filtering.
- `autoagent0/adapters/hugsim/context.py` owns current route/task/camera/ego
  context extraction helpers.
- `autoagent0/prompts/orchestrator.py` owns current scoring, intervention, and
  planner-gate prompt builders.
- `autoagent0/core/orchestrator.py` owns current VLM output coercion and
  selection helpers.
- `autoagent0/experts/rule_based.py` is the normalized wrapper around the
  existing Rule-Planner provider.

`planners/common/vlm_selector.py` remains the active runtime integration point
and compatibility facade. Existing code should continue importing from it where
needed, but new shared helper edits should go into `autoagent0/` first and be
re-exported through the facade only when required for compatibility.

## Mapping From Existing Methods

- Solo VLM intervention maps to an AutoAgent0 learned-intervention flow.
- Choice A / Method A maps to an agentic rule-merge flow: learned and rule-based
  trajectories are merged, then the existing intervention/scorer path selects.
- Choice B / Method B maps to an agentic policy-gate flow: learned and
  rule-based candidates are separate, and the VLM gate chooses the planner
  family.
- Standalone `rule_based` remains an expert baseline.

Existing baseline IDs and configs remain the source of truth for current runs:
`rap_vlm`, `drivor_vlm`, `rap_intervention_4cam`,
`drivor_intervention_4cam`, `rule_based`, `rap_impl_a`, `drivor_impl_a`,
`rap_impl_b`, and `drivor_impl_b`.

## Phase-1 Verifier

The verifier is passive in this phase. It always returns `accepted=True` and is
recorded only in debug traces. It does not reject trajectories, trigger
fallbacks, alter metrics, or change selected actions.

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
- passive verifier acceptance
- previous verifier feedback, currently empty

The trace is diagnostic only. It is not used by `eval.json` scoring and should
not affect plan payloads or selected trajectories.
