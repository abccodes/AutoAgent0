# AutoAgent0 Design

## Goal

AutoAgent0 is currently a HUGSIM-backed, SceneSmith-inspired agentic wrapper
around the trajectory generation and VLM routing logic already implemented in
this repo. The current design goal is to make the system look like an explicit
Orchestrator + Designer workflow while preserving existing behavior.

This document describes the current implementation-level design. The active
`rap_autoagent0` and `drivor_autoagent0` methods implement a first bounded
recovery-loop prototype. They do not yet define a deterministic verifier,
memory module, recovery endpoint planner, or rule-metric critic loop. Those
components remain future extension points.

Current v1 scope:
- Active: camera streams, language instruction, orchestrator routing,
  learned/rule-based candidate generation, VLM intervention/scoring, planner
  gate selection, one-redesign AutoAgent0 recovery-loop selection, and final
  trajectory payload construction.
- Deferred: memory, active verifier, recovery endpoint generation, and
  rule-metric critic loops.

## SceneSmith-Inspired Agent Pattern

SceneSmith uses a planner/designer/critic workflow where a planner coordinates
tool calls to a designer and critic, then requests revisions when needed. For
AutoAgent0, we use that structure as an architectural pattern rather than
copying SceneSmith directly.

Mapping:
- SceneSmith Planner -> AutoAgent0 Orchestrator.
- SceneSmith Designer -> learned and rule-based trajectory generators.
- SceneSmith Critic -> future verifier/critique module.
- SceneSmith checkpoint/reset -> future last-safe fallback or recovery behavior.

The current system already has most of the functional pieces. The main design
change is to name them in an agentic way and expose clean boundaries:
- the Orchestrator coordinates which path to run,
- Designers generate candidate trajectories,
- existing VLM intervention/scoring/gating selects between options,
- HUGSIM receives the selected final trajectory through the existing payload
  path.

Current SceneSmith-style tool vocabulary:
- `request_designer(action_generation)`: active. Requests candidates from
  learned and/or rule-based expert modules.
- `select_final_actions`: active. Selects the final trajectory according to the
  current method semantics.
- `request_critique`: active in `*_autoagent0` through the current VLM
  intervention mechanism. Future versions should replace or augment this with a
  deterministic verifier and rule-metric critic.
- `request_design_change(intervention)`: active as a trace/runtime transition in
  `*_autoagent0`. In this prototype it does not call a separate recovery
  endpoint generator; it triggers expanded learned + rule-based candidate
  generation.

## Current Runtime Flow

At each HUGSIM planning step, the simulator provides observations and metadata
to the planner adapter. The active planner configuration determines which
AutoAgent0-style flow is used.

1. HUGSIM supplies camera streams, ego state, route/task instruction, and
   optional privileged state.
2. The active planner adapter builds learned and/or rule-based candidate
   trajectories.
3. The Orchestrator path determines whether candidates are used directly,
   merged into one pool, or separated into learned/rule-based families.
4. Existing VLM intervention/scoring or planner-gate logic selects the final
   trajectory when enabled by the method.
5. The selected local trajectory and debug metadata are returned in the
   normalized HUGSIM plan payload.
6. `closed_loop.py` converts the trajectory into simulator control and writes
   evaluation/debug artifacts.

For Method A/B and existing intervention baselines this remains
behavior-preserving. For `rap_autoagent0` and `drivor_autoagent0`, the active
runtime behavior changes to a bounded recovery loop: critique one default
trajectory first, generate/score expanded candidates only when the critique
requests redesign, then critique the revised candidate once before execution or
fallback.

## Module Design

### Camera Streams

Camera streams are HUGSIM image observations passed into the VLM-facing parts of
the system. Current supported VLM camera modes are:
- front-only: the VLM sees the front camera.
- 4-camera / multiview: the VLM sees front plus additional surrounding context.

The front camera is primary for trajectory overlays because this is where
candidate paths are rendered. Side/rear camera views provide context for nearby
vehicles, adjacent lanes, obstacles, and surrounding road structure. In the
current prompts, multiview context should support judgment of safety and
surroundings rather than override the front-view path geometry.

### Language Instruction

The language instruction comes from the route command or from task metadata.
This instruction is passed into the current VLM intervention, scoring, and
planner-gate prompts.

For normal benchmark runs, the instruction is the route command such as
straight, left, or right. For curated demos, task metadata can override the
generic route text so the VLM receives a task-specific instruction such as
stopping at a marked target or parking near a target.

### Orchestrator

The Orchestrator coordinates which candidate generation and selection path runs.
In the current code, this is mostly implemented through the VLM selector,
planner-flow helpers, and method-specific planner configs.

Current Orchestrator behaviors:
- Base policy: request learned candidates only and use the learned planner's
  default/top candidate behavior.
- VLM intervention: request learned candidates, run the existing intervention
  gate, and invoke the VLM scorer when intervention is needed.
- Method A / Choice A: request learned and rule-based candidates, merge them
  into a single candidate pool, then use the existing VLM scorer to select the
  final trajectory.
- Method B / Choice B: request learned and rule-based candidate families
  separately, run the VLM planner gate to choose the family, then use that
  family's top/default trajectory.
- AutoAgent0 recovery loop: request one learned default trajectory, critique it
  with VLM intervention, request expanded candidates only on rejection, score
  the expanded pool, critique the selected revision, and then execute or fall
  back.

The current Orchestrator is still implemented inside existing HUGSIM planner
clients and `AutoAgent0Runtime`; it is not a standalone long-running agent
server. The `*_autoagent0` path is the first active tool-call-style loop.

### Trajectory Generation

Learned trajectory generation comes from RAP or DrivoR. These remain backend
adapters under `planners/` because they are tied to their own launch scripts,
model environments, config fields, and output formats.

AutoAgent0 treats RAP and DrivoR as learned expert Designers:
- they generate candidate local trajectories,
- they provide learned planner scores or rankings,
- they expose candidate metadata for visualization and VLM scoring,
- they return the selected local trajectory through the normalized payload path.

The learned planner's internal score is useful within that planner family, but
it is not treated as a universal score that can be directly compared with the
rule-based planner's score.

### Rule-Based Trajectory Generation

Rule-based trajectory generation comes from the external Rule-Planner module
through the existing HUGSIM adapter. For this design pass, the rule-based
generator is treated as an abstract working module. Another teammate may improve
its internals, but AutoAgent0 should depend only on the adapter contract.

The rule-based generator can currently participate in four ways:
- standalone `rule_based` baseline,
- Method A candidate proposals merged with learned candidates,
- Method B candidate family considered by the planner gate.
- AutoAgent0 recovery-loop expanded candidates after the default trajectory is
  rejected by VLM critique.

Future recovery-specific rule-based generation is intentionally left
unimplemented here. A later design may allow the Orchestrator to provide a
recovery endpoint or high-level recovery intent, then ask the rule-based module
to generate a recovery trajectory.

### End-to-End Scorer

The end-to-end scorer refers to the existing learned planner scoring or
top-candidate logic from RAP/DrivoR. It operates inside the learned planner
family and helps identify the learned planner's default/top candidate.

This is not a unified cross-policy scorer. RAP/DrivoR scores and rule-based
scores are not assumed to be directly comparable. When the system needs to
compare across learned and rule-based candidate families, it uses the existing
VLM scorer or planner gate instead of comparing raw planner scores.

### Rule-Based Scorer

The rule-based planner has its own scoring/selection logic for rule-based
candidate quality. That score is meaningful within the rule-based family.

Current use:
- In standalone `rule_based`, the rule-based adapter returns the selected
  rule-based trajectory.
- In Method A, rule-based candidates enter the merged pool and are later judged
  by the VLM scorer alongside learned candidates.
- In Method B, the VLM gate first chooses the planner family. If it chooses
  rule-based, the selected rule-based family candidate is used directly.
- In `*_autoagent0`, rule-based scores are only used to order the rule-based
  rows before they are placed into the expanded redesign pool. The VLM scorer
  still performs the cross-family selection.

Future critique metrics may include out-of-road detection, collision detection,
TTC, symbolic object state, bounding-box state, and map checks. These metrics
would be closer to a SceneSmith Critic or Verifier, but they are not active in
the current v1 design.

### Final Trajectory And States

The final output is the selected local trajectory plus planner debug metadata.
HUGSIM consumes this through the existing plan payload path, then
`closed_loop.py` converts the trajectory to simulator controls and records
outputs.

Current payload generation is normalized through AutoAgent0 helpers, but the
backend adapters still own planner-specific details. This split keeps the
planner-specific RAP/DrivoR/Rule-Planner logic stable while giving the
Orchestrator a consistent view of final actions and debug state.

### Verifier Agent

The deterministic verifier agent is future work. A passive verifier trace
exists in the behavior-preserving scaffolding, but it always accepts and does
not affect actions, fallbacks, metrics, or selected trajectories.

For the active `*_autoagent0` prototype, `request_critique` uses the current VLM
intervention mechanism as a temporary VLM Critic. The VLM Critic checks a single
candidate trajectory and maps `should_intervene = false` to accepted and
`should_intervene = true` to rejected/redesign-needed. This is not the final
rule-based verifier; it is a bootstrap critique mechanism so the agentic loop
can be tested before map/TTC/collision checks are implemented.

A future active verifier could use symbolic state and visual context for:
- out-of-road or map boundary checks,
- collision checks,
- TTC checks,
- lane/rule checks,
- structured rejection reasons that are fed back to the Orchestrator.

This future verifier should be treated as the eventual SceneSmith-style Critic,
but the current VLM intervention/scorer should not be renamed into a full Critic
agent yet.

### Judge And Memory Agent / Memory

Judge and memory behavior is out of scope for this pass. The current design does
not specify memory schema, update policy, retrieval policy, or how memory
changes control behavior.

Future memory may store:
- reusable recovery patterns,
- successful rule-based behaviors,
- summarized failure modes,
- high-level language rules such as how to handle repeated stop-sign or
  recovery scenarios.

Those details should be filled in after the active Orchestrator + Designer flow
is stable.

## Current Method Mapping

Current public methods remain the source of truth for experiments.

| Method | AutoAgent0 interpretation | Current behavior |
| --- | --- | --- |
| `rap_vlm` | Learned Designer only | RAP generates candidates; learned/VLM path returns selected trajectory. |
| `drivor_vlm` | Learned Designer only | DrivoR generates candidates; learned/VLM path returns selected trajectory. |
| `rap_intervention_4cam` | Learned Designer + intervention selection | RAP candidates with current VLM intervention/scorer path. |
| `drivor_intervention_4cam` | Learned Designer + intervention selection | DrivoR candidates with current VLM intervention/scorer path. |
| `rule_based` | Rule-based Designer only | Rule-Planner adapter returns the selected rule-based trajectory. |
| `rap_impl_a` | Ablation: learned + rule-based Designers, merged pool | RAP and rule-based candidates are merged; VLM scorer selects final trajectory. |
| `drivor_impl_a` | Ablation: learned + rule-based Designers, merged pool | DrivoR and rule-based candidates are merged; VLM scorer selects final trajectory. |
| `rap_impl_b` | Ablation: learned + rule-based Designers, planner gate | VLM planner gate chooses RAP family or rule-based family. |
| `drivor_impl_b` | Ablation: learned + rule-based Designers, planner gate | VLM planner gate chooses DrivoR family or rule-based family. |
| `rap_autoagent0` | Agentic recovery loop | RAP default trajectory is critiqued first; expanded RAP + rule-based candidates are scored only when critique requests redesign. |
| `drivor_autoagent0` | Agentic recovery loop | DrivoR default trajectory is critiqued first; expanded DrivoR + rule-based candidates are scored only when critique requests redesign. |

## Active Recovery-Loop Prototype

The first active AutoAgent0 recovery-loop prototype keeps the Critic simple:
`request_critique(...)` uses the existing VLM intervention mechanism. A default
learned trajectory is checked first. If the VLM intervention says no redesign is
needed, the default trajectory is executed. If it requests intervention, the
Orchestrator asks for an expanded learned + rule-based candidate pool, uses the
existing VLM scorer to select a revised trajectory, critiques the revised
trajectory once more, and then either executes it or falls back to a hold
trajectory.

This prototype deliberately does not implement memory, deterministic map/TTC
verification, multi-iteration redesign, or a new rule-based scorer. Method A/B
remain runnable ablations and should not be treated as the final AutoAgent0
workflow.

Concrete implemented loop:

1. `request_designer(mode="default", k=1)` selects the learned planner's default
   candidate.
2. `request_critique(phase="default")` calls VLM intervention on that one
   candidate.
3. If accepted, `select_final_actions` executes the default trajectory.
4. If rejected, `request_design_change` records the VLM critique reason and
   corrective action.
5. `request_designer(mode="recovery", k=10)` builds an expanded learned +
   rule-based candidate pool.
6. The existing VLM scorer selects one revised candidate from that pool.
7. `request_critique(phase="revised")` calls VLM intervention on the revised
   candidate.
8. If accepted, the revised candidate is executed. If rejected, the current
   fallback is a hold trajectory.

The loop does not currently keep adding more trajectories after the revised
candidate is rejected. `redesign_candidate_budget` caps the expanded pool at 10
in the current configs. `max_redesign_attempts` is present in config for the
future, but the current implementation is effectively fixed at one redesign
attempt.

## Codebase Mapping

Active implementation anchors:
- `closed_loop.py`: HUGSIM rollout, simulator step loop, output writing, and
  final evaluation artifacts.
- `planners/rap/client.py`, `planners/drivor/client.py`, and
  `planners/rule_based/client.py`: backend adapters for learned and rule-based
  expert modules.
- `autoagent0/core/designer.py`: candidate-row normalization and AutoAgent0
  trajectory candidate batches.
- `autoagent0/core/runtime.py`: SceneSmith-style facade that exposes the
  current behavior-preserving flow and the opt-in recovery-loop flow through
  `request_designer(...)`, `request_critique(...)`, `request_design_change(...)`,
  and `select_final_actions(...)`-style tool calls.
- `autoagent0/core/config.py`: config/env bridge for the `autoagent0:` planner
  config block used by `rap_autoagent0` and `drivor_autoagent0`.
- `autoagent0/core/planner_flow.py`: current base-policy, Method A, and Method
  B selection-flow helpers.
- `autoagent0/core/orchestrator.py`: current VLM decision parsing/coercion and
  selected-candidate helpers.
- `autoagent0/prompts/orchestrator.py`: current intervention, scoring, and
  planner-gate prompts.
- `autoagent0/adapters/hugsim/`: HUGSIM-specific context, runtime, video,
  results, geometry, default trajectory, overlay, and IO helpers.

## Future Extension Points

The next agentic extensions should be added only after the current design is
stable and smoke-tested:
- Active verifier/critic with rule-metric checks.
- Verifier rejection reasons fed back into the Orchestrator.
- Recovery-specific rule-based generation from a recovery endpoint or high-level
  recovery intent.
- Last-safe checkpoint/fallback behavior inspired by SceneSmith reset semantics.
- Memory for reusable recovery patterns and successful rule-based behaviors.

## Verification

This document is a design update only. No runtime tests are required when
changing only this file.

If future runtime code changes follow this design, run the all-method smoke
suite:

```bash
bash scripts/baselines/smoke/submit_method_smoke.sh
```
