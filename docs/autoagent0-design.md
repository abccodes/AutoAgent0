# AutoAgent0 Design

## TODO

Status legend:
- `Active`: exists and needs refinement.
- `Partial`: prototype exists but the module is not complete.
- `Future`: not implemented yet.

| Priority | Module | Status | Next Work |
| --- | --- | --- | --- |
| 1 | Orchestrator / runtime loop | Active | Validate the bounded `request_critique -> request_design_change -> request_designer -> select_final_actions` loop across more scenes. |
| 2 | Critic / verifier | Partial | Add deterministic verifier checks for off-road, collision, TTC, and map constraints. Start passive/log-only, then make it active once validated. |
| 3 | Rule-based scorer | Future | Define a clean rule-based scorer interface before changing cross-policy final selection logic. Do not compare raw RAP/DrivoR scores directly against Rule-Planner scores. |
| 4 | Recovery designer | Future | Replace the current candidate-normalization placeholder with targeted trajectory generation, e.g. "generate N more recovery trajectories." |
| 5 | Memory / judge-memory agent | Future | Keep memory out of the immediate runtime path until verifier and recovery behavior are stable; later store reusable recovery patterns and successful rule-based behaviors. |
| 6 | Simulator adapters | Partial | Keep HUGSIM as the active backend while defining the stable interface a future Fail2Drive/CARLA adapter must implement. |
| 7 | Debugging and evaluation | Active | Keep `agent_trace`, VLM JSON, `front.mp4`, and `video.mp4` aligned as runtime behavior changes. |

## Goal

AutoAgent0 is currently a HUGSIM-backed, SceneSmith-inspired agentic runtime
around the trajectory generation and VLM routing logic already implemented in
this repo. The current design goal is to make the active `*_autoagent0` methods
look like an explicit Planner + Designer + Critic workflow while preserving
existing Method A/B, intervention, and base-policy baselines as ablations.

This document defines the intended workflow and current design limits. File
ownership and module inventory live in `docs/autoagent0-architecture.md`.

The active `rap_autoagent0` and `drivor_autoagent0` methods implement a first
bounded recovery-loop prototype. They do not yet define a deterministic
verifier, memory module, recovery endpoint planner, or rule-metric critic loop.
Those components remain future extension points.

Current scope:
- Active: camera streams, language instruction, orchestrator routing,
  learned/rule-based candidate generation, VLM intervention/scoring, planner
  gate selection, bounded AutoAgent0 recovery-loop selection, and final
  trajectory payload construction.
- Deferred: memory, active verifier, recovery endpoint generation, and
  rule-metric critic loops.

## SceneSmith-Inspired Agent Pattern

SceneSmith uses a planner/designer/critic workflow where a planner coordinates
tool calls to a designer and critic, then requests revisions when needed. For
AutoAgent0, we use that structure as an architectural pattern rather than
copying SceneSmith directly.

Mapping:
- SceneSmith Planner -> AutoAgent0 runtime planner / orchestrator.
- SceneSmith Designer -> learned and rule-based trajectory generators.
- SceneSmith Critic -> active VLM Critic for `*_autoagent0`, future
  deterministic verifier/rule-metric critic.
- SceneSmith checkpoint/reset -> future last-safe fallback or recovery behavior.

The current system already has most of the functional pieces. The main design
change is to name them in an agentic way and expose clean boundaries:
- the Orchestrator coordinates which path to run,
- Designers generate candidate trajectories,
- legacy VLM intervention/scoring/gating selects between options for existing
  baselines,
- AutoAgent0 role-specific prompts critique and select actions for
  `*_autoagent0`,
- HUGSIM receives the selected final trajectory through the existing payload
  path.

Current SceneSmith-style tool vocabulary:
- `request_designer(action_generation)`: active. Requests candidates from
  learned and/or rule-based expert modules.
- `select_final_actions`: active. Selects the final trajectory according to the
  current method semantics.
- `request_critique`: active in `*_autoagent0` through
  `autoagent0/prompts/critic.py`. Future versions should replace or augment
  this with a deterministic verifier and rule-metric critic.
- `request_design_change(intervention)`: active as a trace/runtime transition in
  `*_autoagent0`. In this prototype it does not call a separate recovery
  endpoint generator; it triggers expanded learned + rule-based candidate
  generation.

## Current Runtime Flow

At each HUGSIM planning step, the simulator provides observations and metadata.
The active planner config determines the flow.

1. HUGSIM supplies camera streams, ego state, route/task instruction, and
   optional privileged state.
2. The active planner adapter builds learned and/or rule-based candidate
   trajectories.
3. The Orchestrator path determines whether candidates are used directly,
   merged into one pool, or separated into learned/rule-based families.
4. Legacy VLM intervention/scoring or planner-gate logic selects the final
   trajectory for existing methods. The active `*_autoagent0` path uses the
   AutoAgent0 Critic and Planner prompts instead.
5. The selected local trajectory and debug metadata are returned in the
   normalized HUGSIM plan payload.
6. `closed_loop.py` converts the trajectory into simulator control and writes
   evaluation/debug artifacts.

Method A/B and existing intervention baselines remain ablations. They should
not be treated as the final AutoAgent0 workflow. The `rap_autoagent0` and
`drivor_autoagent0` methods are the active agentic prototype.

## Design Components

### Inputs

The runtime uses camera streams, ego state, route/task instruction, candidate
trajectories, and optional privileged state. HUGSIM is the current source of
those inputs. Future simulator adapters should provide the same semantic
information without exposing HUGSIM-specific structures to the core workflow.

The front camera remains the primary visual surface for projected trajectories.
Multiview context is supporting evidence for obstacles, adjacent lanes, nearby
vehicles, and surrounding road layout.

### Orchestrator

The Orchestrator coordinates the tool loop. In the current prototype it is not
a standalone service; it is the runtime policy that decides when to critique,
when to request redesign, and when to emit the final action.

Current Orchestrator behavior for `*_autoagent0`:
- request one learned default trajectory,
- critique it,
- accept it immediately if safe enough,
- request redesign only after rejection,
- select one revised candidate from the expanded pool,
- critique the revised candidate,
- repeat selection/critique up to the configured redesign limit,
- emit the first accepted revision or the last VLM Planner-selected revision.

### Trajectory Generation

The current Designer is a boundary, not a full agent yet. It normalizes existing
candidate rows from RAP/DrivoR and Rule-Planner into a shared candidate view.
It does not create new trajectories, decide candidate counts, or dynamically
call tools.

Future Designer behavior should be explicit and targeted: the Orchestrator
should be able to request something like "generate 5 recovery trajectories that
move left and reduce speed." That requires structured rejection reasons from
the Critic/verifier first.

### Rule-Based Trajectory Generation

Learned trajectories come from RAP/DrivoR. Rule-based trajectories come from
the external Rule-Planner adapter. Their internal scores are family-local; they
are not directly comparable. Cross-family selection currently uses the VLM
Planner/scorer.

### Rule-Based Scorer

The rule-based scorer is not designed yet. Current rule-based scores only order
rule-based candidates before they enter the redesign pool. Do not add logic that
directly compares raw learned planner scores to raw rule-based scores.

The future rule-based scorer should support cross-family final selection and
verifier-style critique using map, collision, TTC, symbolic object state, and
lane/rule checks.

### Final Trajectory And States

The final action is a selected local trajectory plus debug metadata. HUGSIM
converts that trajectory into simulator controls. This should remain the only
runtime output contract until a non-HUGSIM adapter is introduced.

### Verifier Agent

The deterministic verifier agent is future work. The current `request_critique`
implementation is the AutoAgent0 VLM Critic. It checks one candidate trajectory
and returns `accepted`, `severity_score`, `corrective_action`, `confidence`,
and `reasoning`.

This is not the final verifier. It is a bootstrap Critic so the loop can be
tested before map/TTC/collision checks exist.

A future active verifier could use symbolic state and visual context for:
- out-of-road or map boundary checks,
- collision checks,
- TTC checks,
- lane/rule checks,
- structured rejection reasons that are fed back to the Orchestrator.

This future verifier should be treated as the eventual safety-grounded
SceneSmith-style Critic. The current VLM Critic is intentionally minimal and
visual-only.

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

## Active Recovery-Loop Prototype

The prototype keeps the Critic and Planner prompts minimal. A default learned
trajectory is critiqued first. Expanded learned + rule-based candidates are
requested only after rejection.

This prototype deliberately does not implement memory, deterministic map/TTC
verification, dynamic recovery endpoint generation, or a new rule-based scorer.
Method A/B remain runnable ablations and should not be treated as the final
AutoAgent0 workflow.

Implemented loop:

1. `request_designer(mode="default", k=1)` selects the learned planner's default
   candidate.
2. `request_critique(phase="default")` calls the AutoAgent0 VLM Critic on that
   one candidate.
3. If accepted, `select_final_actions` executes the default trajectory.
4. If rejected, `request_design_change` records the VLM critique reason and
   corrective action.
5. `request_designer(mode="recovery", k=10)` builds an expanded learned +
   rule-based candidate pool.
6. The AutoAgent0 Planner prompt selects one revised candidate from that pool.
7. `request_critique(phase="revised")` calls the AutoAgent0 VLM Critic on the
   revised candidate.
8. If accepted, execute the revised candidate.
9. If rejected, repeat up to `max_redesign_attempts`.
10. If all attempts are rejected, execute the last VLM Planner-selected
    revision. TODO: replace this threshold behavior after the rule-based scorer
    and deterministic verifier are designed.

The loop does not currently keep adding new trajectories after each rejected
revision. `redesign_candidate_budget` caps the expanded pool at 10 in the
current configs. `max_redesign_attempts = 3`; repeated attempts reuse the same
expanded learned + rule-based candidate pool for now.

## Future Extension Points

Next extensions:
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
