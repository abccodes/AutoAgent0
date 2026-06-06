# Rules

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

# HUGSIM Methods and Run Guide

Keep this section short. Detailed method/run documentation lives in `docs/` so
`AGENTS.md` stays focused on operating rules and navigation.

Use these references instead of duplicating details here:
- `docs/hugsim-methods-and-runs.md`: operational run commands, method families,
  dataset roots, output layout, smoke tests, full benchmarks, and archived-run
  notes.
- `docs/autoagent0-architecture.md`: AutoAgent0 package/file ownership, active
  recovery loop, debug traces, and video outputs.
- `docs/autoagent0-design.md`: AutoAgent0 design intent, current limitations,
  and next research modules.
- `docs/baseline-management.md`: canonical baseline registry, validated-run
  tracking, output conventions, migration/archive policy.
- `docs/curated-demo-tasks.md`: stop/park demo task behavior and outputs.
- `docs/shared-server-setup.md`: shared server assumptions and setup notes.

High-signal rules for future work:
- Prefer canonical launchers under `scripts/baselines/...`; root-level launchers
  are compatibility shims unless a doc says otherwise.
- Keep new shared AutoAgent0 logic under `autoagent0/`; keep RAP/DrivoR and
  Rule-Planner-specific integration in `planners/`.
- Do not move or vendor external planner repos into this repository.
- Keep Method A/B and legacy intervention methods runnable as ablations while
  developing `rap_autoagent0` and `drivor_autoagent0`.
- Before trusting videos or scores from reused output directories, check mtimes
  for `output.txt`, `eval.json`, `front.mp4`, and `video.mp4`.
- Archive stale or historical outputs only when their semantics are understood;
  delete results only when they are clearly useless and known invalid.
