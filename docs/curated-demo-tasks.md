# Curated Demo Tasks

This repo now supports an optional `task` block in scenario YAMLs for small curated demos on top of the existing NuScenes-backed HUGSIM setup.

The feature is intentionally narrow:

- it does not change the current dataset/backend layout,
- it does not create a new benchmark,
- it is meant for a few hand-picked demo scenes,
- `park here` / `stop here` targets are manual scene-local poses.

## Supported Task Types

Current task types:

- `stop_at_target`
- `park_at_target`

Both use a manually specified `target_pose` in scene-local coordinates:

- `x`: lateral offset in the local planning frame
- `y`: forward distance in the local planning frame
- `yaw`: optional heading in radians

## Scenario Schema

Example:

```yaml
task:
  type: park_at_target
  instruction: Park at the marked target.
  target_pose: [2.5, 18.0, 0.0]
  position_tolerance_m: 3.0
  heading_tolerance_deg: 35.0
```

Notes:

- If `task` is omitted, existing behavior is unchanged.
- `instruction` overrides the coarse `command` text for the VLM selector path.
- The front-camera candidate overlay renders a target marker labeled `STOP` or `PARK`.
- Current `park_at_target` demos should be interpreted as a forward pull-over / parking-like stop near the target, not a reverse parking maneuver.

## Runtime Behavior

When a task is present, simulator `info` includes:

- `task_active`
- `task_type`
- `task_instruction`
- `task_target_pose_local`
- `task_target_world`
- `task_goal_status`

`task_goal_status` currently reports:

- `position_error_m`
- `heading_error_deg`
- `reached`

These fields are additive and do not replace the existing `command`, `rc`, or collision outputs.

## Included Demo Configs

Starting-point configs:

- `configs/benchmark/nuscenes_demo/scene-0038-stop-demo.yaml`
- `configs/benchmark/nuscenes_demo/scene-0411-park-demo.yaml`

The target poses are seed values for curation, not guaranteed benchmark-grade final placements. Adjust them after visual inspection if needed.
