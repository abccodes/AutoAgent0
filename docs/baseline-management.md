# Baseline Management

This repo now uses a three-root benchmark output layout:

- canonical baselines:
  - `/bigdata/aidan/outputs/benchmark/out/baselines/`
- archived legacy or experimental outputs:
  - `/bigdata/aidan/outputs/benchmark/out/archive/`
- overwrite-friendly smoke/debug runs:
  - `/bigdata/aidan/outputs/benchmark/out/debug/`

## Source of truth

Two files define the baseline system:

- `configs/baselines/registry.yaml`
  - what the system knows how to run
  - canonical `baseline_id` values
  - canonical planner config paths
  - canonical dataset and suite support
- `configs/baselines/validated_runs.yaml`
  - what outputs are currently trusted
  - which trusted outputs are canonical versus historical
  - which important baselines are still pending

## Canonical naming

Canonical outputs use:

`/bigdata/aidan/outputs/benchmark/out/baselines/<baseline_id>/<dataset>/<suite>/<run_variant>/<scene_dir>`

Examples:

- `.../baselines/drivor_intervention_4cam/waymo/3easy/qwen3-vl-8b-instruct/...`
- `.../baselines/rap_impl_b/kitti360/3easy/qwen3-vl-8b-instruct/...`
- `.../baselines/rap_vlm/nuscenes/full/rap-base-legacy/...`

## Archive conventions

The archive tree is used for:

- dated historical roots moved under `archive/date_roots/`
- small experimental one-off roots moved under `archive/legacy_experiments/`
- future duplicate or superseded outputs if needed

Archive outputs should not be used as the primary source of benchmark numbers unless they are explicitly promoted in `validated_runs.yaml`.

## Historical base-policy note

The old `05_26_26` Waymo/KITTI runs that were originally intended as intervention runs are preserved as valid historical **base-policy** baselines, because in practice:

- the intervention gate executed
- it fell back on every frame
- the scorer never ran

Those now live under canonical baseline paths as historical base-policy runs, not under archive.

## Useful commands

Print trusted baseline inventory:

```bash
/bigdata/jason/drivor_evaluation/HUGSIM/.pixi/envs/default/bin/python \
  scripts/baselines/common/print_baseline_inventory.py
```

Preview smoke-run submission without launching:

```bash
DRY_RUN=1 bash scripts/baselines/smoke/submit_dataset_3easy.sh waymo drivor_intervention_4cam debug
```

Preview full NuScenes baseline submission without launching:

```bash
DRY_RUN=1 bash scripts/baselines/full/submit_nuscenes_full_baselines.sh
```

## Policy

- New trusted baselines should go under `out/baselines/...`
- New smoke tests should go under `out/debug/...`
- Small one-off experiments should go to `out/archive/legacy_experiments/...`
- Dated buckets should not be created again
- Only delete output directories when they are clearly useless and already known invalid
