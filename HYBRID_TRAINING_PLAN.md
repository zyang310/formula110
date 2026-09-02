# Hybrid Formula 110 Training Plan

## Architecture

The runtime controller in `src/controllers/hybrid_track_policy.py` uses only legal `RobotSensors` and returns `RobotCommand`. Offline scripts may use privileged simulator state to evaluate candidates, sample track geometry, and seed start poses.

The system has four layers:

- `scripts/hybrid_training_v1/evaluate_controller_batch.py`: local Gradescope-like 30-second, 60 Hz, no-marshal evaluator.
- `scripts/hybrid_training_v1/build_track_policy_data.py`: privileged fixed-track sampler that writes compact progress-indexed data to `src/controllers/hybrid_track_policy_data.py`.
- `scripts/hybrid_training_v1/mine_maneuvers_cem.py`: smoke-safe CEM loop for short open-loop maneuver discovery.
- `scripts/hybrid_training_v1/mine_sector_trajectories.py`: privileged sector CEM loop with entry speed/lateral/heading perturbations and per-tick trace capture for distillation.
- `scripts/hybrid_training_v1/tune_hybrid_policy.py`: smoke-safe stochastic tuning for controller gains, speed profile shape, and emergency thresholds.

## Build Or Update Track Data

```bash
python scripts/hybrid_training_v1/build_track_policy_data.py --bins 128
```

## Smoke Checks

```bash
python scripts/hybrid_training_v1/evaluate_controller_batch.py \
  --module controllers.hybrid_track_policy \
  --seeds 110 2026 \
  --seconds 3

python scripts/hybrid_training_v1/mine_maneuvers_cem.py --smoke --seconds 2

python scripts/hybrid_training_v1/mine_sector_trajectories.py \
  --smoke \
  --start-progresses 0 \
  --start-speeds 20 \
  --workers 2 \
  --duration 0.5 \
  --segments 3 \
  --trace-keep 1 \
  --output artifacts/sectors/smoke_sector_trajectories.json

python scripts/hybrid_training_v1/tune_hybrid_policy.py --smoke --seconds 2 --compact-output
```

## Mine Fast Sector Trajectories

The current polished hybrid controller runs clean 13.8-14.1 second laps. A sub-5 lap on the 183.066 m track needs roughly 36.6 m/s average progress, so pure gain tuning is not enough. Use sector CEM to discover high-speed lines from realistic entry speeds, then distill the traced best candidates into progress-indexed target speed, offset, and feedforward action rows.

Compact probe:

```bash
python scripts/hybrid_training_v1/mine_sector_trajectories.py \
  --start-progresses 0 45 90 135 \
  --start-speeds 18 30 \
  --workers 4 \
  --duration 1.5 \
  --segments 8 \
  --population 18 \
  --generations 6 \
  --keep 8 \
  --trace-keep 3 \
  --damage-limit 0.98 \
  --confirm-full \
  --output artifacts/sectors/probe_4x2_1p5s.json
```

The probe exposes which areas need a different line. On the 2026-08-28 run, the best 1.5 s sectors reached 32.9-33.7 m/s average at progress 0 m and 90 m with zero damage, while progress 45 m and 135 m remained slower and often required wall contact. Those corner sectors should get denser progress starts, lateral offsets, and longer horizons before distillation.

Sector mining supports `--workers N` for process-level parallelism. It writes a checkpoint to the output JSON after each completed sector; `complete: false` means the run was interrupted or is still partial, and `complete: true` means every requested sector finished.

## Apply Tuned Parameters

Use compact tuner output to avoid the long `history` field, then apply only `best_params`:

```bash
python scripts/hybrid_training_v1/tune_hybrid_policy.py \
  --seeds 110 2026 \
  --seconds 30 \
  --trials 160 \
  --confirm-full \
  --compact-output \
  --output artifacts/policies/tune_160_speed_profile.json

python scripts/hybrid_training_v1/apply_best_policy_params.py artifacts/policies/tune_160_speed_profile.json
```

Preview the exact replacement without editing:

```bash
python scripts/hybrid_training_v1/apply_best_policy_params.py artifacts/policies/tune_160_speed_profile.json --dry-run
```

## Full Training Commands To Approve Later

These are intentionally not launched automatically.

```bash
python scripts/hybrid_training_v1/build_track_policy_data.py --bins 192

python scripts/hybrid_training_v1/evaluate_controller_batch.py \
  --module controllers.hybrid_track_policy \
  --seeds 110 2026 \
  --seconds 30 \
  --json

python scripts/hybrid_training_v1/mine_maneuvers_cem.py \
  --seed 110 \
  --seconds 6 \
  --population 64 \
  --generations 40 \
  --confirm-full

uv run python scripts/hybrid_training_v1/mine_sector_trajectories.py \
  --start-progresses 0 15 30 45 60 75 90 105 120 135 150 165 \
  --start-speeds 12 20 28 36 44 \
  --lateral-offsets -1.6 -0.8 0 0.8 1.6 \
  --heading-errors -10 0 10 \
  --workers 8 \
  --duration 2.0 \
  --segments 12 \
  --population 72 \
  --generations 36 \
  --keep 16 \
  --trace-keep 5 \
  --damage-limit 0.98 \
  --confirm-full \
  --output artifacts/sectors/full_sector_grid.json

python scripts/hybrid_training_v1/tune_hybrid_policy.py \
  --seeds 110 2026 \
  --robustness-seeds 1 7 42 73 314 911 \
  --seconds 30 \
  --trials 240 \
  --compact-output \
  --confirm-full

python scripts/hybrid_training_v1/evaluate_controller_batch.py \
  --module controllers.hybrid_track_policy \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
          20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 \
          40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 \
          60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 \
          80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 \
  --seconds 30 \
  --json
```

Estimated training time:

- First useful version: 4-8 hours.
- Stronger full maneuver mining: 12-24 hours.
