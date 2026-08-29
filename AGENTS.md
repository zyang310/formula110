# Formula 110 Agent Guide

This file applies to the repository root. Formula 110 is a deterministic,
fixed-timestep Python 3.11 racing simulator built on Panda3D Bullet and Ursina.

## Read before editing

- Product, controller, and evaluation contract: [README.md](README.md)
- Sensor types, units, signs, and sentinels: [SENSORS.md](SENSORS.md)
- Setup and first controller: [GETTING_STARTED.md](GETTING_STARTED.md)
- Grading, isolation, and export behavior: [autograder/README.md](autograder/README.md)
- Controller tuning, monitoring, and baking commands:
  [scripts/controller_training/README.md](scripts/controller_training/README.md)
- Build/tool configuration: [pyproject.toml](pyproject.toml)

Use `uv sync --managed-python`; run tools through `uv run`. Python must remain
`>=3.11,<3.12`.

## Architecture

| Area | Owner |
| --- | --- |
| Public controller boundary | `src/racing/student/api.py`; frozen sensors/commands, loader, validation, metadata |
| Controllers | `src/controllers/`; student code, helpers, inference assets |
| Track domain | `src/racing/track/`; layouts, sampled centerline, coordinates |
| Physics | `src/racing/physics/engine.py`; Bullet vehicles, actuation, impacts, damage, elimination |
| Race domain | `src/racing/race/`; sensors, progress, runtime recovery, scoring, headless orchestration |
| Application | `src/racing/game/`; configs, CLI, recording, graphical composition and loops |
| Presentation | `src/racing/graphics/`, `controls/`, `sound/` |
| Distribution | `scripts/`, `autograder/gradescope/`, `src/racing/assets/` |
| Verification | `tests/`, aligned by subsystem |

`src/racing/__init__.py` is the student-facing facade. `src/racing/main.py` is a
broader compatibility facade. Add implementations to their owning subpackage;
only add facade exports deliberately.

Panda/Ursina imports and `Any` belong at physics, graphics, sound, and app
boundaries. Keep domain/config/result APIs typed and engine-independent.

## Runtime paths

- `racing` and `python -m racing` enter `src/racing/game/cli.py`.
- Single-car and watched races are composed in `src/racing/game/app.py` by
  `build_scene()` and `build_head_to_head_viewer_scene()`.
- Headless evaluation enters
  `src/racing/race/head_to_head.py::run_headless_head_to_head()`.
- Shared spawn, progress, contact, marshal, and scoring behavior belongs in
  `src/racing/race/runtime.py`, not duplicated between graphical and headless
  loops.

Every fixed tick must retain this order: build pre-action sensors -> call the
controller -> apply its command -> step Bullet -> sample contacts/projection ->
apply damage -> update progress/runtime -> marshal. Human JSONL records use the
same pre-action observation convention.

Headless determinism depends on the fixed timestep and seeded per-race RNGs in
`race/runtime.py` and `race/head_to_head.py`. Do not introduce wall-clock or
module-global randomness into simulator paths. The simulator seed does not seed
controller libraries.

## Invariants that cross modules

- Controllers return `RobotCommand` from immutable `RobotSensors`. Stateful
  policies must use `create_controller()` or `copy_for_car()` so cars and races
  do not share state. See the controller contract in the README.
- Throttle and steer are signed `[-1, 1]`. Reversing direction brakes first;
  preserve `RobotVehicle.pending_drive_direction` in `physics/engine.py`.
- World coordinates are Y-up with driving on X/Z; heading `0` is `+Z`, and
  positive steering/bearing turns right. Reuse `track/spatial.py`.
- Node names are behavior: barriers start `track-barrier`, the LiDAR-ignored
  floor is `grass-and-track-floor`, and chassis names must be unique.
- `odometry.distance_m` is absolute travel, not race progress. Scored distance
  is best forward progress minus marshal penalties; damage/contact do not scale
  it.
- Marshal reset preserves tick and odometry continuity, resets derivative and
  contact state, and does not clear damage.
- Track width and centerline assumptions span `track/world.py`,
  `race/progress.py`, `graphics/track_mesh.py`, `graphics/track_rendering.py`,
  camera, sensors, spawn clearance, and off-track recovery. Change them
  together.
- Vehicle dimensions span collision bounds, wheel points, spawn height, grid
  clearance, and visuals. `VehiclePhysicsConfig` is the source of truth.

For detailed sensor behavior, link to `SENSORS.md`; do not duplicate it here.

## Cross-cutting changes

- Public sensor: `student/api.py` -> `race/sensors.py` ->
  `game/recording.py` -> `racing/__init__.py` -> docs/tests -> autograder review.
- Race rule/stat: `race/rules.py` or `race/runtime.py` ->
  `race/head_to_head.py` -> watched viewer -> CLI/config -> result tests.
- Physics behavior: `physics/engine.py` -> runtime/sensors -> visuals/audio when
  observable -> physics and integration tests.
- Track behavior: track domain -> progress -> render/collision geometry ->
  camera/sensors/runtime -> track, spawn, and rendering tests.
- Versioned JSON: update schema version, serializer/`to_dict()`, consumers,
  documentation, and tests; keep output valid with `allow_nan=False`.
- CLI option: parser validation -> immutable config -> app/runner -> CLI tests.

## Verification

- Targeted tests: `uv run pytest tests/test_<area>.py -q`
- Full suite: `uv run pytest -q`
- Types: `uv run pyright`
- Touched-file lint: `uv run ruff check <paths>`
- Touched-file format check: `uv run ruff format --check <paths>`
- Distribution changes: `uv build`

The full test suite does not open the GUI, run a complete 30-second race, or
execute Gradescope's Ubuntu setup. Add the corresponding manual/offscreen,
headless race, or archive check when those areas change.

Whole-tree Ruff currently has unrelated baseline findings. Do not run broad
`ruff --fix` or formatting as part of a focused change; restrict edits to the
touched paths and report remaining baseline failures.

## Autograder and generated files

- `autograder/gradescope/race_worker.py` has a separate trusted trial loop;
  review it when tick, sensor, physics, or race semantics change.
- `control_worker.py` is CPU-only, timed, and limited to 512 MiB process-tree
  RSS. Do not weaken its isolation or protocol limits.
- Gradescope installs a fixed dependency set; it does not install arbitrary
  controller dependencies from `pyproject.toml` or `uv.lock`.
- `scripts/export_student_controllers.py` follows static `controllers.*`
  imports. Dynamic imports and non-Python assets require `--all-controllers`.
- `artifacts/` is ignored output. Audio-processing scripts overwrite tracked
  WAVs by default; update `src/racing/assets/audio/THIRD_PARTY_AUDIO.md` when
  audio changes.
- Commit both `pyproject.toml` and `uv.lock` when dependencies change.
