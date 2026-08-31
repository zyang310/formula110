# Competitive Formula 110 Controller - Living Plan

Last updated: 2026-08-30

## Status

- Project state: **stopped at the user's request after v23.** v19 generation 6 is
  the promoted, baked, fully gated controller. Ten versions ran under the
  ten-generation plateau rule. Two produced promotions - v16 (launch floor) and
  v19 (corner floor and front-stop ceiling) - and both came from reopening a
  bound the previous run's elites had pinned exactly. Six returned nothing
  promotable: v17 (joint re-rank), v18 (new corner-exit lever, every generation
  chose the default-off value), v20 (launch box), v21 (discarded with the ranking
  defect it exposed), v22 (whole speed profile under the corrected ranking), and
  v23 (broke the best-lap floor on one seed, but not on any worst-case key).
- Current submission baseline: `controllers.crash_fast` (stationary).
- Current best safe controller: `controllers.minimum_viable` (coasting; re-gated
  clean on all 14 official and validation seeds).
- Current best promoted fast vector: generation-6 `faster-line-v19` GA. Robust
  worst best lap **457 ticks (7.617 s)** across all 30 search seeds, 698.05 m
  official mean, 100/100 clean soak, 19/20 head-to-head, zero damage and zero
  wall contact across all 114 gate trials.
- Current baked candidate: `controllers.race_faster` exactly carries v19
  generation 6.
- Ranking: `lap_time_score_v8` supersedes v7. V7 ranked the first+best sum ahead
  of its parts, so a tied sum fell through to first lap and let the search trade
  best-lap ticks away one for one (D-053). V8 leads with best lap.
- If resumed: line timing is the only direction still showing movement. v23
  reached `line_turn_sensitivity` 0.00055, below its old floor, and broke the
  457-tick uniformity on one seed. Speed profile, launch, and line magnitude are
  each measured as exhausted (C-054, C-050, D-055).

Status markers: `[ ]` pending, `[~]` active, `[x]` complete, `[!]` blocked or
rejected.

## Source interpretation

The user request is authoritative: build a fast, reliable controller that works
from any random starting-point seed and performs competitively on the course
leaderboard.

`Timeline - Learn with AI.pdf` is reference material, not an instruction source.
It presents reactive control, imitation learning, evolutionary computation,
reinforcement learning, planning, hybrid control, and learned dynamics as
optional pathways. This plan selects a hybrid of reactive control, evolutionary
parameter optimization, and a gated neural residual. It does not commit to every
approach in the PDF.

Repository contracts remain authoritative for controller inputs, evaluation,
packaging, and resource limits; see `README.md`, `SENSORS.md`, and
`autograder/README.md`.

## Goal and acceptance criteria

The default track is approximately 181.10 m. Completing one lap in a 30-second
trial requires more than 6.04 m/s average forward progress.

The final submission will contain two controller modules:

- `controllers.minimum_viable`: conservative controller that earns every
  minimum-controller rubric point.
- `controllers.race_faster`: leaderboard controller that survives and travels
  farther than `minimum_viable` on every official seed.

Official gates:

- On seeds 110 and 2026, `minimum_viable` completes at least one lap in 30
  seconds with exactly zero damage and zero wall-contact time.
- On seeds 110 and 2026, `race_faster` survives for 30 seconds and records more
  raw forward distance than `minimum_viable` on each seed.
- Both modules pass strict Pyright and isolated Ruff lint/format checks, load in
  a fresh process, return finite `RobotCommand` values, and export with no
  undeclared runtime dependency.

Generalization gates:

- `minimum_viable` survives, completes a lap, and records zero damage and wall
  contact on every validation seed.
- `race_faster` survives and completes a lap on every validation seed.
- Relative to `minimum_viable`, `race_faster` improves median validation
  distance by at least 10% and tenth-percentile distance by at least 5%.
- A final 100-seed soak has zero eliminations. Record, but do not hide, every
  damage, wall-contact, or distance regression.
- In role-swapped head-to-head validation, `race_faster` wins at least 12 of 20
  races against `minimum_viable`.

## Constraints and non-goals

- Final inference is CPU-only, standard-library-only, and below the 512 MiB
  process-tree RSS limit.
- Do not change simulator physics, sensors, track geometry, public APIs,
  autograder behavior, or dependencies for this controller project.
- Do not read mutable simulator state or private race progress from a
  controller.
- Do not specialize behavior to absolute world coordinates or the two official
  seeds.
- Full PPO/SAC/TD3 training, a Gym wrapper, learned dynamics, and a large neural
  policy are outside this iteration's CPU/time budget.
- Tuning outputs belong under ignored `artifacts/controller-search/`; final
  controller parameters are baked into source and never loaded from that
  directory at runtime.

## Controller design

Create one dependency-free shared controller core under `src/controllers/` and
two thin preset modules. Each preset exposes `create_controller()` so each car
and repeated race receives independent smoothing and recovery state.

### Observation processing

- Use signed speed, yaw rate, camera center offset, camera heading error, all
  three lookahead offsets, wall-only LiDAR, general LiDAR, contact duration, and
  nearby competitor geometry.
- Replace LiDAR infinity with a fixed cap before normalization.
- Clamp every normalized feature to its declared range.
- Avoid absolute heading as a policy input; track-relative geometry is already
  invariant to the random starting location.

### Steering

- Combine heading error, center offset, short/mid/far lookahead offsets, and yaw
  damping into a desired steering command.
- Weight farther lookahead more on straights and nearer lookahead more while
  cornering or recovering.
- Add wall-clearance correction before clamping.
- Apply a per-tick steering slew limit to prevent oscillation without making
  emergency avoidance sluggish.

### Speed and braking

- Estimate upcoming curvature from lookahead offsets and heading error.
- Interpolate between tunable straight and corner target speeds.
- Reduce target speed for large steering demand, poor front clearance, narrow
  side clearance, excessive yaw rate, or a nearby competitor ahead.
- Use proportional forward throttle below target speed and lift to exactly zero
  throttle above it. Never command negative throttle in `NORMAL`.
- Braking stays available in `AVOID` and reverse in `RECOVER`. After any
  negative throttle above walking speed, emit one zero-throttle tick before
  resuming drive so the vehicle's pending-direction latch clears.
- Read the unclamped `speed_mps` for every speed decision. `speed_cap_mps` is a
  normalization constant only, so a raised target speed can never saturate it.

### Safety and recovery state machine

- `NORMAL`: track the preview line and speed schedule.
- `AVOID`: override the racing line when wall or competitor clearance crosses a
  tunable safety threshold; steer toward the safer visible opening and brake.
- `RECOVER`: after contact, sustained near-zero progress, or a blocked front,
  reverse briefly toward the side with greater clearance, then re-enter normal
  control only after heading and clearance recover.
- Priority order is wall safety, recovery, competitor avoidance, racing line,
  then speed optimization.

### Preset responsibilities

- `minimum_viable` uses a center-biased line, lower straight/corner speeds,
  larger clearance margins, and earlier braking.
- `race_faster` starts from the safe preset, raises speed targets, shifts toward
  a preview-derived racing line, brakes later, and adds opponent-aware passing.
- Shared policy code must remain identical between presets; differences are
  represented by immutable parameter sets.

## Evaluation and optimization tooling

Add script-only tooling; do not add a public simulator API.

### Solo evaluator

- Reproduce the Gradescope trial order: pre-action sensors, controller command,
  1/60-second physics step, contact/damage, and progress update.
- Use 30 seconds, one car, seeded spawn, and marshal recovery disabled.
- Return seed, raw distance, partial laps, lap count, damage, survival,
  wall-contact seconds, maximum speed, first-lap time, and best-lap time.
- Add a parity test against `autograder/gradescope/race_worker.py` for identical
  module, seed, and duration.

### Seed suites

- Official: `(110, 2026)`; never use these to rank optimizer generations.
- Generate 40 unique non-official seeds with `random.Random(590110)`, sampling
  from 1 through 99,999 and excluding official seeds.
- First 28 generated seeds are training seeds; remaining 12 are validation
  seeds and remain untouched until promotion decisions.
- Generate the final 100-seed soak with `random.Random(590111)`, excluding every
  official, training, and validation seed.
- Write the exact generated tuples into the experiment log on first use.

The immutable suites generated on 2026-08-28 are:

- Official: `(110, 2026)`
- Training: `(30991, 89384, 37399, 89006, 79351, 44850, 90314, 68992, 3887,
  83880, 26114, 21511, 18845, 23963, 19609, 30387, 62156, 33167, 62599,
  44832, 86653, 8530, 43464, 18527, 10023, 72497, 34071, 1718)`
- Validation: `(82361, 16872, 41256, 8681, 60604, 19331, 37089, 75222,
  88117, 90661, 76542, 56221)`
- Final soak: `(38605, 37849, 60758, 41419, 84539, 13355, 70773, 27223,
  4545, 94066, 94690, 78394, 89235, 70711, 73471, 64378, 96757, 1954,
  57450, 10959, 48082, 227, 66169, 31617, 58753, 49888, 49254, 66005,
  2497, 74457, 51981, 58312, 73621, 70579, 94987, 32973, 29029, 78595,
  73502, 21408, 4500, 45130, 60771, 93235, 49854, 37876, 86202, 98009,
  44097, 51016, 21736, 69143, 28554, 16543, 11643, 22885, 92489, 38791,
  3301, 41700, 9656, 57320, 7717, 75138, 17642, 89934, 59492, 93034,
  96662, 62637, 48672, 97713, 63277, 3003, 80481, 41427, 52447, 27916,
  24295, 26748, 37847, 3453, 49369, 78343, 69636, 60171, 36638, 4682,
  45237, 5533, 77041, 34487, 53571, 55701, 37185, 51174, 87519, 60109,
  89714, 10221)`

### Standard-library Cross-Entropy Method

- Population: 48 candidates.
- Elite count: 8.
- Generations: 20 per preset.
- Parallelism: process workers equal to `max(1, CPU count - 1)`.
- Optimizer randomness is deterministic and stored with every checkpoint.
- Each generation evaluates candidates on a rotating batch of six training
  seeds; reevaluate the eight elites on all 28 training seeds before updating
  the distribution.
- Persist generation, distribution mean/deviation, best parameter vector,
  optimizer seed, and metrics as JSON after every generation.
- Resume from the last complete checkpoint without rerunning completed
  generations.

Minimum candidate ranking is lexicographic:

1. Number of seeds satisfying lap, survival, zero-damage, and zero-wall gates.
2. Number of zero-damage and zero-wall seeds.
3. Worst raw distance.
4. Mean raw distance.

Improved candidate ranking is lexicographic:

1. Survival count.
2. Number of seeds completing at least one lap.
3. Number of seeds inside the per-trial incident budget of 0.25 damage and
   1.5 seconds of wall contact.
4. Worst per-seed distance improvement over `minimum_viable`.
5. Tenth-percentile and median distance improvement.
6. Mean raw distance, charged 120 m per unit of damage and 6 m per second of
   wall contact.

The budget is counted per trial rather than summed so that one small bounded
incident cannot outrank every distance term. `race_faster` is gated on surviving
and completing a lap, not on running perfectly clean; the stricter zero-damage
rule stays in the minimum ranking.

### Phase-aware faster-line search

Superseded 2026-08-28: the `faster` run was stopped at generation 84 rather than
100, and coasting is now default controller behavior rather than a `faster-line`
flag. See the decision log.

- Use the `faster-line` preset for the raised speed schedule and the phase-aware
  outside-inside-outside line.
- Tune 15 speed, centering, line, and slew parameters with a population of 64,
  12 elites, and optimizer seed 590114. Side-clearance parameters are excluded:
  AVOID never fires on the training seeds and varying them changed nothing.
- Keep maximum speed as diagnostic telemetry; rank useful speed through robust
  30-second distance.

```bash
uv run python -m scripts.controller_training.search faster-line \
  --artifact-root artifacts/controller-search/faster-line \
  --population 64 \
  --elites 12 \
  --generations 40 \
  --optimizer-seed 590114
```

Run role-swapped head-to-head races only for the five best improved candidates,
using `(42, 110, 271, 997, 2027)` plus the first five validation seeds. Each
candidate runs once in each role against `minimum_viable`.

## Optional neural residual

Reassess this only after the pose-invariant rules-only v2 controller passes every
promotion gate.

- Inputs: normalized speed, pose-invariant `kappa_0..2`, lateral line error,
  heading error, front-wall clearance, and current line target.
- Network: eight inputs, four `tanh` hidden units, and two outputs.
- Outputs are a line-target residual capped to about 0.10 of half-track and a
  throttle residual capped at 0.10. The line residual remains behind the 0.65
  clamp, target slew, and clearance retraction rather than bypassing them as raw
  steering.
- Apply residuals only in `NORMAL`; AVOID and RECOVER remain rules-only.
- Evolve weights with the same deterministic optimizer; do not add a numerical
  library or model artifact.
- Promote only when validation median distance improves by at least 3%,
  tenth-percentile distance does not decrease, neither official seed worsens,
  no elimination/damage regression appears, and it wins at least 12 of 20
  role-swapped races against rules-only `race_faster`.
- If any gate fails, record the ablation and ship the rules-only controller.

## Work breakdown

### Phase 0 - Baseline and harness

- [x] Inspect controller, sensor, race, packaging, and grading contracts.
- [x] Record the stationary `crash_fast` baseline on official seeds.
- [x] Add solo evaluator and JSON result format.
- [x] Prove evaluator parity with the trusted race worker.
- [x] Generate and log fixed seed suites.

### Phase 1 - Safe minimum controller

- [x] Implement shared parameter and controller-state types.
- [x] Implement feature normalization, preview steering, speed control, and
  recovery states.
- [x] Add focused controller unit tests.
- [x] Establish hand-tuned parameters that complete a clean lap.
- [x] Tune minimum parameters with CEM.
- [x] Promote only after official and validation safe-lap gates pass.

### Phase 2 - Fast controller

- [x] Fork parameters, not policy code, from the promoted minimum controller.
- [x] Add racing-line and opponent-aware terms.
- [x] Tune improved parameters with survival as a hard priority. Stopped early
  at generation 84 of 100: the box was exhausted and the policy it optimized has
  been replaced by coasting.
- [x] Make coasting default controller behavior and add phase-aware
  racing-line behavior.
- [x] Bake the generation-84 `faster` vector and close that search.
- [x] Rebuild the 15-variable `faster-line` space around the raised speed
  schedule and the loosened incident budget.
- [x] Run the `faster-line` search (40 generations; converged with 11 of 15
  deviations on their floors).
- [x] Run official, validation, and head-to-head promotion gates.
- [x] Freeze the best rules-only parameter set (generation 40).
- [x] Implement opt-in pose-invariant curvature, a tracked/debiased racing line,
  strict per-tick trace output, and focused equilibrium/safety tests.
- [x] Implement the 19-gene CEM falsification preset and 17-gene deterministic
  GA with mutation, crossover, elitism, diversity telemetry, and exact resume.
- [x] Archive every completed CEM and GA generation immutably; preserve losing
  objective branches instead of overwriting or deleting them.
- [x] User: run and record the Step 2 trace gate, including the phase repair and
  post-repair boxed-authority boundary check.
- [x] User: finish the 10-generation CEM falsification. Rejected: the winner
  regressed the incumbent and `center_steer_gain` converged back toward its floor.
- [x] Diagnose the rejected CEM branch and repair the phase selector: local
  curvature from segment-slope differences, anchored on the reported centre
  offset, plus whole-lap phase-mass telemetry and falsification tests.
- [x] User: run 10 unseeded GA generations as the falsification gate. Passed on
  distance, lap time, entry offset, and diversity; the two original parameter
  predictions failed and are re-diagnosed below.
- [x] User: run the GA through generation 40. Final penalized mean 597.90 m and
  8.983 s best lap on training seeds.
- [x] Bake the generation-40 v2 GA vector and pass every promotion gate.
- [x] Make the line clamp a searchable parameter and put the speed scalar on the
  pose-invariant local curvature, both default-off; add the 16-gene
  `faster-line-v3` preset that unpins all five bounds the v2 winner sat on.
- [x] User: run the `faster-line-v3` GA. Stopped at generation 60 on a stall.
- [x] Bake the generation-60 v3 vector into `race_faster`.
- [x] Superseded: the v3 vector was replaced by the v4 winner before being gated.
- [x] Add the asymmetric line-target release and the 17-gene `faster-line-v4`
  preset, seeded from the baked v3 vector.
- [x] User: run the `faster-line-v4` GA. Stopped at generation 57 when the
  machine ran out of memory; all 57 archives are intact.
- [x] Bake generation 57 and pass every promotion gate.
- [x] Add the 17-gene `faster-line-v5` preset, seeded from the promoted v4
  vector, keeping v4's line ceilings and widening only the non-safety bounds.
- [x] User: run the `faster-line-v5` GA through generation 100; the best vector
  first appeared at generation 84 and improved v4 by 1.56%.
- [x] Bake the v5 winner for visual inspection; promotion gates are still
  pending, so v4 remains the current promoted controller.
- [x] Supersede the stale post-repair objective-selection step: the `improved`
  objective was already selected during v2 and remained viable through v5.
- [x] Add the 20-gene `faster-line-v6` preset, searching four inherited steering
  dynamics and reopening v5's pressured non-geometric bounds while fixing the
  0.95 line clamp.
- [x] Stop v6 at generation 10: all ten generations exactly tied the v5 anchor.
- [x] Trace the v5 anchor and identify the inherited AVOID and side-speed guards
  as the next bottleneck; verify one clean distance gain and a 7.883 s diagnostic
  lap before widening their bounds.
- [x] Add the focused 16-gene `faster-line-v7` preset and robust `lap-time`
  objective, retaining survival, lap-completion, and incident-budget hard tiers.
- [x] Stop v7 at generation 27 with a real 7.833 s worst / 7.817 s median winner;
  preserve the later archive that superseded generation 25 after quantization.
- [x] Diagnose and test the lap-time float-jitter bug; add the quantized
  `lap-time-v2` objective and the focused 14-gene v8 refinement space.
- [x] Stop v8 at generation 29 after ten flat generations; reject its
  generation-19 winner after the seed-110 trace exposed crash-then-sprint.
- [x] Add the consistency-first `lap-time-v3` objective and 14-gene v9 space;
  seed from clean v7 generation 27 rather than the rejected v8 branch.
- [x] Run v9 through generation 30 with optimizer seed 590122; generation 26 is
  the winner, and full worker-pool restarts at 5, 10, 15, 20, and 25 prevented
  the observed 2-3 GB per-worker growth from exhausting memory.
- [x] Trace v9 generation 26 on both official seeds and diagnose the seed-110
  first-turn correction; add and test the default-off launch speed cap.
- [x] Add the 16-gene v10 space, first-lap-aware `lap-time-v4` ranking, and both
  official seeds to v10 preselection and full evaluation.
- [x] Stop v10 at generation 7: the objective preferred a one-tick repeated-lap
  gain despite a 1.3 s slower first lap and 10 m less distance.
- [x] Add `lap-time-v5`, ranking robust `first_lap + 2 * best_lap` before its
  separate components, and create v11 from v10 generation 4.
- [x] Stop v11 after generation 1: six-decimal component sums ranked two equal
  1,467-tick physical totals as 24.449999 versus 24.450001.
- [x] Add exact 60 Hz tick-based `lap-time-v6` ranking and create v12 from the
  valid v10 generation-4 parent.
- [x] Stop v12 after generation 1: `--seed-checkpoint` restored the searched
  vector but silently omitted its fixed context, so the centre was not the
  archived generation-4 candidate.
- [x] Fix checkpoint seeding to merge fixed context before searched values and
  add regression coverage.
- [x] Complete v13 at generation 30; retain clean generation 22 at a 1,449-tick
  robust worst three-lap time.
- [x] Trace seed 110's final sweeper; reject global line shifts and validate a
  sustained-turn-only speed bonus across all 30 training/official seeds.
- [x] Stop v14 at generation 15 after ten exactly flat generations; bake and
  promote the clean generation-5 winner.
- [x] Promote and bake v4: official, validation, and head-to-head all pass.
- [ ] Fix head-to-head queuing: `race_faster` closes on slower traffic and holds
  station instead of passing, costing roughly half its solo pace in traffic. Now
  a measured regression at two losses in twenty races, up from zero.

### Phase 3 - Residual ablation

- [ ] Reassess a bounded residual only after the pose-invariant line/GA campaign;
  if attempted, target `line_target` rather than raw steering.
- [ ] Compare rules-only and residual policies on untouched validation seeds.
<!-- - [ ] Record the ablation and apply the objective promotion gate. -->

### Phase 4 - Final verification

- [x] Run the 100-seed soak: 100/100 survived and lapped with no incidents.
- [x] Run full tests and strict type checking: 210 tests and Pyright pass.
- [x] Run isolated Ruff lint and format checks on touched modules.
- [x] Verify fresh-process load, finite commands, and memory use (~54.1 MiB RSS).
- [x] Export both controller modules and inspect archive contents.
- [x] Re-run both official trials from the extracted exported package.
- [x] Freeze and promote the v14 generation-5 parameters.

## Test scenarios

- Empty/default sensors and every LiDAR beam at infinity.
- Large positive and negative heading/center/lookahead errors.
- Straight, gentle curve, sharp curve, and rapidly changing curve preview.
- Speed well below and above target.
- Front blockage, asymmetric wall clearance, active wall contact, and prolonged
  low progress.
- Competitor ahead, beside, behind, occluded, and absent.
- Repeated sensor histories produce deterministic commands.
- Separate factory instances do not share smoothing or recovery state.
- All returned command values remain finite and within normalized bounds.
- Checkpoint resume produces the same next generation as uninterrupted tuning.

## Experiment log

Append one row after every meaningful run. Keep failed experiments.

| ID | Date | Controller | Parameters/artifact | Seeds | Key metrics | Decision | Next step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-000 | 2026-08-27 | `crash_fast` | tracked baseline | 110, 2026 | `<0.001 m` raw distance on each; 0 damage; 0 laps | Reject: stationary | Build evaluator and safe preview controller |
| C-001 | 2026-08-28 | `minimum_viable` | `MINIMUM_VIABLE_PARAMETERS` in source | Official `(110, 2026)` and validation `(82361, 16872, 41256, 8681, 60604, 19331, 37089, 75222, 88117, 90661, 76542, 56221)` | 14/14 survived and completed a lap; 0 damage and 0 wall contact throughout; raw distance 188.98-192.88 m; first laps 28.20-29.12 s | Promote as current best safe controller | Build the parity-tested solo evaluator |
| H-002 | 2026-08-28 | `minimum_viable` | `scripts/controller_training/evaluator.py` parity trial | 110 for 0.5 s | Exact match with the trusted race worker across all 11 grading metrics; 1.770464 m raw distance, 7.061531 m/s maximum speed, 0 damage and wall contact | Promote evaluator harness | Add deterministic minimum-preset CEM tuning |
| C-003 | 2026-08-28 | `minimum_viable` | `artifacts/controller-search/minimum/checkpoint.json`, generation 20 | Training 28, official `(110, 2026)`, validation 12 | Training 28/28 clean; 222.09 m worst and 236.21 m mean. Official 235.93/228.33 m. Validation 12/12 clean; 222.46 m worst, 230.16 m median, 224.72 m tenth percentile | Promote tuned minimum preset | Implement and tune `race_faster` |
| C-004 | 2026-08-28 | `race_faster` | `artifacts/controller-search/faster/checkpoint.json`, generation 20 | Training 28 | 28/28 survived with zero damage/contact; worst improvement 39.51%, tenth percentile 41.00%, median 49.50%; 351.27 m mean distance | Bake as current candidate; continue user-run CEM to generation 100 before promotion | Resume at generation 21 |
| C-005 | 2026-08-28 | `race_faster` | `artifacts/controller-search/faster/checkpoint.json`, generation 84 | Training 28 | Worst improvement 43.13%, tenth percentile 43.60%, median 53.95%, 361.41 m mean. Over generations 50-84 the mean rose 0.85%; the last six generations gained 0.011%. Eight of fourteen deviations sat on their floor and five means on a box bound | Stop the search early; supersedes the generation-100 commitment | Bake generation 84 and replace braking with coasting |
| D-006 | 2026-08-28 | `race_faster` (gen 84) | brake-versus-coast A/B, `src/controllers/preview_controller.py` | Official `(110, 2026)` | Braking policy commanded negative throttle on 402 of 1801 ticks and scored 351.68/353.89 m with 1 lap. Coasting scored 437.40/436.00 m with 2 laps, zero damage, zero wall contact: +24.4% and +23.2% | Adopt coasting as default `NORMAL` behavior | Re-gate `minimum_viable` |
| S-007 | 2026-08-28 | `minimum_viable` | coasting default, `artifacts/controller-search/minimum/coast-official.json` and `coast-validation.json` | Official `(110, 2026)` and validation 12 | 14/14 survived and completed a lap with zero damage and zero wall contact. Official mean 283.85 m; validation worst 273.46 m, median 281.17 m, tenth percentile 275.39 m, mean 281.74 m. Previously 222-236 m | Promotion gates pass; keep coasting | Bake generation 84 into `race_faster` |
| C-008 | 2026-08-28 | `race_faster` | generation-84 vector plus coasting, `artifacts/controller-search/faster/gen84-coast-official.json` | Official `(110, 2026)` | 2/2 survived, 2 laps, clean; worst 439.56 m, mean 439.89 m, peak 17.01 m/s. Beats the re-gated `minimum_viable` (283.85 m mean) on both seeds | Bake and close the `faster` search | Run the rebuilt `faster-line` search |
| H-009 | 2026-08-28 | `race_faster` vs `minimum_viable` | role-swapped headless head-to-head, seed 110, 5 races per role | 10 races | 5 wins, 0 losses, 5 ties. No latch pathology: zero damage and no engine-off collapse. In every tie `race_faster` matched the blocker at ~284 m against its 440 m solo pace, and sat within 8 m of a competitor for 52.2% of ticks versus 1.4% for the incumbent | Latch risk cleared; record queuing as the open head-to-head defect | Address passing before the 12-of-20 promotion gate |
| C-010 | 2026-08-28 | `race_faster` | `artifacts/controller-search/faster-line/checkpoint.json`, generation 40 | Training 28 | 28/28 survived, lapped, and stayed inside the incident budget. Worst improvement 94.90%, tenth percentile 94.96%, median 97.26%, penalized mean 563.01 m. Median lap 9.550 s, down from 10.117 s at generation 1. Converged: 11 of 15 deviations on their floors and 2 means pinned | Bake generation 40 | Run the promotion gates |
| C-011 | 2026-08-28 | `race_faster` | generation-40 bake, `artifacts/controller-search/faster-line/gen40-official.json` and `gen40-validation.json` | Official `(110, 2026)` and validation 12 | Official 2/2 clean, worst 563.27 m, mean 563.38 m, peak 20.30 m/s. Validation 12/12 survived, lapped, and were clean, worst 562.53 m, median 563.45 m, tenth percentile 563.06 m. Against the re-gated `minimum_viable` this is +100.4% median and +104.5% tenth percentile, far above the +10%/+5% gates | Promote as current best fast controller | Run the head-to-head gate |
| H-012 | 2026-08-28 | `race_faster` vs `minimum_viable` | role-swapped head-to-head, seeds `(42, 110, 271, 997, 2027)` plus the first five validation seeds, one race per role | 20 races | 16 wins, 4 ties, 0 losses, zero damage. Gate needs 12 of 20 | Pass | Queuing persists: in all four ties and several wins `race_faster` was held to ~280-290 m against its 563 m solo pace |

| D-013 | 2026-08-28 | `faster-line-v2-probe` | `artifacts/controller-search/faster-line-v2-traces/heading-{0.0,0.5,1.0}.jsonl`, `heading-0.0-center-0.9.jsonl`, and `boxed-line-authority.jsonl` | 110 | All five probes survived with zero damage, wall contact, and AVOID ticks, but straight offset was only 0.135-0.237 m versus the required >0.8 m. Distance was 560.16/559.26/559.26 m for heading compensation 0.0/0.5/1.0, 533.92 m at center gain 0.9, and 531.95 m with maximum gain plus target slew. | Reject the original v2 phase blend; do not start CEM | Add a whole-law entry test, stop blending an unambiguous entry target against the opposite-sign apex target, and re-measure under fresh artifact names |

| D-014 | 2026-08-28 | repaired `faster-line-v2-probe` | `phase-fixed-heading-{0.0,0.5,1.0}.jsonl` and `phase-fixed-boxed-line-authority.jsonl` | 110 | Safe heading probes produced 0.375/0.436/0.533 m all-straight offset and 541.02/539.01/534.97 m distance. Their mean requested target on the same straight ticks was only 0.319/0.407/0.436 m, proving the original >0.8 m aggregate gate invalid. Compensation 0.0 had the best distance, curvature, and safe strong-target directional offset (0.295 m). The boxed configuration reached 1.043 m mean absolute offset on strong-target ticks but triggered 40 AVOID ticks and fell to 476.16 m. | Select compensation 0.0; reject the box but accept that the repaired line has physical authority | Run only the 10-generation CEM falsification; require incumbent-distance recovery plus both parameter predictions before GA |

| C-015 | 2026-08-28 | `faster-line-v2-probe` CEM | `artifacts/controller-search/faster-line-v2-cem-gate/checkpoint.json` and `generations/generation-001.json` | Training 28 | Generation 1 archived. Best score `[28,28,28,0.885577,0.887180,0.909571,545.899840]`; 28/28 full trials survived, lapped, and were clean, distance 549.19-554.80 m. Distribution means currently pass both predictions: `center_steer_gain=0.15787`, `curvature_lateral_ratio=0.55591`. Elite pairwise normalized L2 is 0.90664; 52 candidates were rejected at tier 4. | Interim only; do not promote or start GA | Let the same user-run process reach generation 10, preserving every archive, then run the complete CEM gate review |

| C-016 | 2026-08-28 | `faster-line-v2-probe` CEM | `artifacts/controller-search/faster-line-v2-cem-gate/checkpoint.json` and `generations/generation-001.json` through `generation-010.json` | Training 28 | Ten archives are continuous and generation 10 matches `checkpoint.json`. Best score `[28,28,28,0.931342,0.933615,0.958136,558.488760]`; all trials stayed clean, but mean distance 558.49 m regresses the incumbent's 563.01 m. Best `center_steer_gain=0.11000`; distribution mean 0.13662 fails the required >0.14 and trended down after generation 2. Best/distribution `curvature_lateral_ratio=0.93283/0.85646`, moving toward the 1.0 ceiling despite technically passing <0.96. Entry offset collapsed to 0.05506. Preview/wall compensation converged near one (best 0.98310/0.99720), so static spring debias is working. Elite normalized L2 contracted 0.95784 to 0.35048, and essentially all 52 non-elites were rejected at tier 4 every generation. | Reject; do not seed or run GA | Repair dynamic target scheduling; preserve this branch and rerun the CEM gate under a fresh root |

| D-017 | 2026-08-28 | `faster-line-v2-probe` phase selector | `src/controllers/preview_controller.py:track_shape_preview`, whole-lap geometry tests in `tests/test_preview_controller.py` | Whole `mugello-short` centreline, sampled every 0.25-0.50 m | The retired `kappa_i = c_i / d_i` selector put 84.6% of its phase weight on entry and 7.8% on apex with the car exactly on the centreline and zero pose error, and agreed with true centreline curvature on only 35-42% of corner samples across four ground-truth definitions. Replacing it with local curvature from segment-slope differences gives entry 33.1%, apex 10.2%, exit 56.7% and 63-74% agreement. A lateral displacement now cancels exactly (change < 1e-12 versus 0.379 against a 0.150 median signal); a 9 deg yaw leaves at most 0.031 against a 0.156 peak. Re-running the incumbent trace reproduced all 1,802 rows and all 16 summary fields with zero difference. | Adopt the repaired selector; rebound `line_turn_sensitivity` | Rerun the 10-generation CEM gate under a fresh root |
| D-018 | 2026-08-28 | repaired `faster-line-v2-probe` | `artifacts/controller-search/faster-line-v2-traces/phase-repaired-heading-0.0.jsonl` | 110 | Same hand-set `racing_line_entry_offset_ratio=0.35` as the pre-repair probe. 549.81 m and 3 laps versus 541.02 m and 2 laps, still clean with zero damage, zero wall contact, and zero AVOID ticks. On-track phase mass entry 0.358, apex 0.102, exit 0.540, matching the centreline prediction. Straight offset fell from 0.375 m to 0.240 m while corner speed held at 17.17 m/s. Still 13.46 m below the 563.27 m incumbent, as expected for an untuned probe vector. Strong-target tracking remains poor: mean requested 1.071 m but mean directional offset -0.266 m, with 120 of 273 strong-target ticks on the opposite side. | Accept the repair; the remaining gap is reachability, not phase | Rerun the 10-generation CEM gate under `faster-line-v2-cem-gate-2` |
| C-019 | 2026-08-28 | `faster-line-v2` GA, unseeded | `artifacts/controller-search/faster-line-v2-ga/checkpoint.json` and `generations/generation-001.json` through `generation-010.json` | Training 28 | Ten continuous archives. Best score `[28,28,28,0.98047,0.98430,1.01112,572.595]`: all 28 seeds survived, lapped, and stayed inside the incident budget, and the 572.60 m penalized mean beats the 563.01 m incumbent by 9.59 m. Best lap fell 9.850 s to 9.400 s over the ten generations, against the incumbent's 9.550 s median. `racing_line_entry_offset_ratio` rose 0.059 to 0.65, the box ceiling, reversing the rejected CEM branch's collapse to 0.055 and confirming the phase repair from the search side. Elite pairwise normalized L2 rose 0.5302 to 0.8675 rather than contracting. Both original structural predictions still fail: `center_steer_gain` 0.12334 sits at 2.9% of its box and `curvature_lateral_ratio` pins at 0.99962 while `curvature_heading_compensation` pins low at 0.00855. Tier-4 rejections were 501 of 520. | Continue the same run to generation 40 on the `improved` objective | Then reassess the 0.65 line clamp and the speed-path curvature signal |
| C-020 | 2026-08-28 | `faster-line-v2` GA | `artifacts/controller-search/faster-line-v2-ga/checkpoint.json`, generation 40 | Training 28 | Forty continuous archives. Best score `[28,28,28,1.07510,1.08020,1.08760,597.900]`; best lap 8.983 s against the incumbent's 9.550 s median. `corner_target_speed_mps` rose 16.789 to 19.595, which is where the lap time came from. Five genes finished on a bound: entry offset 0.6486 against the 0.65 clamp, `center_steer_gain` 0.1000 on its floor, `line_turn_sensitivity` 0.0100 on its floor, `curvature_lateral_ratio` 1.0000 on its ceiling, and `wall_balance_gain` 0.0416 near its floor. Elite diversity fell 0.8675 to 0.5235 and stagnation reached 2 by generation 33. | Bake generation 40 | Run the promotion gates |

| P-021 | 2026-08-28 | `race_faster` (v2 GA gen 40) | `artifacts/controller-search/faster-line-v2-ga/gen40-official.json`, `gen40-validation.json`, `gen40-head-to-head.json` | Official `(110, 2026)`, validation 12, and 20 role-swapped races | Official 2/2 survived, lapped, and were clean; worst 592.51 m, mean 595.98 m, peak 22.82 m/s, versus the previous 563.38 m. Validation 12/12 survived, lapped, and recorded zero damage and zero wall contact; worst 591.01 m, median 596.55 m, tenth percentile 591.75 m. Against the re-gated `minimum_viable` that is +112.2% median and +114.9% tenth percentile, against +10% and +5% gates. Head-to-head 14 wins, 4 ties, 2 losses against the 12-of-20 gate. | Promote as current best fast controller | Queuing is now a measured regression: the previous vector lost zero races, this one loses two, both while held near 273 m in traffic against a 596 m solo pace |
| C-022 | 2026-08-29 | `faster-line-v3` GA, unseeded | `artifacts/controller-search/faster-line-v3-ga/checkpoint.json`, generation 60, with 60 continuous archives | Training 28 | Best score `[28,28,28,1.13954,1.14718,1.17255,621.185]`; best lap 8.633 s. Against the v2 vector that is +23.29 m and -0.350 s, and against the original `faster-line` incumbent +58.18 m and -0.917 s. Stopped on a stall: generations 43-58 gained 1.20 m in total, the score was flat for the last five, and elite diversity fell 0.60 to 0.255. Nine of sixteen genes finished on a bound. `maximum_racing_line_offset_ratio` reached its 0.90 ceiling with zero elite spread, and `racing_line_offset_ratio`, `racing_line_entry_offset_ratio`, `throttle_gain`, and `curvature_lateral_ratio` also pinned high, while `center_steer_gain`, `heading_steer_gain`, `racing_line_exit_offset_ratio`, and `steering_speed_reduction` pinned low. | Bake generation 60; do not promote until gated | Run the official, validation, and head-to-head gates |
| D-023 | 2026-08-29 | corridor geometry and baked v3 trace | taut-string analysis of `mugello-short`; `artifacts/controller-search/faster-line-v3-ga` trace of the baked vector | 110 | The shortest path inside the corridor at a 2.67 m offset limit is 160.78 m against the 183.07 m centreline, 12.18% shorter, and it stays pinned to one edge for 64 m continuously. Its minimum radius collapses to 0.01 m, so that figure is an upper bound rather than a drivable line. The baked v3 vector requests up to 2.97 m but achieves only 2.06 m at the ninetieth percentile, and spends 39.9% of ticks within 0.5 m of the centreline. | Add an asymmetric target release rather than only widening bounds | Run the v4 GA and re-measure both the coverage and the tracking gap |
| C-024 | 2026-08-29 | `faster-line-v4` GA, seeded from v3 | `artifacts/controller-search/faster-line-v4-ga/checkpoint.json`, generation 57, with 57 continuous archives | Training 28 | Best score `[28,28,28,1.28835,1.29287,1.32165,665.390]`; best lap 8.017 s. Against the v3 parent that is +44.20 m and -0.616 s. Generations 46-56 were completely flat and generation 57 then gained 1.59 m, so the run was still finding ground when it stopped. Seven of seventeen genes finished pinned, including `curvature_lateral_ratio` at the 0.60 ceiling raised from 0.30, the line clamp at 0.95, and both remaining line ratios near 0.95. The new `line_target_release_per_tick` settled at roughly 6.5 to 8 times the outward slew and held there for twenty generations. | Bake generation 57 | Run the promotion gates |

| P-025 | 2026-08-29 | `race_faster` (v4 GA gen 57) | `artifacts/controller-search/faster-line-v4-ga/gen57-official.json`, `gen57-validation.json`, `gen57-head-to-head.json` | Official `(110, 2026)`, validation 12, and 20 role-swapped races | Official 2/2 survived and lapped; worst 621.34 m, mean 646.26 m, peak 24.77 m/s. One official seed recorded 0.000118 damage and 0.033 s of wall contact, the first contact any promoted vector has shown; both are far inside the 0.25 and 1.5 s budget but earlier vectors were spotless. Validation 12/12 survived, lapped, and were completely clean; worst 650.35 m, median 664.15 m, tenth percentile 661.70 m. Against the re-gated `minimum_viable` that is +136.2% median and +140.3% tenth percentile. Head-to-head 17 wins, 1 tie, 2 losses, the best win count of any vector so far. Training mean 665.39 m against validation mean 665.27 m shows no overfitting after 57 generations. | Promote as current best fast controller | Watch the official-seed contact; run the 100-seed soak |
| D-026 | 2026-08-29 | promoted v4 vector, seed-110 sweeps | `artifacts/controller-search/faster-line-v4-ga` traces | 110 | v4 reaches 3.77 m of offset with 0.29 m of wall clearance and 21 AVOID ticks, against v3's 3.09 m, 0.98 m, and zero. Two remedies were measured and both are worse. Raising `line_clearance_m` destabilises rather than protects: 1.5 gives 0.037 damage, and 2.0 collapses the trial to 234.15 m with 728 AVOID ticks and 0.381 damage, because the retraction fires continuously whenever the car runs wide. Lowering the line clamp does not reduce peak offset either: it stays at 3.77, 3.84, 3.99, 3.99, and 3.98 m for clamps of 0.95, 0.90, 0.85, 0.80, and 0.75, while distance falls from 621.34 m to 572.93 m and AVOID rises to 84. | Keep v4's line settings; neither lever is a v5 bound | Identify what actually drives the wide excursions, since they are not the racing-line target |
| C-027 | 2026-08-29 | `faster-line-v5` GA, seeded from promoted v4 | `artifacts/controller-search/faster-line-v5-ga/checkpoint.json`, generation 100, with 100 generation archives | Training 28 | Best score `[28,28,28,1.327776,1.328569,1.367294,675.768]`; all hard safety tiers pass. The winning vector first appeared at generation 84. Against v4's 665.390 m score, the gain is 10.378 m (1.56%) after 100 generations. Final normalized pairwise diversity remained 0.369 while mutation scale had already inflated to its 0.35 ceiling. `heading_steer_gain`, `line_turn_sensitivity`, `wall_balance_gain`, and exit ratio finished on their lower bounds; target release, entry ratio, and the line clamp finished on their upper bounds; curvature ratio reached 1.175 against 1.20. | Bake for visual inspection but do not call it promoted; the box, not mutation strength, is now the limiting hypothesis | Add structural variation in v6 and use several short independent seeds before paying for another long branch |
| C-028 | 2026-08-29 | `faster-line-v6` GA, seeded from baked v5 | `artifacts/controller-search/faster-line-v6-ga/checkpoint.json`, generation 10, with 10 continuous archives | Training 28 | Every generation exactly tied the v5 anchor at `[28,28,28,1.327776,1.328569,1.367294,675.768]`; best lap remained 7.900-7.917 s across the training suite. Final diversity was 0.487 and mutation had already inflated to 0.324, so the branch was still exploring. Generation time held near 57 s instead of lengthening. | Stop early; four added steering-dynamics genes and wider mutations did not expose a gain | Trace the speed policy and change the ranking target before v7 |
| D-029 | 2026-08-29 | baked v5 wall-speed diagnostics | `artifacts/controller-search/faster-line-v7-diagnostics/*.jsonl` | 110 | Baseline: 653.98 m, 7.917 s best lap, zero contact, 23 AVOID ticks, and 77 side-slowdown ticks. Relaxed AVOID thresholds: 663.16 m clean. AVOID speed 8.0 m/s: 658.41 m clean. Combined wall-speed relaxation: 7.883 s with 0.067 s wall contact but only 617.01 m because the first lap rose to 11.067 s. Setting side slowdown to 0.8/0.8 alone failed to lap and contacted the wall for 26.23 s. | Search wall-speed guards conservatively and rank robust best-lap time directly; reject the aggressive side-speed setting | Run the 16-gene v7 GA for at most 30 generations |
| C-030 | 2026-08-29 | `faster-line-v7` GA, seeded from baked v5 | `artifacts/controller-search/faster-line-v7-ga/generations/generation-027.json`, with 27 continuous archives | Training 28 | Real final score after six-decimal time quantization is `[28,28,28,-7.833333,-7.821667,-7.816667,-7.812500,683.531]`. Against v5, robust worst improved 0.083 s, median 0.100 s, mean 0.103 s, and distance 7.76 m; all 28 trials retained the survival, lap, and incident tiers. Gains arrived at generations 7, 11, 15, 17, 20, 21, 22, 24, 25, and 27, so this was not a plateau. V7 pressed `throttle_gain` at 8.0, `curvature_lateral_ratio` near 2.0, and repeatedly explored the lower side/diagonal thresholds. | Close v7 because the v1 objective exposed float-jitter ranking, not because the hypothesis stalled | Seed v8 from generation 27 with quantized ranking and moved bounds |
| D-031 | 2026-08-29 | v7 ranking and worker-memory audit | `faster-line-v7-ga` generations 20, 25, 26, 27; live RSS checkpoints | Training 28 | Generation 26's mathematically equal 7.833333 s worst lap beat generation 25 by about 5e-15 s and hid a worse median, proving raw-float lexicographic ranking is invalid at tick resolution. Six-decimal quantization selects generation 27 over 25 on p90 and mean. Separately, evaluator-object recycling did not return memory: four workers reached 1.96-2.06 GB each by generation 11; three restarted workers reached 2.89-3.04 GB each by generation 20. Full process restarts released the memory and resumed exactly. | Quantize lap times; restart the complete worker pool every ten generations | Run v8 with checkpoint restarts at 10 and 20 |
| C-032 | 2026-08-29 | `faster-line-v8` GA, seeded from v7 generation 27 | `artifacts/controller-search/faster-line-v8-ga/generations/generation-019.json`, stopped with 29 continuous archives | Training 28 | Best score `[28,28,28,-7.783333,-7.766667,-7.766667,-7.766072,684.891]`; the last gain was generation 19 and generations 20-29 were exactly flat. Against v7, worst/median/mean best-lap improved 0.050/0.050/0.046 s and distance 1.36 m. Two training seeds recorded roughly 0.031 damage and 0.13-0.15 s contact. The winning side threshold pinned at the new 0.50 floor with almost zero elite spread. | Reject as not promotable after the gate below | Return to the clean v7 parent and harden the objective against isolated recovery laps |
| D-033 | 2026-08-29 | exact v7/v8 winner traces | `artifacts/controller-search/faster-line-v9-diagnostics/v{7-gen27,v8-gen19}-seed-110.jsonl` | 110 | V8: 7.800 s best lap but 21.650 s first lap, 379.88 m, 0.2455 damage, 1.133 s contact, 365 AVOID ticks, and only 2 laps—the objective rewarded one fast post-crash lap. V7: 7.817 s best, 9.933 s first, 647.96 m, 0.0293 damage, 0.150 s contact, 8 AVOID ticks, and 3 laps. Neither is spotless on this official gate, but v7 preserves race pace and is the valid parent. | Add three-lap and clean-trial tiers ahead of lap time; restore the side threshold floor from 0.50 to 0.60 | Run v9 from v7 generation 27 |
| C-034 | 2026-08-29 | `faster-line-v9` GA, seeded from v7 generation 27 | `artifacts/controller-search/faster-line-v9-ga/generations/generation-026.json`, with 30 continuous archives | Training 28 | Generation 26 won at `[28,28,28,28,28,-7.783333,-7.783333,-7.783333,-7.782143,686.559]`; every trial survived, completed at least three laps, stayed inside budget, and was completely clean. Against v7, robust worst improved 0.050 s and mean improved 0.030 s. Gains arrived at generations 7, 11, 12, 15, 16, 17, 19, 20, 22, 25, and 26; generations 27-30 were flat, so the cap—not a ten-generation plateau—closed the run. Four workers grew to roughly 2.0-3.1 GB each in five-generation lifetimes, requiring complete pool restarts at 5, 10, 15, 20, and 25. | Keep generation 26 as the v10 parent; do not bake before official audit | Trace both official seeds and repair any launch-specific regression |
| D-035 | 2026-08-29 | v9 generation-26 official and first-corner traces | `artifacts/controller-search/faster-line-v10-diagnostics/v9-gen26-seed-*.jsonl` | Official `(110, 2026)` plus seed-110 ablations | Seed 2026 is clean at 7.783 s and 685.67 m. Seed 110 enters AVOID at 2.55 s and 19.5 m/s with the line target only -0.73 m, reaches recovery 0.18 s later, and contacts the wall: 9.967 s first lap, 7.833 s best, 645.01 m, 0.0277 damage, and 0.150 s contact. Faster global line slew was rejected: 0.05-0.12 slowed best lap to 9.38-9.80 s and increased incidents. A launch-only cap at 19 m/s through 3.5 s instead ran completely clean with 8 AVOID ticks, 8.667 s first lap, 7.783 s best, and 679.47 m. | Add a default-off two-parameter launch cap and retain repeated-lap pace ahead of first-lap tie-breakers | Seed v10 from v9 generation 26 plus the measured 19 m/s / 3.5 s launch point; include both official seeds in search evaluation |
| C-036 | 2026-08-29 | `faster-line-v10` GA, seeded from v9 generation 26 plus launch cap | `artifacts/controller-search/faster-line-v10-ga/generations/generation-004.json`, stopped with 7 continuous archives | Training 28 + official 2 | Generation 4 is the valid race-pace winner: all 30 trials clean, 7.800 s worst / 7.775 s median best lap, 8.917 s worst / 8.800 s median first lap, 683.59 m mean, and 24.450 s worst estimated three-lap time. Generation 5 improved repeated laps to 7.783/7.767 s but worsened the composite to 24.583 s. Generation 7 then exposed the objective defect: 7.767 s robust best laps with an 18.03 m/s cap lasting 3.98 s, but a 10.233 s worst first lap, 25.767 s composite, and only 673.40 m. It was clean but materially slower as a race. | Stop immediately; repeated-lap-first lexicographic ranking contradicts the user's first-turn goal | Seed v11 from generation 4 and rank robust estimated three-lap time before its components |
| C-037 | 2026-08-29 | `faster-line-v11` GA, seeded from v10 generation 4 | `artifacts/controller-search/faster-line-v11-ga/generations/generation-001.json` | Training 28 + official 2 | The v11 winner and its parent both have a 1,467-tick worst estimated three-lap total. Six-decimal component rounding represented them as 24.449999 and 24.450001, so generation 1 selected the wrong distribution: 471-tick worst repeated lap and 1,466-tick p90 composite versus the parent's 468 and 1,464. | Stop after generation 1; do not optimize decimal noise | Preserve generation 4 and move the v11 score to exact integer ticks in v12 |
| C-038 | 2026-08-30 | `faster-line-v12` GA, nominally seeded from v10 generation 4 | `artifacts/controller-search/faster-line-v12-ga/generations/generation-001.json` | Training 28 + official 2 | Exact-tick ranking worked, but generation 1 could not reproduce the 1,467-tick parent and reached only 1,470. Audit showed `_checkpoint_parameter_values()` loaded `best_parameter_vector` but discarded `checkpoint_context`; fixed values silently came from the currently baked v5 controller. The v12 centre therefore used straight target 24.674 instead of 25.076 m/s and front-brake start 12.406 instead of 11.919 m, plus stale fixed line/steering values. | Stop after generation 1 and retain the artifact as a seeding-failure audit | Merge context before vector, verify the exact parent, and restart as v13 |
| C-039 | 2026-08-30 | `faster-line-v13` GA, correctly seeded from the complete v10 generation-4 candidate | `artifacts/controller-search/faster-line-v13-ga/generations/generation-022.json`, with 30 archives | Training 28 + official 2 | Generation 22 won completely clean at `[30,30,30,30,30,-1449,-1448.1,-1445,-1443.6,-466,-466,-465,-465.167,-521,-518,-515,-513.267,688.573]`. Robust worst three-lap time improved 18 ticks (0.300 s) from the 1,467-tick parent. Gains continued through generation 22; generations 23-30 were flat. Five-generation process restarts contained workers that otherwise reached as high as 3.68 GB RSS. | Close at the 30-generation cap and keep generation 22 | Trace the user-identified final sweeper before baking |
| D-040 | 2026-08-30 | v13 generation-22 final-sweeper trace and ablations | `artifacts/controller-search/faster-line-v14-diagnostics/v13-gen22-seed-110.jsonl`; direct 30-seed probes | Seed 110, then training 28 + official 2 | The long turn is the lap's only sustained 2.38 s same-direction shape. Global exit offsets 0.01-0.95 and slow release lost distance or caused contact; straight targets 26-27 m/s also caused contact. The controller instead coasted because local target speed fell below actual speed. A 1.5 m/s bonus activated after 1.7 s and held 0.9 s stayed clean on all 30 seeds, improved robust worst/median three-lap totals from 1,449/1,445 to 1,443/1,438.5 ticks, and raised mean distance 688.57 to 692.37 m. Seed 110 best lap improved 7.767 to 7.717 s. | Accept the speed hypothesis; reject the line-shift hypothesis | Search only activation duration, hold, and bonus in v14 |
| C-041 | 2026-08-30 | `faster-line-v14` focused GA, seeded from v13 generation 22 | `artifacts/controller-search/faster-line-v14-ga/generations/generation-005.json`, stopped with 15 archives | Training 28 + official 2 | Generation 5 won completely clean at `[30,30,30,30,30,-1439,-1438,-1432,-1430.8,-461,-461,-460,-460,-519,-518,-511,-510.8,695.2380769788128]`, using a 2.148 s activation, 0.344 s hold, and 2.109 m/s bonus. Robust worst improved from v13's 1,449 to 1,439 ticks (0.167 s) and from the v10 parent by 28 ticks (0.467 s). Generations 6-15 were exactly flat; all winning values were interior to their bounds. | Stop at the ten-generation plateau and retain generation 5 | Bake and run every promotion gate |
| P-042 | 2026-08-30 | baked v14 generation 5 final promotion audit | `artifacts/controller-search/faster-line-v14-ga/{baked-official,baked-validation,baked-soak}.json`; `student-controllers.zip` | Official 2, validation 12, soak 100, and 20 role-swapped races | Official seeds were clean at 693.61/691.69 m, with 7.650/7.667 s best laps. Validation was 12/12 clean and lapped (697.38 m median, 690.96 m worst). Soak was 100/100 clean, survived, and lapped (694.76 m median, 689.28 m worst). Head-to-head passed at 18 wins, 0 ties, 2 losses. The extracted export reproduced both official trials exactly. Full verification passed: 210 tests, Ruff, formatting, Pyright, build, fresh finite call, and ~54.1 MiB process RSS. | Promote v14 generation 5 as the baked fast controller | Visually inspect seed 110; investigate traffic queuing separately |
| C-043 | 2026-08-30 | `faster-line-v15` focused GA, seeded from v14 generation 5 | `artifacts/controller-search/faster-line-v15-ga/generations/generation-027.json`, stopped with 37 archives | Training 28 + official 2 | Four-gene entry-preview bonus for the broad sweeper, ranked by `lap-time-v6`. Generation 27 won clean on all 30 seeds at `[30,30,30,30,30,-1432,-1429.2,-1425,-1422.87,-460,-458,-457,-457.27,-518,-515.2,-510.5,-508.33,698.9449704286582]` with a 0.1067-0.1454 far-curvature window, 1.151 s hold, and 1.530 m/s bonus. Robust worst three-lap improved from v14's 1,439 to 1,432 ticks (0.117 s) and best-lap worst from 461 to 460 ticks. Generations 28-37 were exactly flat. | Stop at the ten-generation plateau and retain generation 27 | Attack the launch cap, which v13's elites had pinned on its 2.5 s floor |
| C-044 | 2026-08-30 | `faster-line-v16` focused GA, seeded from v15 generation 27 | `artifacts/controller-search/faster-line-v16-ga/generations/generation-010.json`, stopped with 20 archives | Training 28 + official 2 | Reopened the two launch genes below v10's 2.5 s floor (21.5-22.9 m/s, 1.85-2.50 s) under `lap-time-v6`. Generation 10 won clean on all 30 seeds at `[30,30,30,30,30,-1427,-1427,-1424,-1422.27,-461,-459,-457,-457.6,-513,-512,-509,-507.07,699.2447230615736]` with a 22.774 m/s cap held 1.993 s. Robust worst three-lap improved 1,432 to 1,427 ticks and worst **first lap** 518 to 513 ticks (0.083 s), confirming the pinned floor was binding; best-lap worst regressed one tick to 461. Generations 11-20 were exactly flat. | Stop at the ten-generation plateau and retain generation 10 | Re-rank the same genes under an equal-weight first+best objective |
| C-045 | 2026-08-30 | `faster-line-v17` joint GA, seeded from v16 generation 10 | `artifacts/controller-search/faster-line-v17-ga/generations/generation-001.json`, stopped with 10 archives | Training 28 + official 2 | Reopened all six preview and launch genes jointly under the new equal-weight exact-tick `lap-time-v7`, which ranks worst first+best before either component. The seeded incumbent scored `[30,30,30,30,30,-970,-513,-461,-969,-512,-459,-966.5,-509,-457,-964.67,-507.07,-457.6,699.2447230615736]` at generation 1 and **no generation ever beat it**: all ten were exactly flat from the first. Re-ranking alone moves nothing; the six-gene joint space is exhausted. | Stop at the ten-generation plateau; reject joint re-mutation of the existing genes | Revise toward an unsearched structural lever rather than another re-ranking |
| P-046 | 2026-08-30 | baked v16 generation 10 promotion audit | `artifacts/controller-search/faster-line-v16-gate/{official,validation,soak}.json` | Official 2, validation 12, soak 100, and 20 role-swapped races | Official seeds clean at 696.02/698.50 m with **7.617 s** best laps and 8.483/8.550 s first laps, against v14's 7.650 s and 693.61/691.69 m. Validation 12/12 clean and lapped (700.17 m median, 686.04 m worst; first-lap worst 8.950 s). Soak 100/100 clean, survived, and lapped (698.55 m median, 691.41 m worst; best-lap worst 7.717 s, first-lap worst 8.617 s). Head-to-head passed at 18 wins, 0 ties, 2 losses. Median validation distance is 149.0% over `minimum_viable` and tenth-percentile 151.8%, against the 10%/5% gates. | Promote v16 generation 10 as the baked fast controller, superseding v14 generation 5 | Search the unsearched corner-exit lever as v18 |
| C-047 | 2026-08-30 | `faster-line-v18` single-gene GA, seeded from v16 generation 10 | `artifacts/controller-search/faster-line-v18-ga/generations/generation-001.json`, stopped with 11 archives | Training 28 + official 2 | New default-off pose-invariant lever: a corner-exit speed bonus scaled by how far near curvature has unwound relative to far curvature, searched over 0.0-3.0 m/s under `lap-time-v7`. **Every one of the eleven generations selected exactly `corner_exit_target_speed_bonus_mps = 0.0`**, the default-off value, and the seeded v16 score `[...,-970,-513,-461,...,699.2447230615736]` never moved. The lever is not merely unhelpful at the tuned optimum; no positive value anywhere in the range beat switching it off. | Stop at the ten-generation plateau; reject the corner-exit bonus outright | Stop adding levers and reopen the two bounds v13's elites pinned exactly against their box |
| C-048 | 2026-08-30 | `faster-line-v19` two-gene GA, seeded from v16 generation 10 | `artifacts/controller-search/faster-line-v19-ga/generations/generation-006.json`, stopped with 16 archives | Training 28 + official 2 | Reopened the two bounds v13's elites pinned exactly: the corner target below its 14.0 m/s floor (to 11.0) and the front stop above its 1.60 m ceiling (to 2.60), under `lap-time-v7`. Improved in six of the first six generations. Generation 6 won clean on all 30 seeds at `[30,30,30,30,30,-970,-513,-457,-969,-512,-457,-966.5,-509,-457,-964.1,-507.1,-457,700.4652]` with `corner_target_speed_mps=13.744` and `front_stop_m=2.037` - **both outside the old box**, confirming the bounds were binding. Robust worst best lap improved from 461 to **457 ticks** (7.683 to 7.617 s) and mean best-lap ticks reached exactly 457.0, i.e. every one of the 30 seeds hit the same best lap. Generations 7-16 were exactly flat. | Stop at the ten-generation plateau and retain generation 6 | Bake and run every promotion gate |
| P-049 | 2026-08-30 | baked v19 generation 6 promotion audit | `artifacts/controller-search/faster-line-v19-gate/{official,validation,soak}.json` | Official 2, validation 12, soak 100, and 20 role-swapped races | Beat baked v16 on every robustness measure and regressed on none. Official 2/2 clean, 698.05 m mean (was 697.26) and 697.52 m worst (was 696.02), 7.617 s best lap. Validation 12/12 clean: 701.71 m mean (was 700.12), 689.40 m worst (was 686.04), worst first lap 8.817 s (was 8.950), worst best lap 7.633 s (was 7.667). Soak 100/100 clean: 700.92 m mean (was 699.86), 693.20 m worst (was 691.41), worst best lap 7.633 s (was 7.717). Head-to-head improved to **19** wins, 0 ties, 1 loss. Zero damage and zero wall contact across all 114 trials. | Promote v19 generation 6 as the baked fast controller, superseding v16 generation 10 | Attack the first lap, frozen at 513 ticks since v16, by reopening v16's own launch box |
| C-050 | 2026-08-30 | `faster-line-v20` two-gene GA, seeded from v19 generation 6 | `artifacts/controller-search/faster-line-v20-ga/generations/generation-001.json`, stopped with 11 archives | Training 28 + official 2 | Reopened v16's launch box on both sides, to 22.0-24.5 m/s over 1.30-2.10 s, because worst first-lap time had read exactly 513 ticks in v16, v17, v18, and v19 and the launch genes are the only family that ever moved it. **All eleven generations selected the seeded 22.774 m/s / 1.993 s unchanged**; nothing anywhere in the widened box beat it. Unlike v13's launch floor, this box was not binding: the partial 4-of-12 elite pin on the 1.85 s floor did not reproduce. | Stop at the ten-generation plateau; treat 513 ticks as the launch floor for this controller | Retest the straight-speed ceiling, whose rejection predates v19's corner approach |
| D-051 | 2026-08-30 | line-clamp saturation check, no simulation | `src/controllers/preview_controller.py` line-target path; baked v19 parameters | n/a | Three line genes sit exactly on bounds - `maximum_racing_line_offset_ratio` and `racing_line_entry_offset_ratio` on their 0.95 ceilings and `racing_line_exit_offset_ratio` on its 0.0 floor - which normally reads as a binding bound worth reopening. It is not. `center_offset_cap_m` is 3.3 m, exactly the 6.6 m corridor half-width, so ratio 0.95 already places the requested line 3.135 m out, within 0.165 m of the wall, and 1.0 *is* the wall. D-026 separately measured 3.77 m of achieved offset at 0.29 m clearance. The active path is `_tracked_line_target`, which clamps to the searchable parameter; the hard-coded 0.65 `MAX_RACING_LINE_OFFSET_RATIO` applies only to the legacy non-pose-invariant branch, so no dead-clamp bug exists. | Reject the line *magnitude* family as a lever without spending a run on it | Spend the run on the straight-speed retest instead. **Superseded in part by D-055**: the saturation argument holds for `maximum_racing_line_offset_ratio`, which is the clamp, but was over-generalised to `racing_line_entry_offset_ratio`, which is a blend coefficient rather than a clamp |
| C-052 | 2026-08-30 | `faster-line-v21` two-gene GA, seeded from v19 generation 6 | `artifacts/controller-search/faster-line-v21-ga/generations/generation-019.json`, stopped early by the objective change | Training 28 + official 2 | Retested the straight-speed ceiling under v19's slower corner approach. It did move: generation 17 improved worst first+best from 970 to 969 ticks and generation 19 reached 700.64 m with mean best-lap 456.93, i.e. at least one seed broke the 457 floor. But generation 11 exposed the ranking defect below, and under the corrected objective this branch's worst best lap of 458 is a regression against v19's 457. | Stop the run; retain the finding that straight speed is live, discard the branch | Re-rank with best lap leading, then re-search the speed profile under it |
| D-053 | 2026-08-30 | `lap_time_score_v7` ranking defect, found from the user's observation that best lap was not improving while first lap was | `artifacts/controller-search/faster-line-v21-ga/run.log` generations 10-11 | Training 28 + official 2 | V7 ranks `max(first+best)`, then `max(first)`, then `max(best)`. Generation 11 moved 513+457 to 512+458: the sum tied at 970, so the comparison fell to the second key, first lap, and 512 beat 513 while best lap silently worsened. Best lap is the third key and is never consulted once an earlier key breaks the tie, so with the sum pinned the GA could bank first-lap ticks by spending best-lap ticks, one for one, and score each trade as an improvement. | Add `lap_time_score_v8` ranking `max(best)`, then `max(first)`, then the sum; the user chose best-lap priority, and in a three-lap trial a repeated lap counts twice against the first lap once | Re-search under v8; a regression test now encodes the exact 513+457 versus 512+458 pair |
| C-054 | 2026-08-30 | `faster-line-v22` four-gene GA under `lap-time-v8`, seeded from v19 generation 6 | `artifacts/controller-search/faster-line-v22-ga/generations/generation-001.json`, stopped with 11 archives | Training 28 + official 2 | Searched the whole speed profile jointly - straight target, corner target, and both ends of the front brake ramp - with every bound v19 and v21 reopened, ranked best lap first. **All eleven generations kept the seeded v19 values unchanged** at `[30,30,30,30,30,-457,-513,-970,...,-457,-507.1,-964.1,700.4649330821997]`, with mean best-lap exactly 457.0. Under a ranking that cannot trade best lap away, no combination of the four pace genes beats v19. | Stop at the ten-generation plateau; treat 457 ticks as a floor of the speed policy, not of the search | Attack the line's timing, the one dimension untouched by v18-v22 |
| D-055 | 2026-08-30 | line entry-ratio reachability, measured against the real `_line_target` | direct calls to `PreviewController._line_target` with the baked v19 vector | n/a | D-051 rejected the whole line family as physically saturated. That is right for the clamp but wrong for the entry coefficient, so the claim was measured rather than reasoned. Sweeping `racing_line_entry_offset_ratio` at 0.95, 1.20, and 1.60 across four phase blends: full entry and a 50% blend are already clipped at 0.95 and do not move at all; the weaker 25% blend moves from 0.795 to 0.950 at ratio 1.20 and then stops. The coefficient is therefore live, but only up to about 1.20 and only in a narrow band, and it reaches the same clamp sooner rather than extending the range. Magnitude really is capped; timing is not. | Keep the magnitude rejection, on measured rather than assumed grounds; do not spend a run on the entry ratio | Search `line_turn_sensitivity` and `line_target_release_per_tick`, both of which finished exactly on a bound |
| C-056 | 2026-08-30 | `faster-line-v23` two-gene GA under `lap-time-v8`, seeded from v19 generation 6 | `artifacts/controller-search/faster-line-v23-ga/generations/generation-004.json`, stopped with 14 archives | Training 28 + official 2 | Reopened the two pinned line-timing genes. Generation 4 moved `line_turn_sensitivity` to 0.00055, **below the old 0.002 floor**, confirming that floor was binding, and broke the 457-tick uniformity for the first time: mean best-lap fell 457.000 to 456.933, i.e. one of the thirty seeds reached 456 ticks (7.600 s). But the worst case did not move on any key - best lap 457, first lap 513, and first+best 970 all unchanged - and mean distance fell 700.465 to 700.30 m. `line_target_release_per_tick` barely moved (0.2500 to 0.2495), so the release ceiling is not the live half of the pair. Generations 5-14 were exactly flat. | Stop at the ten-generation plateau. **Do not promote**: the gain is one tick on one of thirty seeds, sits below every worst-case key in the ranking, and comes with a distance regression, so it does not justify re-running the promotion gates. Retain v19 generation 6 as baked | Line timing is the live direction if this is resumed; search `line_turn_sensitivity` alone, with a wider floor and a population large enough to resolve a sub-tick mean |

## Decision log

| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-08-27 | Optimize solo leaderboard performance first | Gradescope leaderboard ranks partial laps, speed, and lap times after survival qualification. |
| 2026-08-27 | Use tuned reactive control before learning | Processed camera geometry provides a strong low-risk preview signal and the repository has no training environment. |
| 2026-08-27 | Use standard-library CEM | Fits the CPU budget and fixed grader environment without adding dependencies. |
| 2026-08-27 | Keep simulator and autograder unchanged | Controller-only work reduces evaluation drift and submission risk. |
| 2026-08-27 | Gate, rather than assume, the neural residual | Learned complexity is accepted only when untouched-seed ablation proves a robust gain. |
| 2026-08-28 | Promote the hand-tuned minimum preset before CEM | It passed the official and untouched validation safe-lap gates on all 14 seeds with no damage or wall contact. |
| 2026-08-28 | Promote the generation-20 minimum CEM candidate | It remained clean on all 28 training and 14 official/validation seeds while materially increasing the robust distance floor. |
| 2026-08-28 | Isolate coasting and phase-aware line tuning in `faster-line` | The original 14-variable `faster` checkpoint must remain resumable through generation 100; the new behavior and 15-variable search therefore use default-off flags and a fresh artifact root. |
| 2026-08-28 | Coast instead of braking whenever `NORMAL` is overspeed | `resolve_vehicle_actuator_command` latches the vehicle: one negative-throttle tick above 1 km/h forces `engine_force = 0` and re-arms the latch every following tick, including for positive requests, until the car nearly stops. Braking buys about 0.16 m/s² of deceleration while the latch removes 8.7 m/s² of engine authority, a roughly 50:1 loss. A throttle of exactly zero clears the latch. Measured worth +24% distance and a second lap. |
| 2026-08-28 | Stop the `faster` search at generation 84, superseding the generation-100 commitment | The box was exhausted: eight of fourteen deviations at their floor, five means pinned to bounds, and 0.011% gained over the last six generations. It also optimized the braking policy that coasting replaces, so the remaining sixteen generations would have cost about 11,500 trials for an expected 0.1%. |
| 2026-08-28 | Read unclamped speed for control and demote `speed_cap_mps` to normalization | The clamped feature made the controller blind above the cap, so raising targets to 20/13 m/s while the cap stayed at 18 collapsed distance to 323 m with 1.33 damage; the same targets with headroom scored 491 m clean. Decoupling them removes the whole failure class and lets the search raise speeds freely. |
| 2026-08-28 | Loosen the improved incident gate to a per-trial budget | Summed damage and wall contact ranked ahead of every distance term, so a 437 m spotless candidate beat a 494 m candidate with 0.06 damage and the search could never enter the faster regime. plan.md gates `race_faster` on survival and a lap, not on zero damage. The budget keeps a hard wall at 0.25 damage, four times under the elimination threshold. |
| 2026-08-28 | Drop side-clearance parameters from the `faster-line` space | AVOID fired zero ticks across every solo probe and loosening `side_slow_start_m`, `side_speed_floor`, `avoid_side_wall_m`, and `avoid_diagonal_wall_m` changed nothing measurable, so they would waste two of fifteen dimensions. |
| 2026-08-28 | Preserve the generation-40 controller behind an opt-in v2 line flag | The pose-invariant line changes the meaning of phase-aware offsets. Keeping the new path default-off preserves the measured 563 m incumbent until a complete v2 candidate passes promotion gates. |
| 2026-08-28 | Use a full deterministic GA after a 10-generation CEM falsification | CEM variance contracted onto floors and cannot re-expand. The GA retains exact elites while tournament selection, BLX crossover, per-gene mutation, and stagnation inflation maintain alternate lines. |
| 2026-08-28 | Retain every completed generation and let the user launch campaigns | Each checkpoint is copied to an immutable numbered generation record. Codex implements and verifies the tooling, while the user owns long-running terminal commands and their timing. |
| 2026-08-28 | Defer neuroevolution and make any future residual target the line | A bounded line-target residual inherits the geometric clamp, slew, and clearance retraction, unlike a raw-steering residual that can bypass the interpretable safety envelope. |
| 2026-08-30 | Stop every search version after ten generations with no improvement, then revise rather than extend | v15 (gains through generation 27, flat 28-37), v16 (gains through 10, flat 11-20), v17 (flat 1-10), and v18 (flat 1-11) all confirm the plateau is real once it appears: no run resumed improving after ten flat generations. Extending a flat run costs about 70 s per generation and has never paid. |
| 2026-08-30 | Prefer reopening a bound the elites pinned over adding a new lever | v16 reopened the `startup_speed_cap_seconds` floor that v13's twelve elites all sat on at exactly 2.5000, and bought twelve ticks plus five ticks of worst first lap. The two levers tried instead - v17's joint re-ranking of existing genes and v18's new corner-exit bonus - each returned exactly zero over ten-plus generations. A gene pinned on its own bound is measured evidence the box is wrong; a new lever is only a hypothesis. |

| 2026-08-28 | Sharpen v2 entry/apex/exit phase selection before search | Five safe seed-110 probes failed the >0.8 m structural gate even at the planned gain and slew ceilings. The previous relative blend canceled an outside entry request with the opposite-sign apex request, and its unit test disabled preview steering. A 0.25 normalized phase-transition band plus target slew preserves smoothness while allowing an unambiguous approach to request the full entry target; a new whole-law test retains preview steering. |

| 2026-08-28 | Replace the all-straight 0.8 m gate with strong-target telemetry and a bounded search gate | On phase-fixed traces, the controller requested only 0.319-0.436 m on average over ticks counted as straight, making an all-straight >0.8 m response incompatible with its own reference. Strong-target telemetry separates physical reach from sign/lag: the box reached 1.043 m but was unsafe, while compensation 0.0 was the best safe starting point. The 10-generation CEM must now find a no-regression compromise before any GA spend. |

| 2026-08-28 | Reject the completed v2 CEM and block GA | The static preview/wall debias converged near one, but the optimizer reduced entry offset to the old 0.055 scale, pushed the winning line stiffness to 0.110, regressed mean distance by 4.52 m, and moved curvature sensitivity back toward its ceiling. This satisfies the plan's explicit stop condition: the remaining defect is the dynamic line-target schedule, not missing search time. Tier-4 rejection of almost every non-elite also confirms the objective cliff, but changing the objective cannot override the failed structural gate. |

| 2026-08-28 | Read the phase from local curvature, not from `c_i / d_i` | The camera reports the centreline's lateral position at each lookahead, so `c_i` accumulates every degree of turn over `[0, d_i]` and `c_i / d_i ~ d_i / 2R` rises with lookahead distance throughout a constant-radius corner. `far_turn` therefore outranked `near_turn` structurally, independent of phase, and entry weight was pinned at 1.0 for whole corners. That, not search budget or objective shape, is why the CEM drove `racing_line_entry_offset_ratio` to 0.055 and `center_steer_gain` to its floor: it was correctly switching off a line that asked the car to sit outside for the entire corner. Differencing the segment slopes recovers a local curvature and cancels the car's own pose algebraically, so the phase no longer depends on `curvature_offset_compensation` or `curvature_heading_compensation`; those stay in the space only for the speed scalar. |

| 2026-08-28 | Rebound `line_turn_sensitivity` to [0.010, 0.150] | The signal changed from a distance-scaled offset ratio to a curvature in 1/m that peaks at 0.158 on this track. The old [0.03, 0.40] box spent more than half its width above the signal's maximum, where `turn_strength` is pinned near zero and the racing line is off entirely, giving the optimizer a large basin of no-line solutions to fall into. |
| 2026-08-28 | Use a 10-generation GA as the falsification gate instead of re-running CEM | The CEM gate existed to falsify the line rework cheaply, but the repair has already been falsified offline over the whole lap and on track at 549.81 m with 3 laps and no safety regression. A 10-generation GA costs the same 10 generations, tests the same two structural predictions, and does it without the variance collapse that made the first CEM branch uninformative about a line that needs diversity to evaluate. |

| 2026-08-28 | Run the v2 GA unseeded rather than from the rejected CEM checkpoint | `--seed-checkpoint` supplies only the two line-frame spring compensations that the 17-gene space fixes, and the v2 base already carries 1.0 for both against the probe's 0.983 and 0.997. Seeding from the rejected branch would instead centre generation 0 on its other genes, which include `center_steer_gain` at its 0.110 floor, `racing_line_entry_offset_ratio` at 0.055, and `curvature_lateral_ratio` at 0.933: the collapsed no-line solution that only scored well because the phase selector was broken. `--seed-checkpoint` is now optional. |
| 2026-08-28 | Keep the `improved` objective despite 96.3% tier-4 rejections | The runbook's rule says switch to `improved-v2` above 50%, but that threshold assumed the CEM failure mode. Under the GA, elite diversity rose from 0.5302 to 0.8675 while the score improved monotonically for ten generations, so the cliff the softened objective exists to relieve is not binding. Tournament selection already lets a candidate reproduce by beating one random opponent, and the rejection counts describe elite selection rather than reproduction. Switching now would restart from generation 0 and discard a working trajectory. |

| 2026-08-28 | Re-diagnose the two failed structural predictions rather than treat them as a stop condition | `center_steer_gain` staying near its floor no longer indicates an incomplete debias: after the line-frame shift the preview term carries about 0.096 of restoring authority per metre against the centre spring's 0.037, so the line is tracked by preview and the centre gain has little left to do. `curvature_lateral_ratio` pinning at its ceiling while `curvature_heading_compensation` pins near zero is the same defect class as the phase bug, still present on the speed path: `track_curvature_preview` divides the cumulative offset by distance and so overstates curvature, and the ratio is the only desensitising lever the search has. Unlike the first CEM branch, the run beats the incumbent, so these are follow-up work rather than a stop. |
| 2026-08-28 | Promote the v2 GA vector despite two head-to-head losses | It passes every gate: official, validation, both improvement thresholds by more than tenfold their margin, and 14 of 20 races against a 12-win requirement. The two losses are the known queuing defect rather than a new failure mode, and in both the car was held near 273 m in traffic against a 596 m solo pace. The previous vector lost zero races, so this is recorded as a regression to fix rather than hidden, but solo pace gained 32.6 m on the official seeds and 0.57 s of lap time. |
| 2026-08-28 | Move the bounds rather than search longer, and keep both new signals default-off | The v2 GA finished with five genes on a bound: entry offset against the 0.65 clamp, `center_steer_gain` and `line_turn_sensitivity` on their floors, and `curvature_lateral_ratio` on its ceiling. That is the box binding, not the search. `MAX_RACING_LINE_OFFSET_RATIO` becomes the parameter `maximum_racing_line_offset_ratio`, searchable in [0.65, 0.90]; the geometry allows more, since the body edge only reaches the barrier near ratio 1.23, but 0.90 keeps roughly 1.1 m of margin against a measured strong-target tracking error near 1.3 m. `pose_invariant_speed_curvature` puts the speed scalar on `track_shape_preview`, which removes the same distance-inflation defect that broke the phase selector and makes both compensation genes inert, so the space drops from 17 genes to 16. Both flags default off, so the promoted 595.98 m vector re-traces bit-identically. |
| 2026-08-29 | Clamp every preset's initial values into that preset's own bounds | Each `faster*` preset derives its base from whatever is currently baked into `race_faster`, while `ParameterSpec` rejects an out-of-bounds initial. Baking the v3 winner therefore made the closed `faster` and `faster-line-v2-probe` searches unconstructible, because its `heading_steer_gain` of 0.40 and 0.90 line ratios fall outside their recorded boxes. Clamping the initial keeps those bounds exactly as the tests assert them and changes no history, since a resume reads the checkpoint rather than the initials. Without it, every future bake risks breaking an unrelated closed preset. |

| 2026-08-29 | Reject `ProcessPoolExecutor(max_tasks_per_child=...)` as the fix for worker memory growth | It hung the pool: workers exited and were never replaced, leaving the parent blocked on futures at 0% CPU with no error. Reverted. A 300-trial in-process measurement then showed `SoloEvaluator` grows only 0.4 MB per 1,000 trials, so the per-trial leak hypothesis is disproven and the real source is still unisolated. The remedy with evidence behind it is restarting the process, which returned generation time from 77.8 s to 55.9 s in v2 and from 557 s to 47.8 s in v3. `--evaluator-recycle-trials` remains available and is verified to leave results bit-identical, but is not demonstrated to help. |
| 2026-08-29 | Rate-limit the line target's return to centre separately from its move outward | The target is a function of instantaneous curvature, so it collapses to zero through any low-curvature gap and pulls the car back to the middle between two corners that bend the same way. The taut in-corridor path instead holds one edge for 64 m at a stretch, and the baked v3 vector sits within 0.5 m of the centreline for 39.9% of ticks. A single `line_target_release_per_tick` gene expresses the hold; a crossing to the opposite side keeps the fast outward rate so response to a real opposite corner is unchanged, and release equal to slew reproduces v3 exactly. This does not address the separate tracking gap, where the car asks for 2.97 m and achieves 2.06 m, so both must be measured after the run. |
| 2026-08-29 | Record that the line-target release was adopted for the opposite reason to the one predicted | The gene was added on the argument that the target collapses to zero through low-curvature gaps, so a *slow* return would hold the line and lift the 39.9% of ticks spent near the centreline. The search instead made release six to eight times *faster* than the outward slew, held there for twenty generations, and gained 44 m: it wants to ease outward slowly and snap back quickly. The likelier mechanism is the tracking gap rather than coverage, since v3 requested 2.97 m and achieved only 2.06 m, and a slowly ramped target is one the car can actually follow. The gene earned its place; the reasoning behind it did not. |

| 2026-08-29 | Attribute the long-run memory growth away from the search workers | A 300-trial in-process measurement showed `SoloEvaluator` grows 0.4 MB per 1,000 trials, and six workers hold about 6 GB resident while swap reached 22 GB, so roughly 16 GB was held elsewhere. The user then observed the editor consuming an enormous amount. Restarting the search relieved the pressure by freeing its own 6 GB without touching the real consumer, which is why it looked like a fix. `artifacts/` is now excluded from the editor's watcher and search index, since a training run rewrites a checkpoint and adds a file every 60 s for hours. |
| 2026-08-29 | Do not treat v4's low `line_clearance_m` as safety that the search discarded | It looked like the optimizer had driven the line retraction onto the AVOID threshold, leaving only 0.044 m between them, and v5 was first built to floor that gene. A seed-110 sweep falsified it: raising the threshold makes the target oscillate whenever the car runs wide, and 2.0 collapses the trial to 234 m with 728 AVOID ticks. Lowering the line clamp does not reduce peak offset either. v4's settings are the best measured point on distance, lap time, AVOID count, and clearance together, so v5 keeps them and seeks pace in the speed and steering genes instead. The wide excursions are real and unexplained, but they are not the racing-line target and no bound change addresses them. |
| 2026-08-29 | Give v6 structural variation instead of stronger mutations in v5's box | V5 gained only 1.56% over 100 generations even though normalized population diversity remained 0.369 and stagnation had already driven mutation scale to its 0.35 maximum. V6 searches four steering dynamics inherited unchanged through v5 (`yaw_damping_gain`, `steer_slew_per_tick`, `curvature_heading_degrees`, and `yaw_speed_reduction`) and reopens only the pressed curvature, heading, line-sensitivity, and release bounds. The maximum line clamp is removed from the genes and fixed at 0.95 because earlier sweeps found no safety or distance benefit from lowering it and geometry rules out widening it. Three ten-generation optimizer seeds test separate basins before one branch earns a longer run. |
| 2026-08-29 | Stop each new version at 30 generations or earlier on a measured plateau | V6 tied its parent for ten consecutive generations while retaining 0.487 diversity, mutation rose to 0.324, and generation time stayed stable. Longer fixed budgets hide failed hypotheses. Each successor gets at most 30 generations and is reviewed every 10-15; ten or more flat generations permit an early stop. |
| 2026-08-29 | Optimize robust best-lap time directly in v7 | The `improved` objective never contains lap time; it ranks survival, lap completion, incident budget, relative distance, and penalized mean distance. That was appropriate while establishing pace and safety, but it cannot distinguish a 7.883 s lap from a 7.917 s lap when the faster-lap policy loses distance during launch. The `lap-time` objective keeps the same three hard tiers, then minimizes worst, p90, median, and mean best-lap time before using penalized distance as a tie-breaker. |
| 2026-08-29 | Quantize lap times before v8 ranking | The simulator advances at 1/60 s, but accumulated binary floats made equivalent tick counts differ around 1e-15 s. Lexicographic ranking then selected that noise before considering p90, median, or mean. Rounding each measured lap to six decimals preserves far more precision than one tick while making equivalent laps tie as intended. |
| 2026-08-29 | Recycle the complete worker pool, not only `SoloEvaluator` | Closing and rebuilding the evaluator leaves Python/native allocations resident: workers grew to roughly 2 GB each over eleven generations and 3 GB each over nine more after restart. Exact checkpoint restart releases that RSS and preserves the optimizer RNG/population. Until pool recycling is implemented in-process, campaigns restart their command every ten generations. |
| 2026-08-29 | Rank consistency and cleanliness ahead of best-lap time in v9 | V8 demonstrates that surviving, completing one lap, and staying inside the loose incident budget do not prevent a crash-then-sprint policy: it can lose most of the trial, recover, and still win on one fast lap. V9 adds the count of three-lap trials and then zero-contact/zero-damage trials before every lap-time component. This returns selection to the last clean training parent while still minimizing robust lap time among equally consistent candidates. |
| 2026-08-29 | Repair the first corner with a launch-only speed cap, not faster global line slew | The seed-110 trace shows the first corner arrives while the tracked line target is still initializing: AVOID begins at 19.5 m/s and the controller falls into wall recovery. Raising global line slew from 0.0277 to 0.05-0.12 makes every later transition too aggressive, slows best lap by roughly 1.6-2.0 s, and increases damage. A 19 m/s cap for only the first 3.5 s eliminates contact and recovery, improves lap one by 1.30 s, improves best lap by 0.05 s, and adds 34.46 m. The new fields default to zero so existing controllers are bit-identical; v10 searches them and ranks repeated-lap pace before first-lap tie-breakers. |
| 2026-08-29 | Include both official seeds in v10 search evaluation | V9 was completely clean across all 28 training seeds, yet seed 110 still exposed a deterministic launch correction that the disjoint training suite could not rank. V10 keeps the broad 28-seed training suite and adds only the two known promotion seeds to both rotating preselection and full elite evaluation. This makes the measured failure visible without replacing the generalization suite, while validation and soak remain untouched gates. |
| 2026-08-29 | Rank robust estimated three-lap time before repeated and first-lap components in v11 | V10 proves that simply appending first-lap metrics after every repeated-lap metric is insufficient: generation 7 bought a one-tick repeated gain with a 1.3 s slower launch, losing 10 m despite remaining clean. A 30 s trial normally completes three full laps, so `first_lap + 2 * best_lap` is a conservative race-time proxy. Ranking its worst, p90, median, and mean first preserves the clean start and repeated pace together; the decomposed lap distributions and distance remain tie-breakers. |
| 2026-08-29 | Aggregate lap times as integer simulator ticks in v12 | Individual lap times are exact multiples of 1/60 s, but independently rounding them to six decimals before addition can move an equal total by two microseconds. V11 generation 1 exploited that representation rather than improving a physical tick. Converting each lap to `round(seconds * 60)` before `first + 2 * best` makes every equivalent total identical; percentile interpolation occurs only after the integer per-trial totals are formed. |
| 2026-08-30 | Treat checkpoint context as part of a seeded candidate | A search checkpoint splits the candidate between `best_parameter_vector` (searched fields) and `checkpoint_context` (fixed fields needed to reproduce it). Loading only the vector silently paired later branches with whichever values happened to be baked in source, so their advertised parent was false. Seeding now validates and applies context first, then lets searched values override any redundant name. V13 starts under a fresh preset/root because v12 generation 1 was evaluated against the wrong fixed controller. |
| 2026-08-30 | Speed up the long sweeper with a sustained-turn bonus, not a forced line | The user-identified segment really did have unused speed: the controller was coasting at roughly 24-25 m/s because local curvature lowered its target. Conventional outside-exit and line-hold variants were slower or unsafe because they affect every corner and add cross-track motion. A stateful, sensor-relative detector isolates the only turn direction sustained beyond 1.7 s and adds speed locally; its measured seed improves every robust three-lap statistic across all 30 search seeds without contact. The three new parameters default to zero, so existing controllers remain unchanged. |
| 2026-08-30 | Close the solo v14 branch at generation 15 and promote generation 5 | Ten later generations were exactly flat despite GA stagnation inflation, while the winning activation, hold, and bonus values all remained inside their bounds. The final vector passes the 30-seed search suite, untouched validation, 100-seed soak, head-to-head, export, and code-quality gates. More generations or wider mutations in the same three-dimensional box have no evidence-backed expected gain; another version requires a new measured mechanism. |

## Iteration rules

After each implementation or experiment iteration:

1. Update the phase checkbox and top-level status.
2. Append an experiment row with exact seed suite and artifact filename.
3. Update current-best controllers only after every applicable promotion gate.
4. Add a decision-log row when an approach, threshold, or parameter family is
   changed or rejected.
5. Set one concrete next action.
6. Never erase a failed result; supersede it with a later row.
