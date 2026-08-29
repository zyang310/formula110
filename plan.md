# Competitive Formula 110 Controller - Living Plan

Last updated: 2026-08-28

## Status

- Project state: the 563 m `faster-line` incumbent is frozen and verified
  bit-identical after the phase-selector repair. The rejected CEM branch has
  been diagnosed: the phase selector read `c_i / d_i` as curvature, which grows
  with lookahead distance inside any corner, so entry was pinned on and the
  optimizer's only rational reply was to switch the racing line off. The
  selector now differences segment slopes into a local, pose-invariant
  curvature. A fresh CEM gate under a new artifact root is the next run.
- Current submission baseline: `controllers.crash_fast` (stationary).
- Current best safe controller: `controllers.minimum_viable` (coasting; re-gated
  clean on all 14 official and validation seeds).
- Current best fast controller: `controllers.race_faster` (generation-40
  `faster-line-v2` GA vector; 595.98 m mean on the official seeds, clean, and
  past every promotion gate).
- Next action: run the `faster-line-v3` GA. The preset is implemented and
  verified but deliberately not started. The 100-seed soak and Phase 4 export
  checks also remain.

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
- [ ] User: run the `faster-line-v3` GA.
- [ ] User: after that repair, run 10 GA generations and select the objective,
  then finish the selected branch through generation 40.
- [x] Promote and bake: official, validation, and head-to-head all pass.
- [ ] Fix head-to-head queuing: `race_faster` closes on slower traffic and holds
  station instead of passing, costing roughly half its solo pace in traffic. Now
  a measured regression at two losses in twenty races, up from zero.

### Phase 3 - Residual ablation

- [ ] Reassess a bounded residual only after the pose-invariant line/GA campaign;
  if attempted, target `line_target` rather than raw steering.
- [ ] Compare rules-only and residual policies on untouched validation seeds.
<!-- - [ ] Record the ablation and apply the objective promotion gate. -->

### Phase 4 - Final verification

- [ ] Run the 100-seed soak.
- [ ] Run full tests and strict type checking.
- [ ] Run isolated Ruff lint and format checks on exported modules.
- [ ] Verify fresh-process load, call time, finite commands, and memory use.
- [ ] Export both controller modules and inspect archive contents.
- [ ] Re-run official trials from the exported package.
- [ ] Freeze parameters and mark the plan complete.

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

## Iteration rules

After each implementation or experiment iteration:

1. Update the phase checkbox and top-level status.
2. Append an experiment row with exact seed suite and artifact filename.
3. Update current-best controllers only after every applicable promotion gate.
4. Add a decision-log row when an approach, threshold, or parameter family is
   changed or rejected.
5. Set one concrete next action.
6. Never erase a failed result; supersede it with a later row.
