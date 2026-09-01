# Controller training runbook

Commands for tuning, monitoring, baking, and gating the preview controller
presets. Everything here writes to the git-ignored `artifacts/` tree; shipped
parameters are always formatted through `scripts.controller_training.bake`,
then baked into `src/controllers/`; runtime code never reads `artifacts/`.

Run every command from the repository root.

## Presets

| Preset | Tunes | Ranking | Status |
| --- | --- | --- | --- |
| `minimum` | 12 conservative gains for `controllers.minimum_viable` | `minimum_score`: safe laps, clean trials, worst then mean distance | Complete at generation 20 |
| `faster` | 14 gains for `controllers.race_faster` | `improved_score` | **Closed** at generation 84; kept only as an audit trail |
| `faster-line` | 15 speed, steering, and racing-line parameters | `improved_score` | **Closed** at generation 40; current 563 m incumbent |
| `faster-line-v2-probe` | 19 pose-invariant line and speed parameters | `improved_score` with CEM | First gate rejected; rerun after the phase repair |
| `faster-line-v2` | 17 parameters after fixing the two line-frame spring compensations | `improved_score` or `improved_score_v2` with GA | **Closed** at generation 40; current 595.98 m incumbent |
| `faster-line-v3` | 16 parameters; unpins the five v2 bounds and adds the searchable line clamp | `improved_score` with GA | **Closed** at generation 60; current 621 m baked vector |
| `faster-line-v4` | 17 parameters; raises v3's pinned ceilings and adds the asymmetric target release | `improved_score` with GA | **Closed** at generation 57; promoted 665.39 m vector |
| `faster-line-v5` | 17 parameters; retains the safe line ceiling and reopens pace bounds | `improved_score` with GA | **Closed** at generation 100; 675.77 m baked candidate, not promotion-gated |
| `faster-line-v6` | 20 parameters; fixes the line clamp, adds four steering-dynamics genes, and reopens v5's pressured non-geometric bounds | `improved_score` with GA | **Closed** at generation 10; exactly flat against v5 |
| `faster-line-v7` | 16 focused speed and wall-policy parameters | robust `lap_time_score` with GA | **Closed** at generation 27; real winner 7.833 s worst / 7.817 s median |
| `faster-line-v8` | 14 v7 refinements with moved wall/curvature bounds | quantized `lap_time_score_v2` with GA | **Closed** at generation 29; rejected crash-then-sprint optimum |
| `faster-line-v9` | 14 safe wall/speed refinements | consistency-first `lap_time_score_v3` with GA | **Closed** at generation 30; generation 26 reached 7.783 s robust worst, but seed 110 still corrected on lap one |
| `faster-line-v10` | 16 startup, corner, steering, and wall refinements | official-aware `lap_time_score_v4` with GA | **Closed** at generation 7; rejected a clean but 10.233 s slow-launch optimum |
| `faster-line-v11` | v10's 16-gene first-corner box | robust three-lap `lap_time_score_v5` with GA | **Closed** at generation 1; decimal component rounding split equal tick totals |
| `faster-line-v12` | v10's 16-gene first-corner box | exact-tick three-lap `lap_time_score_v6` with GA | **Closed** at generation 1; seed loader omitted the parent's fixed context |
| `faster-line-v13` | v10's 16-gene first-corner box with complete checkpoint seeding | exact-tick three-lap `lap_time_score_v6` with GA | **Closed** at generation 30; generation 22 is clean on all 30 seeds at 24.150 s robust worst |
| `faster-line-v14` | 3 long-sweeper activation/hold/speed-bonus parameters | exact-tick three-lap `lap_time_score_v6` with GA | **Closed** at generation 15; generation 5 is promoted at 23.983 s robust worst |
| `faster-line-v15` | 4 sweeper entry-preview window/hold/bonus parameters | exact-tick three-lap `lap_time_score_v6` with GA | **Closed** at generation 37; generation 27 is clean on all 30 seeds at 1,432 ticks |
| `faster-line-v16` | 2 launch-cap parameters, reopened below v10's pinned 2.5 s floor | exact-tick three-lap `lap_time_score_v6` with GA | **Closed** at generation 20; generation 10 is **promoted** at 1,427 ticks and 7.617 s official best lap |
| `faster-line-v17` | v15's 4 preview genes and v16's 2 launch genes, jointly | equal-weight first+best `lap_time_score_v7` with GA | **Closed** at generation 10; zero improvement over its seed in any generation |
| `faster-line-v18` | 1 default-off pose-invariant corner-exit speed bonus | equal-weight first+best `lap_time_score_v7` with GA | **Closed** at generation 11; every generation selected the default-off 0.0, rejecting the lever |
| `faster-line-v19` | 2 parameters; reopens the corner-target floor and front-stop ceiling v13's elites pinned | equal-weight first+best `lap_time_score_v7` with GA | **Closed** at generation 16; generation 6 is **promoted** at 457 robust worst best-lap ticks |
| `faster-line-v20` | 2 launch parameters, reopening v16's own box on both sides | equal-weight first+best `lap_time_score_v7` with GA | **Closed** at generation 11; every generation kept the seeded launch, establishing the 513-tick first-lap floor |
| `faster-line-v21` | 2 parameters; retests the straight-speed ceiling under v19's slower corner approach | equal-weight first+best `lap_time_score_v7` with GA | **Discarded** at generation 19; it moved, but exposed the v7 ranking defect and its branch regresses best lap |
| `faster-line-v22` | 4 parameters; the whole speed profile, jointly | best-lap-first `lap_time_score_v8` with GA | **Closed** at generation 11; every generation kept the seeded v19 values |
| `faster-line-v23` | 2 line-timing parameters that finished exactly on a bound | best-lap-first `lap_time_score_v8` with GA | **Closed** at generation 14; broke the 457-tick floor on one seed, but not on any worst-case key, so not promoted |
| `faster-line-v24` | 8 global speed-cap and startup brake-turn drift parameters | best-lap-first `lap_time_score_v8` with GA | **Closed** after 10 flat generations; every winner left the new drift disabled |
| `faster-line-v25` | 9 local-corridor speed-bonus and reusable drift parameters | bounded-incident `lap_time_score_v9` with GA | **Closed and promoted** at generation 10; 30/30 search trials clean, with 456-tick official best laps |
| `faster-line-v26` | 8 long-corridor throttle, detection, braking, and steering-speed parameters | live-seed `speed_max_score_v1` with GA | **Closed** at generation 16 after 10 flat generations; generation 6 is baked separately into `race_speedmax` |

## Train

```bash
uv run python -m scripts.controller_training.search faster-line \
  --artifact-root artifacts/controller-search/faster-line \
  --population 64 \
  --elites 12 \
  --generations 40 \
  --optimizer-seed 590114
```

Re-running the exact same command **resumes** from the last completed
generation; it never repeats finished work.

Plumbing smoke test before committing to a long run (about a minute). Always
give it its own artifact root:

```bash
uv run python -m scripts.controller_training.search faster-line \
  --artifact-root artifacts/controller-search/scratch \
  --population 8 --elites 3 --generations 2 --optimizer-seed 590114 --seconds 3.0
```

Useful flags: `--workers` (defaults to CPU count minus one), `--seconds` (trial
duration, default 30.0).

## Pose-invariant line and GA campaign

Run this campaign manually. The commands below deliberately use three different
artifact roots so a CEM checkpoint can never be mistaken for a GA checkpoint or
an objective-v2 restart.

### 1. Record the incumbent trace

```bash
uv run python -m scripts.controller_training.trace controllers.race_faster \
  --seed 110 \
  --output artifacts/controller-search/faster-line-v2-traces/incumbent-seed-110.jsonl
```

The last JSONL row is a summary. It should reproduce roughly 1,801 ticks, 574
straight ticks, 207 corner ticks, 0.126 m mean absolute straight offset, zero
AVOID ticks, and 563.27 m with zero damage/contact.

### 2. Choose heading compensation and run the structural probe

Run the three heading-compensation traces. The `phase-fixed` names deliberately
preserve the five failed pre-fix traces as diagnostic evidence:

```bash
for heading_compensation in 0.0 0.5 1.0; do
  uv run python -m scripts.controller_training.trace \
    --preset faster-line-v2-probe \
    --seed 110 \
    --set curvature_heading_compensation="$heading_compensation" \
    --set racing_line_entry_offset_ratio=0.35 \
    --output "artifacts/controller-search/faster-line-v2-traces/phase-fixed-heading-${heading_compensation}.jsonl"
done
```

Choose the smallest value tied for the lowest straight curvature without a
phase-sign, damage, contact, or AVOID regression. The completed phase-fixed
seed-110 sweep selected `0.0`: it retained zero safety incidents, the best
distance (541.02 m), the lowest straight curvature (0.0463), and the best safe
strong-target directional offset (0.295 m).

The original all-straight `>0.8 m` gate is retired. It was dimensionally clear
but measured the wrong population: the mean requested target on those same
ticks was only 0.319-0.436 m, so the gate required roughly twice the reference
being tracked. Use the trace's `strong_line_target_*` fields instead. The boxed
probe established physical authority (1.043 m mean absolute offset while
`abs(line_target_m) >= 0.8`), but its 40 AVOID ticks and 476.16 m distance reject
that configuration. The 10-generation CEM is therefore the next bounded test
for a safe compromise. Do not start the GA unless CEM also satisfies the two
parameter predictions below and its winner recovers the incumbent distance.

### 3. Run the 10-generation CEM falsification

If the selected heading compensation differs from the preset's `0.5`, update
`FASTER_LINE_V2_BASE_PARAMETERS` before starting this run.

```bash
uv run python -m scripts.controller_training.search faster-line-v2-probe \
  --optimizer cem \
  --objective improved \
  --artifact-root artifacts/controller-search/faster-line-v2-cem-gate \
  --population 64 \
  --elites 12 \
  --generations 10 \
  --optimizer-seed 590115 \
  --workers 4
```

Inspect the two falsifiable predictions:

```bash
uv run python -c "
import json
p='artifacts/controller-search/faster-line-v2-cem-gate/checkpoint.json'
d=json.load(open(p))
m=dict(zip(d['parameter_names'],d['distribution_mean']))
print('center_steer_gain',m['center_steer_gain'],'PASS' if m['center_steer_gain'] > 0.14 else 'FAIL')
print('curvature_lateral_ratio',m['curvature_lateral_ratio'],'PASS' if m['curvature_lateral_ratio'] < 0.96 else 'FAIL')
"
```

Stop and fix the line debias if either prediction fails. The final GA reads this
checkpoint directly: its best 19-gene vector becomes generation 0's incumbent,
while the winning preview/wall compensation values become fixed context.

**Recorded outcome (superseded by the phase-selector repair):** the completed
generation-10 branch under `faster-line-v2-cem-gate` is rejected. All ten immutable archives exist, but the
distribution ended at `center_steer_gain=0.13662` (fails `>0.14`) and
`curvature_lateral_ratio=0.85646` (passes but trended upward); the winning vector
used `center_steer_gain=0.11000`, `curvature_lateral_ratio=0.93283`, and only
`0.05506` entry offset. Its clean 558.49 m training mean regresses the 563.01 m
incumbent. Preview/wall compensation converged near one, isolating the remaining
defect to dynamic target scheduling. **Do not run the GA commands below until a
fresh CEM branch passes under a new artifact root.**

The cause was found afterwards and is now fixed in the controller: the phase
selector read `kappa_i = c_i / d_i` as curvature, but `c_i` accumulates the
whole turn over `[0, d_i]`, so `c_i / d_i` grows with lookahead distance
throughout any corner and `far_turn` outranked `near_turn` regardless of phase.
On the bare centreline with zero pose error that pinned 84.6% of the phase
weight on entry against 7.8% on apex. `track_shape_preview` now differences the
segment slopes into a local curvature, which also cancels the car's pose
algebraically. Rerun the gate under a fresh root:

```bash
uv run python -m scripts.controller_training.search faster-line-v2-probe \
  --optimizer cem \
  --objective improved \
  --artifact-root artifacts/controller-search/faster-line-v2-cem-gate-2 \
  --population 64 \
  --elites 12 \
  --generations 10 \
  --optimizer-seed 590115 \
  --workers 4
```

Check the phase mix first with a single trace; entry pinning is now visible
directly in the summary rather than only in hindsight:

```bash
uv run python -m scripts.controller_training.trace \
  --preset faster-line-v2-probe \
  --seed 110 \
  --set racing_line_entry_offset_ratio=0.35 \
  --output artifacts/controller-search/faster-line-v2-traces/phase-repaired-heading-0.0.jsonl

uv run python -c "
import json
row = json.loads(open('artifacts/controller-search/faster-line-v2-traces/phase-repaired-heading-0.0.jsonl').readlines()[-1])
print({k: round(row[k], 4) for k in ('phase_entry_mass','phase_apex_mass','phase_exit_mass')})
print('corner ticks', row['phase_tick_count'])
"
```

An `phase_entry_mass` back above roughly 0.55 means the selector has regressed
to pinning and no amount of search time will help.

### 4. Run the first 10 GA generations

Run this **unseeded**. `--seed-checkpoint` only supplies the two line-frame
spring compensations that the 17-gene space fixes rather than searches, and the
v2 base already carries 1.0 for both, against the probe's 0.983 and 0.997.
Seeding from `faster-line-v2-cem-gate` would instead centre generation 0 on that
branch's other genes, which include `center_steer_gain` on its floor and a 0.055
entry offset: the collapsed no-line solution that only scored well because the
phase selector was broken. Unseeded, generation 0 member 0 is the incumbent.

```bash
uv run python -m scripts.controller_training.search faster-line-v2 \
  --optimizer ga \
  --objective improved \
  --artifact-root artifacts/controller-search/faster-line-v2-ga \
  --population 64 \
  --elites 12 \
  --generations 10 \
  --optimizer-seed 590115 \
  --workers 4
```

Check whether more than half of rejected candidates first lost at score tier 4:

```bash
uv run python -c "
import glob,json
paths=sorted(glob.glob('artifacts/controller-search/faster-line-v2-ga/generations/generation-*.json'))[:10]
rows=[json.load(open(p))['metrics']['diversity'] for p in paths]
tier4=sum(r['rejection_count_by_score_tier'][3] for r in rows)
rejected=sum(sum(r['rejection_count_by_score_tier'])+r['rejection_count_tied_at_cutoff'] for r in rows)
print({'tier4':tier4,'rejected':rejected,'fraction':tier4/rejected})
"
```

Also check that the phase selector has not regressed to entry pinning, and that
both structural predictions now hold:

```bash
uv run python -c "
import json
d=json.load(open('artifacts/controller-search/faster-line-v2-ga/checkpoint.json'))
b=dict(zip(d['parameter_names'],d['best_parameter_vector']))
print('center_steer_gain     ',round(b['center_steer_gain'],5),'PASS' if b['center_steer_gain']>0.14 else 'FAIL')
print('curvature_lateral_ratio',round(b['curvature_lateral_ratio'],5),'PASS' if b['curvature_lateral_ratio']<0.96 else 'FAIL')
print('entry offset          ',round(b['racing_line_entry_offset_ratio'],5),'PASS' if b['racing_line_entry_offset_ratio']>0.12 else 'FAIL')
print('best score',[round(x,4) for x in d['metrics']['best_score']])
"
```

If the fraction is at most `0.5`, resume the same run through generation 40:

```bash
uv run python -m scripts.controller_training.search faster-line-v2 \
  --optimizer ga \
  --objective improved \
  --artifact-root artifacts/controller-search/faster-line-v2-ga \
  --population 64 \
  --elites 12 \
  --generations 40 \
  --optimizer-seed 590115 \
  --workers 4
```

If the fraction is greater than `0.5`, leave the first ten generations intact
and start the softened objective from generation 0 in its own root:

```bash
uv run python -m scripts.controller_training.search faster-line-v2 \
  --optimizer ga \
  --objective improved-v2 \
  --artifact-root artifacts/controller-search/faster-line-v2-ga-soft \
  --population 64 \
  --elites 12 \
  --generations 40 \
  --optimizer-seed 590115 \
  --workers 4
```

### Every generation is retained

After each completed generation, the search atomically writes `checkpoint.json`
and immediately copies that exact record to
`generations/generation-NNN.json`. This applies to CEM and GA, including the
first ten generations of a rejected objective branch. Generation archives are
immutable: a conflicting rerun raises instead of overwriting history.

To count and inspect them:

```bash
find artifacts/controller-search/faster-line-v2-ga/generations -name 'generation-*.json' -print | sort
```

Never delete a branch merely because it lost. Preserve it as the experiment
record and use a new artifact root for a restart.

### Three things that will bite you

- **`--optimizer-seed` must match the checkpoint** or the resume refuses to
  start. `faster-line` is `590114`, the closed `faster` run is `590113`, and the
  CLI default is `590112`.
- **To extend a finished run**, re-run it with a larger `--generations`. To
  *restart* one, use a new artifact root. Never delete the old checkpoint or
  numbered generations; they are the complete experiment record.
- **Never rank generations on the official seeds.** Suites are fixed in
  `seeds.py`: 2 official, 28 training, 12 validation, 100 soak. The optimizer
  only ever sees training seeds.

### Set `--workers` by memory, not by core count

Each worker holds a persistent headless Panda3D `ShowBase` so it does not
rebuild the scene per trial. That costs roughly **950 MB of resident memory per
worker**, and the default is CPU count minus one.

On a 24 GB machine the default of 9 workers needs about 8.5 GB of evaluators and
drives the system into swap. Symptoms: generations slow from ~40 s to several
minutes, workers sit at 0% CPU in state `U` (uninterruptible page-in wait), and
their RSS collapses to a few MB as they are paged out. Budget about 1 GB per
worker against free RAM and cap the flag accordingly:

```bash
--workers 4        # ~3.7 GB of evaluators
```

Fewer resident workers finish faster than more swapped ones. Check with:

```bash
sysctl vm.swapusage
ps -o pid,stat,rss,%cpu,command -p $(pgrep -f controller_training | tr '\n' ',' | sed 's/,$//')
```

Stopping a run is safe at any point: checkpoints are written atomically after
each generation, so a resume loses at most the generation in flight.

## Watch a run in progress

Search output is buffered, so read the checkpoint rather than the log. It is
rewritten after every generation.

```bash
uv run python -c "import json;d=json.load(open('artifacts/controller-search/faster-line/checkpoint.json'));print(d['generation'],[round(x,4) for x in d['metrics']['best_score']])"
```

```bash
# How many generations are done
ls artifacts/controller-search/faster-line/generations | wc -l

# Is it still running
ps aux | grep "[c]ontroller_training.search"
```

`best_score` is a 7-tuple compared lexicographically, highest wins:

| # | Meaning |
| --- | --- |
| 1 | Seeds survived (of 28) |
| 2 | Seeds completing at least one lap |
| 3 | Seeds inside the incident budget: damage ≤ 0.25 and wall contact ≤ 1.5 s |
| 4 | Worst per-seed distance improvement over `minimum_viable` |
| 5 | Tenth-percentile improvement |
| 6 | Median improvement |
| 7 | Mean distance, charged 120 m per damage unit and 6 m per contact second |

So `[28, 28, 28, 0.93, 0.93, 0.95, 557.5]` means all 28 training seeds survived,
lapped, and stayed inside the budget, with the worst seed 93% further than the
baseline.

### Lap-time trend

Lap time is diagnostic telemetry, not part of the ranking. Track it per
generation with:

```bash
uv run python -c "
import json, glob, statistics
for p in sorted(glob.glob('artifacts/controller-search/faster-line/generations/generation-*.json')):
    d=json.load(open(p)); res=d['metrics'].get('generation_best_results') or []
    if not res: continue
    laps=[r['best_lap_time_seconds'] for r in res if r.get('best_lap_time_seconds') is not None]
    dist=[r['raw_distance_m'] for r in res]
    print(d['generation'], round(min(laps),3) if laps else '-', round(statistics.median(laps),3) if laps else '-', round(statistics.fmean(dist),1))
"
```

Three caveats before drawing conclusions from it:

- **It is quantized to one physics tick, 1/60 s ≈ 16.7 ms.** Consecutive
  generations often differ by exactly one tick, so lap time looks flat during
  real progress. Confirm a plateau against mean distance, which is continuous.
- **The search never optimizes it.** Ranking is survival, laps, incident budget,
  then distance. Faster laps are a side effect, so lap time can stall while the
  score still improves.
- **`min` across seeds is the luckiest spawn.** Use the median. The value is
  `null` until a car crosses the line twice, since one crossing from a random
  spawn is a partial lap, not a lap time.

### Knowing when to stop

A run is exhausted when the deviations collapse onto their floors and the means
pin to their bounds. Check before spending more generations:

```bash
uv run python -c "
import json
d=json.load(open('artifacts/controller-search/faster-line/checkpoint.json'))
for n,m,s in zip(d['parameter_names'],d['distribution_mean'],d['distribution_deviation']):
    print(f'{n:>32} mean={m:9.4f} dev={s:8.5f}')
"
```

The `faster` preset was stopped at generation 84 because 8 of its 14 deviations
sat on the floor and 5 means were pinned to a bound, buying 0.011% over six
generations.

## The v3 campaign

The v2 GA winner finished with five genes on a bound, so v3 moves the bounds
rather than searching longer:

| gene | v2 result | v3 bound | why |
| --- | --- | --- | --- |
| `racing_line_entry_offset_ratio` | 0.6486, on the 0.65 clamp | ratios to 0.90 | the clamp, not the search, was binding |
| `maximum_racing_line_offset_ratio` | fixed constant | searchable [0.65, 0.90] | half-track is 3.3 m and the hull half-width 0.63 m, so the body edge only reaches the 4.7 m barrier near ratio 1.23; 0.90 keeps about 1.1 m of margin |
| `center_steer_gain` | 0.1000, on its floor | floor to 0.0 | after the line-frame debias the preview term carries the line |
| `line_turn_sensitivity` | 0.0100, on its floor | floor to 0.002 | the signal is smaller than the first rebound assumed |
| `curvature_lateral_ratio` | 1.0000, on its ceiling | redefined, [0.02, 0.30] | the ceiling was forced by the distance-inflated speed signal |

`pose_invariant_speed_curvature` also switches the speed scalar onto
`track_shape_preview`, which cancels pose algebraically. That makes
`curvature_offset_compensation` and `curvature_heading_compensation` inert, so
they leave the space: 17 genes become 16. Its initial `curvature_lateral_ratio`
of 0.155 was calibrated on a seed-110 trace of the promoted vector as the value
whose `curvature` distribution best matches what that vector actually ran.

Both new behaviours are default-off, so the promoted 595.98 m controller is
reproducible unchanged; a re-traced seed 110 matches it on all 1,802 rows.

```bash
caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v3 \
  --optimizer ga \
  --objective improved \
  --artifact-root artifacts/controller-search/faster-line-v3-ga \
  --population 64 \
  --elites 12 \
  --generations 40 \
  --optimizer-seed 590116 \
  --workers 3
```

Run it on mains power. On battery this machine idle-sleeps and thermally sleeps
mid-run: during the v2 GA, 3,535 s of a 5,820 s window was spent asleep, and two
Thermal Emergency Sleep events cost 876 s and 15 s. `caffeinate -ims` blocks idle
and system sleep but **not** thermal sleep, which is what the power adapter and
the lower worker count are for.

Generation 0 anchors on the promoted vector, so elitism holds 597.90 m as a
floor. Promote only against the v2 incumbent's gates: 595.98 m official mean,
596.55 m validation median, and 14 of 20 head-to-head wins.

## The v4 campaign

v3 finished with nine of sixteen genes on a bound, so v4 again moves bounds, and
adds one structural gene.

**Why the release rate exists.** The line target is a function of *instantaneous*
curvature, so `turn_strength` collapses to zero wherever local curvature is low
and the target is dragged back to the centreline, even when the next corner bends
the same way and the car should simply stay out. Measured on the baked v3 vector
at seed 110: it requests up to 2.97 m but achieves only 2.06 m at the ninetieth
percentile, and sits within 0.5 m of the centreline for 39.9% of ticks. The taut
shortest path inside this corridor instead stays pinned to one edge for 64 m at a
stretch and is 12.2% shorter than the centreline.

`line_target_release_per_tick` rate-limits the target moving back toward the
centre *on the same side*, separately from the outward `line_target_slew_per_tick`.
A crossing to the opposite side still uses the fast outward rate, so a genuine
opposite corner is answered exactly as before. Setting release equal to slew
reproduces v3, which is where generation 0 starts.

```bash
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v4 \
  --optimizer ga \
  --objective improved \
  --artifact-root artifacts/controller-search/faster-line-v4-ga \
  --population 64 --elites 12 --generations 100 \
  --optimizer-seed 590117 --workers 6
```

Create the artifact directory first (`mkdir -p`) if your shell autocorrects
unknown paths, run on mains power, and **restart the process whenever generation
time passes roughly 150 s** — see the note below on worker memory growth.

## The v5-v7 campaigns

The v5 run completed all 100 generations at a best penalized mean of 675.77 m,
up 10.38 m (1.56%) from v4. Its winning vector first appeared at generation 84.
The final population still had 0.369 normalized pairwise diversity and the GA
had already expanded its mutation scale to the 0.35 maximum. More mutation in
the same box is therefore unlikely to address the plateau.

V6 changes the search structure instead. It makes `yaw_damping_gain`,
`steer_slew_per_tick`, `curvature_heading_degrees`, and `yaw_speed_reduction`
searchable, reopens the v5 limits on curvature response, heading gain, line-turn
sensitivity, and target release, and fixes `maximum_racing_line_offset_ratio`
at the measured-safe 0.95. Generation 0 is the baked v5 winner.

The primary v6 branch ran for ten generations with optimizer seed 590119. It
retained 0.487 normalized pairwise diversity and mutation inflated to 0.324,
but no candidate beat the v5 anchor on any score component. It was stopped
there instead of spending the planned 40 generations in a demonstrably flat
box.

A seed-110 trace then identified what v6 did not search: 23 AVOID ticks capped
speed at 3.4 m/s and 77 more ticks invoked side slowdown. Relaxing only the
AVOID thresholds increased distance from 653.98 m to 663.16 m cleanly. Raising
the AVOID speed increased it to 658.41 m cleanly. A combined diagnostic reached
a 7.883 s lap, down from 7.917 s, with 0.067 s of wall contact. Aggressively
weakening side slowdown alone failed to finish a lap, so v7 searches these
guards inside conservative bounds.

V7 also introduces `lap-time`, a ranking objective that retains the existing
survival, lap-completion, and incident-budget tiers, then minimizes worst,
ninetieth-percentile, median, and mean best-lap time in that order. Penalized
distance is only the last tie-breaker. Run at most 30 generations and stop early
after ten or more flat generations:

```bash
mkdir -p artifacts/controller-search/faster-line-v7-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v7 \
  --optimizer ga --objective lap-time \
  --artifact-root artifacts/controller-search/faster-line-v7-ga \
  --population 64 --elites 12 --generations 30 \
  --optimizer-seed 590120 --workers 4 --evaluator-recycle-trials 120
```

V7 was stopped at generation 27. Its real final winner has a 7.833 s robust
worst lap, 7.817 s median, 7.812 s mean, 683.53 m penalized mean, and all 28
trials inside every hard tier. Generation 26 exposed a ranking precision bug:
lap times that differed by only about 5e-15 s were treated as materially
different before median time was considered. V8 rounds measured lap times to
six decimal places before ranking, seeds from generation 27, removes the two
speed-reduction genes that stayed at zero, and moves the throttle, curvature,
and wall thresholds v7 pressed.

```bash
mkdir -p artifacts/controller-search/faster-line-v8-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v8 \
  --optimizer ga --objective lap-time-v2 \
  --seed-checkpoint artifacts/controller-search/faster-line-v7-ga/generations/generation-027.json \
  --artifact-root artifacts/controller-search/faster-line-v8-ga \
  --population 64 --elites 12 --generations 30 \
  --optimizer-seed 590121 --workers 4 --evaluator-recycle-trials 120
```

The evaluator-object recycle does not return RSS to macOS. Restart the complete
command at generations 10 and 20; exact checkpoint resume releases worker
memory without changing optimizer state.

V8's training winner reached a 7.783 s worst and 7.767 s median best lap, but it
was not promotable. Two training seeds had small incidents, and the seed-110
gate exposed the underlying exploit: 0.246 damage, 1.133 s contact, 365 AVOID
ticks, a 21.65 s first lap, and only 379.88 m before it recovered and set one
7.800 s lap. V9 rejects that failure mode before comparing best-lap time: after
survival, lap, and incident tiers, it ranks three-lap completion and completely
clean trials. It restarts from v7 generation 27, whose training suite was clean.

```bash
mkdir -p artifacts/controller-search/faster-line-v9-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v9 \
  --optimizer ga --objective lap-time-v3 \
  --seed-checkpoint artifacts/controller-search/faster-line-v7-ga/generations/generation-027.json \
  --artifact-root artifacts/controller-search/faster-line-v9-ga \
  --population 64 --elites 12 --generations 30 \
  --optimizer-seed 590122 --workers 4 --evaluator-recycle-trials 120
```

V9 completed 30 generations. Generation 26 was the winner: all 28 training
trials were clean and completed at least three laps, with a 7.783 s robust
worst/median best lap and 686.56 m mean distance. The second official seed also
ran clean at 7.783 s, but seed 110 exposed a first-turn correction: the car
entered AVOID at 19.5 m/s, contacted the wall for 0.150 s, took 9.967 s for lap
one, and covered 645.01 m.

A default-off launch cap fixes that specific transient without slowing later
laps. At 19 m/s for only the first 3.5 s, the same seed-110 vector ran clean,
cut lap one to 8.667 s, set a 7.783 s best lap, and covered 679.47 m. V10 makes
the cap and duration searchable, reopens only the active corner/wall dynamics,
and evaluates the two official seeds alongside all 28 training seeds. Its v4
objective keeps v9's safety/three-lap/clean tiers, then ranks robust best-lap
time, robust first-lap time, and distance.

```bash
mkdir -p artifacts/controller-search/faster-line-v10-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v10 \
  --optimizer ga --objective lap-time-v4 \
  --seed-checkpoint artifacts/controller-search/faster-line-v9-ga/generations/generation-026.json \
  --artifact-root artifacts/controller-search/faster-line-v10-ga \
  --population 64 --elites 12 --generations 30 \
  --optimizer-seed 590123 --workers 4 --evaluator-recycle-trials 120
```

On this machine, restart the complete command after every five completed v10
generations: v9 workers reached roughly 2.0-3.1 GB each in a five-generation
pool lifetime. Exact checkpoint resume preserves the optimizer state.

V10 stopped at generation 7 because its ranking exposed a bad trade: a one-tick
repeated-lap gain outranked a 1.3 s slower first lap. That candidate remained
clean, but its 10.233 s worst first lap and 673.40 m mean were worse race pace
than generation 4's 8.917 s and 683.59 m. V11 uses the same measured search box
but ranks robust estimated three-lap time (`first_lap + 2 * best_lap`) before
separate repeated/first-lap components. Generation 4 is the seed: its composite
worst is 24.450 s, versus 24.583 s at generation 5 and 25.767 s at generation 7.

```bash
mkdir -p artifacts/controller-search/faster-line-v11-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v11 \
  --optimizer ga --objective lap-time-v5 \
  --seed-checkpoint artifacts/controller-search/faster-line-v10-ga/generations/generation-004.json \
  --artifact-root artifacts/controller-search/faster-line-v11-ga \
  --population 64 --elites 12 --generations 30 \
  --optimizer-seed 590124 --workers 4 --evaluator-recycle-trials 120
```

V11 stopped after generation 1. Its winner and the generation-4 parent both
have a worst estimated three-lap total of exactly 1,467 simulator ticks, but
rounding each constituent lap to six decimal seconds produced 24.449999 versus
24.450001 and selected the worse distribution. V12 converts every measured lap
to an integer 60 Hz tick count before summing or ranking; equal physical totals
therefore tie exactly.

```bash
mkdir -p artifacts/controller-search/faster-line-v12-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v12 \
  --optimizer ga --objective lap-time-v6 \
  --seed-checkpoint artifacts/controller-search/faster-line-v10-ga/generations/generation-004.json \
  --artifact-root artifacts/controller-search/faster-line-v12-ga \
  --population 64 --elites 12 --generations 30 \
  --optimizer-seed 590125 --workers 4 --evaluator-recycle-trials 120
```

V12 stopped after generation 1 because `--seed-checkpoint` restored only the
searched vector and ignored `checkpoint_context`. The resulting centre had the
right 16 genes but stale fixed values—for example, straight target 24.674
instead of 25.076 m/s and front-brake start 12.406 instead of 11.919 m—so it was
not the archived parent. The loader now merges fixed context first and searched
values second, with regression coverage. V13 is the clean restart.

```bash
mkdir -p artifacts/controller-search/faster-line-v13-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v13 \
  --optimizer ga --objective lap-time-v6 \
  --seed-checkpoint artifacts/controller-search/faster-line-v10-ga/generations/generation-004.json \
  --artifact-root artifacts/controller-search/faster-line-v13-ga \
  --population 64 --elites 12 --generations 30 \
  --optimizer-seed 590126 --workers 4 --evaluator-recycle-trials 120
```

V13 completed all 30 generations. Generation 22 won with every training and
official seed clean: its robust three-lap worst/median are 1,449/1,445 ticks
(24.150/24.083 s), versus 1,467/1,455.5 ticks for its v10 parent. A seed-110
trace then showed the car coasting through the only sustained 2.38 s sweeper.
Global exit-line offsets and slower line release were rejected because they
lost 2-126 m or caused contact. A default-off speed bonus, activated only after
a sustained same-direction turn, generalized: the measured 1.5 m/s bonus after
1.7 s with a 0.9 s hold kept all 30 seeds clean, improved the robust worst to
1,443 ticks (24.050 s), and raised mean distance from 688.57 to 692.37 m.

```bash
mkdir -p artifacts/controller-search/faster-line-v14-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v14 \
  --optimizer ga --objective lap-time-v6 \
  --seed-checkpoint artifacts/controller-search/faster-line-v13-ga/generations/generation-022.json \
  --artifact-root artifacts/controller-search/faster-line-v14-ga \
  --population 64 --elites 12 --generations 30 \
  --optimizer-seed 590127 --workers 4 --evaluator-recycle-trials 120
```

V14 stopped after generation 15 because generations 6-15 were exactly flat.
Generation 5 won with a 2.148 s sustained-turn activation, 0.344 s release
hold, and 2.109 m/s speed bonus. It remained clean on all 30 search seeds and
improved robust worst three-lap time to 1,439 ticks (23.983 s), ten ticks faster
than v13 and 28 ticks faster than the complete v10 parent. The baked controller
then passed 2/2 official, 12/12 validation, and 100/100 clean soak trials plus
the 20-race head-to-head gate at 18 wins. Its best official lap is 7.650 s.


### V15 - sweeper entry preview

```bash
mkdir -p artifacts/controller-search/faster-line-v15-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v15 \
  --optimizer ga --objective lap-time-v6 \
  --seed-checkpoint artifacts/controller-search/faster-line-v14-ga/generations/generation-005.json \
  --artifact-root artifacts/controller-search/faster-line-v15-ga \
  --population 64 --elites 12 --generations 40 \
  --optimizer-seed 590128 --workers 4 --evaluator-recycle-trials 120
```

Stopped after generation 37; generations 28-37 were exactly flat. Generation 27
holds a 0.1067-0.1454 far-curvature window, a 1.151 s hold, and a 1.530 m/s
bonus, clean on all 30 seeds at 1,432 robust worst ticks.

### V16 - reopening the pinned launch floor

V13's twelve elites pinned `startup_speed_cap_seconds` at exactly 2.5000, the
floor of v10's box. V16 reopens that gene below its own floor and nothing else.

```bash
mkdir -p artifacts/controller-search/faster-line-v16-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v16 \
  --optimizer ga --objective lap-time-v6 \
  --seed-checkpoint artifacts/controller-search/faster-line-v15-ga/generations/generation-027.json \
  --artifact-root artifacts/controller-search/faster-line-v16-ga \
  --population 64 --elites 12 --generations 30 \
  --optimizer-seed 590129 --workers 4 --evaluator-recycle-trials 120
```

Stopped after generation 20; generations 11-20 were exactly flat. Generation 10
caps the launch at 22.774 m/s for 1.993 s, improving robust worst three-lap time
to 1,427 ticks and worst first lap from 518 to 513 ticks. **This is the promoted
vector**: 7.617 s official best lap, 100/100 clean soak, 18/20 head-to-head.

### V17 - joint re-ranking, falsified

The three-lap objective weights repeated pace twice, so `lap-time-v7` was added
to rank worst first+best ticks ahead of either component. V17 reopens all six
preview and launch genes jointly under it.

```bash
mkdir -p artifacts/controller-search/faster-line-v17-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v17 \
  --optimizer ga --objective lap-time-v7 \
  --seed-checkpoint artifacts/controller-search/faster-line-v16-ga/generations/generation-010.json \
  --artifact-root artifacts/controller-search/faster-line-v17-ga \
  --population 64 --elites 12 --generations 30 \
  --optimizer-seed 590130 --workers 4 --evaluator-recycle-trials 120
```

Stopped after generation 10 with **zero** improvement in any generation: the
seeded v16 incumbent won generation 1 and was never beaten. Re-ranking existing
genes moves nothing, so later versions must add an unsearched lever or reopen a
binding bound rather than re-mutate the same box.

### V18 - the corner-exit speed bonus

A new default-off, pose-invariant lever: accelerate as near curvature unwinds
relative to far curvature, instead of waiting for the whole speed scalar to fall
back toward straight-line pace. One gene, so the population is halved.

```bash
mkdir -p artifacts/controller-search/faster-line-v18-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v18 \
  --optimizer ga --objective lap-time-v7 \
  --seed-checkpoint artifacts/controller-search/faster-line-v16-ga/generations/generation-010.json \
  --artifact-root artifacts/controller-search/faster-line-v18-ga \
  --population 32 --elites 8 --generations 30 \
  --optimizer-seed 590131 --workers 5 --evaluator-recycle-trials 120
```

Stopped after generation 11; generations 2-11 were exactly flat. **Every**
generation selected `corner_exit_target_speed_bonus_mps = 0.0` - the default-off
value - so no positive bonus anywhere in the 0.0-3.0 m/s range beat switching the
lever off. The hypothesis is rejected, not merely untuned. The parameter stays in
`ControllerParameters` at its 0.0 default and costs nothing at runtime.

### V19 - reopening the last two pinned bounds

V13's elites pinned two more genes exactly against their box: the winning
`corner_target_speed_mps` sat on the 14.0 m/s floor and `front_stop_m` on the
1.60 m ceiling. Both shape the same corner approach, so v19 searches them
together, releasing the corner target down to 11.0 m/s and the front stop up to
2.60 m - back toward the 2.5-3.0 m ceilings v5-v7 used before v10 narrowed the
box. This is the same signature v16 converted into twelve ticks.

```bash
mkdir -p artifacts/controller-search/faster-line-v19-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v19 \
  --optimizer ga --objective lap-time-v7 \
  --seed-checkpoint artifacts/controller-search/faster-line-v16-ga/generations/generation-010.json \
  --artifact-root artifacts/controller-search/faster-line-v19-ga \
  --population 48 --elites 10 --generations 30 \
  --optimizer-seed 590132 --workers 5 --evaluator-recycle-trials 120
```

Stopped after generation 16; generations 7-16 were exactly flat. It improved in
**six of its first six generations**. Generation 6 settles at
`corner_target_speed_mps=13.744` and `front_stop_m=2.037` - both outside the box
v10 gave them, which is the direct confirmation that the bounds, not the
controller, were the limit. Robust worst best lap went 461 to **457 ticks**
(7.683 to 7.617 s), and mean best-lap ticks hit exactly 457.0, so every one of
the 30 search seeds reaches the same best lap. **This is the promoted vector**:
100/100 clean soak, 19/20 head-to-head, and better than baked v16 on official,
validation, and soak distance with no regression anywhere.

### V20 - the frozen first lap

Worst first-lap time has read 513 ticks in v16, v17, v18, and v19 alike. The
launch genes are the only family that has ever moved it (v16 took it from 518),
and v16's own box is now partly pinned: four of twelve elites finished on the
1.85 s floor and the winning cap sits within a tenth of the 22.9 m/s ceiling.
V20 reopens that box on both sides while holding v19's corner genes fixed.

```bash
mkdir -p artifacts/controller-search/faster-line-v20-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v20 \
  --optimizer ga --objective lap-time-v7 \
  --seed-checkpoint artifacts/controller-search/faster-line-v19-ga/generations/generation-006.json \
  --artifact-root artifacts/controller-search/faster-line-v20-ga \
  --population 48 --elites 10 --generations 30 \
  --optimizer-seed 590133 --workers 5 --evaluator-recycle-trials 120
```

Stopped after generation 11 with **zero** improvement: all eleven generations kept
the seeded 22.774 m/s cap held 1.993 s, and nothing in the widened box beat it.
The partial pin that motivated the run - four of twelve v16 elites on the 1.85 s
floor - did not reproduce as a real bound. Treat 513 ticks as this controller's
first-lap floor and stop attacking the launch.

Note that a bound sitting at a *physical* limit is not a lever either. Three line
genes read as pinned (`maximum_racing_line_offset_ratio` and
`racing_line_entry_offset_ratio` at 0.95, `racing_line_exit_offset_ratio` at 0.0),
but `center_offset_cap_m` is 3.3 m - exactly the corridor half-width - so 0.95
already requests a line 3.135 m out and 1.0 is the wall itself. Check what a bound
means physically before reopening it.

### V21 - the straight-speed ceiling, retested

D-040 rejected 26-27 m/s straight targets for causing wall contact. That was
measured against the old corner approach: a 14.0 m/s corner target braking to a
1.60 m front stop. V19 replaced both with 13.744 m/s and 2.037 m, so the car now
arrives slower and brakes earlier, and the rejection's premise no longer holds.
V21 retests the ceiling with the brake ramp free to start beyond v4's 14.0 m.

```bash
mkdir -p artifacts/controller-search/faster-line-v21-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v21 \
  --optimizer ga --objective lap-time-v7 \
  --seed-checkpoint artifacts/controller-search/faster-line-v19-ga/generations/generation-006.json \
  --artifact-root artifacts/controller-search/faster-line-v21-ga \
  --population 48 --elites 10 --generations 30 \
  --optimizer-seed 590134 --workers 5 --evaluator-recycle-trials 120
```

V21 did move - generation 17 improved worst first+best from 970 to 969 ticks, and
generation 19 reached mean best-lap 456.93, so at least one seed broke the 457
floor - but it also exposed a defect in the ranking itself.

## The v7 ranking defect, and `lap-time-v8`

`lap_time_score_v7` ranks `max(first+best)`, then `max(first)`, then `max(best)`.
At v21 generation 11 the search moved 513+457 to 512+458. Both sum to 970, so the
primary key tied and the comparison fell through to first lap, where 512 beats
513 - while best lap quietly got worse. Best lap is the third key and is never
consulted once an earlier key breaks the tie.

With the sum pinned, that makes the two laps a zero-sum trade the search will
happily take: it banks a first-lap tick by spending a best-lap tick, every time,
and scores each trade as an improvement. Symptom to watch for: first lap creeping
down while best lap creeps up, with the sum never moving.

`lap_time_score_v8` leads with `max(best)`, then `max(first)`, then the sum. Best
lap is also the better primary on its own terms - a thirty-second trial runs
three laps, so a repeated lap counts twice against the first lap once.
`test_lap_time_score_v8_refuses_to_trade_best_lap_for_first_lap` pins the exact
513+457 versus 512+458 pair so the regression cannot return.

### V22 - the whole speed profile, re-searched under v8

```bash
mkdir -p artifacts/controller-search/faster-line-v22-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v22 \
  --optimizer ga --objective lap-time-v8 \
  --seed-checkpoint artifacts/controller-search/faster-line-v19-ga/generations/generation-006.json \
  --artifact-root artifacts/controller-search/faster-line-v22-ga \
  --population 64 --elites 12 --generations 40 \
  --optimizer-seed 590135 --workers 5 --evaluator-recycle-trials 120
```

Stopped after generation 11 with **zero** improvement: all four pace genes stayed
exactly at the seeded v19 values, and mean best-lap was exactly 457.0, meaning all
30 seeds hit the same lap. Under a ranking that cannot trade best lap away, the
speed profile has nothing left. 457 ticks is a floor of the policy, not of the
search.

### V23 - line timing, the last untouched dimension

Speed, launch, and line magnitude are each now measured as exhausted. Magnitude is
genuinely capped: `center_offset_cap_m` is the 3.3 m corridor half-width, so the
0.95 clamp already requests 3.135 m and 1.0 is the wall. Timing is not capped, and
two of its genes finished exactly on a bound - `line_turn_sensitivity` on its
0.002 floor, meaning the search wanted an even sharper response than a box that
already commands full offset for the gentlest curvature, and
`line_target_release_per_tick` on its 0.25 ceiling, meaning it wanted the target
to relax faster than allowed.

```bash
mkdir -p artifacts/controller-search/faster-line-v23-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v23 \
  --optimizer ga --objective lap-time-v8 \
  --seed-checkpoint artifacts/controller-search/faster-line-v19-ga/generations/generation-006.json \
  --artifact-root artifacts/controller-search/faster-line-v23-ga \
  --population 48 --elites 10 --generations 30 \
  --optimizer-seed 590136 --workers 5 --evaluator-recycle-trials 120
```

When chaining presets this deep, check that the fixed context is complete before
launching: each preset deletes the genes it searches, so a later preset that
searches none of them must add every one back or the run silently reverts it to a
base default. `test_faster_line_v23_reopens_the_pinned_line_timing_bounds` caught
exactly that, which is the same failure that cost v12 its run.

Stopped after generation 14; generations 5-14 were exactly flat. Generation 4
reached `line_turn_sensitivity` 0.00055, **below the old 0.002 floor**, which
confirms that floor was binding, and broke the 457-tick uniformity for the first
time: mean best-lap fell from 457.000 to 456.933, meaning one of the thirty seeds
reached 456 ticks (7.600 s).

It was **not promoted.** The gain is one tick on one seed of thirty; it sits below
every worst-case key in the ranking (best lap 457, first lap 513, first+best 970
are all unchanged); and mean distance regressed from 700.465 to 700.30 m. That
does not justify re-running the promotion gates, so `race_faster` still carries
v19 generation 6. The checkpoint is retained if the direction is picked up again -
note that `line_target_release_per_tick` barely moved (0.2500 to 0.2495), so
sensitivity is the live half of the pair and deserves a run of its own.

### V24 - startup brake-turn drift and the global speed ceiling

V24 tested the seed-110 first-turn idea directly: while the car is still in its
startup window, sufficient steering and a close front return can trigger one
short negative-throttle pulse. The requested steering is held through the pulse,
then the existing neutral latch-release tick restores drive. The new behavior is
default-off, so a seeded incumbent remains exactly reproducible.

```bash
mkdir -p artifacts/controller-search/faster-line-v24-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v24 \
  --optimizer ga --objective lap-time-v8 \
  --seed-checkpoint artifacts/controller-search/faster-line-v19-ga/generations/generation-006.json \
  --artifact-root artifacts/controller-search/faster-line-v24-ga \
  --population 48 --elites 10 --generations 10 \
  --optimizer-seed 590137 --workers 5 --evaluator-recycle-trials 120
```

The standalone pulse was promising on seed 110, moving the first lap from 509
to 501 ticks, but the joint global-speed/drift search could not improve the
30-seed robust score. All ten generations stayed flat and selected the disabled
anchor, so the run was closed under the plateau rule. A global 30 m/s target was
also unsafe in direct probes. `speed_cap_mps` is only sensor normalization; it
does not constrain vehicle speed, so raising it cannot make the car faster.

### V25 - local long-corridor boost and reusable drift

V25 replaced the global target with a bounded, geometry-triggered boost. A
pose-invariant near/far-curvature check recognizes a sustained straight, adds a
local target-speed bonus for a short window, then arms the same brake-turn pulse
for the next hard corner. This carries more speed down a long corridor without
leaving the higher target active around the whole lap.

It also introduces `lap_time_score_v9`. V9 keeps survival, a completed lap, and
three timed laps as hard requirements, but expands the search incident budget
from 0.25 to 0.50 damage and from 1.5 to 2.0 contact seconds. Within that bounded
budget it ranks best-lap distribution first, then first lap and their sum. This
lets a genuinely faster low-contact line compete during mutation; clean trials
still win later tie-breaks, and promotion retains the stricter clean gates.

```bash
mkdir -p artifacts/controller-search/faster-line-v25-ga
nocorrect caffeinate -ims uv run python -m scripts.controller_training.search faster-line-v25 \
  --optimizer ga --objective lap-time-v9 \
  --seed-checkpoint artifacts/controller-search/faster-line-v19-ga/generations/generation-006.json \
  --artifact-root artifacts/controller-search/faster-line-v25-ga \
  --population 48 --elites 10 --generations 10 \
  --optimizer-seed 590138 --workers 5 --evaluator-recycle-trials 120
```

Generation 10 won with a 0.555 m/s corridor target bonus held for 0.236 s and a
0.412 negative-throttle pulse held for about five ticks. It was nevertheless
fully clean on all 30 search trials. Against v19, mean best lap improved from
457.000 to 455.633 ticks, mean first lap from 507.10 to 505.57 ticks, and mean
distance from 700.465 to 702.384 m. The official seeds both reached 456-tick
(7.600 s) best laps; the 100-seed soak remained 100/100 clean and measured a
27.215 m/s peak. This generation is baked into `controllers.race_faster`.

### V26 full-throttle boundary from the human trace

`artifacts/human-driving.jsonl` reached 36.444 m/s instantaneously and 33.695
m/s over its best rolling one-second window. Its useful pattern was a 1.283 s
full-throttle corridor segment followed by a coast before turning. The two long
`-1.0` brake-turn pulses were not copied: they shed roughly 12-14 m/s, whereas
v25's five-tick `-0.412` pulse loses less than 1 m/s.

V26 reopened the long-straight target and hold parameters under a speed-first
objective on the five live Gradescope seeds:

```bash
uv run python -m scripts.controller_training.search faster-line-v26 \
  --optimizer ga --objective speed-max-v1 \
  --seed-checkpoint artifacts/controller-search/faster-line-v25-ga/generations/generation-010.json \
  --artifact-root artifacts/controller-search/faster-line-v26-ga \
  --population 48 --elites 10 --generations 16 \
  --optimizer-seed 590139 --workers 5 --evaluator-recycle-trials 120
```

Generation 6 won and generations 7-16 were exactly flat. It reaches
34.636-36.401 m/s peak across the five seeds. Validation with the live grading
worker completed 2-3 laps on every seed and measured a 33.587 m/s mean of each
seed's best rolling one-second speed, up from 26.309 m/s for `race_faster`.
Damage ranged from 0.151 to 0.763, which is deliberately accepted by this
specialized variant. The target-speed bonus finished on its 25 m/s ceiling, but
the requested target is already about 50 m/s and therefore holds the normalized
throttle command at its legal `1.0` ceiling. Raising the target again cannot add
motor command; the next limit is usable corridor duration.

The winner is baked into `controllers.race_speedmax`, leaving the clean
lap-time controller unchanged.

### Finding the next revision

When a run stops on the plateau rule, do not extend it and do not reach for a new
lever first. Ask the closed run which of its own bounds it was pushing against:
a gene whose winner sits *exactly* on its floor or ceiling is measured evidence
that the box, not the controller, is the limit.

```bash
PYTHONPATH="$PWD" uv run python - \
  artifacts/controller-search/faster-line-v13-ga/checkpoint.json faster-line-v13 <<'EOF'
import json, sys
from scripts.controller_training.search import preset_configuration
checkpoint, preset = sys.argv[1], sys.argv[2]
d = json.load(open(checkpoint))
names, elites, win = d["parameter_names"], d["elite_values"], d["best_parameter_vector"]
spec = {s.name: s for s in preset_configuration(preset)[1].specs}
for i, n in enumerate(names):
    s, w = spec[n], win[n]
    lo = sum(1 for row in elites if abs(row[i] - s.minimum) <= 1e-9)
    hi = sum(1 for row in elites if abs(row[i] - s.maximum) <= 1e-9)
    flag = ("WINNER ON FLOOR" if abs(w - s.minimum) <= 1e-9 else
            "WINNER ON CEILING" if abs(w - s.maximum) <= 1e-9 else "")
    print(f"{n:38s} [{s.minimum:8.3f},{s.maximum:8.3f}] win {w:9.4f} "
          f"elites@floor {lo:2d} @ceil {hi:2d}  {flag}")
EOF
```

On v13's closed checkpoint this prints four pinned genes, and the record since is
unambiguous:

| Pinned gene | Reopened by | Result |
| --- | --- | --- |
| `startup_speed_cap_seconds` on its 2.5 s floor (8/12 elites) | v16 | Landed at 1.993 s; **+12 ticks**, promoted |
| `corner_target_speed_mps` on its 14.0 m/s floor | v19 | Landed at 13.647 m/s, outside the old box |
| `front_stop_m` on its 1.60 m ceiling | v19 | Landed at 1.939 m, outside the old box |
| `avoid_side_wall_m` on its 0.60 m floor (4/12 elites) | *deliberately not reopened* | D-033 rejected 0.50 as the crash-then-sprint optimum; this floor is a safety constraint, not a search artifact |

The two revisions that instead re-ranked existing genes (v17) or added a brand
new lever (v18) each returned **exactly zero** over ten-plus generations. Reopen a
measured bound before inventing a mechanism, and never reopen a bound that an
earlier decision row put there on purpose.

### Stopping rule

`search` has no built-in plateau stop; every generation is archived and the run
resumes from its checkpoint, so campaigns are stopped by hand. Stop a run once
ten consecutive generations pass without the archived `best_score` improving,
then revise rather than extend. Check the current count with:

```bash
uv run python - artifacts/controller-search/faster-line-v19-ga <<'EOF'
import json, sys, glob, os
root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "generations", "*.json")))
gens = [json.load(open(f)) for f in files]
best, last = tuple(gens[0]["best_score"]), gens[0]["generation"]
for g in gens[1:]:
    if tuple(g["best_score"]) > best:
        best, last = tuple(g["best_score"]), g["generation"]
print(f"latest={gens[-1]['generation']} last_improved={last} flat={gens[-1]['generation'] - last}")
EOF
```

## Bake the best vector into source

```bash
# Latest checkpoint of the current run
uv run python -m scripts.controller_training.bake

# Or any specific checkpoint / archived generation
uv run python -m scripts.controller_training.bake \
  artifacts/controller-search/faster-line/generations/generation-030.json
```

For the v2 GA winner, pass the final branch checkpoint and v2 preset:

```bash
uv run python -m scripts.controller_training.bake \
  artifacts/controller-search/faster-line-v2-ga/checkpoint.json \
  --preset faster-line-v2
```

Use `faster-line-v2-ga-soft/checkpoint.json` instead if the softened branch was
selected. The GA checkpoint stores the preview/wall compensation values fixed
from the CEM probe, so the bake output includes them even though they are not in
the final 17-gene vector.

It prints a ready-to-paste `RACE_FASTER_PARAMETERS` block and separates the
values this search actually tuned from ones carried over from an earlier search,
so parameters outside the current space (`brake_gain`, `side_speed_floor`, the
competitor gains) are not silently dropped. It only prints — it never edits
source.

Paste the block over the one in `src/controllers/race_faster.py`, then verify
with the suite commands below.

## Evaluate and gate

```bash
# Fixed suites: official (2), validation (12), training (28), soak (100)
uv run python -m scripts.controller_training.suite controllers.race_faster --suite official
uv run python -m scripts.controller_training.suite controllers.minimum_viable --suite validation

# Arbitrary seeds, and save the record
uv run python -m scripts.controller_training.suite controllers.race_faster \
  --seeds 110 2026 --output artifacts/controller-search/faster-line/official.json

# One trial, printed as JSON
uv run python -m scripts.controller_training.evaluator controllers.race_faster --seed 110 --seconds 30.0
```

Promotion gates from `plan.md`:

- `minimum_viable`: on all 14 official and validation seeds — survives, completes
  a lap, **zero** damage and **zero** wall contact.
- `race_faster`: survives and completes a lap on every validation seed, beats
  `minimum_viable` on both official seeds, improves median distance by 10% and
  tenth-percentile by 5%, and wins at least 12 of 20 role-swapped head-to-head
  races.

### Head-to-head

Solo trials have no competitors, so the search cannot see passing behavior at
all. Check it separately:

```bash
uv run python -c "
from racing.race.head_to_head import run_headless_head_to_head
from racing.student.api import load_student_controller
r = run_headless_head_to_head(
    challenger_controller=load_student_controller('controllers.race_faster'),
    incumbent_controller=load_student_controller('controllers.minimum_viable'),
    race_count=5, round_seconds=30.0, random_seed=110)
for x in r.races:
    print(x.race_index, x.winner, round(x.challenger.raw_distances_m[0],1), round(x.incumbent.raw_distances_m[0],1))
"
```

Swap the two controllers and re-run for the other role. Ties count against the
12-of-20 gate.

## Verification before committing

```bash
uv run pytest -q
uv run pyright
uv run ruff check src/controllers scripts/controller_training tests
uv run ruff format --check src/controllers scripts/controller_training tests
```

Whole-tree Ruff has unrelated baseline findings, so keep lint scoped to touched
paths and do not run a broad `--fix`.

## Artifact layout

```
artifacts/controller-search/
  seed-manifest.json              the fixed seed suites
  <preset>/
    checkpoint.json               latest generation; resume reads this
    generations/generation-NNN.json   immutable per-generation archive
    *.json                        suite records you saved with --output
```

`artifacts/` is git-ignored. Deleting a preset directory discards that run's
history, so keep it until the parameters are baked and gated.
