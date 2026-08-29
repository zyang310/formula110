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
| `faster-line-v3` | 16 parameters; unpins the five v2 bounds and adds the searchable line clamp | `improved_score` with GA | Ready, not yet run |

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
