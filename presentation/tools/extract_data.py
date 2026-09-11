#!/usr/bin/env python3
"""Rebuild presentation/data/deck-data.json from files in this repository.

From the repository root:

  PYTHONPATH=. uv run python presentation/tools/extract_data.py [--paths FILE] [--check]

Reads the search checkpoints in artifacts/controller-search/, the agent summary
in presentation/data/agent-usage.json, and the track geometry from racing.
Trajectories come from --paths (the output of capture_paths.py); without it the
trajectories already in deck-data.json are kept. Checkpoint times come from
presentation/data/checkpoint-times.json because git does not keep file
modification times; checkpoints missing from it are added from their mtime.
--check compares against the current deck-data.json instead of overwriting it.
"""

import argparse
import bisect
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "presentation" / "data"
SEARCH = REPO / "artifacts" / "controller-search"
TIMES_FILE = DATA / "checkpoint-times.json"
DECK = DATA / "deck-data.json"

# Promoted versions, in order; their running best median lap is the chart's curve.
LINEAGE = [
    "faster", "faster-line", "faster-line-v2-ga", "faster-line-v3-ga", "faster-line-v4-ga", "faster-line-v5-ga",
    "faster-line-v7-ga", "faster-line-v9-ga", "faster-line-v13-ga", "faster-line-v14-ga", "faster-line-v15-ga",
    "faster-line-v16-ga", "faster-line-v19-ga", "faster-line-v25-ga", "faster-line-v27b-ga",
]
REJECTED = [
    ("faster-line-v8-ga", "crash-then-sprint"),
    ("faster-line-v10-ga", "slow launch"),
    ("faster-line-v30-ga-neutral-round3", "human maneuver: safe, slower"),
]
CEM_TRIALS = {"faster": 720, "faster-line": 720, "faster-line-v2-cem-gate": 720}  # 64 x 6 seeds + 12 elites x 28
MINIMUM_TRIALS = 20 * 512  # minimum preset: 20 CEM generations of 48 x 6 + 8 x 28
FASTER_UNARCHIVED = 19 * 720  # generations 1-19 of `faster` ran before per-generation archiving existed
HAND_TUNED_LAP = 28.2  # plan.md C-001: hand-tuned first laps 28.20-29.12 s


def short(name: str) -> str:
    fixed = {"minimum": "minimum CEM", "faster": "CEM · faster", "faster-line": "CEM · faster-line"}
    if name in fixed:
        return fixed[name]
    name = name.replace("faster-line-", "").replace("-ga", "")
    return "v27" if name == "v27b" else name


def median_best(results):
    if not results:
        return None
    laps = [r["best_lap_time_seconds"] for r in results if r.get("best_lap_time_seconds") is not None]
    return dict(n=len(results), med_best=round(statistics.median(laps), 4) if laps else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", type=Path, help="trajectory JSON written by capture_paths.py")
    parser.add_argument("--check", action="store_true", help="compare with deck-data.json instead of writing it")
    args = parser.parse_args()

    times = json.loads(TIMES_FILE.read_text()) if TIMES_FILE.exists() else {}

    def mtime(path: Path) -> float:
        key = path.relative_to(SEARCH).as_posix()
        if key not in times:
            times[key] = os.path.getmtime(path)
        return times[key]

    old = json.loads(DECK.read_text()) if DECK.exists() else {}
    usage = json.loads((DATA / "agent-usage.json").read_text(encoding="utf-8"))

    # ---- one row per archived generation, branches in the order they started
    branches = []
    for d in SEARCH.iterdir():
        files = sorted((d / "generations").glob("*.json")) if d.is_dir() else []
        if files:
            branches.append((min(mtime(f) for f in files), d.name))
    rows = []
    for _, name in sorted(branches):
        incumbent = None
        for f in sorted((SEARCH / name / "generations").glob("generation-*.json")):
            g = json.loads(f.read_text())
            m = g.get("metrics", {})
            if m.get("new_overall_best_results"):
                incumbent = median_best(m["new_overall_best_results"])
            rows.append(dict(
                preset=name, gen=g.get("generation"), t=round(mtime(f)), sel=len(m.get("selection_seeds") or []),
                gen_best=median_best(m.get("generation_best_results") or []), incumbent=incumbent,
                config=g.get("optimizer_config") or {},
            ))

    # ---- simulated races over time
    minimum_t = mtime(SEARCH / "minimum" / "checkpoint.json")
    trial_events = [(minimum_t, MINIMUM_TRIALS)]
    first_faster = True
    for r in rows:
        if r["preset"] in CEM_TRIALS:
            n = CEM_TRIALS[r["preset"]]
            if r["preset"] == "faster" and first_faster:
                n += FASTER_UNARCHIVED
                first_faster = False
        else:
            c = r["config"]
            n = (c.get("population_size") or 0) * r["sel"] + (c.get("elite_count") or 0) * ((r["gen_best"] or {}).get("n") or 0)
        trial_events.append((r["t"], n))
    trial_events.sort()
    trial_times, trial_totals, running = [], [], 0
    for t, n in trial_events:
        running += n
        trial_times.append(t)
        trial_totals.append(running)

    def trials_at(t: float) -> int:
        i = bisect.bisect_right(trial_times, t)
        return trial_totals[i - 1] if i else 0

    # ---- promoted curve and rejected markers
    points = [
        dict(t=round(mtime(SEARCH / "smoke" / "checkpoint.json")), lap=HAND_TUNED_LAP, v="hand-tuned"),
        dict(t=round(minimum_t), lap=23.75, v=short("minimum"), g=None),
    ]
    best = 23.75
    for r in rows:
        inc = r["incumbent"] or r["gen_best"]
        if r["preset"] in LINEAGE and inc and inc["med_best"] is not None and inc["med_best"] < best - 1e-9:
            best = inc["med_best"]
            points.append(dict(t=r["t"], lap=round(best, 4), v=short(r["preset"]), g=r["gen"]))
    rejected = []
    for preset, label in REJECTED:
        last = [r for r in rows if r["preset"] == preset][-1]
        inc = last["incumbent"] or last["gen_best"]
        rejected.append(dict(t=last["t"], lap=inc["med_best"], v=short(preset), label=label))

    s = usage["summary"]
    journey = dict(
        points=points,
        rejected=rejected,
        prompts=[dict(p, trials=trials_at(p["t"])) for p in usage["prompts"]],
        chats=usage["chats"],
        segs=usage["segs"],
        tokmap=usage["tokmap"],
        trialmap=[[round(t), n] for t, n in zip(trial_times, trial_totals)],
        summary=dict(
            tokens_total_M=s["tokens_total_M"], fresh_M=s["fresh_M"], out_M=s["out_M"], codex_M=s["codex_M"],
            claude_M=s["claude_M"], reviewer_M=s["reviewer_M"], trials_total=trial_totals[-1],
            sim_days=round(trial_totals[-1] * 30 / 86400, 1), t_first=s["t_first"], t_last=s["t_last"], generations=len(rows),
        ),
    )

    # ---- CEM ellipses (v2 gate) and GA population (v19), with their preset bounds
    from scripts.controller_training.search import preset_configuration

    def bounds(preset: str) -> dict:
        return {spec.name: [spec.minimum, spec.maximum] for spec in preset_configuration(preset)[1].specs}

    genes = ["center_steer_gain", "racing_line_entry_offset_ratio"]
    probe = bounds("faster-line-v2-probe")
    cem = []
    for f in sorted((SEARCH / "faster-line-v2-cem-gate" / "generations").glob("*.json")):
        d = json.loads(f.read_text())
        idx = [d["parameter_names"].index(g) for g in genes]
        vec = d["best_parameter_vector"]
        cem.append(dict(
            gen=d["generation"], mean=[d["distribution_mean"][i] for i in idx], dev=[d["distribution_deviation"][i] for i in idx],
            best=[vec[g] if isinstance(vec, dict) else vec[i] for g, i in zip(genes, idx)],
        ))
    v19, v13 = bounds("faster-line-v19"), bounds("faster-line-v13")
    ga = []
    for f in sorted((SEARCH / "faster-line-v19-ga" / "generations").glob("*.json")):
        d = json.loads(f.read_text())
        names = d["parameter_names"]
        ga.append(dict(
            gen=d["generation"], pop=[[round(v, 4) for v in row] for row in d["population"]],
            elites=[[round(v, 4) for v in row] for row in d["elite_values"]], best=[d["best_parameter_vector"][n] for n in names],
        ))
    search = dict(
        cem=dict(genes=genes, bounds=[probe[g] for g in genes], gens=cem),
        ga=dict(genes=names, bounds=[v19[n] for n in names], old_bounds=[v13.get(n) for n in names], gens=ga),
    )

    # ---- track and trajectories
    from racing.track import world

    track = dict(c=[[round(p.x, 2), round(p.z, 2)] for p in world.sampled_track_centerline()], half=world.TRACK_WIDTH / 2)
    if args.paths:
        paths = {}
        for key, v in json.loads(args.paths.read_text()).items():
            tick_rows = v["rows"]
            xy = []
            for r in tick_rows[::2]:
                xy += [round(r[0], 2), round(r[1], 2), round(r[2], 1)]
            paths[key] = dict(xy=xy, crossings=v["crossings"], best=v["best_lap"], first=v["first_lap"], dist=v["distance_m"],
                              damage=v["damage"], contact=v["wall_contact_s"], n=len(tick_rows))
            if key in ("v27", "human"):
                paths[key]["thr"] = [round(r[4] * 100) for r in tick_rows]
                paths[key]["st"] = [round(r[3] * 100) for r in tick_rows]
    else:
        paths = old["paths"]

    data = dict(track=track, paths=paths, journey=journey, search=search)
    TIMES_FILE.write_text(json.dumps(times, indent=1, sort_keys=True) + "\n")
    if args.check:
        problems = differences(json.loads(json.dumps(data)), old)
        print("deck-data.json is up to date" if not problems else "differences:\n  " + "\n  ".join(problems))
        sys.exit(1 if problems else 0)
    DECK.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {DECK.relative_to(REPO)}: {len(rows)} generations, {len(points)} curve points")


def differences(new, old, path="", found=None):
    found = [] if found is None else found
    if len(found) >= 12:
        return found
    if isinstance(new, dict) and isinstance(old, dict):
        for k in sorted(set(new) | set(old)):
            if k not in new or k not in old:
                found.append(f"{path}/{k}: only in {'new' if k in new else 'old'}")
            else:
                differences(new[k], old[k], f"{path}/{k}", found)
    elif isinstance(new, list) and isinstance(old, list):
        if len(new) != len(old):
            found.append(f"{path}: {len(new)} items vs {len(old)}")
        else:
            for i, (a, b) in enumerate(zip(new, old)):
                differences(a, b, f"{path}[{i}]", found)
    elif new != old:
        found.append(f"{path}: {new!r} vs {old!r}")
    return found


if __name__ == "__main__":
    main()
