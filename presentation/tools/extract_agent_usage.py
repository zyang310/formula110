#!/usr/bin/env python3
"""Summarize this project's Codex and Claude Code usage into data/agent-usage.json.

Runs only where the agent logs live (Zhi's laptop): it reads ~/.codex and
~/.claude. The output holds token counts, chat names and times - no transcript
content beyond the five prompts quoted on the slides, which are fixed below.

  python3 presentation/tools/extract_agent_usage.py
"""

import bisect
import datetime as dt
import glob
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "agent-usage.json"
CODEX = Path.home() / ".codex"
CLAUDE = Path.home() / ".claude" / "projects" / "-Users-zhihangyang-Code-Class-Work-COMP-590H-formula110"
PROJECT = "formula110"
CODEX_EXCLUDE = {
    "01a063ad-a791-7881-9a2e-72cbf6a63657",  # Fix TargetGame manager script (another project)
    "01a0653d-e9d7-7ef2-a27d-284b70904564",  # Explore memory error visualizer (another project)
    "01a08d2e-f452-7913-bdfc-96554d898036",  # Draft visual project presentation
}
CLAUDE_NAMES = {
    "7f63a75b-58c2-477f-ab50-09d9febf8cd0": "Looser bounds, outside–inside line",
    "c518ba47-b235-457d-9072-886bfaa929eb": '"Roadblock implementing GA"',
    "7a68c1ae-b84b-4565-8298-a997d501b4e6": "GA goal run (v19–v23)",
    "bf77c56f-6817-4783-a2e2-548ac3f7b36d": '"Break down the approach"',
}
PROMPTS = [
    ("2026-08-28T18:11:00Z", "Is there a way to remove the braking? I feel like instead of breaking we just won’t accelerate"),
    ("2026-08-29T19:13:00Z", "Aren’t we just wasting compute power right now"),
    ("2026-08-31T15:20:00Z", "Think about drifting… we can turn without losing all of our speed"),
    ("2026-08-31T22:34:00Z", "check human-driving.jsonl … I tried to drift a little bit"),
    ("2026-09-01T23:18:00Z", "I think I had good instances of drifting… the large curve near the end"),
]


def ep(stamp: str) -> float:
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


# Sessions after the controller work ended (building this presentation) are not counted.
CUTOFF = ep("2026-09-05T00:00:00Z")


def codex_events():
    names = {}
    for line in open(CODEX / "session_index.jsonl", encoding="utf-8"):
        try:
            d = json.loads(line)
            names[d["id"]] = d["thread_name"]
        except Exception:
            continue
    events = []
    for path in sorted(glob.glob(str(CODEX / "sessions" / "**" / "*.jsonl"), recursive=True)):
        fid = os.path.basename(path)[-42:-6]
        lines = open(path, encoding="utf-8").readlines()
        meta = None
        for line in lines:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") == "session_meta":
                meta = d["payload"]
                break
        if not meta or PROJECT not in (meta.get("cwd") or ""):
            continue
        sid = meta.get("id")
        if sid in CODEX_EXCLUDE or fid in CODEX_EXCLUDE:
            continue
        source = meta.get("source")
        kind = "reviewer" if isinstance(source, dict) and "guardian" in json.dumps(source) else ("subagent" if fid != sid else "chat")
        prev = None
        for line in lines:
            try:
                d = json.loads(line)
            except Exception:
                continue
            payload = d.get("payload", {})
            if d.get("type") == "event_msg" and payload.get("type") == "token_count" and payload.get("info"):
                u = payload["info"]["total_token_usage"]
                cur = (u["total_tokens"], u["output_tokens"], u["input_tokens"] - u["cached_input_tokens"] + u["output_tokens"])
                delta = cur if prev is None else tuple(a - b for a, b in zip(cur, prev))
                prev = cur
                t = ep(d["timestamp"])
                if any(delta) and t < CUTOFF:
                    events.append(dict(t=t, agent="codex", chat=sid, kind=kind, total=delta[0], out=delta[1], fresh=delta[2]))
    return events, names


def claude_events():
    events = []
    for path in glob.glob(str(CLAUDE / "*.jsonl")):
        sid = os.path.basename(path)[:-6]
        seen = set()
        for line in open(path, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            m = d.get("message") or {}
            if d.get("type") == "assistant" and isinstance(m, dict) and m.get("usage"):
                if m.get("id") in seen:
                    continue
                seen.add(m.get("id"))
                t = ep(d["timestamp"])
                if t >= CUTOFF:
                    continue
                u = m["usage"]
                inp, read, write, out = (u.get(k, 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"))
                events.append(dict(t=t, agent="claude", chat=sid, kind="chat", total=inp + read + write + out, out=out, fresh=inp + write + out))
    return events


def main() -> None:
    codex, names = codex_events()
    ev = codex + claude_events()
    by_time = sorted(ev, key=lambda e: e["t"])
    times, cumulative, total = [], [], 0
    for e in by_time:
        total += e["total"]
        times.append(e["t"])
        cumulative.append(total)

    def tokens_at(t: float) -> int:
        i = bisect.bisect_right(times, t)
        return cumulative[i - 1] if i else 0

    chats = {}
    for e in by_time:
        if e["kind"] == "reviewer":
            continue
        c = chats.setdefault(e["chat"], dict(agent=e["agent"], t0=e["t"], t1=e["t"], total=0))
        c["t1"] = e["t"]
        c["total"] += e["total"]
    named = []
    for sid, c in sorted(chats.items(), key=lambda kv: kv[1]["t0"]):
        name = names.get(sid) or CLAUDE_NAMES.get(sid, "")
        if name:
            named.append(dict(agent=c["agent"], name=name, t0=round(c["t0"]), t1=round(c["t1"]), total=round(c["total"] / 1e6, 2)))

    # time -> cumulative tokens, thinned to one sample per 0.4 M tokens or 20 minutes
    tuples = sorted((e["t"], e["total"], e["agent"]) for e in ev)
    tokmap, running, last_t, last_c = [], 0, None, -1e18
    for i, (t, n, _agent) in enumerate(tuples):
        running += n
        if last_t is None or running - last_c >= 4e5 or t - last_t >= 1200 or i == len(tuples) - 1:
            tokmap.append([round(t), round(running / 1e6, 3)])
            last_t, last_c = t, running
    segs = {"codex": [], "claude": []}
    for t, n, agent in tuples:
        s = segs[agent]
        if s and t - s[-1][1] < 1500:
            s[-1][1] = t
            s[-1][2] += n
        else:
            s.append([t, t, n])
    segs = {a: [[round(x0), round(x1), round(n / 1e6, 2)] for x0, x1, n in v] for a, v in segs.items()}

    usage = dict(
        tokmap=tokmap,
        segs=segs,
        chats=named,
        prompts=[dict(t=round(ep(stamp)), text=text, tok=round(tokens_at(ep(stamp)) / 1e6, 2)) for stamp, text in PROMPTS],
        summary=dict(
            tokens_total_M=round(sum(e["total"] for e in ev) / 1e6, 1),
            fresh_M=round(sum(e["fresh"] for e in ev) / 1e6, 2),
            out_M=round(sum(e["out"] for e in ev) / 1e6, 2),
            codex_M=round(sum(e["total"] for e in ev if e["agent"] == "codex") / 1e6, 1),
            claude_M=round(sum(e["total"] for e in ev if e["agent"] == "claude") / 1e6, 1),
            reviewer_M=round(sum(e["total"] for e in ev if e["kind"] == "reviewer") / 1e6, 1),
            t_first=round(by_time[0]["t"]),
            t_last=round(by_time[-1]["t"]),
        ),
    )
    OUT.write_text(json.dumps(usage, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(OUT.parents[2])}: {len(named)} chats, {usage['summary']['tokens_total_M']} M tokens")


if __name__ == "__main__":
    main()
