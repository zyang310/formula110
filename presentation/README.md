# The 7.19-Second Lap — Formula 110 presentation

A 14-slide, 3Blue1Brown-style deck: our two approaches, the development loop
with the AI agents, the lap-time journey from 28.2 s to 7.19 s, and what we
learned.

Live link (Zhi's artifact, private until shared):
<https://claude.ai/code/artifact/7d36b83a-ef11-4de7-be95-eb81ccc148cf>

## Present

- **Online:** open the link above.
- **Offline:** open `presentation/index.html` in Chrome, Safari or Firefox.
  Fonts and data are embedded, so it needs no internet.

| Key | Action |
| --- | --- |
| → · Space · Enter | Next animation step, then next slide |
| ← · Backspace | Back |
| Home · End | First · last slide |
| N | Speaker notes |
| F | Full screen (or use the browser's own full screen) |

Slide 7 has x-axis buttons (tokens, calendar time, simulated races); slide 11
has race controls. Add `#7` to the URL to open a specific slide, or `?all` to
show every step already revealed. Below 820 px wide the deck becomes one
scrolling page.

## Edit

| To change | Edit |
| --- | --- |
| Slide text, order, speaker notes | `parts/body.html` — one `<section class="slide">` per slide; notes live in `<aside class="notes">` |
| Colours, fonts, spacing | `parts/style.css` — tokens at the top |
| Slide 7 beat text | `BEATS` in `parts/viz_journey.js` |
| Title, controller map, race | `parts/viz_track.js` |
| Task, approaches, seeds, loops, CEM/GA, curvature, learning bars | `parts/viz_misc.js` |
| Navigation and animation engine | `parts/app.js` |
| Numbers behind the charts | `data/deck-data.json` (generated — see Data) |

Then rebuild and open `index.html` to check:

```bash
python3 presentation/build.py
```

Commit `parts/`, `data/`, `index.html` and `artifact.html` together.

Step reveals: give an element `data-s="n"` to reveal it on step `n`, and set
the section's `data-steps` to the highest `n`. Add class `up` for a small
rise; an SVG path with class `draw` animates as a stroke being drawn.

## Share and update the link

- **To let someone view it:** open the artifact and use its Share menu.
  Artifacts are private by default.
- **Only the owner can update that link.** After pulling changes and
  rebuilding, Zhi asks Claude Code to "republish `presentation/artifact.html`
  to <the link above>". Anyone else who publishes `artifact.html` from their
  own account gets a separate link.

## Data

`data/deck-data.json` holds every number on the slides. Rebuild it from the
repository, then rebuild the deck:

```bash
PYTHONPATH=. uv run python presentation/tools/extract_data.py
python3 presentation/build.py
```

It reads:

- `artifacts/controller-search/*/generations/*.json` and `checkpoint.json`:
  the committed search checkpoints behind the lap-time curve, the
  simulated-race counts and the CEM/GA populations. To put a new promoted
  search on the curve, add its folder name to `LINEAGE` in `extract_data.py`.
- `data/checkpoint-times.json`: when each checkpoint was written. Git doesn't
  keep file times, so they're recorded here; new checkpoints are added from
  their local file time.
- `data/agent-usage.json`: Codex and Claude Code token counts and chat times.
  It holds no transcript content beyond the five prompts quoted on the slides.
  Only Zhi can regenerate it, because it reads his local agent logs:
  `python3 presentation/tools/extract_agent_usage.py`.
- The race trajectories. Re-record the three seed-110 runs (this uses
  `artifacts/human-driving.jsonl`) and pass them in:

  ```bash
  PYTHONPATH=. uv run python presentation/tools/capture_paths.py paths.json
  PYTHONPATH=. uv run python presentation/tools/extract_data.py --paths paths.json
  ```

Add `--check` to `extract_data.py` to confirm `deck-data.json` is up to date
without changing it. The ~400 MB of diagnostic traces (`*.jsonl` under
`artifacts/controller-search/`) are not committed.

## Fonts

CMU Serif and CMU Typewriter (Computer Modern Unicode, Andrey V. Panov),
subset to WOFF, under the SIL Open Font License 1.1 — see `fonts/OFL.txt`.
