# Presentation outline — "The 7.19-Second Lap"

Eight slides, high level, about **five minutes**. No deep technical detail:
what a controller is, the two approaches we tried, how the lap got faster and
what surprised us, a real race, and what we take from it. The deck lives in
[`presentation/`](presentation/README.md).

- **Present:** open `presentation/index.html`, or the online copy at
  <https://claude.ai/code/artifact/7d36b83a-ef11-4de7-be95-eb81ccc148cf>
- **Keys:** → next step · ← back · **N** speaker notes · **F** full screen
- **After any edit:** `python3 presentation/build.py`

Each slide's speaker notes (**N**) open with its time budget, how many times to
press →, the clock time to leave by, and the line to say.

## The run

| # | Slide | Presses | Time | What you say |
| --- | --- | --- | --- | --- |
| 1 | The 7.19-second lap | 0 | 0:15 | "Same track, same start. Grey was our first safe car, yellow is what we shipped: 28 seconds down to 7.19." |
| 2 | What a controller is | 3 | 0:35 | "It's a function, sixty times a second. No map, no plan — it reacts to what's just ahead, following rules we wrote." |
| 3 | Two ways to learn a lap | 3 | 0:35 | "Lucas tried memorizing the track; it ran out of memory and wouldn't transfer. I tuned a reflex, and that's what we kept." |
| 4 | How the controller drives | 4 | 0:35 | "Read the bend ahead, take the widest line through it, pick a speed — and once a lap, drift the way we did by hand." |
| 5 | Two loops: the laptop and the chat | 2 | 0:30 | "The laptop raced 570,000 times. The chats decided what was worth trying: 30 versions, 52 logged decisions." |
| 6 | How the lap time came down | 5 | 1:20 | Step the six beats. Each drop has the surprise that caused it: the brake locked the accelerator, the score got gamed, the plateau was the car, ten seconds of driving beat all of it. |
| 7 | Seed 110, raced three ways | 0 | 0:30 | "Baseline, a human at the keyboard, and what we shipped: laps of 8.20, 7.20, 7.25. Watch the first hairpin." + leaderboard placement |
| 8 | What we learned — and the open question | 2 | 0:50 | "Tuning got us most of the way; new moves got us the rest. Which leaves the open question: is Codex a more efficient learner than traditional ML?" |

**Total ≈ 5:10.** Slide 6 carries the most weight; if you're running long, stop
after the plateau beat.

## Where the final numbers went

The results table is no longer in the deck (it's in `extras.html`). Say the
headline numbers over the race on slide 7 — **7.190 s mean best lap and 8.160 s
first lap across the five grading seeds, every run clean, 100 of 100 endurance
runs survived** — and give the leaderboard placement there.

## Jargon, explained in passing

| Slide | Terms |
| --- | --- |
| 3 Two ways to learn a lap | seed · cross-entropy search · genetic algorithm |
| 5 Two loops | version · generation · elite · crossover |

Edit one by finding `class="terms"` in `body.html`.

## Slides we cut

Nine slides are parked in `presentation/parts/extras.html`, kept word for word:
the results table, the sensor diagram, the seed suites and ranking, CEM versus
the genetic algorithm, the curvature bug, a glossary, an appendix divider, the
standalone open-question slide, and the surprises table. They are **not** built.
Note the glossary there still describes elites as "the only ones allowed to
breed", which is wrong — see slide 5 below. To bring a slide back, move its
`<section>` into `body.html` and rebuild.

## Where to edit

| What | File |
| --- | --- |
| Slide text, order, speaker notes | `presentation/parts/body.html` |
| The line under a title · the terms strip · the notes cue | `class="sh-lead"` · `class="terms"` · `class="cue"` |
| Colours, type, spacing | `presentation/parts/style.css` |
| Slide 6's beats and their surprises | `BEATS` at the top of `presentation/parts/viz_journey.js` |
| Title map, controller map, race | `presentation/parts/viz_track.js` |
| Other drawings | `presentation/parts/viz_misc.js` |

---

## 1 · The 7.19-second lap (`s-title`)

Open on the result. Two cars replay real 30-second runs from seed 110 — grey
`minimum_viable`, yellow `race_faster` — with live lap times. 28 s is the
first controller's only lap; 7.19 s is what we shipped, measured on the five
grading seeds.

## 2 · What a controller is (`s-what`) — foundation

A loop: **Sense** → **Decide** → **Act**, 60 times a second. It's a function;
the car has no map and no plan; ours answers with rules (*if the road bends this
much, aim this far off centre and hold this speed*). So a lap gets faster only
by a straighter path or more speed through the corners.

## 3 · Two ways to learn a lap (`s-approaches`) — **approaches**

- **Lucas · alternative — "Memorize the track"**: search offline for the best
  throttle and steering at each point of the lap. About one lap in 30 s; a
  multi-hour run exhausted memory and crashed the laptop; the result fits one
  track and one starting pose. Abandoned Aug 30.
- **Zhi · primary — "Tune a reflex"**: the controller from slide 2, with its
  parameters tuned across many random starts.

*Have Lucas check his column — it comes from the lab notes.*

## 4 · How the controller drives (`s-controller`)

A real lap on seed 110 coloured by speed. Four stages:

1. **Read the road** — how sharply is it bending 4, 9 and 16 m ahead?
2. **Choose a line** — start a corner on the outside edge, touch the inside
   edge at its middle, finish on the outside again. The bigger, gentler arc
   lets the car carry more speed.
3. **Choose a speed** — aim for 25 m/s on straights and 14 m/s in corners;
   when it's too fast, let go of the accelerator and coast rather than brake.
4. **Drift like we did** — by imitating our own recorded laps, once a lap it
   brakes hard into a corner to swing the car round.

*Stage 4 is imitation in the everyday sense: we watched our recording and coded
the move by hand, then let the search tune it. No model was trained on the
recording, so avoid calling it "imitation learning" if asked.*

## 5 · Two loops: the laptop and the chat (`s-loops`) — **development strategy**

**Version** — one search run we set up: which parameters are open to search,
how wide each one's range is, and how candidates are scored. Everything else
stays locked at the previous winner, and the search starts from that winner.
**Generation** — one round inside a version. Every generation in a version
searches the same parameters; only the values being tried change. Examples:
v16 searched just the 2 start-speed-cap parameters; v19 re-searched corner
speed and braking distance with wider ranges; v11 kept v10's 16 parameters and
only changed the score. (All of this is in slide 5's speaker notes too.)

**Inner loop, each round (about a minute):**

1. Race every candidate in the batch from a few starting points (6 training
   seeds, later plus the official or grading seeds).
2. Race the best dozen again from every starting point (28–33) for a reliable
   score. Nothing is trained here — it's the same candidate, tested more.
3. Keep that dozen unchanged, and breed the rest of the next batch. Each child
   gets two parents, each picked as the best of three random batch members.
   **Crossover**: for every parameter, the child takes a random value between
   its parents' values (reaching a little beyond either one). **Mutation**:
   each value has a 1-in-4 chance of a small random nudge.

**The first batch** of 64: the previous winner, 15 small random variations of
it, and 48 completely random settings within the allowed ranges.

**Outer loop:** question → hypothesis → build it → launch a search → promote or
reject, across 20 conversations, 30 versions and 52 logged decisions.

## 6 · How the lap time came down (`s-journey`) — **evidence and failures**

A single step chart over the dates Aug 28 – Sep 1: blue dots are promoted
controllers, red rings are rejected candidates, brown diamonds are our
questions. Each beat shows the surprise that caused the drop:

| Beat | What it says | Ends at |
| --- | --- | --- |
| 1 A car that finishes | Our first controller followed the middle of the track: one lap took 28.2 s | 23.75 s |
| 2 Search harvests the slack | 84 rounds over 14 parameters — target speeds, steering strengths, braking distances; stopped when it stopped improving | 15.5 s |
| 3 "Why does it brake?" | **Surprise:** tapping the brake put the car into "about to reverse" mode — until it nearly stopped, the accelerator only braked harder. So it coasts instead | 9.00 s |
| 4 The score got gamed | **Surprise:** the top car hit a wall, limped, then set one fast lap and won on best-lap. Now it must finish three clean laps first; ranges widened where results sat on their edge | 7.78 s |
| 5 The plateau | **Surprise:** every start gave the identical lap; five versions moved nothing. Got here via a speed cap for the first two seconds, extra speed through the one long bend, and wider ranges | 7.617 s |
| 6 Play the game | Ten seconds of driving by hand showed the drift; copying the full human sequence was slower | 7.183 s |

## 7 · Seed 110, raced three ways (`s-race`) — **representative run + results**

Three real runs from the same spawn: grey baseline (one lap, 19.50 s), brown
human replay (8.55 s, recording ends at 10.4 s), yellow v27 (8.20, 7.20,
7.25 s). Say the grading-seed numbers and the leaderboard placement here.

## 8 · What we learned — and the open question (`s-learned`)

Three bars showing where the 21 seconds came from (search vs new behaviour),
then four large bullets:

- Tuning numbers got us most of the way; the last big gains came from new
  moves, like coasting and drifting.
- The search exploits any loophole in its score — it rewarded a car that
  crashed and then sprinted.
- When the search stalls, change what the car can do, not how long you search.
- Our best idea came from driving the car ourselves.

Closing line: **570,000 races · 504 M tokens · 10.4 s of driving**, under the
open question — *is Codex a more efficient learner than traditional ML?*

---

## Before you present

- [ ] Decide where to say the leaderboard placement — slide 7 is the natural
      spot.
- [ ] Confirm v27 is the controller you submitted, not v30.
- [ ] Have Lucas check his column on slide 3 and his role on slide 5.
- [ ] Split the slides between you; the notes are written so either of you can
      take any slide.
- [ ] Time one run-through — the table is an estimate, not a measurement.
