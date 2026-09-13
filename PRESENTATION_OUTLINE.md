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
| 2 | What a controller is | 3 | 0:35 | "It's a function, sixty times a second. No map, no plan. Ours follows rules we wrote — and a search found the numbers inside them." |
| 3 | Two ways to learn a lap | 3 | 0:35 | "Lucas tried memorizing the track; it ran out of memory and wouldn't transfer. I tuned a reflex, and that's what we kept." |
| 4 | How the controller drives | 4 | 0:35 | "Read the bend ahead, pick a line, pick a speed, stay out of the wall. It lifts off instead of braking, and it drifts once a lap." |
| 5 | Two loops: the laptop and the chat | 2 | 0:30 | "The laptop tried 570,000 races. The chats decided what was worth trying: 30 versions, 52 logged decisions." |
| 6 | Lap time, one chat at a time | 5 | 1:20 | Step the six beats. Each drop has the surprise that caused it: braking killed the engine, the score got gamed, the plateau was the car, ten seconds of driving beat all of it. |
| 7 | Seed 110, raced three ways | 0 | 0:30 | "Baseline, a human at the keyboard, and what we shipped: laps of 8.20, 7.20, 7.25. Watch the first hairpin." + leaderboard placement |
| 8 | What we learned — and the open question | 2 | 0:50 | "Early, search bought the seconds; late, only new behaviour did. Which leaves the open question: is Codex a more efficient learner than traditional ML?" |

**Total ≈ 5:10.** Slide 6 carries the most weight; if you're running long, stop
after the plateau beat.

## Where the final numbers went

The results table slide is no longer in the deck (it's in `extras.html`). Say
the headline numbers over the race on slide 7 — **7.190 s mean best lap and
8.160 s first lap across the five grading seeds, every run clean, 100 of 100
endurance runs survived** — and give the leaderboard placement there. Ask me if
you'd rather have that line printed on a slide.

## Jargon, explained in passing

| Slide | Terms |
| --- | --- |
| 3 Two ways to learn a lap | seed · cross-entropy search · genetic algorithm |
| 5 Two loops | generation · elite |

Edit one by finding `class="terms"` in `body.html`.

## Slides we cut

Nine slides are parked in `presentation/parts/extras.html`, kept word for word:
the results table, the sensor diagram, the seed suites and ranking, CEM versus
the genetic algorithm, the curvature bug, a glossary, an appendix divider, the
standalone open-question slide, and the surprises table. They are **not** built.
To bring one back, move its `<section>` into `body.html` and rebuild.

## Where to edit

| What | File |
| --- | --- |
| Slide text, order, speaker notes | `presentation/parts/body.html` |
| The line under a title · the terms strip · the notes cue | `class="sh-lead"` · `class="terms"` · `class="cue"` |
| Colours, type, spacing | `presentation/parts/style.css` |
| Slide 6's beats and their surprises | `BEATS` at the top of `presentation/parts/viz_journey.js` |
| Title map, controller map, race | `presentation/parts/viz_track.js` |
| Other drawings | `presentation/parts/viz_misc.js` |

To change when something appears, edit `data-s="n"` on the element and
`data-steps` on the section.

---

## 1 · The 7.19-second lap (`s-title`)

Open on the result. Two cars replay real 30-second runs from seed 110 — grey
`minimum_viable`, yellow `race_faster` — with live lap times. 28 s is the
hand-tuned baseline's only lap; 7.19 s is what we shipped, measured on the five
grading seeds.

## 2 · What a controller is (`s-what`) — foundation

A loop: **Sense** → **Decide** → **Act**, 60 times a second. Then: it's a
function; the car has no map and no plan; ours answers with rules; **we wrote
the rules, a search found the numbers**; so a lap gets faster only by a
straighter path or more speed through the corners.

## 3 · Two ways to learn a lap (`s-approaches`) — **approaches**

- **Lucas · alternative — "Memorize the track"**: search offline for the best
  throttle and steering at each point of the lap. About one lap in 30 s; a
  multi-hour run exhausted memory and crashed the laptop; the result fits one
  track and one starting pose. Abandoned Aug 30.
- **Zhi · primary — "Tune a reflex"**: the controller from slide 2, with its
  numbers tuned across many random starts. Works from any start; clean on 14 of
  14 seeds before tuning; every candidate raced on 28–33 seeds.

*Have Lucas check his column — it comes from the lab notes.*

## 4 · How the controller drives (`s-controller`)

A real lap on seed 110 coloured by speed. Four plain stages: **read the road**
(how sharply is it bending 4, 9 and 16 m ahead?) · **choose a line** (swing
wide, cut the corner, run wide again) · **choose a speed** (about 25 m/s on
straights, 14 in corners; above that it lifts off rather than brakes) · **stay
out of the wall** (and once a lap it brakes hard into a corner to rotate — the
trick copied from a human). 116 numbers set all of it.

## 5 · Two loops: the laptop and the chat (`s-loops`) — **development strategy**

Inner loop: race the candidates, keep the best, breed the next batch — about a
minute a round, 771 rounds, ≈ 570,000 races, no tokens. Outer loop: question →
hypothesis → build it → launch a search → promote or reject, across 20
conversations, 30 versions and 52 logged decisions. Plus who did what.

## 6 · Lap time, one chat at a time (`s-journey`) — **evidence and failures**

The heart of the talk. A step chart from 28.2 s to 7.18 s; bands under the axis
are the 20 chats (blue Codex, purple Claude Code), brown diamonds are our
questions, red rings are candidates we rejected. Buttons re-plot the curve
against agent tokens, calendar time or simulated races. Each beat shows the
surprise that caused the drop:

| Beat | The surprise | Ends at |
| --- | --- | --- |
| 1 Chats as generations | — | 23.75 s |
| 2 Search harvests the slack | — | 15.5 s |
| 3 "Why does it brake?" | Braking switched the engine off until the car nearly stopped | 9.00 s |
| 4 Widen the limits, fix the score | Our first fast candidate crashed, recovered, sprinted one lap and won on points | 7.78 s |
| 5 The plateau | Every start returned the identical lap; five versions moved nothing | 7.617 s |
| 6 Play the game | Ten seconds of driving by hand showed a move the search never tried | 7.183 s |

## 7 · Seed 110, raced three ways (`s-race`) — **representative run + results**

Three real runs from the same spawn: grey baseline (one lap, 19.50 s), brown
human replay (8.55 s, recording ends at 10.4 s), yellow v27 (8.20, 7.20,
7.25 s), with the throttle traces underneath. Say the grading-seed numbers and
the leaderboard placement here.

## 8 · What we learned — and the open question (`s-learned`)

Three bars showing where the 21 seconds came from:

| Phase | Search | New behaviour |
| --- | --- | --- |
| 28.2 → 9.0 s | 12.70 s (66%) | 6.50 s (34%) |
| 9.0 → 7.9 s | 0.48 s (44%) | 0.62 s (56%) |
| 7.9 → 7.18 s | 0.12 s (16%) | 0.60 s (84%) |

Four large bullets — search bought the early seconds, new behaviour bought the
last ones; whatever you score is what you get; test on starts you never tuned
on; our best ideas came from driving the car ourselves. Then the closing line:
**570,000 races · 504 M tokens · 10.4 s of driving**, under the open question
in so many words — *is Codex a more efficient learner than traditional ML?*

---

## Before you present

- [ ] Decide where to say the leaderboard placement — slide 7 is the natural
      spot now that the results table is out.
- [ ] Confirm v27 is the controller you submitted, not v30.
- [ ] Have Lucas check his column on slide 3 and his role on slide 5.
- [ ] Split the slides between you; the notes are written so either of you can
      take any slide.
- [ ] Time one run-through — the table is an estimate, not a measurement.
