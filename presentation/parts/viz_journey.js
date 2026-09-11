/* ================= journey: lap time vs tokens / time / races ================= */
const J = DATA.journey;
function mapFn(map, gap, step) {
  const ts = map.map(r => r[0]), vs = map.map(r => r[1]), n = ts.length;
  return t => {
    if (t < ts[0]) return 0;
    if (t >= ts[n - 1]) return vs[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ts[m] <= t) lo = m; else hi = m; }
    if (step || ts[hi] - ts[lo] > gap) return vs[lo];
    return vs[lo] + (vs[hi] - vs[lo]) * (t - ts[lo]) / (ts[hi] - ts[lo]);
  };
}
const fTok = mapFn(J.tokmap, 900, false), fTrial = mapFn(J.trialmap, 0, true);
const T0 = edtEpoch(8, 26, 12, 0), T1 = edtEpoch(9, 2, 14, 0);
const DAYS = []; for (let d = 27; d <= 33; d++) DAYS.push(d <= 31 ? edtEpoch(8, d, 0, 0) : edtEpoch(9, d - 31, 0, 0));
const MODES = {
  tok: { f: fTok, d: [0, 510], ticks: [0, 100, 200, 300, 400, 500], fmt: v => (v ? v + ' M' : '0'), title: 'agent tokens processed (millions) — 97% are cached re-reads of the conversation' },
  time: { f: t => t, d: [T0, T1], ticks: DAYS, fmt: fmtDay, title: 'calendar time (EDT)' },
  trials: { f: fTrial, d: [0, 580000], ticks: [0, 1e5, 2e5, 3e5, 4e5, 5e5], fmt: v => (v ? v / 1000 + 'k' : '0'), title: 'simulated 30-second races — ≈ 198 days of driving in total' },
};
const PTS = J.points.slice().sort((a, b) => a.t - b.t);
const REJ = [{ t: edtEpoch(8, 28, 20, 18), lap: 9.65, v: 'v2 CEM gate', label: 'CEM gate' }].concat(J.rejected);
const QUOTE = i => J.prompts[i].text;
const BEATS = [
  { T: edtEpoch(8, 28, 12, 50), z: 0, date: 'Aug 26 – 28', title: 'Chats as generations', text: 'Each band under the axis is one conversation with an agent — Codex in blue, Claude Code in purple. For the first 45 M tokens the car hadn’t turned a wheel: we explored the repo, planned imitation learning, then wrote plan.md.', stat: '20 chats · 504 M tokens' },
  { T: edtEpoch(8, 28, 13, 13), key: 'minimum CEM', z: 0, date: 'Aug 28 · midday', title: 'First, a car that finishes', text: 'A hand-tuned centre-line controller ran one 28.2 s lap, clean on 14 of 14 seeds. Twenty CEM generations on its 12 gains: 23.75 s. (Until a car laps twice, its only lap is its best lap.)', stat: '28.2 → 23.75 s' },
  { T: edtEpoch(8, 28, 14, 22), key: 'CEM · faster', z: 0, date: 'Aug 28 · afternoon', title: 'Search harvests the slack', text: 'CEM on 14 gains of the same controller for 84 generations. We stopped when 8 of 14 deviations had collapsed onto their floor.', stat: '15.5 s' },
  { T: edtEpoch(8, 28, 17, 47), key: 'CEM · faster-line', prompt: 0, z: 0, date: 'Aug 28 · evening', title: '“Why does it brake?”', text: 'One negative-throttle tick latches the drivetrain. Coasting instead of braking, plus an outside–inside–outside line: +24% distance and a second lap.', stat: '9.57 s' },
  { T: edtEpoch(8, 28, 23, 32), key: 'v2', z: 0, date: 'Aug 28 · night', title: 'CEM out, GA in', text: 'The CEM gate (red) collapsed onto a broken curvature signal — the failure slide explains it. With local curvature fixed, a genetic algorithm reached 9.4 s in ten generations and 9.0 s in forty.', stat: '9.00 s' },
  { T: edtEpoch(8, 29, 17, 38), key: 'v5', z: 0, date: 'Aug 29', title: 'Move the box, not the search', text: 'v3–v5 reopened bounds the elites were pinned against and added one structural gene: the line may snap back to centre faster than it moves out. Searches ran from the terminal; the agent watched memory.', stat: '7.90 s' },
  { T: edtEpoch(8, 29, 23, 20), key: 'v9', prompt: 1, z: 0, date: 'Aug 29 · night', title: 'Rank what you mean', text: 'Ranking on lap time exposed exploits: v8 (red) crashed, recovered, then set one fast lap. Three-lap and clean tiers now come before speed — and that question became our ten-flat-generations rule.', stat: '7.78 s' },
  { T: edtEpoch(8, 31, 11, 48), key: 'v19', z: 1, date: 'Aug 30 – 31', title: 'The plateau', text: 'Launch cap, sweeper boost, reopened bounds: 7.617 s — exactly 457 ticks on all 30 seeds. Then v17, v18, v20, v22 and v24 each improved nothing.', stat: '457 ticks' },
  { T: edtEpoch(8, 31, 23, 50), key: 'v27', prompt: 3, z: 1, date: 'Aug 31', title: 'Play the game', text: '“Think about drifting…”, then ten seconds of recorded human driving: a corridor boost followed by a 0.15 s brake-turn pulse into the next corner. Clean on all 33 seeds.', stat: '7.183 s' },
  { T: edtEpoch(9, 2, 13, 30), key: 'v27', prompt: 4, z: 1, date: 'Sep 1 – 2', title: 'Copy the idea, not the trajectory', text: 'Replaying the human’s full sequence — rotate, carry, hold, settle — was safe on 33 seeds but 0.068 s slower (red). v27 stayed. Now flip the x-axis above.', stat: 'v27 kept' },
];

HOOKS['s-journey'] = {
  init(sl) {
    const svg = this.svg = sl.querySelector('#jr-svg');
    const W = 1000, Hh = 660; this.L = 66; this.R = 985; this.T = 16; this.B = 524;
    svg.setAttribute('viewBox', `0 0 ${W} ${Hh}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMin meet');
    const defs = S('defs', {}, svg);
    const cp = S('clipPath', { id: 'jr-clip' }, defs);
    S('rect', { x: this.L, y: this.T - 8, width: this.R - this.L + 8, height: this.B - this.T + 8 }, cp);
    this.gY = S('g', {}, svg);
    this.gX = S('g', {}, svg);
    S('line', { x1: this.L, x2: this.R, y1: this.B, y2: this.B, class: 'ax' }, svg);
    S('text', { x: this.L - 10, y: this.B + 38, 'font-size': 13, 'text-anchor': 'end', text: 'Codex' }, svg);
    S('text', { x: this.L - 10, y: this.B + 56, 'font-size': 13, 'text-anchor': 'end', text: 'Claude Code' }, svg);
    this.xTitle = S('text', { x: (this.L + this.R) / 2, y: this.B + 110, 'font-size': 14, 'text-anchor': 'middle', class: 'tx-mute' }, svg);
    S('text', { x: 4, y: this.T + 4, 'font-size': 13, class: 'tx-mute', text: 'best lap' }, svg);
    // chat bands
    this.chats = J.chats.map((c, i) => {
      const y = c.agent === 'codex' ? this.B + 27 : this.B + 45;
      const g = S('g', {}, svg);
      const r = S('rect', { y, height: 13, rx: 2, fill: c.agent === 'codex' ? COL.blue : COL.purple, 'fill-opacity': 0.55, stroke: COL.bg, 'stroke-width': 1 }, g);
      S('title', { text: `Chat ${i + 1} · ${c.name} · ${fmtDay(c.t0)} ${fmtHM(c.t0)} · ${c.total.toFixed(1)} M tokens` }, r);
      const tx = S('text', { y: y - 3, 'font-size': 11, 'text-anchor': 'middle', class: 'tx-mute', text: String(i + 1) }, g);
      return { c, r, tx };
    });
    const plot = S('g', { 'clip-path': 'url(#jr-clip)' }, svg);
    this.plateau = S('g', {}, plot);
    this.plLine = S('line', { stroke: COL.mute, 'stroke-dasharray': '5 5', 'stroke-width': 1.2 }, this.plateau);
    this.plText = S('text', { 'font-size': 13, class: 'tx-mute', text: '457 ticks = 7.617 s · every seed, v16 → v24' }, this.plateau);
    this.curve = S('path', { fill: 'none', stroke: COL.blue, 'stroke-width': 2.6, 'stroke-linejoin': 'round' }, plot);
    this.dots = PTS.map(p => {
      const c = S('circle', { r: 3.4, fill: COL.blue }, plot);
      S('title', { text: `${p.v}${p.g ? ' · gen ' + p.g : ''} · ${p.lap.toFixed(3)} s · ${fmtDay(p.t)} ${fmtHM(p.t)}` }, c);
      return c;
    });
    this.rej = REJ.map(r => {
      const g = S('g', {}, plot);
      S('circle', { r: 6.5, fill: COL.bg, stroke: COL.red, 'stroke-width': 2 }, g);
      S('text', { x: 10, y: -8, 'font-size': 13, fill: COL.red, text: r.label }, g);
      S('title', { text: `${r.v} · ${r.lap.toFixed(3)} s · rejected: ${r.label}` }, g);
      return { r, g };
    });
    this.prompts = J.prompts.map(p => {
      const g = S('g', {}, svg);
      S('path', { d: 'M0 -7 L6 0 L0 7 L-6 0 Z', fill: COL.brown, stroke: COL.bg, 'stroke-width': 1.5 }, g);
      S('title', { text: `${fmtDay(p.t)} ${fmtHM(p.t)} · “${p.text}”` }, g);
      return { p, g };
    });
    this.end = S('circle', { r: 6.5, fill: COL.yellow, stroke: COL.bg, 'stroke-width': 2.5 }, plot);
    this.ring = S('circle', { r: 17, fill: 'none', stroke: COL.yellow, 'stroke-width': 1.6 }, plot);
    const lg = S('g', { transform: `translate(${this.R - 330},${this.T + 8})` }, svg);
    S('circle', { cx: 0, cy: 0, r: 4, fill: COL.blue }, lg); S('text', { x: 9, y: 4.5, 'font-size': 13, text: 'promoted best' }, lg);
    S('circle', { cx: 118, cy: 0, r: 5, fill: COL.bg, stroke: COL.red, 'stroke-width': 2 }, lg); S('text', { x: 128, y: 4.5, 'font-size': 13, text: 'rejected' }, lg);
    S('path', { d: 'M212 -6 L217 0 L212 6 L207 0 Z', fill: COL.brown }, lg); S('text', { x: 223, y: 4.5, 'font-size': 13, text: 'our question' }, lg);

    this.panel = sl.querySelector('#jr-panel');
    this.st = { mode: 'tok', from: 'tok', mk: 1, z: 0, rev: BEATS[0].T, beat: 0 };
    ['tok', 'time', 'trials'].forEach(m => sl.querySelector('#jr-x-' + m).addEventListener('click', () => this.setMode(m)));
    this.render();
    this.fillPanel(0);
  },
  nx(m, t) { const M = MODES[m]; return (M.f(t) - M.d[0]) / (M.d[1] - M.d[0]); },
  X(t) { const s = this.st; return this.L + lerp(this.nx(s.from, t), this.nx(s.mode, t), s.mk) * (this.R - this.L); },
  ydom() { const z = this.st.z; return [lerp(6.6, 7.05, z), lerp(29.5, 8.2, z)]; },
  Y(v) { const [a, b] = this.ydom(); return this.B - (v - a) / (b - a) * (this.B - this.T); },
  render() {
    const s = this.st, L = this.L, R = this.R;
    this.gY.replaceChildren();
    [[[8, 12, 16, 20, 24, 28], 1 - s.z, 0], [[7.2, 7.4, 7.6, 7.8, 8.0], s.z, 1]].forEach(([ticks, op, dp]) => {
      if (op < 0.02) return;
      ticks.forEach(v => {
        const y = this.Y(v); if (y < this.T - 2 || y > this.B + 1) return;
        S('line', { x1: L, x2: R, y1: y, y2: y, class: 'gr', opacity: op }, this.gY);
        S('text', { x: L - 10, y: y + 4.5, 'font-size': 13, 'text-anchor': 'end', opacity: op, text: v.toFixed(dp) + ' s' }, this.gY);
      });
    });
    this.gX.replaceChildren();
    [[s.from, 1 - s.mk], [s.mode, s.mk]].forEach(([m, op]) => {
      if (op < 0.02 || (m === s.mode && s.from === s.mode && op !== s.mk)) return;
      const M = MODES[m];
      M.ticks.forEach(v => {
        const x = L + (v - M.d[0]) / (M.d[1] - M.d[0]) * (R - L);
        S('line', { x1: x, x2: x, y1: this.B, y2: this.B + 5, class: 'ax', opacity: op }, this.gX);
        S('text', { x, y: this.B + 86, 'font-size': 13, 'text-anchor': 'middle', opacity: op, text: M.fmt(v) }, this.gX);
      });
    });
    this.xTitle.textContent = MODES[s.mode].title;
    this.chats.forEach(({ c, r, tx }) => {
      const x0 = this.X(c.t0), x1 = Math.max(this.X(c.t1), x0 + 1.5);
      r.setAttribute('x', x0.toFixed(1)); r.setAttribute('width', (x1 - x0).toFixed(1));
      tx.setAttribute('x', ((x0 + x1) / 2).toFixed(1));
      tx.setAttribute('opacity', x1 - x0 >= 15 ? 1 : 0);
    });
    // curve up to s.rev
    const vis = PTS.filter(p => p.t <= s.rev);
    let d = '';
    vis.forEach((p, i) => {
      const x = this.X(p.t).toFixed(1), y = this.Y(p.lap).toFixed(1);
      d += i ? `H${x}V${y}` : `M${x} ${y}`;
    });
    if (vis.length) d += `H${this.X(Math.min(s.rev, T1)).toFixed(1)}`;
    this.curve.setAttribute('d', d);
    PTS.forEach((p, i) => {
      const c = this.dots[i], on = p.t <= s.rev;
      c.setAttribute('cx', this.X(p.t).toFixed(1)); c.setAttribute('cy', this.Y(p.lap).toFixed(1));
      c.setAttribute('opacity', on ? 1 : 0);
    });
    const last = vis[vis.length - 1];
    this.end.setAttribute('opacity', last ? 1 : 0);
    if (last) { this.end.setAttribute('cx', this.X(last.t)); this.end.setAttribute('cy', this.Y(last.lap)); }
    this.rej.forEach(({ r, g }) => {
      g.setAttribute('transform', `translate(${this.X(r.t).toFixed(1)},${this.Y(r.lap).toFixed(1)})`);
      g.setAttribute('opacity', r.t <= s.rev ? 1 : 0);
    });
    const pb = BEATS[s.beat].prompt;
    this.prompts.forEach(({ p, g }, i) => {
      g.setAttribute('transform', `translate(${this.X(p.t).toFixed(1)},${this.B - 12})`);
      g.setAttribute('opacity', p.t <= s.rev ? (i === pb ? 1 : 0.55) : 0);
    });
    const ply = this.Y(7.6167), plx = this.X(edtEpoch(8, 30, 20, 35));
    this.plLine.setAttribute('x1', plx); this.plLine.setAttribute('x2', R); this.plLine.setAttribute('y1', ply); this.plLine.setAttribute('y2', ply);
    this.plText.setAttribute('x', Math.min(plx + 8, R - 290)); this.plText.setAttribute('y', ply - 8);
    this.plateau.setAttribute('opacity', s.z > 0.5 && s.rev >= edtEpoch(8, 30, 20, 35) ? s.z : 0);
    const kb = BEATS[s.beat].key, kp = kb && PTS.filter(p => p.v === kb).pop();
    this.ring.setAttribute('opacity', kp && kp.t <= s.rev ? 1 : 0);
    if (kp) { this.ring.setAttribute('cx', this.X(kp.t)); this.ring.setAttribute('cy', this.Y(kp.lap)); }
  },
  fillPanel(k) {
    const b = BEATS[k], p = b.prompt != null ? J.prompts[b.prompt] : null;
    this.panel.innerHTML =
      `<div class="jr-date">${b.date} · beat ${k + 1} of ${BEATS.length}</div><h3>${b.title}</h3>` +
      (p ? `<blockquote class="q">${p.text}<cite>— Zhi, to Codex · ${fmtDay(p.t)}, ${fmtHM(p.t)}</cite></blockquote>` : '') +
      `<p>${b.text}</p><div class="spacer"></div><div class="jr-stat">${b.stat}</div>`;
  },
  step(k) {
    if (k < 0) { this.stop && this.stop(); this.st.rev = BEATS[0].T; this.st.z = 0; this.st.beat = 0; this.render(); return; }
    const s = this.st, b = BEATS[k], r0 = s.rev, z0 = s.z;
    s.beat = k;
    this.fillPanel(k);
    if (this.stop) this.stop();
    const target = ENV.reader ? edtEpoch(9, 2, 13, 30) : b.T;
    this.stop = tween(1300, e => { s.rev = lerp(r0, target, e); s.z = lerp(z0, b.z, e); this.render(); });
  },
  setMode(m) {
    const s = this.st;
    if (m === s.mode) return;
    s.from = s.mk < 0.5 ? s.from : s.mode; s.mode = m; s.mk = 0;
    this.svg.closest('.slide').querySelectorAll('[id^="jr-x-"]').forEach(b => b.setAttribute('aria-pressed', String(b.id === 'jr-x-' + m)));
    if (this.stopM) this.stopM();
    this.stopM = tween(1200, e => { s.mk = e; this.render(); }, () => { s.from = s.mode; s.mk = 1; this.render(); });
  },
};

/* ================= chronology ribbon ================= */
const RIBBON = {
  init() {
    const svg = this.svg = document.getElementById('rb-svg');
    svg.setAttribute('viewBox', '0 0 1000 56');
    const x = this.x = t => 4 + (t - T0) / (T1 - T0) * 992;
    this.win = S('rect', { y: 0, height: 44, rx: 3, fill: COL.ink, 'fill-opacity': 0.07 }, svg);
    DAYS.forEach(t => {
      S('line', { x1: x(t), x2: x(t), y1: 2, y2: 44, class: 'gr' }, svg);
      S('text', { x: x(t) + 4, y: 55, 'font-size': 10.5, text: fmtDay(t) }, svg);
    });
    S('text', { x: x(T0), y: 55, 'font-size': 10.5, text: 'Aug 26' }, svg);
    [['codex', COL.blue, 28], ['claude', COL.purple, 36]].forEach(([a, c, y]) => {
      (J.segs[a] || []).forEach(([t0, t1]) => S('rect', { x: x(t0), y, width: Math.max(1.2, x(t1) - x(t0)), height: 5, rx: 1, fill: c, 'fill-opacity': 0.7 }, svg));
    });
    const yl = l => 3 + (1 - (l - 7) / 22) * 20;
    let d = '';
    PTS.forEach((p, i) => { d += i ? `H${x(p.t).toFixed(1)}V${yl(p.lap).toFixed(1)}` : `M${x(p.t).toFixed(1)} ${yl(p.lap).toFixed(1)}`; });
    d += `H${x(T1)}`;
    S('path', { d, fill: 'none', stroke: COL.blue, 'stroke-width': 1.4, 'stroke-opacity': 0.9 }, svg);
    this.head = S('line', { y1: 0, y2: 44, stroke: COL.yellow, 'stroke-width': 1.6 }, svg);
    this.cur = [T0, T1];
    this.set(this.cur);
  },
  set([a, b]) {
    const xa = this.x(a), xb = this.x(b);
    this.win.setAttribute('x', xa.toFixed(1)); this.win.setAttribute('width', Math.max(0, xb - xa).toFixed(1));
    this.head.setAttribute('x1', xb.toFixed(1)); this.head.setAttribute('x2', xb.toFixed(1));
    this.head.setAttribute('opacity', this.forceHead || b - a < 3 * 86400 ? 1 : 0);
  },
  update(sl, k) {
    let r;
    this.forceHead = sl.id === 's-journey';
    if (sl.id === 's-journey') { const T = BEATS[Math.max(0, k)].T; r = [T0, T]; }
    else r = parseWhen(sl.dataset.when) || [T0, T1];
    const c0 = this.cur.slice();
    if (this.stop) this.stop();
    this.stop = tween(900, e => { this.cur = [lerp(c0[0], r[0], e), lerp(c0[1], r[1], e)]; this.set(this.cur); });
  },
};
window.RIBBON = RIBBON;
