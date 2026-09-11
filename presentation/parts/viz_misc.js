/* ================= task schematic ================= */
HOOKS['s-task'] = {
  init(sl) {
    const svg = sl.querySelector('#task-svg');
    svg.setAttribute('viewBox', '0 0 600 560');
    const k = 24, cx0 = 300, cy0 = 500, R = 26, ox = 12; // car 0.5 m left of the centre line
    const Cx = cx0 + ox + R * k; // turn centre (right-hand bend)
    const arc = (rad, s0, s1) => { const pts = []; for (let s = s0; s <= s1; s += 0.4) { const a = s / R; pts.push([Cx - rad * Math.cos(a), cy0 - rad * Math.sin(a)]); } return pts; };
    const RK = R * k;
    S('path', { d: openD(arc(RK, -3, 21)), class: 'road', 'stroke-width': 6.6 * k }, svg);
    [-4.7, 4.7].forEach(o => S('path', { d: openD(arc(RK + o * k, -3, 21)), class: 'wall' }, svg));
    [-3.3, 3.3].forEach(o => S('path', { d: openD(arc(RK + o * k, -3, 21)), class: 'edge' }, svg));
    S('path', { d: openD(arc(RK, -3, 21)), class: 'cl' }, svg);
    const car = S('g', { transform: `translate(${cx0},${cy0})` }, svg);
    S('path', { d: 'M0 -18 L10 14 L0 8 L-10 14 Z', fill: COL.ink }, car);
    // 1 · lookahead offsets along the heading
    const g1 = S('g', { 'data-s': 1 }, svg);
    S('line', { x1: cx0, x2: cx0, y1: cy0 - 20, y2: cy0 - 17 * k, stroke: COL.mute, 'stroke-dasharray': '4 5', 'stroke-width': 1.2 }, g1);
    [4, 9, 16].forEach((d, i) => {
      const y = cy0 - d * k, x = Cx - Math.sqrt(RK * RK - (cy0 - y) * (cy0 - y));
      S('line', { x1: cx0, x2: x, y1: y, y2: y, stroke: COL.blue, 'stroke-width': 2 }, g1);
      S('circle', { cx: x, cy: y, r: 6, fill: COL.blue }, g1);
      const t = S('text', { x: (cx0 + x) / 2, y: y - 9, 'font-size': 17, 'text-anchor': 'middle', 'font-style': 'italic', class: 'tx-ink' }, g1);
      t.textContent = 'c'; S('tspan', { 'font-size': 12, dy: 4, text: String(i) }, t);
      S('text', { x: cx0 - 12, y: y + 5, 'font-size': 14, 'text-anchor': 'end', text: d + ' m' }, g1);
    });
    // 2 · LiDAR rays to the walls
    const g2 = S('g', { 'data-s': 2 }, svg);
    for (let a = -90; a <= 90; a += 22.5) {
      const th = a * Math.PI / 180, dx = Math.sin(th), dy = -Math.cos(th);
      let hit = null;
      for (let r = 10; r < 520; r += 1.5) {
        const x = cx0 + dx * r, y = cy0 + dy * r, dd = Math.hypot(x - Cx, y - cy0);
        if (dd < RK - 4.7 * k || dd > RK + 4.7 * k) { hit = [x, y]; break; }
      }
      if (!hit) continue;
      S('line', { x1: cx0, y1: cy0, x2: hit[0], y2: hit[1], stroke: COL.ink2, 'stroke-opacity': 0.35, 'stroke-width': 1 }, g2);
      S('circle', { cx: hit[0], cy: hit[1], r: 3, fill: COL.ink2 }, g2);
    }
    // 3 · outputs
    const g3 = S('g', { 'data-s': 3 }, svg);
    S('path', { d: `M${cx0 + 40} ${cy0 + 26} v-54`, stroke: COL.yellow, 'stroke-width': 3, fill: 'none' }, g3);
    S('path', { d: `M${cx0 + 32} ${cy0 - 20} l8 -12 l8 12`, stroke: COL.yellow, 'stroke-width': 3, fill: 'none' }, g3);
    S('text', { x: cx0 + 52, y: cy0 + 4, 'font-size': 16, class: 'tx-ink', text: 'throttle' }, g3);
    S('path', { d: `M${cx0 - 30} ${cy0 + 34} A 34 34 0 0 0 ${cx0 + 22} ${cy0 + 38}`, stroke: COL.yellow, 'stroke-width': 3, fill: 'none' }, g3);
    S('text', { x: cx0 - 40, y: cy0 + 56, 'font-size': 16, class: 'tx-ink', 'text-anchor': 'end', text: 'steer' }, g3);
  },
};

/* ================= approaches ================= */
HOOKS['s-approaches'] = {
  init(sl) {
    const a = sl.querySelector('#appr-a');
    a.setAttribute('viewBox', '0 0 520 150');
    S('line', { x1: 20, x2: 500, y1: 110, y2: 110, class: 'ax' }, a);
    let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let sct = 0; sct < 12; sct++) {
      const x0 = 20 + sct * 40;
      S('line', { x1: x0, x2: x0, y1: 104, y2: 116, class: 'ax' }, a);
      for (let j = 0; j < 8; j++) {
        const h = 10 + rnd() * 60;
        S('rect', { x: x0 + 4 + j * 4.3, y: 108 - h, width: 3, height: h, rx: 1, fill: COL.teal, 'fill-opacity': sct % 2 ? 0.55 : 0.85 }, a);
      }
    }
    S('text', { x: 20, y: 138, 'font-size': 14, text: 'position on the lap →' }, a);
    S('text', { x: 500, y: 138, 'font-size': 14, 'text-anchor': 'end', 'font-style': 'italic', text: 'a(s) = throttle, steer for each sector' }, a);
    S('text', { x: 20, y: 22, 'font-size': 13, class: 'tx-mute', text: 'schematic' }, a);

    const b = sl.querySelector('#appr-b');
    b.setAttribute('viewBox', '0 0 520 150');
    S('rect', { x: 205, y: 22, width: 110, height: 62, rx: 6, fill: 'none', stroke: COL.blue, 'stroke-width': 2 }, b);
    const f = S('text', { x: 260, y: 62, 'font-size': 26, 'text-anchor': 'middle', 'font-style': 'italic', class: 'tx-ink' }, b);
    f.textContent = 'f'; S('tspan', { 'font-size': 16, dy: 6, text: 'θ' }, f);
    [['road ahead', 36], ['speed', 54], ['walls', 72]].forEach(([l, y]) => {
      S('line', { x1: 120, x2: 200, y1: y, y2: y, class: 'ax' }, b);
      S('text', { x: 114, y: y + 4.5, 'font-size': 14, 'text-anchor': 'end', text: l }, b);
    });
    S('line', { x1: 320, x2: 392, y1: 53, y2: 53, class: 'ax' }, b);
    S('text', { x: 398, y: 58, 'font-size': 14, text: 'throttle, steer' }, b);
    for (let j = 0; j < 6; j++) {
      const x = 212 + j * 19.2;
      S('circle', { cx: x, cy: 112, r: 7, fill: 'none', stroke: COL.blue, 'stroke-width': 1.6 }, b);
      const ang = (j * 53 % 180 - 90) * Math.PI / 180;
      S('line', { x1: x, y1: 112, x2: x + Math.sin(ang) * 6, y2: 112 - Math.cos(ang) * 6, stroke: COL.ink, 'stroke-width': 1.5 }, b);
    }
    S('text', { x: 330, y: 117, 'font-size': 14, 'font-style': 'italic', text: 'θ: 116 numbers' }, b);
    S('path', { d: 'M440 66 C 470 110, 430 140, 330 132', fill: 'none', stroke: COL.mute, 'stroke-width': 1.3, 'stroke-dasharray': '4 4' }, b);
    S('text', { x: 452, y: 118, 'font-size': 13, class: 'tx-mute', text: 'search' }, b);
  },
};

/* ================= seed suites ================= */
HOOKS['s-eval'] = {
  init(sl) {
    const svg = sl.querySelector('#eval-seeds');
    svg.setAttribute('viewBox', '0 0 600 430');
    const rowY = { off: 40, train: 104, val: 204, live: 268, soak: 332 };
    const dots = (n, perRow, y, col, hollow, r) => {
      for (let i = 0; i < n; i++) {
        const x = 22 + (i % perRow) * 19, yy = y + Math.floor(i / perRow) * 19;
        S('circle', { cx: x, cy: yy, r: r || 6.5, fill: hollow ? 'none' : col, stroke: hollow ? col : 'none', 'stroke-width': 1.6 }, svg);
      }
    };
    const lab = (y, text, sub) => {
      const t = S('text', { x: 14, y: y - 16, 'font-size': 14, class: 'tx-ink', text }, svg);
      if (sub) S('tspan', { class: 'tx-mute', text: '  ' + sub }, t);
    };
    lab(rowY.off, 'official · 2', 'seeds 110 and 2026 — never used to rank until v10'); dots(2, 28, rowY.off, COL.yellow);
    lab(rowY.train, 'training · 28', 'the optimizer ranks on these'); dots(28, 28, rowY.train, COL.blue);
    lab(rowY.val, 'validation · 12', 'untouched until promotion'); dots(12, 28, rowY.val, COL.green);
    lab(rowY.live, 'live grading · 5', '110, 2026 + three new, from Aug 31'); dots(5, 28, rowY.live, COL.ink2, true);
    lab(rowY.soak, 'soak · 100', 'the final gate'); dots(100, 28, rowY.soak, COL.base, false, 4.5);
    const g = S('g', { 'data-s': 1 }, svg);
    S('path', { d: `M12 ${rowY.train + 14} V${rowY.train + 26} H560 V${rowY.train + 14}`, fill: 'none', stroke: COL.blue, 'stroke-width': 1.4 }, g);
    S('text', { x: 14, y: rowY.train + 44, 'font-size': 13, fill: COL.blue, text: 'optimizer’s view: training · + official from v10 · + live grading from v27' }, g);
  },
};

/* ================= two loops ================= */
HOOKS['s-loops'] = {
  init(sl) {
    const svg = sl.querySelector('#loops-svg');
    svg.setAttribute('viewBox', '-50 -10 620 540');
    const c = 260, Ro = 214, Ri = 104;
    const node = (r, deg, text, col, anchor, p) => {
      const a = (deg - 90) * Math.PI / 180, x = c + r * Math.cos(a), y = c + r * Math.sin(a);
      S('circle', { cx: x, cy: y, r: 5, fill: col }, p);
      const out = r > 150 ? 22 : -14;
      S('text', { x: x + Math.cos(a) * out, y: y + Math.sin(a) * out + 5, 'font-size': 14, 'text-anchor': anchor || 'middle', class: 'tx-ink', text }, p);
    };
    const gi = S('g', {}, svg);
    S('circle', { cx: c, cy: c, r: Ri, fill: 'none', stroke: COL.blue, 'stroke-width': 2, class: 'draw', 'data-s': 0 }, gi);
    [['sample', 0], ['race', 90], ['select', 180], ['mutate', 270]].forEach(([t, d]) => node(Ri, d, t, COL.blue, 'middle', gi));
    S('text', { x: c, y: c - 6, 'font-size': 15, 'text-anchor': 'middle', fill: COL.blue, text: '≈ 1 minute' }, gi);
    S('text', { x: c, y: c + 14, 'font-size': 13, 'text-anchor': 'middle', class: 'tx-mute', text: '771 generations' }, gi);
    const go = S('g', { 'data-s': 1 }, svg);
    S('circle', { cx: c, cy: c, r: Ro, fill: 'none', stroke: COL.yellow, 'stroke-width': 2, class: 'draw', 'data-s': 1 }, go);
    [['question', 0, 'middle'], ['hypothesis', 60, 'start'], ['lever + tests', 120, 'start'], ['launch search', 180, 'middle'], ['read checkpoints', 240, 'end'], ['promote · reject', 300, 'end']].forEach(([t, d, an]) => node(Ro, d, t, COL.yellow, an, go));
    S('text', { x: c, y: c + Ri + 56, 'font-size': 13, 'text-anchor': 'middle', class: 'tx-mute', text: 'hours per turn · 20 chats' }, go);
    this.di = S('circle', { r: 7, fill: COL.blue, stroke: COL.bg, 'stroke-width': 2 }, gi);
    this.do = S('circle', { r: 7, fill: COL.yellow, stroke: COL.bg, 'stroke-width': 2 }, go);
    this.c = c; this.Ri = Ri; this.Ro = Ro; this.t = 0;
    this.place();
  },
  place() {
    const a = this.t * 2 * Math.PI / 3 - Math.PI / 2, b = this.t * 2 * Math.PI / 24 - Math.PI / 2;
    this.di.setAttribute('cx', this.c + this.Ri * Math.cos(a)); this.di.setAttribute('cy', this.c + this.Ri * Math.sin(a));
    this.do.setAttribute('cx', this.c + this.Ro * Math.cos(b)); this.do.setAttribute('cy', this.c + this.Ro * Math.sin(b));
  },
  enter() { if (ENV.reduce) return; this.loop = dt => { this.t += dt; this.place(); }; addLoop(this.loop); },
  leave() { if (this.loop) removeLoop(this.loop); },
};

/* ================= CEM vs GA (real checkpoints) ================= */
function plotFrame(svg, dx, dy, xl, yl, fx, fy) {
  svg.setAttribute('viewBox', '0 0 520 400');
  const L = 64, R = 505, T = 14, B = 340;
  const X = v => L + (v - dx[0]) / (dx[1] - dx[0]) * (R - L), Y = v => B - (v - dy[0]) / (dy[1] - dy[0]) * (B - T);
  const id = 'clip' + Math.random().toString(36).slice(2, 7);
  S('rect', { x: L, y: T, width: R - L, height: B - T }, S('clipPath', { id }, S('defs', {}, svg)));
  S('line', { x1: L, x2: R, y1: B, y2: B, class: 'ax' }, svg); S('line', { x1: L, x2: L, y1: T, y2: B, class: 'ax' }, svg);
  fx.forEach(v => { S('line', { x1: X(v), x2: X(v), y1: B, y2: B + 5, class: 'ax' }, svg); S('text', { x: X(v), y: B + 21, 'font-size': 13, 'text-anchor': 'middle', text: String(v) }, svg); });
  fy.forEach(v => { S('line', { x1: L - 5, x2: L, y1: Y(v), y2: Y(v), class: 'ax' }, svg); S('text', { x: L - 9, y: Y(v) + 4.5, 'font-size': 13, 'text-anchor': 'end', text: String(v) }, svg); });
  S('text', { x: (L + R) / 2, y: B + 46, 'font-size': 14, 'text-anchor': 'middle', class: 'tt', text: xl }, svg);
  S('text', { x: 14, y: (T + B) / 2, 'font-size': 14, 'text-anchor': 'middle', class: 'tt', transform: `rotate(-90 14 ${(T + B) / 2})`, text: yl }, svg);
  return { X, Y, L, R, T, B, g: S('g', { 'clip-path': `url(#${id})` }, svg) };
}
HOOKS['s-search'] = {
  init(sl) {
    const cem = DATA.search.cem, ga = DATA.search.ga;
    // CEM panel
    const a = this.A = plotFrame(sl.querySelector('#cem-svg'), [0, 0.3], [0, 0.16], 'center_steer_gain', 'entry offset ratio', [0, 0.1, 0.2, 0.3], [0, 0.05, 0.1, 0.15]);
    const svgA = sl.querySelector('#cem-svg');
    S('line', { x1: a.X(0.1), x2: a.X(0.1), y1: a.T, y2: a.B, stroke: COL.red, 'stroke-dasharray': '5 5', 'stroke-width': 1.3 }, svgA);
    S('text', { x: a.X(0.1) + 6, y: a.T + 14, 'font-size': 13, fill: COL.red, text: 'lower bound' }, svgA);
    this.ghosts = S('g', {}, a.g);
    this.ell = S('ellipse', { fill: COL.blue, 'fill-opacity': 0.12, stroke: COL.blue, 'stroke-width': 2 }, a.g);
    this.mean = S('circle', { r: 4, fill: COL.blue }, a.g);
    this.cemBest = S('path', { d: 'M-7 -7 L7 7 M-7 7 L7 -7', stroke: COL.yellow, 'stroke-width': 2.5, opacity: 0 }, svgA);
    this.cemLbl = S('text', { x: a.R - 4, y: a.T + 14, 'font-size': 13, 'text-anchor': 'end', class: 'tx-ink' }, svgA);
    this.cem = cem.gens;
    // GA panel
    const b = this.Bp = plotFrame(sl.querySelector('#ga-svg'), [11, 16], [1.3, 2.6], 'corner_target_speed_mps', 'front_stop_m', [11, 12, 13, 14, 15, 16], [1.4, 1.8, 2.2, 2.6]);
    const svgB = sl.querySelector('#ga-svg');
    S('rect', { x: b.X(14), y: b.Y(1.6), width: b.X(16) - b.X(14), height: b.Y(1.3) - b.Y(1.6), fill: COL.ink, 'fill-opacity': 0.04, stroke: COL.mute, 'stroke-dasharray': '5 5', 'stroke-width': 1.3 }, svgB);
    S('text', { x: b.X(14) + 6, y: b.Y(1.6) - 7, 'font-size': 13, class: 'tx-mute', text: 'v13’s box' }, svgB);
    this.gaDots = ga.gens[0].pop.map(() => S('circle', { r: 4.2, fill: COL.blue, 'fill-opacity': 0.8, stroke: COL.bg, 'stroke-width': 1 }, b.g));
    this.gaBest = S('path', { d: 'M0 -9 L8 0 L0 9 L-8 0 Z', fill: COL.yellow, stroke: COL.bg, 'stroke-width': 2, opacity: 0 }, svgB);
    this.gaLbl = S('text', { x: b.R - 4, y: b.T + 14, 'font-size': 13, 'text-anchor': 'end', class: 'tx-ink' }, svgB);
    this.gaWin = S('text', { 'font-size': 13, class: 'tx-ink', opacity: 0 }, svgB);
    this.ga = ga.gens;
    this.drawCem(0, 0); this.drawGa(0, 0);
  },
  drawCem(i, e) {
    const g = this.cem, A = this.A, p = g[i], q = g[Math.min(i + 1, g.length - 1)];
    const m0 = lerp(p.mean[0], q.mean[0], e), m1 = lerp(p.mean[1], q.mean[1], e), s0 = lerp(p.dev[0], q.dev[0], e), s1 = lerp(p.dev[1], q.dev[1], e);
    this.ell.setAttribute('cx', A.X(m0)); this.ell.setAttribute('cy', A.Y(m1));
    this.ell.setAttribute('rx', Math.abs(A.X(m0 + s0) - A.X(m0))); this.ell.setAttribute('ry', Math.abs(A.Y(m1 + s1) - A.Y(m1)));
    this.mean.setAttribute('cx', A.X(m0)); this.mean.setAttribute('cy', A.Y(m1));
    this.cemLbl.textContent = `generation ${e > 0.5 ? q.gen : p.gen} of ${g[g.length - 1].gen}`;
  },
  drawGa(i, e) {
    const g = this.ga, B = this.Bp, p = g[i], q = g[Math.min(i + 1, g.length - 1)];
    this.gaDots.forEach((d, k) => {
      const a = p.pop[k] || p.pop[0], b = q.pop[k] || a;
      d.setAttribute('cx', B.X(lerp(a[0], b[0], e)).toFixed(1)); d.setAttribute('cy', B.Y(lerp(a[1], b[1], e)).toFixed(1));
    });
    this.gaLbl.textContent = `generation ${e > 0.5 ? q.gen : p.gen} of ${g[g.length - 1].gen}`;
  },
  play(kind) {
    const g = kind === 'cem' ? this.cem : this.ga, draw = (kind === 'cem' ? this.drawCem : this.drawGa).bind(this);
    if (this['stop' + kind]) this['stop' + kind]();
    let i = 0;
    if (kind === 'cem') this.ghosts.replaceChildren();
    const nextGen = () => {
      if (i >= g.length - 1) { this.finish(kind); return; }
      if (kind === 'cem') {
        const gh = this.ell.cloneNode(); gh.setAttribute('fill-opacity', 0); gh.setAttribute('stroke-opacity', 0.25); gh.setAttribute('stroke-width', 1); this.ghosts.appendChild(gh);
      }
      this['stop' + kind] = tween(kind === 'cem' ? 520 : 430, e => draw(i, e), () => { i++; nextGen(); });
    };
    nextGen();
  },
  finish(kind) {
    if (kind === 'cem') {
      const b = this.cem[this.cem.length - 1].best;
      this.cemBest.setAttribute('transform', `translate(${this.A.X(b[0])},${this.A.Y(b[1])})`); this.cemBest.setAttribute('opacity', 1);
    } else {
      const b = this.ga[this.ga.length - 1].best, B = this.Bp;
      this.gaBest.setAttribute('transform', `translate(${B.X(b[0])},${B.Y(b[1])})`); this.gaBest.setAttribute('opacity', 1);
      this.gaWin.setAttribute('x', B.X(b[0]) - 12); this.gaWin.setAttribute('y', B.Y(b[1]) - 16); this.gaWin.setAttribute('text-anchor', 'end');
      this.gaWin.textContent = `winner: ${b[0].toFixed(2)} m/s, ${b[1].toFixed(2)} m`; this.gaWin.setAttribute('opacity', 1);
    }
  },
  step(k) {
    if (k < 0) { ['cem', 'ga'].forEach(x => this['stop' + x] && this['stop' + x]()); this.cemBest.setAttribute('opacity', 0); this.gaBest.setAttribute('opacity', 0); this.gaWin.setAttribute('opacity', 0); this.ghosts.replaceChildren(); this.drawCem(0, 0); this.drawGa(0, 0); return; }
    if (ENV.reader || ENV.all) { this.drawCem(this.cem.length - 1, 0); this.finish('cem'); this.drawGa(this.ga.length - 1, 0); this.finish('ga'); return; }
    if (k === 1) this.play('cem');
    if (k === 2) this.play('ga');
  },
};

/* ================= curvature failure ================= */
HOOKS['s-curv'] = {
  init(sl) {
    const svg = sl.querySelector('#curv-svg');
    svg.setAttribute('viewBox', '0 0 460 520');
    const k = 24, x0 = 110, y0 = 486, R = 24, c = d => d * d / (2 * R);
    const P = d => [x0 + c(d) * k, y0 - d * k];
    const road = []; for (let d = -1; d <= 18.5; d += 0.25) road.push(P(d));
    S('path', { d: openD(road), class: 'road', 'stroke-width': 6.6 * k }, svg);
    S('path', { d: openD(road), class: 'cl' }, svg);
    S('path', { d: 'M0 -16 L9 12 L0 7 L-9 12 Z', fill: COL.ink, transform: `translate(${x0},${y0})` }, svg);
    S('text', { x: 320, y: 470, 'font-size': 16, 'font-style': 'italic', class: 'tx-mute', text: 'radius R = 24 m' }, svg);
    const g1 = S('g', { 'data-s': 1 }, svg);
    S('line', { x1: x0, x2: x0, y1: y0 - 18, y2: y0 - 17 * k, stroke: COL.mute, 'stroke-dasharray': '4 5', 'stroke-width': 1.2 }, g1);
    const D = [4, 9, 16];
    D.forEach((d, i) => {
      const [x, y] = P(d);
      S('line', { x1: x0, x2: x, y1: y, y2: y, stroke: COL.red, 'stroke-width': 2 }, g1);
      S('circle', { cx: x, cy: y, r: 5.5, fill: COL.ink }, g1);
      const t = S('text', { x: x + 10, y: y + 5, 'font-size': 17, 'font-style': 'italic', class: 'tx-ink' }, g1);
      t.textContent = 'c'; S('tspan', { 'font-size': 12, dy: 4, text: String(i) }, t);
      S('text', { x: x0 - 10, y: y + 5, 'font-size': 14, 'text-anchor': 'end', text: 'd = ' + d + ' m' }, g1);
    });
    const g3 = S('g', { 'data-s': 3 }, svg);
    const pts = [[x0, y0]].concat(D.map(P));
    S('path', { d: openD(pts), fill: 'none', stroke: COL.green, 'stroke-width': 2.4, class: 'draw', 'data-s': 3 }, g3);
    S('text', { x: 300, y: 150, 'font-size': 14, fill: COL.green, text: 'segment slopes' }, g3);

    const b = sl.querySelector('#curv-bars');
    b.setAttribute('viewBox', '0 0 560 190');
    const L = 150, Rr = 540, row = (i) => 24 + i * 32, X = v => L + v / 4.2 * (Rr - L);
    S('text', { x: L, y: 12, 'font-size': 13, class: 'tx-mute', text: 'how much sharper the point looks than the nearest one' }, b);
    const cd = D.map(d => c(d) / d), rel = cd.map(v => v / cd[0]);
    [1, 2, 3, 4].forEach(v => { S('line', { x1: X(v), x2: X(v), y1: 18, y2: 176, class: 'gr' }, b); S('text', { x: X(v), y: 188, 'font-size': 12, 'text-anchor': 'middle', text: v + '×' }, b); });
    const gr = S('g', {}, b);
    rel.forEach((v, i) => {
      S('rect', { x: L, y: row(i), width: X(v) - L, height: 20, rx: 3, fill: COL.red }, gr);
      S('text', { x: L - 10, y: row(i) + 15, 'font-size': 14, 'text-anchor': 'end', text: `c/d at ${D[i]} m` }, gr);
      S('text', { x: X(v) + 6, y: row(i) + 15, 'font-size': 13, class: 'tx-ink', text: v.toFixed(2) + '×' }, gr);
    });
    const gg = S('g', { 'data-s': 3 }, b);
    ['κ near', 'κ far'].forEach((l, i) => {
      const y = row(3) + i * 0 + (i ? 26 : 0) + 8;
      S('rect', { x: L, y, width: X(1) - L, height: 18, rx: 3, fill: COL.green }, gg);
      S('text', { x: L - 10, y: y + 14, 'font-size': 14, 'text-anchor': 'end', 'font-style': 'italic', text: l }, gg);
      S('text', { x: X(1) + 6, y: y + 14, 'font-size': 13, class: 'tx-ink', text: '1.00× — same curve, same answer' }, gg);
    });
  },
};

/* ================= learned: where the seconds came from ================= */
HOOKS['s-learned'] = {
  init(sl) {
    const svg = sl.querySelector('#learned-bars');
    svg.setAttribute('viewBox', '0 0 640 330');
    const rows = [
      { label: '28.2 → 9.0 s', when: 'Aug 28', s: 12.70, n: 6.50, sv: 'hand-tuned → CEM ×2', nv: 'coasting + racing line, curvature fix' },
      { label: '9.0 → 7.9 s', when: 'Aug 29', s: 0.483, n: 0.617, sv: 'v3, v5 bounds', nv: 'v4 line-release gene' },
      { label: '7.9 → 7.18 s', when: 'Aug 29–31', s: 0.116, n: 0.600, sv: 'v7, v9 objectives', nv: 'launch cap, sweeper ×2, corridor, drift' },
    ];
    const L = 150, R = 630;
    const lg = S('g', { transform: 'translate(150,14)' }, svg);
    S('rect', { x: 0, y: -9, width: 14, height: 10, rx: 2, fill: COL.blue }, lg); S('text', { x: 20, y: 0, 'font-size': 13, class: 'tx-ink', text: 'search inside a fixed controller' }, lg);
    S('rect', { x: 238, y: -9, width: 14, height: 10, rx: 2, fill: COL.yellow }, lg); S('text', { x: 258, y: 0, 'font-size': 13, class: 'tx-ink', text: 'new structure: behaviour, feature, fix' }, lg);
    rows.forEach((r, i) => {
      const y = 46 + i * 94, tot = r.s + r.n, xs = L + (r.s / tot) * (R - L);
      S('text', { x: L - 12, y: y + 20, 'font-size': 16, 'text-anchor': 'end', class: 'tx-ink', text: r.label }, svg);
      S('text', { x: L - 12, y: y + 38, 'font-size': 13, 'text-anchor': 'end', text: r.when }, svg);
      S('rect', { x: L, y, width: Math.max(0, xs - L - 1), height: 30, rx: 3, fill: COL.blue }, svg);
      S('rect', { x: xs + 1, y, width: Math.max(0, R - xs - 1), height: 30, rx: 3, fill: COL.yellow }, svg);
      const pct = Math.round(100 * r.s / tot);
      S('text', { x: L, y: y + 50, 'font-size': 13, class: 'tx-ink', text: `${r.s.toFixed(r.s < 1 ? 2 : 1)} s · ${pct}%` }, svg);
      S('text', { x: L, y: y + 67, 'font-size': 12, text: r.sv }, svg);
      S('text', { x: R, y: y + 50, 'font-size': 13, 'text-anchor': 'end', class: 'tx-ink', text: `${r.n.toFixed(r.n < 1 ? 2 : 1)} s · ${100 - pct}%` }, svg);
      S('text', { x: R, y: y + 67, 'font-size': 12, 'text-anchor': 'end', text: r.nv }, svg);
    });
  },
};
