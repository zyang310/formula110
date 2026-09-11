/* ================= track geometry (1 m = 10 units, y = -z) ================= */
const TRK = (() => {
  const c = DATA.track.c, n = c.length;
  const P = c.map(([x, z]) => [x * 10, -z * 10]);
  const N = P.map((p, i) => {
    const a = P[(i - 1 + n) % n], b = P[(i + 1) % n];
    const tx = b[0] - a[0], ty = b[1] - a[1], L = Math.hypot(tx, ty) || 1;
    return [-ty / L, tx / L];
  });
  const off = d => P.map((p, i) => [p[0] + N[i][0] * d * 10, p[1] + N[i][1] * d * 10]);
  const xs = P.map(p => p[0]), ys = P.map(p => p[1]);
  const box = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  const s = [0];
  for (let i = 1; i <= n; i++) { const a = P[i - 1], b = P[i % n]; s.push(s[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1])); }
  return { P, N, off, box, s, n };
})();

function drawTrackBase(svg, margin) {
  const [x0, y0, x1, y1] = TRK.box, m = margin == null ? 60 : margin;
  svg.setAttribute('viewBox', `${(x0 - m).toFixed(0)} ${(y0 - m).toFixed(0)} ${(x1 - x0 + 2 * m).toFixed(0)} ${(y1 - y0 + 2 * m).toFixed(0)}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  const g = S('g', {}, svg);
  S('path', { d: closedD(TRK.off(4.7)), class: 'wall' }, g);
  S('path', { d: closedD(TRK.off(-4.7)), class: 'wall' }, g);
  S('path', { d: closedD(TRK.P), class: 'road', 'stroke-width': 66 }, g);
  S('path', { d: closedD(TRK.off(3.3)), class: 'edge' }, g);
  S('path', { d: closedD(TRK.off(-3.3)), class: 'edge' }, g);
  S('path', { d: closedD(TRK.P), class: 'cl' }, g);
  return g;
}

function carPath(key) {
  const d = DATA.paths[key], xy = d.xy, m = xy.length / 3, pts = new Array(m);
  for (let i = 0; i < m; i++) pts[i] = [xy[3 * i] * 10, -xy[3 * i + 1] * 10, xy[3 * i + 2]];
  const cr = d.crossings || [];
  return { pts, dur: (m - 1) / 30, meta: d, laps: cr.map((c, i) => c - (i ? cr[i - 1] : 0)) };
}
const PATHS = { minimum: carPath('minimum'), v27: carPath('v27'), human: carPath('human') };
function posAt(p, t) {
  const f = clamp(t * 30, 0, p.pts.length - 1), i = Math.floor(f), k = f - i;
  const a = p.pts[i], b = p.pts[Math.min(i + 1, p.pts.length - 1)];
  return [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k)];
}
function trailPts(p, t, span) {
  const i1 = Math.floor(clamp(t * 30, 0, p.pts.length - 1)), i0 = Math.max(0, i1 - Math.round(span * 30));
  const pts = p.pts.slice(i0, i1 + 1);
  pts.push(posAt(p, t));
  return pts;
}
function mkCar(svg, key, color, r) {
  const p = PATHS[key];
  const trail = S('path', { fill: 'none', stroke: color, 'stroke-width': 6, 'stroke-opacity': 0.45, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, svg);
  const dot = S('circle', { r: r || 10, fill: color, stroke: COL.bg, 'stroke-width': 3 }, svg);
  return {
    p, trail, dot,
    update(t) {
      const tt = Math.min(t, p.dur), q = posAt(p, tt);
      dot.setAttribute('cx', q[0].toFixed(1));
      dot.setAttribute('cy', q[1].toFixed(1));
      trail.setAttribute('d', openD(trailPts(p, tt, 1.4)));
      const over = t > p.dur + 0.05;
      dot.setAttribute('opacity', over ? 0.35 : 1);
      trail.setAttribute('opacity', over ? 0 : 1);
      return q;
    },
    lapsAt(t) { return (p.meta.crossings || []).filter(c => c <= t).length; },
  };
}
const SPEED_STOPS = [[10, [31, 78, 110]], [17, [58, 156, 192]], [24, [126, 211, 200]], [32, [244, 211, 69]]];
function speedColor(v) {
  const s = SPEED_STOPS;
  if (v <= s[0][0]) return `rgb(${s[0][1]})`;
  for (let i = 1; i < s.length; i++) {
    if (v <= s[i][0]) {
      const k = (v - s[i - 1][0]) / (s[i][0] - s[i - 1][0]), a = s[i - 1][1], b = s[i][1];
      return `rgb(${Math.round(lerp(a[0], b[0], k))},${Math.round(lerp(a[1], b[1], k))},${Math.round(lerp(a[2], b[2], k))})`;
    }
  }
  return `rgb(${s[s.length - 1][1]})`;
}
function readRow(box, color, name) {
  const row = H('div', {}, box);
  H('span', { class: 'sw', style: 'background:' + color }, row);
  H('span', { text: name }, row);
  const v = H('span', { class: 'c-mute', style: 'margin-left:.6em' }, row);
  return v;
}

/* ================= title ================= */
HOOKS['s-title'] = {
  init(sl) {
    const svg = sl.querySelector('#title-svg');
    drawTrackBase(svg, 40);
    const p = PATHS.v27, c = p.meta.crossings;
    S('path', { d: openD(p.pts.slice(Math.round(c[0] * 30), Math.round(c[1] * 30) + 1)), fill: 'none', stroke: COL.yellow, 'stroke-opacity': 0.2, 'stroke-width': 4, 'stroke-linejoin': 'round' }, svg);
    const st = p.pts[0];
    S('circle', { cx: st[0], cy: st[1], r: 22, fill: 'none', stroke: COL.mute, 'stroke-width': 1.5, 'stroke-dasharray': '3 5' }, svg);
    S('text', { x: st[0] + 30, y: st[1] + 50, 'font-size': 17, class: 'tx-mute', text: 'seed 110 start' }, svg);
    this.cars = [mkCar(svg, 'minimum', COL.base), mkCar(svg, 'v27', COL.yellow)];
    const box = sl.querySelector('#title-read');
    this.clock = H('div', { class: 'c-mute' }, box);
    this.rows = [readRow(box, COL.base, 'minimum_viable'), readRow(box, COL.yellow, 'race_faster · v27')];
    this.t = 0;
    this.draw(ENV.reduce ? 22 : 0);
  },
  draw(t) {
    this.cars.forEach((car, i) => {
      car.update(t);
      const n = car.lapsAt(t), laps = car.p.laps;
      this.rows[i].textContent = n ? 'lap ' + laps.slice(0, n).map(x => x.toFixed(2) + ' s').join(', ') : 'lap 1 in progress';
    });
    this.clock.textContent = 't = ' + t.toFixed(1) + ' s of 30';
  },
  enter() {
    if (ENV.reduce) return;
    this.t = 0;
    this.loop = dt => { this.t += dt; if (this.t > 31.5) this.t = 0; this.draw(Math.min(this.t, 30)); };
    addLoop(this.loop);
  },
  leave() { if (this.loop) removeLoop(this.loop); },
};

/* ================= controller map ================= */
HOOKS['s-controller'] = {
  init(sl) {
    const svg = sl.querySelector('#ctrl-svg');
    drawTrackBase(svg, 40);
    const p = PATHS.v27, c = p.meta.crossings, i0 = Math.round(c[0] * 30), i1 = Math.round(c[1] * 30);
    const lap = p.pts.slice(i0, i1 + 1);

    // 1 · lookahead at the tightest corner: car 14 m before max turning angle
    const n = TRK.n, P = TRK.P;
    let best = 0, bi = 0;
    for (let i = 0; i < n; i++) {
      let a = 0;
      for (let k = -3; k <= 3; k++) {
        const A = P[(i + k - 1 + n) % n], B = P[(i + k + n) % n], C = P[(i + k + 1 + n) % n];
        const h1 = Math.atan2(B[1] - A[1], B[0] - A[0]), h2 = Math.atan2(C[1] - B[1], C[0] - B[0]);
        let d = h2 - h1; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
        a += Math.abs(d);
      }
      if (a > best) { best = a; bi = i; }
    }
    const walk = (i, dist) => { // walk along centre line by dist units (signed)
      let j = i, acc = 0; const dir = dist >= 0 ? 1 : -1;
      while (acc < Math.abs(dist)) { const k = (j + dir + n) % n; acc += Math.hypot(P[k][0] - P[j][0], P[k][1] - P[j][1]); j = k; }
      return j;
    };
    const ci = walk(bi, -140);
    let carI = 0, bd = 1e18;
    lap.forEach((q, k) => { const d = Math.hypot(q[0] - P[ci][0], q[1] - P[ci][1]); if (d < bd) { bd = d; carI = k; } });
    const car = lap[carI];
    const g1 = S('g', { 'data-s': 1 }, svg);
    [40, 90, 160].forEach((dd, k) => {
      const q = P[walk(ci, dd)];
      S('line', { x1: car[0], y1: car[1], x2: q[0], y2: q[1], stroke: COL.blue, 'stroke-width': 1.5, 'stroke-opacity': 0.7 }, g1);
      S('circle', { cx: q[0], cy: q[1], r: 6, fill: COL.blue }, g1);
      S('text', { x: q[0] + 10, y: q[1] - 10, 'font-size': 17, class: 'tx-ink', 'font-style': 'italic', text: ['4 m', '9 m', '16 m'][k] }, g1);
    });
    S('circle', { cx: car[0], cy: car[1], r: 10, fill: COL.ink, stroke: COL.bg, 'stroke-width': 3 }, g1);

    // 2 · the line itself vs the baseline
    const mp = PATHS.minimum.pts;
    S('path', { d: openD(mp), fill: 'none', stroke: COL.base, 'stroke-width': 3, 'stroke-opacity': 0.8, class: 'draw', 'data-s': 2 }, svg);
    S('path', { d: openD(lap), fill: 'none', stroke: COL.ink2, 'stroke-width': 4.5, 'stroke-linejoin': 'round', class: 'draw', 'data-s': 2 }, svg);

    // 3 · speed colour
    const g3 = S('g', { 'data-s': 3 }, svg);
    for (let k = 1; k < lap.length; k++) {
      const a = lap[k - 1], b = lap[k];
      S('line', { x1: a[0].toFixed(1), y1: a[1].toFixed(1), x2: b[0].toFixed(1), y2: b[1].toFixed(1), stroke: speedColor((a[2] + b[2]) / 2), 'stroke-width': 7, 'stroke-linecap': 'round' }, g3);
    }
    // 4 · drift pulses (negative throttle) inside the flying lap
    const g4 = S('g', { 'data-s': 4 }, svg);
    const thr = p.meta.thr, from = Math.round(c[0] * 60), to = Math.round(c[1] * 60);
    const clusters = [];
    for (let t = from; t <= to; t++) {
      if (thr[t] < 0) { const last = clusters[clusters.length - 1]; if (last && t - last[1] <= 3) last[1] = t; else clusters.push([t, t]); }
    }
    clusters.forEach(([a, b]) => {
      const q = posAt(p, (a + b) / 120);
      S('circle', { cx: q[0], cy: q[1], r: 34, fill: 'none', stroke: COL.ink, 'stroke-width': 1.6, 'stroke-dasharray': '4 4' }, g4);
      S('text', { x: q[0] + 42, y: q[1] + 6, 'font-size': 18, class: 'tx-ink', text: `drift pulse · ${((b - a + 1) / 60).toFixed(2)} s` }, g4);
    });
    const lg = sl.querySelector('#ctrl-legend');
    lg.innerHTML = `<span><span class="sw" style="background:${COL.base}"></span>baseline, <code>minimum_viable</code></span>` +
      `<span>speed <span style="display:inline-block;width:7em;height:.55em;border-radius:3px;vertical-align:middle;margin:0 .45em;background:linear-gradient(90deg,${speedColor(10)},${speedColor(17)},${speedColor(24)},${speedColor(32)})"></span>10 → 32 m/s</span>` +
      `<span class="c-mute">one flying lap of v27, 7.20 s</span>`;
  },
};

/* ================= representative race ================= */
HOOKS['s-race'] = {
  init(sl) {
    const svg = sl.querySelector('#race-svg');
    drawTrackBase(svg, 30);
    this.cars = [mkCar(svg, 'minimum', COL.base), mkCar(svg, 'human', COL.brown), mkCar(svg, 'v27', COL.yellow)];
    const box = sl.querySelector('#race-read');
    this.rows = [
      readRow(box, COL.base, 'minimum_viable'),
      readRow(box, COL.brown, 'Human (Zhi), replayed'),
      readRow(box, COL.yellow, 'race_faster · v27'),
    ];
    // throttle strip
    const st = sl.querySelector('#race-strip');
    const W = 1000, Hh = 92, X0 = 70, X1 = 930;
    st.setAttribute('viewBox', `0 0 ${W} ${Hh}`);
    const xt = t => X0 + (t / 30) * (X1 - X0), yv = v => 38 - v * 28;
    this.xt = xt;
    S('line', { x1: X0, x2: X1, y1: yv(0), y2: yv(0), class: 'ax' }, st);
    [1, -1].forEach(v => S('line', { x1: X0, x2: X1, y1: yv(v), y2: yv(v), class: 'gr' }, st));
    [['+1', 1], ['0', 0], ['−1', -1]].forEach(([l, v]) => S('text', { x: X0 - 10, y: yv(v) + 5, 'font-size': 14, 'text-anchor': 'end', text: l }, st));
    S('text', { x: 4, y: 20, 'font-size': 14, class: 'tx-ink', text: 'throttle' }, st);
    for (let s = 0; s <= 30; s += 5) S('text', { x: xt(s), y: Hh - 2, 'font-size': 13, 'text-anchor': 'middle', text: s + ' s' }, st);
    const line = (arr, col) => {
      const pts = arr.map((v, i) => [xt(i / 60), yv(v / 100)]);
      S('path', { d: openD(pts), fill: 'none', stroke: col, 'stroke-width': 1.6, 'vector-effect': 'non-scaling-stroke', 'stroke-linejoin': 'round' }, st);
    };
    line(DATA.paths.human.thr, COL.brown);
    line(DATA.paths.v27.thr, COL.yellow);
    S('text', { x: X1 + 8, y: yv(0.95) + 5, 'font-size': 13, text: 'v27', fill: COL.ink2 }, st);
    S('text', { x: xt(10.5) + 6, y: yv(-0.8), 'font-size': 13, text: 'human ends', fill: COL.ink2 }, st);
    this.head = S('line', { y1: 4, y2: 74, stroke: COL.ink, 'stroke-width': 1.2 }, st);

    this.t = 0; this.rate = 1; this.playing = true; this.hold = 0;
    const $ = id => sl.querySelector('#' + id);
    this.btnPlay = $('race-play'); this.scrub = $('race-scrub'); this.lbl = $('race-t');
    this.btnPlay.addEventListener('click', () => { this.playing = !this.playing; this.sync(); });
    $('race-restart').addEventListener('click', () => { this.t = 0; this.playing = true; this.draw(); this.sync(); });
    [['race-sp-05', 0.5], ['race-sp-1', 1], ['race-sp-2', 2]].forEach(([id, r]) => $(id).addEventListener('click', () => {
      this.rate = r;
      ['race-sp-05', 'race-sp-1', 'race-sp-2'].forEach(x => $(x).setAttribute('aria-pressed', String(x === id)));
    }));
    this.scrub.addEventListener('input', () => { this.t = +this.scrub.value; this.draw(); });
    this.draw();
  },
  sync() {
    this.btnPlay.textContent = this.playing ? 'Pause' : 'Play';
    this.btnPlay.setAttribute('aria-pressed', String(!this.playing));
  },
  draw() {
    const t = this.t;
    this.cars.forEach((car, i) => {
      const q = car.update(t), n = car.lapsAt(t), laps = car.p.laps;
      let s = (t <= car.p.dur ? q[2].toFixed(0) + ' m/s' : 'recording ended at ' + car.p.dur.toFixed(1) + ' s');
      if (n) s += ' · lap ' + laps.slice(0, n).map(x => x.toFixed(2) + ' s').join(', ');
      this.rows[i].textContent = s;
    });
    const x = this.xt(t).toFixed(1);
    this.head.setAttribute('x1', x); this.head.setAttribute('x2', x);
    this.scrub.value = t.toFixed(2);
    this.lbl.textContent = t.toFixed(1) + ' s';
  },
  enter() {
    this.sync();
    if (ENV.reduce) { this.t = 12; this.draw(); return; }
    this.loop = dt => {
      if (!this.playing) return;
      if (this.t >= 30) { this.hold += dt; if (this.hold > 1.5) { this.t = 0; this.hold = 0; } }
      else this.t = Math.min(30, this.t + dt * this.rate);
      this.draw();
    };
    addLoop(this.loop);
  },
  leave() { if (this.loop) removeLoop(this.loop); },
};
