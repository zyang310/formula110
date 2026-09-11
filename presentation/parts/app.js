/* ================= core helpers ================= */
const NS = 'http://www.w3.org/2000/svg';
const COL = {
  bg: '#0D0F13', line: '#232A33', line2: '#39424F', mute: '#8C96A4', ink: '#EEF1F4', ink2: '#C9D1DB',
  blue: '#58C4DD', yellow: '#F4D345', green: '#83C167', red: '#FC6255', teal: '#5CD0B3',
  purple: '#A77BD6', brown: '#D9904A', base: '#7D8795', road: '#1A1F27',
};
function S(tag, a, p) {
  const e = document.createElementNS(NS, tag);
  for (const k in a || {}) {
    if (k === 'text') e.textContent = a[k];
    else if (a[k] != null) e.setAttribute(k, a[k]);
  }
  if (p) p.appendChild(e);
  return e;
}
function H(tag, a, p) {
  const e = document.createElement(tag);
  for (const k in a || {}) {
    if (k === 'text') e.textContent = a[k];
    else if (k === 'html') e.innerHTML = a[k];
    else if (a[k] != null) e.setAttribute(k, a[k]);
  }
  if (p) p.appendChild(e);
  return e;
}
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const HOOKS = {};
const ENV = {
  reader: matchMedia('(max-width: 820px)').matches,
  reduce: matchMedia('(prefers-reduced-motion: reduce)').matches,
  all: new URLSearchParams(location.search).has('all'),
};
function prepDraw(root) {
  root.querySelectorAll('.draw').forEach(el => {
    try { el.style.setProperty('--len', el.getTotalLength().toFixed(1)); } catch (e) { /* not a geometry element */ }
  });
}
function openD(pts) { return 'M' + pts.map(p => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join('L'); }
function closedD(pts) { return openD(pts) + 'Z'; }

/* time: labels in EDT (UTC-4), independent of the viewer's zone */
const EDT = -4 * 3600;
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
function edt(t) { return new Date((t + EDT) * 1000); }
function fmtDay(t) { const d = edt(t); return MON[d.getUTCMonth()] + ' ' + d.getUTCDate(); }
function fmtHM(t) { const d = edt(t); return String(d.getUTCHours()).padStart(2, '0') + ':' + String(d.getUTCMinutes()).padStart(2, '0'); }
function edtEpoch(mo, d, h, mi) { return Date.UTC(2026, mo - 1, d, h, mi || 0) / 1000 - EDT; }
function parseWhen(s) {
  if (!s || s === 'all') return null;
  return s.split('|').map(x => {
    const m = x.match(/(\d+)-(\d+) (\d+):(\d+)/);
    return edtEpoch(+m[1], +m[2], +m[3], +m[4]);
  });
}

/* one rAF runner for every animation on the page */
const LOOPS = new Set();
let rafId = 0, lastTs = 0;
function runLoops(ts) {
  const dt = lastTs ? Math.min(0.1, (ts - lastTs) / 1000) : 0;
  lastTs = ts;
  LOOPS.forEach(f => f(dt));
  if (LOOPS.size) rafId = requestAnimationFrame(runLoops);
  else { rafId = 0; lastTs = 0; }
}
function addLoop(f) { LOOPS.add(f); if (!rafId) { lastTs = 0; rafId = requestAnimationFrame(runLoops); } }
function removeLoop(f) { LOOPS.delete(f); }
function tween(ms, fn, done) {
  if (ENV.reduce || ENV.reader || ENV.all || ms <= 0) { fn(1); if (done) done(); return () => {}; }
  let el = 0;
  const f = dt => {
    el += dt * 1000;
    const k = Math.min(1, el / ms);
    fn(smooth(k));
    if (k >= 1) { removeLoop(f); if (done) done(); }
  };
  addLoop(f);
  return () => removeLoop(f);
}

/* ================= deck ================= */
let SLIDES = [], cur = 0, step = 0;
const maxStep = sl => +sl.dataset.steps || 0;
function setSteps(sl, k) {
  sl.querySelectorAll('[data-s]').forEach(e => e.classList.toggle('on', +e.dataset.s <= k));
  const h = HOOKS[sl.id];
  if (h && h.step) h.step(k);
}
function show(i, k) {
  i = clamp(i, 0, SLIDES.length - 1);
  const was = SLIDES[cur], sl = SLIDES[i];
  if (was !== sl || !sl.classList.contains('cur')) {
    if (was && was !== sl) {
      was.classList.remove('cur');
      const h = HOOKS[was.id];
      if (h && h.leave) h.leave();
      setSteps(was, -1);
    }
    sl.classList.add('cur');
    const h = HOOKS[sl.id];
    if (h && h.enter) h.enter();
  }
  cur = i;
  step = clamp(k, 0, maxStep(sl));
  setSteps(sl, step);
  document.getElementById('rb-count').innerHTML = '<b>' + (i + 1) + '</b> / ' + SLIDES.length;
  if (window.RIBBON) RIBBON.update(sl, step);
  fillNotes();
  try { history.replaceState(null, '', location.pathname + location.search + '#' + (i + 1)); } catch (e) { /* sandboxed frame */ }
}
function next() {
  const sl = SLIDES[cur];
  if (step < maxStep(sl)) { step++; setSteps(sl, step); if (window.RIBBON) RIBBON.update(sl, step); }
  else if (cur < SLIDES.length - 1) show(cur + 1, ENV.all ? maxStep(SLIDES[cur + 1]) : 0);
}
function prev() {
  const sl = SLIDES[cur];
  if (step > 0) { step--; setSteps(sl, step); if (window.RIBBON) RIBBON.update(sl, step); }
  else if (cur > 0) show(cur - 1, maxStep(SLIDES[cur - 1]));
}
function fillNotes() {
  const box = document.getElementById('notes');
  const n = SLIDES[cur].querySelector('aside.notes');
  document.getElementById('notes-title').textContent = 'Speaker notes · slide ' + (cur + 1);
  document.getElementById('notes-body').innerHTML = n ? n.innerHTML : '';
  box.hidden = !(document.getElementById('rb-notes').getAttribute('aria-pressed') === 'true');
}
function toggleNotes() {
  const b = document.getElementById('rb-notes');
  b.setAttribute('aria-pressed', b.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
  fillNotes();
}

function boot() {
  SLIDES = [...document.querySelectorAll('.slide')];
  SLIDES.forEach(sl => { const h = HOOKS[sl.id]; if (h && h.init) { try { h.init(sl); } catch (e) { console.error(sl.id, e); } } });
  prepDraw(document);
  if (window.RIBBON) RIBBON.init();

  if (ENV.reader) {
    SLIDES.forEach(sl => { sl.classList.add('cur'); setSteps(sl, maxStep(sl)); });
    const io = new IntersectionObserver(es => es.forEach(en => {
      const h = HOOKS[en.target.id];
      if (!h) return;
      if (en.isIntersecting) { if (h.enter) h.enter(); } else if (h.leave) h.leave();
    }), { threshold: 0.2 });
    SLIDES.forEach(sl => io.observe(sl));
    return;
  }

  const hi = parseInt((location.hash || '#1').slice(1), 10);
  cur = 0;
  show(isNaN(hi) ? 0 : hi - 1, ENV.all ? maxStep(SLIDES[isNaN(hi) ? 0 : hi - 1]) : 0);

  document.getElementById('rb-next').addEventListener('click', next);
  document.getElementById('rb-prev').addEventListener('click', prev);
  document.getElementById('rb-notes').addEventListener('click', toggleNotes);
  addEventListener('keydown', e => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (k === 'ArrowRight' || k === 'PageDown' || k === ' ' || k === 'Enter') { if (k === 'Enter' && e.target.tagName === 'BUTTON') return; e.preventDefault(); next(); }
    else if (k === 'ArrowLeft' || k === 'PageUp' || k === 'Backspace') { e.preventDefault(); prev(); }
    else if (k === 'Home') show(0, 0);
    else if (k === 'End') show(SLIDES.length - 1, 0);
    else if (k === 'n' || k === 'N') toggleNotes();
    else if (k === 'f' || k === 'F') {
      try { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); } catch (err) { /* blocked in some frames */ }
    }
  });
  let tx = null;
  addEventListener('touchstart', e => { tx = e.touches[0].clientX; }, { passive: true });
  addEventListener('touchend', e => {
    if (tx == null) return;
    const dx = e.changedTouches[0].clientX - tx;
    tx = null;
    if (Math.abs(dx) > 60) (dx < 0 ? next : prev)();
  });
}
