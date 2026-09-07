// Shared by index.html and video.html: the tooltip, the legend, and the
// line chart with its crosshair. Plain script; everything hangs off `Charts`.
// The page supplies <div id="tip"> and the .linechart/.legend styles
// (style.css).
(function () {
  const SVGNS = 'http://www.w3.org/2000/svg';
  const svgEl = (tag, attrs) => {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
    return n;
  };
  const tipEl = () => document.getElementById('tip');

  // --- tooltip ------------------------------------------------------------
  function moveTip(ev) {
    const tip = tipEl();
    const r = tip.getBoundingClientRect();
    let x = (ev.clientX ?? 0) + 14, y = (ev.clientY ?? 0) + 16;
    if (x + r.width > innerWidth - 8) x = innerWidth - r.width - 8;
    if (y + r.height > innerHeight - 8) y = (ev.clientY ?? 0) - r.height - 12;
    tip.style.left = x + 'px';
    tip.style.top = Math.max(8, y) + 'px';
  }
  const hideTip = () => { tipEl().style.opacity = '0'; };

  // One value with a coloured key, a name line, a meta line.
  function showTip(ev, color, valueText, nameText, metaText) {
    const tip = tipEl();
    tip.innerHTML = '';
    const v = document.createElement('div');
    v.className = 't-val';
    const key = document.createElement('span');
    key.className = 'key';
    key.style.background = color;
    v.append(key, document.createTextNode(valueText));
    const n = document.createElement('div');
    n.className = 't-name';
    n.textContent = nameText;
    const m = document.createElement('div');
    m.className = 't-meta';
    m.textContent = metaText;
    tip.append(v, n, m);
    tip.style.opacity = '1';
    moveTip(ev);
  }

  // A title and one line per series, for the crosshair.
  function showRowsTip(ev, title, rows) {
    const tip = tipEl();
    tip.innerHTML = '';
    const h = document.createElement('div');
    h.className = 't-name';
    h.style.marginTop = '0';
    h.textContent = title;
    tip.appendChild(h);
    for (const r of rows) {
      const line = document.createElement('div');
      line.className = 't-val';
      line.style.fontSize = '13px';
      line.style.marginTop = '4px';
      const key = document.createElement('span');
      key.className = 'key';
      key.style.background = r.color;
      line.append(key, document.createTextNode(r.value));
      const sub = document.createElement('div');
      sub.className = 't-meta';
      sub.style.marginTop = '1px';
      sub.textContent = r.label;
      tip.append(line, sub);
    }
    tip.style.opacity = '1';
    moveTip(ev);
  }

  // --- legend (identity never rests on colour alone) -----------------------
  function legend(el, items) {
    el.innerHTML = '';
    for (const it of items) {
      const li = document.createElement('li');
      const sw = document.createElement('span');
      sw.className = 'swatch';
      sw.style.background = it.color;
      const tx = document.createElement('span');
      tx.textContent = it.label;
      li.append(sw, tx);
      el.appendChild(li);
    }
  }

  // --- line chart ---------------------------------------------------------
  // series: [{ label, color, dash?, levels: [{ <xKey>: x, ... }] }]
  // pick(level) -> y or null (null points are skipped); fmtVal(y) -> text.
  // opts: xKey (default 'concurrency'), xTitle(x) for the crosshair title,
  // log (log y axis), xNumeric (x positioned by value, not by rank).
  function drawLine(svg, series, pick, fmtVal, opts) {
    const xKey = (opts && opts.xKey) || 'concurrency';
    const xTitle = (opts && opts.xTitle) || ((c) => c + (c === 1 ? ' concurrent request' : ' concurrent requests'));
    svg.innerHTML = '';
    // Render at the container's real pixel width rather than scaling a fixed
    // viewBox, so tick and label text stays at its true size on a phone.
    const W = Math.max(280, Math.floor(svg.parentElement.clientWidth) || 880);
    const narrow = W < 560;
    const H = narrow ? 240 : 300;
    // No right gutter when narrow: end labels are dropped there anyway.
    const L = narrow ? 42 : 56, R = narrow ? 14 : 104, T = 14, B = 36;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const pw = W - L - R, ph = H - T - B;
    const xs = [...new Set(series.flatMap(s => s.levels.map(l => l[xKey])))].sort((a, b) => a - b);
    if (!xs.length) return;
    const vals = series.flatMap(s => s.levels.map(pick)).filter(v => v != null);
    if (!vals.length) return;
    const rawMax = Math.max(...vals);
    const mag = Math.pow(10, Math.floor(Math.log10(rawMax)));
    // Log y when the series span decades (4 s next to 1000 s would flatten
    // the fast ones to the axis); numeric x when the levels are not shared
    // steps (clip lengths at two frame rates).
    const log = !!(opts && opts.log);
    const yMin = log ? Math.pow(10, Math.floor(Math.log10(Math.min(...vals)))) : 0;
    const yMax = log ? Math.pow(10, Math.ceil(Math.log10(rawMax))) : Math.ceil(rawMax / (mag / 2)) * (mag / 2);
    const numeric = !!(opts && opts.xNumeric) && xs.length > 1;
    const xMin = xs[0], xMax = xs[xs.length - 1];
    const X = (c) => L + (xs.length === 1 ? pw / 2
      : numeric ? ((c - xMin) / (xMax - xMin)) * pw : (xs.indexOf(c) / (xs.length - 1)) * pw);
    const Y = log
      ? (v) => T + (1 - (Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * ph
      : (v) => T + (1 - v / yMax) * ph;

    // gridlines + y ticks. Decimals come from the step size, not the value, so
    // a 0.5 step never renders 1.5 as "2". Log axes tick at 1-2-5 per decade.
    const ticks = [];
    if (log) {
      const decades = Math.round(Math.log10(yMax) - Math.log10(yMin));
      for (let d = yMin; d < yMax; d *= 10) ticks.push(...(decades <= 3 ? [d, 2 * d, 5 * d] : [d]));
      ticks.push(yMax);
    } else {
      for (let i = 0; i <= 4; i++) ticks.push((yMax / 4) * i);
    }
    // On a log axis each tick has its own magnitude, so 0.5 and 500 sit on
    // the same axis; decimals then follow the tick, not a shared step.
    const decFor = (step) => step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
    const dec = decFor(yMax / 4);
    ticks.forEach((v, i) => {
      const y = Y(v);
      svg.appendChild(svgEl('line', { class: i === 0 ? 'axisline' : 'grid', x1: L, x2: L + pw, y1: y, y2: y }));
      const t = svgEl('text', { class: 'tick', x: L - 10, y: y + 4, 'text-anchor': 'end' });
      const d = log ? decFor(v) : dec;
      t.textContent = d ? v.toFixed(d) : Math.round(v).toLocaleString('en-US');
      svg.appendChild(t);
    });
    // x ticks; on a numeric axis, labels that would overlap are dropped
    let lastTick = -Infinity;
    for (const c of xs) {
      if (numeric && X(c) - lastTick < 30) continue;
      lastTick = X(c);
      const t = svgEl('text', { class: 'tick', x: X(c), y: T + ph + 20, 'text-anchor': 'middle' });
      t.textContent = numeric ? Math.round(c) : c;
      svg.appendChild(t);
    }

    const cross = svgEl('line', { class: 'cross', y1: T, y2: T + ph, x1: L, x2: L, opacity: 0 });
    svg.appendChild(cross);

    // series
    for (const s of series) {
      const pts = s.levels.filter(l => pick(l) != null);
      if (!pts.length) continue;
      const attrs = {
        class: 'ser', stroke: s.color,
        points: pts.map(l => `${X(l[xKey])},${Y(pick(l))}`).join(' '),
      };
      if (s.dash) attrs['stroke-dasharray'] = '6 5';
      svg.appendChild(svgEl('polyline', attrs));
      for (const l of pts) {
        svg.appendChild(svgEl('circle', {
          class: 'ring', cx: X(l[xKey]), cy: Y(pick(l)), r: s.dash ? 3 : 4.5, fill: s.color,
        }));
      }
    }

    // direct end labels, skipped entirely if any two would collide
    const ends = series
      .map(s => { const p = s.levels.filter(l => pick(l) != null).pop(); return p ? { s, p } : null; })
      .filter(Boolean)
      .sort((a, b) => Y(pick(a.p)) - Y(pick(b.p)));
    const collide = ends.some((e, i) => i > 0 && Math.abs(Y(pick(e.p)) - Y(pick(ends[i - 1].p))) < 14);
    if (!collide && !narrow) {
      for (const e of ends) {
        const x = X(e.p[xKey]), y = Y(pick(e.p));
        svg.appendChild(svgEl('circle', { cx: x + 12, cy: y - 3, r: 3.5, fill: e.s.color }));
        const t = svgEl('text', { class: 'endlab', x: x + 20, y: y });
        t.textContent = fmtVal(pick(e.p));
        svg.appendChild(t);
      }
    }

    // crosshair hit layer
    const hit = svgEl('rect', { class: 'hit', x: L, y: T, width: pw, height: ph, tabindex: 0 });
    svg.appendChild(hit);
    let idx = -1;
    const place = (i, ev) => {
      idx = i;
      const c = xs[i];
      cross.setAttribute('x1', X(c));
      cross.setAttribute('x2', X(c));
      cross.setAttribute('opacity', 1);
      const rows = [];
      for (const s of series) {
        const l = s.levels.find(l => l[xKey] === c);
        if (l && pick(l) != null) rows.push({ color: s.color, label: s.label, value: fmtVal(pick(l)) });
      }
      showRowsTip(ev, xTitle(c), rows);
    };
    hit.addEventListener('pointermove', (ev) => {
      const box = svg.getBoundingClientRect();
      const rel = (ev.clientX - box.left) / box.width * W;
      let best = 0, bd = Infinity;
      xs.forEach((c, i) => { const d = Math.abs(X(c) - rel); if (d < bd) { bd = d; best = i; } });
      place(best, ev);
    });
    hit.addEventListener('pointerleave', () => { cross.setAttribute('opacity', 0); hideTip(); });
    hit.addEventListener('focus', () => {
      const box = svg.getBoundingClientRect();
      place(0, { clientX: box.left + 60, clientY: box.top + 40 });
    });
    hit.addEventListener('blur', () => { cross.setAttribute('opacity', 0); hideTip(); });
    hit.addEventListener('keydown', (ev) => {
      if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
      ev.preventDefault();
      const box = svg.getBoundingClientRect();
      const next = Math.min(xs.length - 1, Math.max(0, (idx < 0 ? 0 : idx) + (ev.key === 'ArrowRight' ? 1 : -1)));
      place(next, { clientX: box.left + (X(xs[next]) / W) * box.width, clientY: box.top + 40 });
    });
  }

  // The validated 8-slot categorical order, assigned in fixed order and never
  // cycled: a 9th series needs faceting, not a generated hue.
  const SERIES = Array.from({ length: 8 }, (_, i) => `var(--series-${i + 1})`);

  window.Charts = { svgEl, moveTip, hideTip, showTip, showRowsTip, legend, drawLine, SERIES };
})();
