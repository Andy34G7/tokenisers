/**
 * Tokenisers Lab — benchmark dashboard
 */

// Global Application State
const state = {
  data: null,
  activeTab: 'overview',
  activeFamilyFilter: 'ALL',
  sortColumn: 'macro_val_bpb',
  sortAscending: true,
  tokkaActiveMetric: 'bytes_per_token',
  selectedRadarModels: ['bpe_frequency', 'wp_bpe_ft', 'wp_bpe_pig', 'unigram_lm'],
  radarSlots: {},
  charts: {
    quadrant: null,
    extrinsicBar: null,
    extrinsicCurve: null,
    tokkaBar: null,
    radar: null,
  },
  theme: readStoredTheme(),
};

// Families map to fixed categorical slots (--fam-1 … --fam-8 in index.html)
const FAMILIES = [
  'Information-Theoretic',
  'Frequency BPE',
  'EM / Probabilistic',
  'Structural / Multi-Stage',
  'Morphology / Guided',
  'Regularized BPE',
  'Information-Theoretic (Improved)',
  'Custom',
];
// Secondary encoding so families stay distinguishable without colour
// Only filled styles: line-only ones (star, cross) vanish behind the surface ring
const FAMILY_SHAPES = [['circle', 0], ['rect', 0], ['triangle', 0], ['rectRot', 0], ['triangle', 180], ['rectRounded', 45], ['rectRot', 0], ['circle', 0]];
const DASHES = [[], [6, 3], [2, 3], [10, 3, 2, 3]];

function familySlot(family) {
  const i = FAMILIES.indexOf(family);
  return i === -1 ? FAMILIES.length : i + 1;
}

function familyVar(family) {
  return `var(--fam-${familySlot(family)})`;
}

function getFamilyColor(family) {
  const hex = cssVar(`--fam-${familySlot(family)}`);
  const [shape, rotation] = FAMILY_SHAPES[familySlot(family) - 1];
  return { border: hex, bg: withAlpha(hex, 0.75), shape, rotation };
}

// Preset Texts for Live Playground
const PRESETS = {
  morphology: 'The internationalization of unhappily disproportionate misunderstandings requires robust subword tokenization.',
  code: `pub fn compute_pmi(freq_uv: u64, freq_u: u64, freq_v: u64, total: u64) -> f64 {
    let p_uv = (freq_uv as f64) / (total as f64);
    let p_u = (freq_u as f64) / (total as f64);
    let p_v = (freq_v as f64) / (total as f64);
    (p_uv / (p_u * p_v)).log2()
}`,
  multilingual: 'Tokenizers in Deutsch: Unabhängigkeitserklärung; Français: internationalisation; 汉语: 自然语言处理; 日本語: 機械学習モデル; Русский: сверхпроводимость; العربية: معالجة اللغات الطبيعية.',
  agglutinative: 'megszentségteleníthetetlenségeskedéseitekért epäjärjestelmällistyttämättömyydelläänsäkäänkö unconstitutionality hypercholesterolemia',
};

// Lifecycle Initialization
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  fetchBenchmarkData();
  setupEventListeners();
  loadPlaygroundPreset('morphology');
});

// Helpers
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

function isNum(v) {
  return v !== null && v !== undefined && !Number.isNaN(v);
}

const DASH = '<span class="muted">–</span>';

function fmt(v, digits = 3) {
  return isNum(v) ? v.toFixed(digits) : DASH;
}

function pct(v) {
  return isNum(v) ? (v * 100).toFixed(1) + '%' : DASH;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function withAlpha(hex, a) {
  const n = parseInt(hex.replace('#', ''), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

function famLabel(family) {
  return `<span class="fam"><span class="dot" style="--c: ${familyVar(family)}"></span>${esc(family)}</span>`;
}

// Theme Management
function readStoredTheme() {
  try {
    const t = localStorage.getItem('tokenisers-theme');
    return t === 'light' || t === 'dark' ? t : null;
  } catch {
    return null;
  }
}

function initTheme() {
  if (state.theme) document.documentElement.setAttribute('data-theme', state.theme);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (!state.theme) renderCharts();
  });
}

function isDarkTheme() {
  return state.theme ? state.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function setTheme(themeName) {
  state.theme = themeName;
  try {
    localStorage.setItem('tokenisers-theme', themeName);
  } catch {}
  document.documentElement.setAttribute('data-theme', themeName);
  renderCharts();
}

// Shared Chart.js styling, read from the CSS tokens so both themes stay in sync
function chartTheme() {
  const t = {
    text: cssVar('--text'),
    text2: cssVar('--text-2'),
    muted: cssVar('--muted'),
    grid: cssVar('--grid'),
    surface: cssVar('--surface'),
    border: cssVar('--border-strong'),
  };
  Chart.defaults.font.family = cssVar('--sans');
  Chart.defaults.font.size = 11.5;
  Chart.defaults.color = t.muted;
  Chart.defaults.borderColor = t.grid;
  return t;
}

function tooltipStyle(t) {
  return {
    backgroundColor: t.surface,
    titleColor: t.text,
    bodyColor: t.text2,
    borderColor: t.border,
    borderWidth: 1,
    padding: 10,
    cornerRadius: 8,
    boxPadding: 4,
    usePointStyle: true,
    titleFont: { weight: '600' },
  };
}

function legendStyle(t, position = 'top') {
  return {
    position,
    align: 'start',
    labels: { color: t.text2, usePointStyle: true, pointStyleWidth: 8, boxHeight: 7, padding: 14, font: { size: 11.5 } },
  };
}

function axis(t, title, extra = {}) {
  return {
    grid: { color: t.grid, drawTicks: false },
    border: { display: false },
    ticks: { color: t.muted, padding: 6 },
    title: title ? { display: true, text: title, color: t.text2, font: { size: 11.5, weight: '500' } } : { display: false },
    ...extra,
  };
}

// 1-2-5 ticks only, so log axes don't crowd
function logAxis(t) {
  return { type: 'logarithmic', ticks: { color: t.muted, padding: 6, callback: (v) => ([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].includes(v) ? v : '') } };
}

// Draws dashed median guides on the quadrant chart
const medianLinesPlugin = {
  id: 'medianLines',
  beforeDatasetsDraw(chart, _args, opts) {
    if (!opts || !isNum(opts.x) || !isNum(opts.y)) return;
    const { ctx, chartArea: a, scales } = chart;
    const x = scales.x.getPixelForValue(opts.x);
    const y = scales.y.getPixelForValue(opts.y);
    ctx.save();
    ctx.strokeStyle = opts.color;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, a.top);
    ctx.lineTo(x, a.bottom);
    ctx.moveTo(a.left, y);
    ctx.lineTo(a.right, y);
    ctx.stroke();
    ctx.restore();
  },
};

// Data Fetching
async function fetchBenchmarkData() {
  const refreshBtn = document.getElementById('refresh-btn');
  if (refreshBtn) refreshBtn.classList.add('spinning');

  try {
    const res = await fetch('/api/benchmarks');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    state.data = json;
    renderAllViews();
  } catch (err) {
    console.error('Failed to load benchmark data:', err);
  } finally {
    if (refreshBtn) refreshBtn.classList.remove('spinning');
  }
}

// Master Render Function
function renderAllViews() {
  if (!state.data) return;

  renderMeta();
  renderKPIs();
  renderLeaderboard();
  renderIntrinsicTab();
  renderCharts();

  renderMath();
}

// KaTeX auto-render skips single-dollar inline math unless asked
function renderMath() {
  if (!window.renderMathInElement) return;
  window.renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
    ],
    throwOnError: false,
  });
}

function renderCharts() {
  if (!state.data) return;
  renderQuadrantChart();
  renderExtrinsicTab();
  renderTokkaTab();
  renderRadarView();
}

function renderMeta() {
  const n = Object.keys(state.data.models).length;
  const sub = document.getElementById('brand-sub');
  if (sub) sub.textContent = `${n} subword tokenisers · 11 languages · downstream LM, compression & morphology`;
  const foot = document.getElementById('footer-updated');
  if (foot && state.data.timestamp) foot.textContent = `Data loaded ${new Date(state.data.timestamp).toLocaleString()}`;
}

// Tab Switching
function switchTab(tabName) {
  state.activeTab = tabName;

  document.querySelectorAll('[id^="tab-nav-"]').forEach((btn) => {
    btn.setAttribute('aria-selected', String(btn.id === `tab-nav-${tabName}`));
  });

  const tabIds = ['overview', 'extrinsic', 'tokka', 'intrinsic', 'wpbpe', 'playground'];
  tabIds.forEach((id) => {
    document.getElementById(`tab-${id}`)?.classList.toggle('hidden', id !== tabName);
  });

  // Charts rendered while hidden have zero size; resize once visible
  setTimeout(() => {
    Object.values(state.charts).forEach((c) => c && c.resize());
  }, 50);
}

// 1. KPI Summary Strip
function renderKPIs() {
  const sum = state.data.summary;
  if (!sum) return;

  const set = (id, kpi, format) => {
    const valEl = document.getElementById(`kpi-${id}-val`);
    const modEl = document.getElementById(`kpi-${id}-model`);
    if (valEl) valEl.textContent = kpi ? format(kpi.value) : '–';
    if (modEl) modEl.textContent = kpi ? kpi.display_name : '–';
  };

  set('bpb', sum.best_downstream_quality, (v) => v.toFixed(3));
  set('renyi', sum.highest_information_efficiency, (v) => v.toFixed(3));
  set('morph', sum.best_morphological_f1, (v) => (v * 100).toFixed(1) + '%');
  set('bpt', sum.best_compression, (v) => v.toFixed(2));
  set('speed', sum.fastest_throughput, (v) => v.toFixed(1));
}

// 2. Leaderboard Table
function renderLeaderboard() {
  const tbody = document.getElementById('leaderboard-tbody');
  if (!tbody || !state.data) return;

  const models = Object.values(state.data.models);
  const filtered = models.filter((m) => state.activeFamilyFilter === 'ALL' || m.family === state.activeFamilyFilter);

  filtered.sort((a, b) => {
    const valA = a[state.sortColumn];
    const valB = b[state.sortColumn];
    // Missing values always sink to the bottom
    if (!isNum(valA)) return isNum(valB) ? 1 : 0;
    if (!isNum(valB)) return -1;
    return state.sortAscending ? valA - valB : valB - valA;
  });

  document.querySelectorAll('#leaderboard-table th.sortable').forEach((th) => {
    if (th.dataset.sort === state.sortColumn) th.setAttribute('aria-sort', state.sortAscending ? 'ascending' : 'descending');
    else th.removeAttribute('aria-sort');
  });

  const countBadge = document.getElementById('leaderboard-count-badge');
  if (countBadge) countBadge.textContent = filtered.length;

  const bpbs = filtered.map((m) => m.macro_val_bpb).filter(isNum);
  const minBpb = Math.min(...bpbs);
  const maxBpb = Math.max(...bpbs);

  tbody.innerHTML = filtered
    .map((m, idx) => {
      // Bar inverted so the lowest BPB gets the longest bar
      const barWidth = isNum(m.macro_val_bpb) && maxBpb > minBpb ? Math.round(((maxBpb - m.macro_val_bpb) / (maxBpb - minBpb)) * 100) : 100;
      const gen2 = m.generation && m.generation.includes('Gen 2') ? ' <span class="tag">Gen 2</span>' : '';

      return `
      <tr>
        <td class="rank num">${idx + 1}</td>
        <td>
          <div class="model-name">${esc(m.display_name)}${gen2}</div>
          <div class="model-desc" title="${esc(m.description)}">${esc(m.description)}</div>
        </td>
        <td>${famLabel(m.family)}</td>
        <td class="r num">
          <div class="bar-cell">
            ${isNum(m.macro_val_bpb) ? `<span class="bar"><i style="width: ${Math.max(barWidth, 4)}%"></i></span>` : ''}
            <span class="hl">${fmt(m.macro_val_bpb)}</span>
          </div>
        </td>
        <td class="r num">${fmt(m.bytes_per_token)}</td>
        <td class="r num">${fmt(m.renyi_e_2_0)}</td>
        <td class="r num">${pct(m.morph_f1)}</td>
        <td class="r num">${fmt(m.decode_mb_per_sec, 1)}</td>
        <td class="r"><button class="link-btn" onclick="inspectModelInPlayground('${esc(m.key)}')" title="Open in playground">Try →</button></td>
      </tr>`;
    })
    .join('');
}

function sortLeaderboard(col) {
  if (state.sortColumn === col) {
    state.sortAscending = !state.sortAscending;
  } else {
    state.sortColumn = col;
    // Lower is better for bpb, higher is better for others
    state.sortAscending = col === 'macro_val_bpb';
  }
  renderLeaderboard();
}

// 3. Trade-off Quadrant Scatter Plot
const METRIC_LABELS = {
  bytes_per_token: 'Bytes / token',
  renyi_e_2_0: 'Rényi E₂',
  morph_f1: 'Morphology F1',
  decode_mb_per_sec: 'Decode MB/s',
  macro_val_bpb: 'Macro val BPB',
  english_bpb: 'English val BPB',
  multilingual_bpb: 'Multilingual val BPB',
  code_bpb: 'Code val BPB',
};

function quadrantValue(m, metric) {
  if (metric === 'english_bpb') return m.val_domains?.english?.bpb;
  if (metric === 'multilingual_bpb') return m.val_domains?.multilingual?.bpb;
  if (metric === 'code_bpb') return m.val_domains?.code?.bpb;
  return m[metric];
}

function median(vals) {
  const s = [...vals].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

function renderQuadrantChart() {
  const canvas = document.getElementById('quadrant-chart');
  if (!canvas || !state.data) return;

  const xMetric = document.getElementById('quadrant-x-select')?.value || 'bytes_per_token';
  const yMetric = document.getElementById('quadrant-y-select')?.value || 'macro_val_bpb';

  const points = Object.values(state.data.models)
    .map((m) => ({ x: quadrantValue(m, xMetric), y: quadrantValue(m, yMetric), model: m }))
    .filter((p) => isNum(p.x) && isNum(p.y));

  if (state.charts.quadrant) state.charts.quadrant.destroy();
  if (points.length === 0) return;

  const families = FAMILIES.filter((f) => points.some((p) => p.model.family === f));
  const datasets = families.map((fam) => {
    const color = getFamilyColor(fam);
    return {
      label: fam,
      data: points.filter((p) => p.model.family === fam),
      backgroundColor: color.border,
      borderColor: cssVar('--surface'),
      borderWidth: 2,
      pointStyle: color.shape,
      rotation: color.rotation,
      pointRadius: 7,
      pointHoverRadius: 9,
    };
  });

  const t = chartTheme();
  // Invert BPB so lower (better) sits at the top
  const isYInverted = yMetric.includes('bpb');

  state.charts.quadrant = new Chart(canvas, {
    type: 'scatter',
    data: { datasets },
    plugins: [medianLinesPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: legendStyle(t),
        medianLines: { x: median(points.map((p) => p.x)), y: median(points.map((p) => p.y)), color: t.border },
        tooltip: {
          ...tooltipStyle(t),
          callbacks: {
            title: (items) => items[0].raw.model.display_name,
            label: (ctx) => [
              `${METRIC_LABELS[xMetric]}: ${ctx.raw.x.toFixed(3)}`,
              `${METRIC_LABELS[yMetric]}: ${ctx.raw.y.toFixed(3)}`,
            ],
            footer: (items) => items[0].raw.model.family,
          },
          footerColor: t.muted,
          footerFont: { weight: '400', size: 11 },
        },
      },
      scales: {
        x: axis(t, `${METRIC_LABELS[xMetric]}  →  better`),
        y: axis(t, `${METRIC_LABELS[yMetric]}  →  better`, { reverse: isYInverted }),
      },
    },
  });
}

// 4. Downstream LM Validation BPB & Convergence Curves
function renderExtrinsicTab() {
  if (!state.data) return;

  const models = Object.values(state.data.models)
    .filter((m) => isNum(m.macro_val_bpb))
    .sort((a, b) => a.macro_val_bpb - b.macro_val_bpb);
  const t = chartTheme();

  // Grouped Bar Chart
  const barCanvas = document.getElementById('extrinsic-bar-chart');
  if (barCanvas) {
    if (state.charts.extrinsicBar) state.charts.extrinsicBar.destroy();

    const bar = (label, data, color) => ({ label, data, backgroundColor: color, borderRadius: 2, borderSkipped: 'start', categoryPercentage: 0.75, barPercentage: 0.9 });

    state.charts.extrinsicBar = new Chart(barCanvas, {
      type: 'bar',
      data: {
        labels: models.map((m) => m.display_name.replace('WP-BPE ', 'WP-')),
        datasets: [
          bar('English', models.map((m) => m.val_domains?.english?.bpb ?? null), cssVar('--fam-1')),
          bar('Multilingual', models.map((m) => m.val_domains?.multilingual?.bpb ?? null), cssVar('--fam-2')),
          bar('Code', models.map((m) => m.val_domains?.code?.bpb ?? null), cssVar('--fam-3')),
          bar('Macro avg', models.map((m) => m.macro_val_bpb), cssVar('--fam-8')),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        interaction: { mode: 'index', axis: 'y', intersect: false },
        plugins: {
          legend: legendStyle(t),
          tooltip: { ...tooltipStyle(t), callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.x?.toFixed(3)}` } },
        },
        scales: {
          x: axis(t, 'Bits per byte'),
          y: axis(t, null, { grid: { display: false }, ticks: { color: t.text2, font: { size: 11 } } }),
        },
      },
    });
  }

  // Convergence Line Chart
  const curveCanvas = document.getElementById('extrinsic-curve-chart');
  if (curveCanvas) {
    if (state.charts.extrinsicCurve) state.charts.extrinsicCurve.destroy();

    // Same family shares a hue; dash pattern separates members within it
    const seen = {};
    const datasets = models
      .filter((m) => m.curve && m.curve.length > 0)
      .map((m) => {
        const color = getFamilyColor(m.family);
        const n = (seen[m.family] = (seen[m.family] ?? -1) + 1);
        return {
          label: m.display_name,
          data: m.curve.map((pt) => ({ x: pt.bytes_consumed / 1000, y: pt.train_bpb, step: pt.step })),
          borderColor: color.border,
          backgroundColor: color.border,
          borderDash: DASHES[n % DASHES.length],
          borderWidth: 1.75,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.25,
        };
      });

    state.charts.extrinsicCurve = new Chart(curveCanvas, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
        plugins: {
          legend: { ...legendStyle(t, 'bottom'), labels: { ...legendStyle(t).labels, usePointStyle: false, boxWidth: 14, boxHeight: 2, padding: 8, font: { size: 10.5 } } },
          tooltip: { ...tooltipStyle(t), usePointStyle: false, callbacks: { title: (i) => `${i[0].parsed.x.toFixed(0)} KB · step ${i[0].raw.step}`, label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(3)}` } },
        },
        scales: {
          x: axis(t, 'Bytes ingested (KB)', { type: 'linear', min: 0 }),
          // Log scale keeps the ~100 BPB first step and the ~2 BPB tail both readable
          y: axis(t, 'Train BPB (log)', logAxis(t)),
        },
      },
    });
  }

  // Table
  const tbody = document.getElementById('extrinsic-tbody');
  if (tbody) {
    const baseModel = models.find((m) => m.key === 'bpe_frequency');
    const baseMacroBpb = baseModel ? baseModel.macro_val_bpb : 2.738;

    tbody.innerHTML = models
      .map((m) => {
        const delta = m.macro_val_bpb - baseMacroBpb;
        const isBase = m.key === 'bpe_frequency';
        const deltaHtml = isBase
          ? '<span class="muted">baseline</span>'
          : `<span class="${delta <= 0 ? 'delta-good' : 'delta-bad'}">${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(3)}</span>`;

        return `
        <tr>
          <td class="model-name">${esc(m.display_name)}</td>
          <td>${famLabel(m.family)}</td>
          <td class="r num">${fmt(m.val_domains?.english?.bpb)}</td>
          <td class="r num">${fmt(m.val_domains?.multilingual?.bpb)}</td>
          <td class="r num">${fmt(m.val_domains?.code?.bpb)}</td>
          <td class="r num hl">${fmt(m.macro_val_bpb)}</td>
          <td class="r num">${deltaHtml}</td>
        </tr>`;
      })
      .join('');
  }
}

// 5. Tokka-Bench 11-Language Heatmap & Grouped Chart
// +1: higher is better, -1: lower is better, 0: neutral magnitude
const TOKKA_DIRECTION = { bytes_per_token: 1, subword_fertility: -1, continued_word_rate: -1, unique_tokens: 0 };

function renderTokkaTab() {
  if (!state.data || !state.data.tokka_matrix) return;

  const matrix = state.data.tokka_matrix;
  const models = Object.keys(matrix);
  const langs = ['en', 'de', 'fr', 'ru', 'zh', 'ja', 'ar', 'hi', 'hu', 'fi', 'code'];
  const metric = state.tokkaActiveMetric;
  const dir = TOKKA_DIRECTION[metric] ?? 0;

  const lo = document.getElementById('tokka-scale-lo');
  const hi = document.getElementById('tokka-scale-hi');
  if (lo) lo.textContent = dir === 0 ? 'fewer' : 'worse';
  if (hi) hi.textContent = dir === 0 ? 'more' : 'better';

  const display = (v) => (isNum(v) ? (metric === 'continued_word_rate' ? v.toFixed(1) + '%' : metric === 'unique_tokens' ? String(Math.round(v)) : v.toFixed(2)) : '–');

  // Shade by rank within each column, so outliers (e.g. CJK) don't wash out the rest
  const cols = [...langs, 'macro_avg'];
  const sorted = {};
  cols.forEach((c) => {
    sorted[c] = models.map((k) => matrix[k]?.[c]?.[metric]).filter(isNum).sort((a, b) => a - b);
  });
  const shade = (v, c) => {
    if (!isNum(v)) return '';
    const vals = sorted[c];
    const lo = vals.indexOf(v);
    const hi = vals.lastIndexOf(v);
    let s = vals.length > 1 ? (lo + hi) / 2 / (vals.length - 1) : 1;
    if (dir === -1) s = 1 - s;
    return `background-color: color-mix(in srgb, var(--accent) ${Math.round(4 + s * 51)}%, var(--surface));`;
  };

  const tbody = document.getElementById('tokka-heatmap-tbody');
  if (tbody) {
    tbody.innerHTML = models
      .map((mKey) => {
        const meta = state.data.models[mKey] || {};
        const row = matrix[mKey];
        const cells = langs
          .map((lang) => {
            const v = row?.[lang]?.[metric];
            return `<td class="cell" style="${shade(v, lang)}">${display(v)}</td>`;
          })
          .join('');
        const macroVal = row?.macro_avg?.[metric];

        return `
        <tr>
          <td class="model-name">${esc(meta.display_name || mKey)}</td>
          ${cells}
          <td class="cell avg" style="${shade(macroVal, 'macro_avg')}">${display(macroVal)}</td>
        </tr>`;
      })
      .join('');
  }

  // Bar Chart: series are tokenisers, so each gets its own categorical slot
  const barCanvas = document.getElementById('tokka-bar-chart');
  if (barCanvas) {
    if (state.charts.tokkaBar) state.charts.tokkaBar.destroy();
    const t = chartTheme();

    const datasets = models.slice(0, 6).map((mKey, i) => {
      const meta = state.data.models[mKey] || {};
      return {
        label: meta.display_name || mKey,
        data: langs.map((lang) => matrix[mKey]?.[lang]?.[metric] ?? null),
        backgroundColor: cssVar(`--fam-${i + 1}`),
        borderRadius: 2,
        borderSkipped: 'start',
        categoryPercentage: 0.8,
        barPercentage: 0.9,
      };
    });

    // Some scripts are orders of magnitude apart; switch to log so the rest stay visible
    const all = datasets.flatMap((d) => d.data).filter((v) => isNum(v) && v > 0);
    const useLog = Math.max(...all) / Math.min(...all) > 20;
    const metricLabel = document.querySelector(`.tokka-metric-btn[data-metric="${metric}"]`)?.textContent || metric;

    state.charts.tokkaBar = new Chart(barCanvas, {
      type: 'bar',
      data: { labels: langs.map((l) => (l === 'code' ? 'Code' : l.toUpperCase())), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: legendStyle(t), tooltip: tooltipStyle(t) },
        scales: {
          x: axis(t, null, { grid: { display: false } }),
          y: axis(t, useLog ? `${metricLabel} (log)` : metricLabel, useLog ? logAxis(t) : {}),
        },
      },
    });
  }
}

// 6. Intrinsic & Morphology Table
function renderIntrinsicTab() {
  if (!state.data) return;

  const tbody = document.getElementById('intrinsic-tbody');
  if (!tbody) return;

  const searchTerm = (document.getElementById('intrinsic-search')?.value || '').toLowerCase();
  const models = Object.values(state.data.models).filter((m) => m.display_name.toLowerCase().includes(searchTerm) || m.family.toLowerCase().includes(searchTerm));

  if (models.length === 0) {
    tbody.innerHTML = '<tr><td colspan="15" class="muted" style="text-align: center; padding: 24px">No tokenisers match.</td></tr>';
    return;
  }

  tbody.innerHTML = models
    .map((m) => {
      const intr = m.intrinsic || {};
      return `
      <tr>
        <td><span class="fam" style="color: var(--text)"><span class="dot" style="--c: ${familyVar(m.family)}"></span><span class="model-name">${esc(m.display_name)}</span></span></td>
        <td class="r num">${intr.vocab_size ?? DASH}</td>
        <td class="r num">${fmt(intr.bytes_per_token)}</td>
        <td class="r num">${fmt(intr.fertility, 2)}</td>
        <td class="r num">${fmt(intr.renyi_e_1_0)}</td>
        <td class="r num hl">${fmt(intr.renyi_e_2_0)}</td>
        <td class="r num">${fmt(intr.renyi_e_2_5)}</td>
        <td class="r num">${fmt(intr.gini_coefficient)}</td>
        <td class="r num">${pct(intr.active_fraction)}</td>
        <td class="r num hl">${pct(intr.morph_f1)}</td>
        <td class="r num">${pct(intr.morph_precision)}</td>
        <td class="r num">${pct(intr.morph_recall)}</td>
        <td class="r num">${pct(intr.identifier_f1)}</td>
        <td class="r num">${fmt(intr.encode_mb_per_sec, 2)}</td>
        <td class="r num">${fmt(intr.decode_mb_per_sec, 1)}</td>
      </tr>`;
    })
    .join('');
}

// 7. Multi-Axis Radar View
// Each selected model keeps its slot until deselected, so toggling never repaints the others
function radarSlot(mKey) {
  if (!state.radarSlots[mKey]) {
    const used = new Set(Object.values(state.radarSlots));
    const free = [1, 2, 3, 4, 5, 6, 7].find((s) => !used.has(s));
    state.radarSlots[mKey] = free || 8;
  }
  return state.radarSlots[mKey];
}

function renderRadarView() {
  if (!state.data) return;

  const canvas = document.getElementById('radar-chart');
  const togglesContainer = document.getElementById('radar-toggles');
  if (!canvas || !togglesContainer) return;

  const allModels = Object.values(state.data.models);
  state.selectedRadarModels.forEach(radarSlot);

  togglesContainer.innerHTML = allModels
    .map((m) => {
      const on = state.selectedRadarModels.includes(m.key);
      const dot = on ? `<span class="dot" style="--c: var(--fam-${state.radarSlots[m.key]})"></span>` : '';
      return `<button class="chip" aria-pressed="${on}" onclick="toggleRadarModel('${esc(m.key)}')">${dot}${esc(m.display_name)}</button>`;
    })
    .join('');

  if (state.charts.radar) state.charts.radar.destroy();

  const getMinMax = (extractor) => {
    const vals = allModels.map(extractor).filter(isNum);
    return { min: Math.min(...vals), max: Math.max(...vals) };
  };

  const axes = [
    (m) => m.bytes_per_token,
    (m) => m.renyi_e_2_0,
    (m) => m.morph_f1,
    (m) => m.identifier_f1,
    (m) => (m.macro_val_bpb ? 1 / m.macro_val_bpb : null),
    (m) => m.decode_mb_per_sec,
  ];
  const ranges = axes.map(getMinMax);

  const normalize = (val, mm) => {
    if (!isNum(val)) return 0.2;
    if (mm.max === mm.min) return 1.0;
    return Math.max(0.1, Math.min(1.0, (val - mm.min) / (mm.max - mm.min)));
  };

  const datasets = state.selectedRadarModels
    .map((mKey) => {
      const m = state.data.models[mKey];
      if (!m) return null;
      const hex = cssVar(`--fam-${state.radarSlots[mKey]}`);
      return {
        label: m.display_name,
        data: axes.map((f, i) => normalize(f(m), ranges[i])),
        backgroundColor: withAlpha(hex, 0.12),
        borderColor: hex,
        borderWidth: 2,
        pointBackgroundColor: hex,
        pointRadius: 3,
      };
    })
    .filter(Boolean);

  const t = chartTheme();

  state.charts.radar = new Chart(canvas, {
    type: 'radar',
    data: {
      labels: ['Compression', 'Rényi E₂', 'Morphology F1', 'Identifier F1', 'Downstream (1/BPB)', 'Decode speed'],
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: legendStyle(t, 'bottom'),
        tooltip: { ...tooltipStyle(t), callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.r.toFixed(2)}` } },
      },
      scales: {
        r: {
          min: 0,
          max: 1.0,
          ticks: { display: false, stepSize: 0.25 },
          pointLabels: { color: t.text2, font: { size: 11.5, weight: '500' } },
          grid: { color: t.grid },
          angleLines: { color: t.grid },
        },
      },
    },
  });
}

function toggleRadarModel(mKey) {
  if (state.selectedRadarModels.includes(mKey)) {
    if (state.selectedRadarModels.length > 1) {
      state.selectedRadarModels = state.selectedRadarModels.filter((k) => k !== mKey);
      delete state.radarSlots[mKey];
    }
  } else {
    state.selectedRadarModels.push(mKey);
  }
  renderRadarView();
}

// 8. Live Interactive Tokenizer Playground
function loadPlaygroundPreset(type) {
  const input = document.getElementById('playground-input');
  if (input && PRESETS[type]) {
    input.value = PRESETS[type];
    runLiveTokenize();
  }
}

function inspectModelInPlayground(modelKey) {
  switchTab('playground');
  const target = document.getElementById(`tok-card-${modelKey}`);
  if (target) {
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    target.classList.add('flash');
    setTimeout(() => target.classList.remove('flash'), 1600);
  }
}

async function runLiveTokenize() {
  const inputEl = document.getElementById('playground-input');
  const vocabEl = document.getElementById('playground-vocab-input');
  const loadingEl = document.getElementById('playground-loading');
  const resultsContainer = document.getElementById('playground-results-container');
  const btn = document.getElementById('playground-tokenize-btn');

  const text = inputEl?.value || '';
  const vocabSize = parseInt(vocabEl?.value || '400', 10);

  if (loadingEl) loadingEl.classList.remove('hidden');
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/tokenize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, vocab_size: vocabSize }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    renderPlaygroundResults(data.results, data.raw_bytes);
  } catch (err) {
    console.error('Tokenize error:', err);
    if (resultsContainer) {
      resultsContainer.innerHTML = `<div class="error">Tokenisation failed: ${esc(err.message)}</div>`;
    }
  } finally {
    if (loadingEl) loadingEl.classList.add('hidden');
    if (btn) btn.disabled = false;
  }
}

function renderPlaygroundResults(results, rawBytes) {
  const container = document.getElementById('playground-results-container');
  if (!container) return;

  if (!results || results.length === 0) {
    container.innerHTML = '<div class="loading">No tokeniser results available.</div>';
    return;
  }

  container.innerHTML = results
    .map((res) => {
      // Make whitespace visible without changing the token's width much
      const pillHtml = res.token_details
        .map((tok) => {
          const piece = esc(tok.piece).replace(/ /g, '<span class="ws">␣</span>').replace(/\n/g, '<span class="ws">↵</span>');
          return `<span class="tk" title="ID ${tok.id} · ${tok.byte_len} bytes">${piece}</span>`;
        })
        .join('');

      return `
      <div id="tok-card-${esc(res.name)}" class="tok-row">
        <div class="tok-head">
          <div class="left">
            <span class="model-name">${esc(res.display_name)}</span>
            ${famLabel(res.family)}
          </div>
          <div class="tok-stats">
            <span><b>${esc(res.num_tokens)}</b> tokens</span>
            <span><b>${esc(res.bytes_per_token)}</b> B/tok</span>
            <span><b>${esc(res.encode_ms)}</b> ms</span>
          </div>
        </div>
        <div class="tokens">${pillHtml}</div>
      </div>`;
    })
    .join('');
}

// Event Listeners Setup
function setupEventListeners() {
  // Family Filter Chips
  document.querySelectorAll('.family-filter-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.family-filter-btn').forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
      state.activeFamilyFilter = btn.dataset.family;
      renderLeaderboard();
    });
  });

  // Quadrant Selectors
  document.getElementById('quadrant-x-select')?.addEventListener('change', renderQuadrantChart);
  document.getElementById('quadrant-y-select')?.addEventListener('change', renderQuadrantChart);

  // Tokka Metric Switcher
  document.querySelectorAll('.tokka-metric-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tokka-metric-btn').forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
      state.tokkaActiveMetric = btn.dataset.metric;
      renderTokkaTab();
    });
  });

  // Intrinsic Search
  document.getElementById('intrinsic-search')?.addEventListener('input', renderIntrinsicTab);

  // Playground Vocab Slider sync
  const vocabSlider = document.getElementById('playground-vocab-slider');
  const vocabInput = document.getElementById('playground-vocab-input');
  if (vocabSlider && vocabInput) {
    vocabSlider.addEventListener('input', (e) => {
      vocabInput.value = e.target.value;
    });
    vocabInput.addEventListener('input', (e) => {
      vocabSlider.value = e.target.value;
    });
  }

  // Ctrl/Cmd+Enter tokenises
  document.getElementById('playground-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runLiveTokenize();
  });

  document.getElementById('refresh-btn')?.addEventListener('click', fetchBenchmarkData);
  document.getElementById('theme-btn')?.addEventListener('click', () => setTheme(isDarkTheme() ? 'light' : 'dark'));
}
