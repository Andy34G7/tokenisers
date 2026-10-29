/**
 * Tokenisers Lab — Algorithmic Benchmark Intelligence Dashboard
 * Inspired by Artificial Analysis (artificialanalysis.ai)
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
  charts: {
    quadrant: null,
    extrinsicBar: null,
    extrinsicCurve: null,
    tokkaBar: null,
    radar: null,
  },
  theme: localStorage.getItem('tokenisers-theme') || 'dark',
};

// Family Color Scheme
const FAMILY_COLORS = {
  'Information-Theoretic': { bg: 'rgba(56, 189, 248, 0.75)', border: '#38bdf8' },
  'Information-Theoretic (Improved)': { bg: 'rgba(129, 140, 248, 0.85)', border: '#818cf8' },
  'Frequency BPE': { bg: 'rgba(245, 158, 11, 0.75)', border: '#f59e0b' },
  'EM / Probabilistic': { bg: 'rgba(236, 72, 153, 0.75)', border: '#ec4899' },
  'Structural / Multi-Stage': { bg: 'rgba(16, 185, 129, 0.75)', border: '#10b981' },
  'Morphology / Guided': { bg: 'rgba(139, 92, 246, 0.75)', border: '#8b5cf6' },
  'Regularized BPE': { bg: 'rgba(100, 116, 139, 0.75)', border: '#64748b' },
  'Custom': { bg: 'rgba(148, 163, 184, 0.75)', border: '#94a3b8' },
};

function getFamilyColor(family) {
  return FAMILY_COLORS[family] || FAMILY_COLORS['Custom'];
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

// Theme Management
function initTheme() {
  document.documentElement.setAttribute('data-theme', state.theme);
}

function setTheme(themeName) {
  state.theme = themeName;
  localStorage.setItem('tokenisers-theme', themeName);
  document.documentElement.setAttribute('data-theme', themeName);
  updateChartThemes();
}

function isDarkTheme() {
  return ['dark', 'night', 'dim', 'synthwave', 'halloween', 'forest', 'black', 'luxury', 'dracula'].includes(state.theme);
}

function updateChartThemes() {
  const isDark = isDarkTheme();
  const textColor = isDark ? '#cbd5e1' : '#334155';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';

  Object.values(state.charts).forEach((chart) => {
    if (!chart) return;
    if (chart.options.scales) {
      Object.values(chart.options.scales).forEach((scale) => {
        if (scale.ticks) scale.ticks.color = textColor;
        if (scale.grid) scale.grid.color = gridColor;
        if (scale.title) scale.title.color = textColor;
      });
    }
    if (chart.options.plugins && chart.options.plugins.legend) {
      chart.options.plugins.legend.labels.color = textColor;
    }
    chart.update();
  });
}

// Data Fetching
async function fetchBenchmarkData() {
  const refreshBtn = document.getElementById('refresh-btn');
  if (refreshBtn) refreshBtn.classList.add('loading');

  try {
    const res = await fetch('/api/benchmarks');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    state.data = json;
    renderAllViews();
  } catch (err) {
    console.error('Failed to load benchmark data:', err);
  } finally {
    if (refreshBtn) refreshBtn.classList.remove('loading');
  }
}

// Master Render Function
function renderAllViews() {
  if (!state.data) return;

  renderKPIs();
  renderLeaderboard();
  renderQuadrantChart();
  renderExtrinsicTab();
  renderTokkaTab();
  renderIntrinsicTab();
  renderRadarView();

  if (window.renderMathInElement) {
    window.renderMathInElement(document.body);
  }
}

// Tab Switching
function switchTab(tabName) {
  state.activeTab = tabName;

  // Toggle Tab Navigation Pill styling
  document.querySelectorAll('[id^="tab-nav-"]').forEach((btn) => {
    btn.classList.remove('tab-active');
  });
  const activeNav = document.getElementById(`tab-nav-${tabName}`);
  if (activeNav) activeNav.classList.add('tab-active');

  // Toggle Tab content visibility
  const tabIds = ['overview', 'extrinsic', 'tokka', 'intrinsic', 'wpbpe', 'playground'];
  tabIds.forEach((id) => {
    const el = document.getElementById(`tab-${id}`);
    if (el) {
      if (id === tabName) {
        el.classList.remove('hidden');
      } else {
        el.classList.add('hidden');
      }
    }
  });

  // Re-size charts to adjust for tab layout display
  setTimeout(() => {
    if (tabName === 'overview' && state.charts.quadrant) state.charts.quadrant.resize();
    if (tabName === 'extrinsic') {
      if (state.charts.extrinsicBar) state.charts.extrinsicBar.resize();
      if (state.charts.extrinsicCurve) state.charts.extrinsicCurve.resize();
    }
    if (tabName === 'tokka' && state.charts.tokkaBar) state.charts.tokkaBar.resize();
    if (tabName === 'wpbpe' && state.charts.radar) state.charts.radar.resize();
  }, 50);
}

// 1. KPI Summary Strip
function renderKPIs() {
  const sum = state.data.summary;
  if (!sum) return;

  // BPB
  const bpbValEl = document.getElementById('kpi-bpb-val');
  const bpbModEl = document.getElementById('kpi-bpb-model');
  if (bpbValEl) bpbValEl.textContent = sum.best_downstream_quality ? sum.best_downstream_quality.value.toFixed(3) : '--';
  if (bpbModEl) bpbModEl.textContent = sum.best_downstream_quality ? sum.best_downstream_quality.display_name : 'Macro Val BPB';

  // Rényi
  const renyiValEl = document.getElementById('kpi-renyi-val');
  const renyiModEl = document.getElementById('kpi-renyi-model');
  if (renyiValEl) renyiValEl.textContent = sum.highest_information_efficiency ? sum.highest_information_efficiency.value.toFixed(3) : '--';
  if (renyiModEl) renyiModEl.textContent = sum.highest_information_efficiency ? sum.highest_information_efficiency.display_name : 'Highest Rényi';

  // Morph F1
  const morphValEl = document.getElementById('kpi-morph-val');
  const morphModEl = document.getElementById('kpi-morph-model');
  if (morphValEl) morphValEl.textContent = sum.best_morphological_f1 ? (sum.best_morphological_f1.value * 100).toFixed(1) + '%' : '--';
  if (morphModEl) morphModEl.textContent = sum.best_morphological_f1 ? sum.best_morphological_f1.display_name : 'Sigmorphon F1';

  // Bytes per token
  const bptValEl = document.getElementById('kpi-bpt-val');
  const bptModEl = document.getElementById('kpi-bpt-model');
  if (bptValEl) bptValEl.textContent = sum.best_compression ? sum.best_compression.value.toFixed(2) : '--';
  if (bptModEl) bptModEl.textContent = sum.best_compression ? sum.best_compression.display_name : 'Bytes / Token';

  // Speed
  const speedValEl = document.getElementById('kpi-speed-val');
  const speedModEl = document.getElementById('kpi-speed-model');
  if (speedValEl) speedValEl.textContent = sum.fastest_throughput ? sum.fastest_throughput.value.toFixed(1) + ' MB/s' : '--';
  if (speedModEl) speedModEl.textContent = sum.fastest_throughput ? sum.fastest_throughput.display_name : 'Decode Speed';
}

// 2. Executive Leaderboard Table
function renderLeaderboard() {
  const tbody = document.getElementById('leaderboard-tbody');
  if (!tbody || !state.data) return;

  const models = Object.values(state.data.models);
  const filtered = models.filter((m) => {
    if (state.activeFamilyFilter === 'ALL') return true;
    return m.family === state.activeFamilyFilter;
  });

  filtered.sort((a, b) => {
    let valA = a[state.sortColumn];
    let valB = b[state.sortColumn];
    if (valA === null || valA === undefined) valA = state.sortAscending ? 999999 : -999999;
    if (valB === null || valB === undefined) valB = state.sortAscending ? 999999 : -999999;
    return state.sortAscending ? valA - valB : valB - valA;
  });

  const countBadge = document.getElementById('leaderboard-count-badge');
  if (countBadge) countBadge.textContent = `${filtered.length} Models`;

  // Find min/max for visual progress
  const bpbs = filtered.map((m) => m.macro_val_bpb).filter((v) => v !== null);
  const minBpb = Math.min(...bpbs);
  const maxBpb = Math.max(...bpbs);

  tbody.innerHTML = filtered
    .map((m, idx) => {
      const famColor = getFamilyColor(m.family);
      const bpb = m.macro_val_bpb ? m.macro_val_bpb.toFixed(3) : '<span class="opacity-40">N/A</span>';
      const bpt = m.bytes_per_token ? m.bytes_per_token.toFixed(3) : '<span class="opacity-40">N/A</span>';
      const renyi = m.renyi_e_2_0 ? m.renyi_e_2_0.toFixed(3) : '<span class="opacity-40">N/A</span>';
      const morph = m.morph_f1 ? (m.morph_f1 * 100).toFixed(1) + '%' : '<span class="opacity-40">N/A</span>';
      const speed = m.decode_mb_per_sec ? m.decode_mb_per_sec.toFixed(1) : '<span class="opacity-40">N/A</span>';

      // Bar percentage: inverted so lowest BPB has highest bar width
      let barWidth = 0;
      if (m.macro_val_bpb && maxBpb > minBpb) {
        barWidth = Math.round(((maxBpb - m.macro_val_bpb) / (maxBpb - minBpb)) * 100);
      }

      return `
      <tr class="hover">
        <td class="text-center font-mono font-bold text-base-content/60">${idx + 1}</td>
        <td>
          <div class="font-bold flex items-center gap-1.5">
            <span>${m.display_name}</span>
            ${m.generation.includes('Gen 2') ? '<span class="badge badge-accent badge-xs">Gen 2</span>' : ''}
          </div>
          <div class="text-[10px] text-base-content/60 truncate max-w-xs">${m.description}</div>
        </td>
        <td>
          <span class="badge badge-xs" style="background-color: ${famColor.bg}; color: #fff; border-color: ${famColor.border};">${m.family}</span>
        </td>
        <td class="font-mono">
          <div class="flex items-center gap-2">
            <span class="font-bold text-primary">${bpb}</span>
            ${m.macro_val_bpb ? `<div class="w-12 bg-base-300 rounded-full h-1.5 overflow-hidden"><div class="bg-primary h-1.5 rounded-full" style="width: ${barWidth}%"></div></div>` : ''}
          </div>
        </td>
        <td class="font-mono font-medium">${bpt}</td>
        <td class="font-mono">${renyi}</td>
        <td class="font-mono font-medium text-accent">${morph}</td>
        <td class="font-mono text-base-content/80">${speed}</td>
        <td class="text-center">
          <button class="btn btn-ghost btn-xs text-primary" onclick="inspectModelInPlayground('${m.key}')" title="Test in Interactive Playground">
            Play
          </button>
        </td>
      </tr>
    `;
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

// 3. Artificial Analysis Quadrant Scatter Plot
function renderQuadrantChart() {
  const canvas = document.getElementById('quadrant-chart');
  if (!canvas || !state.data) return;

  const xMetric = document.getElementById('quadrant-x-select')?.value || 'bytes_per_token';
  const yMetric = document.getElementById('quadrant-y-select')?.value || 'macro_val_bpb';

  const models = Object.values(state.data.models);

  // Extract points
  const points = [];
  const xVals = [];
  const yVals = [];

  models.forEach((m) => {
    let xVal = m[xMetric];
    let yVal = null;

    if (yMetric === 'macro_val_bpb') {
      yVal = m.macro_val_bpb;
    } else if (yMetric === 'english_bpb') {
      yVal = m.val_domains?.english?.bpb;
    } else if (yMetric === 'multilingual_bpb') {
      yVal = m.val_domains?.multilingual?.bpb;
    } else if (yMetric === 'code_bpb') {
      yVal = m.val_domains?.code?.bpb;
    } else {
      yVal = m[yMetric];
    }

    if (xVal !== null && xVal !== undefined && yVal !== null && yVal !== undefined) {
      xVals.push(xVal);
      yVals.push(yVal);
      const famColor = getFamilyColor(m.family);
      points.push({
        x: xVal,
        y: yVal,
        model: m,
        color: famColor.border,
        bgColor: famColor.bg,
      });
    }
  });

  if (points.length === 0) return;

  // Compute medians for quadrant lines
  const sortedX = [...xVals].sort((a, b) => a - b);
  const sortedY = [...yVals].sort((a, b) => a - b);
  const medianX = sortedX[Math.floor(sortedX.length / 2)];
  const medianY = sortedY[Math.floor(sortedY.length / 2)];

  // Group by family for clean Chart.js datasets
  const families = [...new Set(points.map((p) => p.model.family))];
  const datasets = families.map((fam) => {
    const famPoints = points.filter((p) => p.model.family === fam);
    const color = getFamilyColor(fam);
    return {
      label: fam,
      data: famPoints.map((p) => ({ x: p.x, y: p.y, model: p.model })),
      backgroundColor: color.bg,
      borderColor: color.border,
      borderWidth: 2,
      pointRadius: 7,
      pointHoverRadius: 10,
    };
  });

  const isDark = isDarkTheme();
  const textColor = isDark ? '#cbd5e1' : '#334155';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';

  if (state.charts.quadrant) {
    state.charts.quadrant.destroy();
  }

  const isYInverted = yMetric.includes('bpb');

  state.charts.quadrant = new Chart(canvas, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: {
            boxWidth: 10,
            font: { size: 11 },
            color: textColor,
          },
        },
        tooltip: {
          backgroundColor: isDark ? 'rgba(15, 23, 42, 0.95)' : 'rgba(255, 255, 255, 0.95)',
          titleColor: isDark ? '#f8fafc' : '#0f172a',
          bodyColor: isDark ? '#cbd5e1' : '#334155',
          borderColor: isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: (ctx) => {
              const m = ctx.raw.model;
              return [
                `${m.display_name} (${m.family})`,
                `X (${xMetric}): ${ctx.raw.x.toFixed(3)}`,
                `Y (${yMetric}): ${ctx.raw.y.toFixed(3)}`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: xMetric.replace(/_/g, ' ').toUpperCase(),
            color: textColor,
            font: { size: 11, weight: 'bold' },
          },
          ticks: { color: textColor, font: { size: 10 } },
          grid: { color: gridColor },
        },
        y: {
          reverse: isYInverted, // Invert Y-axis if BPB so lower BPB is visually at the top!
          title: {
            display: true,
            text: yMetric.replace(/_/g, ' ').toUpperCase() + (isYInverted ? ' (Inverted: UP = Superior Quality)' : ''),
            color: textColor,
            font: { size: 11, weight: 'bold' },
          },
          ticks: { color: textColor, font: { size: 10 } },
          grid: { color: gridColor },
        },
      },
    },
  });
}

// 4. Extrinsic LM Validation BPB & Convergence Curves
function renderExtrinsicTab() {
  if (!state.data) return;

  const models = Object.values(state.data.models).filter((m) => m.macro_val_bpb !== null);

  // Grouped Bar Chart
  const barCanvas = document.getElementById('extrinsic-bar-chart');
  if (barCanvas) {
    if (state.charts.extrinsicBar) state.charts.extrinsicBar.destroy();

    const labels = models.map((m) => m.display_name.replace('WP-BPE ', 'WP-'));
    const englishBpb = models.map((m) => m.val_domains?.english?.bpb || 0);
    const multiBpb = models.map((m) => m.val_domains?.multilingual?.bpb || 0);
    const codeBpb = models.map((m) => m.val_domains?.code?.bpb || 0);
    const macroBpb = models.map((m) => m.macro_val_bpb || 0);

    const isDark = isDarkTheme();
    const textColor = isDark ? '#cbd5e1' : '#334155';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';

    state.charts.extrinsicBar = new Chart(barCanvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'English BPB', data: englishBpb, backgroundColor: '#38bdf8' },
          { label: 'Multilingual BPB', data: multiBpb, backgroundColor: '#a855f7' },
          { label: 'Code BPB', data: codeBpb, backgroundColor: '#10b981' },
          { label: 'Macro Avg BPB', data: macroBpb, backgroundColor: '#f59e0b' },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { color: textColor, font: { size: 10 } } },
        },
        scales: {
          x: { ticks: { color: textColor, font: { size: 9 }, maxRotation: 45 }, grid: { display: false } },
          y: { title: { display: true, text: 'Bits-Per-Byte (BPB)', color: textColor }, ticks: { color: textColor }, grid: { color: gridColor } },
        },
      },
    });
  }

  // Convergence Line Chart
  const curveCanvas = document.getElementById('extrinsic-curve-chart');
  if (curveCanvas) {
    if (state.charts.extrinsicCurve) state.charts.extrinsicCurve.destroy();

    // Select subset of distinct models for clean visualization
    const curveModels = models.filter((m) => m.curve && m.curve.length > 0);
    const datasets = curveModels.map((m) => {
      const color = getFamilyColor(m.family);
      return {
        label: m.display_name,
        data: m.curve.map((pt) => ({ x: pt.step, y: pt.train_bpb })),
        borderColor: color.border,
        backgroundColor: color.bg,
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.2,
      };
    });

    const isDark = isDarkTheme();
    const textColor = isDark ? '#cbd5e1' : '#334155';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';

    state.charts.extrinsicCurve = new Chart(curveCanvas, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { color: textColor, boxWidth: 8, font: { size: 9 } } },
        },
        scales: {
          x: { title: { display: true, text: 'Training Steps (Ingested Token Stream)', color: textColor, font: { size: 10 } }, ticks: { color: textColor }, grid: { color: gridColor } },
          y: {
            title: { display: true, text: 'Train BPB', color: textColor, font: { size: 10 } },
            ticks: { color: textColor },
            grid: { color: gridColor },
            max: 20, // Clip initial spike for readable trajectory
          },
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
      .sort((a, b) => a.macro_val_bpb - b.macro_val_bpb)
      .map((m) => {
        const en = m.val_domains?.english?.bpb?.toFixed(3) || 'N/A';
        const multi = m.val_domains?.multilingual?.bpb?.toFixed(3) || 'N/A';
        const code = m.val_domains?.code?.bpb?.toFixed(3) || 'N/A';
        const macro = m.macro_val_bpb ? m.macro_val_bpb.toFixed(3) : 'N/A';
        const delta = m.macro_val_bpb ? (m.macro_val_bpb - baseMacroBpb).toFixed(3) : 'N/A';
        const deltaClass = parseFloat(delta) <= 0 ? 'text-success font-bold' : 'text-error';

        return `
        <tr class="hover">
          <td class="font-bold">${m.display_name}</td>
          <td><span class="badge badge-xs badge-outline">${m.family}</span></td>
          <td class="font-mono">${en}</td>
          <td class="font-mono">${multi}</td>
          <td class="font-mono">${code}</td>
          <td class="font-mono font-bold text-primary">${macro}</td>
          <td class="font-mono ${deltaClass}">${delta > 0 ? '+' + delta : delta}</td>
        </tr>
      `;
      })
      .join('');
  }
}

// 5. Tokka-Bench 11-Language Heatmap & Grouped Chart
function renderTokkaTab() {
  if (!state.data || !state.data.tokka_matrix) return;

  const matrix = state.data.tokka_matrix;
  const models = Object.keys(matrix);
  const langs = ['en', 'de', 'fr', 'ru', 'zh', 'ja', 'ar', 'hi', 'hu', 'fi', 'code'];
  const metric = state.tokkaActiveMetric;

  // Heatmap Table
  const tbody = document.getElementById('tokka-heatmap-tbody');
  if (tbody) {
    tbody.innerHTML = models
      .map((mKey) => {
        const meta = state.data.models[mKey] || {};
        const row = matrix[mKey];
        const macroVal = row?.macro_avg?.[metric];

        const cells = langs
          .map((lang) => {
            const v = row?.[lang]?.[metric];
            const displayVal = v !== undefined && v !== null ? (metric === 'continued_word_rate' ? v.toFixed(1) + '%' : v.toFixed(2)) : '--';
            return `<td class="heatmap-cell font-mono text-center">${displayVal}</td>`;
          })
          .join('');

        const macroDisplay = macroVal !== undefined && macroVal !== null ? (metric === 'continued_word_rate' ? macroVal.toFixed(1) + '%' : macroVal.toFixed(2)) : '--';

        return `
        <tr class="hover">
          <td class="font-bold">${meta.display_name || mKey}</td>
          ${cells}
          <td class="heatmap-cell font-mono font-bold text-primary bg-primary/10 text-center">${macroDisplay}</td>
        </tr>
      `;
      })
      .join('');
  }

  // Bar Chart
  const barCanvas = document.getElementById('tokka-bar-chart');
  if (barCanvas) {
    if (state.charts.tokkaBar) state.charts.tokkaBar.destroy();

    const isDark = isDarkTheme();
    const textColor = isDark ? '#cbd5e1' : '#334155';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';

    const datasets = models.slice(0, 6).map((mKey) => {
      const meta = state.data.models[mKey] || {};
      const color = getFamilyColor(meta.family);
      const vals = langs.map((lang) => matrix[mKey]?.[lang]?.[metric] || 0);

      return {
        label: meta.display_name || mKey,
        data: vals,
        backgroundColor: color.bg,
        borderColor: color.border,
        borderWidth: 1,
      };
    });

    state.charts.tokkaBar = new Chart(barCanvas, {
      type: 'bar',
      data: {
        labels: langs.map((l) => l.toUpperCase()),
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { color: textColor, boxWidth: 10, font: { size: 10 } } },
        },
        scales: {
          x: { ticks: { color: textColor }, grid: { display: false } },
          y: {
            title: { display: true, text: metric.replace(/_/g, ' ').toUpperCase(), color: textColor, font: { size: 10 } },
            ticks: { color: textColor },
            grid: { color: gridColor },
          },
        },
      },
    });
  }
}

// 6. Intrinsic & Morphology Leaderboard Tab
function renderIntrinsicTab() {
  if (!state.data) return;

  const tbody = document.getElementById('intrinsic-tbody');
  if (!tbody) return;

  const searchTerm = (document.getElementById('intrinsic-search')?.value || '').toLowerCase();
  const models = Object.values(state.data.models).filter((m) => {
    return m.display_name.toLowerCase().includes(searchTerm) || m.family.toLowerCase().includes(searchTerm);
  });

  tbody.innerHTML = models
    .map((m) => {
      const intr = m.intrinsic || {};
      return `
      <tr class="hover">
        <td class="font-bold">${m.display_name}</td>
        <td class="font-mono text-center">${intr.vocab_size || '--'}</td>
        <td class="font-mono">${intr.bytes_per_token ? intr.bytes_per_token.toFixed(3) : '--'}</td>
        <td class="font-mono">${intr.fertility ? intr.fertility.toFixed(2) : '--'}</td>
        <td class="font-mono">${intr.renyi_e_1_0 ? intr.renyi_e_1_0.toFixed(3) : '--'}</td>
        <td class="font-mono font-bold text-primary">${intr.renyi_e_2_0 ? intr.renyi_e_2_0.toFixed(3) : '--'}</td>
        <td class="font-mono">${intr.renyi_e_2_5 ? intr.renyi_e_2_5.toFixed(3) : '--'}</td>
        <td class="font-mono">${intr.gini_coefficient ? intr.gini_coefficient.toFixed(3) : '--'}</td>
        <td class="font-mono">${intr.active_fraction ? (intr.active_fraction * 100).toFixed(1) + '%' : '--'}</td>
        <td class="font-mono">${intr.morph_precision ? (intr.morph_precision * 100).toFixed(1) + '%' : '--'}</td>
        <td class="font-mono">${intr.morph_recall ? (intr.morph_recall * 100).toFixed(1) + '%' : '--'}</td>
        <td class="font-mono font-bold text-accent">${intr.morph_f1 ? (intr.morph_f1 * 100).toFixed(1) + '%' : '--'}</td>
        <td class="font-mono">${intr.identifier_f1 ? (intr.identifier_f1 * 100).toFixed(1) + '%' : '--'}</td>
        <td class="font-mono">${intr.encode_mb_per_sec ? intr.encode_mb_per_sec.toFixed(2) : '--'}</td>
        <td class="font-mono font-bold text-success">${intr.decode_mb_per_sec ? intr.decode_mb_per_sec.toFixed(1) : '--'}</td>
      </tr>
    `;
    })
    .join('');
}

// 7. Multi-Axis Radar View & WP-BPE Deep Dive
function renderRadarView() {
  if (!state.data) return;

  const canvas = document.getElementById('radar-chart');
  const togglesContainer = document.getElementById('radar-toggles');
  if (!canvas || !togglesContainer) return;

  const allModels = Object.values(state.data.models);

  // Render toggle pill buttons
  togglesContainer.innerHTML = allModels
    .map((m) => {
      const isSelected = state.selectedRadarModels.includes(m.key);
      const btnClass = isSelected ? 'btn-primary' : 'btn-outline';
      return `
      <button class="btn btn-xs ${btnClass}" onclick="toggleRadarModel('${m.key}')">
        ${m.display_name}
      </button>
    `;
    })
    .join('');

  if (state.charts.radar) state.charts.radar.destroy();

  // Metrics to normalize for 6-axis radar:
  // 1. Compression: bytes_per_token
  // 2. Info Efficiency: renyi_e_2_0
  // 3. Morphology: morph_f1
  // 4. Identifiers: identifier_f1
  // 5. Downstream Quality: 1 / macro_val_bpb
  // 6. Decode Speed: decode_mb_per_sec
  const getMinMax = (extractor) => {
    const vals = allModels.map(extractor).filter((v) => v !== null && v !== undefined && !isNaN(v));
    return { min: Math.min(...vals), max: Math.max(...vals) };
  };

  const bptMM = getMinMax((m) => m.bytes_per_token);
  const renyiMM = getMinMax((m) => m.renyi_e_2_0);
  const morphMM = getMinMax((m) => m.morph_f1);
  const identMM = getMinMax((m) => m.identifier_f1);
  const bpbMM = getMinMax((m) => (m.macro_val_bpb ? 1 / m.macro_val_bpb : null));
  const speedMM = getMinMax((m) => m.decode_mb_per_sec);

  const normalize = (val, mm) => {
    if (val === null || val === undefined || isNaN(val)) return 0.2;
    if (mm.max === mm.min) return 1.0;
    return Math.max(0.1, Math.min(1.0, (val - mm.min) / (mm.max - mm.min)));
  };

  const datasets = state.selectedRadarModels
    .map((mKey) => {
      const m = state.data.models[mKey];
      if (!m) return null;
      const color = getFamilyColor(m.family);

      const normData = [
        normalize(m.bytes_per_token, bptMM),
        normalize(m.renyi_e_2_0, renyiMM),
        normalize(m.morph_f1, morphMM),
        normalize(m.identifier_f1, identMM),
        normalize(m.macro_val_bpb ? 1 / m.macro_val_bpb : null, bpbMM),
        normalize(m.decode_mb_per_sec, speedMM),
      ];

      return {
        label: m.display_name,
        data: normData,
        backgroundColor: color.bg.replace('0.75', '0.25'),
        borderColor: color.border,
        borderWidth: 2,
        pointBackgroundColor: color.border,
        pointRadius: 3,
      };
    })
    .filter(Boolean);

  const isDark = isDarkTheme();
  const textColor = isDark ? '#cbd5e1' : '#334155';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.08)';

  state.charts.radar = new Chart(canvas, {
    type: 'radar',
    data: {
      labels: [
        'Compression (Bytes/Tok)',
        'Entropy (Rényi E_2.0)',
        'Morphology F1',
        'Identifier F1',
        'Downstream (1/BPB)',
        'Throughput (MB/s)',
      ],
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { color: textColor, font: { size: 10 } } },
      },
      scales: {
        r: {
          min: 0,
          max: 1.0,
          ticks: { display: false },
          pointLabels: { color: textColor, font: { size: 10, weight: 'bold' } },
          grid: { color: gridColor },
          angleLines: { color: gridColor },
        },
      },
    },
  });
}

function toggleRadarModel(mKey) {
  if (state.selectedRadarModels.includes(mKey)) {
    if (state.selectedRadarModels.length > 1) {
      state.selectedRadarModels = state.selectedRadarModels.filter((k) => k !== mKey);
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
  const targetCard = document.getElementById(`tok-card-${modelKey}`);
  if (targetCard) {
    targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    targetCard.classList.add('ring-2', 'ring-primary');
    setTimeout(() => targetCard.classList.remove('ring-2', 'ring-primary'), 2000);
  }
}

const PILL_COLOR_STYLES = [
  'bg-blue-500/15 border-blue-500/30 text-blue-300',
  'bg-emerald-500/15 border-emerald-500/30 text-emerald-300',
  'bg-amber-500/15 border-amber-500/30 text-amber-300',
  'bg-purple-500/15 border-purple-500/30 text-purple-300',
  'bg-pink-500/15 border-pink-500/30 text-pink-300',
  'bg-cyan-500/15 border-cyan-500/30 text-cyan-300',
];

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
      resultsContainer.innerHTML = `
        <div class="alert alert-error text-xs">
          <span>Failed to execute tokenization: ${err.message}</span>
        </div>
      `;
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
    container.innerHTML = '<div class="text-xs text-base-content/60">No tokenizer results available.</div>';
    return;
  }

  container.innerHTML = results
    .map((res) => {
      const famColor = getFamilyColor(res.family);

      // Render token pills
      const pillHtml = res.token_details
        .map((tok, idx) => {
          const colorClass = PILL_COLOR_STYLES[idx % PILL_COLOR_STYLES.length];
          // Visual treatment for spaces
          const displayPiece = tok.piece
            .replace(/ /g, '<span class="token-space">␣</span>')
            .replace(/\n/g, '<span class="token-space">↵\\n</span>');

          const tooltip = `Token ID: #${tok.id} | ${tok.byte_len} bytes`;

          return `
          <span class="token-pill ${colorClass}" title="${tooltip}" data-token-id="${tok.id}">
            <span>${displayPiece}</span>
          </span>
        `;
        })
        .join('');

      return `
      <div id="tok-card-${res.name}" class="card bg-base-200/60 border border-base-content/10 shadow-sm transition-all duration-300">
        <div class="card-body p-4 space-y-2.5">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-base-content/5 pb-2">
            <div class="flex items-center space-x-2">
              <span class="font-bold text-sm text-base-content">${res.display_name}</span>
              <span class="badge badge-xs" style="background-color: ${famColor.bg}; color: #fff; border-color: ${famColor.border};">${res.family}</span>
            </div>
            <div class="flex items-center space-x-2 text-xs font-mono">
              <span class="badge badge-sm badge-neutral"><strong>${res.num_tokens}</strong>&nbsp;tokens</span>
              <span class="badge badge-sm badge-outline"><strong>${res.bytes_per_token}</strong>&nbsp;B/tok</span>
              <span class="badge badge-sm badge-ghost text-base-content/60">${res.encode_ms} ms</span>
            </div>
          </div>
          <div class="p-2.5 rounded-lg bg-base-100/90 border border-base-content/5 leading-relaxed overflow-x-auto custom-scroll">
            ${pillHtml}
          </div>
        </div>
      </div>
    `;
    })
    .join('');
}

// Event Listeners Setup
function setupEventListeners() {
  // Family Filter Buttons
  document.querySelectorAll('.family-filter-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.family-filter-btn').forEach((b) => {
        b.classList.remove('btn-primary');
        b.classList.add('btn-outline');
      });
      e.target.classList.remove('btn-outline');
      e.target.classList.add('btn-primary');

      state.activeFamilyFilter = e.target.getAttribute('data-family');
      renderLeaderboard();
    });
  });

  // Quadrant Selectors
  document.getElementById('quadrant-x-select')?.addEventListener('change', renderQuadrantChart);
  document.getElementById('quadrant-y-select')?.addEventListener('change', renderQuadrantChart);

  // Tokka Metric Switcher
  document.querySelectorAll('.tokka-metric-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.tokka-metric-btn').forEach((b) => {
        b.classList.remove('btn-primary');
        b.classList.add('btn-outline');
      });
      e.target.classList.remove('btn-outline');
      e.target.classList.add('btn-primary');

      state.tokkaActiveMetric = e.target.getAttribute('data-metric');
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

  // Refresh Button
  document.getElementById('refresh-btn')?.addEventListener('click', fetchBenchmarkData);
}
