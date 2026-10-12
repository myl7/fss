'use strict';

// Static comparison page over the measured records exported by the figure
// pipeline (doc/site/data.json). No dependencies: the charts are hand-built
// SVG and the independent variable is always the domain size N.

const FAMILY = {DCF: 'dcf', GrottoDCF: 'dcf', DPF: 'dpf', HalfTreeDPF: 'dpf',
  PackedHalfTreeDPF: 'dpf', VDPF: 'dpf', DMPF: 'dmpf', VDMPF: 'dmpf'};
const OPERATIONS = ['Gen', 'Eval', 'EvalAll'];
const LIBRARY_NAMES = {fss: 'FSS (this repo)', fss_v070: 'FSS 0.7.0', fss_v060: 'FSS 0.6.0',
  google_dpf: 'Google DPF', gpu_dpf: 'GPU-DPF', ezpc: 'EzPC', libdpf: 'libdpf',
  libfss: 'libfss', servan_vdpf: 'Servan-Schreiber VDPF'};
const LIBRARY_SHORT = {fss: 'FSS', fss_v070: 'FSS 0.7.0', fss_v060: 'FSS 0.6.0',
  google_dpf: 'Google', gpu_dpf: 'GPU-DPF', ezpc: 'EzPC', libdpf: 'libdpf',
  libfss: 'libfss', servan_vdpf: 'Servan-Schreiber'};
const SCHEME_NAMES = {DPF: 'DPF', DCF: 'DCF', HalfTreeDPF: 'HalfTree DPF',
  PackedHalfTreeDPF: 'Packed HalfTree DPF', VDPF: 'VDPF', GrottoDCF: 'Grotto DCF',
  DMPF: 'DMPF', VDMPF: 'VDMPF'};
const SCHEME_ORDER = ['DPF', 'HalfTreeDPF', 'PackedHalfTreeDPF', 'VDPF', 'DCF', 'GrottoDCF', 'DMPF', 'VDMPF'];
const PRG_NAMES = {'AES-NI': 'AES-128 MMO, AES-NI', 'OpenSSL-AES': 'AES-128 MMO, OpenSSL',
  AES128: 'AES-128', 'AES128-MMO/OpenSSL': 'AES-128 MMO, OpenSSL', 'AES128-MMO/RustCrypto': 'AES-128 MMO, RustCrypto',
  'AES128/OpenSSL': 'AES-128, OpenSSL EVP', 'AES128-MMO-NI': 'AES-128 MMO, AES-NI',
  'AES-software': 'AES-128, software', 'ChaCha20-x1': 'ChaCha20 ×1 block', 'ChaCha20-x2': 'ChaCha20 ×2 blocks',
  'ChaCha20-x4': 'ChaCha20 ×4 blocks', ChaCha12: 'ChaCha12', Salsa12: 'Salsa12'};
const TIMING_NAMES = {wall_time_operation: 'wall-clock time per operation',
  cuda_event_operation_kernel_launches: 'CUDA events around kernel launches',
  cuda_event_native_key_generation_api: 'CUDA events around native GPU key-generation calls',
  cuda_event_native_point_evaluation_api: 'CUDA events around native GPU point-evaluation calls',
  cuda_event_packed_eval_all_kernel: 'CUDA events around the packed full-domain kernel',
  cuda_event_native_dpf_hybrid_api: 'CUDA events around the native dpf_hybrid call'};
const PREFERRED_B = ['libdpf', 'google_dpf', 'ezpc', 'libfss', 'servan_vdpf', 'gpu_dpf', 'fss_v070', 'fss_v060', 'fss'];

let variants = new Map();
const state = {platform: 'cpu', a: null, b: null, op: null};

function outputQualifier(row) {
  const group = {bytes: 'XOR byte group', prime_field: `${row.logical_output_bits}-bit prime field`,
    additive_mod_2_64: 'additive mod 2^64', additive_mod_2_128: 'additive mod 2^128', xor_128: 'XOR 128-bit'}[row.group];
  return {packed_128_binary_leaf: '1-bit packed', packed_bits: `${row.logical_output_bits}-bit packed`,
    packed_lanes: `${row.logical_output_bits}-bit packed`, bool_scalar: '1-bit bool',
    gmp_scalar: `${row.logical_output_bits}-bit prime field`,
    uint128_scalar: `${row.logical_output_bits}-bit integer`}[row.output_storage] ?? group ??
    (row.logical_output_bits ? `${row.logical_output_bits}-bit output` : null);
}

function buildVariants(records) {
  for (const row of records) {
    const key = `${row.platform}|${row.variant}`;
    if (!variants.has(key)) variants.set(key, {key, platform: row.platform, family: FAMILY[row.scheme],
      variant: row.variant, library: row.library, scheme: row.scheme, ops: new Map(), sample: row});
    const entry = variants.get(key);
    if (!entry.ops.has(row.operation)) entry.ops.set(row.operation, []);
    entry.ops.get(row.operation).push({bits: row.domain_bits, ns: row.ns_per_key, keys: row.num_keys});
  }
  for (const entry of variants.values()) {
    for (const [operation, points] of entry.ops) {
      points.sort((x, y) => x.bits - y.bits);
      entry.ops.set(operation, Object.freeze(points));
    }
  }
}

function platformVariants(platform) {
  return [...variants.values()].filter(entry => entry.platform === platform)
    .sort((a, b) => (a.library === 'fss') - (b.library === 'fss') || a.library.localeCompare(b.library) ||
      SCHEME_ORDER.indexOf(a.scheme) - SCHEME_ORDER.indexOf(b.scheme));
}

function normalizeSelection() {
  const list = platformVariants(state.platform);
  if (!state.a || state.a.platform !== state.platform) {
    const previous = state.a;
    state.a = (previous && list.find(entry => entry.library === previous.library && entry.scheme === previous.scheme)) ??
      list.find(entry => entry.library === 'fss' && entry.scheme === 'DPF') ??
      list.find(entry => entry.library === 'fss') ?? list[0];
  }
  const familyList = list.filter(entry => entry.family === state.a.family && entry.key !== state.a.key);
  if (!state.b || state.b.platform !== state.platform || state.b.family !== state.a.family || state.b.key === state.a.key) {
    state.b = familyList.find(entry => entry.library === state.b?.library) ??
      PREFERRED_B.map(library => familyList.find(entry => entry.library === library)).find(Boolean) ??
      familyList[0];
  }
  const available = OPERATIONS.filter(operation => state.a.ops.has(operation) || state.b.ops.has(operation));
  if (!available.includes(state.op)) state.op = available[0] ?? null;
}

function optionLabel(entry) {
  // Library prefix keeps variants apart when two libraries share a scheme and
  // PRG backend (FSS and FSS 0.7.0 both run AES-NI MMO DCF).
  return `${LIBRARY_SHORT[entry.library] ?? entry.library} ${SCHEME_NAMES[entry.scheme]} — ` +
    `${[outputQualifier(entry.sample), PRG_NAMES[entry.sample.prg] ?? entry.sample.prg].filter(Boolean).join(', ')}`;
}

function optionMarkup(entries, selected) {
  return entries.map(entry =>
    `<option value="${entry.key}"${entry.key === selected?.key ? ' selected' : ''}>${optionLabel(entry)}</option>`).join('');
}

function populateSelects() {
  document.getElementById('select-a').innerHTML = optionMarkup(platformVariants(state.platform), state.a);
  const familyList = platformVariants(state.platform).filter(entry => entry.family === state.a.family);
  document.getElementById('select-b').innerHTML = optionMarkup(familyList, state.b);
  document.getElementById('select-b').disabled = familyList.length <= 1;
}

function variantLabel(entry) {
  const suffix = entry.library === 'servan_vdpf' ? '' : ` ${SCHEME_NAMES[entry.scheme]}`;
  return `${LIBRARY_NAMES[entry.library] ?? entry.library}${suffix}`;
}

function configLine(entry) {
  const row = entry.sample;
  const keysPerOperation = OPERATIONS.filter(operation => entry.ops.has(operation))
    .map(operation => `${operation}: K=${entry.ops.get(operation)[0].keys.toLocaleString('en-US')}`);
  const statistic = row.statistic === 'median_of_process_medians' ? 'median of 5 process medians' : 'median of 5 repetitions';
  const bits = row.logical_output_bits;
  const storage = Number.isFinite(row.storage_output_bits) ? ` in ${row.storage_output_bits}-bit storage` : '';
  return [outputQualifier(row) + (bits ? ` (${bits} logical bit${bits > 1 ? 's' : ''}${storage})` : ''),
    `PRG: ${PRG_NAMES[row.prg] ?? row.prg}`, keysPerOperation.join(', '),
    row.num_points > 1 ? `t=${row.num_points} points` : null,
    TIMING_NAMES[row.timing_boundary] ?? row.timing_boundary, statistic].filter(Boolean).join(' · ');
}

function renderConfigSummary() {
  document.getElementById('config-summary').innerHTML =
    `<div class="config-a">A — ${variantLabel(state.a)}: ${configLine(state.a)}</div>` +
    `<div class="config-b">B — ${variantLabel(state.b)}: ${configLine(state.b)}</div>`;
}

const superscript = value => String(value).split('').map(char => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(char)] ?? char).join('');
const trimmed = value => String(parseFloat(value.toPrecision(4)));

function axisTick(value) {
  if (value >= 1e6) return `${trimmed(value / 1e6)}M`;
  if (value >= 1e3) return `${trimmed(value / 1e3)}k`;
  return trimmed(value);
}

function timeUnit(nsValues) {
  const max = Math.max(...nsValues);
  return max >= 1e6 ? 'ms' : max >= 1e3 ? 'µs' : 'ns';
}
const unitScale = {ns: 1, 'µs': 1e3, ms: 1e6};

// Log-scale y ticks: decades on wide ranges, 2× and 5× steps in between, and
// minor steps on narrow ranges so a nearly flat curve still shows a readable
// logarithmic axis.
function yTicks(min, max, spacingPx, position) {
  const ticks = [];
  const span = Math.log10(max / min);
  const factors = span > 2.2 ? [1] : span < 0.9 ? [1, 2, 3, 4, 5, 6, 8, 9] : [1, 2, 5];
  for (let exponent = Math.floor(Math.log10(min)); exponent <= Math.ceil(Math.log10(max)); exponent++) {
    for (const factor of factors) {
      const value = factor * 10 ** exponent;
      if (value >= min && value <= max && (!ticks.length || Math.abs(position(value) - position(ticks[ticks.length - 1])) >= spacingPx)) ticks.push(value);
    }
  }
  if (ticks.length < 3) {
    for (const share of [0, 1 / 3, 2 / 3, 1]) {
      ticks.push(10 ** (Math.log10(min) + span * share));
    }
  }
  return ticks;
}

function panelElement(operation, series) {
  const width = 320, height = 240, left = 50, right = 10, top = 10, bottom = 42;
  const allPoints = series.flatMap(item => item.entry.ops.get(operation));
  const xs = allPoints.map(point => point.bits), ys = allPoints.map(point => point.ns);
  const xMin = Math.min(...xs) - 0.3, xMax = Math.max(...xs) + 0.3;
  const yMin = Math.min(...ys) * 0.6, yMax = Math.max(...ys) * 1.9;
  const x = value => left + (value - xMin) / (xMax - xMin) * (width - left - right);
  const y = value => top + (Math.log10(yMax) - Math.log10(value)) / (Math.log10(yMax) - Math.log10(yMin)) * (height - top - bottom);
  const unit = timeUnit(ys);
  const svg = [`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${operation} comparison">`];
  for (const tick of yTicks(yMin, yMax, 14, y)) {
    svg.push(`<line x1="${left}" x2="${width - right}" y1="${y(tick).toFixed(1)}" y2="${y(tick).toFixed(1)}" stroke="#e5e7e9" stroke-width="1"/>`);
    svg.push(`<text x="${left - 6}" y="${(y(tick) + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="#55595e">${axisTick(tick)}</text>`);
  }
  svg.push(`<rect x="${left}" y="${top}" width="${width - left - right}" height="${height - top - bottom}" fill="none" stroke="#333" stroke-width="1"/>`);
  const xTicks = [...new Set(xs)].sort((a, b) => a - b).filter((value, index, list) => list.length <= 6 || index % 2 === 0);
  for (const tick of xTicks) {
    svg.push(`<text x="${x(tick).toFixed(1)}" y="${height - bottom + 16}" text-anchor="middle" font-size="11" fill="#55595e">2${superscript(tick)}</text>`);
  }
  svg.push(`<text x="${(left + width - right) / 2}" y="${height - 6}" text-anchor="middle" font-size="11.5" fill="#55595e">Domain size, N</text>`);
  svg.push(`<text x="12" y="${(top + height - bottom) / 2}" text-anchor="middle" font-size="11.5" fill="#55595e" transform="rotate(-90 12 ${(top + height - bottom) / 2})">Time (${unit}/key)</text>`);
  for (const item of series) {
    const points = item.entry.ops.get(operation);
    svg.push(`<polyline fill="none" stroke="${item.color}" stroke-width="2"${item.dash ? ` stroke-dasharray="${item.dash}"` : ''} points="${
      points.map(point => `${x(point.bits).toFixed(1)},${y(point.ns).toFixed(1)}`).join(' ')}"/>`);
    for (const point of points) svg.push(`<circle cx="${x(point.bits).toFixed(1)}" cy="${y(point.ns).toFixed(1)}" r="2.7" fill="${item.color}"/>`);
  }
  svg.push('</svg>');
  const missing = tag => series.length === 1 ? ` (${tag}: no data)` : '';
  const title = `<p class="panel-title">${operation} <span class="series">· <span class="series-a">${variantLabel(state.a)}</span>` +
    ` vs <span class="series-b">${variantLabel(state.b)}</span>${state.a.ops.has(operation) ? '' : missing('A')}${state.b.ops.has(operation) ? '' : missing('B')}</span></p>`;
  const panel = document.createElement('div');
  panel.className = 'panel';
  panel.innerHTML = title + svg.join('');
  return panel;
}

function renderPanels() {
  const container = document.getElementById('panels');
  container.innerHTML = '';
  const available = OPERATIONS.filter(operation => state.a.ops.has(operation) || state.b.ops.has(operation));
  for (const operation of available) {
    const series = [];
    if (state.a.ops.has(operation)) series.push({entry: state.a, color: 'var(--a)', dash: null});
    if (state.b.ops.has(operation)) series.push({entry: state.b, color: 'var(--b)', dash: '5 3'});
    container.appendChild(panelElement(operation, series));
  }
}

function renderTabs() {
  const tabs = document.getElementById('operation-tabs');
  const available = OPERATIONS.filter(operation => state.a.ops.has(operation) || state.b.ops.has(operation));
  tabs.innerHTML = available.map(operation =>
    `<button type="button" data-operation="${operation}"${operation === state.op ? ' class="active"' : ''}>${operation}</button>`).join('');
  for (const button of tabs.querySelectorAll('button')) {
    button.addEventListener('click', () => { state.op = button.dataset.operation; update(); });
  }
}

function renderTable() {
  const head = document.querySelector('#values-table thead');
  const body = document.querySelector('#values-table tbody');
  const note = document.getElementById('table-note');
  if (!state.op) {
    head.innerHTML = body.innerHTML = '';
    note.textContent = 'No measurements for this pair.';
    return;
  }
  const aPoints = state.a.ops.has(state.op) ? state.a.ops.get(state.op) : [];
  const bPoints = state.b.ops.has(state.op) ? state.b.ops.get(state.op) : [];
  const unitA = aPoints.length ? timeUnit(aPoints.map(point => point.ns)) : null;
  const unitB = bPoints.length ? timeUnit(bPoints.map(point => point.ns)) : null;
  head.innerHTML = '<tr><th scope="col">N</th>' +
    `<th scope="col"><span class="head-a">A — ${variantLabel(state.a)}</span>${unitA ? `, ${unitA}/key` : ''}</th>` +
    `<th scope="col"><span class="head-b">B — ${variantLabel(state.b)}</span>${unitB ? `, ${unitB}/key` : ''}</th>` +
    '<th scope="col">A ÷ B</th></tr>';
  const aByBits = new Map(aPoints.map(point => [point.bits, point.ns]));
  const bByBits = new Map(bPoints.map(point => [point.bits, point.ns]));
  const bits = [...new Set([...aByBits.keys(), ...bByBits.keys()])].sort((x, y) => x - y);
  const cell = (value, unit) => value === undefined ? '<td class="missing">—</td>' :
    `<td>${trimmed(value / unitScale[unit])}&#8239;${unit}</td>`;
  let hasGap = false;
  const rows = bits.map(bit => {
    const a = aByBits.get(bit), b = bByBits.get(bit);
    if (a === undefined || b === undefined) hasGap = true;
    return `<tr><td>2${superscript(bit)} (${(2 ** bit).toLocaleString('en-US')})</td>${cell(a, unitA)}${cell(b, unitB)}` +
      `<td>${a !== undefined && b !== undefined ? `${(a / b).toFixed(2)}×` : '—'}</td></tr>`;
  });
  body.innerHTML = rows.join('');
  const details = [];
  for (const [tag, entry] of [['A', state.a], ['B', state.b]]) {
    if (entry.ops.has(state.op) && entry.ops.get(state.op)[0].keys > 1) {
      details.push(`${tag}: per-key time amortized over K=${entry.ops.get(state.op)[0].keys.toLocaleString('en-US')} keys`);
    }
  }
  if (hasGap) details.push('—: not measured at that domain size for that implementation');
  note.textContent = details.join('. ') + (details.length ? '.' : '');
}

function renderProvenance(runs) {
  const unique = field => [...new Set(runs.map(run => run[field]).filter(Boolean))];
  const dates = runs.map(run => run.date).filter(Boolean).sort();
  const range = dates.length ? `${dates[0].slice(0, 10)} – ${dates[dates.length - 1].slice(0, 10)}` : null;
  document.getElementById('provenance').textContent =
    [range ? `Measured ${range}` : null, unique('host').length ? `on host ${unique('host').join(', ')}` : null,
      unique('cpu').length ? `(${unique('cpu').join('; ')})` : null,
      unique('gpu').length ? `with ${unique('gpu').join('; ')}` : null,
      unique('git_revision').length ? `at fss revisions ${unique('git_revision').map(revision => revision.slice(0, 7)).join(', ')}` : null]
      .filter(Boolean).join(' ') + '.';
}

function syncPlatformToggle() {
  for (const button of document.querySelectorAll('#platform-toggle button')) {
    button.classList.toggle('active', button.dataset.platform === state.platform);
  }
}

let suppressHashChange = false;

function update() {
  normalizeSelection();
  populateSelects();
  renderConfigSummary();
  renderPanels();
  renderTabs();
  renderTable();
  syncPlatformToggle();
  const hash = `p=${state.platform}&a=${encodeURIComponent(state.a.variant)}&b=${encodeURIComponent(state.b.variant)}` +
    (state.op ? `&op=${state.op}` : '');
  if (location.hash !== `#${hash}`) {
    suppressHashChange = true;
    location.hash = hash;
  }
  document.title = `${variantLabel(state.a)} vs ${variantLabel(state.b)} — FSS benchmark comparison`;
}

function restoreFromHash() {
  const parameters = new URLSearchParams(location.hash.slice(1));
  const platform = parameters.get('p');
  if (platform === 'cpu' || platform === 'gpu') state.platform = platform;
  for (const slot of ['a', 'b']) {
    const variant = parameters.get(slot);
    if (variant) state[slot] = [...variants.values()].find(entry => entry.platform === state.platform && entry.variant === variant) ?? null;
  }
  const operation = parameters.get('op');
  if (OPERATIONS.includes(operation)) state.op = operation;
}

function wireControls() {
  document.getElementById('select-a').addEventListener('change', event => {
    state.a = variants.get(event.target.value);
    update();
  });
  document.getElementById('select-b').addEventListener('change', event => {
    state.b = variants.get(event.target.value);
    update();
  });
  for (const button of document.querySelectorAll('#platform-toggle button')) {
    button.addEventListener('click', () => {
      state.platform = button.dataset.platform;
      update();
    });
  }
}

async function main() {
  const response = await fetch('data.json');
  if (!response.ok) throw new Error(`data.json: HTTP ${response.status}`);
  const data = await response.json();
  buildVariants(data.records);
  restoreFromHash();
  wireControls();
  renderProvenance(data.runs);
  update();
  // Follow shared-comparison links pasted into the same tab.
  window.addEventListener('hashchange', () => {
    if (suppressHashChange) { suppressHashChange = false; return; }
    restoreFromHash();
    update();
  });
}

main().catch(error => {
  document.getElementById('table-note').textContent = `failed to load data: ${error.message}`;
});
