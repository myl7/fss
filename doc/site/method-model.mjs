// Shared identities, styles, and time units for static figures and the site.
const LIBRARIES = {fss: 'FSS', fss_v060: 'FSS 0.6.0', fss_v070: 'FSS 0.7.0',
  google_dpf: 'Google', gpu_dpf: 'GPU-DPF', libfss: 'libfss', libdpf: 'libdpf',
  ezpc: 'EzPC', servan_vdpf: 'Servan-Schreiber', mastic: 'Mastic', plasma: 'PLASMA',
  'dpf-cpp': 'dpf-cpp', dpfplusplus: 'dpf++'};
const SCHEMES = {HalfTreeDPF: 'HalfTree', PackedHalfTreeDPF: 'Packed HalfTree', GrottoDCF: 'Grotto DCF'};
const COLORS = {DPF: '#1670b8', DCF: '#008579', HalfTreeDPF: '#164a79',
  PackedHalfTreeDPF: '#607a2f', VDPF: '#78449b', GrottoDCF: '#b18838', DMPF: '#1670b8', VDMPF: '#78449b'};
const BASELINES = {fss_v060: '#7d8796', fss_v070: '#a685b4', ezpc: '#c07852',
  gpu_dpf: '#729875', libdpf: '#ac8831', libfss: '#a95d71', google_dpf: '#877a67',
  servan_vdpf: '#c2564a', mastic: '#8f6394', plasma: '#688e99', 'dpf-cpp': '#67794c', dpfplusplus: '#a16335'};
const DASHES = ['', '7,3', '2,3', '9,3,2,3', '5,2,2,2', '11,4'];
export const unitScale = {ns: 1, 'µs': 1e3, ms: 1e6};
export const scaleTime = (ns, unit) => ns / unitScale[unit];
export function timeUnit(values) {
  const max = Math.max(...values);
  return max >= 1e6 ? 'ms' : max >= 1e3 ? 'µs' : 'ns';
}
export function timingMetric(row) {
  const denominator = row.time_denominator ?? (row.operation === 'Eval' ? 'query' : 'key');
  const count = denominator === 'query' ? row.query_count ?? row.num_keys ?? 1 : row.key_count ?? row.num_keys ?? 1;
  const ns = row.ns_per_unit ?? (row.median_ns !== undefined ? row.median_ns / count : row.ns_per_key);
  if (!(ns > 0) || !Number.isFinite(ns)) throw new Error(`invalid timing for ${row.label ?? row.variant}`);
  return {ns, denominator, count};
}
export function variantQualifiers(row) {
  const value = row.variant ?? '';
  const tags = [];
  const keyGeneration = ['Gen', 'gen'].includes(row.operation) && row.library === 'fss' && ['VDPF', 'VDMPF'].includes(row.scheme);
  if (value.includes('-packedkey-setup')) tags.push('prepare only');
  else if (value.includes('-packedkey-oneshot')) tags.push('pack + query');
  else if (value.includes('-packedkey')) tags.push('prepared query');
  else if (!keyGeneration && !/(merkle|chains)/.test(value)) tags.push('direct');
  if (!keyGeneration && /-merkle(?:-fold)?(?:-|$)/.test(value)) tags.push('Merkle');
  if (!keyGeneration && /-chains(?:-|$)/.test(value)) tags.push('chain');
  if (value.includes('-rawprp')) tags.push('raw PRP');
  else if (row.library === 'fss' && ['DMPF', 'VDMPF'].includes(row.scheme) && row.platform === 'cpu') tags.push('OpenSSL PRP');
  if (value.includes('-simd8')) tags.push('SIMD8');
  return tags;
}
export function outputQualifier(row) {
  const bits = row.logical_output_bits;
  return {packed_128_binary_leaf: '1-bit packed', packed_bits: `${bits}-bit packed`,
    packed_lanes: `${bits}-bit packed`, bool_scalar: '1-bit bool', gmp_scalar: `${bits}-bit field`,
    field128x2_plus_counter: '2×Field128 + counter', u64_mod_p_scalar: 'u64 mod 2^63−25'}[row.output_storage] ??
    {bytes: `XOR${bits ?? ''}`, xor_128: 'XOR128', uint: `mod 2^${bits}`,
      'additive_mod_2^64': 'mod 2^64', 'additive_mod_2^128': 'mod 2^128',
      additive_mod_2_64: 'mod 2^64', additive_mod_2_128: 'mod 2^128'}[row.group] ??
    (bits ? `${bits}-bit output` : row.group);
}
export function methodId(row) {
  return [row.library, row.scheme, row.variant ?? row.prg ?? 'native', row.group ?? '',
    row.logical_output_bits ?? '', row.output_storage ?? '', row.num_points ?? '', row.platform ?? ''].join(':');
}
export function describeMethod(row) {
  const name = `${LIBRARIES[row.library] ?? row.library} ${SCHEMES[row.scheme] ?? row.scheme}${row.scheme === 'PackedHalfTreeDPF' ? ' (exp.)' : ''}`;
  const config = [outputQualifier(row), row.prg, ...variantQualifiers(row),
    row.num_points ? `t=${row.num_points}` : null].filter(Boolean).join(', ');
  return {name, config, label: `${name} (${config})`};
}
export function buildMethodStyles(rows) {
  const result = new Map();
  for (const row of rows) {
    const color = row.library === 'fss' ? COLORS[row.scheme] : BASELINES[row.library] ?? '#555';
    const variant = row.variant ?? '';
    let index = variant.includes('-packedkey-setup') ? 3 : variant.includes('-packedkey-oneshot') ? 2 :
      variant.includes('-packedkey') || /-merkle(?:-fold)?(?:-|$)/.test(variant) ? 1 : 0;
    if (variant.includes('-rawprp')) index += 4;
    if (row.group === 'uint') index += 8;
    if (row.prg === 'AES-software') index += 12;
    if (row.prg === 'OpenSSL-AES') index += 16;
    if (variant.includes('-simd8')) index += 20;
    if (index >= 36) throw new Error(`unsupported method style combination: ${variant}`);
    // Styles depend on the method, so adding a series or exporting a subset
    // does not change existing curves. Panel checks catch unsupported aliases.
    // Rotate HalfTree's marker assignment so its default is a square beside
    // DPF's circle. The permutation preserves all six fallback marker slots.
    const markerOffset = row.library === 'fss' && row.scheme === 'HalfTreeDPF' ? 1 : 0;
    const marker = (Math.floor(index / DASHES.length) + markerOffset) % 6;
    result.set(methodId(row), {color, dash: DASHES[index % DASHES.length], marker});
  }
  return result;
}
export function assertPanelIdentities(rows) {
  const methods = [...new Map(rows.map(row => [row.method, row])).values()];
  for (const [name, values] of [['labels', methods.map(row => row.methodLabel)],
    ['styles', methods.map(row => JSON.stringify(row.style))]]) {
    if (new Set(values).size !== values.length) throw new Error(`duplicate panel ${name}`);
  }
}

export function buildRowLegend(panels) {
  // A row may use one visual style for different operation-specific shapes
  // or proof choices. Describe those operations in one legend entry without
  // combining their curves, timing records, or source identities.
  const groups = new Map();
  const operationNames = {gen: 'Gen', prepare: 'Prepare', eval: 'Eval', eval_all: 'EvalAll'};
  for (const panel of panels) for (const row of panel.rows) {
    const construction = [row.library, row.scheme, row.prg, row.group, row.logical_output_bits, row.num_points];
    const key = JSON.stringify([construction, row.style]);
    if (!groups.has(key)) groups.set(key, {row, methods: new Set(), descriptions: new Map()});
    const group = groups.get(key);
    group.methods.add(row.method);
    const description = row.compactConfig;
    if (!group.descriptions.has(description)) group.descriptions.set(description, new Set());
    group.descriptions.get(description).add(operationNames[panel.operation] ?? panel.operation);
  }
  const entries = [...groups.values()].map(group => ({row: group.row, methods: [...group.methods],
    label: group.descriptions.size === 1 ? group.row.methodLabel : `${group.row.legendName} [${[...group.descriptions]
      .map(([description, operations]) => `${[...operations].join('/')}: ${description}`).join('; ')}]`}));
  for (const [name, values] of [['labels', entries.map(entry => entry.label)],
    ['styles', entries.map(entry => JSON.stringify(entry.row.style))]]) {
    if (new Set(values).size !== values.length) throw new Error(`duplicate row legend ${name}`);
  }
  return entries;
}
