import {assertPanelIdentities, buildMethodStyles, buildRowLegend, describeMethod, methodId, timingMetric} from '../../site/method-model.mjs';
const FAMILIES = [
  ['dcf', 'DCF', ['DCF', 'GrottoDCF']],
  ['dpf', 'DPF full width', ['DPF', 'HalfTreeDPF']],
  ['packed', 'Packed DPF', ['PackedHalfTreeDPF']],
  ['vdpf', 'VDPF', ['VDPF', 'VIDPF']],
  ['dmpf', 'DMPF / VDMPF', ['DMPF', 'VDMPF']],
];
const OPERATIONS = ['gen', 'eval', 'eval_all'];
const historical = row => ['fss_v060', 'fss_v070'].includes(row.library);
const prepared = row => /-packedkey/.test(row.variant);
const extra = row => /-rawprp|-simd8/.test(row.variant) || row.group === 'uint' ||
  row.prg === 'AES-software' || row.library === 'fss' && row.platform === 'cpu' && /-merkle/.test(row.variant);
// Include failed records and planned cases so a missing coordinate breaks a
// curve even when every method in its panel failed at that coordinate.
export function expectedCoordinates(input) {
  const coordinates = new Map();
  const add = row => {
    const scan = row.scan === 'threads' || row.sweep === 'block' ? 'threads' : 'domain';
    const key = `${row.platform}|${scan}`;
    const value = Number(scan === 'threads' ? row.requested_threads_per_block ?? row.threads_per_block : row.domain_bits);
    if (!(value > 0)) return;
    if (!coordinates.has(key)) coordinates.set(key, new Set());
    coordinates.get(key).add(value);
  };
  (Array.isArray(input) ? input : input.records ?? []).forEach(add);
  const walk = metadata => {
    (metadata?.cases ?? []).forEach(add);
    (metadata?.runs ?? []).forEach(run => walk(run.metadata));
  };
  if (!Array.isArray(input)) walk(input.metadata);
  return new Map([...coordinates].map(([key, values]) => [key, [...values].sort((a, b) => a - b)]));
}
export function normalizeRecords(input) {
  const records = Array.isArray(input) ? input : input.records;
  if (!Array.isArray(records)) throw new Error('missing benchmark records');
  const good = records.filter(row => row.status === 'ok' && row.metadata_status !== 'requires_recollection');
  const styles = buildMethodStyles(good);
  const coordinates = expectedCoordinates(input);
  return good.map(row => {
    const metric = timingMetric(row);
    const device = row.platform;
    const family = row.library === 'libdpf' || ['dpf-cpp', 'dpfplusplus'].includes(row.library) ? 'packed' :
      FAMILIES.find(([, , schemes]) => schemes.includes(row.scheme))?.[0];
    const operation = {Gen: 'gen', Eval: 'eval', EvalAll: 'eval_all', Prepare: 'prepare'}[row.operation];
    if (!['cpu', 'gpu'].includes(device)) throw new Error('invalid platform');
    if (!family) throw new Error('invalid scheme');
    if (!operation) throw new Error('invalid operation');
    const exponent = Number(row.domain_bits);
    if (!Number.isInteger(exponent) || exponent < 1) throw new Error('invalid log_n');
    const method = methodId(row), description = describeMethod(row);
    return {...row, method, methodLabel: description.label, legendName: description.name,
      compactConfig: description.config, style: styles.get(method), device, family, operation,
      log_n: exponent, expected_coordinates: coordinates.get(`${device}|${row.scan === 'threads' ? 'threads' : 'domain'}`),
      denominator: metric.denominator, count: metric.count,
      ns_per_unit: metric.ns, ms_per_unit: metric.ns / 1e6};
  });
}
function panelsFor(selected, operations, families = FAMILIES) {
  const panels = [];
  families.forEach(([family, familyLabel], row) => {
    operations.forEach((operation, column) => {
      const rows = selected.filter(item => item.family === family && item.operation === operation);
      if (rows.length) {
        assertPanelIdentities(rows);
        const denominators = new Set(rows.map(item => item.denominator));
        if (denominators.size !== 1) throw new Error('mixed time denominators');
        panels.push({family, familyLabel, operation, rows, row, column, denominator: rows[0].denominator});
      }
    });
  });
  return panels;
}
export function figureGroups(records) {
  const figures = [];
  for (const device of ['cpu', 'gpu']) {
    const selected = records.filter(row => row.device === device && row.scan !== 'threads' && !historical(row));
    const overview = selected.filter(row => !prepared(row) && !extra(row) && !['mastic', 'plasma', 'dpf-cpp', 'dpfplusplus'].includes(row.library));
    // The ordinary DPF reference is repeated in the packed row where measured.
    const packedReference = overview.filter(row => row.library === 'fss' && row.scheme === 'DPF' && overview.some(item => item.family === 'packed' && item.operation === row.operation)).map(row => ({...row, family: 'packed'}));
    figures.push({id: `${device}-operations`, title: `${device.toUpperCase()} benchmark overview`,
      panels: panelsFor([...overview, ...packedReference], OPERATIONS)});
    const supplements = [
      ['prepared', 'Prepared-key point evaluation', selected.filter(row => row.library === 'fss' && row.group !== 'uint' && ['DPF', 'HalfTreeDPF'].includes(row.scheme) && ['prepare', 'eval'].includes(row.operation))],
      ['proof', 'Proof accumulators', selected.filter(row => row.scheme === 'VDPF' && row.library === 'fss' && row.operation === 'eval_all')],
      ['prp', 'PRP backend comparison', selected.filter(row => ['DMPF', 'VDMPF'].includes(row.scheme))],
      ['output-groups', 'Output groups and PRG backends', selected.filter(row => ['DPF', 'DCF', 'HalfTreeDPF', 'VDPF'].includes(row.scheme) && row.library === 'fss' && !prepared(row))],
      ['additional', 'Additional native implementations', selected.filter(row => ['mastic', 'plasma', 'dpf-cpp', 'dpfplusplus'].includes(row.library) || row.library === 'fss' && !prepared(row) && !extra(row))],
    ];
    for (const [suffix, title, rows] of supplements) {
      const distinctVariants = new Set(rows.filter(row => row.library === 'fss').map(row => row.variant));
      if (suffix === 'prepared' && !rows.some(prepared) || suffix === 'prp' && !rows.some(row => /-rawprp/.test(row.variant)) ||
        suffix === 'additional' && !rows.some(row => row.library !== 'fss') ||
        suffix === 'output-groups' && !rows.some(row => row.group === 'uint' || row.prg === 'AES-software') ||
        suffix === 'proof' && !rows.some(row => /-chains/.test(row.variant)) || !distinctVariants.size) continue;
      figures.push({id: `${device}-${suffix}`, title: `${device.toUpperCase()}: ${title}`, panels: suffix === 'prepared' ? preparedPanels(rows) : panelsFor(rows, suffix === 'proof' ? ['eval_all'] : OPERATIONS)});
    }
    const speedup = pairedSpeedups(selected.filter(row => row.group !== 'uint'));
    if (speedup.length) figures.push({id: `${device}-prepared-speedup`, title: `${device.toUpperCase()}: prepared query speedup`,
      isSpeedup: true, panels: preparedPanels(speedup)});
  }
  const threads = records.filter(row => row.device === 'gpu' && row.scan === 'threads' && row.block_size_tunable !== false && !['DMPF', 'VDMPF'].includes(row.scheme));
  figures.push({id: 'gpu-block-size', title: 'GPU block-size sensitivity', isThreads: true, panels: panelsFor(threads, ['eval', 'eval_all'])});
  return figures.filter(figure => figure.panels.length).map(figure => {
    const rows = new Map();
    for (const panel of figure.panels) {
      const row = figure.isThreads ? 0 : panel.row;
      if (!rows.has(row)) rows.set(row, []);
      rows.get(row).push(panel);
    }
    return {...figure, rowLegends: [...rows].map(([rowIndex, panels]) => ({rowIndex, entries: buildRowLegend(panels)}))};
  });
}
function preparedPanels(rows) {
  return ['prepare', 'eval'].flatMap((operation, row) => ['DPF', 'HalfTreeDPF'].flatMap((scheme, column) => {
    const selected = rows.filter(item => item.scheme === scheme && item.operation === operation);
    if (!selected.length) return [];
    assertPanelIdentities(selected);
    return [{family: scheme, familyLabel: scheme === 'DPF' ? 'DPF' : 'HalfTree',
      operation, rows: selected, row, column, denominator: operation === 'prepare' ? 'key' : 'query'}];
  }));
}
export function pairedSpeedups(records) {
  const relevant = row => JSON.stringify([row.device, row.scheme, row.log_n, row.group, row.logical_output_bits,
    row.prg, row.key_count, row.query_count, row.time_denominator, row.raw_result, row.raw_result_sha256, row.source_sha256]);
  const direct = new Map(records.filter(row => row.library === 'fss' && row.operation === 'eval' && !prepared(row)).map(row => [relevant(row), row]));
  return records.filter(row => row.operation === 'eval' && prepared(row)).flatMap(row => {
    // Only pair when source hashes exist and the configuration matches. The
    // paired raw receipt remains explicit in exported provenance.
    const reference = direct.get(relevant(row));
    if (!reference || !row.raw_result || !row.raw_result_sha256 || !row.source_sha256 || !Object.keys(row.source_sha256).length) return [];
    return [{...row, speedup: reference.ns_per_unit / row.ns_per_unit,
      paired_reference: {raw_result: reference.raw_result, raw_result_sha256: reference.raw_result_sha256,
        ns_per_unit: reference.ns_per_unit, source_sha256: reference.source_sha256}}];
  });
}
export function methodInventory(records, figures) {
  return [...new Map(records.map(row => [row.method, row])).values()].map(row => ({
    method: row.method, label: row.methodLabel, platform: row.device,
    figures: figures.filter(figure => figure.panels.some(panel => panel.rows.some(item => item.method === row.method))).map(figure => figure.id),
    comparison_site: records.some(item => item.method === row.method && item.scan !== 'threads'),
  }));
}
