import assert from 'node:assert/strict';
import {expectedCoordinates, figureGroups, methodInventory, normalizeRecords, pairedSpeedups} from './figure-model.mjs';
import {assertPanelIdentities, buildMethodStyles, buildRowLegend, describeMethod, methodId, scaleTime, timeUnit, timingMetric, variantQualifiers} from '../../site/method-model.mjs';
const measured = {platform: 'gpu', scheme: 'DPF', operation: 'Eval', library: 'fss',
  variant: 'DPF-bytes-ChaCha20-x2', group: 'bytes', prg: 'ChaCha20-x2',
  domain_bits: 8, num_keys: 4, median_ns: 2000000, ns_per_key: 500000, status: 'ok'};
assert.equal(normalizeRecords({records: [measured]})[0].ms_per_unit, 0.5);
assert.throws(() => normalizeRecords({records: [{...measured, median_ns: 0, ns_per_key: 0}]}), /invalid timing/);
assert.deepEqual(figureGroups([]), []);
assert.deepEqual(buildMethodStyles([measured]).get(methodId(measured)), buildMethodStyles([measured, {...measured, variant: 'DPF-bytes-AES-NI-packedkey'}]).get(methodId(measured)));
assert.equal(timingMetric({...measured, key_count: 1, query_count: 4, time_denominator: 'query'}).ns, 500000);
for (const [value, unit, expected] of [[999, 'ns', 999], [1000, 'µs', 1], [999999, 'µs', 999.999], [1000000, 'ms', 1]]) {
  assert.equal(timeUnit([value]), unit);
  assert.equal(scaleTime(value, unit), expected);
}
assert.deepEqual(variantQualifiers({...measured, variant: 'VDMPF-bytes-AES-NI-chains-rawprp'}), ['chain', 'raw PRP']);
const cpuRow = (scheme, operation, library = 'fss', suffix = '') => ({...measured, platform: 'cpu', scheme,
  operation, library, prg: 'AES-NI', variant: `${scheme}-bytes-AES-NI${suffix}`});
const records = normalizeRecords({records: [cpuRow('DCF', 'EvalAll'), cpuRow('GrottoDCF', 'EvalAll'),
  cpuRow('DPF', 'EvalAll'), cpuRow('HalfTreeDPF', 'EvalAll'), cpuRow('PackedHalfTreeDPF', 'EvalAll'),
  cpuRow('DPF', 'EvalAll', 'libdpf'), cpuRow('DMPF', 'EvalAll'), cpuRow('VDMPF', 'Gen'),
  cpuRow('VDPF', 'EvalAll', 'fss', '-chains'), cpuRow('VDPF', 'EvalAll', 'fss', '-merkle'),
  cpuRow('VDPF', 'EvalAll', 'servan_vdpf'), cpuRow('DPF', 'Eval'),
  cpuRow('DPF', 'Eval', 'fss', '-packedkey'), cpuRow('DPF', 'Eval', 'fss', '-packedkey-oneshot')]});
const figures = figureGroups(records), overview = figures.find(figure => figure.id === 'cpu-operations');
assert.deepEqual([...new Set(overview.panels.map(panel => panel.family))], ['dcf', 'dpf', 'packed', 'vdpf', 'dmpf']);
assert.deepEqual(overview.panels.find(panel => panel.family === 'dpf' && panel.operation === 'eval_all').rows.map(row => row.scheme), ['DPF', 'HalfTreeDPF']);
assert.equal(overview.panels.find(panel => panel.family === 'vdpf').rows.length, 2);
assert.equal(figures.find(figure => figure.id === 'cpu-prepared').panels[0].rows.length, 3);
assert.equal(figures.find(figure => figure.id === 'cpu-proof').panels[0].rows.length, 2);
for (const figure of figures) for (const panel of figure.panels) assertPanelIdentities(panel.rows);
const inventory = methodInventory(records, figures);
assert(inventory.every(method => method.figures.length > 0));
assert.equal(new Set(records.filter(row => row.operation === 'eval').map(row => describeMethod({...row, operation: 'Eval'}).label)).size, 3);
assert.throws(() => assertPanelIdentities([{method:'a',methodLabel:'same',style:{}},{method:'b',methodLabel:'same',style:{}}]), /duplicate panel labels/);
assert.throws(() => assertPanelIdentities([{method:'a',methodLabel:'a',style:{}},{method:'b',methodLabel:'b',style:{}}]), /duplicate panel styles/);
const pair = normalizeRecords([cpuRow('DPF', 'Eval'), cpuRow('DPF', 'Eval', 'fss', '-packedkey')].map(row => ({...row, source_sha256: {header: 'same'}, raw_result: 'same-receipt.json', raw_result_sha256: 'same-receipt-hash'})));
assert.equal(pairedSpeedups(pair).length, 1);
assert.equal(pairedSpeedups(pair.map((row, index) => index ? {...row, source_sha256: {header: 'different'}} : row)).length, 0);
assert.equal(pairedSpeedups(pair.map((row, index) => index ? {...row, raw_result: 'another-host/receipt.json'} : row)).length, 0);
const sharedProof = figureGroups(normalizeRecords([
  cpuRow('VDPF', 'Gen'), cpuRow('VDPF', 'Eval', 'fss', '-chains'), cpuRow('VDPF', 'EvalAll', 'fss', '-chains'),
])).find(figure => figure.id === 'cpu-operations');
const proofLegend = sharedProof.rowLegends[0].entries;
assert.equal(proofLegend.length, 1);
assert.equal(proofLegend[0].methods.length, 2);
assert.match(proofLegend[0].label, /Gen:.*XOR|Gen:.*bytes/);
assert.match(proofLegend[0].label, /Eval\/EvalAll:.*chain/);
assert(!proofLegend[0].label.includes('direct'));
const ezpc = normalizeRecords([
  {...measured, library:'ezpc', group:'native', logical_output_bits:1, output_storage:'materialized', prg:'AES128', operation:'Gen'},
  {...measured, library:'ezpc', group:'native', logical_output_bits:1, output_storage:'materialized', prg:'AES128', operation:'Eval'},
  {...measured, library:'ezpc', group:'native', logical_output_bits:1, output_storage:'packed_bits', prg:'AES128', operation:'EvalAll'},
]);
const ezpcLegend = buildRowLegend(['gen','eval','eval_all'].map(operation => ({operation, rows:ezpc.filter(row=>row.operation===operation)})));
assert.equal(ezpcLegend.length, 1);
assert.match(ezpcLegend[0].label, /Gen\/Eval:.*1-bit output/);
assert.match(ezpcLegend[0].label, /EvalAll:.*1-bit packed/);
assert.throws(() => buildRowLegend([{operation:'eval', rows:[
  {...ezpc[0], library:'one'}, {...ezpc[0], library:'another', method:'another'},
]}]), /duplicate row legend/);
const threadRows = normalizeRecords([{...measured, scan:'threads', threads_per_block:32},
  {...measured, scheme:'DMPF', variant:'DMPF-bytes-ChaCha20-x2', scan:'threads', threads_per_block:256, block_size_tunable:false}]);
assert.equal(figureGroups(threadRows)[0].panels[0].rows.length, 1);
const domainAndThreads = normalizeRecords([measured, {...measured, scan:'threads', threads_per_block:32}]);
assert(methodInventory(domainAndThreads, figureGroups(domainAndThreads))[0].comparison_site);
const overlapRows = ['DPF','HalfTreeDPF'].flatMap(scheme => ['Gen','Eval','EvalAll'].map(operation => cpuRow(scheme,operation)));
const overlapStyles = buildMethodStyles(overlapRows);
for (const row of overlapRows) {
  assert.equal(overlapStyles.get(methodId(row)).marker, row.scheme === 'HalfTreeDPF' ? 1 : 0);
  assert.equal(overlapStyles.get(methodId(row)).dash, '');
}
const halfTreeFallbacks = [cpuRow('HalfTreeDPF','Eval'), cpuRow('HalfTreeDPF','Eval','fss','-packedkey-oneshot-rawprp'),
  {...cpuRow('HalfTreeDPF','Eval'),group:'uint'}, {...cpuRow('HalfTreeDPF','Eval'),prg:'AES-software',variant:'HalfTreeDPF-bytes-AES-software'},
  {...cpuRow('HalfTreeDPF','Eval'),prg:'OpenSSL-AES',variant:'HalfTreeDPF-bytes-OpenSSL-AES'}, cpuRow('HalfTreeDPF','Eval','fss','-simd8'),
  {...cpuRow('HalfTreeDPF','Eval','fss','-packedkey-oneshot-rawprp-simd8'),group:'uint'}];
const fallbackStyles = [...buildMethodStyles(halfTreeFallbacks).values()];
assert.equal(fallbackStyles.length, halfTreeFallbacks.length);
assert(fallbackStyles.every(style => Number.isInteger(style.marker) && style.marker >= 0 && style.marker < 6));
assert.equal(new Set(fallbackStyles.map(style => JSON.stringify(style))).size, fallbackStyles.length);
assert.equal(fallbackStyles.at(-1).marker, 0, 'the final HalfTree fallback slot must wrap to a supported marker');
const missingMiddle = {records: [measured, {...measured, domain_bits:10, status:'known_failure'}, {...measured, domain_bits:12}]};
assert.deepEqual(expectedCoordinates(missingMiddle).get('gpu|domain'), [8,10,12]);
assert(normalizeRecords(missingMiddle).every(row => row.expected_coordinates.join(',') === '8,10,12'));
assert.deepEqual(expectedCoordinates({records:[measured], metadata:{runs:[{metadata:{cases:[{platform:'gpu', domain_bits:10}]}}]}}).get('gpu|domain'), [8,10]);
console.log('figure identities, layout, denominator, pairing, and unit checks passed');
