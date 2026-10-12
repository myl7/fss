import assert from 'node:assert/strict';
import {figureGroups, normalizeRecords} from './figure-model.mjs';

const measured = {platform: 'gpu', scheme: 'DPF', operation: 'Eval', library: 'fss',
  label: 'FSS', domain_bits: 8, num_keys: 4, median_ns: 2000000, ns_per_key: 500000, status: 'ok'};
assert.equal(normalizeRecords({records: [measured]})[0].ms_per_key, 0.5);
assert.throws(() => normalizeRecords({records: [{...measured, ns_per_key: 0}]}), /invalid timing/);
assert.throws(() => normalizeRecords({records: [{...measured, ns_per_key: undefined}]}), /invalid timing/);
assert.deepEqual(figureGroups([]), []);
assert.equal(figureGroups(normalizeRecords({records: [measured]}))[0].id, 'gpu-operations');
assert.equal(figureGroups(normalizeRecords({records: [{...measured, scan: 'threads', threads_per_block: 32}]}))[0].id, 'gpu-block-size');

const cpuRow = (scheme, operation, library = 'fss') => ({...measured, platform: 'cpu', scheme, operation, library});
const records = normalizeRecords({records: [
  cpuRow('DPF', 'EvalAll'), cpuRow('HalfTreeDPF', 'EvalAll'), cpuRow('PackedHalfTreeDPF', 'EvalAll'),
  cpuRow('GrottoDCF', 'EvalAll'), cpuRow('DMPF', 'EvalAll'), cpuRow('VDMPF', 'Gen'),
  cpuRow('VDPF', 'EvalAll'), cpuRow('VDPF', 'EvalAll', 'servan_vdpf')]});
const figures = Object.fromEntries(figureGroups(records).map(figure => [figure.id, figure]));
// Every scheme family shares one figure row: VDPF joins the DPF row with both
// the FSS and the Servan-Schreiber reference curve, Grotto DCF joins the DCF
// row, and the DMPF family gets its own row. Grid cells without measurements
// stay absent while the remaining rows and columns keep their coordinates.
const operations = figures['cpu-operations'];
assert.deepEqual(operations.panels.map(panel => `${panel.familyLabel}:${panel.operation}:${panel.row}:${panel.column}`).sort(), [
  'DCF:eval_all:0:2',
  'DMPF / VDMPF:eval_all:2:2',
  'DMPF / VDMPF:gen:2:0',
  'DPF / VDPF:eval_all:1:2',
]);
assert.equal(operations.panels.length, 4);
const dpfPanel = operations.panels.find(panel => panel.family === 'dpf' && panel.operation === 'eval_all');
assert.deepEqual(dpfPanel.rows.map(row => row.scheme).sort(),
  ['DPF', 'HalfTreeDPF', 'PackedHalfTreeDPF', 'VDPF', 'VDPF']);
assert.deepEqual(dpfPanel.rows.map(row => row.library).sort(),
  ['fss', 'fss', 'fss', 'fss', 'servan_vdpf']);
const dcfPanel = operations.panels.find(panel => panel.family === 'dcf');
assert.deepEqual(dcfPanel.rows.map(row => row.scheme), ['GrottoDCF']);
console.log('figure model checks passed');
