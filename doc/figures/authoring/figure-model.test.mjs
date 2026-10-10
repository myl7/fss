import assert from 'node:assert/strict';
import {figureGroups, normalizeRecords} from './figure-model.mjs';

const measured = {platform: 'gpu', scheme: 'DPF', operation: 'Eval', library: 'fss',
  label: 'FSS', domain_bits: 8, num_keys: 4, median_ns: 2000000, ns_per_key: 500000, status: 'ok'};
assert.equal(normalizeRecords({records: [measured]})[0].ms_per_key, 0.5);
assert.throws(() => normalizeRecords({records: [{...measured, ns_per_key: 0}]}), /invalid timing/);
assert.throws(() => normalizeRecords({records: [{...measured, ns_per_key: undefined}]}), /invalid timing/);
assert.deepEqual(figureGroups([]), []);
assert.equal(figureGroups(normalizeRecords({records: [measured]}))[0].panels.length, 1);
assert.equal(figureGroups(normalizeRecords({records: [{...measured, scan: 'threads', threads_per_block: 32}]}))[0].id, 'gpu-block-size');

const cpuRow = (scheme, operation, library = 'fss') => ({...measured, platform: 'cpu', scheme, operation, library});
const records = normalizeRecords({records: [
  cpuRow('DPF', 'EvalAll'), cpuRow('HalfTreeDPF', 'EvalAll'), cpuRow('PackedHalfTreeDPF', 'EvalAll'),
  cpuRow('GrottoDCF', 'EvalAll'), cpuRow('DMPF', 'EvalAll'), cpuRow('VDMPF', 'Gen'),
  cpuRow('VDPF', 'EvalAll', 'servan_vdpf')]});
const figures = Object.fromEntries(figureGroups(records).map(figure => [figure.id, figure]));
// Specialty schemes stay off the comparison figure and get dedicated figures.
assert.deepEqual(figures['cpu-eval-all'].panels.flatMap(panel => panel.rows).map(row => row.scheme).sort(),
  ['DPF', 'HalfTreeDPF', 'PackedHalfTreeDPF']);
const grotto = figures['cpu-grotto'].panels.flatMap(panel => panel.rows);
assert.deepEqual(grotto.map(row => row.primitive), ['dcf']);
const dmpf = figures['cpu-dmpf'].panels.flatMap(panel => panel.rows).map(row => `${row.primitive}:${row.scheme}`);
assert.deepEqual(dmpf, ['dmpf:VDMPF', 'dmpf:DMPF', 'dmpf:VDPF']);
console.log('figure model checks passed');
