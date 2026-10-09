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
console.log('figure model checks passed');
