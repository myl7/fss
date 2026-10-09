// Normalize measured timing records without interpolating missing measurements.
export function normalizeRecords(input) {
  const records = Array.isArray(input) ? input : input.records;
  if (!Array.isArray(records)) throw new Error('missing benchmark records');
  return records.filter(row => row.status === 'ok').map(row => {
    const keys = Number(row.num_keys ?? 1);
    const time = Number(row.ns_per_key) / 1e6;
    if (!(keys > 0) || !(time > 0) || !Number.isFinite(time)) {
      throw new Error(`invalid timing for ${row.label ?? row.method}`);
    }
    const device = row.platform;
    const primitive = row.scheme === 'DCF' ? 'dcf' : 'dpf';
    const operation = {Gen: 'gen', Eval: 'eval', EvalAll: 'eval_all'}[row.operation];
    if (!['cpu', 'gpu'].includes(device)) throw new Error('invalid platform');
    if (!['DPF', 'DCF', 'HalfTreeDPF'].includes(row.scheme)) throw new Error('invalid scheme');
    if (!operation) throw new Error('invalid operation');
    const exponent = Number(row.domain_bits);
    if (!Number.isInteger(exponent) || exponent < 1) throw new Error('invalid log_n');
    const variant = (row.variant ?? '').replace(/^(HalfTreeDPF|DPF|DCF)[-_]?/, '');
    const method = `${row.library}:${variant}${row.scheme === 'HalfTreeDPF' ? ':half-tree' : ''}`;
    const label = row.label ?? `${row.library}${row.scheme === 'HalfTreeDPF' ? ' HalfTree' : ''}${row.variant ? ` (${row.variant})` : ''}`;
    return {...row, method, label, device, primitive, operation, log_n: exponent, keys, ms_per_key: time};
  });
}

export function figureGroups(records) {
  const specs = [
    ['cpu-point', 'CPU point operations', 'cpu', ['gen', 'eval']],
    ['gpu-point', 'GPU point operations', 'gpu', ['gen', 'eval']],
    ['cpu-eval-all', 'CPU full-domain evaluation', 'cpu', ['eval_all']],
    ['gpu-eval-all', 'GPU full-domain evaluation', 'gpu', ['eval_all']],
    ['gpu-block-size', 'GPU block-size sensitivity', 'gpu', ['eval', 'eval_all']],
  ];
  return specs.map(([id, title, device, operations]) => {
    const isThreads = id === 'gpu-block-size';
    const selected = records.filter(row => row.device === device && operations.includes(row.operation) &&
      (isThreads ? row.scan === 'threads' : row.scan !== 'threads'));
    const panels = [];
    for (const primitive of ['dpf', 'dcf']) for (const operation of operations) {
      const rows = selected.filter(row => row.primitive === primitive && row.operation === operation);
      if (rows.length) panels.push({primitive, operation, rows});
    }
    return {id, title, isThreads, panels};
  }).filter(figure => figure.panels.length);
}
