// Normalize measured timing records without interpolating missing measurements.
const SCHEMES = ['DPF', 'DCF', 'HalfTreeDPF', 'PackedHalfTreeDPF', 'GrottoDCF', 'DMPF', 'VDMPF', 'VDPF'];
// Schemes shown on the library-comparison figures; specialty schemes get
// dedicated single-library figures because their functionality has no
// like-for-like third-party counterpart on the same axes. VDPF joins the
// comparison axes with both an FSS and the Servan-Schreiber reference curve.
const MAIN_SCHEMES = ['DPF', 'DCF', 'HalfTreeDPF', 'PackedHalfTreeDPF', 'VDPF'];

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
    const primitive = ['DCF', 'GrottoDCF'].includes(row.scheme) ? 'dcf' : ['DMPF', 'VDMPF'].includes(row.scheme) ? 'dmpf' : 'dpf';
    const operation = {Gen: 'gen', Eval: 'eval', EvalAll: 'eval_all'}[row.operation];
    if (!['cpu', 'gpu'].includes(device)) throw new Error('invalid platform');
    if (!SCHEMES.includes(row.scheme)) throw new Error('invalid scheme');
    if (!operation) throw new Error('invalid operation');
    const exponent = Number(row.domain_bits);
    if (!Number.isInteger(exponent) || exponent < 1) throw new Error('invalid log_n');
    const variant = (row.variant ?? '').replace(/^(PackedHalfTreeDPF|HalfTreeDPF|GrottoDCF|VDMPF|VDPF|DMPF|DPF|DCF)[-_]?/, '');
    // The scheme suffix keeps series apart when two schemes share a variant
    // (DPF/DCF/VDPF bytes-AES-NI on comparison figures, DMPF/VDMPF on the
    // shared specialty figure); otherwise the renderer groups them into one
    // polyline.
    const suffix = {HalfTreeDPF: ':half-tree', PackedHalfTreeDPF: ':packed', DCF: ':dcf', VDPF: ':vdpf',
      GrottoDCF: ':grotto', DMPF: ':dmpf', VDMPF: ':vdmpf'}[row.scheme] ?? '';
    const method = `${row.library}:${variant}${suffix}`;
    const label = row.label ?? `${row.library}${row.scheme === 'HalfTreeDPF' ? ' HalfTree' : ''}${row.variant ? ` (${row.variant})` : ''}`;
    return {...row, method, label, device, primitive, operation, log_n: exponent, keys, ms_per_key: time};
  });
}

export function figureGroups(records) {
  const specs = [
    ['cpu-point', 'CPU point operations', 'cpu', ['gen', 'eval'], MAIN_SCHEMES],
    ['gpu-point', 'GPU point operations', 'gpu', ['gen', 'eval'], MAIN_SCHEMES],
    ['cpu-eval-all', 'CPU full-domain evaluation', 'cpu', ['eval_all'], MAIN_SCHEMES],
    ['gpu-eval-all', 'GPU full-domain evaluation', 'gpu', ['eval_all'], MAIN_SCHEMES],
    ['gpu-block-size', 'GPU block-size sensitivity', 'gpu', ['eval', 'eval_all'], MAIN_SCHEMES],
    ['cpu-grotto', 'CPU Grotto DCF (1-bit comparison output)', 'cpu', ['gen', 'eval', 'eval_all'], ['GrottoDCF']],
    ['cpu-dmpf', 'CPU DMPF and VDMPF family', 'cpu', ['gen', 'eval', 'eval_all'], ['DMPF', 'VDMPF']],
  ];
  return specs.map(([id, title, device, operations, schemes]) => {
    const isThreads = id === 'gpu-block-size';
    const selected = records.filter(row => row.device === device && operations.includes(row.operation) &&
      schemes.includes(row.scheme) && (isThreads ? row.scan === 'threads' : row.scan !== 'threads'));
    const panels = [];
    for (const primitive of ['dpf', 'dcf', 'dmpf']) for (const operation of operations) {
      const rows = selected.filter(row => row.primitive === primitive && row.operation === operation);
      if (rows.length) panels.push({primitive, operation, rows});
    }
    return {id, title, isThreads, panels};
  }).filter(figure => figure.panels.length);
}
