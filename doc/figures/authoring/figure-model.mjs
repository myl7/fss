// Normalize measured timing records without interpolating missing measurements.
const SCHEMES = ['DPF', 'DCF', 'HalfTreeDPF', 'PackedHalfTreeDPF', 'GrottoDCF', 'DMPF', 'VDMPF', 'VDPF', 'VIDPF'];
// Scheme families share one figure row: comparison DCFs together with Grotto
// DCF, point-function schemes together with their verifiable VDPF
// counterparts, and the multi-point DMPF family. Families without measured
// panels simply leave their grid cells empty.
const FAMILIES = [
  ['dcf', 'DCF', ['DCF', 'GrottoDCF']],
  ['dpf', 'DPF / VDPF', ['DPF', 'HalfTreeDPF', 'PackedHalfTreeDPF', 'VDPF', 'VIDPF']],
  ['dmpf', 'DMPF / VDMPF', ['DMPF', 'VDMPF']],
];
const OPERATIONS = ['gen', 'eval', 'eval_all'];

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
    const family = FAMILIES.find(([, , schemes]) => schemes.includes(row.scheme))?.[0];
    const operation = {Gen: 'gen', Eval: 'eval', EvalAll: 'eval_all'}[row.operation];
    if (!['cpu', 'gpu'].includes(device)) throw new Error('invalid platform');
    if (!family) throw new Error('invalid scheme');
    if (!operation) throw new Error('invalid operation');
    const exponent = Number(row.domain_bits);
    if (!Number.isInteger(exponent) || exponent < 1) throw new Error('invalid log_n');
    const variant = (row.variant ?? '').replace(/^(PackedHalfTreeDPF|HalfTreeDPF|GrottoDCF|VDMPF|VDPF|DMPF|DPF|DCF)[-_]?/, '');
    // The scheme suffix keeps series apart when two schemes share a variant
    // (DPF/DCF/VDPF bytes-AES-NI in the DPF and DCF rows, DMPF/VDMPF in the
    // multi-point row); otherwise the renderer groups them into one polyline.
    const suffix = {HalfTreeDPF: ':half-tree', PackedHalfTreeDPF: ':packed', DCF: ':dcf', VDPF: ':vdpf', VIDPF: ':vidpf',
      GrottoDCF: ':grotto', DMPF: ':dmpf', VDMPF: ':vdmpf'}[row.scheme] ?? '';
    const method = `${row.library}:${variant}${suffix}`;
    const label = row.label ?? `${row.library}${row.scheme === 'HalfTreeDPF' ? ' HalfTree' : ''}${row.variant ? ` (${row.variant})` : ''}`;
    return {...row, method, label, device, family, operation, log_n: exponent, keys, ms_per_key: time};
  });
}

export function figureGroups(records) {
  const specs = [
    ['cpu-operations', 'CPU benchmark operations', 'cpu', OPERATIONS],
    ['gpu-operations', 'GPU benchmark operations', 'gpu', OPERATIONS],
    ['gpu-block-size', 'GPU block-size sensitivity', 'gpu', ['eval', 'eval_all']],
  ];
  return specs.map(([id, title, device, operations]) => {
    const isThreads = id === 'gpu-block-size';
    const selected = records.filter(row => row.device === device && operations.includes(row.operation) &&
      (isThreads ? row.scan === 'threads' : row.scan !== 'threads'));
    const panels = [];
    if (isThreads) {
      for (const family of FAMILIES) for (const operation of operations) {
        const rows = selected.filter(row => row.family === family[0] && row.operation === operation);
        if (rows.length) panels.push({family: family[0], familyLabel: family[1], operation, rows});
      }
    } else {
      // One family per grid row and one operation per grid column, so the
      // Gen, Eval, and EvalAll columns stay aligned across families.
      FAMILIES.forEach(([family, familyLabel], row) => {
        operations.forEach((operation, column) => {
          // The historical FSS releases stay on the comparison site with full
          // provenance; the figures compare current implementations.
          const rows = selected.filter(row_ => row_.family === family && row_.operation === operation
            && !['fss_v060', 'fss_v070'].includes(row_.library));
          if (rows.length) panels.push({family, familyLabel, operation, rows, row, column});
        });
      });
    }
    return {id, title, isThreads, panels};
  }).filter(figure => figure.panels.length);
}
