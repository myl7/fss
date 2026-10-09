// Browser-side D3 authoring. Exported SVGs contain no JavaScript dependencies.
export function renderFigures(d3, figures, metadata) {
  const colors = ['#bd7656', '#739b7a', '#947fa9', '#a39250', '#608b92', '#a8728a', '#747b83'];
  const fssColors = ['#1670b8', '#164a79', '#4392cc', '#276ca6', '#405ec4', '#2c649c'];
  const symbols = [d3.symbolCircle, d3.symbolSquare, d3.symbolTriangle, d3.symbolDiamond, d3.symbolCross, d3.symbolStar];
  const methods = [...new Set(figures.flatMap(f => f.panels.flatMap(p => p.rows.map(r => r.method))))].sort((a, b) => {
    const priority = value => /^fss:/.test(value) ? 0 : 1;
    return priority(a) - priority(b) || a.localeCompare(b);
  });
  const methodIndex = method => methods.indexOf(method);
  const ownMethods = methods.filter(method => /^fss:/.test(method));
  const baselineColors = {fss_v060: '#7d8796', fss_v070: '#a685b4', ezpc: '#c07852',
    gpu_dpf: '#729875', libdpf: '#ac9861', libfss: '#688e99', google_dpf: '#877a67'};
  const color = method => /^fss:/.test(method) ? fssColors[ownMethods.indexOf(method) % fssColors.length] :
    method.startsWith('libfss:additive') ? '#9b657d' : baselineColors[method.split(':')[0]] ?? colors[methodIndex(method) % colors.length];
  const symbol = method => symbols[methodIndex(method) % symbols.length];
  const W = 980, gap = 86, left = 77, right = 24, top = 38;
  for (const figure of figures) {
    const panelH = figure.panels.length === 1 ? 370 : 310;
    const cols = Math.min(2, figure.panels.length);
    const panelW = (W - left - right - gap * (cols - 1)) / cols;
    const rows = Math.ceil(figure.panels.length / cols);
    const legendRows = [...new Map(figure.panels.flatMap(p => p.rows).map(r => [`${r.method}:${configuration(r)}`, r])).values()];
    const legendEntries = legendRows.map(row => ({row, lines: wrap(`${legendName(row, figure.id)}  ·  ${configuration(row)}`, 108)}));
    const legendStart = top + rows * panelH + 10;
    const legendH = legendEntries.reduce((sum, entry) => sum + entry.lines.length * 19 + 8, 0) + (figure.isThreads ? 54 : 34);
    const H = legendStart + legendH;
    const svg = d3.select('#chart').append('svg').attr('xmlns', 'http://www.w3.org/2000/svg')
      .attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`)
      .attr('data-figure', figure.id).attr('role', 'img').attr('aria-labelledby', `${figure.id}-title`)
      .style('font-family', 'Arial, Helvetica, sans-serif').style('font-size', '14px').style('background', '#fff');
    svg.append('title').attr('id', `${figure.id}-title`).text(figure.title);
    svg.append('rect').attr('data-layer', 'background').attr('width', W).attr('height', H).attr('fill', '#fff');
    svg.append('text').attr('data-name', 'figure-title').attr('x', left).attr('y', 22)
      .attr('font-size', 17).attr('font-weight', 600).attr('fill', '#202020').text(figure.title);
    figure.panels.forEach((panel, index) => {
      const timeUnit = panel.operation !== 'eval_all' ? (figure.id.startsWith('gpu') ? 'ns' : 'µs') : 'ms';
      const timeMultiplier = {ns: 1e6, 'µs': 1e3, ms: 1}[timeUnit];
      const x0 = left + (index % cols) * (panelW + gap), y0 = top + Math.floor(index / cols) * panelH;
      const plotH = panelH - 94;
      const g = svg.append('g').attr('data-panel', `${panel.primitive}-${panel.operation}`).attr('transform', `translate(${x0},${y0})`);
      const xValue = row => figure.isThreads ? Number(row.threads_per_block) : row.log_n;
      const values = panel.rows.map(xValue);
      if (values.some(value => !(value > 0))) throw new Error('invalid x coordinate');
      const xExtent = figure.isThreads ? [16, 1024] : d3.extent(values);
      if (xExtent[0] === xExtent[1]) { xExtent[0] -= 0.5; xExtent[1] += 0.5; }
      const xPadding = figure.isThreads ? (Math.log2(xExtent[1]) - Math.log2(xExtent[0])) * 0.025 : (xExtent[1] - xExtent[0]) * 0.025;
      if (figure.isThreads) { xExtent[0] /= 2 ** xPadding; xExtent[1] *= 2 ** xPadding; }
      else { xExtent[0] -= xPadding; xExtent[1] += xPadding; }
      const x = (figure.isThreads ? d3.scaleLog().base(2) : d3.scaleLinear()).domain(xExtent).range([0, panelW]);
      const timeValue = row => row.ms_per_key * timeMultiplier;
      const yExtent = d3.extent(panel.rows, timeValue);
      // Pad in log coordinates so wide-range plots retain marker clearance.
      const logMin = Math.log10(yExtent[0]), logMax = Math.log10(yExtent[1]);
      const logSpan = Math.max(logMax - logMin, 0.15);
      const logPadding = Math.max(logSpan * 0.025, logSpan * 7 / (plotH - 14), 0.055);
      const lower = 10 ** (logMin - logPadding), upper = 10 ** (logMax + logPadding);
      const y = d3.scaleLog().domain([lower, upper]).range([plotH, 0]);
      const candidates = [];
      for (let exponent = Math.floor(Math.log10(lower)); exponent <= Math.ceil(Math.log10(upper)); exponent++) {
        for (const factor of Math.log10(upper / lower) > 2.2 ? [1] : [1, 2, 5]) {
          const value = factor * 10 ** exponent;
          if (value >= lower && value <= upper) candidates.push(value);
        }
      }
      if (Math.log10(upper / lower) < 0.8) candidates.splice(0, candidates.length, ...d3.ticks(lower, upper, 6));
      const yTicks = [];
      for (const value of candidates) {
        if (!yTicks.length || Math.abs(y(value) - y(yTicks[yTicks.length - 1])) >= 27) yTicks.push(value);
      }
      const tickValues = figure.isThreads ? [16, 32, 64, 128, 256, 512, 1024] : [...new Set(values)].sort((a, b) => a - b);
      g.append('g').attr('data-layer', 'grid').call(d3.axisLeft(y).tickValues(yTicks).tickSize(-panelW).tickFormat(''))
        .call(grid => grid.select('.domain').remove()).call(grid => grid.selectAll('line').attr('stroke', '#e3e5e8').attr('stroke-width', 0.6));
      g.append('rect').attr('data-layer', 'frame').attr('width', panelW).attr('height', plotH).attr('fill', 'none').attr('stroke', '#333').attr('stroke-width', 0.8);
      g.append('g').attr('data-axis', 'x').attr('transform', `translate(0,${plotH})`)
        .call(d3.axisBottom(x).tickValues(tickValues).tickFormat(value => figure.isThreads ? value : `2${superscript(value)}`).tickSizeOuter(0))
        .call(axis => axis.select('.domain').remove());
      g.append('g').attr('data-axis', 'y').call(d3.axisLeft(y).tickValues(yTicks).tickFormat(value =>
        value < 0.001 || value >= 10000 ? d3.format('.0e')(value).replace('e+', 'e') :
          value >= 1 ? d3.format('.4~g')(value) : d3.format('.2~g')(value)).tickSizeOuter(0))
        .call(axis => axis.select('.domain').remove());
      g.selectAll('[data-axis] text').attr('font-size', 13).attr('fill', '#333');
      g.append('text').attr('data-name', 'x-label').attr('x', panelW / 2).attr('y', plotH + 43).attr('text-anchor', 'middle')
        .text(figure.isThreads ? 'Threads per block, T' : 'Domain size, N');
      g.append('text').attr('data-name', 'y-label').attr('transform', `translate(-58,${plotH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle').text(`Time (${timeUnit}/key)`);
      const series = d3.group(panel.rows, row => row.method);
      for (const [method, points] of series) {
        points.sort((a, b) => xValue(a) - xValue(b));
        const layer = g.append('g').attr('data-series', method);
        layer.append('path').attr('data-layer', 'curve').attr('d', d3.line().x(row => x(xValue(row))).y(row => y(timeValue(row)))(points))
          .attr('fill', 'none').attr('stroke', color(method)).attr('stroke-width', /^fss:/.test(method) ? 2.6 : 1.8);
        layer.selectAll('path.point').data(points).join('path').attr('class', 'point').attr('data-point-id', row => xValue(row))
          .attr('transform', row => `translate(${x(xValue(row))},${y(timeValue(row))})`)
          .attr('d', d3.symbol().type(symbol(method)).size(45)).attr('fill', color(method)).attr('stroke', '#fff').attr('stroke-width', 0.6);
      }
      const operation = {gen: 'Gen', eval: 'Eval', eval_all: 'EvalAll'}[panel.operation];
      g.append('text').attr('data-name', 'panel-label').attr('x', panelW / 2).attr('y', plotH + 69)
        .attr('text-anchor', 'middle').attr('font-size', 15).attr('font-weight', 600)
        .text(`(${String.fromCharCode(97 + index)}) ${panel.primitive.toUpperCase()} ${operation}`);
    });
    let legendOffset = 0;
    legendEntries.forEach(({row, lines}) => {
      const y = legendStart + legendOffset;
      legendOffset += lines.length * 19 + 8;
      const legend = svg.append('g').attr('data-legend-item', row.label).attr('transform', `translate(${left},${y})`);
      legend.append('line').attr('x2', 25).attr('stroke', color(row.method)).attr('stroke-width', 2);
      legend.append('path').attr('transform', 'translate(12,0)').attr('d', d3.symbol().type(symbol(row.method)).size(45)).attr('fill', color(row.method));
      const text = legend.append('text').attr('x', 35).attr('y', 4).attr('fill', '#333');
      lines.forEach((line, index) => text.append('tspan').attr('x', 35).attr('dy', index ? 19 : 0).text(line));
    });
    if (figure.isThreads) svg.append('text').attr('data-name', 'unsupported-note').attr('x', left).attr('y', H - 33)
      .attr('font-size', 12).attr('fill', '#555').text('N=2²⁰. Point Eval is amortized over K=262,144 keys. Software AES: T=1024 unsupported.');
    svg.append('text').attr('data-name', 'timing-note').attr('x', left).attr('y', H - 13).attr('font-size', 12).attr('fill', '#555')
      .text(`Logarithmic time axis. ${figure.id === 'gpu-point' ? 'Amortized over K=262,144 keys. ' : ''}${metadata.figure_timing_note ?? (figure.id.startsWith('gpu') ? 'CUDA event timed native GPU calls.' : 'See benchmark methodology for timing boundaries.')}`);
  }
}

function superscript(value) {
  return String(value).split('').map(char => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(char)] ?? char).join('');
}

function configuration(row) {
  let output = [row.logical_output_bits ? `${row.logical_output_bits}-bit logical` : null,
    row.storage_output_bits ? `${row.storage_output_bits}-bit storage` : null].filter(Boolean).join(' / ');
  if (row.output_storage === 'packed_128_binary_leaf') output = '1-bit packed / 128-bit leaves';
  else if (row.output_storage === 'packed_bits') output = `${row.logical_output_bits}-bit packed output`;
  else if (row.library === 'gpu_dpf' && row.storage_output_bits === 128) output = '128-bit unsigned outputs';
  else if (row.output_storage === 'gmp_scalar') output = `${row.logical_output_bits}-bit prime-field / GMP scalar`;
  const group = {bytes: 'XOR', additive_mod_2_64: 'additive mod 2^64',
    'additive_mod_2^64': 'additive mod 2^64', prime_field: 'prime field',
    xor_128: 'XOR', 'additive_mod_2^128': 'additive mod 2^128'}[row.group] ?? row.group;
  const showGroup = !['gmp_scalar', 'packed_128_binary_leaf', 'packed_bits'].includes(row.output_storage) && row.library !== 'gpu_dpf';
  return [output, showGroup ? group : null, row.prg,
    row.keys > 1 ? `K=${row.keys.toLocaleString('en-US')}` : 'K=1'].filter(Boolean).join(' / ');
}

function legendName(row, figureId) {
  const library = {fss: 'FSS', fss_v070: 'FSS 0.7.0', fss_v060: 'FSS 0.6.0',
    gpu_dpf: 'GPU-DPF', ezpc: 'EzPC', libdpf: 'libdpf', google_dpf: 'Google DPF', libfss: 'libfss'}[row.library] ?? row.library;
  const scheme = ['gpu-eval-all', 'gpu-block-size'].includes(figureId) && row.library === 'fss' ? ` ${row.scheme}` : '';
  return `${library}${scheme}`;
}

function wrap(text, limit) {
  const lines = [''];
  for (const word of text.split(/\s+/)) {
    const index = lines.length - 1;
    if (lines[index] && lines[index].length + word.length + 1 > limit) lines.push(word);
    else lines[index] += `${lines[index] ? ' ' : ''}${word}`;
  }
  return lines;
}
