// Browser-side D3 authoring. Exported SVGs contain no JavaScript dependencies.
export function renderFigures(d3, figures, metadata) {
  const symbols = [d3.symbolCircle, d3.symbolSquare, d3.symbolTriangle, d3.symbolDiamond, d3.symbolCross, d3.symbolStar];
  const methods = [...new Set(figures.flatMap(f => f.panels.flatMap(p => p.rows.map(r => r.method))))].sort();
  const methodIndex = method => methods.indexOf(method);
  const style = new Map(figures.flatMap(f => f.panels.flatMap(p => p.rows.map(r => [r.method, r.style]))));
  const color = method => style.get(method).color;
  const dash = method => style.get(method).dash;
  const symbol = method => symbols[style.get(method).marker];
  const W = 980, gap = 86, left = 70, right = 20, top = 40;
  for (const figure of figures) {
    const grid = !figure.isThreads;
    const cols = grid ? Math.max(...figure.panels.map(panel => panel.column)) + 1 : Math.min(2, figure.panels.length);
    const colGap = grid ? 34 : gap;
    const panelW = (W - left - right - colGap * (cols - 1)) / cols;
    const panelH = figure.panels.length === 1 ? 370 : grid ? 292 : 310;
    // One legend per grid row keeps each entry close to the curves it names;
    // the supplementary block-size figure keeps one legend below its panels.
    const rowIndices = [...new Set(figure.panels.map(panel => grid ? panel.row : 0))].sort((a, b) => a - b);
    const legendColumns = 3;
    const rowLayouts = rowIndices.map(rowIndex => {
      const panels = figure.panels.filter(panel => (grid ? panel.row : 0) === rowIndex);
      const entries = figure.rowLegends.find(legend => legend.rowIndex === rowIndex).entries
        .map(entry => ({...entry, lines: wrap(entry.label, (W - left - right) / legendColumns - 70)}))
        .sort((a, b) => methodIndex(a.row.method) - methodIndex(b.row.method));
      const slots = Array.from({length: legendColumns}, () => 0);
      for (const entry of entries) slots[slots.indexOf(Math.min(...slots))] += entry.lines.length * 19;
      const height = Math.max(...slots) + 14;
      return {rowIndex, panels, entries, height};
    });
    const legendStarts = [];
    let cursor = top;
    for (const rowLayout of rowLayouts) {
      const panelsTop = cursor;
      // The supplementary block-size figure wraps its panels across two-column
      // bands; the grid figures always place one band per family row.
      const bands = grid ? 1 : Math.ceil(rowLayout.panels.length / cols);
      cursor += panelH * bands + (bands - 1) * 18 + 6;
      legendStarts[rowLayout.rowIndex] = cursor;
      cursor += rowLayout.height + (grid ? 22 : 0);
    }
    const H = cursor + (figure.isThreads ? 74 : 52);
    const svg = d3.select('#chart').append('svg').attr('xmlns', 'http://www.w3.org/2000/svg')
      .attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`)
      .attr('data-figure', figure.id).attr('role', 'img').attr('aria-labelledby', `${figure.id}-title`)
      .style('font-family', 'Arial, Helvetica, sans-serif').style('font-size', '14px').style('background', '#fff');
    svg.append('title').attr('id', `${figure.id}-title`).text(figure.title);
    svg.append('rect').attr('data-layer', 'background').attr('width', W).attr('height', H).attr('fill', '#fff');
    svg.append('text').attr('data-name', 'figure-title').attr('x', left).attr('y', 22)
      .attr('font-size', 17).attr('font-weight', 600).attr('fill', '#202020').text(figure.title);
    // The leftmost existing panel of each grid row; rows can start past
    // column 0 when the family lacks that operation's implementation.
    const rowMinColumn = new Map();
    if (grid) for (const panel of figure.panels) {
      rowMinColumn.set(panel.row, Math.min(rowMinColumn.get(panel.row) ?? Infinity, panel.column));
    }
    const unitOf = panel => panel.operation === 'prepare' ? 'ns' : panel.operation !== 'eval_all' ? (figure.id.startsWith('gpu') ? 'ns' : 'µs') : 'ms';
    // The unit of the label each row/band carries, from its leftmost panel.
    const bandUnit = new Map();
    figure.panels.forEach((panel, index) => {
      const leftmost = grid ? panel.column === rowMinColumn.get(panel.row) : index % cols === 0;
      if (leftmost) bandUnit.set(grid ? panel.row : Math.floor(index / cols), unitOf(panel));
    });
    figure.panels.forEach((panel, index) => {
      const timeUnit = unitOf(panel);
      const timeMultiplier = {ns: 1e6, 'µs': 1e3, ms: 1}[timeUnit];
      const x0 = left + (grid ? panel.column : index % cols) * (panelW + colGap);
      const y0 = grid ? legendStarts[panel.row] - panelH - 6 : top + Math.floor(index / cols) * (panelH + 18);
      const plotH = panelH - 96;
      const g = svg.append('g').attr('data-panel', `${panel.family}-${panel.operation}`).attr('transform', `translate(${x0},${y0})`);
      const xValue = row => figure.isThreads ? Number(row.threads_per_block) : row.log_n;
      const values = [...new Set(panel.rows.flatMap(row => row.expected_coordinates ?? [xValue(row)]))];
      if (values.some(value => !(value > 0))) throw new Error('invalid x coordinate');
      const xExtent = figure.isThreads ? [16, 1024] : d3.extent(values);
      if (xExtent[0] === xExtent[1]) { xExtent[0] -= 0.5; xExtent[1] += 0.5; }
      const xPadding = figure.isThreads ? (Math.log2(xExtent[1]) - Math.log2(xExtent[0])) * 0.025 : (xExtent[1] - xExtent[0]) * 0.025;
      if (figure.isThreads) { xExtent[0] /= 2 ** xPadding; xExtent[1] *= 2 ** xPadding; }
      else { xExtent[0] -= xPadding; xExtent[1] += xPadding; }
      const x = (figure.isThreads ? d3.scaleLog().base(2) : d3.scaleLinear()).domain(xExtent).range([0, panelW]);
      const timeValue = row => figure.isSpeedup ? row.speedup : row.ms_per_unit * timeMultiplier;
      const yExtent = d3.extent(panel.rows, timeValue);
      if (figure.isSpeedup) { yExtent[0] = Math.min(yExtent[0], 1); yExtent[1] = Math.max(yExtent[1], 1); }
      // Pad in log coordinates so wide-range plots retain marker clearance.
      const logMin = Math.log10(yExtent[0]), logMax = Math.log10(yExtent[1]);
      const logSpan = Math.max(logMax - logMin, 0.15);
      const logPadding = Math.max(logSpan * 0.025, logSpan * 7 / (plotH - 14), 0.055);
      const lower = 10 ** (logMin - logPadding), upper = 10 ** (logMax + logPadding);
      const y = d3.scaleLog().domain([lower, upper]).range([plotH, 0]);
      // Log ticks: decade ticks on wide ranges, 1-2-5 steps in between, and
      // minor steps on narrow ranges, where nearly flat curves (a point query
      // that does not scale with N) would otherwise leave too few labeled
      // levels to see that the axis is logarithmic.
      const span = Math.log10(upper / lower);
      const factors = span > 2.2 ? [1] : span < 0.9 ? [1, 2, 3, 4, 5, 6, 8, 9] : [1, 2, 5];
      const candidates = [];
      for (let exponent = Math.floor(Math.log10(lower)); exponent <= Math.ceil(Math.log10(upper)); exponent++) {
        for (const factor of factors) {
          const value = factor * 10 ** exponent;
          if (value >= lower && value <= upper) candidates.push(value);
        }
      }
      const spacing = span < 0.9 ? 22 : 27;
      let yTicks = [];
      for (const value of candidates) {
        if (!yTicks.length || Math.abs(y(value) - y(yTicks[yTicks.length - 1])) >= spacing) yTicks.push(value);
      }
      if (yTicks.length < 3) yTicks = d3.ticks(lower, upper, 5);
      const tickValues = figure.isThreads ? [16, 32, 64, 128, 256, 512, 1024] : [...new Set(values)].sort((a, b) => a - b);
      g.append('g').attr('data-layer', 'grid').call(d3.axisLeft(y).tickValues(yTicks).tickSize(-panelW).tickFormat(''))
        .call(gridLayer => gridLayer.select('.domain').remove()).call(gridLayer => gridLayer.selectAll('line').attr('stroke', '#e3e5e8').attr('stroke-width', 0.6));
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
      // Only the leftmost existing panel of a grid row (or block-size band)
      // carries the y-axis label: the per-panel label would reach past the 34px
      // column gap into the neighboring plot, and a row whose first column is
      // empty (GPU DMPF has no Gen panel) starts further right. A panel whose
      // unit differs from the unit that label states notes its own unit in the
      // panel label instead.
      const leftmost = grid ? panel.column === rowMinColumn.get(panel.row) : index % cols === 0;
      const unitNote = figure.isSpeedup ? '' : !leftmost && (timeUnit !== bandUnit.get(grid ? panel.row : Math.floor(index / cols)) || panel.denominator !== 'key') ? ` (${timeUnit}/${panel.denominator})` : '';
      if (leftmost) {
        g.append('text').attr('data-name', 'y-label').attr('transform', `translate(-52,${plotH / 2}) rotate(-90)`)
          .attr('text-anchor', 'middle').text(figure.isSpeedup ? 'Direct / variant time' : `Time (${timeUnit}/${panel.denominator})`);
      }
      if (figure.isSpeedup) g.append('line').attr('data-name', 'direct-reference').attr('x2', panelW).attr('y1', y(1)).attr('y2', y(1)).attr('stroke', '#777').attr('stroke-dasharray', '3,3');
      const series = d3.group(panel.rows, row => row.method);
      for (const [method, points] of series) {
        points.sort((a, b) => xValue(a) - xValue(b));
        const layer = g.append('g').attr('data-series', method);
        layer.append('path').attr('data-layer', 'curve').attr('d', d3.line().defined(row => row !== undefined).x(row => x(xValue(row))).y(row => y(timeValue(row)))([...new Set(values)].sort((a, b) => a - b).map(value => points.find(row => xValue(row) === value))))
          .attr('fill', 'none').attr('stroke', color(method)).attr('stroke-dasharray', dash(method)).attr('stroke-width', /^fss:/.test(method) ? 2.6 : 1.8);
        layer.selectAll('path.point').data(points).join('path').attr('class', 'point').attr('data-point-id', row => xValue(row))
          .attr('transform', row => `translate(${x(xValue(row))},${y(timeValue(row))})`)
          .attr('d', d3.symbol().type(symbol(method)).size(45)).attr('fill', color(method)).attr('stroke', '#fff').attr('stroke-width', 0.6);
      }
      const operation = {gen: 'Gen', eval: 'Eval', eval_all: 'EvalAll', prepare: 'Prepare'}[panel.operation];
      g.append('text').attr('data-name', 'panel-label').attr('x', panelW / 2).attr('y', plotH + 69)
        .attr('text-anchor', 'middle').attr('font-size', 13).attr('font-weight', 600)
        .text(`(${String.fromCharCode(97 + index)}) ${panel.familyLabel} ${operation}${unitNote}`);
    });
    rowLayouts.forEach(rowLayout => {
      const legendTop = legendStarts[rowLayout.rowIndex];
      const columnWidth = (W - left - right) / legendColumns;
      // Each entry lands in the column whose next free slot is highest, so
      // multi-line entries never overlap the entries below them.
      const columnY = Array.from({length: legendColumns}, () => legendTop);
      rowLayout.entries.forEach(({row, label, lines, methods}) => {
        const column = columnY.indexOf(Math.min(...columnY));
        const y = columnY[column];
        columnY[column] = y + lines.length * 19;
        const legend = svg.append('g').attr('data-legend-item', label).attr('data-method', row.method).attr('data-methods', JSON.stringify(methods)).attr('transform', `translate(${left + column * columnWidth},${y + 4})`);
        legend.append('line').attr('x2', 48).attr('stroke', color(row.method)).attr('stroke-dasharray', dash(row.method)).attr('stroke-width', /^fss:/.test(row.method) ? 2.6 : 1.8);
        legend.append('path').attr('transform', 'translate(24,0)').attr('d', d3.symbol().type(symbol(row.method)).size(45)).attr('fill', color(row.method));
        const text = legend.append('text').attr('x', 60).attr('y', 4).attr('fill', '#333');
        lines.forEach((line, lineIndex) => text.append('tspan').attr('x', 60).attr('dy', lineIndex ? 19 : 0).text(line));
      });
    });
    if (figure.isThreads) svg.append('text').attr('data-name', 'unsupported-note').attr('x', left).attr('y', H - 51)
      .attr('font-size', 12).attr('fill', '#555').text('N=2²⁰. Point Eval uses independent keys. Fixed-256-thread multi-point kernels are excluded.');
    const defaultNote = figure.id === 'gpu-operations'
      ? 'Log time. Gen: ns/key. Eval: ns/query (independent keys or one-key batch). EvalAll: ms/key.'
      : figure.id.startsWith('gpu') ? 'Logarithmic time axis. CUDA-event-timed native GPU calls.'
        : 'Log time. Gen: µs/key. Prepare: ns/key. Eval: µs/query. EvalAll: ms/key. One CPU thread.';
    svg.append('text').attr('data-name', 'timing-note').attr('x', left).attr('y', H - 31).attr('font-size', 12).attr('fill', '#555')
      .text(figure.isSpeedup ? 'Paired source hashes and configuration match. Reference: direct Eval. Preparation is excluded from prepared query timing.' : metadata.figure_timing_note ?? defaultNote);
    svg.append('text').attr('data-name', 'methodology-note').attr('x', left).attr('y', H - 13).attr('font-size', 12).attr('fill', '#555')
      .text('Counts, native outputs, PRGs, and timing boundaries: doc/bench_figures.md. Variants: supplementary figures and comparison site.');
  }
}

function superscript(value) {
  return String(value).split('').map(char => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(char)] ?? char).join('');
}

function wrap(text, limit) {
  const lines = [''];
  const context = document.createElement('canvas').getContext('2d');
  context.font = '14px Arial';
  for (const word of text.split(/\s+/)) {
    const index = lines.length - 1;
    if (lines[index] && context.measureText(`${lines[index]} ${word}`).width > limit) lines.push(word);
    else lines[index] += `${lines[index] ? ' ' : ''}${word}`;
  }
  return lines;
}
