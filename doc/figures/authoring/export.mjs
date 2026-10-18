#!/usr/bin/env node
// Deterministic, offline D3 rendering with headless Chromium.
import {readFile, writeFile, mkdir, copyFile, readdir, unlink} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {createHash} from 'node:crypto';
import {expectedCoordinates, normalizeRecords, figureGroups, methodInventory} from './figure-model.mjs';

const sourceDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(sourceDir, '../../..');
const dataPath = process.argv[2];
if (!dataPath) throw new Error('usage: node doc/figures/authoring/export.mjs normalized.json [--publish]');
const publish = process.argv.includes('--publish');
const runtime = process.env.FSS_FIGURE_NODE_MODULES ?? '/Users/myl/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.PLAYWRIGHT_BROWSERS_PATH ??= path.join(root, 'build/figure-tools/browsers');
const require = createRequire(path.join(runtime, 'package.json'));
const {chromium} = require('playwright');
const d3Path = process.env.FSS_FIGURE_D3 ?? path.join(root, 'build/figure-tools/node_modules/d3/dist/d3.min.js');
const dataSource = await readFile(path.resolve(dataPath), 'utf8');
const input = JSON.parse(dataSource);
const dataSha256 = createHash('sha256').update(dataSource).digest('hex');

// The comparison site ships the same successful domain-sweep measurements as
// the figures, trimmed to the fields its tables show. Block-sweep records are
// excluded because the site's independent variable is always the domain size.
function siteData() {
  const fields = ['library', 'scheme', 'variant', 'label', 'platform', 'operation', 'domain_bits',
    'domain_size', 'num_keys', 'ns_per_key', 'median_ns', 'key_count', 'query_count', 'domain_output_count', 'ns_per_domain_output', 'legacy_quantity_fields', 'ns_per_unit', 'time_denominator', 'workload', 'proof_accumulator', 'proof_generation_included', 'setup_included', 'metadata_status', 'native_counters', 'requested_threads_per_block', 'launch_metadata_source', 'quantity_source', 'source_sha256', 'raw_result_sha256', 'config', 'group', 'prg', 'logical_output_bits', 'storage_output_bits',
    'output_storage', 'num_points', 'threads_per_block', 'timing_boundary', 'statistic'];
  const coordinates = expectedCoordinates(input);
  const records = input.records
    .filter(row => row.status === 'ok' && row.scan !== 'threads' && row.metadata_status !== 'requires_recollection')
    .map(row => ({...Object.fromEntries(fields.filter(field => row[field] !== undefined).map(field => [field, row[field]])),
      expected_domain_bits: coordinates.get(`${row.platform}|domain`)}));
  // Merged inputs nest run lists, so provenance needs the leaf run manifests.
  const leaves = [];
  const walk = metadata => {
    for (const run of metadata?.runs ?? []) {
      if (run.metadata?.runs) walk(run.metadata);
      else leaves.push(run.metadata ?? {});
    }
  };
  walk(input.metadata);
  if (!leaves.length && input.metadata?.started_utc) leaves.push(input.metadata);
  const line = (text, pattern) => text?.match(pattern)?.[1]?.trim() ?? null;
  const runs = leaves.map(metadata => ({date: metadata.started_utc, host: metadata.host,
    cpu: line(metadata.cpu_model, /^Model name:\s*(.+)$/m),
    gpu: line(metadata.gpu_model, /(NVIDIA [^|\n]+?)(?=\s*\.{3}|\s{2,}(?:On|Off)\b)/),
    governor: metadata.governor, git_revision: metadata.git_revision}));
  return {schema_version: 1, source_sha256: dataSha256, runs, records,
    post_timing_source_delta: input.metadata?.post_timing_source_delta,
    failure_metadata_correction: input.metadata?.failure_metadata_correction};
}
function leafRuns(metadata) {
  if (metadata?.runs) return metadata.runs.flatMap(run => run.metadata?.runs ? leafRuns(run.metadata) : [run]);
  return [{metadata}];
}
const records = normalizeRecords(input);
const figures = figureGroups(records);
if (!figures.length) throw new Error('no successful measurements to render');
const outDir = path.join(root, 'build/figures');
await mkdir(outDir, {recursive: true});
const d3Source = await readFile(d3Path, 'utf8');
const renderSource = (await readFile(path.join(sourceDir, 'render.mjs'), 'utf8')).replace('export function renderFigures', 'function renderFigures');
const json = JSON.stringify({figures, metadata: input.metadata ?? {}}).replaceAll('<', '\\u003c');
const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>FSS measured benchmarks</title>
<style>*{box-sizing:border-box}body{margin:0;background:white}svg{display:block;margin-bottom:20px}</style>
</head><body><main id="chart"></main><script>${d3Source}</script><script>${renderSource}
const payload=${json};renderFigures(d3,payload.figures,payload.metadata);window.figureReady=true;</script></body></html>`;
await writeFile(path.join(outDir, 'figures.html'), html);
const browser = await chromium.launch({headless: true});
try {
  const page = await browser.newPage({viewport: {width: 1000, height: 1000}, deviceScaleFactor: 2});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setContent(html, {waitUntil: 'load'});
  await page.waitForFunction(() => window.figureReady === true);
  await page.evaluate(() => document.fonts.ready);
  if (errors.length) throw new Error(errors.join('\n'));
  for (const figure of figures) {
    const locator = page.locator(`svg[data-figure="${figure.id}"]`);
    const fields = ['library', 'scheme', 'operation', 'variant', 'domain_bits', 'domain_size',
      'expected_coordinates', 'num_keys', 'median_ns', 'time_ns', 'key_count', 'query_count', 'domain_output_count', 'ns_per_domain_output', 'legacy_quantity_fields', 'ns_per_unit', 'time_denominator', 'workload', 'proof_accumulator', 'proof_generation_included', 'setup_included', 'requested_threads_per_block', 'launch_metadata_source', 'quantity_source', 'paired_reference', 'source_sha256', 'threads_per_block', 'scan', 'ns_per_key', 'logical_output_bits',
      'storage_output_bits', 'output_storage', 'group', 'prg', 'timing_boundary',
      'config', 'raw_result', 'benchmark', 'native_counters', 'metadata_status', 'raw_result_sha256'];
    const runs = leafRuns(input.metadata);
    const provenance = {
      schema_version: 1,
      source_sha256: dataSha256,
      post_timing_source_delta: input.metadata?.post_timing_source_delta,
      failure_metadata_correction: input.metadata?.failure_metadata_correction,
      sources: runs.map(run => {
        const metadata = run.metadata ?? {};
        return {source: run.source, sha256: run.sha256, date: metadata.started_utc,
          hardware: {host: metadata.host, cpu: metadata.cpu_model, gpu: metadata.gpu_model,
            governor: metadata.governor},
          git_revision: metadata.git_revision, git_status: metadata.git_status, runner_sha256: metadata.runner_sha256,
          source_sha256: metadata.source_sha256};
      }),
      measurements: figure.panels.flatMap(panel => panel.rows.map(row =>
        Object.fromEntries(fields.filter(field => row[field] !== undefined).map(field => [field, row[field]])))),
    };
    await locator.evaluate((element, metadata) => {
      const node = document.createElementNS('http://www.w3.org/2000/svg', 'metadata');
      node.setAttribute('data-name', 'benchmark-provenance');
      node.textContent = JSON.stringify(metadata);
      element.insertBefore(node, element.firstChild);
    }, provenance);
    const svg = await locator.evaluate(element => element.outerHTML);
    await writeFile(path.join(outDir, `${figure.id}.svg`), svg);
    await locator.screenshot({path: path.join(outDir, `${figure.id}.png`)});
    const overflow = await locator.evaluate(element => {
      return [...element.querySelectorAll('text')].filter(text => {
        const box = text.getBoundingClientRect(), canvas = element.getBoundingClientRect();
        return box.left < canvas.left - 1 || box.right > canvas.right + 1 || box.top < canvas.top - 1 || box.bottom > canvas.bottom + 1;
      }).map(text => text.textContent);
    });
    if (overflow.length) throw new Error(`text exceeds ${figure.id} canvas: ${overflow.join(', ')}`);
    const legendErrors = await locator.evaluate(element => {
      const errors = [];
      for (const legend of element.querySelectorAll('[data-legend-item]')) {
        const method = legend.getAttribute('data-method');
        const curve = [...element.querySelectorAll('[data-series]')].find(layer => layer.getAttribute('data-series') === method)?.querySelector('[data-layer="curve"]');
        const sample = legend.querySelector('line');
        if (!curve || !sample) { errors.push(`missing legend curve: ${method}`); continue; }
        for (const field of ['stroke', 'stroke-dasharray', 'stroke-width']) {
          if (sample.getAttribute(field) !== curve.getAttribute(field)) errors.push(`legend ${field} differs: ${method}`);
        }
        const period = (sample.getAttribute('stroke-dasharray') ?? '').split(',').reduce((sum, value) => sum + Number(value), 0);
        if (Number(sample.getAttribute('x2')) < period * 2) errors.push(`legend sample hides the dash pattern: ${method}`);
      }
      return errors;
    });
    if (legendErrors.length) throw new Error(legendErrors.join('\n'));
    if (publish) for (const extension of ['svg', 'png']) {
      await copyFile(path.join(outDir, `${figure.id}.${extension}`), path.join(root, 'doc/figures', `${figure.id}.${extension}`));
    }
    console.log(`${figure.id}: ${figure.panels.length} panels, ${figure.panels.reduce((sum, panel) => sum + panel.rows.length, 0)} measured points`);
  }
  await writeFile(path.join(outDir, 'manifest.json'), JSON.stringify({source: path.resolve(dataPath),
    source_sha256: dataSha256, post_timing_source_delta: input.metadata?.post_timing_source_delta,
    failure_metadata_correction: input.metadata?.failure_metadata_correction,
    records: records.length, figures: figures.map(figure => figure.id), methods: methodInventory(records, figures)}, null, 2) + '\n');
  const site = JSON.stringify(siteData()) + '\n';
  await writeFile(path.join(outDir, 'site-data.json'), site);
  if (publish) {
    await mkdir(path.join(root, 'doc/site'), {recursive: true});
    await writeFile(path.join(root, 'doc/site/data.json'), site);
    // Drop published figures that this export no longer produces.
    const keep = new Set(figures.flatMap(figure => ['svg', 'png'].map(extension => `${figure.id}.${extension}`)));
    for (const file of await readdir(path.join(root, 'doc/figures'))) {
      if (/\.(svg|png)$/.test(file) && !keep.has(file)) {
        await unlink(path.join(root, 'doc/figures', file));
      }
    }
    console.log(`site data: ${JSON.parse(site).records.length} domain-sweep records`);
  }
} finally {
  await browser.close();
}
