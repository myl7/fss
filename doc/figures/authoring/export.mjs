#!/usr/bin/env node
// Deterministic, offline D3 rendering with headless Chromium.
import {readFile, writeFile, mkdir, copyFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {createHash} from 'node:crypto';
import {normalizeRecords, figureGroups} from './figure-model.mjs';

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
      'num_keys', 'threads_per_block', 'scan', 'ns_per_key', 'logical_output_bits',
      'storage_output_bits', 'output_storage', 'group', 'prg', 'timing_boundary',
      'config', 'raw_result_sha256'];
    const runs = input.metadata?.runs ?? [{metadata: input.metadata}];
    const provenance = {
      schema_version: 1,
      source_sha256: dataSha256,
      sources: runs.map(run => {
        const metadata = run.metadata ?? {};
        return {sha256: run.sha256, date: metadata.started_utc,
          hardware: {host: metadata.host, cpu: metadata.cpu_model, gpu: metadata.gpu_model,
            governor: metadata.governor},
          git_revision: metadata.git_revision, runner_sha256: metadata.runner_sha256,
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
    if (publish) for (const extension of ['svg', 'png']) {
      await copyFile(path.join(outDir, `${figure.id}.${extension}`), path.join(root, 'doc/figures', `${figure.id}.${extension}`));
    }
    console.log(`${figure.id}: ${figure.panels.length} panels, ${figure.panels.reduce((sum, panel) => sum + panel.rows.length, 0)} measured points`);
  }
  await writeFile(path.join(outDir, 'manifest.json'), JSON.stringify({source: path.resolve(dataPath),
    records: records.length, figures: figures.map(figure => figure.id)}, null, 2) + '\n');
} finally {
  await browser.close();
}
