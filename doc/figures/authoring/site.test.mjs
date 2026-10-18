// Render the real static page and check unit conversion against a known receipt.
import assert from 'node:assert/strict';
import {readFile, mkdir} from 'node:fs/promises';
import {createServer} from 'node:http';
import {createRequire} from 'node:module';
import path from 'node:path';
import {figureGroups, normalizeRecords} from './figure-model.mjs';
const root = path.resolve(import.meta.dirname, '../../..');
const runtime = process.env.FSS_FIGURE_NODE_MODULES ?? '/Users/myl/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const require = createRequire(path.join(runtime, 'package.json'));
process.env.PLAYWRIGHT_BROWSERS_PATH ??= path.join(root, 'build/figure-tools/browsers');
const {chromium} = require('playwright');
const actual = JSON.parse(await readFile(process.argv[2] ?? path.join(root, 'build/figures/site-data.json'), 'utf8'));
const server = createServer(async (request, response) => {
  const name = request.url.split('?')[0] === '/' ? 'index.html' : request.url.slice(1).split('?')[0];
  try {
    if (name === 'data.json') {
      response.setHeader('Content-Type', 'application/json');
      response.end(JSON.stringify(actual));
    } else {
      response.setHeader('Content-Type', /\.m?js$/.test(name) ? 'text/javascript' : name.endsWith('.css') ? 'text/css' : 'text/html');
      response.end(await readFile(path.join(root, 'doc/site', name)));
    }
  } catch { response.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({headless: true});
try {
  const page = await browser.newPage({viewport: {width: 1100, height: 1150}, deviceScaleFactor: 1});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(url);
  await page.waitForFunction(() => document.querySelectorAll('#select-a option').length > 1);
  const labels = await page.locator('#select-a option').allTextContents();
  assert.equal(new Set(labels).size, labels.length, 'selector labels must identify every variant');
  await mkdir(path.join(root, 'build/figures'), {recursive: true});
  await page.screenshot({path: path.join(root, 'build/figures/site-comparison.png'), fullPage: true});
  assert.deepEqual(errors, []);
  const fixture = {library:'fss', scheme:'DPF', variant:'DPF-bytes-AES-NI', platform:'cpu', operation:'EvalAll',
    domain_bits:8, domain_size:256, num_keys:1, key_count:1, domain_output_count:256, time_denominator:'key',
    group:'bytes', prg:'AES-NI', logical_output_bits:127, ns_per_unit:1000000};
  await page.route('**/data.json', route => route.fulfill({contentType:'application/json', body:JSON.stringify({runs:[], records:[fixture,
    {...fixture, library:'google_dpf', variant:'DPF-xor_128-AES128', prg:'AES128'}]})}));
  await page.goto(url);
  await page.waitForFunction(() => document.querySelectorAll('#values-table tbody tr').length === 1);
  assert.match(await page.locator('#values-table tbody').textContent(), /1\s*ms/);
  assert.match(await page.locator('#panels').textContent(), /Time \(ms\/key\)/);
  const ticks = await page.locator('#panels svg text').allTextContents();
  assert(ticks.includes('1'), '1,000,000 ns must appear as the 1 ms axis tick');
  assert(!ticks.includes('1M'), 'raw ns ticks must not be labeled ms');
  assert.deepEqual(errors, []);
  const gapRecords = [fixture, {...fixture, library:'google_dpf', variant:'DPF-xor_128-AES128', prg:'AES128'}]
    .flatMap(row => [row, {...row, domain_bits:12, domain_size:4096}])
    .map(row => ({...row, status:'ok', expected_domain_bits:[8,10,12]}));
  await page.route('**/data.json', route => route.fulfill({contentType:'application/json', body:JSON.stringify({runs:[], records:gapRecords})}));
  await page.goto(url);
  await page.waitForFunction(() => document.querySelectorAll('#values-table tbody tr').length === 3);
  const gapRow = page.locator('#values-table tbody tr').nth(1);
  assert.equal(await gapRow.locator('td.missing').count(), 2, 'an entirely missing N must remain visible in the table');
  const segments = await page.locator('#panels polyline').evaluateAll(nodes => nodes.map(node => node.getAttribute('points').trim().split(/\s+/).filter(Boolean).length));
  assert(segments.length === 4 && segments.every(count => count === 1), 'site curves must not bridge an entirely missing N');
  const renderSource = (await readFile(path.join(root,'doc/figures/authoring/render.mjs'),'utf8')).replace('export function renderFigures', 'function renderFigures');
  const d3 = await readFile(path.join(root,'build/figure-tools/node_modules/d3/dist/d3.min.js'),'utf8');
  const figures = figureGroups(normalizeRecords({records:[...gapRecords, {...fixture, domain_bits:10, status:'known_failure'}]}));
  await page.setContent(`<main id="chart"></main><script>${d3}</script><script>${renderSource};renderFigures(d3,${JSON.stringify(figures)},{});</script>`);
  const paths = await page.locator('[data-layer="curve"]').evaluateAll(nodes => nodes.map(node => node.getAttribute('d')));
  assert(paths.length > 0 && paths.every(path => (path.match(/M/g) ?? []).length === 2 && !path.includes('L')), 'static curves must not bridge an entirely missing N');
  console.log(`site selector identities and 1,000,000 ns = 1 ms table/axis and whole-panel missing-N gap checks passed (${labels.length} CPU variants)`);
} finally {
  await browser.close();
  server.close();
}
