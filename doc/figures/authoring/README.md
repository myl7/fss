# Benchmark figure authoring

The renderer consumes `chart-data.json` from `third_party/figure_bench.py`.
It draws successful positive measurements with resolved quantity metadata.
Missing measurements stay absent. The domain coordinates include all record
statuses and the planned cases in run manifests. Lines stop at a missing
coordinate even when every method in a panel lacks that measurement. The
comparison site retains the coordinate as an empty table row.

Install dependencies in the build directory:

```sh
npm install --prefix build/figure-tools --no-audit --no-fund d3@7.9.0 playwright@1.62.1
PLAYWRIGHT_BROWSERS_PATH="$PWD/build/figure-tools/browsers" \
  node build/figure-tools/node_modules/playwright/cli.js install chromium --only-shell
python3 -m unittest discover -s third_party -p test_figure_bench.py -q
node doc/figures/authoring/figure-model.test.mjs
FSS_FIGURE_NODE_MODULES="$PWD/build/figure-tools/node_modules" \
  node doc/figures/authoring/export.mjs build/third_party/sweep/results/merged/chart-data.json
FSS_FIGURE_NODE_MODULES="$PWD/build/figure-tools/node_modules" \
  node doc/figures/authoring/site.test.mjs
```

`FSS_FIGURE_NODE_MODULES` selects the Playwright installation. The exporter
also supports the bundled Codex runtime. `FSS_FIGURE_D3` selects another local
D3 bundle. Rendering makes no network requests. Browser downloads, authoring
HTML, SVGs, 2x PNGs, source manifest, and site QA screenshots stay under `build`.

Inspect the PNGs at 980 display pixels wide. After visual review, repeat the
export command with `--publish` to copy SVG/PNG assets to `doc/figures` and
write `doc/site/data.json`. Publication drops previously generated figures
that this export no longer produces. Raw receipts remain under `build`.

The overview has five rows: DCF, full-width DPF, packed DPF, VDPF, and
DMPF/VDMPF. Its columns are Gen, Eval, and EvalAll. Default backends appear
in the overview. A missing operation stays absent. An ordinary DPF reference
joins a packed panel only when a packed method was actually measured there.
The absolute log axes retain slow third-party baselines.

Supplementary figures separate prepared-key setup/query/one-shot costs,
proof accumulators, PRP backends, and output groups or PRG backends. Prepared-key static panels focus on the default byte group. The site
retains integer-group preparation and query variants. Prepared
query speedups require matching configuration and source hashes. The direct
reference receipt is embedded in the speedup point's provenance. Setup is a
separately timed `Prepare` operation, never a subtraction of medians. The
block sweep uses actual T and excludes fixed-256-thread multi-point kernels.

Gen, Prepare, and EvalAll use time/key. Eval uses time/query. Counts identify
independent-key queries versus many queries on one key. Overview CPU point
times use µs, GPU point times use ns, preparation uses ns/key, and full-domain times use ms. Every
panel states its denominator. The comparison site applies the same metric
and converts tick values through the same unit scale as its tables.

`doc/site/method-model.mjs` is the shared method-description, style, and unit
model. Labels parse combined variants, including chain plus raw PRP. Colors
identify constructs and line styles/markers identify variants. Automated
checks reject duplicate complete styles or visible labels within a panel and
its shared row legend. When one construction keeps the same style but has
operation-specific outputs or proof choices, one legend entry states each
operation and its configuration. Curves and source records remain separate.
`figure-model.mjs` selects panels, `render.mjs` draws SVGs with D3, and
`export.mjs` renders them in headless Chromium. The method inventory in
`build/figures/manifest.json` lists every method's static figures and site
availability, including historical methods that remain site-only.

Each SVG embeds source hashes, leaf run manifests with hardware and revision,
and every plotted point's original record, raw receipt hash, explicit work
counts, setup boundary, and requested/actual T. Legacy `num_keys` and
`ns_per_key` are retained as historical fields. Consumers use `key_count`,
`query_count`, `domain_output_count`, `time_denominator`, and `ns_per_unit`.
