# Benchmark figure authoring

The renderer consumes the normalized JSON emitted by `third_party/figure_bench.py`.
It uses successful measurements with a positive `ns_per_key` value. It does
not interpolate missing domain sizes or create placeholder curves. The
`schema-example.json` file deliberately contains no measurements.

Install the rendering dependencies in the build directory:

```sh
npm install --prefix build/figure-tools --no-audit --no-fund d3@7.9.0 playwright@1.62.1
PLAYWRIGHT_BROWSERS_PATH="$PWD/build/figure-tools/browsers" \
  node build/figure-tools/node_modules/playwright/cli.js install chromium --only-shell
node doc/figures/authoring/figure-model.test.mjs
FSS_FIGURE_NODE_MODULES="$PWD/build/figure-tools/node_modules" \
  node doc/figures/authoring/export.mjs build/third_party/sweep/results/figure-data/chart-data.json
```

`FSS_FIGURE_NODE_MODULES` selects the directory containing Playwright. The
exporter also has a default for the bundled Codex runtime used for these
figures. Set `FSS_FIGURE_D3` to the local
`d3.min.js` bundle to use another D3 installation. No network requests are
made while rendering.

The authoring HTML, static SVGs, 2x PNGs, and source manifest are written to
`build/figures`. Inspect the PNGs at a displayed width of 980 pixels. After
visual review, repeat the export command with `--publish` to copy the SVG and
PNG publication assets to `doc/figures`.

Each static SVG embeds a compact JSON `<metadata>` record. It contains the
normalized source SHA-256, measurement dates and hardware, source revision
and hashes, and every plotted timing with its original configuration.
Raw benchmark logs and normalized datasets remain under `build`.

Each main figure uses domain size N on the horizontal axis, with ticks at
powers of two. The supplementary block-size figure uses actual CUDA threads
per block T. The vertical axis is logarithmic time per key, in ns for GPU
point operations, µs for CPU point operations, and ms for full evaluation. GPU
point operations use amortized timings at the measured K. Full-domain
evaluation uses K=1. The legends carry native output and PRG configurations.

`figure-model.mjs` maps the benchmark schema to figures. `render.mjs` owns
the D3 SVG layout, stable styles, labels, and legends. `export.mjs` builds
self-contained authoring HTML and exports it with headless Chromium.
