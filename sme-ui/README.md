# M4 SME Dashboard

This subdirectory (`m4-sme-docs`) contains a React-based dashboard, built with Vite, for visualizing hardware capabilities and throughput data related to the Apple M4 SME (Scalable Matrix Extension) experiments.

## Fetching Live Hardware Data

To ensure the visualizations reflect the actual hardware capabilities of your Apple Silicon device, you can fetch live data by running:

```bash
npm run fetch-data
```

This command acts as a pipeline that runs two scripts:

1. **`update_probe.cjs`**: Executes the shell script `../probes/probe_instructions.sh`. This script compiles C payloads using Apple `clang` and uses `objdump` to test which SME instructions are valid (returning `OK`) and which fail (returning `SIGILL`). The result is parsed and saved directly to `src/data/probe_results.json`.
2. **`update_throughput.cjs`**: Executes `../tests/run_full_throughput_tests.sh`. This runs intensive benchmark tests across different compute units (GPU, BNNS, NEON, SME) and calculates the throughput in TOPS (Tera Operations Per Second). The summary output is parsed and saved to `src/data/throughput_results.json`.

*Note: Running throughput tests takes about 2-3 minutes.*

## Building the Dashboard

The dashboard uses Vite for fast development and building. Because we want a highly portable dashboard that doesn't require a web server to view, it is configured to bundle into a single HTML file.

This is achieved using `vite-plugin-singlefile` which inline all CSS and JavaScript into the final document.

To build the dashboard:

```bash
npm run build
```

This will run TypeScript type-checking (`tsc -b`) followed by `vite build`. The output will be a single `index.html` file inside the `dist` directory that you can open locally in any browser or share directly without deploying any static assets.

## Development

If you prefer to run a local development server with hot-module replacement (HMR), simply use:

```bash
npm run dev
```
