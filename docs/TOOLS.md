# Tools

## Hardware Probe

[`probes/probe.sh`](../probes/probe.sh) is a comprehensive SME/SVE2/SSVE instruction probe for Apple Silicon. It compiles and executes hundreds of individual ARM instructions to determine which ones your hardware actually supports.

**Usage:**

```bash
cd probes
./probe.sh                  # Run full probe (instructions + throughput)
./probe.sh --skip ops       # Skip instruction probes, run only throughput tests
./probe.sh --skip tp        # Skip throughput tests, run only instruction probes
```

**Output:** A table of instructions with status: `WORKS`, `SIGILL` (illegal instruction), or `COMPILE_FAIL`. This is how we determined that FP8 instructions are non-functional on M4, while int8 SMOPA/UMOPA work correctly.

A reference list of all A64 SME instructions is available at [`probes/sme_instruction_list.txt`](../probes/sme_instruction_list.txt).

## Throughput Benchmark

[`tests/run_full_throughput_tests.sh`](../tests/run_full_throughput_tests.sh) measures concurrent throughput across five independent compute paths to prove that SME and BNNS contend for the same hardware. See [docs/DISCOVERY.md](DISCOVERY.md) for the full results and analysis.

**Usage:**

```bash
bash tests/run_full_throughput_tests.sh
```

The script is fully self-contained — it generates all source code (C, Metal, ARM assembly, Swift) inline, builds in a temp directory, and cleans up after itself. Each worker self-times with nanosecond-precision timestamps. The analyzer finds the proven-concurrent window and computes throughput from interior heartbeats only, discarding startup and shutdown artifacts.

**Workers tested:**
- **GPU** (Metal): `char4` INT8 multiply-accumulate
- **SME** (N threads): ARM SMOPA int8 outer product on ZA tiles
- **BNNS** (N threads): Apple Accelerate INT8 fully-connected
- **CBLAS** (N threads): Apple Accelerate SGEMM FP32
- **NEON** (N threads): ARM NEON FP32 fused multiply-add

Thread counts are auto-detected from your hardware topology.

## Dashboard

The React-based dashboard dynamically compiles and runs probing payloads against your hardware, then presents the results in an interactive single-page application.

**Usage:**

```bash
./dashboard.sh
```

This will:
1. Build the MNIST demo pipeline.
2. Run live hardware probes against your SME hardware.
3. Build the Vite React application into a single `dist/index.html` file.
4. Open the dashboard in your browser.

Pre-built dashboard snapshots for M4 and M5 are available in the [`dashboards/`](../dashboards/) directory.

For dashboard development details (HMR, data fetching, build process), see [`sme-ui/README.md`](../sme-ui/README.md).

## PyTorch Comparison

[`scripts/mnist_pytorch_train_gpu.py`](../scripts/mnist_pytorch_train_gpu.py) trains the same MNIST network using PyTorch CPU (Eager) for comparison against the SME bytecode interpreter. Requires Python 3.10+ with `torch` and `torchvision` installed.
