# Discovery: SME2 and the "Neural Engine"

This page documents the research behind `ane` — how we found that Apple Silicon's SME2 matrix tiles are the same hardware that Apple's "Neural Engine" software stack dispatches to.

> **Important distinction:** The "Neural Engine" as Apple markets it is a combination of hardware (the SME2 `za` matrix tiles on the CPU) and a software scheduling framework (CoreML, BNNS). This project accesses the matrix hardware directly, bypassing the software stack entirely. That does not make this "the Neural Engine" — it means we are driving the same silicon through a different interface.

---

## Background

While looking for an ARM equivalent to Accelerate on some ASUS GX10 machines (ARM CPUs + unified memory + CUDA GPU), I found ARM's documentation for the **SME2** extension in ARMv9.2. I had Claude write some assembly to test it — intending to run on the GX10 — but the tests got built and executed on my M4 instead.

They ran. So I kept going.

## Probe Results

A [comprehensive probe](../probes/probe.sh) of the M4's SME2 support revealed:

- `bfloat16` `fmopa` operations ran at the exact same speed as `fp32` `fmopa` — unexpected given the halved memory bandwidth.
- Many basic `za` tile arithmetic instructions (`add`, `mul`, `sub`) returned `SIGILL`. The fused variants (`fmla`) worked fine.
- **None** of the `FP8` instructions worked. Zero.
- `int8` and `uint8` MOPA operations worked correctly.
- Four isolated 4096-byte `za` tile arrays were accessible across CPU cores — significant matrix compute capacity that Apple does not document or advertise.

The four separate `za` matrix arrays, the focus on 8-bit integer operations, the missing FP8 support, the lack of documentation — this matched everything I had heard about the behavior of Apple's "Neural Engine" through its software APIs.

## Contention Benchmark

To test whether SME and the "Neural Engine" share hardware, we ran a comprehensive benchmark measuring five independent compute paths — GPU (Metal `char4` INT8), CPU SME (`smopa` INT8), Apple's BNNS INT8 (`BNNSFilterCreateLayerFullyConnected` with `BNNSDataTypeInt8`), CBLAS SGEMM FP32, and NEON FP32 FMA — both in isolation and in every combination, with heartbeat logging and proven-concurrent overlap windows.

Results on Apple M4 Max:

| Test | GPU | SME | BNNS INT8 | NEON | Total |
|---|---|---|---|---|---|
| GPU alone | 37.4 | | | | 37.4 |
| SME alone (4 threads) | | 8.5 | | | 8.5 |
| BNNS INT8 alone (4 threads) | | | 4.2 | | 4.2 |
| GPU + SME | 37.4 | 8.5 | | | **45.9** |
| GPU + BNNS | 37.3 | | 4.0 | | 41.3 |
| **SME + BNNS** | | **5.2** | **1.6** | | **6.9** |
| GPU + SME + BNNS | 37.0 | 5.2 | 1.6 | | 43.8 |
| GPU + SME + BNNS + NEON | 35.8 | 4.9 | 1.7 | 0.7 | 43.2 |

The critical result is **SME + BNNS**: when both run simultaneously, SME drops from 8.5 to 5.2 TOPS (39% drop) and BNNS drops from 4.2 to 1.6 TOPS (62% drop). They are fighting for the same hardware. The GPU is completely unaffected by any CPU-side workload.

Run the tests yourself:

```bash
bash tests/run_full_throughput_tests.sh
```

The test script is fully self-contained — it generates all source code inline, builds in a temp directory, and cleans up after itself. Each worker process self-times with nanosecond-precision epoch timestamps. The analyzer finds the proven-concurrent window (where all processes in a test are confirmed running) and computes throughput from interior heartbeats only, discarding startup and shutdown artifacts.

## Evidence

1. **SME and BNNS INT8 contend for the same hardware.** Running them simultaneously drops both — total throughput is less than either alone. If BNNS used separate silicon, both would maintain full speed.
2. **GPU is fully independent.** No throughput drop when paired with any CPU workload — it is genuinely separate hardware.
3. **Same limitations.** Apple's "Neural Engine" has the same data type support as SME: no FP8, same int8/bf16/fp32 types.
4. **Direct SME access outperforms the software stack by 2x.** Raw `smopa` at 8.5 TOPS vs BNNS at 4.2 TOPS on the same hardware, same data types.
5. **Same official documentation:** none.

## Practical Implications

For **32-bit floating point** workloads, `cblas_sgemm` from Accelerate is likely your best bet — it uses `fmopa` under the hood and Apple is very good at scheduling it.

For **8-bit integer** workloads, this repository's `smopa` kernels achieve over twice the throughput of Apple's own BNNS INT8 path. The `luti4` instructions can be used for efficient quantized inference or training.

This is **not production ready**. Test and use at your own risk. The SME2 instructions are undocumented by Apple and behavior may change between chip generations.

## M5 Notes

Testing on Apple M5 Max has revealed performance discrepancies where the M4 outperforms the M5 in certain SME throughput benchmarks. Investigation has found that `SMLALL` (signed multiply-add long long) runs approximately 5.3x faster than `SMOPA` on M5 — suggesting Apple may be shifting the microarchitecture toward different instruction paths across generations. This is under active investigation.
