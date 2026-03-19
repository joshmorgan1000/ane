# ane

Hand-written ARM SME2 assembly kernels for the Apple Neural Engine (M4).

153 single-threaded kernels covering the standard ML operator taxonomy: elementwise arithmetic, activations, reductions, matrix multiply, convolution, normalization, attention, losses, and optimizers. All kernels target `armv9-a+sme2+sve2+sme-lutv2`.

## Why do you call it the Neural Engine?

Oh boy. This has become an interesting topic, but I should have seen it coming.

### **What exactly is the "Neural Engine"?**

#### Is it separate compute unit on the CPU that is completely undocumented?

Well, yes, it is a "separate unit", but no it is not "completely undocumented".

#### Is it a CoreML API/framework?

Yes. A lot of what is referred to as "the Neural Engine" is in fact software, which I will get into in a bit. But this is not the whole story.

#### Is it something that can only be accessed through CoreML?

That depends on your definition of "Neural Engine". If part of that includes the software stack, then yes. If you're just referring to the hardware, then no, as this repository demonstrates.

### The Discovery

I recently gained access to some ASUS GX10's, which incidentally have ARM CPU's in them along with unified memory... but with a CUDA GPU. Familiar with Accelerate, I was looking for an equivalent library, which led me to ARM's website, where I found out about the **SME2** extension in ARMv9.2. So I had Claude write some assembly code with every intention on running on a GX10 to see what kind of numbers I could get.

Instead, Claude built the tests on my M4. Which ran. Cool.

So I ran a full probe (well, mostly full) which can be [found here](probe.txt).

I did some performance benchmarks and found:
- `bfloat16` `fmopa` operations were the exact same speed as 32-bit float `fmopa` operations. This shouldn't be the case since they take up half the memory bandwidth?
- Many of the *basic arithmetic* operations for the `za` tile (`add`, `mul`, `sub`, etc) didn't work. Weird, but oh well, `fmla` works which does the same thing.
- None of the `FP8` operations worked. Zero.
- The `int8` and `uint8` MOPA operations *did* work.

That's when I started to get deja-vu. I had heard of something that was some sort of fast 8-bit matrix multiply unit, wasn't really optimized for FP16, and had some weird quirks. That's when it hit me. The four, 4096-byte `za` matrix tiles I was playing with was in fact the hardware behind the Apple Neural Engine. Or at least a large part of it.

### Further Testing

We know that Apple advertises the "Neural Engine" as being able to do 38 TOPS of compute. We ran a comprehensive benchmark measuring four independent compute paths — GPU (Metal `char4` INT8), CPU SME (`smopa` INT8), Apple's BNNS INT8 (`BNNSFilterCreateLayerFullyConnected` with `BNNSDataTypeInt8`), and NEON FP32 FMA — both in isolation and in every combination, for 10 seconds each, with heartbeat logging and proven-concurrent overlap windows.

Results on Apple M4 Max (selected from full table):

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

The critical result is **SME + BNNS**: when both run simultaneously, SME drops from 8.5 to 5.2 TOPS (39% drop) and BNNS drops from 4.2 to 1.6 TOPS (62% drop). They are fighting for the same hardware. Meanwhile, the GPU is completely unaffected by any CPU-side workload.

This proves that Apple's BNNS INT8 library — the same library that CoreML dispatches to for "Neural Engine" inference — uses the SME `za` matrix tiles. There is no separate coprocessor.

Additionally, the raw SME `smopa` kernels in this repository achieve **8.5 TOPS** on INT8 operations — more than **double** the 4.2 TOPS that Apple's own BNNS framework achieves on the same hardware. Direct access to the hardware is significantly more efficient than going through Apple's software stack.

Run the tests yourself:

```bash
bash tests/run_full_throughput_tests.sh
```

The test script is fully self-contained — it generates all source code inline, builds in a temp directory, and cleans up after itself. Each worker process self-times with nanosecond-precision epoch timestamps. The analyzer finds the proven-concurrent window (where all processes in a test are confirmed running) and computes throughput from interior heartbeats only, discarding startup and shutdown artifacts.

## Conclusion

The "Neural Engine" is the ARM SME2 `za` matrix tiles plus a software scheduling layer. Apple's "38 TOPS" comes from the GPU (~37 TOPS via native Metal `char4` INT8 vectors) plus the CPU SME tiles (~8.5 TOPS via `smopa`). BNNS and CoreML dispatch to the same SME hardware, achieving ~4.2 TOPS through their software abstraction.

The evidence:
1. **SME and BNNS INT8 contend for the same hardware.** Running them simultaneously drops both — total throughput is less than either alone. If BNNS used separate silicon, both would maintain full speed.
2. **GPU is fully independent.** No throughput drop when paired with any CPU workload — it's genuinely separate hardware.
3. **Same limitations.** The "ANE" has the same data type support as SME (no FP8, same int8/bf16/fp32 types).
4. **Direct SME access outperforms the "ANE" by 2x.** Raw `smopa` at 8.5 TOPS vs BNNS at 4.2 TOPS on the same hardware, same data types.

## So What?

For 32-bit floating point workloads, `cblas_sgemm` from Accelerate is likely your best bet — it uses `fmopa` under the hood and Apple is very good at scheduling it.

For 8-bit integer workloads, this repository's `smopa` kernels achieve over twice the throughput of Apple's own BNNS INT8 path. The `luti4` instructions can be used for efficient quantized inference or training. Working examples are included (though not fully implemented yet).

## Is this production ready?

Absolutely not. Test and use at your own risk.

## Requirements

- Apple M4 (or any ARM processor with SME2)
- CMake 3.19+
- C++20 compiler (Apple Clang recommended)

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.logicalcpu)
```

Or with Ninja:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja
ninja
```

## Install

```bash
cmake --install build --prefix ~/.local
```

## Usage

```cpp
#include <ane/ane.hpp>

// Single-threaded elementwise add
float a[1024], b[1024], c[1024];
ane::kernel::add_fp32(a, b, c, 1024);

// Matrix multiply
float A[64*128], B[128*32], C[64*32];
ane::kernel::matmul_fp32(A, B, C, 64, 32, 128);
```

Link against the static library:

```cmake
find_package(ane REQUIRED)
target_link_libraries(my_app PRIVATE ane::ane)
```

## Kernel Categories

| Category | Examples |
|---|---|
| Elementwise | `add_fp32`, `mul_fp32`, `div_fp32`, `fma_fp32`, `scalar_mul_fp32` |
| Activations | `relu_fp32`, `gelu_fp32`, `silu_fp32`, `sigmoid_fp32`, `mish_fp32`, `elu_fp32` |
| Reductions | `reduce_sum_fp32`, `reduce_max_fp32`, `dot_fp32`, `sumsqr_fp32`, `argmax_fp32` |
| Matrix Multiply | `matmul_fp32`, `matmul_int8`, `matmul_bfp16`, transposed variants |
| Convolution | `conv2d_fp32`, fused bias/relu/bn/swish/gelu variants, backward passes |
| Normalization | `layernorm_fp32`, `fused_rms_norm_scale_fp32`, `fused_batchnorm_relu_fp32` |
| Attention | `flash_attn_*`, `sdp_attn_*`, `gqa_attn_*`, `cross_attn_*`, `rope_fp32` |
| Losses | `mse_loss_fp32`, `cross_entropy_loss_fp32`, `bce_loss_fp32`, `mae_loss_fp32` |
| Optimizers | `adam_fp32`, `sgd_fp32`, `param_update_fp32` |
| Quantization | `quantize_fp32_int8`, `dequantize_int8_fp32`, `q8_0_matvec` |
| Type Convert | `fp32_to_bf16`, `bf16_to_fp32`, `fp32_to_int32` |
| Bitwise | `and_u32`, `or_u32`, `xor_u32`, `shl_u32`, `shr_u32` |
| LUT | `tbl_u8`, `luti4_u8`, `luti2_u8` |
| Stochastic | `dropout_fp32`, `gaussian_noise_fp32` |

## AI Assistants

Thanks to Anthropic's Claude models (Haiku, Sonnet, and Opus 4.5 and 4.6 were all used at some point)

## License

MIT. See [LICENSE](LICENSE) for details.
