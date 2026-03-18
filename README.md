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

We know that Apple advertises the "Neural Engine" as being able to do 38 TOPS of compute. We measured the actual INT8 throughput of each compute path:

| Test | INT8 TOPS |
|---|---|
| GPU peak ALU (Metal `char4` vectors, 1M threads) | ~37 TOPS |
| CPU SME (`smopa` int8, 4 threads) | ~8.4 TOPS |
| GPU + CPU SME simultaneously | **~44 TOPS** |
| GPU + CPU SME + CoreML "ANE" simultaneously | **~44 TOPS** (no gain) |

The GPU alone nearly matches Apple's 38 TOPS claim. Running both GPU and CPU SME at once exceeds it at ~44 TOPS.

The critical test: we ran a CoreML model (with `.all` compute units, which requests the "ANE") simultaneously with the GPU and CPU workloads. If the ANE were a separate compute unit, total throughput should jump to ~80 TOPS. Instead, CoreML completed **zero inferences** — the GPU and CPU were fully saturated, leaving no hardware available for CoreML. Total throughput stayed at ~44 TOPS.

Run the tests yourself: `cd tests/throughput && bash ../run_full_throughput_tests.sh`

## Conclusion

The "Neural Engine" is not a separate compute unit. Apple's "38 TOPS" comes from the GPU (~37 TOPS using native `char4` int8 vectors) plus the CPU's SME `za` matrix tiles (~8.4 TOPS via `smopa`). Running CoreML with "ANE" dispatch alongside saturated GPU+CPU workloads adds zero throughput — because there is no additional hardware to run on.

The evidence:
1. **Throughput test**: GPU + CPU = ~44 TOPS. Adding CoreML "ANE" = still ~44 TOPS, 0 inferences completed.
2. **Same limitations**: The "ANE" has the same data type support as SME (no FP8, same int8/bf16/fp32 types) and cannot run simultaneously with SME workloads.

If you are dealing with 32-bit floating point matrix multiply workloads, you're better off just calling `cblas_sgemm` from Accelerate. The CBLAS implementations call the `fmopa` instructions under the hood, and Apple is better at scheduling and optimizing than you likely are.

## There's Still Hope

What CBLAS does not have is the 8-bit `smopa` instructions, as well as the `luti4` instructions which can be used for very efficient quantized inference or training. Working examples are included in thie repository (although not fully implemented yet).

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
