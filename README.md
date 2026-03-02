# ane

Hand-written ARM SME2 assembly kernels for the Apple Neural Engine (M4).

153 single-threaded kernels covering the standard ML operator taxonomy: elementwise arithmetic, activations, reductions, matrix multiply, convolution, normalization, attention, losses, and optimizers. All kernels target `armv9-a+sme2+sve2+sme-lutv2`.

## Why do you call it the Neural Engine?

Plenty of reasons:
- The performance characteristcs are identical to the marketed ANE performance
- They have the same exact restrictions (prefers int8, can't do FP16, etc)
- You never see the ANE and these operations in the same room at the same time. Run a CoreML model while this is running, suddenly both's performance is halved.

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
