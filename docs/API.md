# ANE Bytecode Language

## Your First Program

Add two arrays of floats together:

```cpp
#include <ane/ane.hpp>

float* a = /* 1024 floats, 64-byte aligned */;
float* b = /* 1024 floats, 64-byte aligned */;
float* c = /* 1024 floats, 64-byte aligned */;

ane::script add(R"(
    x: ZVEC_F32;
    y: ZVEC_F32;
    _LOOP_:;
        x.load(params[0]);
        y.load(params[1]);
        x = x + y;
        x.save(params[2]);
        params[0]++;
        params[1]++;
        params[2]++;
    goto _LOOP_ 64;
)");

add.exec({a, b, c});
// c now contains a[i] + b[i] for all 1024 floats
```

That's a complete, working program. Here's what's happening:

- **`x: ZVEC_F32`** declares a variable that holds 16 floats at a time.
- **`x.load(params[0])`** reads the next 16 floats from your first data pointer into `x`.
- **`x = x + y`** adds them element-wise.
- **`x.save(params[2])`** writes the 16 results to your output pointer.
- **`params[0]++`** advances the pointer by 16 floats (64 bytes) so the next iteration reads the next chunk.
- **`goto _LOOP_ 64`** repeats the body 64 times. 64 iterations × 16 floats = 1024 floats total.
- **`add.exec({a, b, c})`** runs the whole thing. `params[0]` gets `a`, `params[1]` gets `b`, `params[2]` gets `c`.

The program compiles once and can be reused with different pointers:
```cpp
add.exec({a2, b2, c2});  // same program, different data
```

---

## Example: Dot Product

Multiply two arrays element-wise and accumulate the results. This is the building block for distance calculations, similarity scores, and matrix math.

```cpp
ane::script dot(R"(
    a: ZVEC_F32;
    b: ZVEC_F32;
    sum: ZVEC_F32;

    // Zero the accumulator (load from a zeroed buffer)
    sum.load(params[3]);

    _LOOP_:;
        a.load(params[0]);
        b.load(params[1]);
        a = a * b;
        sum = sum + a;
        params[0]++;
        params[1]++;
    goto _LOOP_ 64;

    // Write the partial sums out
    sum.save(params[2]);
)");

// params[3] points to a zeroed 64-byte buffer (for initializing sum)
alignas(64) float zero_buf[16] = {};
dot.exec({vec_a, vec_b, result_buf, zero_buf});
// result_buf now holds 16 partial sums — add them up on the CPU to get the final scalar
```

This shows a few new things:
- **Three variables** — `a`, `b`, and `sum`. The compiler assigns each one a separate internal slot.
- **Accumulation** — `sum` persists across loop iterations. You load it once before the loop, update it inside, and store it after.
- **The loop only advances two pointers** — `sum` stays in a variable, never touching memory until the final `save`.

---

## Example: Fused Multiply-Accumulate with Matrix Scratchpad

For matrix multiplication, the hardware has a dedicated **scratchpad** — a 32×32 grid of values that persists across instructions. You don't access it directly; instead you clear it, feed data into it with outer product operations, and read the result out.

```cpp
ane::program matmul;

// Set up data pointers
matmul.emit(ane::Op::set_param, uint8_t(0), reinterpret_cast<uintptr_t>(A));
matmul.emit(ane::Op::set_param, uint8_t(1), reinterpret_cast<uintptr_t>(B));

// Clear the scratchpad
matmul.emit(ane::Op::zero_za);

// Accumulate: load rows of A and columns of B, compute outer products
matmul.emit(ane::Op::acc_smopa,
    uint32_t(k_steps),   // how many 128-byte chunks to process
    reinterpret_cast<uintptr_t>(A),
    reinterpret_cast<uintptr_t>(B));

// Read out the 32×32 result
ane::z_tiles result;
matmul.emit(ane::Op::store_tiles, reinterpret_cast<uintptr_t>(result.ptr()));

matmul.exec();
int32_t* output = result.as_i32();  // 1024 int32 values
```

For a standard BLAS-style matrix multiply with arbitrary dimensions, transposition, and scaling, there's a single instruction that handles everything:

```cpp
ane::program gemm;
gemm.emit(ane::Op::cblas_sgemm,
    uint8_t(0),         // flags: 0 = no transpose
    uint32_t(100),      // M rows
    uint32_t(200),      // N columns
    uint32_t(300),      // K shared dimension
    uint32_t(300),      // row stride of A
    uint32_t(200),      // row stride of B
    uint32_t(200),      // row stride of C
    1.0f,               // alpha (C = alpha*A*B + beta*C)
    0.0f,               // beta
    reinterpret_cast<uintptr_t>(A),
    reinterpret_cast<uintptr_t>(B),
    reinterpret_cast<uintptr_t>(C));
gemm.exec();
```

---

## The Environment

### What you get

| Resource | Size | What it's for |
|----------|------|--------------|
| **30 variables** | 16 floats each (64 bytes) | Your working data. Declared with `name: ZVEC_F32;` in the DSL. |
| **8 data pointers** | one address each | Point to your arrays in memory. Set via `params[0]` through `params[7]`. |
| **1 scratchpad** | 32×32 values (4KB) | Matrix multiplication workspace. Managed by `zero_za`, outer product ops, and `store_tiles`. |
| **1 lookup table** | 64 bytes | Small table for expanding packed/quantized data. Loaded with `load_zt0`. |

### Rules

- **One session per `exec()`.** Everything runs as a single burst. Variables persist across instructions within the program, but are gone after `exec()` returns.
- **One loop per program.** Max 255 iterations.
- **All pointers must be 64-byte aligned.** Use `std::aligned_alloc(64, size)` or `ane::z_stream<T>`.
- **Buffer sizes should be multiples of 64 bytes** (16 floats, 32 int16s, or 64 int8s).
- **Variables hold 16 floats at a time.** To process 1024 floats, loop 64 times (1024 ÷ 16 = 64).

---

## DSL Reference

### Declaring Variables

```
name: ZVEC_F32;
```
Creates a variable that holds 16 float32 values. You can declare up to 30 variables.

### Loading and Saving Data

```
x.load(params[N]);     // read 16 values from data pointer N into x
x.save(params[N]);     // write x (16 values) to data pointer N
params[N]++;           // advance data pointer N by 16 values (64 bytes)
```

`load` and `save` do not move the pointer — you advance it yourself with `++`.

### Arithmetic

```
result = a + b;       // element-wise addition (16 floats at once)
result = a - b;       // subtraction
result = a * b;       // multiplication
```

The destination can be the same as a source: `x = x + y;` is fine.

### Loops

```
_LABEL_:;
    // ... body ...
goto _LABEL_ N;       // repeat the body N times (max 255)
```

The label name must start and end with an underscore. The semicolon after the colon is required.

### Comments

```
// single-line comment
/* multi-line
   comment */
```

### Calling a script

```cpp
ane::script s(R"( ... source ... )");
s.exec({ptr0, ptr1, ptr2});   // params[0]=ptr0, params[1]=ptr1, etc.
```

The script compiles on first `exec()` and caches the result. Subsequent calls with different pointers reuse the compiled program.

---

## Instruction Catalog

Below is every available instruction grouped by what it does. Each entry lists the parameters in order and notes which of your **variables get overwritten** — meaning any data in those slots is lost after the instruction runs.

When using the DSL, you don't need most of these directly — the compiler generates them for you. They're documented here for the `ane::program` builder and for understanding what's happening under the hood.

### Data Movement

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `set_param` | `idx:u8`, `ptr:u64` | Store a memory address in data pointer slot idx (0-7) | nothing |
| `load_param` | `idx:u8` | Read 16 values from data pointer idx into v0 | v0 |
| `store_param` | `idx:u8` | Write v0 to data pointer idx | nothing |
| `advance_param` | `idx:u8` | Advance data pointer idx by 64 bytes | nothing |
| `load` | `ptr:u64` | Read 16 values from a fixed address into v0 | v0 |
| `store` | `ptr:u64` | Write v0 to a fixed address | nothing |
| `mov_zreg` | `src:u8`, `dst:u8` | Copy variable src to variable dst | dst only |

### Control Flow

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `loop_begin` | `count:u8` | Start a loop (1-255 iterations) | nothing |
| `loop_end` | `offset:u16` | End of loop (offset computed automatically) | nothing |

### Float32 Arithmetic (register-addressed)

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `fadd_zreg` | `dst:u8`, `src1:u8`, `src2:u8` | dst = src1 + src2 | dst, v0-v1 |
| `fsub_zreg` | `dst:u8`, `src1:u8`, `src2:u8` | dst = src1 - src2 | dst, v0-v1 |
| `fmul_zreg` | `dst:u8`, `src1:u8`, `src2:u8` | dst = src1 × src2 | dst, v0-v1 |
| `fmla_zreg` | `dst:u8`, `src1:u8`, `src2:u8` | dst += src1 × src2 | dst, v0-v2 |

### Bitwise (register-addressed)

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `and_zreg` | `dst:u8`, `src1:u8`, `src2:u8` | dst = src1 AND src2 | dst, v0-v1 |
| `orr_zreg` | `dst:u8`, `src1:u8`, `src2:u8` | dst = src1 OR src2 | dst, v0-v1 |
| `eor_zreg` | `dst:u8`, `src1:u8`, `src2:u8` | dst = src1 XOR src2 | dst, v0-v1 |
| `not_zreg` | `dst:u8`, `src:u8` | dst = NOT src | dst, v0 |
| `lsl_zreg` | `dst:u8`, `src:u8`, `amount:u8` | dst = src << amount | dst, v0-v1 |
| `lsr_zreg` | `dst:u8`, `src:u8`, `amount:u8` | dst = src >> amount (logical) | dst, v0-v1 |
| `asr_zreg` | `dst:u8`, `src:u8`, `amount:u8` | dst = src >> amount (arithmetic) | dst, v0-v1 |

### Lookup Tables

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `load_zt0` | `ptr:u64` | Load 64-byte lookup table | lookup table |
| `luti2_zreg` | `dst:u8`, `src:u8` | 2-bit index lookup (src → table → dst) | dst, v0-v1 |
| `luti4_zreg` | `dst:u8`, `src:u8` | 4-bit index lookup (src → table → dst) | dst, v0-v1 |

### Scratchpad / Matrix Operations

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `zero_za` | *(none)* | Clear the scratchpad to zero | scratchpad |
| `store_tiles` | `dst_ptr:u64` | Write scratchpad (4KB) to memory | v4-v11 |
| `load_bias` | `src_ptr:u64` | Load 4KB into scratchpad from memory | v4-v11, scratchpad |
| `scale_store` | `scale:f32`, `dst_ptr:u64` | Convert scratchpad int32→float×scale, write to memory | v4-v11, v16 |
| `smopa_zreg` | `tile:u8`, `src1:u8`, `src2:u8` | Signed int8 outer product into scratchpad tile | v0-v1, tile |
| `umopa_zreg` | `tile:u8`, `src1:u8`, `src2:u8` | Unsigned int8 outer product | v0-v1, tile |
| `usmopa_zreg` | `tile:u8`, `src1:u8`, `src2:u8` | Unsigned × signed outer product | v0-v1, tile |
| `fmopa_zreg` | `tile:u8`, `src1:u8`, `src2:u8` | Float32 outer product | v0-v1, tile |

### BLAS Matrix Multiply

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `cblas_sgemm` | `flags:u8`, `M:u32`, `N:u32`, `K:u32`, `lda:u32`, `ldb:u32`, `ldc:u32`, `alpha:f32`, `beta:f32`, `A:u64`, `B:u64`, `C:u64` | C = alpha×op(A)×op(B) + beta×C | v0-v21, scratchpad |

flags: bit 0 = transpose A, bit 1 = transpose B.

### Fused Array Operations

These process entire arrays from memory with internal loops — no `loop_begin`/`loop_end` needed.

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `elementwise_add_fp32` | `count:u32`, `a:u64`, `b:u64`, `out:u64` | out[i] = a[i] + b[i] | v0-v1 |
| `elementwise_scaled_add_fp32` | `count:u32`, `scale:f32`, `a:u64`, `b:u64`, `out:u64` | out[i] = a[i] + scale×b[i] | v0-v1, v16 |
| `elementwise_mul_fp32` | `count:u32`, `a:u64`, `b:u64`, `out:u64` | out[i] = a[i] × b[i] | v0-v1 |
| `relu_backward_fp32` | `count:u32`, `act:u64`, `grad:u64`, `out:u64` | out[i] = (act[i]>0) ? grad[i] : 0 | v0-v2, v17 |
| `l2_squared_fp32` | `dim:u32`, `a:u64`, `b:u64`, `out:u64` | out = sum((a[i]-b[i])²) | v0-v1, v4 |
| `l2_squared_bf16` | `dim:u32`, `a:u64`, `b:u64`, `out:u64` | (bfloat16 inputs, float32 result) | v0-v7 |
| `l2_squared_f64` | `dim:u32`, `a:u64`, `b:u64`, `out:u64` | (float64) | v0-v1, v4 |
| `cosine_dist_fp32` | `dim:u32`, `a:u64`, `b:u64`, `out:u64` | out = 1 - dot(a,b)/(‖a‖×‖b‖) | v0-v1, v4-v6 |
| `cosine_dist_bf16` | `dim:u32`, `a:u64`, `b:u64`, `out:u64` | (bfloat16 inputs) | v0-v11 |
| `cosine_dist_f64` | `dim:u32`, `a:u64`, `b:u64`, `out:u64` | (float64) | v0-v1, v4-v6 |
| `normalize_fp32` | `dim:u32`, `vec:u64` | vec[i] /= ‖vec‖ (in-place) | v0, v4, v16 |
| `reduce_sum_fp32` | `count:u32`, `src:u64`, `out:u64` | out = sum of all elements | v0-v1 |
| `count_matches` | `count:u32`, `pred:u64`, `labels:u64`, `out:u64` | out = number of matches | v0-v1 |

### Fused Matrix Multiply

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `dense_fp32` | `M:u32`, `N:u32`, `K:u32`, `scale:f32`, `flags:u8`, `A:u64`, `B:u64`, `bias:u64`, `C:u64` | C = scale×(A×B+bias), optional ReLU (flags bit 0) | v0-v21, scratchpad |
| `dense_i8` | `M:u32`, `N:u32`, `K:u32`, `sa:f32`, `sb:f32`, `flags:u8`, `A:u64`, `B_packed:u64`, `bias:u64`, `C:u64` | Int8 matmul (B must be pre-packed via `pack_b_i8`) | v0-v21, scratchpad |
| `dense_u8s8` | *(same as dense_i8)* | Unsigned A × signed B matmul | v0-v21, scratchpad |
| `acc_smopa` | `k_steps:u32`, `rows:u64`, `cols:u64` | Fused load+signed outer product loop | v0-v3, scratchpad |
| `acc_umopa` | *(same)* | Unsigned | v0-v3, scratchpad |
| `acc_usmopa` | *(same)* | Unsigned × signed | v0-v3, scratchpad |
| `acc_sumopa` | *(same)* | Signed × unsigned (**result is transposed**) | v0-v3, scratchpad |
| `smopa_2x2` | *(none)* | Outer product on pre-loaded v0-v3 | scratchpad |
| `umopa_2x2` | *(none)* | Unsigned | scratchpad |
| `usmopa_2x2` | *(none)* | Unsigned × signed | scratchpad |

### Fused Lookup Tables

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `luti4_op` | `count:u32`, `elem_size:u8`, `table:u64`, `indices:u64`, `out:u64` | 4-bit lookup, full array | v0-v1, lookup table |
| `luti2_op` | *(same)* | 2-bit lookup, full array | v0-v1, lookup table |

### Transforms

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `dct2_forward_fp32` | `dim:u32`, `src:u64`, `dst:u64` | H.264 DCT-II forward (groups of 4) | v0-v8 |
| `dct2_inverse_fp32` | `dim:u32`, `src:u64`, `dst:u64` | DCT-II inverse | v0-v8 |
| `transpose_fp32` | `M:u32`, `N:u32`, `src:u64`, `dst:u64` | Transpose M×N matrix | minimal |
| `transpose_i8` | `M:u32`, `N:u32`, `src:u64`, `dst:u64` | Transpose M×N int8 matrix | v0-v7, scratchpad |

### Quantization

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `quantize_fp32_i8` | `count:u32`, `src:u64`, `dst:u64`, `scale_out:u64` | Float→int8 with auto-scale | v0, v16-v19 |
| `quantize_fp32_i8_channelwise` | `M:u32`, `N:u32`, `src:u64`, `dst:u64`, `scales:u64` | Per-row float→int8 | v0, v16-v19 |
| `dequantize_i8_fp32` | `count:u32`, `scale:f32`, `src:u64`, `dst:u64` | Int8→float | v0, v16 |
| `pack_b_i8` | `K:u32`, `N:u32`, `src:u64`, `dst:u64` | Repack int8 for `dense_i8` | v0-v3 |
| `welford_stats_fp32` | `n_vecs:u32`, `dim:u32`, `src:u64`, `stats:u64` | Mean/stddev/maxabs/scale (double precision) | v0-v7 |
| `quantize_pack_4bit_fp32` | `n:u32`, `dim:u32`, `src:u64`, `stats:u64`, `raw_out:u64`, `dct_src:u64`, `dct_out:u64` | Float→4-bit packed SoA (dual) | minimal |
| `quantize_accum_2bit` | `count:u32`, `packed:u64`, `scale:u64`, `accum:u64` | 2-bit ternary decode + bf16 accumulate | v0-v5 |
| `accum_8bit` | `count:u32`, `data:u64`, `scale:u64`, `accum:u64` | Int8 × bf16 scale → bf16 accumulate | v0-v6 |

### Threshold & Bitmap

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `threshold_bitmap_fp32` | `dim:u32`, `threshold:f32`, `src:u64`, `bitmap:u64` | Float > threshold → packed bits | v0-v1, v16-v17 |
| `threshold_8bit` | `n_bytes:u32`, `threshold:u8`, `src:u64`, `bitmap:u64` | 8-bitplane counter → threshold → bits | minimal |
| `bitmap_score_pipeline` | `n_streams:u32`, `n_bytes:u32`, `n_vecs:u32`, `score_min:u32`, `max_cand:u32`, `streams:u64`, `flags:u64`, `cand_out:u64`, `count_out:u64` | Full bitmap accumulate→threshold→extract | v0-v7 |

### SoA Distance

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `soa_sub_scale_bf16` | `count:u32`, `src:u64`, `scalar:f32`, `scale:f32`, `accum:u64` | accum += bf16((src×scale - scalar)²) | v0-v3, v16-v17 |
| `soa_luti2_accum` | `count:u32`, `packed:u64`, `table:u64`, `accum:u64` | LUTI2 expand + bf16 accumulate | v0-v5, lookup table |
| `soa_luti4_accum` | `count:u32`, `packed:u64`, `table:u64`, `accum:u64` | LUTI4 expand + bf16 accumulate | v0-v5, lookup table |

### Classification

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `softmax_argmax_fp32` | `batch:u32`, `dim:u32`, `logits:u64`, `probs:u64`, `labels:u64`, `grad:u64`, `argmax:u64` | Softmax + cross-entropy grad + argmax | v0-v7, v16-v31 |
| `scatter_tile_fp32` | `M:u32`, `N:u32`, `stride:u32`, `row:u32`, `col:u32`, `src:u64`, `dst:u64` | Copy tile into larger matrix | v0-v1 |
