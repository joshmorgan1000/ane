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

## Example: Dot Product (Program Builder)

The `fdot_zreg` instruction computes a full dot product — multiply and horizontal sum — in one opcode. Use width=4 to process 64 elements per call:

```cpp
ane::program dot;

// z0 = accumulator (starts zeroed by default)
dot.emit(ane::Op::broadcast_scalar_zreg, uint8_t(0), 0.0f);

dot.begin_loop(16);  // 16 iterations × 64 elements = 1024 total
    // Load 64 input elements into z1:z4
    for (uint8_t v = 0; v < 4; v++) {
        dot.emit(ane::Op::load_param, uint8_t(0));           // z0 = next 16 floats from param[0]
        dot.emit(ane::Op::mov_zreg, uint8_t(0), uint8_t(1 + v));
        dot.emit(ane::Op::advance_param, uint8_t(0));
    }
    // Load 64 weight elements into z5:z8
    for (uint8_t v = 0; v < 4; v++) {
        dot.emit(ane::Op::load_param, uint8_t(1));
        dot.emit(ane::Op::mov_zreg, uint8_t(0), uint8_t(5 + v));
        dot.emit(ane::Op::advance_param, uint8_t(1));
    }
    // Restore accumulator, then accumulate 4 products element-wise
    dot.emit(ane::Op::mov_zreg, uint8_t(9), uint8_t(0));    // z0 = saved accumulator
    dot.emit(ane::Op::fmla_wide_zreg, uint8_t(4));           // z0 += z1*z5 + z2*z6 + z3*z7 + z4*z8
    dot.emit(ane::Op::mov_zreg, uint8_t(0), uint8_t(9));    // save accumulator
dot.end_loop();

// Horizontal sum to scalar
dot.emit(ane::Op::mov_zreg, uint8_t(9), uint8_t(0));
dot.emit(ane::Op::faddv_zreg, uint8_t(0), uint8_t(0));      // z0 = broadcast(sum(z0))
dot.emit(ane::Op::store, reinterpret_cast<uintptr_t>(result));

dot.emit(ane::Op::set_param, uint8_t(0), reinterpret_cast<uintptr_t>(vec_a));
dot.emit(ane::Op::set_param, uint8_t(1), reinterpret_cast<uintptr_t>(vec_b));
dot.exec();
// result[0..15] all contain the scalar dot product
```

## Example: Dot Product (DSL)

The same operation using the scripting DSL — the compiler handles register allocation:

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

alignas(64) float zero_buf[16] = {};
dot.exec({vec_a, vec_b, result_buf, zero_buf});
// result_buf holds 16 partial sums — add on CPU to get the scalar
```

The DSL version uses element-wise multiply + add. The program builder version uses `fmla_wide_zreg` which is more efficient (4 products per opcode dispatch) but requires manual register management.

**Multi-vector variables** (`ZVEC2_F32`, `ZVEC4_F32`) let the DSL compiler auto-expand operations across wider data. See "Declaring Variables" in the DSL Reference below.

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
name: ZVEC_F32;       // 1 slot  — holds 16 floats (64 bytes)
name: ZVEC2_F32;      // 2 slots — holds 32 floats (128 bytes)
name: ZVEC4_F32;      // 4 slots — holds 64 floats (256 bytes)
```

`ZVEC_F32` is the single-width type. `ZVEC2_F32` and `ZVEC4_F32` declare **multi-vector** variables that occupy 2 or 4 consecutive internal slots. The compiler tracks the width and auto-expands all operations — loads, saves, and arithmetic are emitted once per slot.

With 30 slots total, you can fit 30 single-width variables, 15 double-width, or 7 quad-width (with 2 slots left over).

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

For multi-vector variables, the compiler expands each operation across all slots:
```
a: ZVEC4_F32;
b: ZVEC4_F32;
c = a + b;            // emits 4 add instructions (one per slot pair)
```

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

### Intrinsic Functions

High-level operations (GEMM, softmax, etc.) can be called directly from DSL scripts as intrinsic functions. Pointer arguments use `params[N]` — the actual memory addresses are filled in at `exec()` time.

```
// fp32 matrix multiply: C = A × B (no transpose, alpha=1, beta=0)
sgemm(0, 128, 256, 64, 64, 256, 256, 1.0, 0.0, params[0], params[1], params[2]);

// bf16 matrix multiply with transposed B
bfgemm(2, 128, 256, 64, 64, 256, 256, 1.0, 0.0, params[0], params[1], params[2]);

// Partial softmax for distributed computation
softmax_partial(1024, params[0], params[1], params[2], params[3]);

// RMS normalization
rms_norm(4096, 0.00001, params[0], params[1], params[2]);
```

Available intrinsics: `sgemm`, `bfgemm`, `igemm`, `ugemm`, `usgemm`, `gemm_tile`, `softmax`, `softmax_partial`, `softmax_correct`, `reduce_sum_sq`, `reduce_col_sum`, `rms_norm`, `layer_norm`, `silu`, `silu_backward`, `gelu`, `softmax_backward`, `rope`, `causal_mask`, `adam_step`, `q8_0_gemv`, `q4_0_gemv`, `elementwise_add`, `elementwise_mul`, `reduce_sum`.

Arguments are integer literals, float literals (`1.0`, `0.5f`), or `params[N]` references. Negative floats are written as `-1.0`. See the Instruction Catalog below for the parameter order of each operation.

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

All CBLAS-style GEMM ops share the same encoding and semantics: `C = alpha × op(A) × op(B) + beta × C`. Flags bit 0 = transpose A, bit 1 = transpose B. All use zero heap allocation — computation is entirely in ZA tiles and a 128-byte stack frame.

| Instruction | Parameters | Inputs | Accumulator | K alignment |
|-------------|-----------|--------|-------------|-------------|
| `cblas_sgemm` | `flags:u8`, `M:u32`, `N:u32`, `K:u32`, `lda:u32`, `ldb:u32`, `ldc:u32`, `alpha:f32`, `beta:f32`, `A:u64`, `B:u64`, `C:u64` | fp32 × fp32 | fp32 (FMOPA) | K % 16 == 0 |
| `cblas_bfgemm` | *(same encoding)* | bf16 × bf16 | fp32 (BFMOPA) | K % 32 == 0 |
| `cblas_igemm` | *(same encoding)* | i8 × i8 | i32→fp32 (SMOPA) | K % 64 == 0 |
| `cblas_ugemm` | *(same encoding)* | u8 × u8 | i32→fp32 (UMOPA) | K % 64 == 0 |
| `cblas_usgemm` | *(same encoding)* | u8 × i8 | i32→fp32 (USMOPA) | K % 64 == 0 |

All overwrite v0-v21 and the scratchpad.

For `bfgemm`: A and B are bf16 arrays, C is fp32. lda/ldb are in bf16 elements, ldc in fp32 elements.

For `igemm`/`ugemm`/`usgemm`: A and B are i8/u8 arrays, C is fp32. lda/ldb are in byte elements, ldc in fp32 elements. The int32 accumulator is converted to fp32 and multiplied by alpha before storing.

**DSL usage:**
```
sgemm(0, 128, 256, 64, 64, 256, 256, 1.0, 0.0, params[0], params[1], params[2]);
bfgemm(0, 128, 256, 64, 64, 256, 256, 1.0, 0.0, params[0], params[1], params[2]);
igemm(0, 128, 256, 128, 128, 256, 256, 0.00390625, 0.0, params[0], params[1], params[2]);
```

### Tile-Range GEMM (Multi-Threaded / Multi-Node)

Computes only a subset of output tiles from a fp32 GEMM. Same inner kernel as `cblas_sgemm`, but the outer tile loop is bounded by `(ti_start, tj_start, ti_count, tj_count)`. Assign different tile ranges to different threads or nodes.

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `gemm_tile_fp32` | `flags:u8`, `M:u32`, `N:u32`, `K:u32`, `lda:u32`, `ldb:u32`, `ldc:u32`, `alpha:f32`, `beta:f32`, `ti_start:u32`, `tj_start:u32`, `ti_count:u32`, `tj_count:u32`, `A:u64`, `B:u64`, `C:u64` | C[ti..ti+ti_count, tj..tj+tj_count] = alpha×op(A)×op(B) + beta×C | v0-v21, scratchpad |

M and N are the **full** matrix dimensions (used for edge predication on the last tile). ti_start should be a multiple of 16, tj_start a multiple of 32.

**Multi-threaded example** (C++, 4 threads each computing a quarter of the rows):
```cpp
// Thread i computes rows [i*quarter .. (i+1)*quarter)
uint32_t quarter = (M + 3) / 4;
for (int t = 0; t < 4; t++) {
    // Each thread builds its own program:
    ane::program p;
    p.emit(ane::Op::gemm_tile_fp32,
        uint8_t(0), M, N, K, lda, ldb, ldc, 1.0f, 0.0f,
        uint32_t(t * quarter),  // ti_start
        uint32_t(0),            // tj_start
        uint32_t(quarter),      // ti_count
        uint32_t(N),            // tj_count (all columns)
        ptr_A, ptr_B, ptr_C);
    p.exec();  // each thread runs in its own smstart/smstop session
}
```

**DSL usage:**
```
gemm_tile(0, 1024, 1024, 512, 512, 1024, 1024, 1.0, 0.0, 0, 0, 256, 1024, params[0], params[1], params[2]);
```

### Decomposition Building Blocks

These ops split reductions into partial computations that can be distributed across threads or nodes, then merged.

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `softmax_partial_fp32` | `dim:u32`, `in:u64`, `out:u64`, `max_out:u64`, `sum_out:u64` | out = exp(in - max), stores max and sum(exp) scalars | v0-v7, v16-v19, v26-v31 |
| `softmax_correct_fp32` | `dim:u32`, `local_max:f32`, `global_max:f32`, `inout:u64`, `sum_ptr:u64` | inout *= exp(local_max - global_max), corrects sum | v0, v16, v28-v31 |
| `reduce_sum_sq_fp32` | `dim:u32`, `in:u64`, `out:u64` | out = sum(in[i]²) | v0-v1 |
| `reduce_col_sum_fp32` | `M:u32`, `N:u32`, `stride:u32`, `src:u64`, `dst:u64` | dst[j] = sum(src[i*stride+j]) — bias gradient | v0-v1 |

dim and N must be multiples of 16 (cntw).

**Distributed softmax pattern** (across N shards):

```
// Step 1: Each shard computes partial softmax
softmax_partial(shard_dim, params[0], params[1], params[2], params[3]);
// params[2] = &local_max, params[3] = &local_sum

// Step 2: All-reduce to find global_max = max(all local_max values)
//         (done by your communication layer, not by ANE)

// Step 3: Each shard applies correction
softmax_correct(shard_dim, local_max, global_max, params[1], params[3]);

// Step 4: All-reduce to find global_sum = sum(all corrected local_sum values)

// Step 5: Each shard divides by global_sum
//         Use elementwise_mul with scale = 1.0/global_sum
elementwise_mul(shard_dim, params[1], params[4], params[1]);
// where params[4] points to a buffer filled with 1.0/global_sum
```

**Distributed RMS norm pattern:**

```
// Step 1: Each shard computes partial sum of squares
reduce_sum_sq(shard_dim, params[0], params[1]);

// Step 2: All-reduce sum across shards, divide by total_dim, add eps
//         scale = rsqrt(global_sum_sq / total_dim + eps)
//         (done on CPU or with frsqrt_zreg)

// Step 3: Each shard applies: out[i] = in[i] * scale * weight[i]
//         Use existing rms_norm_fp32 with pre-computed scale, or compose
//         from fscale_zreg + fmul_zreg in a loop
```

### Dot Product & Wide FMA

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `fdot_zreg` | `width:u8` | Dot product with jump-point per width. See below. | v0 |
| `fmla_wide_zreg` | `width:u8` | Multi-product accumulate into v0. See below. | v0 |

**`fdot_zreg`** computes a scalar dot product and broadcasts the result to all 16 lanes of v0:

| Width | Input A | Input B | Formula |
|-------|---------|---------|---------|
| 1 | v0 | v1 | `v0 = broadcast(sum(v0[i] × v1[i]))` — 16 products |
| 2 | v0, v1 | v2, v3 | `v0 = broadcast(sum(v0·v2) + sum(v1·v3))` — 32 products |
| 4 | v0–v3 | v4–v7 | `v0 = broadcast(sum(v0·v4) + ... + sum(v3·v7))` — 64 products |

The width selects a different code path inside the handler — no extra dispatch overhead.

**`fmla_wide_zreg`** accumulates multiple element-wise products into v0 (no horizontal sum):

| Width | Formula |
|-------|---------|
| 1 | `v0 += v1 × v2` |
| 2 | `v0 += v1×v3 + v2×v4` |
| 4 | `v0 += v1×v5 + v2×v6 + v3×v7 + v4×v8` |

Use `fmla_wide` in a loop, then `faddv_zreg` at the end to reduce to scalar. This is the building block for GEMV: load input chunks into v1–v4, dequantized weights into v5–v8, accumulate with width=4.

### Clamp / Max / Min

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `fclamp_zreg` | `flags:u8`, `dst:u8`, `src:u8`, `lo:f32`, `hi:f32` | Clamp, max, or min per element | dst, v0-v1 |

flags: 3 = clamp(lo..hi), 1 = max(x, lo) (ReLU when lo=0), 2 = min(x, hi).

### Scalar Operations

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `broadcast_scalar_zreg` | `dst:u8`, `value:f32` | Fill all 16 lanes of dst with value | dst |
| `fscale_zreg` | `dst:u8`, `src:u8`, `scalar:f32` | dst = src × scalar | dst, v0 |
| `faddv_zreg` | `dst:u8`, `src:u8` | Horizontal sum: dst = broadcast(sum(src)) | dst |
| `frsqrt_zreg` | `dst:u8`, `src:u8` | dst[i] = 1/sqrt(src[i]) | dst, v0 |

### Transformer Kernels

These process entire arrays from memory — designed for LLM inference and training.

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `rms_norm_fp32` | `dim:u32`, `eps:f32`, `in:u64`, `weight:u64`, `out:u64` | RMS normalization per row | v0-v4, v16 |
| `layer_norm_fp32` | `dim:u32`, `eps:f32`, `in:u64`, `gamma:u64`, `beta:u64`, `out:u64` | Full layer normalization (mean subtraction + variance) | v0-v4, v16-v17 |
| `softmax_fp32` | `dim:u32`, `in:u64`, `out:u64` | Numerically stable softmax | v0-v7, v16-v19 |
| `silu_fp32` | `count:u32`, `in:u64`, `out:u64` | SiLU activation: x·sigmoid(x) | v0-v4, v16-v19 |
| `gelu_fp32` | `count:u32`, `in:u64`, `out:u64` | GeLU activation (tanh approximation) | v0-v7, v16-v19 |
| `rope_fp32` | `dim:u32`, `pos:u32`, `theta:f32`, `in:u64`, `out:u64` | Rotary position embedding | v0-v7, v20-v31 |
| `causal_mask_fp32` | `dim:u32`, `stride:u32`, `ptr:u64` | Fill upper triangle with -inf for causal attention | v0, v16 |

`layer_norm_fp32` computes full layer normalization: `out = ((x - mean) / sqrt(var + eps)) * gamma + beta`. Three passes over the data: sum for mean, sum-of-squared-deviations for variance, then normalize+scale+shift. For Llama-family models, use `rms_norm_fp32` instead.

`gelu_fp32` uses the tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.

`causal_mask_fp32` sets `ptr[i * stride + j] = -inf` for all `j > i`. Apply this to attention scores before softmax for autoregressive (decoder) models. dim is both the row and column count (square matrix). stride is the row stride in fp32 elements.

### Backward Passes

Training backward kernels for the operations above. These compute gradients with respect to the inputs.

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `silu_backward_fp32` | `dim:u32`, `x:u64`, `dy:u64`, `dx:u64` | dx = dy · σ(x) · (1 + x·(1-σ(x))) | v0-v7, v26-v31 |
| `softmax_backward_fp32` | `dim:u32`, `s:u64`, `dy:u64`, `dx:u64` | dx = s · (dy - sum(s·dy)) | v0-v1, v16 |

`silu_backward_fp32` takes the forward input `x` (not the forward output), the upstream gradient `dy`, and writes `dx`. It recomputes sigmoid internally.

`softmax_backward_fp32` takes the forward softmax output `s`, the upstream gradient `dy`, and writes `dx`. Two passes: first computes `dot = sum(s·dy)`, then `dx = s·(dy - dot)`.

dim must be a multiple of 16 for both.

### Optimizer

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `adam_step_fp32` | `count:u32`, `lr:f32`, `beta1:f32`, `beta2:f32`, `eps:f32`, `t:u32`, `params:u64`, `grads:u64`, `m:u64`, `v:u64` | Fused Adam update step | v0-v6, v16-v22 |

Single pass over all four arrays. Updates `m` and `v` in-place, then applies the bias-corrected update to `params`. The timestep `t` is 1-based (first call uses t=1). Bias correction factors `(1-β1^t)` and `(1-β2^t)` are pre-computed before the loop.

**DSL usage:**
```
adam_step(4096, 0.001, 0.9, 0.999, 0.00000001, 1, params[0], params[1], params[2], params[3]);
```

### Flash Attention

| Instruction | Parameters | What it does | Overwrites |
|-------------|-----------|--------------|------------|
| `flash_attention_fp32` | `N:u32`, `d:u32`, `flags:u8`, `Q:u64`, `K:u64`, `V:u64`, `out:u64` | out = softmax(Q@K^T/sqrt(d)) @ V | all ZA tiles, v0-v24 |

Fused tiled attention using the FlashAttention online softmax algorithm. Processes one attention head at a time. Tiles the Q@K^T score matrix into 16×16 blocks — never materializes the full N×N score matrix.

- `N` = sequence length (must be multiple of 16)
- `d` = head dimension (must be multiple of 16)
- `flags` bit 0 = causal mask (mask where key position > query position)
- Q, K, V: N×d fp32 row-major matrices
- out: N×d fp32 output (zeroed internally before computation)

For multi-head attention, call once per head with offset pointers. For grouped-query attention, share K/V pointers across the query heads in each group.

**DSL usage:**
```
// Single head, seq_len=256, head_dim=64, causal
flash_attention(256, 64, 1, params[0], params[1], params[2], params[3]);
```

**Multi-head attention (C++):**
```cpp
for (int h = 0; h < n_heads; h++) {
    size_t offset = h * seq_len * head_dim * sizeof(float);
    ane::program attn;
    attn.emit(ane::Op::flash_attention_fp32,
        uint32_t(seq_len), uint32_t(head_dim), uint8_t(1),  // causal
        reinterpret_cast<uintptr_t>(Q + offset),
        reinterpret_cast<uintptr_t>(K + offset),
        reinterpret_cast<uintptr_t>(V + offset),
        reinterpret_cast<uintptr_t>(out + offset));
    attn.exec();
}
```

### Quantized GEMV

All quantized GEMV ops share the same encoding: `[opcode][M:u32][K:u32][in:u64][W:u64][out:u64]`. `in` is fp32, `W` is the quantized weight matrix in the specified block format, `out` is fp32. Computes `output[m] = sum_k dequant(W[m,k]) * input[k]` for all M rows.

| Instruction | Parameters | Block Size | Bits/Weight | K alignment | What it does |
|-------------|-----------|------------|-------------|-------------|--------------|
| `q8_0_gemv` | `M:u32`, `K:u32`, `in:u64`, `W:u64`, `out:u64` | 34 bytes/32 vals | 8.5 | K % 32 == 0 | fp16 scale + 32×int8 |
| `q4_0_gemv` | *(same)* | 18 bytes/32 vals | 4.5 | K % 32 == 0 | fp16 scale + 16 packed bytes (4-bit nibbles) |
| `q4_k_gemv` | *(same)* | 144 bytes/256 vals | 4.5 | K % 256 == 0 | Super-block: fp16 d/dmin + 6-bit sub-block scales + 4-bit quants |
| `q2_k_gemv` | *(same)* | 84 bytes/256 vals | 2.625 | K % 256 == 0 | Super-block: 4-bit sub-block scales + 2-bit quants |
| `q3_k_gemv` | *(same)* | 110 bytes/256 vals | 3.4375 | K % 256 == 0 | Super-block: 2-bit low + 1-bit high plane + 6-bit signed scales |
| `q5_k_gemv` | *(same)* | 176 bytes/256 vals | 5.5 | K % 256 == 0 | Super-block: 4-bit low + 1-bit high plane + 6-bit scales |
| `q6_k_gemv` | *(same)* | 210 bytes/256 vals | 6.5625 | K % 256 == 0 | Super-block: 4-bit low + 2-bit high + int8 scales |

The K-quant formats (Q2_K through Q6_K) use **super-blocks** of 256 values with per-sub-block scales for higher quality at low bit widths. The dequantization uses vectorized bit extraction via SVE shifts and masks — each sub-block is processed as 16 elements at a time.

**DSL usage:**
```
q4_k_gemv(128, 4096, params[0], params[1], params[2]);
// 128 output rows, K=4096, fp32 input, Q4_K weights, fp32 output
```

### Embedding Lookup (GET_ROWS)

| Instruction | Parameters | Table Format | What it does |
|-------------|-----------|-------------|--------------|
| `get_rows_fp32` | `n_rows:u32`, `dim:u32`, `table:u64`, `indices:u64`, `out:u64` | fp32 | Copy rows by index |
| `get_rows_q8_0` | *(same)* | Q8_0 | Lookup + dequant to fp32 |
| `get_rows_q4_0` | *(same)* | Q4_0 | Lookup + dequant to fp32 |

Gather `n_rows` rows from an embedding table by integer index. `indices` is a `uint32_t` array of row indices. Each selected row (`dim` elements) is dequantized (if quantized) and written as fp32 to the output.

- `dim` must be a multiple of 16 for fp32, 32 for Q8_0/Q4_0
- The table can be arbitrarily large (vocab_size × dim); only the indexed rows are touched
- Output is contiguous: `n_rows × dim` fp32 values

**DSL usage:**
```
// Look up 4 token embeddings from a Q4_0 table with dim=4096
get_rows_q4_0(4, 4096, params[0], params[1], params[2]);
```

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
| `reduce_col_sum_fp32` | `M:u32`, `N:u32`, `stride:u32`, `src:u64`, `dst:u64` | dst[j] = sum(src[i*stride + j]) for i in 0..M-1 | v0-v1 |
| `count_matches` | `count:u32`, `pred:u64`, `labels:u64`, `out:u64` | out = number of matches | v0-v1 |

`reduce_col_sum_fp32` computes the column-wise sum of an M×N matrix — the bias gradient operation in dense backward passes. N must be a multiple of 16. The stride parameter is the row stride in fp32 elements (typically equal to the full matrix width, which may be larger than N when operating on a column subrange).

**DSL usage:**
```
// Sum 256 rows of 128 columns with row stride 512
reduce_col_sum(256, 128, 512, params[0], params[1]);
```

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
