# ANE Bytecode Instruction Reference

Every instruction runs inside a single streaming SVE session (`smstart`/`smstop`). Z-registers and ZA tiles persist across instructions within the same `program::exec()` call.

**Register conventions:**
- `z0` is the "mailbox" register for `load`/`store`/`load_param`/`store_param`
- `z0-z1` are scratch for trampoline-based arithmetic ops
- `z2+` are safe for user variables (via `mov_zreg`)
- `ZA` (za0.s-za3.s / za0.d-za7.d) persists unless explicitly zeroed or written
- `ZT0` persists unless a LUT operation reloads it
- `x20` is the loop counter register

---

## Composable Primitives

### `load` (0x1F)
Load one z-vector (VL bytes) from memory into `z0`.
```
Encoding: [0x1F][ptr:u64]
Clobbers: z0
Preserves: z1-z31, ZA, ZT0
```

### `store` (0x20)
Store one z-vector from `z0` to memory.
```
Encoding: [0x20][ptr:u64]
Clobbers: nothing
Reads: z0
Preserves: z0-z31, ZA, ZT0
```

### `mov_zreg` (0x34)
Copy z{src} to z{dst}. Uses stack relay via trampoline tables.
```
Encoding: [0x34][src:u8][dst:u8]
Clobbers: z{dst} (obviously), stack temp (restored)
Preserves: all other z-regs, ZA, ZT0
```

### `set_param` (0x37)
Store a pointer into param table slot [0-7].
```
Encoding: [0x37][idx:u8][ptr:u64]
Clobbers: nothing (writes to stack param table at sp+128)
Preserves: z0-z31, ZA, ZT0
```

### `load_param` (0x38)
Load one z-vector from `param[idx]` pointer into `z0`.
```
Encoding: [0x38][idx:u8]
Clobbers: z0
Preserves: z1-z31, ZA, ZT0
```

### `store_param` (0x39)
Store `z0` to `param[idx]` pointer.
```
Encoding: [0x39][idx:u8]
Clobbers: nothing
Reads: z0
Preserves: z0-z31, ZA, ZT0
```

### `advance_param` (0x3A)
Advance `param[idx]` pointer by VL bytes (64 on M4/M5).
```
Encoding: [0x3A][idx:u8]
Clobbers: nothing (modifies param table entry)
Preserves: z0-z31, ZA, ZT0
```

### `loop_begin` (0x35)
Set the loop counter register to `count`.
```
Encoding: [0x35][count:u8]
Clobbers: x20 (loop counter)
Preserves: z0-z31, ZA, ZT0
Max iterations: 255
```

### `loop_end` (0x36)
Decrement loop counter. If nonzero, rewind instruction pointer by `offset` bytes.
```
Encoding: [0x36][offset:u16]
Clobbers: x20 (decremented)
Preserves: z0-z31, ZA, ZT0
Note: offset is computed by program::end_loop() or the DSL compiler
```

---

## Register-Addressed Arithmetic (float32)

All use the trampoline tables. Operands are loaded into z0/z1 via stack relay, result stored back via stack relay. **Only z{dst} is modified; all other z-regs survive.**

### `fadd_zreg` (0x3B)
`z{dst}.s = z{src1}.s + z{src2}.s`
```
Encoding: [0x3B][dst:u8][src1:u8][src2:u8]
Clobbers: z{dst}, z0-z1 (scratch, restored via trampoline)
Preserves: all z-regs except z{dst}, ZA, ZT0
```

### `fsub_zreg` (0x3C)
`z{dst}.s = z{src1}.s - z{src2}.s`
```
Encoding: [0x3C][dst:u8][src1:u8][src2:u8]
Clobbers: z{dst}, z0-z1 (scratch)
Preserves: all z-regs except z{dst}, ZA, ZT0
```

### `fmul_zreg` (0x3D)
`z{dst}.s = z{src1}.s * z{src2}.s`
```
Encoding: [0x3D][dst:u8][src1:u8][src2:u8]
Clobbers: z{dst}, z0-z1 (scratch)
Preserves: all z-regs except z{dst}, ZA, ZT0
```

### `fmla_zreg` (0x3E)
`z{dst}.s += z{src1}.s * z{src2}.s` (fused multiply-accumulate)
```
Encoding: [0x3E][dst:u8][src1:u8][src2:u8]
Clobbers: z{dst}, z0-z2 (scratch)
Preserves: all z-regs except z{dst}, ZA, ZT0
Note: dst is both input (accumulator) and output
```

---

## Register-Addressed Bitwise Operations

Operate on full 512-bit vectors. Lane width is irrelevant for bitwise ops.

### `and_zreg` (0x3F)
`z{dst} = z{src1} AND z{src2}`
```
Encoding: [0x3F][dst:u8][src1:u8][src2:u8]
Clobbers: z{dst}, z0-z1 (scratch)
Preserves: all z-regs except z{dst}, ZA, ZT0
```

### `orr_zreg` (0x40)
`z{dst} = z{src1} OR z{src2}`
```
Encoding: [0x40][dst:u8][src1:u8][src2:u8]
Clobbers: z{dst}, z0-z1 (scratch)
Preserves: all z-regs except z{dst}, ZA, ZT0
```

### `eor_zreg` (0x41)
`z{dst} = z{src1} XOR z{src2}`
```
Encoding: [0x41][dst:u8][src1:u8][src2:u8]
Clobbers: z{dst}, z0-z1 (scratch)
Preserves: all z-regs except z{dst}, ZA, ZT0
```

### `not_zreg` (0x42)
`z{dst} = NOT z{src}`
```
Encoding: [0x42][dst:u8][src:u8]
Clobbers: z{dst}, z0 (scratch)
Preserves: all z-regs except z{dst}, ZA, ZT0
```

---

## ZA Tile Operations

### `zero_za` (0x01)
Zero all ZA tiles.
```
Encoding: [0x01]
Clobbers: ZA (za0.s-za3.s / za0.d-za7.d all zeroed)
Preserves: z0-z31, ZT0
```

### `store_tiles` (0x06)
Store the 2x2 tile group (za0-za3.s) as row-major int32 to memory.
```
Encoding: [0x06][dst_ptr:u64]
Clobbers: z4-z11 (extraction temporaries)
Reads: ZA (za0.s-za3.s)
Preserves: z0-z3, z12-z31, ZT0
Output: 32x32 int32 matrix (4096 bytes on M4/M5)
```

### `load_bias` (0x0A)
Load int32 data from memory into ZA tiles (reverse of store_tiles).
```
Encoding: [0x0A][src_ptr:u64]
Clobbers: z4-z11 (load temporaries), ZA (za0.s-za3.s written)
Preserves: z0-z3, z12-z31, ZT0
```

### `scale_store` (0x0B)
Extract ZA tiles, convert int32 to float32, multiply by scale, store.
```
Encoding: [0x0B][scale:f32][dst_ptr:u64]
Clobbers: z4-z11, z16 (scale broadcast)
Reads: ZA (za0.s-za3.s)
Preserves: z0-z3, z12-z15, z17-z31, ZT0
```

---

## Fused Accumulator Kernels

**WARNING: These kernels clobber z0-z3 and accumulate into ZA. Use `zero_za` or `load_bias` to initialize ZA before calling.**

### `acc_smopa` (0x02)
Fused load + signed int8 outer product loop.
```
Encoding: [0x02][k_steps:u32][row_ptr:u64][col_ptr:u64]
Clobbers: z0-z3 (loaded per iteration)
Writes: ZA (za0.s-za3.s accumulated)
Preserves: z4-z31, ZT0
Each k_step: loads 2 z-vecs of rows + 2 z-vecs of cols, 4x smopa
```

### `acc_umopa` (0x03)
Same as acc_smopa but unsigned int8.
```
Encoding: [0x03][k_steps:u32][row_ptr:u64][col_ptr:u64]
Clobbers: z0-z3
Writes: ZA (accumulated)
```

### `acc_usmopa` (0x04)
Unsigned rows x signed cols.
```
Encoding: [0x04][k_steps:u32][row_ptr:u64][col_ptr:u64]
Clobbers: z0-z3
Writes: ZA (accumulated)
```

### `acc_sumopa` (0x05)
Signed rows x unsigned cols (via usmopa with swapped operands — tiles are transposed).
```
Encoding: [0x05][k_steps:u32][row_ptr:u64][col_ptr:u64]
Clobbers: z0-z3
Writes: ZA (accumulated, TRANSPOSED layout)
```

### `smopa_2x2` (0x07)
4x smopa on pre-loaded z0-z3 (no memory access). Use after `load_rows_i8`/`load_cols_i8`.
```
Encoding: [0x07]
Reads: z0-z3
Writes: ZA (accumulated)
Preserves: z0-z31 (reads only), ZT0
```

### `umopa_2x2` (0x08)
Same as smopa_2x2 but unsigned.
```
Encoding: [0x08]
Reads: z0-z3
Writes: ZA (accumulated)
```

### `usmopa_2x2` (0x09)
Same as smopa_2x2 but unsigned x signed.
```
Encoding: [0x09]
Reads: z0-z3
Writes: ZA (accumulated)
```

---

## Fused Elementwise Kernels (Memory-to-Memory)

**WARNING: These operate on memory pointers with a count. They clobber z0-z1 and z16 (for scaled ops). They do NOT touch ZA or user z-regs above z1.**

### `elementwise_add_fp32` (0x0C)
`out[i] = a[i] + b[i]`
```
Encoding: [0x0C][count:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
Clobbers: z0-z1
Preserves: z2-z31, ZA, ZT0
count = number of float32 elements (must be multiple of SVLs=16)
```

### `elementwise_scaled_add_fp32` (0x0D)
`out[i] = a[i] + scale * b[i]`
```
Encoding: [0x0D][count:u32][scale:f32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
Clobbers: z0-z1, z16 (scale broadcast)
Preserves: z2-z15, z17-z31, ZA, ZT0
```

### `elementwise_mul_fp32` (0x0E)
`out[i] = a[i] * b[i]`
```
Encoding: [0x0E][count:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
Clobbers: z0-z1
Preserves: z2-z31, ZA, ZT0
```

### `relu_backward_fp32` (0x0F)
`out[i] = (hidden[i] > 0) ? grad[i] : 0`
```
Encoding: [0x0F][count:u32][hidden_ptr:u64][grad_ptr:u64][out_ptr:u64]
Clobbers: z0-z2, z17 (zero constant)
Preserves: z3-z16, z18-z31, ZA, ZT0
```

---

## Fused LUT Kernels (Memory-to-Memory)

**WARNING: These load ZT0 from the table pointer. ZT0 is clobbered.**

### `luti4_op` (0x13)
4-bit table lookup via ZT0.
```
Encoding: [0x13][count:u32][elem_size:u8][table_ptr:u64][indices_ptr:u64][output_ptr:u64]
elem_size: 0=byte(.b), 1=halfword(.h), 2=word(.s)
Clobbers: z0-z1, ZT0
Preserves: z2-z31, ZA
count = number of z-vectors to process
```

### `luti2_op` (0x14)
2-bit table lookup via ZT0.
```
Encoding: [0x14][count:u32][elem_size:u8][table_ptr:u64][indices_ptr:u64][output_ptr:u64]
elem_size: 0=byte(.b), 1=halfword(.h), 2=word(.s)
Clobbers: z0-z1, ZT0
Preserves: z2-z31, ZA
```

---

## Fused Dense/Matmul Kernels

**WARNING: These are heavyweight. They use large stack frames, clobber many z-regs, and write ZA.**

### `dense_fp32` (0x15)
Full fp32 matmul via FMOPA with optional ReLU.
```
Encoding: [0x15][M:u32][N:u32][K:u32][scale:f32][flags:u8]
         [A_ptr:u64][B_ptr:u64][bias_ptr:u64][C_ptr:u64]
flags: bit 0 = apply ReLU
Clobbers: z0-z21, ZA (za0-za2), 128-byte stack frame
Preserves: z22-z31, ZT0
```

### `dense_i8` (0x18)
INT8 matmul via SMOPA with dequantization.
```
Encoding: [0x18][M:u32][N:u32][K:u32][scale_a:f32][scale_b:f32][flags:u8]
         [A_i8_ptr:u64][B_packed_ptr:u64][bias_ptr:u64][C_ptr:u64]
flags: bit 0 = apply ReLU
Clobbers: z0-z21, ZA (za0-za2), 96-byte stack frame
Preserves: z22-z31, ZT0
Note: B must be in SMOPA panel format (use pack_b_i8 first)
```

### `dense_u8s8` (0x1E)
UINT8 x INT8 matmul via USMOPA.
```
Encoding: [0x1E][M:u32][N:u32][K:u32][scale_a:f32][scale_b:f32][flags:u8]
         [A_u8_ptr:u64][B_packed_ptr:u64][bias_ptr:u64][C_ptr:u64]
Clobbers: z0-z21, ZA, 96-byte stack frame
Preserves: z22-z31, ZT0
```

---

## Fused Distance/Reduction Kernels

### `l2_squared_fp32` (0x21)
`result = sum((a[i] - b[i])^2)` for float32.
```
Encoding: [0x21][dim:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
Clobbers: z0-z1, z4 (accumulator)
Preserves: z2-z3, z5-z31, ZA, ZT0
out_ptr receives one float32 scalar
```

### `l2_squared_bf16` (0x22)
Same for bfloat16 inputs, fp32 accumulation.
```
Encoding: [0x22][dim:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
Clobbers: z0-z7
Preserves: z8-z31, ZA, ZT0
```

### `l2_squared_f64` (0x23)
Same for float64.
```
Encoding: [0x23][dim:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
Clobbers: z0-z1, z4
Preserves: z2-z3, z5-z31, ZA, ZT0
```

### `cosine_dist_fp32` (0x24)
`result = 1 - dot(a,b) / (||a|| * ||b||)` for float32.
```
Encoding: [0x24][dim:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
Clobbers: z0-z1, z4-z6 (three accumulators)
Preserves: z2-z3, z7-z31, ZA, ZT0
```

### `cosine_dist_bf16` (0x25)
Same for bfloat16 inputs.
```
Encoding: [0x25][dim:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
Clobbers: z0-z11
Preserves: z12-z31, ZA, ZT0
```

### `cosine_dist_f64` (0x26)
Same for float64.
```
Encoding: [0x26][dim:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
Clobbers: z0-z1, z4-z6
Preserves: z2-z3, z7-z31, ZA, ZT0
```

### `normalize_fp32` (0x27)
In-place normalize to unit length: `vec[i] /= ||vec||`.
```
Encoding: [0x27][dim:u32][vec_ptr:u64]
Clobbers: z0, z4 (norm accum), z16 (inv_norm broadcast)
Preserves: z1-z3, z5-z15, z17-z31, ZA, ZT0
Two-pass: compute ||v||, then multiply by 1/||v||
```

### `reduce_sum_fp32` (0x17)
Horizontal sum of float32 array.
```
Encoding: [0x17][count:u32][src_ptr:u64][out_ptr:u64]
Clobbers: z0-z1
Preserves: z2-z31, ZA, ZT0
```

### `count_matches` (0x16)
Count matching predictions vs labels.
```
Encoding: [0x16][count:u32][pred_ptr:u64][labels_ptr:u64][result_ptr:u64]
Clobbers: z0-z1
Preserves: z2-z31, ZA, ZT0
```

---

## Fused Transform Kernels

### `dct2_forward_fp32` (0x28)
H.264 4-point integer butterfly DCT-II forward.
```
Encoding: [0x28][dim:u32][src_ptr:u64][dst_ptr:u64]
Clobbers: z0-z8
Preserves: z9-z31, ZA, ZT0
dim must be multiple of 4 (and multiple of SVLs for vectorization)
```

### `dct2_inverse_fp32` (0x29)
Inverse of the above.
```
Encoding: [0x29][dim:u32][src_ptr:u64][dst_ptr:u64]
Clobbers: z0-z8
Preserves: z9-z31, ZA, ZT0
```

### `transpose_fp32` (0x11)
Transpose M x N float32 matrix.
```
Encoding: [0x11][M:u32][N:u32][src_ptr:u64][dst_ptr:u64]
Clobbers: s0 (scalar temp)
Preserves: z1-z31, ZA, ZT0
Note: scalar loop, not vectorized
```

### `transpose_i8` (0x1D)
Transpose M x N int8 matrix.
```
Encoding: [0x1D][M:u32][N:u32][src_ptr:u64][dst_ptr:u64]
Clobbers: z0-z7, ZA (za0 used as temp)
Preserves: z8-z31, ZT0
```

---

## Fused Threshold/Bitmap Kernels

### `threshold_bitmap_fp32` (0x2A)
Compare float32 > threshold, produce packed bitmap.
```
Encoding: [0x2A][dim:u32][threshold:f32][src_ptr:u64][bitmap_out:u64]
Clobbers: z0-z1, z16-z17
Preserves: z2-z15, z18-z31, ZA, ZT0
```

### `threshold_8bit` (0x2D)
Reconstruct 8-bit counters from 8 bitplanes, threshold to bitmap.
```
Encoding: [0x2D][n_bytes:u32][threshold:u8][src_ptr:u64][bitmap_out:u64]
Clobbers: scalar x-regs only (GP register bit manipulation)
Preserves: z0-z31, ZA, ZT0
```

### `bitmap_score_pipeline` (0x33)
Full pipeline: ripple-carry accumulate N bitmap streams, threshold, extract candidate IDs.
```
Encoding: [0x33][n_streams:u32][n_bytes:u32][n_vectors:u32][score_min:u32]
         [max_candidates:u32][streams_ptr:u64][is_high_ptr:u64]
         [candidates_out:u64][count_out:u64]
Clobbers: z0-z7 (bitplane ripple carry), large stack frame
Preserves: z8-z31, ZT0
Note: ZA may be used internally
```

---

## Fused Statistics/Quantization Kernels

### `welford_stats_fp32` (0x2B)
Online Welford mean/stddev/maxabs/scale across n_vectors of dim.
```
Encoding: [0x2B][n_vectors:u32][dim:u32][src_ptr:u64][stats_out:u64]
Clobbers: z0-z7 (accumulation in f64), 48-byte stack frame
Preserves: z8-z31, ZT0
stats_out: 4 x dim doubles (mean, stddev, maxabs, scale)
All accumulation in double precision
```

### `quantize_fp32_i8` (0x19)
Per-tensor symmetric quantize float32 to signed int8.
```
Encoding: [0x19][count:u32][src_ptr:u64][dst_ptr:u64][scale_out:u64]
Clobbers: z0, z16-z19
Preserves: z1-z15, z20-z31, ZA, ZT0
```

### `quantize_fp32_i8_channelwise` (0x1C)
Per-row symmetric quantize float32 to int8.
```
Encoding: [0x1C][M:u32][N:u32][src_ptr:u64][dst_ptr:u64][scales_out:u64]
Clobbers: z0, z16-z19
Preserves: z1-z15, z20-z31, ZA, ZT0
```

### `dequantize_i8_fp32` (0x1A)
Dequantize signed int8 to float32.
```
Encoding: [0x1A][count:u32][scale:f32][src_ptr:u64][dst_ptr:u64]
Clobbers: z0, z16 (scale broadcast)
Preserves: z1-z15, z17-z31, ZA, ZT0
```

### `quantize_pack_4bit_fp32` (0x2C)
Quantize float32 to signed 4-bit SoA packed nibbles (dual source).
```
Encoding: [0x2C][n:u32][dim:u32][src_ptr:u64][stats_ptr:u64]
         [raw_out:u64][dct_src:u64][dct_out:u64]
Clobbers: scalar regs, 80-byte stack frame
Preserves: z0-z31 (scalar processing), ZA, ZT0
```

### `quantize_accum_2bit` (0x2E)
2-bit ternary decode, scale by bf16, accumulate into bf16.
```
Encoding: [0x2E][count:u32][packed_ptr:u64][scale_ptr:u64][accum_ptr:u64]
Clobbers: z0-z5 (decode + arithmetic)
Preserves: z6-z31, ZA, ZT0
```

### `accum_8bit` (0x2F)
INT8 scale-accumulate: `accum[i] += data[i] * scale[i]` in bf16.
```
Encoding: [0x2F][count:u32][data_ptr:u64][scale_ptr:u64][accum_ptr:u64]
Clobbers: z0-z6
Preserves: z7-z31, ZA, ZT0
```

### `pack_b_i8` (0x1B)
Pack K x N row-major int8 into SMOPA panel format.
```
Encoding: [0x1B][K:u32][N:u32][src_ptr:u64][dst_ptr:u64]
Clobbers: z0-z3
Preserves: z4-z31, ZA, ZT0
```

---

## Fused SoA Accumulator Kernels

### `soa_sub_scale_bf16` (0x30)
`accum[i] += bf16((src[i]*scale - scalar)^2)` for L2-like SoA distance.
```
Encoding: [0x30][count:u32][src_ptr:u64][scalar:f32][scale:f32][accum_ptr:u64]
Clobbers: z0-z3, z16-z17
Preserves: z4-z15, z18-z31, ZA, ZT0
```

### `soa_luti2_accum` (0x31)
LUTI2 expand 2-bit indices via ZT0, accumulate into bf16.
```
Encoding: [0x31][count:u32][packed_ptr:u64][table_ptr:u64][accum_ptr:u64]
Clobbers: z0-z5, ZT0
Preserves: z6-z31, ZA
```

### `soa_luti4_accum` (0x32)
LUTI4 expand 4-bit indices via ZT0, accumulate into bf16.
```
Encoding: [0x32][count:u32][packed_ptr:u64][table_ptr:u64][accum_ptr:u64]
Clobbers: z0-z5, ZT0
Preserves: z6-z31, ZA
```

---

## Softmax/Special

### `softmax_argmax_fp32` (0x12)
Batched softmax + cross-entropy backward + argmax.
```
Encoding: [0x12][batch:u32][dim:u32][logits_ptr:u64][probs_ptr:u64]
         [labels_ptr:u64][grad_out_ptr:u64][argmax_ptr:u64]
Clobbers: z0-z7, z16-z31 (exp polynomial coefficients), large stack usage
Preserves: z8-z15, ZA, ZT0
WARNING: Clobbers almost all z-regs above z7
```

### `scatter_tile_fp32` (0x10)
Scatter GROUP_DIM tile to strided matrix.
```
Encoding: [0x10][M:u32][N:u32][N_stride:u32][tile_row:u32][tile_col:u32]
         [src_ptr:u64][dst_ptr:u64]
Clobbers: z0-z1
Preserves: z2-z31, ZA, ZT0
```

---

## NOT YET IMPLEMENTED (Needed)

These composable primitives are identified as gaps:

- `lsl_zreg` / `lsr_zreg` / `asr_zreg` — bitwise shifts
- `load_zt0` — load ZT0 table register without running a LUT operation
- `luti2_zreg` / `luti4_zreg` — composable LUT (result in z-reg, not memory)
- `smopa_zreg` / `umopa_zreg` / `usmopa_zreg` / `fmopa_zreg` — composable outer product accumulate
- `faddv_zreg` — horizontal reduction (sum z-vector to scalar)
- `fsqrt_zreg` / `fdiv_zreg` — square root and division
- `f64` variants of arithmetic ops — double-precision fadd/fsub/fmul/fmla
- Scalar broadcast — fill z-reg with a single value
