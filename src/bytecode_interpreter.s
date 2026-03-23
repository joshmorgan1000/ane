.text
.p2align 4
.global _stream_exec
_stream_exec:
    stp     x29, x30, [sp, #-128]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     d8,  d9,  [sp, #80]
    stp     d10, d11, [sp, #96]
    stp     d12, d13, [sp, #112]
    // ── x0 = data, x1 = size ──
    mov     x19, x0                // instruction pointer
    add     x21, x0, x1            // end = data + size
    // ── Enter streaming mode ──
    smstart
    // ── Load jump table base (must be after smstart, adrp is fine in streaming) ──
    adrp    x25, .L_jump_table@PAGE
    add     x25, x25, .L_jump_table@PAGEOFF
    // ── Dispatch ──
.L_dispatch:
    cmp     x19, x21               // IP >= bytecodes end?
    b.hs    .L_exit                  // done — no explicit halt needed
    ldrb    w9, [x19], #1          // fetch opcode, advance IP
    ldrsw   x10, [x25, x9, lsl #2] // load relative offset (32-bit signed)
    add     x10, x25, x10          // absolute target = table_base + offset
    br      x10
// ================================================================
// Jump Table (PC-relative offsets, avoids text relocations on macOS)
// ================================================================
.p2align 2
.L_jump_table:
    .long   .L_exit          - .L_jump_table  // 0x00 (reserved)
    .long   .L_op_zero_za    - .L_jump_table  // 0x01
    .long   .L_op_acc_smopa  - .L_jump_table  // 0x02
    .long   .L_op_acc_umopa  - .L_jump_table  // 0x03
    .long   .L_op_acc_usmopa - .L_jump_table  // 0x04
    .long   .L_op_acc_sumopa - .L_jump_table  // 0x05
    .long   .L_op_store_tiles - .L_jump_table // 0x06
    .long   .L_op_load_rows_i8 - .L_jump_table // 0x07
    .long   .L_op_load_cols_i8 - .L_jump_table // 0x08
    .long   .L_op_smopa_2x2  - .L_jump_table // 0x09
    .long   .L_op_umopa_2x2  - .L_jump_table // 0x0A
    .long   .L_op_usmopa_2x2 - .L_jump_table // 0x0B
    .long   .L_op_load_bias  - .L_jump_table // 0x0C
    .long   .L_op_scale_store - .L_jump_table // 0x0D
    .long   .L_op_dense_scale_relu_i8 - .L_jump_table // 0x0E
    .long   .L_op_dense_scale_i8        - .L_jump_table // 0x0F
    .long   .L_op_elementwise_add_fp32  - .L_jump_table // 0x10
    .long   .L_op_elementwise_scaled_add_fp32 - .L_jump_table // 0x11
    .long   .L_op_elementwise_mul_fp32 - .L_jump_table // 0x12
    .long   .L_op_relu_backward_fp32 - .L_jump_table // 0x13
    .long   .L_op_quantize_fp32_i8 - .L_jump_table // 0x14
    .long   .L_op_pack_rows_i8 - .L_jump_table // 0x15
    .long   .L_op_pack_cols_i8 - .L_jump_table // 0x16
    .long   .L_op_scatter_tile_fp32 - .L_jump_table // 0x17
    .long   .L_op_transpose_fp32 - .L_jump_table // 0x18
    .long   .L_op_softmax_argmax_fp32 - .L_jump_table // 0x19
    .long   .L_op_luti4 - .L_jump_table // 0x1A
    .long   .L_op_luti2 - .L_jump_table // 0x1B
    .long   .L_op_dense_fp32 - .L_jump_table // 0x1C
// ================================================================
// EXIT — reached end of bytecodes
// ================================================================
.L_exit:
    smstop
    ldp     d12, d13, [sp, #112]
    ldp     d10, d11, [sp, #96]
    ldp     d8,  d9,  [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #128
    ret
// ================================================================
// ZERO_ZA (0x01)
// ================================================================
.L_op_zero_za:
    zero    {za}
    b       .L_dispatch
// ================================================================
// ACC_SMOPA (0x02) — fused ld1b+smopa loop
// Encoding: [0x02] [k_steps: 4 bytes LE]
// Operands: row_ptr, col_ptr
// ================================================================
.L_op_acc_smopa:
    ldr     w22, [x19]             // k_steps (4 bytes LE)
    add     x19, x19, #4           // advance IP past immediate
    ldr     x8, [x19], #8          // consume row_ptr
    ldr     x11, [x19], #8         // consume col_ptr
    ptrue   p0.b
    cbz     w22, .L_dispatch
.L_acc_smopa_loop:
    ld1b    {z0.b}, p0/z, [x8]
    ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x11]
    ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
    smopa   za0.s, p0/m, p0/m, z0.b, z2.b
    smopa   za1.s, p0/m, p0/m, z0.b, z3.b
    smopa   za2.s, p0/m, p0/m, z1.b, z2.b
    smopa   za3.s, p0/m, p0/m, z1.b, z3.b
    addvl   x8, x8, #2
    addvl   x11, x11, #2
    sub     w22, w22, #1
    cbnz    w22, .L_acc_smopa_loop
    b       .L_dispatch
// ================================================================
// ACC_UMOPA (0x03) — fused ld1b+umopa loop
// ================================================================
.L_op_acc_umopa:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8
    ldr     x11, [x19], #8
    ptrue   p0.b
    cbz     w22, .L_dispatch
.L_acc_umopa_loop:
    ld1b    {z0.b}, p0/z, [x8]
    ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x11]
    ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
    umopa   za0.s, p0/m, p0/m, z0.b, z2.b
    umopa   za1.s, p0/m, p0/m, z0.b, z3.b
    umopa   za2.s, p0/m, p0/m, z1.b, z2.b
    umopa   za3.s, p0/m, p0/m, z1.b, z3.b
    addvl   x8, x8, #2
    addvl   x11, x11, #2
    sub     w22, w22, #1
    cbnz    w22, .L_acc_umopa_loop
    b       .L_dispatch
// ================================================================
// ACC_USMOPA (0x04) — fused ld1b+usmopa loop
// ================================================================
.L_op_acc_usmopa:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8
    ldr     x11, [x19], #8
    ptrue   p0.b
    cbz     w22, .L_dispatch
.L_acc_usmopa_loop:
    ld1b    {z0.b}, p0/z, [x8]
    ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x11]
    ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
    usmopa  za0.s, p0/m, p0/m, z0.b, z2.b
    usmopa  za1.s, p0/m, p0/m, z0.b, z3.b
    usmopa  za2.s, p0/m, p0/m, z1.b, z2.b
    usmopa  za3.s, p0/m, p0/m, z1.b, z3.b
    addvl   x8, x8, #2
    addvl   x11, x11, #2
    sub     w22, w22, #1
    cbnz    w22, .L_acc_usmopa_loop
    b       .L_dispatch
// ================================================================
// ACC_SUMOPA (0x05) — fused ld1b+usmopa loop, signed rows × unsigned cols
// No hardware sumopa exists. We use usmopa (unsigned × signed) with
// swapped operand registers: cols (unsigned) as first arg, rows (signed)
// as second. This transposes the tile accumulation — the caller must
// account for the transposed layout when reading ZA tiles.
// ================================================================
.L_op_acc_sumopa:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8
    ldr     x11, [x19], #8
    ptrue   p0.b
    cbz     w22, .L_dispatch
.L_acc_sumopa_loop:
    ld1b    {z0.b}, p0/z, [x8]
    ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x11]
    ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
    usmopa  za0.s, p0/m, p0/m, z2.b, z0.b
    usmopa  za1.s, p0/m, p0/m, z3.b, z0.b
    usmopa  za2.s, p0/m, p0/m, z2.b, z1.b
    usmopa  za3.s, p0/m, p0/m, z3.b, z1.b
    addvl   x8, x8, #2
    addvl   x11, x11, #2
    sub     w22, w22, #1
    cbnz    w22, .L_acc_sumopa_loop
    b       .L_dispatch
// ================================================================
// STORE_TILES (0x06)
// Stores the 2x2 tile group as (2*SVLs) × (2*SVLs) row-major int32.
// Output is z_vector* — each z_vector is 64 bytes = 16 int32s = one SVLs row.
// A full row of output is 2 z_vectors wide (left half + right half).
// Consumes one operand: destination z_vector pointer.
//
// Layout (M4, SVLs=16, GROUP_DIM=32):
//   Row  0: z_vec[0]=za0_row0_left, z_vec[1]=za1_row0_right
//   Row  1: z_vec[2]=za0_row1_left, z_vec[3]=za1_row1_right
//   ...
//   Row 15: z_vec[30]=za0_row15_left, z_vec[31]=za1_row15_right
//   Row 16: z_vec[32]=za2_row0_left, z_vec[33]=za3_row0_right
//   ...
//   Row 31: z_vec[62]=za2_row15_left, z_vec[63]=za3_row15_right
// ================================================================
.L_op_store_tiles:
    ldr     x8, [x19], #8          // consume destination pointer
    ptrue   p0.s
    cntw    x9                     // SVLs (16 on M4)
    lsl     x10, x9, #3           // row stride in bytes: 2*SVLs*4 = SVLs*8
    // ---- Upper half: za0 (left) + za1 (right), SVLs rows ----
    mov     w12, #0
.L_se_store_upper:
    mova    {z4.s-z7.s}, za0h.s[w12, 0:3]
    mova    {z8.s-z11.s}, za1h.s[w12, 0:3]
    st1w    {z4.s}, p0, [x8]
    st1w    {z8.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z5.s}, p0, [x8]
    st1w    {z9.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z6.s}, p0, [x8]
    st1w    {z10.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z7.s}, p0, [x8]
    st1w    {z11.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .L_se_store_upper
    // ---- Lower half: za2 (left) + za3 (right), SVLs rows ----
    mov     w12, #0
.L_se_store_lower:
    mova    {z4.s-z7.s}, za2h.s[w12, 0:3]
    mova    {z8.s-z11.s}, za3h.s[w12, 0:3]
    st1w    {z4.s}, p0, [x8]
    st1w    {z8.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z5.s}, p0, [x8]
    st1w    {z9.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z6.s}, p0, [x8]
    st1w    {z10.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z7.s}, p0, [x8]
    st1w    {z11.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .L_se_store_lower
    b       .L_dispatch
// ================================================================
// LOAD_ROWS_I8 (0x07) — load 128 bytes into z0, z1
// ================================================================
.L_op_load_rows_i8:
    ldr     x8, [x19], #8
    ptrue   p0.b
    ld1b    {z0.b}, p0/z, [x8]
    cntb    x9
    ld1b    {z1.b}, p0/z, [x8, x9]
    b       .L_dispatch
// ================================================================
// LOAD_COLS_I8 (0x08) — load 128 bytes into z2, z3
// ================================================================
.L_op_load_cols_i8:
    ldr     x8, [x19], #8
    ptrue   p0.b
    ld1b    {z2.b}, p0/z, [x8]
    cntb    x9
    ld1b    {z3.b}, p0/z, [x8, x9]
    b       .L_dispatch
// ================================================================
// SMOPA_2x2 (0x09) — 4× smopa, no load
// ================================================================
.L_op_smopa_2x2:
    ptrue   p0.b
    smopa   za0.s, p0/m, p0/m, z0.b, z2.b
    smopa   za1.s, p0/m, p0/m, z0.b, z3.b
    smopa   za2.s, p0/m, p0/m, z1.b, z2.b
    smopa   za3.s, p0/m, p0/m, z1.b, z3.b
    b       .L_dispatch
// ================================================================
// UMOPA_2x2 (0x0A) — 4× umopa, no load
// ================================================================
.L_op_umopa_2x2:
    ptrue   p0.b
    umopa   za0.s, p0/m, p0/m, z0.b, z2.b
    umopa   za1.s, p0/m, p0/m, z0.b, z3.b
    umopa   za2.s, p0/m, p0/m, z1.b, z2.b
    umopa   za3.s, p0/m, p0/m, z1.b, z3.b
    b       .L_dispatch
// ================================================================
// USMOPA_2x2 (0x0B) — 4× usmopa, no load
// ================================================================
.L_op_usmopa_2x2:
    ptrue   p0.b
    usmopa  za0.s, p0/m, p0/m, z0.b, z2.b
    usmopa  za1.s, p0/m, p0/m, z0.b, z3.b
    usmopa  za2.s, p0/m, p0/m, z1.b, z2.b
    usmopa  za3.s, p0/m, p0/m, z1.b, z3.b
    b       .L_dispatch
// ================================================================
// LOAD_BIAS (0x0C)
// Loads int32 bias data from memory into ZA tiles (reverse of store_tiles).
// Same layout as store: (2*SVLs) × (2*SVLs) row-major int32.
// Consumes one operand: source pointer.
// This replaces zero_za when you want to accumulate on top of bias.
// ================================================================
.L_op_load_bias:
    ldr     x8, [x19], #8          // consume source pointer
    ptrue   p0.s
    cntw    x9                     // SVLs (16 on M4)
    lsl     x10, x9, #3           // row stride in bytes: 2*SVLs*4
    // ---- Upper half: → za0 (left) + za1 (right), SVLs rows ----
    mov     w12, #0
.L_se_loadbias_upper:
    ld1w    {z4.s}, p0/z, [x8]
    ld1w    {z8.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    ld1w    {z5.s}, p0/z, [x8]
    ld1w    {z9.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    ld1w    {z6.s}, p0/z, [x8]
    ld1w    {z10.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    ld1w    {z7.s}, p0/z, [x8]
    ld1w    {z11.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    mova    za0h.s[w12, 0:3], {z4.s-z7.s}
    mova    za1h.s[w12, 0:3], {z8.s-z11.s}
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .L_se_loadbias_upper
    // ---- Lower half: → za2 (left) + za3 (right), SVLs rows ----
    mov     w12, #0
.L_se_loadbias_lower:
    ld1w    {z4.s}, p0/z, [x8]
    ld1w    {z8.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    ld1w    {z5.s}, p0/z, [x8]
    ld1w    {z9.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    ld1w    {z6.s}, p0/z, [x8]
    ld1w    {z10.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    ld1w    {z7.s}, p0/z, [x8]
    ld1w    {z11.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    mova    za2h.s[w12, 0:3], {z4.s-z7.s}
    mova    za3h.s[w12, 0:3], {z8.s-z11.s}
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .L_se_loadbias_lower
    b       .L_dispatch
// ================================================================
// SCALE_STORE (0x0D)
// Reads ZA tiles (int32), converts to float32, multiplies by a scalar
// scale factor, and stores as float32.
// Encoding: [0x0D] [scale: 4 bytes LE float]
// Consumes one operand: destination pointer.
//
// This is the dequantization step: int32_accum × scale → float32 output.
// The scale is a float immediate in the bytecode stream.
// ================================================================
.L_op_scale_store:
    ldr     s16, [x19]             // load float scale into s16 (bottom of z16)
    add     x19, x19, #4           // advance IP past float immediate
    ldr     x8, [x19], #8          // consume destination pointer
    ptrue   p0.s
    cntw    x9                     // SVLs
    lsl     x10, x9, #3           // row stride
    // Broadcast scale into z16 for fmul
    mov     z16.s, s16
    // ---- Upper half: za0 (left) + za1 (right) ----
    mov     w12, #0
.L_se_scalestore_upper:
    // Extract 4 rows from za0, za1
    mova    {z4.s-z7.s}, za0h.s[w12, 0:3]
    mova    {z8.s-z11.s}, za1h.s[w12, 0:3]
    // Convert int32 → float32 (scvtf) and multiply by scale
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z16.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z16.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z16.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z16.s
    scvtf   z8.s, p0/m, z8.s
    fmul    z8.s, z8.s, z16.s
    scvtf   z9.s, p0/m, z9.s
    fmul    z9.s, z9.s, z16.s
    scvtf   z10.s, p0/m, z10.s
    fmul    z10.s, z10.s, z16.s
    scvtf   z11.s, p0/m, z11.s
    fmul    z11.s, z11.s, z16.s
    // Store float32 results
    st1w    {z4.s}, p0, [x8]
    st1w    {z8.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z5.s}, p0, [x8]
    st1w    {z9.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z6.s}, p0, [x8]
    st1w    {z10.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z7.s}, p0, [x8]
    st1w    {z11.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .L_se_scalestore_upper
    // ---- Lower half: za2 (left) + za3 (right) ----
    mov     w12, #0
.L_se_scalestore_lower:
    mova    {z4.s-z7.s}, za2h.s[w12, 0:3]
    mova    {z8.s-z11.s}, za3h.s[w12, 0:3]
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z16.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z16.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z16.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z16.s
    scvtf   z8.s, p0/m, z8.s
    fmul    z8.s, z8.s, z16.s
    scvtf   z9.s, p0/m, z9.s
    fmul    z9.s, z9.s, z16.s
    scvtf   z10.s, p0/m, z10.s
    fmul    z10.s, z10.s, z16.s
    scvtf   z11.s, p0/m, z11.s
    fmul    z11.s, z11.s, z16.s
    st1w    {z4.s}, p0, [x8]
    st1w    {z8.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z5.s}, p0, [x8]
    st1w    {z9.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z6.s}, p0, [x8]
    st1w    {z10.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z7.s}, p0, [x8]
    st1w    {z11.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .L_se_scalestore_lower
    b       .L_dispatch
// ================================================================
// DENSE_FUSED_I8 (0x0E) — fully fused fp32→i8 quantize+pack+smopa+dequant(+relu)→fp32
// Full-matrix version with internal tiling (arbitrary M, N, K).
// Bytecode: [0x0E][M:u32][N:u32][K:u32][scale:f32][flags:u8] = 17 bytes
//   M, N, K = full matrix dimensions (arbitrary, not limited to 32)
//   scale = dequant scale (typically 1.0)
//   flags bit 0: 1=apply ReLU, 0=no ReLU
// Operands: A (M×K row-major, stride K), B (K×N row-major, stride N), C (M×N row-major, stride N)
// Stack layout (dynamic):
//   [sp+0   .. +128):   context block (saved ptrs, dims, scales)
//   [sp+128 .. +384):   packed scratch (256 bytes: 128 rows + 128 cols)
//   [sp+384 .. +4480):  tile_out scratch (4096 bytes = 32×32×4)
//   [sp+4480 .. +4480+M_pad*K_pad): qa (quantized A, int8)
//   [sp+4480+M_pad*K_pad .. +4480+M_pad*K_pad+K_pad*N_pad): qb (quantized B, int8)
// ================================================================
.L_op_dense_scale_relu_i8:
    // ── Parse immediates + operands ──
    ldr     w0, [x19]              // M
    ldr     w1, [x19, #4]          // N
    ldr     w2, [x19, #8]          // K
    ldr     s16, [x19, #12]        // scale (f32)
    ldrb    w18, [x19, #16]        // flags (bit 0 = relu)
    add     x19, x19, #17
    ldr     x5, [x19], #8          // A
    ldr     x6, [x19], #8          // B
    ldr     x7, [x19], #8          // C
    // ── Compute derived values ──
    add     w14, w2, #3
    and     w14, w14, #0xFFFFFFFC  // K_pad = (K+3) & ~3
    lsr     w15, w14, #2           // k_steps = K_pad / 4
    add     w3, w0, #31
    and     w3, w3, #0xFFFFFFE0   // M_pad = (M+31) & ~31
    add     w4, w1, #31
    and     w4, w4, #0xFFFFFFE0   // N_pad = (N+31) & ~31
    ptrue   p0.s
    cntw    x9                     // SVLs = 16
    // ── Allocate stack frame ──
    //   fixed: 128 (context) + 256 (packed) + 4096 (tile_out) = 4480
    //   dynamic: M_pad*K_pad (qa) + K_pad*N_pad (qb)
    mul     w10, w3, w14           // M_pad * K_pad
    mul     w11, w14, w4           // K_pad * N_pad
    add     w10, w10, w11          // qa + qb total
    add     x10, x10, #4096       // + fixed (4480 = 4096 + 384)
    add     x10, x10, #384
    add     x10, x10, #63
    and     x10, x10, #~63        // round up to 64-byte alignment
    sub     sp, sp, x10
    // Save context to [sp+0..127]
    stp     x5, x6, [sp, #0]      // [0] A, [8] B
    str     x7, [sp, #16]         // [16] C
    stp     w0, w1, [sp, #24]     // [24] M, [28] N
    stp     w2, w14, [sp, #32]    // [32] K, [36] K_pad
    stp     w4, w3, [sp, #40]     // [40] N_pad, [44] M_pad
    stp     w15, w18, [sp, #48]   // [48] k_steps, [52] flags
    str     s16, [sp, #56]        // [56] scale
    str     x10, [sp, #72]        // [72] frame_size
    // ── Compute buffer pointers ──
    add     x11, sp, #4096        // qa_base = sp + 4480
    add     x11, x11, #384
    mul     w16, w3, w14          // M_pad * K_pad
    add     x13, x11, x16         // qb_base = qa_base + M_pad*K_pad
    // Zero qa and qb
    add     x8, x16, x11         // end of qa  (recompute: qa_base + M_pad*K_pad = qb_base)
    mul     w17, w14, w4          // K_pad * N_pad
    add     x8, x16, x17         // total bytes to zero = M_pad*K_pad + K_pad*N_pad
    mov     x16, x11             // cursor at qa_base
.L_fuse_zero_scratch:
    cbz     x8, .L_fuse_zero_done
    stp     xzr, xzr, [x16], #16
    sub     x8, x8, #16
    cbnz    x8, .L_fuse_zero_scratch
.L_fuse_zero_done:
    // ════════════════════════════════════════════════════════════
    // Phase 1: Absmax of A (M rows × K cols, contiguous stride K)
    // ════════════════════════════════════════════════════════════
    fmov    z18.s, #0.0            // running absmax
    mov     x8, x5                 // A cursor
    mul     x16, x0, x2            // M * K total elements
    mov     x17, xzr               // index
    whilelt p1.s, xzr, x16
.L_fuse_absmax_a:
    ld1w    {z0.s}, p1/z, [x8]
    fabs    z0.s, p1/m, z0.s
    fmax    z18.s, p1/m, z18.s, z0.s
    add     x8, x8, x9, lsl #2
    incw    x17
    whilelt p1.s, x17, x16
    b.first .L_fuse_absmax_a
    fmaxv   s18, p0, z18.s        // absmax_a → s18
    // Compute scale_a = 127/absmax_a, inv_a = absmax_a/127
    fmov    s17, #1.0
    fcmp    s18, #0.0
    fcsel   s18, s17, s18, eq
    movz    w8, #0x42FE, lsl #16   // 127.0f
    fmov    s17, w8
    fdiv    s28, s17, s18          // s28 = 127/absmax_a (scale_a)
    fdiv    s29, s18, s17          // s29 = absmax_a/127 (inv_a)
    str     s29, [sp, #60]         // save inv_a at [60]
    // ════════════════════════════════════════════════════════════
    // Phase 2: Absmax of B (K rows × N cols, contiguous stride N)
    // ════════════════════════════════════════════════════════════
    fmov    z18.s, #0.0
    ldr     x6, [sp, #8]          // reload B
    ldr     w2, [sp, #32]         // K
    ldr     w1, [sp, #28]         // N
    lsl     x3, x1, #2            // stride N in bytes
    mov     w12, #0
.L_fuse_absmax_b:
    cmp     w12, w2
    b.ge    .L_fuse_absmax_b_done
    mov     x8, x6
    mov     x17, xzr
    whilelt p1.s, xzr, x1
.L_fuse_absmax_b_inner:
    ld1w    {z0.s}, p1/z, [x8]
    fabs    z0.s, p1/m, z0.s
    fmax    z18.s, p1/m, z18.s, z0.s
    add     x8, x8, x9, lsl #2
    incw    x17
    whilelt p1.s, x17, x1
    b.first .L_fuse_absmax_b_inner
    add     x6, x6, x3            // next B row (stride N*4 bytes)
    add     w12, w12, #1
    b       .L_fuse_absmax_b
.L_fuse_absmax_b_done:
    fmaxv   s18, p0, z18.s
    fmov    s17, #1.0
    fcmp    s18, #0.0
    fcsel   s18, s17, s18, eq
    movz    w8, #0x42FE, lsl #16
    fmov    s17, w8
    fdiv    s30, s17, s18          // s30 = 127/absmax_b (scale_b)
    fdiv    s17, s18, s17          // s17 = inv_b
    str     s17, [sp, #64]         // save inv_b at [64]
    // ════════════════════════════════════════════════════════════
    // Phase 3: Quantize A → qa (M_pad rows × K_pad cols, zero-padded)
    // A is M rows of K contiguous floats, rows beyond M stay zero.
    // ════════════════════════════════════════════════════════════
    ldr     x5, [sp, #0]          // A
    ldr     w0, [sp, #24]         // M
    ldr     w2, [sp, #32]         // K
    ldr     w14, [sp, #36]        // K_pad
    add     x11, sp, #4096        // qa_base = sp + 4480
    add     x11, x11, #384
    mov     z16.s, s28             // broadcast scale_a
    mov     w12, #0
.L_fuse_quant_a_row:
    cmp     w12, w0
    b.ge    .L_fuse_quant_a_done
    mov     x8, x5                 // src cursor (row start in A)
    mov     x13, x11               // dst cursor (row start in qa)
    mov     x17, xzr
    whilelt p1.s, xzr, x2
.L_fuse_quant_a_inner:
    ld1w    {z0.s}, p1/z, [x8]
    fmul    z0.s, z0.s, z16.s
    frinti  z0.s, p1/m, z0.s
    fcvtzs  z0.s, p1/m, z0.s
    mov     z1.s, #127
    mov     z2.s, #-127
    smin    z0.s, p1/m, z0.s, z1.s
    smax    z0.s, p1/m, z0.s, z2.s
    st1b    {z0.s}, p1, [x13]
    add     x8, x8, x9, lsl #2
    add     x13, x13, x9
    incw    x17
    whilelt p1.s, x17, x2
    b.first .L_fuse_quant_a_inner
    lsl     x8, x2, #2            // K * 4 bytes
    add     x5, x5, x8            // next A row
    add     x11, x11, x14         // next qa row (stride K_pad)
    add     w12, w12, #1
    b       .L_fuse_quant_a_row
.L_fuse_quant_a_done:
    // ════════════════════════════════════════════════════════════
    // Phase 4: Quantize B → qb (K_pad rows × N_pad cols, zero-padded)
    // B is K rows of N contiguous floats, stride N.
    // ════════════════════════════════════════════════════════════
    ldr     x6, [sp, #8]          // B
    ldr     w2, [sp, #32]         // K
    ldr     w1, [sp, #28]         // N
    ldr     w14, [sp, #36]        // K_pad
    ldr     w4, [sp, #40]         // N_pad
    lsl     x3, x1, #2            // N * 4 bytes (B row stride)
    ldr     w16, [sp, #44]        // M_pad
    mul     w10, w16, w14         // M_pad * K_pad = qa size
    add     x11, sp, #4096         // qa_base = sp + 4480
    add     x11, x11, #384
    add     x11, x11, x10         // qb_base
    mov     z16.s, s30             // broadcast scale_b
    mov     w12, #0
.L_fuse_quant_b_row:
    cmp     w12, w2
    b.ge    .L_fuse_quant_b_done
    mov     x8, x6                 // src cursor
    mov     x13, x11               // dst cursor
    mov     x17, xzr
    whilelt p1.s, xzr, x1
.L_fuse_quant_b_inner:
    ld1w    {z0.s}, p1/z, [x8]
    fmul    z0.s, z0.s, z16.s
    frinti  z0.s, p1/m, z0.s
    fcvtzs  z0.s, p1/m, z0.s
    mov     z1.s, #127
    mov     z2.s, #-127
    smin    z0.s, p1/m, z0.s, z1.s
    smax    z0.s, p1/m, z0.s, z2.s
    st1b    {z0.s}, p1, [x13]
    add     x8, x8, x9, lsl #2
    add     x13, x13, x9
    incw    x17
    whilelt p1.s, x17, x1
    b.first .L_fuse_quant_b_inner
    add     x6, x6, x3            // next B row (stride N*4 bytes)
    add     x11, x11, x4          // next qb row (stride N_pad bytes)
    add     w12, w12, #1
    b       .L_fuse_quant_b_row
.L_fuse_quant_b_done:
    // ════════════════════════════════════════════════════════════
    // Phase 5: Compute combined dequant scale
    // ════════════════════════════════════════════════════════════
    ldr     s28, [sp, #60]         // inv_a
    ldr     s29, [sp, #64]         // inv_b
    ldr     s30, [sp, #56]         // user scale
    fmul    s28, s28, s29
    fmul    s28, s28, s30          // combined = inv_a * inv_b * scale
    str     s28, [sp, #56]         // overwrite scale with combined
    // ════════════════════════════════════════════════════════════
    // Phase 6: Tile loop — for each 32×32 tile block
    //   x0 = ti (row tile offset), x1 = tj (col tile offset)
    //   Iterates ti = 0..M_pad-1 step 32, tj = 0..N_pad-1 step 32
    // ════════════════════════════════════════════════════════════
    ldr     w3, [sp, #44]          // M_pad
    ldr     w4, [sp, #40]          // N_pad
    ldr     w14, [sp, #36]         // K_pad
    ldr     w15, [sp, #48]         // k_steps
    mov     w0, #0                 // ti = 0
.L_fuse_tile_row:
    cmp     w0, w3                 // ti < M_pad
    b.ge    .L_fuse_tile_done
    mov     w1, #0                 // tj = 0
.L_fuse_tile_col:
    cmp     w1, w4                 // tj < N_pad
    b.ge    .L_fuse_tile_row_next
    // ── Save tile coords to context ──
    stp     w0, w1, [sp, #80]     // [80] ti, [84] tj
    // ── Compute qa_tile = qa_base + ti*K_pad ──
    add     x11, sp, #4096        // qa_base = sp + 4480
    add     x11, x11, #384
    mul     w10, w0, w14          // ti * K_pad
    add     x11, x11, x10         // qa_tile = qa_base + ti*K_pad
    // ── Compute qb_tile = qb_base + tj (column offset in qb row) ──
    ldr     w16, [sp, #44]        // M_pad
    mul     w10, w16, w14         // M_pad * K_pad
    add     x13, sp, #4096
    add     x13, x13, #384
    add     x13, x13, x10         // qb_base
    add     x13, x13, x1          // qb_tile = qb_base + tj
    // ── Zero ZA for this tile ──
    zero    {za}
    ptrue   p0.b
    add     x16, sp, #128         // packed scratch
    mov     w12, #0               // t = 0 (k-step counter)
    cbz     w15, .L_fuse_tile_smopa_done
.L_fuse_tile_kstep:
    // ── Pack rows: gather 4 bytes from each of 32 qa rows into packed scratch ──
    mov     x8, x16               // dst = packed_rows scratch
    mov     w17, #0               // row counter
.L_fuse_tile_pr:
    madd    x10, x17, x14, x11    // x10 = qa_tile + row*K_pad
    add     x10, x10, x12, lsl #2 // + t*4
    ldr     w22, [x10]
    str     w22, [x8], #4
    add     w17, w17, #1
    cmp     w17, #32
    b.lt    .L_fuse_tile_pr
    // ── Pack cols: load 4 qb rows at k-step t, zip interleave ──
    //   qb row r starts at qb_tile + r * N_pad
    //   We need rows t*4+0 .. t*4+3
    lsl     w17, w12, #2           // t*4
    mul     x10, x17, x4          // (t*4) * N_pad
    add     x8, x13, x10          // qb_tile + (t*4)*N_pad
    ld1b    {z0.b}, p0/z, [x8]
    add     x8, x8, x4            // += N_pad
    ld1b    {z1.b}, p0/z, [x8]
    add     x8, x8, x4
    ld1b    {z2.b}, p0/z, [x8]
    add     x8, x8, x4
    ld1b    {z3.b}, p0/z, [x8]
    zip1    z4.b, z0.b, z2.b
    zip1    z5.b, z1.b, z3.b
    zip1    z6.b, z4.b, z5.b
    zip2    z7.b, z4.b, z5.b
    add     x8, x16, #128         // packed_cols scratch
    st1b    {z6.b}, p0, [x8]
    add     x10, x8, #64
    st1b    {z7.b}, p0, [x10]
    // ── Load packed data into z0-z3 and run smopa ──
    ld1b    {z0.b}, p0/z, [x16]
    add     x8, x16, #64
    ld1b    {z1.b}, p0/z, [x8]
    add     x8, x16, #128
    ld1b    {z2.b}, p0/z, [x8]
    add     x8, x16, #192
    ld1b    {z3.b}, p0/z, [x8]
    smopa   za0.s, p0/m, p0/m, z0.b, z2.b
    smopa   za1.s, p0/m, p0/m, z0.b, z3.b
    smopa   za2.s, p0/m, p0/m, z1.b, z2.b
    smopa   za3.s, p0/m, p0/m, z1.b, z3.b
    add     w12, w12, #1
    cmp     w12, w15
    b.lt    .L_fuse_tile_kstep
.L_fuse_tile_smopa_done:
    // ── Dequant + optional ReLU → tile_out scratch (32×32 floats) ──
    ldr     s28, [sp, #56]         // combined dequant scale
    mov     z16.s, s28             // broadcast
    ldr     w18, [sp, #52]         // flags
    and     w18, w18, #1           // isolate relu
    ptrue   p0.s
    cntw    x9
    fmov    z17.s, #0.0            // for ReLU
    add     x7, sp, #384          // tile_out scratch base
    mov     x10, #128              // tile_out row stride = 32*4 = 128 bytes
    // Store upper half: za0 (left 16 cols) + za1 (right 16 cols) → rows 0..15
    mov     w12, #0
.L_fuse_tile_store_upper:
    mova    {z4.s-z7.s}, za0h.s[w12, 0:3]
    mova    {z8.s-z11.s}, za1h.s[w12, 0:3]
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z16.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z16.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z16.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z16.s
    scvtf   z8.s, p0/m, z8.s
    fmul    z8.s, z8.s, z16.s
    scvtf   z9.s, p0/m, z9.s
    fmul    z9.s, z9.s, z16.s
    scvtf   z10.s, p0/m, z10.s
    fmul    z10.s, z10.s, z16.s
    scvtf   z11.s, p0/m, z11.s
    fmul    z11.s, z11.s, z16.s
    cbz     w18, .L_fuse_tile_su_norelu
    fmax    z4.s, p0/m, z4.s, z17.s
    fmax    z5.s, p0/m, z5.s, z17.s
    fmax    z6.s, p0/m, z6.s, z17.s
    fmax    z7.s, p0/m, z7.s, z17.s
    fmax    z8.s, p0/m, z8.s, z17.s
    fmax    z9.s, p0/m, z9.s, z17.s
    fmax    z10.s, p0/m, z10.s, z17.s
    fmax    z11.s, p0/m, z11.s, z17.s
.L_fuse_tile_su_norelu:
    st1w    {z4.s}, p0, [x7]
    st1w    {z8.s}, p0, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z5.s}, p0, [x7]
    st1w    {z9.s}, p0, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z6.s}, p0, [x7]
    st1w    {z10.s}, p0, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z7.s}, p0, [x7]
    st1w    {z11.s}, p0, [x7, x9, lsl #2]
    add     x7, x7, x10
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .L_fuse_tile_store_upper
    // Store lower half: za2 (left) + za3 (right) → rows 16..31
    mov     w12, #0
.L_fuse_tile_store_lower:
    mova    {z4.s-z7.s}, za2h.s[w12, 0:3]
    mova    {z8.s-z11.s}, za3h.s[w12, 0:3]
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z16.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z16.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z16.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z16.s
    scvtf   z8.s, p0/m, z8.s
    fmul    z8.s, z8.s, z16.s
    scvtf   z9.s, p0/m, z9.s
    fmul    z9.s, z9.s, z16.s
    scvtf   z10.s, p0/m, z10.s
    fmul    z10.s, z10.s, z16.s
    scvtf   z11.s, p0/m, z11.s
    fmul    z11.s, z11.s, z16.s
    cbz     w18, .L_fuse_tile_sl_norelu
    fmax    z4.s, p0/m, z4.s, z17.s
    fmax    z5.s, p0/m, z5.s, z17.s
    fmax    z6.s, p0/m, z6.s, z17.s
    fmax    z7.s, p0/m, z7.s, z17.s
    fmax    z8.s, p0/m, z8.s, z17.s
    fmax    z9.s, p0/m, z9.s, z17.s
    fmax    z10.s, p0/m, z10.s, z17.s
    fmax    z11.s, p0/m, z11.s, z17.s
.L_fuse_tile_sl_norelu:
    st1w    {z4.s}, p0, [x7]
    st1w    {z8.s}, p0, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z5.s}, p0, [x7]
    st1w    {z9.s}, p0, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z6.s}, p0, [x7]
    st1w    {z10.s}, p0, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z7.s}, p0, [x7]
    st1w    {z11.s}, p0, [x7, x9, lsl #2]
    add     x7, x7, x10
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .L_fuse_tile_store_lower
    // ── Copy valid portion of tile_out → C ──
    // C_tile = C_base + ti*N + tj (in floats)
    // Copy min(32, M-ti) rows, each min(32, N-tj) floats
    ldr     x7, [sp, #16]         // C_base
    ldp     w0, w1, [sp, #80]     // ti, tj
    ldr     w2, [sp, #24]         // M
    ldr     w3, [sp, #28]         // N
    // rows_valid = min(32, M - ti)
    sub     w5, w2, w0             // M - ti
    mov     w8, #32
    cmp     w5, w8
    csel    w5, w5, w8, lt         // w5 = min(M-ti, 32)
    // cols_valid = min(32, N - tj)
    sub     w6, w3, w1             // N - tj
    cmp     w6, w8
    csel    w6, w6, w8, lt         // w6 = min(N-tj, 32)
    // C_tile = C + (ti * N + tj) * 4
    mul     w8, w0, w3             // ti * N
    add     w8, w8, w1             // + tj
    add     x7, x7, x8, lsl #2    // C_tile ptr
    lsl     x10, x3, #2           // N * 4 = C row stride in bytes
    add     x11, sp, #384         // tile_out base
    // Generate col predicate: whilelt for cols_valid
    whilelt p2.s, xzr, x6         // p2 masks valid columns
    mov     w12, #0               // row counter
.L_fuse_tile_copy:
    cmp     w12, w5
    b.ge    .L_fuse_tile_copy_done
    // Load from tile_out row: left 16 + right 16
    ld1w    {z0.s}, p2/z, [x11]
    // Check if cols_valid > 16: need right half too
    cmp     w6, #16
    b.le    .L_fuse_tile_copy_left_only
    // For the right half, create offset predicate
    sub     w17, w6, #16
    whilelt p3.s, xzr, x17
    add     x8, x11, x9, lsl #2   // tile_out + 16*4
    ld1w    {z1.s}, p3/z, [x8]
    st1w    {z0.s}, p2, [x7]
    st1w    {z1.s}, p3, [x7, x9, lsl #2]
    b       .L_fuse_tile_copy_next
.L_fuse_tile_copy_left_only:
    st1w    {z0.s}, p2, [x7]
.L_fuse_tile_copy_next:
    add     x7, x7, x10           // next C row
    add     x11, x11, #128        // next tile_out row
    add     w12, w12, #1
    b       .L_fuse_tile_copy
.L_fuse_tile_copy_done:
    // ── Advance to next tile column ──
    ldp     w0, w1, [sp, #80]     // reload ti, tj
    ldr     w4, [sp, #40]         // N_pad
    ldr     w14, [sp, #36]        // K_pad
    ldr     w15, [sp, #48]        // k_steps
    ldr     w3, [sp, #44]         // M_pad
    add     w1, w1, #32           // tj += 32
    b       .L_fuse_tile_col
.L_fuse_tile_row_next:
    add     w0, w0, #32           // ti += 32
    b       .L_fuse_tile_row
.L_fuse_tile_done:
    // ── Deallocate stack and dispatch ──
    ldr     x10, [sp, #72]         // frame size
    add     sp, sp, x10
    b       .L_dispatch
// ================================================================
// DENSE_SCALE_I8 (0x0F)
// Fused: zero → smopa accumulate → scvtf → fmul(scale) → store fp32
// Bytecode: [0x0F][k_steps:u32][scale:f32]
// Operands: rows_ptr, cols_ptr, output_ptr
// ================================================================
.L_op_dense_scale_i8:
    ldr     w22, [x19]
    ldr     s16, [x19, #4]
    add     x19, x19, #8
    ldr     x8, [x19], #8
    ldr     x11, [x19], #8
    ldr     x13, [x19], #8
    zero    {za}
    ptrue   p0.b
    cbz     w22, .L_scale_dequant
.L_scale_acc:
    ld1b    {z0.b}, p0/z, [x8]
    ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x11]
    ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
    smopa   za0.s, p0/m, p0/m, z0.b, z2.b
    smopa   za1.s, p0/m, p0/m, z0.b, z3.b
    smopa   za2.s, p0/m, p0/m, z1.b, z2.b
    smopa   za3.s, p0/m, p0/m, z1.b, z3.b
    addvl   x8, x8, #2
    addvl   x11, x11, #2
    sub     w22, w22, #1
    cbnz    w22, .L_scale_acc
.L_scale_dequant:
    ptrue   p0.s
    mov     z16.s, s16
    cntw    x9
    lsl     x10, x9, #3
    mov     w12, #0
.L_scale_upper:
    mova    {z4.s-z7.s}, za0h.s[w12, 0:3]
    mova    {z8.s-z11.s}, za1h.s[w12, 0:3]
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z16.s
    scvtf   z8.s, p0/m, z8.s
    fmul    z8.s, z8.s, z16.s
    st1w    {z4.s}, p0, [x13]
    st1w    {z8.s}, p0, [x13, x9, lsl #2]
    add     x13, x13, x10
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z16.s
    scvtf   z9.s, p0/m, z9.s
    fmul    z9.s, z9.s, z16.s
    st1w    {z5.s}, p0, [x13]
    st1w    {z9.s}, p0, [x13, x9, lsl #2]
    add     x13, x13, x10
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z16.s
    scvtf   z10.s, p0/m, z10.s
    fmul    z10.s, z10.s, z16.s
    st1w    {z6.s}, p0, [x13]
    st1w    {z10.s}, p0, [x13, x9, lsl #2]
    add     x13, x13, x10
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z16.s
    scvtf   z11.s, p0/m, z11.s
    fmul    z11.s, z11.s, z16.s
    st1w    {z7.s}, p0, [x13]
    st1w    {z11.s}, p0, [x13, x9, lsl #2]
    add     x13, x13, x10
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .L_scale_upper
    mov     w12, #0
.L_scale_lower:
    mova    {z4.s-z7.s}, za2h.s[w12, 0:3]
    mova    {z8.s-z11.s}, za3h.s[w12, 0:3]
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z16.s
    scvtf   z8.s, p0/m, z8.s
    fmul    z8.s, z8.s, z16.s
    st1w    {z4.s}, p0, [x13]
    st1w    {z8.s}, p0, [x13, x9, lsl #2]
    add     x13, x13, x10
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z16.s
    scvtf   z9.s, p0/m, z9.s
    fmul    z9.s, z9.s, z16.s
    st1w    {z5.s}, p0, [x13]
    st1w    {z9.s}, p0, [x13, x9, lsl #2]
    add     x13, x13, x10
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z16.s
    scvtf   z10.s, p0/m, z10.s
    fmul    z10.s, z10.s, z16.s
    st1w    {z6.s}, p0, [x13]
    st1w    {z10.s}, p0, [x13, x9, lsl #2]
    add     x13, x13, x10
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z16.s
    scvtf   z11.s, p0/m, z11.s
    fmul    z11.s, z11.s, z16.s
    st1w    {z7.s}, p0, [x13]
    st1w    {z11.s}, p0, [x13, x9, lsl #2]
    add     x13, x13, x10
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .L_scale_lower
    b       .L_dispatch
// ================================================================
// ELEMENTWISE_ADD_FP32 (0x10)
// Pure SVE: load a, load b, fadd, store
// Bytecode: [0x10][count:u32]
// Operands: a_ptr, b_ptr, output_ptr
// ================================================================
.L_op_elementwise_add_fp32:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8
    ldr     x11, [x19], #8
    ldr     x13, [x19], #8
    ptrue   p0.s
    cntw    x9
    cbz     w22, .L_dispatch
.L_add_loop:
    ld1w    {z0.s}, p0/z, [x8]
    ld1w    {z1.s}, p0/z, [x11]
    fadd    z0.s, z0.s, z1.s
    st1w    {z0.s}, p0, [x13]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x13, x13, x9, lsl #2
    sub     w22, w22, w9
    cbnz    w22, .L_add_loop
    b       .L_dispatch
// ================================================================
// ELEMENTWISE_SCALED_ADD_FP32 (0x11)
// out = a + scale * b  (used for SGD: W = W + (-lr) * grad)
// Bytecode: [0x11][count:u32][scale:f32]
// Operands: a_ptr, b_ptr, output_ptr
// ================================================================
.L_op_elementwise_scaled_add_fp32:
    ldr     w22, [x19]
    ldr     s16, [x19, #4]
    add     x19, x19, #8
    ldr     x8, [x19], #8
    ldr     x11, [x19], #8
    ldr     x13, [x19], #8
    ptrue   p0.s
    mov     z16.s, s16
    cntw    x9
    cbz     w22, .L_dispatch
.L_scadd_loop:
    ld1w    {z0.s}, p0/z, [x8]
    ld1w    {z1.s}, p0/z, [x11]
    fmla    z0.s, p0/m, z1.s, z16.s
    st1w    {z0.s}, p0, [x13]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x13, x13, x9, lsl #2
    sub     w22, w22, w9
    cbnz    w22, .L_scadd_loop
    b       .L_dispatch
// ================================================================
// ELEMENTWISE_MUL_FP32 (0x12)
// out = a * b  (used for ReLU backward mask)
// Bytecode: [0x12][count:u32]
// Operands: a_ptr, b_ptr, output_ptr
// ================================================================
.L_op_elementwise_mul_fp32:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8
    ldr     x11, [x19], #8
    ldr     x13, [x19], #8
    ptrue   p0.s
    cntw    x9
    cbz     w22, .L_dispatch
.L_mul_loop:
    ld1w    {z0.s}, p0/z, [x8]
    ld1w    {z1.s}, p0/z, [x11]
    fmul    z0.s, z0.s, z1.s
    st1w    {z0.s}, p0, [x13]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x13, x13, x9, lsl #2
    sub     w22, w22, w9
    cbnz    w22, .L_mul_loop
    b       .L_dispatch
// ================================================================
// RELU_BACKWARD_FP32 (0x13)
// out[i] = (hidden[i] > 0) ? grad[i] : 0
// Bytecode: [0x13][count:u32]
// Operands: hidden_ptr, grad_ptr, output_ptr
// ================================================================
.L_op_relu_backward_fp32:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8          // hidden (post-relu forward values)
    ldr     x11, [x19], #8         // grad (incoming gradient)
    ldr     x13, [x19], #8         // output
    ptrue   p0.s
    fmov    z17.s, #0.0
    cntw    x9
    cbz     w22, .L_dispatch
.L_relubw_loop:
    ld1w    {z0.s}, p0/z, [x8]     // hidden
    ld1w    {z1.s}, p0/z, [x11]    // grad
    fcmgt   p1.s, p0/z, z0.s, z17.s // p1 = (hidden > 0)
    mov     z2.s, #0
    sel     z2.s, p1, z1.s, z2.s   // z2 = p1 ? grad : 0
    st1w    {z2.s}, p0, [x13]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x13, x13, x9, lsl #2
    sub     w22, w22, w9
    cbnz    w22, .L_relubw_loop
    b       .L_dispatch
// ================================================================
// QUANTIZE_FP32_I8 (0x14)
// Quantize fp32 array to int8: find absmax, scale to [-127,127], output i8
// Writes inverse_scale (1/scale) to a float pointer for later dequant
// Bytecode: [0x14][count:u32]
// Operands: src_fp32_ptr, dst_i8_ptr, inv_scale_ptr
// ================================================================
.L_op_quantize_fp32_i8:
    ldr     w22, [x19]             // count (fp32 elements)
    add     x19, x19, #4
    ldr     x8, [x19], #8          // src fp32
    ldr     x11, [x19], #8         // dst i8
    ldr     x13, [x19], #8         // inv_scale output (float*)
    ptrue   p0.s
    cntw    x9                     // SVLs
    // Pass 1: find absmax across all elements
    fmov    z16.s, #0.0            // running max = 0
    mov     x14, x8                // save src ptr
    mov     w15, w22               // save count
    cbz     w22, .L_quant_store_scale
.L_quant_absmax:
    ld1w    {z0.s}, p0/z, [x8]
    fabs    z0.s, p0/m, z0.s
    fmax    z16.s, p0/m, z16.s, z0.s
    add     x8, x8, x9, lsl #2
    sub     w22, w22, w9
    cbnz    w22, .L_quant_absmax
    // Horizontal max reduce z16 → s16
    fmaxv   s16, p0, z16.s
    // scale = 127.0 / absmax, handle near-zero
    fmov    s17, #1.0
    movz    w3, #0x42FE, lsl #16    // 127.0 as IEEE754 = 0x42FE0000
    fmov    s18, w3
    fcmp    s16, #0.0
    fccmp   s16, s17, #0, ne       // if absmax < 1e-8 treat as 1e-8
    fcsel   s16, s17, s16, lt      // clamp absmax to at least ~1.0 for safety
    fdiv    s17, s18, s16          // s17 = 127.0 / absmax = scale
    fdiv    s18, s16, s18          // s18 = absmax / 127.0 = inv_scale
    // Write inv_scale
    str     s18, [x13]
    // Pass 2: quantize — dst[i] = clamp(round(src[i] * scale), -127, 127)
    mov     z16.s, s17             // broadcast scale
    mov     x8, x14                // restore src ptr
    mov     w22, w15               // restore count
    cntw    x9
.L_quant_scale:
    ld1w    {z0.s}, p0/z, [x8]
    fmul    z0.s, z0.s, z16.s      // src * scale
    frinti  z0.s, p0/m, z0.s       // round to nearest int
    fcvtzs  z0.s, p0/m, z0.s       // convert to int32
    // Clamp to [-127, 127]
    mov     z17.s, #127
    mov     z18.s, #-127
    smin    z0.s, p0/m, z0.s, z17.s
    smax    z0.s, p0/m, z0.s, z18.s
    // Store low byte of each int32 element: st1b with .s element size
    // This truncates each 32-bit lane to 8 bits and stores contiguously
    st1b    {z0.s}, p0, [x11]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9           // advance by SVLs bytes (not *4)
    sub     w22, w22, w9
    cbnz    w22, .L_quant_scale
.L_quant_store_scale:
    b       .L_dispatch
// ================================================================
// PACK_ROWS_I8 (0x15)
// Pack int8 row-major matrix into SME dot4 row panels.
// Each k-step: z0 = rows 0..15 × 4 bytes, z1 = rows 16..31 × 4 bytes
// Bytecode: [0x15][M:u32][K:u32]
// Operands: src_i8_ptr, dst_packed_ptr
// ================================================================
.L_op_pack_rows_i8:
    ldr     w22, [x19]             // M (must be GROUP_DIM=32)
    ldr     w3, [x19, #4]          // K (stride between rows in src)
    add     x19, x19, #8
    ldr     x8, [x19], #8          // src
    ldr     x11, [x19], #8         // dst
    lsr     w4, w3, #2             // k_steps = K / 4
    // Apple Silicon: no SVE gather in any mode. Use 4-byte loads per row.
    // smstop not needed — ldr w/str w work in streaming mode.
    mov     w14, #0
    cbz     w4, .L_dispatch
.L_pack_rows_t:
    mov     x15, x11
    mov     w12, #0
.L_pr_upper:
    madd    x16, x12, x3, x8
    add     x16, x16, x14, lsl #2
    ldr     w17, [x16]
    str     w17, [x15], #4
    add     w12, w12, #1
    cmp     w12, #16
    b.lt    .L_pr_upper
    mov     w12, #16
.L_pr_lower:
    madd    x16, x12, x3, x8
    add     x16, x16, x14, lsl #2
    ldr     w17, [x16]
    str     w17, [x15], #4
    add     w12, w12, #1
    cmp     w12, #32
    b.lt    .L_pr_lower
    add     x11, x11, #128
    add     w14, w14, #1
    cmp     w14, w4
    b.lt    .L_pack_rows_t
    b       .L_dispatch
// ================================================================
// PACK_COLS_I8 (0x16)
// Pack int8 row-major matrix into SME dot4 column panels.
// Each k-step: z2 = cols 0..15 × 4 bytes, z3 = cols 16..31 × 4 bytes
// Bytecode: [0x16][N:u32][K:u32]
// Operands: src_i8_ptr, dst_packed_ptr
// ================================================================
.L_op_pack_cols_i8:
    ldr     w22, [x19]             // N (stride between rows in src)
    ldr     w3, [x19, #4]          // K
    add     x19, x19, #8
    ldr     x8, [x19], #8          // src (K×N row-major)
    ldr     x11, [x19], #8         // dst packed
    lsr     w4, w3, #2             // k_steps = K/4
    ptrue   p0.b
    mov     w14, #0                // t = 0
    cbz     w4, .L_dispatch
.L_pack_cols_t:
    // For k-step t, load 4 consecutive rows of N bytes each:
    //   row0 = B[(t*4+0)*N ..], row1 = B[(t*4+1)*N ..], etc.
    // Then interleave columns into dot4 format:
    //   dst[c*4+d] = row_d[c]
    lsl     w5, w14, #2            // t*4
    madd    x15, x5, x22, x8      // x15 = src + (t*4)*N = row0 base
    // Load 4 rows (each N bytes, but we only need 32 cols = first 32 bytes)
    // With SVLb=64, ld1b loads 64 bytes — first 32 are our columns, rest is fine to load
    ld1b    {z0.b}, p0/z, [x15]           // row t*4+0
    add     x15, x15, x22
    ld1b    {z1.b}, p0/z, [x15]           // row t*4+1
    add     x15, x15, x22
    ld1b    {z2.b}, p0/z, [x15]           // row t*4+2
    add     x15, x15, x22
    ld1b    {z3.b}, p0/z, [x15]           // row t*4+3
    // 4-way byte interleave: [r0c0,r1c0,r2c0,r3c0, r0c1,r1c1,r2c1,r3c1, ...]
    // Step 1: zip pairs at byte granularity
    zip1    z4.b, z0.b, z2.b      // z4 = [r0_0,r2_0, r0_1,r2_1, r0_2,r2_2, ...]
    zip1    z5.b, z1.b, z3.b      // z5 = [r1_0,r3_0, r1_1,r3_1, r1_2,r3_2, ...]
    // Step 2: zip the pairs to get 4-way interleave
    zip1    z6.b, z4.b, z5.b      // z6 = [r0_0,r1_0,r2_0,r3_0, r0_1,r1_1,r2_1,r3_1, ...]
    zip2    z7.b, z4.b, z5.b      // z7 = upper half columns
    // z6 has cols 0..15 in dot4 format (first 64 bytes)
    // z7 has cols 16..31 in dot4 format (next 64 bytes)
    st1b    {z6.b}, p0, [x11]
    add     x16, x11, #64
    st1b    {z7.b}, p0, [x16]
    add     x11, x11, #128
    add     w14, w14, #1
    cmp     w14, w4
    b.lt    .L_pack_cols_t
    b       .L_dispatch
// ================================================================
// SCATTER_TILE_FP32 (0x17)
// Copy GROUP_DIM×GROUP_DIM tile into a strided output matrix.
// Bytecode: [0x17][dst_row_offset:u32][dst_col_offset:u32][dst_stride_cols:u32]
// Operands: tile_src_ptr, dst_matrix_ptr
// tile_src is GROUP_DIM×GROUP_DIM contiguous, dst is strided.
// ================================================================
.L_op_scatter_tile_fp32:
    ldr     w22, [x19]             // dst_row_offset
    ldr     w3, [x19, #4]          // dst_col_offset
    ldr     w4, [x19, #8]          // dst_stride_cols (N of the full matrix)
    add     x19, x19, #12
    ldr     x8, [x19], #8          // tile src (GROUP_DIM × GROUP_DIM fp32)
    ldr     x11, [x19], #8         // dst matrix base
    ptrue   p0.s
    cntw    x9                     // SVLs = 16
    // dst_start = dst + (row_offset * stride + col_offset) * 4
    madd    w5, w22, w4, w3        // row_off * stride + col_off
    lsl     x5, x5, #2            // offset * 4 bytes
    add     x11, x11, x5           // dst + offset
    lsl     x10, x4, #2           // dst row stride in bytes = stride_cols * 4
    mov     w12, #0                // row counter
.L_scatter_row:
    // Load 32 fp32 from tile (2 z-registers worth)
    ld1w    {z0.s}, p0/z, [x8]
    ld1w    {z1.s}, p0/z, [x8, x9, lsl #2]
    // Store to strided dst (only up to GROUP_DIM cols)
    st1w    {z0.s}, p0, [x11]
    st1w    {z1.s}, p0, [x11, x9, lsl #2]
    add     x8, x8, #128          // next tile row = GROUP_DIM * 4 = 128 bytes
    add     x11, x11, x10         // next dst row = stride * 4 bytes
    add     w12, w12, #1
    cmp     w12, #32               // GROUP_DIM rows
    b.lt    .L_scatter_row
    b       .L_dispatch
// ================================================================
// TRANSPOSE_FP32 (0x18)
// Transpose M×N fp32 matrix → N×M fp32 matrix.
// For each src row i, scatter-store as column i of dst.
// dst[j*M + i] = src[i*N + j]
// Bytecode: [0x18][M:u32][N:u32]
// Operands: src_ptr, dst_ptr
// ================================================================
.L_op_transpose_fp32:
    ldr     w22, [x19]             // M (rows of src)
    ldr     w3, [x19, #4]          // N (cols of src)
    add     x19, x19, #8
    ldr     x8, [x19], #8          // src (M×N fp32 row-major)
    ldr     x11, [x19], #8         // dst (N×M fp32 row-major)
    // dst[j*M + i] = src[i*N + j]
    // Apple Silicon: no scatter stores. Use scalar str per element.
    lsl     x5, x22, #2           // M * 4 = dst row stride
    lsl     x15, x3, #2           // N * 4 = src row stride
    mov     w12, #0
.L_tr_row:
    cmp     w12, w22
    b.ge    .L_tr_done
    lsl     x16, x12, #2
    add     x16, x11, x16         // dst_col_base = dst + i*4
    mov     x17, x8               // src cursor
    mov     w14, #0
.L_tr_elem:
    cmp     w14, w3
    b.ge    .L_tr_next_row
    ldr     s0, [x17], #4
    str     s0, [x16]
    add     x16, x16, x5          // next dst row
    add     w14, w14, #1
    b       .L_tr_elem
.L_tr_next_row:
    add     x8, x8, x15
    add     w12, w12, #1
    b       .L_tr_row
.L_tr_done:
    b       .L_dispatch
// SOFTMAX_ARGMAX_FP32 (0x19)
// Fused softmax + cross-entropy backward + argmax. Processes batch rows.
// Per row: (1) find max, (2) exp(x-max)+sum, (3) normalize, (4) g_out=probs-one_hot, (5) argmax.
// Bytecode: [0x19][batch:u32][cols:u32]
// Operands: logits, probs, labels (uint8_t[batch]), g_out (batch×cols), argmax (int32[batch])
.L_op_softmax_argmax_fp32:
    ldr     w22, [x19]             // batch
    ldr     w3, [x19, #4]          // cols (elements per row, must be multiple of SVLs)
    add     x19, x19, #8
    ldr     x8, [x19], #8          // src base (logits, batch × cols fp32)
    ldr     x11, [x19], #8         // dst base (probs, batch × cols fp32)
    ldr     x16, [x19], #8         // labels base (uint8_t[batch])
    ldr     x17, [x19], #8         // g_out base (batch × cols fp32)
    ldr     x13, [x19], #8         // argmax base (int32_t[batch])
    ptrue   p0.s
    cntw    x9
    lsl     x10, x3, #2           // row stride = cols * 4 bytes
    // ── Load exp constants (once for all rows) ──
    movz    w4, #0x3FB8, lsl #16
    movk    w4, #0xAA3B
    fmov    s28, w4
    mov     z28.s, s28              // log2(e)
    movz    w4, #0x3C1D, lsl #16
    movk    w4, #0x955A
    fmov    s29, w4
    mov     z29.s, s29              // c4
    movz    w4, #0x3D63, lsl #16
    movk    w4, #0x5847
    fmov    s30, w4
    mov     z30.s, s30              // c3
    movz    w4, #0x3E75, lsl #16
    movk    w4, #0xFDF0
    fmov    s31, w4
    mov     z31.s, s31              // c2
    movz    w4, #0x3F31, lsl #16
    movk    w4, #0x7218
    fmov    s27, w4
    mov     z27.s, s27              // c1 = ln(2)
    fmov    z26.s, #1.0
    // ── Batch loop ──
    mov     w18, #0
.L_smb_row:
    cmp     w18, w22
    b.ge    .L_dispatch
    mov     x14, x8                // save row src
    mov     x15, x11               // save row dst
    // ── Pass 1: find max ──
    movz    w4, #0xFF80, lsl #16
    fmov    s16, w4
    mov     z16.s, s16
    mov     w12, w3
.L_smb_max:
    cbz     w12, .L_smb_max_done
    ld1w    {z0.s}, p0/z, [x8]
    fmax    z16.s, p0/m, z16.s, z0.s
    add     x8, x8, x9, lsl #2
    sub     w12, w12, w9
    cbnz    w12, .L_smb_max
.L_smb_max_done:
    fmaxv   s16, p0, z16.s
    mov     z16.s, s16
    // ── Pass 2: exp(x - max) + sum ──
    fmov    z17.s, #0.0
    mov     x8, x14
    mov     x11, x15
    mov     w12, w3
.L_smb_exp:
    cbz     w12, .L_smb_exp_done
    ld1w    {z0.s}, p0/z, [x8]
    fsub    z0.s, z0.s, z16.s
    fmul    z1.s, z0.s, z28.s
    frintm  z2.s, p0/m, z1.s
    fsub    z3.s, z1.s, z2.s
    fmul    z4.s, z29.s, z3.s
    fadd    z4.s, z4.s, z30.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z31.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z27.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z26.s
    fcvtzs  z5.s, p0/m, z2.s
    mov     z6.s, #-127
    smax    z5.s, p0/m, z5.s, z6.s
    add     z5.s, z5.s, #127
    lsl     z5.s, z5.s, #23
    fmul    z4.s, z4.s, z5.s
    st1w    {z4.s}, p0, [x11]
    fadd    z17.s, z17.s, z4.s
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    sub     w12, w12, w9
    b       .L_smb_exp
.L_smb_exp_done:
    // ── Pass 3: normalize ──
    faddv   s17, p0, z17.s
    fmov    s18, #1.0
    fdiv    s17, s18, s17
    mov     z17.s, s17
    mov     x11, x15
    mov     w12, w3
.L_smb_div:
    cbz     w12, .L_smb_ce_backward
    ld1w    {z0.s}, p0/z, [x11]
    fmul    z0.s, z0.s, z17.s
    st1w    {z0.s}, p0, [x11]
    add     x11, x11, x9, lsl #2
    sub     w12, w12, w9
    b       .L_smb_div
.L_smb_ce_backward:
    // ── Pass 4: cross-entropy backward: g_out = probs - one_hot(label) ──
    ld1w    {z0.s}, p0/z, [x15]     // reload normalized probs
    ldrb    w4, [x16], #1           // label for this row (advance labels ptr)
    index   z3.s, #0, #1            // [0, 1, 2, ..., 15]
    mov     z5.s, w4                // broadcast label index
    cmpeq   p3.s, p0/z, z3.s, z5.s // p3 = one active lane at label
    fmov    z6.s, #1.0
    fsub    z0.s, p3/m, z0.s, z6.s  // probs[label] -= 1.0
    st1w    {z0.s}, p0, [x17]       // store g_out
.L_smb_argmax:
    // ── Pass 5: argmax ──
    whilelt p1.s, xzr, x3
    ld1w    {z0.s}, p1/z, [x15]
    movz    w4, #0xFF80, lsl #16
    fmov    s1, w4
    mov     z1.s, s1
    sel     z0.s, p1, z0.s, z1.s
    fmaxv   s2, p0, z0.s
    mov     z2.s, s2
    fcmeq   p2.s, p0/z, z0.s, z2.s
    index   z3.s, #0, #1
    mov     z4.s, #15
    sel     z4.s, p2, z3.s, z4.s
    uminv   s4, p0, z4.s
    fmov    w4, s4
    str     w4, [x13]
    // ── Advance to next row ──
    add     x8, x14, x10
    add     x11, x15, x10
    add     x17, x17, x10          // g_out += row stride
    add     x13, x13, #4           // argmax += sizeof(int32)
    add     w18, w18, #1
    b       .L_smb_row
// ================================================================
// LUTI4 (0x1A)
// 4-bit table lookup using the SME LUTI4 instruction via ZT0.
// Loads the 64-byte table into ZT0, then processes count z-vectors
// of packed nibble indices, producing output z-vectors of looked-up values.
// Bytecode: [0x1A][count:u32][elem_size:u8]
//   elem_size: 0=8-bit (.b), 1=16-bit (.h), 2=32-bit (.s)
// Operands: table_ptr (64 bytes), indices_ptr, output_ptr
// For .b: each input byte has 2 nibbles → 2 output bytes (1v→1v, indices reinterpreted)
// For .h: luti4 {z_dst.h}, zt0, z_idx[imm] — 16-bit output per 4-bit index
// For .s: luti4 {z_dst.s}, zt0, z_idx[imm] — 32-bit output per 4-bit index
// ================================================================
.L_op_luti4:
    ldr     w22, [x19]             // count (number of z-vectors to process)
    ldrb    w3, [x19, #4]          // elem_size (0=.b, 1=.h, 2=.s)
    add     x19, x19, #5
    ldr     x8, [x19], #8          // table_ptr (64 bytes, loaded into ZT0)
    ldr     x11, [x19], #8         // indices_ptr
    ldr     x13, [x19], #8         // output_ptr
    // Load table into ZT0
    ldr     zt0, [x8]
    ptrue   p0.b
    cntb    x9                     // SVLb = 64
    cbz     w22, .L_dispatch
    // Dispatch on element size
    cmp     w3, #1
    b.eq    .L_luti4_h
    cmp     w3, #2
    b.eq    .L_luti4_s
    // Default: 8-bit lookup
.L_luti4_b:
    ld1b    {z0.b}, p0/z, [x11]
    luti4   z1.b, zt0, z0[0]
    st1b    {z1.b}, p0, [x13]
    add     x11, x11, x9
    add     x13, x13, x9
    sub     w22, w22, #1
    cbnz    w22, .L_luti4_b
    b       .L_dispatch
.L_luti4_h:
    ptrue   p0.h
    ld1b    {z0.b}, p0/z, [x11]
    luti4   z1.h, zt0, z0[0]
    st1h    {z1.h}, p0, [x13]
    add     x11, x11, x9
    cntw    x10
    lsl     x10, x10, #1           // SVLh * 2 bytes
    add     x13, x13, x10
    sub     w22, w22, #1
    cbnz    w22, .L_luti4_h
    b       .L_dispatch
.L_luti4_s:
    ptrue   p0.s
    ld1b    {z0.b}, p0/z, [x11]
    luti4   z1.s, zt0, z0[0]
    st1w    {z1.s}, p0, [x13]
    add     x11, x11, x9
    cntw    x10
    lsl     x10, x10, #2           // SVLs * 4 bytes
    add     x13, x13, x10
    sub     w22, w22, #1
    cbnz    w22, .L_luti4_s
    b       .L_dispatch
// ================================================================
// LUTI2 (0x1B)
// 2-bit table lookup using the SME LUTI2 instruction via ZT0.
// Bytecode: [0x1B][count:u32][elem_size:u8]
//   elem_size: 0=8-bit (.b), 1=16-bit (.h), 2=32-bit (.s)
// Operands: table_ptr (64 bytes), indices_ptr, output_ptr
// ================================================================
.L_op_luti2:
    ldr     w22, [x19]
    ldrb    w3, [x19, #4]
    add     x19, x19, #5
    ldr     x8, [x19], #8          // table_ptr
    ldr     x11, [x19], #8         // indices_ptr
    ldr     x13, [x19], #8         // output_ptr
    ldr     zt0, [x8]
    ptrue   p0.b
    cntb    x9
    cbz     w22, .L_dispatch
    cmp     w3, #1
    b.eq    .L_luti2_h
    cmp     w3, #2
    b.eq    .L_luti2_s
.L_luti2_b:
    ld1b    {z0.b}, p0/z, [x11]
    luti2   z1.b, zt0, z0[0]
    st1b    {z1.b}, p0, [x13]
    add     x11, x11, x9
    add     x13, x13, x9
    sub     w22, w22, #1
    cbnz    w22, .L_luti2_b
    b       .L_dispatch
.L_luti2_h:
    ptrue   p0.h
    ld1b    {z0.b}, p0/z, [x11]
    luti2   z1.h, zt0, z0[0]
    st1h    {z1.h}, p0, [x13]
    add     x11, x11, x9
    cntw    x10
    lsl     x10, x10, #1
    add     x13, x13, x10
    sub     w22, w22, #1
    cbnz    w22, .L_luti2_h
    b       .L_dispatch
.L_luti2_s:
    ptrue   p0.s
    ld1b    {z0.b}, p0/z, [x11]
    luti2   z1.s, zt0, z0[0]
    st1w    {z1.s}, p0, [x13]
    add     x11, x11, x9
    cntw    x10
    lsl     x10, x10, #2
    add     x13, x13, x10
    sub     w22, w22, #1
    cbnz    w22, .L_luti2_s
    b       .L_dispatch
// ================================================================
// DENSE_FP32 (0x1C)
// Full fp32 matmul via FMOPA: C = scale * (A @ B) [+ relu]
// Bytecode: [0x1C][M:u32][N:u32][K:u32][scale:f32][flags:u8]
//   flags bit 0: apply ReLU after matmul+scale
// Operands: A (M×K row-major fp32), B (K×N row-major fp32), C (M×N row-major fp32)
// 16×32 output tiles: za0+za1 accumulate, za2 inline transpose.
//   Caller must pad output buffer rows to ((M+31)&~31).
//   Caller must pad A columns (K dim) to ((K+15)&~15) with zeros.
// ================================================================
.L_op_dense_fp32:
    // ── Parse immediates + operands ──
    ldr     w0, [x19]              // M
    ldr     w1, [x19, #4]          // N
    ldr     w2, [x19, #8]          // K
    ldr     s20, [x19, #12]        // scale (f32)
    ldrb    w18, [x19, #16]        // flags (bit 0 = relu)
    add     x19, x19, #17
    ldr     x5, [x19], #8          // A
    ldr     x6, [x19], #8          // B
    ldr     x8, [x19], #8          // bias
    ldr     x7, [x19], #8          // C
    // ── Derived values ──
    add     w3, w0, #15
    and     w3, w3, #0xFFFFFFF0    // M_pad = (M+15) & ~15 (tile rows = 16)
    add     w4, w1, #31
    and     w4, w4, #0xFFFFFFE0    // N_pad = (N+31) & ~31 (tile cols = 32)
    lsr     w15, w2, #4            // k_blocks = K / 16
    ptrue   p0.s
    cntw    x9                     // SVLs = 16
    lsl     x17, x2, #2           // K * 4 = A row stride in bytes
    // ── Branchless ReLU threshold: z21 = relu ? 0.0 : -FLT_MAX ──
    mov     z20.s, s20             // broadcast scale
    and     w18, w18, #1
    movz    w10, #0xFF7F, lsl #16
    movk    w10, #0xFFFF           // -FLT_MAX
    cmp     w18, #0
    csel    w10, w10, wzr, eq      // relu=0 → -FLT_MAX; relu=1 → 0.0
    fmov    s21, w10
    mov     z21.s, s21
    // ── Fixed 128-byte stack frame (context only, no scratch) ──
    sub     sp, sp, #128
    stp     x5, x6, [sp, #0]      // [0] A, [8] B
    str     x7, [sp, #16]         // [16] C
    stp     w0, w1, [sp, #24]     // [24] M, [28] N
    str     w2, [sp, #32]         // [32] K
    stp     w4, w3, [sp, #40]     // [40] N_pad, [44] M_pad  (store before x3 is reused)
    str     w15, [sp, #48]        // [48] k_blocks
    lsl     x3, x1, #2            // N * 4 = B/C row stride in bytes  (now safe to clobber w3)
    stp     x3, x17, [sp, #56]    // [56] B_stride, [64] A_row_stride
    str     x8, [sp, #72]         // [72] bias
    // ── Tile row loop: ti = 0, 16, ... ──
    mov     w0, #0                 // ti = 0
.L_fp32_tile_row:
    str     w0, [sp, #80]          // save ti
    // A_tile_base = A + ti * K * 4
    ldr     x5, [sp, #0]
    ldr     w2, [sp, #32]
    mul     w10, w0, w2
    add     x5, x5, x10, lsl #2   // x5 = A + ti*K*4
    // ── Tile column loop: tj = 0, 32, ... ──
    mov     w1, #0                 // tj = 0
.L_fp32_tile_col:
    str     w1, [sp, #84]          // save tj
    // Load bias into za0/za1 accumulators (bias is 1×N, broadcast across all 16 rows)
    zero    {za2.s, za3.s}
    ldr     x8, [sp, #72]          // bias base
    ptrue   p0.s
    cntw    x9                     // SVLs = 16
    ld1w    {z0.s}, p0/z, [x8, x1, lsl #2]           // bias[tj..tj+15]    → za0 cols
    add     x10, x1, x9
    ld1w    {z4.s}, p0/z, [x8, x10, lsl #2]          // bias[tj+16..tj+31] → za1 cols
    // Broadcast bias[tj..tj+15] into all 16 rows of za0
    mov     z1.d, z0.d
    mov     z2.d, z0.d
    mov     z3.d, z0.d
    mov     w12, #0
    mova    za0h.s[w12, 0:3], {z0.s-z3.s}             // rows 0-3
    mov     w12, #4
    mova    za0h.s[w12, 0:3], {z0.s-z3.s}             // rows 4-7
    mov     w12, #8
    mova    za0h.s[w12, 0:3], {z0.s-z3.s}             // rows 8-11
    mov     w12, #12
    mova    za0h.s[w12, 0:3], {z0.s-z3.s}             // rows 12-15
    // Broadcast bias[tj+16..tj+31] into all 16 rows of za1
    mov     z5.d, z4.d
    mov     z6.d, z4.d
    mov     z7.d, z4.d
    mov     w12, #0
    mova    za1h.s[w12, 0:3], {z4.s-z7.s}             // rows 0-3
    mov     w12, #4
    mova    za1h.s[w12, 0:3], {z4.s-z7.s}             // rows 4-7
    mov     w12, #8
    mova    za1h.s[w12, 0:3], {z4.s-z7.s}             // rows 8-11
    mov     w12, #12
    mova    za1h.s[w12, 0:3], {z4.s-z7.s}             // rows 12-15
    // B_tile = B + tj * 4
    ldr     x6, [sp, #8]
    add     x6, x6, x1, lsl #2
    ldr     x3, [sp, #56]          // B_stride = N*4
    ldr     x17, [sp, #64]         // A_row_stride = K*4
    ldr     w15, [sp, #48]         // k_blocks
    ptrue   p0.s
    cntw    x9
    // x13 = ck byte offset into A rows (advances by 64 per k-block)
    mov     x13, xzr
    // ── K-block loop: process 16 columns of A per iteration ──
    cbz     w15, .L_fp32_kblock_done
.L_fp32_kblock:
    // ── Load 16 A rows into za2 for transpose ──
    zero    {za2.s}
    add     x8, x5, x13           // A_tile_base + ck*4
    mov     w12, #0
    ld1w    {z0.s}, p0/z, [x8]
    add     x8, x8, x17
    ld1w    {z1.s}, p0/z, [x8]
    add     x8, x8, x17
    ld1w    {z2.s}, p0/z, [x8]
    add     x8, x8, x17
    ld1w    {z3.s}, p0/z, [x8]
    add     x8, x8, x17
    mova    za2h.s[w12, 0:3], {z0.s-z3.s}
    mov     w12, #4
    ld1w    {z0.s}, p0/z, [x8]
    add     x8, x8, x17
    ld1w    {z1.s}, p0/z, [x8]
    add     x8, x8, x17
    ld1w    {z2.s}, p0/z, [x8]
    add     x8, x8, x17
    ld1w    {z3.s}, p0/z, [x8]
    add     x8, x8, x17
    mova    za2h.s[w12, 0:3], {z0.s-z3.s}
    mov     w12, #8
    ld1w    {z0.s}, p0/z, [x8]
    add     x8, x8, x17
    ld1w    {z1.s}, p0/z, [x8]
    add     x8, x8, x17
    ld1w    {z2.s}, p0/z, [x8]
    add     x8, x8, x17
    ld1w    {z3.s}, p0/z, [x8]
    add     x8, x8, x17
    mova    za2h.s[w12, 0:3], {z0.s-z3.s}
    mov     w12, #12
    ld1w    {z0.s}, p0/z, [x8]
    add     x8, x8, x17
    ld1w    {z1.s}, p0/z, [x8]
    add     x8, x8, x17
    ld1w    {z2.s}, p0/z, [x8]
    add     x8, x8, x17
    ld1w    {z3.s}, p0/z, [x8]
    mova    za2h.s[w12, 0:3], {z0.s-z3.s}
    // ── Extract 16 columns from za2, FMOPA into za0+za1 ──
    // Cols 0-3
    mov     w12, #0
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z0.s, z5.s
    add     x6, x6, x3
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
    add     x6, x6, x3
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z2.s, z5.s
    add     x6, x6, x3
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z3.s, z5.s
    add     x6, x6, x3
    // Cols 4-7
    mov     w12, #4
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z0.s, z5.s
    add     x6, x6, x3
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
    add     x6, x6, x3
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z2.s, z5.s
    add     x6, x6, x3
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z3.s, z5.s
    add     x6, x6, x3
    // Cols 8-11
    mov     w12, #8
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z0.s, z5.s
    add     x6, x6, x3
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
    add     x6, x6, x3
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z2.s, z5.s
    add     x6, x6, x3
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z3.s, z5.s
    add     x6, x6, x3
    // Cols 12-15
    mov     w12, #12
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z0.s, z5.s
    add     x6, x6, x3
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
    add     x6, x6, x3
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z2.s, z5.s
    add     x6, x6, x3
    ld1w    {z4.s}, p0/z, [x6]
    ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z3.s, z5.s
    add     x6, x6, x3
    // ── Advance k-block ──
    add     x13, x13, #64         // ck += 16 elements (64 bytes)
    subs    w15, w15, #1
    b.ne    .L_fp32_kblock
.L_fp32_kblock_done:
    // ── Store: extract za0+za1 → scale → clamp → C ──
    ldr     x7, [sp, #16]         // C
    ldr     w0, [sp, #80]         // ti
    ldr     w1, [sp, #84]         // tj
    ldr     w14, [sp, #28]        // N
    lsl     x10, x14, #2          // C row stride = N*4
    mul     w8, w0, w14
    add     w8, w8, w1
    add     x7, x7, x8, lsl #2   // C_tile = C + (ti*N + tj)*4
    // Column predicates
    sub     w6, w14, w1            // N - tj
    mov     w8, #32
    cmp     w6, w8
    csel    w6, w6, w8, lt
    whilelt p2.s, xzr, x6         // left cols
    sub     w8, w6, #16
    cmp     w8, #0
    csel    w8, wzr, w8, lt
    whilelt p3.s, xzr, x8         // right cols (0 if N-tj <= 16)
    ptrue   p0.s
    cntw    x9
    // 4 groups of 4 rows = 16 rows
    // Scale via fmul, relu via multi-vector fmax
    mov     z14.d, z21.d           // relu threshold into low reg (z0-z15 required)
    mov     w12, #0
    // Group 0
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    fmul    z0.s, z0.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z3.s, z3.s, z20.s
    fmul    z4.s, z4.s, z20.s
    fmul    z5.s, z5.s, z20.s
    fmul    z6.s, z6.s, z20.s
    fmul    z7.s, z7.s, z20.s
    fmax    {z0.s-z3.s}, {z0.s-z3.s}, z14.s
    fmax    {z4.s-z7.s}, {z4.s-z7.s}, z14.s
    st1w    {z0.s}, p2, [x7]
    st1w    {z4.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z1.s}, p2, [x7]
    st1w    {z5.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z2.s}, p2, [x7]
    st1w    {z6.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z3.s}, p2, [x7]
    st1w    {z7.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    // Group 1
    mov     w12, #4
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    fmul    z0.s, z0.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z3.s, z3.s, z20.s
    fmul    z4.s, z4.s, z20.s
    fmul    z5.s, z5.s, z20.s
    fmul    z6.s, z6.s, z20.s
    fmul    z7.s, z7.s, z20.s
    fmax    {z0.s-z3.s}, {z0.s-z3.s}, z14.s
    fmax    {z4.s-z7.s}, {z4.s-z7.s}, z14.s
    st1w    {z0.s}, p2, [x7]
    st1w    {z4.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z1.s}, p2, [x7]
    st1w    {z5.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z2.s}, p2, [x7]
    st1w    {z6.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z3.s}, p2, [x7]
    st1w    {z7.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    // Group 2
    mov     w12, #8
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    fmul    z0.s, z0.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z3.s, z3.s, z20.s
    fmul    z4.s, z4.s, z20.s
    fmul    z5.s, z5.s, z20.s
    fmul    z6.s, z6.s, z20.s
    fmul    z7.s, z7.s, z20.s
    fmax    {z0.s-z3.s}, {z0.s-z3.s}, z14.s
    fmax    {z4.s-z7.s}, {z4.s-z7.s}, z14.s
    st1w    {z0.s}, p2, [x7]
    st1w    {z4.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z1.s}, p2, [x7]
    st1w    {z5.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z2.s}, p2, [x7]
    st1w    {z6.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z3.s}, p2, [x7]
    st1w    {z7.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    // Group 3
    mov     w12, #12
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    fmul    z0.s, z0.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z3.s, z3.s, z20.s
    fmul    z4.s, z4.s, z20.s
    fmul    z5.s, z5.s, z20.s
    fmul    z6.s, z6.s, z20.s
    fmul    z7.s, z7.s, z20.s
    fmax    {z0.s-z3.s}, {z0.s-z3.s}, z14.s
    fmax    {z4.s-z7.s}, {z4.s-z7.s}, z14.s
    st1w    {z0.s}, p2, [x7]
    st1w    {z4.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z1.s}, p2, [x7]
    st1w    {z5.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z2.s}, p2, [x7]
    st1w    {z6.s}, p3, [x7, x9, lsl #2]
    add     x7, x7, x10
    st1w    {z3.s}, p2, [x7]
    st1w    {z7.s}, p3, [x7, x9, lsl #2]
    // ── Advance tile column ──
    ldr     w1, [sp, #84]          // tj
    ldr     w4, [sp, #40]          // N_pad
    add     w1, w1, #32
    cmp     w1, w4
    b.lt    .L_fp32_tile_col
    // ── Advance tile row ──
    ldr     w0, [sp, #80]          // ti
    ldr     w3, [sp, #44]          // M_pad
    add     w0, w0, #16
    cmp     w0, w3
    b.lt    .L_fp32_tile_row
.L_fp32_tile_done:
    add     sp, sp, #128
    b       .L_dispatch