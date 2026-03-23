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
    .long   .L_exit          - .L_jump_table  // 0x00 reserved (exit)
    .long   .L_op_zero_za    - .L_jump_table  // 0x01 zero_za
    .long   .L_op_acc_smopa  - .L_jump_table  // 0x02 acc_smopa
    .long   .L_op_acc_umopa  - .L_jump_table  // 0x03 acc_umopa
    .long   .L_op_acc_usmopa - .L_jump_table  // 0x04 acc_usmopa
    .long   .L_op_acc_sumopa - .L_jump_table  // 0x05 acc_sumopa
    .long   .L_op_store_tiles - .L_jump_table // 0x06 store_tiles
    .long   .L_op_smopa_2x2  - .L_jump_table // 0x07 smopa_2x2
    .long   .L_op_umopa_2x2  - .L_jump_table // 0x08 umopa_2x2
    .long   .L_op_usmopa_2x2 - .L_jump_table // 0x09 usmopa_2x2
    .long   .L_op_load_bias  - .L_jump_table  // 0x0A load_bias
    .long   .L_op_scale_store - .L_jump_table // 0x0B scale_store
    .long   .L_op_elementwise_add_fp32  - .L_jump_table // 0x0C elementwise_add_fp32
    .long   .L_op_elementwise_scaled_add_fp32 - .L_jump_table // 0x0D elementwise_scaled_add_fp32
    .long   .L_op_elementwise_mul_fp32 - .L_jump_table // 0x0E elementwise_mul_fp32
    .long   .L_op_relu_backward_fp32 - .L_jump_table // 0x0F relu_backward_fp32
    .long   .L_op_scatter_tile_fp32 - .L_jump_table // 0x10 scatter_tile_fp32
    .long   .L_op_transpose_fp32 - .L_jump_table // 0x11 transpose_fp32
    .long   .L_op_softmax_argmax_fp32 - .L_jump_table // 0x12 softmax_argmax_fp32
    .long   .L_op_luti4 - .L_jump_table      // 0x13 luti4_op
    .long   .L_op_luti2 - .L_jump_table      // 0x14 luti2_op
    .long   .L_op_dense_fp32 - .L_jump_table  // 0x15 dense_fp32
    .long   .L_op_count_matches - .L_jump_table // 0x16 count_matches
    .long   .L_op_reduce_sum_fp32 - .L_jump_table // 0x17 reduce_sum_fp32
    .long   .L_op_dense_i8 - .L_jump_table  // 0x18 dense_i8 (INT8 matmul via SMOPA)
    .long   .L_op_quantize_fp32_i8 - .L_jump_table // 0x19 quantize_fp32_i8
    .long   .L_op_dequantize_i8_fp32 - .L_jump_table // 0x1A dequantize_i8_fp32
    .long   .L_op_pack_b_i8 - .L_jump_table   // 0x1B pack_b_i8
    .long   .L_op_quantize_fp32_i8_channelwise - .L_jump_table // 0x1C quantize_fp32_i8_channelwise
    .long   .L_op_transpose_i8 - .L_jump_table // 0x1D transpose_i8
    .long   .L_op_dense_u8s8 - .L_jump_table  // 0x1E dense_u8s8 (UINT8×INT8 matmul via USMOPA)
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
// ================================================================
// DEQUANTIZE_U8_FP32 (0x1D)
// out[i] = (float)src_u8[i] * scale
// Bytecode: [0x1D][count:u32][scale:f32]
// Operands: src_u8_ptr, dst_fp32_ptr
// ================================================================
.L_op_dequantize_u8_fp32:
    ldr     w22, [x19]
    ldr     s16, [x19, #4]
    add     x19, x19, #8
    ldr     x8, [x19], #8
    ldr     x11, [x19], #8
    ptrue   p0.s
    mov     z16.s, s16
    cntw    x9
    cbz     w22, .L_dispatch
.L_dequant_u8_loop:
    ld1b    {z0.s}, p0/z, [x8]
    ucvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z16.s
    st1w    {z0.s}, p0, [x11]
    add     x8, x8, x9
    add     x11, x11, x9, lsl #2
    sub     w22, w22, w9
    cbnz    w22, .L_dequant_u8_loop
    b       .L_dispatch
// ================================================================
// COUNT_MATCHES (0x1F) — compare int32 pred[] vs uint8 labels[]
// Encoding: [0x1F] [count: u32]
// Operands: pred(int32_t*), labels(uint8_t*), result(int32_t*)
// ================================================================
.L_op_count_matches:
    ldr     w22, [x19]             // count (u32)
    add     x19, x19, #4           // advance IP past immediate
    ldr     x8, [x19], #8         // pred ptr (int32_t*)
    ldr     x11, [x19], #8        // labels ptr (uint8_t*)
    ldr     x12, [x19], #8        // result ptr (int32_t*)
    ptrue   p0.s                   // all-true predicate for .s lanes
    mov     x9, #0                 // accumulator for match count
    mov     x10, #0                // loop index
    cntw    x13                    // SVLs = elements per z register (16 on M4)
    cbz     w22, .L_cm_store
    whilelt p1.s, x10, x22        // initial predicate for tail handling
.L_cm_loop:
    ld1w    {z0.s}, p1/z, [x8, x10, lsl #2]  // pred[i..i+15] as int32
    ld1b    {z1.s}, p1/z, [x11, x10]          // labels[i..i+15] zero-extended u8→i32
    cmpeq   p2.s, p1/z, z0.s, z1.s            // p2 = (pred == label) masked by p1
    cntp    x14, p1, p2.s                      // count active matches
    add     x9, x9, x14                        // accumulate
    add     x10, x10, x13                      // advance by SVLs
    whilelt p1.s, x10, x22                     // update predicate
    b.first .L_cm_loop
.L_cm_store:
    str     w9, [x12]              // *result = total matches
    b       .L_dispatch
// ================================================================
// REDUCE_SUM_FP32 (0x20) — horizontal sum of fp32 array
// Bytecode: [0x20][count:u32]
// Operands: src(float*), result(float*)
// ================================================================
.L_op_reduce_sum_fp32:
    ldr     w22, [x19]             // count (u32)
    add     x19, x19, #4           // advance IP past immediate
    ldr     x8, [x19], #8         // src ptr (float*)
    ldr     x12, [x19], #8        // result ptr (float*)
    ptrue   p0.s                   // all-true predicate for .s lanes
    fmov    z0.s, #0.0             // accumulator vector
    mov     x10, #0                // loop index
    cbz     w22, .L_rs_store
    whilelt p1.s, x10, x22        // initial predicate for tail handling
.L_rs_loop:
    ld1w    {z1.s}, p1/z, [x8, x10, lsl #2]  // src[i..i+15]
    fadd    z0.s, p1/m, z0.s, z1.s            // accumulate (masked by p1)
    incw    x10                                // advance by SVLs
    whilelt p1.s, x10, x22                    // update predicate
    b.first .L_rs_loop
.L_rs_store:
    faddv   s0, p0, z0.s          // horizontal reduction → scalar
    str     s0, [x12]             // *result = sum
    b       .L_dispatch
// ================================================================
// QUANTIZE_FP32_I8 (0x19)
// Per-tensor symmetric quantization: i8 = clamp(round(x / scale), -127, 127)
// Computes scale = max(abs(x)) / 127.0, then quantizes.
// Bytecode: [0x19][count:u32]
// Operands: src_fp32, dst_i8, scale_out (float*)
// Writes computed scale to scale_out for later dequantization.
// count must be a multiple of SVLs (16).
// ================================================================
.L_op_quantize_fp32_i8:
    ldr     w22, [x19]             // count (u32, multiple of 16)
    add     x19, x19, #4
    ldr     x8, [x19], #8         // src fp32
    ldr     x11, [x19], #8        // dst i8
    ldr     x12, [x19], #8        // scale_out (float*)
    ptrue   p0.s
    cntw    x9                     // SVLs = 16
    // ── Pass 1: find max(abs(x)) across all elements ──
    fmov    z16.s, #0.0            // accumulator for max(abs)
    mov     x10, #0
    mov     x13, x8                // save src base
.L_qi8_absmax:
    ld1w    {z0.s}, p0/z, [x8]
    fabs    z0.s, p0/m, z0.s
    fmax    z16.s, p0/m, z16.s, z0.s
    add     x8, x8, x9, lsl #2
    add     x10, x10, x9
    cmp     w10, w22
    b.lt    .L_qi8_absmax
    fmaxv   s16, p0, z16.s        // horizontal max → scalar
    // ── Compute scale = absmax / 127.0, inv_scale = 127.0 / absmax ──
    fmov    s17, #1.0
    movz    w4, #0x42FE, lsl #16   // 127.0f = 0x42FE0000
    fmov    s18, w4
    // Guard against zero absmax (all-zero input → scale=1.0, inv=127.0)
    fcmp    s16, #0.0
    b.eq    .L_qi8_zero_scale
    fdiv    s19, s16, s18          // scale = absmax / 127.0
    fdiv    s17, s18, s16          // inv_scale = 127.0 / absmax
    b       .L_qi8_do_quant
.L_qi8_zero_scale:
    fmov    s19, #1.0              // scale = 1.0 for zero input
    fmov    s17, s18               // inv_scale = 127.0
.L_qi8_do_quant:
    str     s19, [x12]            // write scale to scale_out
    mov     z17.s, s17             // broadcast inv_scale
    // ── Clamp bounds ──
    movz    w4, #0xC2FE, lsl #16   // -127.0f
    fmov    s18, w4
    mov     z18.s, s18             // z18 = -127.0 (lo)
    movz    w4, #0x42FE, lsl #16   // +127.0f
    fmov    s19, w4
    mov     z19.s, s19             // z19 = +127.0 (hi)
    // ── Pass 2: quantize ──
    mov     x8, x13                // restore src base
    mov     x10, #0
.L_qi8_quant:
    ld1w    {z0.s}, p0/z, [x8]
    // x_scaled = x * inv_scale
    fmul    z0.s, z0.s, z17.s
    // clamp to [-127, 127]
    fmax    z0.s, p0/m, z0.s, z18.s
    fmin    z0.s, p0/m, z0.s, z19.s
    // round to nearest
    frintn  z0.s, p0/m, z0.s
    // convert to int32
    fcvtzs  z0.s, p0/m, z0.s
    // narrow int32 → int8 and store
    st1b    {z0.s}, p0, [x11]
    add     x8, x8, x9, lsl #2    // src += SVLs * 4
    add     x11, x11, x9           // dst += SVLs (1 byte per element)
    add     x10, x10, x9
    cmp     w10, w22
    b.lt    .L_qi8_quant
    b       .L_dispatch
// ================================================================
// DEQUANTIZE_I8_FP32 (0x1A)
// Dequantize signed int8 to fp32: out = (float)i8 * scale
// Bytecode: [0x1A][count:u32][scale:f32]
// Operands: src_i8, dst_fp32
// count must be a multiple of SVLs (16).
// ================================================================
.L_op_dequantize_i8_fp32:
    ldr     w22, [x19]
    ldr     s16, [x19, #4]
    add     x19, x19, #8
    ldr     x8, [x19], #8         // src i8
    ldr     x11, [x19], #8        // dst fp32
    ptrue   p0.s
    mov     z16.s, s16             // broadcast scale
    cntw    x9
    cbz     w22, .L_dispatch
.L_dequant_i8_loop:
    ld1sb   {z0.s}, p0/z, [x8]    // sign-extend i8→i32
    scvtf   z0.s, p0/m, z0.s      // int32 → float32
    fmul    z0.s, z0.s, z16.s     // out = float_val * scale
    st1w    {z0.s}, p0, [x11]
    add     x8, x8, x9             // src += SVLs bytes
    add     x11, x11, x9, lsl #2  // dst += SVLs * 4
    sub     w22, w22, w9
    cbnz    w22, .L_dequant_i8_loop
    b       .L_dispatch
// ================================================================
// DENSE_I8 (0x18)
// INT8 matmul via SMOPA: C_fp32 = dequant(A_i8 @ B_packed_i8) + bias [+relu]
// 16x32 output tiles (za0=left 16x16, za1=right 16x16).
// Uses za2 for inline A-transpose (same trick as dense_fp32 but for bytes).
//
// A is M×K_pad row-major signed int8 (K_pad = ceil(K/64)*64, zero-padded).
// B_packed is in SMOPA panel format:
//   For col_panel p (16 cols), k_group g (4 k-values):
//     64 bytes at offset (p * k_groups + g) * 64
//     byte[c*4+d] = B[g*4+d, p*16+c], c=0..15, d=0..3
//   k_groups = K_pad / 4
//
// Bytecode: [0x18][M:u32][N:u32][K:u32][scale_a:f32][scale_b:f32][flags:u8]
//   flags bit 0: apply ReLU after dequant
// Operands: A_i8, B_packed_i8, bias_fp32 (1×N), C_fp32
// ================================================================
.L_op_dense_i8:
    // ── Parse immediates + operands ──
    ldr     w0, [x19]              // M
    ldr     w1, [x19, #4]          // N
    ldr     w2, [x19, #8]          // K
    ldr     s20, [x19, #12]        // scale_a (f32)
    ldr     s22, [x19, #16]        // scale_b (f32)
    ldrb    w18, [x19, #20]        // flags (bit 0 = relu)
    add     x19, x19, #21
    ldr     x5, [x19], #8          // A_i8 (M×K_pad row-major)
    ldr     x6, [x19], #8          // B_packed_i8 (SMOPA panel format)
    ldr     x8, [x19], #8          // bias_fp32 (1×N)
    ldr     x7, [x19], #8          // C_fp32 (M×N output)
    // ── Derived values ──
    add     w3, w0, #15
    and     w3, w3, #0xFFFFFFF0    // M_pad = (M+15) & ~15
    add     w4, w1, #31
    and     w4, w4, #0xFFFFFFE0    // N_pad = (N+31) & ~31
    add     w15, w2, #63
    and     w15, w15, #0xFFFFFFC0  // K_pad = (K+63) & ~63
    lsr     w16, w15, #6           // outer_k_blocks = K_pad / 64
    lsr     w14, w15, #2           // k_groups_total = K_pad / 4
    fmul    s20, s20, s22          // combined dequant scale = scale_a * scale_b
    ptrue   p0.s
    cntw    x9
    mov     z20.s, s20             // broadcast combined scale
    // ── Branchless ReLU ──
    and     w18, w18, #1
    movz    w10, #0xFF7F, lsl #16
    movk    w10, #0xFFFF           // -FLT_MAX
    cmp     w18, #0
    csel    w10, w10, wzr, eq
    fmov    s21, w10
    mov     z21.s, s21
    // ── Stack frame: 96 bytes ──
    sub     sp, sp, #96
    stp     x5, x6, [sp, #0]      // [0] A, [8] B_packed
    str     x7, [sp, #16]         // [16] C
    stp     w0, w1, [sp, #24]     // [24] M, [28] N
    str     w15, [sp, #32]        // [32] K_pad
    stp     w4, w3, [sp, #36]     // [36] N_pad, [40] M_pad
    stp     w16, w14, [sp, #44]   // [44] outer_k_blocks, [48] k_groups_total
    str     x8, [sp, #56]         // [56] bias
    mov     x17, x15
    str     x17, [sp, #64]        // [64] A_row_stride = K_pad
    // [72] ti, [76] tj (loop vars)
    // ── Tile row loop ──
    mov     w0, #0
.L_i8_tile_row:
    str     w0, [sp, #72]
    ldr     x5, [sp, #0]
    ldr     x17, [sp, #64]
    mul     w10, w0, w17
    add     x5, x5, x10           // A_tile_base = A + ti*K_pad
    mov     w1, #0
.L_i8_tile_col:
    str     w1, [sp, #76]
    zero    {za0.s, za1.s}
    ldr     w16, [sp, #44]         // outer_k_blocks
    ldr     x17, [sp, #64]
    mov     x13, xzr
    mov     w24, #0
    cbz     w16, .L_i8_k_done
.L_i8_k_outer:
    // ── Load 16 A rows into za2 (.s horizontal slices) ──
    zero    {za2.s}
    ptrue   p0.s
    add     x8, x5, x13
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
    // ── B_packed panel addressing ──
    ldr     x6, [sp, #8]
    ldr     w14, [sp, #48]
    ldr     w1, [sp, #76]
    lsr     w23, w1, #4
    add     w26, w23, #1
    mul     w10, w23, w14
    lsl     x10, x10, #6
    add     x22, x6, x10          // left_panel_base
    mul     w10, w26, w14
    lsl     x10, x10, #6
    add     x23, x6, x10          // right_panel_base
    ptrue   p0.b
    // ── Column group 0 (cols 0-3) ──
    mov     w12, #0
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    add     w10, w24, #0
    lsl     x10, x10, #6
    add     x8, x22, x10
    add     x11, x23, x10
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z0.b, z4.b
    smopa   za1.s, p0/m, p0/m, z0.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z1.b, z4.b
    smopa   za1.s, p0/m, p0/m, z1.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z2.b, z4.b
    smopa   za1.s, p0/m, p0/m, z2.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z3.b, z4.b
    smopa   za1.s, p0/m, p0/m, z3.b, z5.b
    // ── Column group 1 (cols 4-7) ──
    mov     w12, #4
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    add     w10, w24, #4
    lsl     x10, x10, #6
    add     x8, x22, x10
    add     x11, x23, x10
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z0.b, z4.b
    smopa   za1.s, p0/m, p0/m, z0.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z1.b, z4.b
    smopa   za1.s, p0/m, p0/m, z1.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z2.b, z4.b
    smopa   za1.s, p0/m, p0/m, z2.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z3.b, z4.b
    smopa   za1.s, p0/m, p0/m, z3.b, z5.b
    // ── Column group 2 (cols 8-11) ──
    mov     w12, #8
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    add     w10, w24, #8
    lsl     x10, x10, #6
    add     x8, x22, x10
    add     x11, x23, x10
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z0.b, z4.b
    smopa   za1.s, p0/m, p0/m, z0.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z1.b, z4.b
    smopa   za1.s, p0/m, p0/m, z1.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z2.b, z4.b
    smopa   za1.s, p0/m, p0/m, z2.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z3.b, z4.b
    smopa   za1.s, p0/m, p0/m, z3.b, z5.b
    // ── Column group 3 (cols 12-15) ──
    mov     w12, #12
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    add     w10, w24, #12
    lsl     x10, x10, #6
    add     x8, x22, x10
    add     x11, x23, x10
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z0.b, z4.b
    smopa   za1.s, p0/m, p0/m, z0.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z1.b, z4.b
    smopa   za1.s, p0/m, p0/m, z1.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z2.b, z4.b
    smopa   za1.s, p0/m, p0/m, z2.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    smopa   za0.s, p0/m, p0/m, z3.b, z4.b
    smopa   za1.s, p0/m, p0/m, z3.b, z5.b
    // ── Advance K ──
    add     x13, x13, #64
    add     w24, w24, #16
    subs    w16, w16, #1
    b.ne    .L_i8_k_outer
.L_i8_k_done:
    // ── Store: extract za0+za1 → scvtf → fmul scale → fadd bias → relu → C ──
    ptrue   p0.s
    cntw    x9
    ldr     x8, [sp, #56]         // bias
    ldr     w1, [sp, #76]         // tj
    ld1w    {z16.s}, p0/z, [x8, x1, lsl #2]
    add     x10, x1, x9
    ld1w    {z17.s}, p0/z, [x8, x10, lsl #2]
    ldr     x7, [sp, #16]
    ldr     w0, [sp, #72]
    ldr     w14, [sp, #28]
    lsl     x10, x14, #2
    mul     w8, w0, w14
    add     w8, w8, w1
    add     x7, x7, x8, lsl #2
    sub     w6, w14, w1
    mov     w8, #32
    cmp     w6, w8
    csel    w6, w6, w8, lt
    whilelt p2.s, xzr, x6
    sub     w8, w6, #16
    cmp     w8, #0
    csel    w8, wzr, w8, lt
    whilelt p3.s, xzr, x8
    ptrue   p0.s
    cntw    x9
    mov     z14.d, z21.d
    // ── Store macro: scvtf → fmul → fadd bias → relu (repeated for 4 groups) ──
    // Group 0 (rows 0-3)
    mov     w12, #0
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z20.s
    fadd    z0.s, z0.s, z16.s
    scvtf   z1.s, p0/m, z1.s
    fmul    z1.s, z1.s, z20.s
    fadd    z1.s, z1.s, z16.s
    scvtf   z2.s, p0/m, z2.s
    fmul    z2.s, z2.s, z20.s
    fadd    z2.s, z2.s, z16.s
    scvtf   z3.s, p0/m, z3.s
    fmul    z3.s, z3.s, z20.s
    fadd    z3.s, z3.s, z16.s
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z20.s
    fadd    z4.s, z4.s, z17.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z20.s
    fadd    z5.s, z5.s, z17.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z20.s
    fadd    z6.s, z6.s, z17.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z20.s
    fadd    z7.s, z7.s, z17.s
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
    // Group 1 (rows 4-7)
    mov     w12, #4
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z20.s
    fadd    z0.s, z0.s, z16.s
    scvtf   z1.s, p0/m, z1.s
    fmul    z1.s, z1.s, z20.s
    fadd    z1.s, z1.s, z16.s
    scvtf   z2.s, p0/m, z2.s
    fmul    z2.s, z2.s, z20.s
    fadd    z2.s, z2.s, z16.s
    scvtf   z3.s, p0/m, z3.s
    fmul    z3.s, z3.s, z20.s
    fadd    z3.s, z3.s, z16.s
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z20.s
    fadd    z4.s, z4.s, z17.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z20.s
    fadd    z5.s, z5.s, z17.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z20.s
    fadd    z6.s, z6.s, z17.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z20.s
    fadd    z7.s, z7.s, z17.s
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
    // Group 2 (rows 8-11)
    mov     w12, #8
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z20.s
    fadd    z0.s, z0.s, z16.s
    scvtf   z1.s, p0/m, z1.s
    fmul    z1.s, z1.s, z20.s
    fadd    z1.s, z1.s, z16.s
    scvtf   z2.s, p0/m, z2.s
    fmul    z2.s, z2.s, z20.s
    fadd    z2.s, z2.s, z16.s
    scvtf   z3.s, p0/m, z3.s
    fmul    z3.s, z3.s, z20.s
    fadd    z3.s, z3.s, z16.s
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z20.s
    fadd    z4.s, z4.s, z17.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z20.s
    fadd    z5.s, z5.s, z17.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z20.s
    fadd    z6.s, z6.s, z17.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z20.s
    fadd    z7.s, z7.s, z17.s
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
    // Group 3 (rows 12-15)
    mov     w12, #12
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z20.s
    fadd    z0.s, z0.s, z16.s
    scvtf   z1.s, p0/m, z1.s
    fmul    z1.s, z1.s, z20.s
    fadd    z1.s, z1.s, z16.s
    scvtf   z2.s, p0/m, z2.s
    fmul    z2.s, z2.s, z20.s
    fadd    z2.s, z2.s, z16.s
    scvtf   z3.s, p0/m, z3.s
    fmul    z3.s, z3.s, z20.s
    fadd    z3.s, z3.s, z16.s
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z20.s
    fadd    z4.s, z4.s, z17.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z20.s
    fadd    z5.s, z5.s, z17.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z20.s
    fadd    z6.s, z6.s, z17.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z20.s
    fadd    z7.s, z7.s, z17.s
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
    ldr     w1, [sp, #76]
    ldr     w4, [sp, #36]          // N_pad
    add     w1, w1, #32
    cmp     w1, w4
    b.lt    .L_i8_tile_col
    // ── Advance tile row ──
    ldr     w0, [sp, #72]
    ldr     w3, [sp, #40]          // M_pad
    add     w0, w0, #16
    cmp     w0, w3
    b.lt    .L_i8_tile_row
.L_i8_tile_done:
    add     sp, sp, #96
    b       .L_dispatch
// ================================================================
// PACK_B_I8 (0x1B)
// Pack a K×N row-major int8 matrix B into SMOPA panel format.
// Panel format (matching dense_i8 expectations):
//   For col_panel p (16 cols), k_group g (4 k-values, d=0..3, c=0..15):
//     output[(p * k_groups + g) * 64 + c*4 + d] = B[(g*4+d)*N + p*16+c]
//   k_groups = K_pad / 4, K_pad = ceil(K/4)*4
//   N_pad = ceil(N/16)*16
//
// Bytecode: [0x1B][K:u32][N:u32]
// Operands: src_i8 (K×N row-major), dst_packed_i8
// ================================================================
.L_op_pack_b_i8:
    ldr     w22, [x19]             // K
    ldr     w3, [x19, #4]          // N
    add     x19, x19, #8
    ldr     x8, [x19], #8         // src_i8 base
    ldr     x11, [x19], #8        // dst_packed_i8 base
    // ── Derived values ──
    add     w4, w22, #3
    and     w4, w4, #0xFFFFFFFC    // K_pad = ceil(K/4)*4
    lsr     w5, w4, #2             // k_groups = K_pad / 4
    add     w6, w3, #15
    and     w6, w6, #0xFFFFFFF0    // N_pad = ceil(N/16)*16
    lsr     w7, w6, #4             // num_panels = N_pad / 16
    cntb    x9                     // SVLb = 64
    // ── Panel loop: iterate over column panels ──
    mov     w14, #0                // panel index
.L_pack_panel:
    cmp     w14, w7
    b.ge    .L_pack_done
    // panel_col_start = panel * 16
    lsl     w15, w14, #4
    // dst_panel_base = dst + panel * k_groups * 64
    mul     w10, w14, w5
    lsl     x10, x10, #6
    add     x12, x11, x10
    // ── K-group loop: iterate over groups of 4 K rows ──
    mov     w16, #0                // k_group index
.L_pack_kgroup:
    cmp     w16, w5
    b.ge    .L_pack_next_panel
    // k_base = k_group * 4
    lsl     w17, w16, #2
    // For this k_group, we need to load 4 rows (d=0..3), each starting at
    // col panel_col_start, 16 bytes wide.
    // Then 4-way interleave: output[c*4+d] = row_d[c]
    // Load row d: src[(k_base+d)*N + panel_col_start], or zero if k_base+d >= K
    // ── Compute row pointers, zeroing if beyond K ──
    ptrue   p0.b
    // Create predicate for valid columns: whilelt over [panel_col_start, N)
    // but we want exactly 16 bytes
    add     w10, w15, #16
    cmp     w10, w3
    b.le    .L_pack_full_cols
    // Partial column panel — need predicate for valid cols
    mov     x10, x15
    whilelt p1.b, x10, x3         // p1 = mask for valid columns within 16
    b       .L_pack_load_rows
.L_pack_full_cols:
    ptrue   p1.b, vl16            // all 16 columns valid
.L_pack_load_rows:
    // Row 0: k_base+0
    add     w10, w17, #0
    cmp     w10, w22               // if k_base+0 >= K, zero
    b.ge    .L_pack_r0_zero
    mul     w10, w10, w3           // row_offset = (k_base+0)*N
    add     x10, x8, x10
    add     x10, x10, x15         // + panel_col_start
    ld1b    {z0.b}, p1/z, [x10]
    b       .L_pack_r1
.L_pack_r0_zero:
    mov     z0.d, #0
.L_pack_r1:
    add     w10, w17, #1
    cmp     w10, w22
    b.ge    .L_pack_r1_zero
    mul     w10, w10, w3
    add     x10, x8, x10
    add     x10, x10, x15
    ld1b    {z1.b}, p1/z, [x10]
    b       .L_pack_r2
.L_pack_r1_zero:
    mov     z1.d, #0
.L_pack_r2:
    add     w10, w17, #2
    cmp     w10, w22
    b.ge    .L_pack_r2_zero
    mul     w10, w10, w3
    add     x10, x8, x10
    add     x10, x10, x15
    ld1b    {z2.b}, p1/z, [x10]
    b       .L_pack_r3
.L_pack_r2_zero:
    mov     z2.d, #0
.L_pack_r3:
    add     w10, w17, #3
    cmp     w10, w22
    b.ge    .L_pack_r3_zero
    mul     w10, w10, w3
    add     x10, x8, x10
    add     x10, x10, x15
    ld1b    {z3.b}, p1/z, [x10]
    b       .L_pack_interleave
.L_pack_r3_zero:
    mov     z3.d, #0
.L_pack_interleave:
    // 4-way interleave: we need output[c*4+d] = row_d[c] for c=0..15, d=0..3
    // z0 = [r0c0, r0c1, ..., r0c15, 0, ...]  (only first 16 bytes populated)
    // z1 = [r1c0, r1c1, ..., r1c15, 0, ...]
    // z2 = [r2c0, r2c1, ..., r2c15, 0, ...]
    // z3 = [r3c0, r3c1, ..., r3c15, 0, ...]
    // Target: [r0c0,r1c0,r2c0,r3c0, r0c1,r1c1,r2c1,r3c1, ...]
    // zip1/zip2 at .b interleaves adjacent byte pairs:
    // zip1(z0,z1) = [r0c0,r1c0, r0c1,r1c1, r0c2,r1c2, ...]  (low halves)
    // zip2(z0,z1) = [r0c32,r1c32, ...]  (high halves — zeroes for us)
    // zip1(z2,z3) = [r2c0,r3c0, r2c1,r3c1, ...]
    // Then zip at .h:
    // zip1(pair01, pair23).h = [r0c0,r1c0,r2c0,r3c0, r0c1,r1c1,r2c1,r3c1, ...]
    zip1    z4.b, z0.b, z1.b      // [r0c0,r1c0, r0c1,r1c1, ...] low 32 bytes
    zip1    z5.b, z2.b, z3.b      // [r2c0,r3c0, r2c1,r3c1, ...] low 32 bytes
    zip1    z6.h, z4.h, z5.h      // [r0c0,r1c0,r2c0,r3c0, r0c1,r1c1,r2c1,r3c1, ...] full 64 bytes
    // z6 holds all 64 bytes of interleaved output: [r0c0,r1c0,r2c0,r3c0, ..., r0c15,r1c15,r2c15,r3c15]
    // z7 is unused (zip2 high halves are zero since only 16 of 64 input bytes were populated)
    // Store all 64 bytes in one shot using p0 (ptrue .b from earlier)
    lsl     x10, x16, #6          // k_group * 64
    add     x13, x12, x10
    st1b    {z6.b}, p0, [x13]
    // ── Next k_group ──
    add     w16, w16, #1
    b       .L_pack_kgroup
.L_pack_next_panel:
    add     w14, w14, #1
    b       .L_pack_panel
.L_pack_done:
    b       .L_dispatch
// ================================================================
// QUANTIZE_FP32_I8_CHANNELWISE (0x1C)
// Per-row (per-channel) symmetric quantization: each row gets its own scale.
// For each row: scale = max(abs(row)) / 127, out_i8 = clamp(round(x / scale), -127, 127)
// Bytecode: [0x1C][rows:u32][cols:u32]
// Operands: src_fp32 (rows×cols), dst_i8 (rows×cols), scales_out (float[rows])
// cols must be a multiple of SVLs (16).
// ================================================================
.L_op_quantize_fp32_i8_channelwise:
    ldr     w22, [x19]             // rows
    ldr     w3, [x19, #4]          // cols (multiple of SVLs=16)
    add     x19, x19, #8
    ldr     x8, [x19], #8         // src fp32 base
    ldr     x11, [x19], #8        // dst i8 base
    ldr     x12, [x19], #8        // scales_out (float*)
    ptrue   p0.s
    cntw    x9                     // SVLs = 16
    lsl     x15, x3, #2           // src row stride = cols * 4 bytes
    // ── Clamp constant: 127.0f ──
    movz    w4, #0x42FE, lsl #16   // 127.0f = 0x42FE0000
    fmov    s18, w4
    movz    w4, #0xC2FE, lsl #16   // -127.0f
    fmov    s23, w4
    // ── Row loop ──
    mov     w14, #0                // row index
.L_cq_row:
    cmp     w14, w22
    b.ge    .L_cq_done
    // ── Pass 1: find max(abs(x)) across this row ──
    mov     z16.d, #0              // accumulator for max(abs)
    mov     x10, #0                // column counter
    mov     x13, x8                // save row src base
.L_cq_absmax:
    ld1w    {z0.s}, p0/z, [x8]
    fabs    z0.s, p0/m, z0.s
    fmax    z16.s, p0/m, z16.s, z0.s
    add     x8, x8, x9, lsl #2
    add     x10, x10, x9
    cmp     w10, w3
    b.lt    .L_cq_absmax
    fmaxv   s16, p0, z16.s        // horizontal max -> scalar
    // ── Compute scale = absmax / 127.0, inv_scale = 127.0 / absmax ──
    fcmp    s16, #0.0
    b.eq    .L_cq_zero_scale
    fdiv    s19, s16, s18          // scale = absmax / 127.0
    fdiv    s17, s18, s16          // inv_scale = 127.0 / absmax
    b       .L_cq_do_quant
.L_cq_zero_scale:
    fmov    s19, #1.0              // scale = 1.0 for zero row
    fmov    s17, s18               // inv_scale = 127.0
.L_cq_do_quant:
    str     s19, [x12]            // write scale to scales_out[row]
    add     x12, x12, #4          // advance scales_out pointer
    mov     z17.s, s17             // broadcast inv_scale
    // ── Clamp bounds ──
    mov     z18.s, s23             // z18 = -127.0 (lo)
    movz    w4, #0x42FE, lsl #16   // +127.0f
    fmov    s24, w4
    mov     z19.s, s24             // z19 = +127.0 (hi)
    // ── Pass 2: quantize this row ──
    mov     x8, x13                // restore row src base
    mov     x10, #0
.L_cq_quant:
    ld1w    {z0.s}, p0/z, [x8]
    fmul    z0.s, z0.s, z17.s     // x_scaled = x * inv_scale
    fmax    z0.s, p0/m, z0.s, z18.s
    fmin    z0.s, p0/m, z0.s, z19.s
    frintn  z0.s, p0/m, z0.s      // round to nearest
    fcvtzs  z0.s, p0/m, z0.s      // convert to int32
    st1b    {z0.s}, p0, [x11]     // narrow int32 -> int8 and store
    add     x8, x8, x9, lsl #2    // src += SVLs * 4
    add     x11, x11, x9           // dst += SVLs bytes
    add     x10, x10, x9
    cmp     w10, w3
    b.lt    .L_cq_quant
    // ── Next row ──
    add     w14, w14, #1
    b       .L_cq_row
.L_cq_done:
    b       .L_dispatch
// ================================================================
// TRANSPOSE_I8 (0x1D)
// Transpose an M×N int8 matrix to N×M int8.
// Uses ZA tile: load horizontal byte slices, extract vertical byte slices.
// Processes in SVLb×SVLb blocks (64×64 on M4).
//
// Bytecode: [0x1D][M:u32][N:u32]
// Operands: src_i8 (M×N), dst_i8 (N×M)
// ================================================================
.L_op_transpose_i8:
    ldr     w22, [x19]             // M (rows of src)
    ldr     w3, [x19, #4]          // N (cols of src)
    add     x19, x19, #8
    ldr     x8, [x19], #8         // src (M×N i8 row-major)
    ldr     x11, [x19], #8        // dst (N×M i8 row-major)
    cntb    x9                     // SVLb = 64
    // ── Tile block loops: process SVLb×SVLb blocks ──
    // Outer loop: row blocks of SVLb
    mov     w14, #0                // block_row
.L_ti8_block_row:
    cmp     w14, w22
    b.ge    .L_ti8_done
    // Inner loop: col blocks of SVLb
    mov     w15, #0                // block_col
.L_ti8_block_col:
    cmp     w15, w3
    b.ge    .L_ti8_next_block_row
    // ── Load this block into ZA0.b via horizontal slices ──
    zero    {za0.b}
    // Compute row/col predicates for edge blocks
    sub     w10, w22, w14          // remaining rows
    cmp     w10, w9
    csel    w10, w10, w9, lt       // min(remaining_rows, SVLb)
    sub     w16, w3, w15           // remaining cols
    cmp     w16, w9
    csel    w16, w16, w9, lt       // min(remaining_cols, SVLb)
    // Column predicate for loads
    mov     x17, xzr
    whilelt p1.b, x17, x16        // p1 = mask for valid columns
    // ── Load rows into ZA horizontal slices (groups of 4) ──
    mov     w12, #0                // ZA slice index
    mov     w17, #0                // row counter within block
.L_ti8_load_rows:
    cmp     w17, w10
    b.ge    .L_ti8_extract
    // Load up to 4 rows at a time via mova za0h.b[w12, 0:3]
    // Determine how many rows remain (1-4)
    sub     w4, w10, w17
    cmp     w4, #4
    b.lt    .L_ti8_load_partial
    // Full group of 4 rows
    mul     w6, w14, w3            // block_row * N (start of first block row in src)
    add     w6, w6, w15            // + block_col
    // Row 0
    add     w7, w17, #0
    madd    x13, x7, x3, x6       // (row_within_block) * N + base_offset
    add     x13, x8, x13           // absolute src address
    ld1b    {z0.b}, p1/z, [x13]
    // Row 1
    add     x13, x13, x3
    ld1b    {z1.b}, p1/z, [x13]
    // Row 2
    add     x13, x13, x3
    ld1b    {z2.b}, p1/z, [x13]
    // Row 3
    add     x13, x13, x3
    ld1b    {z3.b}, p1/z, [x13]
    mova    za0h.b[w12, 0:3], {z0.b-z3.b}
    add     w12, w12, #4
    add     w17, w17, #4
    b       .L_ti8_load_rows
.L_ti8_load_partial:
    // Load 1-3 remaining rows, zero-fill the rest
    mul     w6, w14, w3
    add     w6, w6, w15
    // Always load at least row 0
    madd    x13, x17, x3, x6
    add     x13, x8, x13
    ld1b    {z0.b}, p1/z, [x13]
    // Row 1?
    add     w7, w17, #1
    cmp     w7, w10
    b.ge    .L_ti8_pad_1
    add     x13, x13, x3
    ld1b    {z1.b}, p1/z, [x13]
    // Row 2?
    add     w7, w17, #2
    cmp     w7, w10
    b.ge    .L_ti8_pad_2
    add     x13, x13, x3
    ld1b    {z2.b}, p1/z, [x13]
    mov     z3.d, #0
    b       .L_ti8_store_partial
.L_ti8_pad_2:
    mov     z2.d, #0
    mov     z3.d, #0
    b       .L_ti8_store_partial
.L_ti8_pad_1:
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0
.L_ti8_store_partial:
    mova    za0h.b[w12, 0:3], {z0.b-z3.b}
    add     w12, w12, #4
    add     w17, w17, #4
    b       .L_ti8_load_rows
.L_ti8_extract:
    // ── Extract vertical slices from ZA and store to dst ──
    // dst[col * M + row] for col in [block_col..block_col+valid_cols)
    // Row predicate for stores: valid rows in this block
    mov     x17, xzr
    whilelt p2.b, x17, x10        // p2 = mask for valid rows in output
    // Extract vertical slices in groups of 4
    mov     w12, #0                // ZA vertical slice index
    mov     w17, #0                // col counter within block
.L_ti8_store_cols:
    cmp     w17, w16
    b.ge    .L_ti8_next_block_col
    // Extract up to 4 vertical slices
    sub     w4, w16, w17
    cmp     w4, #4
    b.lt    .L_ti8_extract_partial
    // Full group of 4 columns
    mova    {z0.b-z3.b}, za0v.b[w12, 0:3]
    // Store col 0: dst[(block_col + col_within_block + 0) * M + block_row]
    add     w6, w15, w17           // absolute col index
    mul     w7, w6, w22            // col * M
    add     w7, w7, w14            // + block_row
    add     x13, x11, x7          // dst address
    st1b    {z0.b}, p2, [x13]
    // Store col 1
    add     x13, x13, x22         // next col in dst = +M bytes
    st1b    {z1.b}, p2, [x13]
    // Store col 2
    add     x13, x13, x22
    st1b    {z2.b}, p2, [x13]
    // Store col 3
    add     x13, x13, x22
    st1b    {z3.b}, p2, [x13]
    add     w12, w12, #4
    add     w17, w17, #4
    b       .L_ti8_store_cols
.L_ti8_extract_partial:
    // Extract 1-3 remaining columns
    mova    {z0.b-z3.b}, za0v.b[w12, 0:3]
    add     w6, w15, w17
    mul     w7, w6, w22
    add     w7, w7, w14
    add     x13, x11, x7
    st1b    {z0.b}, p2, [x13]
    add     w7, w17, #1
    cmp     w7, w16
    b.ge    .L_ti8_extract_done_partial
    add     x13, x13, x22
    st1b    {z1.b}, p2, [x13]
    add     w7, w17, #2
    cmp     w7, w16
    b.ge    .L_ti8_extract_done_partial
    add     x13, x13, x22
    st1b    {z2.b}, p2, [x13]
.L_ti8_extract_done_partial:
    add     w12, w12, #4
    add     w17, w17, #4
    b       .L_ti8_store_cols
.L_ti8_next_block_col:
    add     w15, w15, w9           // block_col += SVLb
    b       .L_ti8_block_col
.L_ti8_next_block_row:
    add     w14, w14, w9           // block_row += SVLb
    b       .L_ti8_block_row
.L_ti8_done:
    b       .L_dispatch
// ================================================================
// DENSE_U8S8 (0x1E)
// UINT8×INT8 matmul via USMOPA: C_fp32 = dequant(A_u8 @ B_packed_i8) + bias [+relu]
// Identical to DENSE_I8 except uses usmopa (unsigned A × signed B) instead of
// smopa (signed × signed). This enables feeding raw uint8 data (e.g. MNIST pixel
// bytes 0-255) directly into the matmul without float normalization+requantize.
//
// A is M×K_pad row-major unsigned uint8 (K_pad = ceil(K/64)*64, zero-padded).
// B_packed is in SMOPA panel format (same as dense_i8, signed int8).
//
// Bytecode: [0x1E][M:u32][N:u32][K:u32][scale_a:f32][scale_b:f32][flags:u8]
//   flags bit 0: apply ReLU after dequant
// Operands: A_u8, B_packed_i8, bias_fp32 (1×N), C_fp32
// ================================================================
.L_op_dense_u8s8:
    // ── Parse immediates + operands ──
    ldr     w0, [x19]              // M
    ldr     w1, [x19, #4]          // N
    ldr     w2, [x19, #8]          // K
    ldr     s20, [x19, #12]        // scale_a (f32)
    ldr     s22, [x19, #16]        // scale_b (f32)
    ldrb    w18, [x19, #20]        // flags (bit 0 = relu)
    add     x19, x19, #21
    ldr     x5, [x19], #8          // A_u8 (M×K_pad row-major)
    ldr     x6, [x19], #8          // B_packed_i8 (SMOPA panel format)
    ldr     x8, [x19], #8          // bias_fp32 (1×N)
    ldr     x7, [x19], #8          // C_fp32 (M×N output)
    // ── Derived values ──
    add     w3, w0, #15
    and     w3, w3, #0xFFFFFFF0    // M_pad = (M+15) & ~15
    add     w4, w1, #31
    and     w4, w4, #0xFFFFFFE0    // N_pad = (N+31) & ~31
    add     w15, w2, #63
    and     w15, w15, #0xFFFFFFC0  // K_pad = (K+63) & ~63
    lsr     w16, w15, #6           // outer_k_blocks = K_pad / 64
    lsr     w14, w15, #2           // k_groups_total = K_pad / 4
    fmul    s20, s20, s22          // combined dequant scale = scale_a * scale_b
    ptrue   p0.s
    cntw    x9
    mov     z20.s, s20             // broadcast combined scale
    // ── Branchless ReLU ──
    and     w18, w18, #1
    movz    w10, #0xFF7F, lsl #16
    movk    w10, #0xFFFF           // -FLT_MAX
    cmp     w18, #0
    csel    w10, w10, wzr, eq
    fmov    s21, w10
    mov     z21.s, s21
    // ── Stack frame: 96 bytes ──
    sub     sp, sp, #96
    stp     x5, x6, [sp, #0]      // [0] A, [8] B_packed
    str     x7, [sp, #16]         // [16] C
    stp     w0, w1, [sp, #24]     // [24] M, [28] N
    str     w15, [sp, #32]        // [32] K_pad
    stp     w4, w3, [sp, #36]     // [36] N_pad, [40] M_pad
    stp     w16, w14, [sp, #44]   // [44] outer_k_blocks, [48] k_groups_total
    str     x8, [sp, #56]         // [56] bias
    mov     x17, x15
    str     x17, [sp, #64]        // [64] A_row_stride = K_pad
    // [72] ti, [76] tj (loop vars)
    // ── Tile row loop ──
    mov     w0, #0
.L_u8s8_tile_row:
    str     w0, [sp, #72]
    ldr     x5, [sp, #0]
    ldr     x17, [sp, #64]
    mul     w10, w0, w17
    add     x5, x5, x10           // A_tile_base = A + ti*K_pad
    mov     w1, #0
.L_u8s8_tile_col:
    str     w1, [sp, #76]
    zero    {za0.s, za1.s}
    ldr     w16, [sp, #44]         // outer_k_blocks
    ldr     x17, [sp, #64]
    mov     x13, xzr
    mov     w24, #0
    cbz     w16, .L_u8s8_k_done
.L_u8s8_k_outer:
    // ── Load 16 A rows into za2 (.s horizontal slices) ──
    zero    {za2.s}
    ptrue   p0.s
    add     x8, x5, x13
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
    // ── B_packed panel addressing ──
    ldr     x6, [sp, #8]
    ldr     w14, [sp, #48]
    ldr     w1, [sp, #76]
    lsr     w23, w1, #4
    add     w26, w23, #1
    mul     w10, w23, w14
    lsl     x10, x10, #6
    add     x22, x6, x10          // left_panel_base
    mul     w10, w26, w14
    lsl     x10, x10, #6
    add     x23, x6, x10          // right_panel_base
    ptrue   p0.b
    // ── Column group 0 (cols 0-3) ──
    mov     w12, #0
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    add     w10, w24, #0
    lsl     x10, x10, #6
    add     x8, x22, x10
    add     x11, x23, x10
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z0.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z0.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z1.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z1.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z2.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z2.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z3.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z3.b, z5.b
    // ── Column group 1 (cols 4-7) ──
    mov     w12, #4
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    add     w10, w24, #4
    lsl     x10, x10, #6
    add     x8, x22, x10
    add     x11, x23, x10
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z0.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z0.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z1.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z1.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z2.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z2.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z3.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z3.b, z5.b
    // ── Column group 2 (cols 8-11) ──
    mov     w12, #8
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    add     w10, w24, #8
    lsl     x10, x10, #6
    add     x8, x22, x10
    add     x11, x23, x10
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z0.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z0.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z1.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z1.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z2.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z2.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z3.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z3.b, z5.b
    // ── Column group 3 (cols 12-15) ──
    mov     w12, #12
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    add     w10, w24, #12
    lsl     x10, x10, #6
    add     x8, x22, x10
    add     x11, x23, x10
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z0.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z0.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z1.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z1.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z2.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z2.b, z5.b
    add     x8, x8, #64
    add     x11, x11, #64
    ld1b    {z4.b}, p0/z, [x8]
    ld1b    {z5.b}, p0/z, [x11]
    usmopa  za0.s, p0/m, p0/m, z3.b, z4.b
    usmopa  za1.s, p0/m, p0/m, z3.b, z5.b
    // ── Advance K ──
    add     x13, x13, #64
    add     w24, w24, #16
    subs    w16, w16, #1
    b.ne    .L_u8s8_k_outer
.L_u8s8_k_done:
    // ── Store: extract za0+za1 → scvtf → fmul scale → fadd bias → relu → C ──
    ptrue   p0.s
    cntw    x9
    ldr     x8, [sp, #56]         // bias
    ldr     w1, [sp, #76]         // tj
    ld1w    {z16.s}, p0/z, [x8, x1, lsl #2]
    add     x10, x1, x9
    ld1w    {z17.s}, p0/z, [x8, x10, lsl #2]
    ldr     x7, [sp, #16]
    ldr     w0, [sp, #72]
    ldr     w14, [sp, #28]
    lsl     x10, x14, #2
    mul     w8, w0, w14
    add     w8, w8, w1
    add     x7, x7, x8, lsl #2
    sub     w6, w14, w1
    mov     w8, #32
    cmp     w6, w8
    csel    w6, w6, w8, lt
    whilelt p2.s, xzr, x6
    sub     w8, w6, #16
    cmp     w8, #0
    csel    w8, wzr, w8, lt
    whilelt p3.s, xzr, x8
    ptrue   p0.s
    cntw    x9
    mov     z14.d, z21.d
    // ── Store macro: scvtf → fmul → fadd bias → relu (repeated for 4 groups) ──
    // Group 0 (rows 0-3)
    mov     w12, #0
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z20.s
    fadd    z0.s, z0.s, z16.s
    scvtf   z1.s, p0/m, z1.s
    fmul    z1.s, z1.s, z20.s
    fadd    z1.s, z1.s, z16.s
    scvtf   z2.s, p0/m, z2.s
    fmul    z2.s, z2.s, z20.s
    fadd    z2.s, z2.s, z16.s
    scvtf   z3.s, p0/m, z3.s
    fmul    z3.s, z3.s, z20.s
    fadd    z3.s, z3.s, z16.s
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z20.s
    fadd    z4.s, z4.s, z17.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z20.s
    fadd    z5.s, z5.s, z17.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z20.s
    fadd    z6.s, z6.s, z17.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z20.s
    fadd    z7.s, z7.s, z17.s
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
    // Group 1 (rows 4-7)
    mov     w12, #4
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z20.s
    fadd    z0.s, z0.s, z16.s
    scvtf   z1.s, p0/m, z1.s
    fmul    z1.s, z1.s, z20.s
    fadd    z1.s, z1.s, z16.s
    scvtf   z2.s, p0/m, z2.s
    fmul    z2.s, z2.s, z20.s
    fadd    z2.s, z2.s, z16.s
    scvtf   z3.s, p0/m, z3.s
    fmul    z3.s, z3.s, z20.s
    fadd    z3.s, z3.s, z16.s
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z20.s
    fadd    z4.s, z4.s, z17.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z20.s
    fadd    z5.s, z5.s, z17.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z20.s
    fadd    z6.s, z6.s, z17.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z20.s
    fadd    z7.s, z7.s, z17.s
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
    // Group 2 (rows 8-11)
    mov     w12, #8
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z20.s
    fadd    z0.s, z0.s, z16.s
    scvtf   z1.s, p0/m, z1.s
    fmul    z1.s, z1.s, z20.s
    fadd    z1.s, z1.s, z16.s
    scvtf   z2.s, p0/m, z2.s
    fmul    z2.s, z2.s, z20.s
    fadd    z2.s, z2.s, z16.s
    scvtf   z3.s, p0/m, z3.s
    fmul    z3.s, z3.s, z20.s
    fadd    z3.s, z3.s, z16.s
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z20.s
    fadd    z4.s, z4.s, z17.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z20.s
    fadd    z5.s, z5.s, z17.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z20.s
    fadd    z6.s, z6.s, z17.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z20.s
    fadd    z7.s, z7.s, z17.s
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
    // Group 3 (rows 12-15)
    mov     w12, #12
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z20.s
    fadd    z0.s, z0.s, z16.s
    scvtf   z1.s, p0/m, z1.s
    fmul    z1.s, z1.s, z20.s
    fadd    z1.s, z1.s, z16.s
    scvtf   z2.s, p0/m, z2.s
    fmul    z2.s, z2.s, z20.s
    fadd    z2.s, z2.s, z16.s
    scvtf   z3.s, p0/m, z3.s
    fmul    z3.s, z3.s, z20.s
    fadd    z3.s, z3.s, z16.s
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z20.s
    fadd    z4.s, z4.s, z17.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z20.s
    fadd    z5.s, z5.s, z17.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z20.s
    fadd    z6.s, z6.s, z17.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z20.s
    fadd    z7.s, z7.s, z17.s
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
    ldr     w1, [sp, #76]
    ldr     w4, [sp, #36]          // N_pad
    add     w1, w1, #32
    cmp     w1, w4
    b.lt    .L_u8s8_tile_col
    // ── Advance tile row ──
    ldr     w0, [sp, #72]
    ldr     w3, [sp, #40]          // M_pad
    add     w0, w0, #16
    cmp     w0, w3
    b.lt    .L_u8s8_tile_row
.L_u8s8_tile_done:
    add     sp, sp, #96
    b       .L_dispatch
