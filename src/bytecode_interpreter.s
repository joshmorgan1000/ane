.text
.p2align 4
.global _stream_exec
_stream_exec:
    stp     x29, x30, [sp, #-192]!
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
    .long   .L_op_load - .L_jump_table      // 0x1F load (SVE streaming copy src→dst)
    .long   .L_op_store - .L_jump_table     // 0x20 store (SVE streaming copy src→dst)
    .long   .L_op_l2_squared_fp32 - .L_jump_table   // 0x21 l2_squared_fp32
    .long   .L_op_l2_squared_bf16 - .L_jump_table   // 0x22 l2_squared_bf16
    .long   .L_op_l2_squared_f64 - .L_jump_table    // 0x23 l2_squared_f64
    .long   .L_op_cosine_dist_fp32 - .L_jump_table  // 0x24 cosine_dist_fp32
    .long   .L_op_cosine_dist_bf16 - .L_jump_table  // 0x25 cosine_dist_bf16
    .long   .L_op_cosine_dist_f64 - .L_jump_table   // 0x26 cosine_dist_f64
    .long   .L_op_normalize_fp32 - .L_jump_table    // 0x27 normalize_fp32
    .long   .L_op_dct2_forward_fp32 - .L_jump_table // 0x28 dct2_forward_fp32
    .long   .L_op_dct2_inverse_fp32 - .L_jump_table // 0x29 dct2_inverse_fp32
    .long   .L_op_threshold_bitmap_fp32 - .L_jump_table // 0x2A threshold_bitmap_fp32
    .long   .L_op_welford_stats_fp32 - .L_jump_table   // 0x2B welford_stats_fp32
    .long   .L_op_quantize_pack_4bit_fp32 - .L_jump_table // 0x2C quantize_pack_4bit_fp32
    .long   .L_op_threshold_8bit - .L_jump_table        // 0x2D threshold_8bit
    .long   .L_op_quantize_accum_2bit - .L_jump_table  // 0x2E quantize_accum_2bit
    .long   .L_op_accum_8bit - .L_jump_table           // 0x2F accum_8bit
    .long   .L_op_soa_sub_scale_bf16 - .L_jump_table  // 0x30 soa_sub_scale_bf16
    .long   .L_op_soa_luti2_accum - .L_jump_table     // 0x31 soa_luti2_accum
    .long   .L_op_soa_luti4_accum - .L_jump_table     // 0x32 soa_luti4_accum
    .long   .L_op_bitmap_score_pipeline - .L_jump_table // 0x33 bitmap_score_pipeline
    .long   .L_op_mov_zreg - .L_jump_table         // 0x34 mov_zreg
    .long   .L_op_loop_begin - .L_jump_table      // 0x35 loop_begin
    .long   .L_op_loop_end - .L_jump_table        // 0x36 loop_end
    .long   .L_op_set_param - .L_jump_table       // 0x37 set_param
    .long   .L_op_load_param - .L_jump_table      // 0x38 load_param
    .long   .L_op_store_param - .L_jump_table     // 0x39 store_param
    .long   .L_op_advance_param - .L_jump_table   // 0x3A advance_param
    .long   .L_op_fadd_zreg - .L_jump_table       // 0x3B fadd_zreg
    .long   .L_op_fsub_zreg - .L_jump_table       // 0x3C fsub_zreg
    .long   .L_op_fmul_zreg - .L_jump_table       // 0x3D fmul_zreg
    .long   .L_op_fmla_zreg - .L_jump_table       // 0x3E fmla_zreg
    .long   .L_op_and_zreg - .L_jump_table        // 0x3F and_zreg
    .long   .L_op_orr_zreg - .L_jump_table        // 0x40 orr_zreg
    .long   .L_op_eor_zreg - .L_jump_table        // 0x41 eor_zreg
    .long   .L_op_not_zreg - .L_jump_table        // 0x42 not_zreg
    .long   .L_op_lsl_zreg - .L_jump_table       // 0x43 lsl_zreg
    .long   .L_op_lsr_zreg - .L_jump_table       // 0x44 lsr_zreg
    .long   .L_op_asr_zreg - .L_jump_table       // 0x45 asr_zreg
    .long   .L_op_load_zt0 - .L_jump_table       // 0x46 load_zt0
    .long   .L_op_luti2_zreg - .L_jump_table     // 0x47 luti2_zreg
    .long   .L_op_luti4_zreg - .L_jump_table     // 0x48 luti4_zreg
    .long   .L_op_smopa_zreg - .L_jump_table     // 0x49 smopa_zreg
    .long   .L_op_umopa_zreg - .L_jump_table     // 0x4A umopa_zreg
    .long   .L_op_usmopa_zreg - .L_jump_table    // 0x4B usmopa_zreg
    .long   .L_op_fmopa_zreg - .L_jump_table     // 0x4C fmopa_zreg
    .long   .L_op_cblas_sgemm - .L_jump_table    // 0x4D cblas_sgemm
    .long   .L_op_fclamp_zreg - .L_jump_table   // 0x4E fclamp_zreg
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
    ldp     x29, x30, [sp], #192
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
// ================================================================
// LOAD (0x1F) — Load one z-vector from memory into z0
// Encoding: [0x1F][src_ptr:u64]
// Loads VL bytes from src_ptr into z0. Use mov_zreg to move z0
// to another register afterward if needed.
// ================================================================
.L_op_load:
    ldr     x8, [x19], #8         // src_ptr
    ptrue   p0.b
    ld1b    {z0.b}, p0/z, [x8]
    b       .L_dispatch
// ================================================================
// STORE (0x20) — Store one z-vector from z0 to memory
// Encoding: [0x20][dst_ptr:u64]
// Stores VL bytes from z0 to dst_ptr. Use mov_zreg to move data
// into z0 before calling if needed.
// ================================================================
.L_op_store:
    ldr     x8, [x19], #8         // dst_ptr
    ptrue   p0.b
    st1b    {z0.b}, p0, [x8]
    b       .L_dispatch
// ================================================================
// L2_SQUARED_FP32 (0x21) — fused L2 squared distance for fp32
// result = sum((a[i] - b[i])^2)
// Encoding: [0x21][dim:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
// dim = number of fp32 elements (multiple of cntw)
// ================================================================
.L_op_l2_squared_fp32:
    ldr     w22, [x19]             // dim (u32)
    add     x19, x19, #4
    ldr     x8, [x19], #8         // a_ptr
    ldr     x11, [x19], #8        // b_ptr
    ldr     x12, [x19], #8        // out_ptr
    ptrue   p0.s
    fmov    z4.s, #0.0             // accumulator
    mov     x10, #0
    cbz     w22, .L_l2f32_store
    whilelt p1.s, x10, x22
.L_l2f32_loop:
    ld1w    {z0.s}, p1/z, [x8, x10, lsl #2]
    ld1w    {z1.s}, p1/z, [x11, x10, lsl #2]
    fsub    z0.s, z0.s, z1.s
    fmla    z4.s, p1/m, z0.s, z0.s
    incw    x10
    whilelt p1.s, x10, x22
    b.first .L_l2f32_loop
.L_l2f32_store:
    faddv   s4, p0, z4.s
    str     s4, [x12]
    b       .L_dispatch
// ================================================================
// L2_SQUARED_BF16 (0x22) — fused L2 squared distance for bf16
// Loads bf16, widens to fp32, accumulates in fp32.
// result = sum((a[i] - b[i])^2) as fp32 scalar
// Encoding: [0x22][dim:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
// dim = number of bf16 elements (multiple of cnth)
// ================================================================
.L_op_l2_squared_bf16:
    ldr     w22, [x19]             // dim (u32, bf16 element count)
    add     x19, x19, #4
    ldr     x8, [x19], #8         // a_ptr (bfloat16*)
    ldr     x11, [x19], #8        // b_ptr (bfloat16*)
    ldr     x12, [x19], #8        // out_ptr (float*)
    ptrue   p0.s
    ptrue   p2.h
    fmov    z4.s, #0.0             // fp32 accumulator (low half)
    fmov    z5.s, #0.0             // fp32 accumulator (high half)
    mov     x10, #0
    cbz     w22, .L_l2bf16_store
    whilelt p1.h, x10, x22
.L_l2bf16_loop:
    ld1h    {z0.h}, p1/z, [x8, x10, lsl #1]   // load bf16 a
    ld1h    {z1.h}, p1/z, [x11, x10, lsl #1]  // load bf16 b
    // Widen bf16 to fp32: bf16 is upper 16 bits of fp32, so lsl #16
    uunpklo z2.s, z0.h             // zero-extend low half → .s
    uunpkhi z3.s, z0.h             // zero-extend high half → .s
    lsl     z2.s, z2.s, #16        // bf16 → fp32 (a_lo)
    lsl     z3.s, z3.s, #16        // bf16 → fp32 (a_hi)
    uunpklo z6.s, z1.h             // zero-extend low half → .s (b)
    uunpkhi z7.s, z1.h             // zero-extend high half → .s (b)
    lsl     z6.s, z6.s, #16        // bf16 → fp32 (b_lo)
    lsl     z7.s, z7.s, #16        // bf16 → fp32 (b_hi)
    fsub    z2.s, z2.s, z6.s       // diff_lo = a_lo - b_lo
    fsub    z3.s, z3.s, z7.s       // diff_hi = a_hi - b_hi
    fmla    z4.s, p0/m, z2.s, z2.s // accum_lo += diff_lo^2
    fmla    z5.s, p0/m, z3.s, z3.s // accum_hi += diff_hi^2
    inch    x10
    whilelt p1.h, x10, x22
    b.first .L_l2bf16_loop
.L_l2bf16_store:
    fadd    z4.s, p0/m, z4.s, z5.s // merge both accumulators
    faddv   s4, p0, z4.s
    str     s4, [x12]
    b       .L_dispatch
// ================================================================
// L2_SQUARED_F64 (0x23) — fused L2 squared distance for f64
// result = sum((a[i] - b[i])^2)
// Encoding: [0x23][dim:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
// dim = number of f64 elements (multiple of cntd)
// ================================================================
.L_op_l2_squared_f64:
    ldr     w22, [x19]             // dim (u32)
    add     x19, x19, #4
    ldr     x8, [x19], #8         // a_ptr
    ldr     x11, [x19], #8        // b_ptr
    ldr     x12, [x19], #8        // out_ptr
    ptrue   p0.d
    fmov    z4.d, #0.0             // accumulator
    mov     x10, #0
    cbz     w22, .L_l2f64_store
    whilelt p1.d, x10, x22
.L_l2f64_loop:
    ld1d    {z0.d}, p1/z, [x8, x10, lsl #3]
    ld1d    {z1.d}, p1/z, [x11, x10, lsl #3]
    fsub    z0.d, z0.d, z1.d
    fmla    z4.d, p1/m, z0.d, z0.d
    incd    x10
    whilelt p1.d, x10, x22
    b.first .L_l2f64_loop
.L_l2f64_store:
    faddv   d4, p0, z4.d
    str     d4, [x12]
    b       .L_dispatch
// ================================================================
// COSINE_DIST_FP32 (0x24) — fused cosine distance for fp32
// result = 1 - dot(a,b) / (||a|| * ||b||)
// Three accumulators: z4=dot(a,b), z5=||a||^2, z6=||b||^2
// Encoding: [0x24][dim:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
// ================================================================
.L_op_cosine_dist_fp32:
    ldr     w22, [x19]             // dim (u32)
    add     x19, x19, #4
    ldr     x8, [x19], #8         // a_ptr
    ldr     x11, [x19], #8        // b_ptr
    ldr     x12, [x19], #8        // out_ptr
    ptrue   p0.s
    fmov    z4.s, #0.0             // dot(a,b) accumulator
    fmov    z5.s, #0.0             // ||a||^2 accumulator
    fmov    z6.s, #0.0             // ||b||^2 accumulator
    mov     x10, #0
    cbz     w22, .L_csf32_reduce
    whilelt p1.s, x10, x22
.L_csf32_loop:
    ld1w    {z0.s}, p1/z, [x8, x10, lsl #2]
    ld1w    {z1.s}, p1/z, [x11, x10, lsl #2]
    fmla    z4.s, p1/m, z0.s, z1.s // dot += a*b
    fmla    z5.s, p1/m, z0.s, z0.s // norm_a += a*a
    fmla    z6.s, p1/m, z1.s, z1.s // norm_b += b*b
    incw    x10
    whilelt p1.s, x10, x22
    b.first .L_csf32_loop
.L_csf32_reduce:
    faddv   s4, p0, z4.s           // dot scalar
    faddv   s5, p0, z5.s           // ||a||^2 scalar
    faddv   s6, p0, z6.s           // ||b||^2 scalar
    fsqrt   s5, s5                 // ||a||
    fsqrt   s6, s6                 // ||b||
    fmul    s5, s5, s6             // ||a|| * ||b||
    fdiv    s4, s4, s5             // dot / (||a|| * ||b||)
    fmov    s7, #1.0
    fsub    s4, s7, s4             // 1 - similarity
    str     s4, [x12]
    b       .L_dispatch
// ================================================================
// COSINE_DIST_BF16 (0x25) — fused cosine distance for bf16
// Loads bf16, widens to fp32, accumulates in fp32.
// result = 1 - dot(a,b) / (||a|| * ||b||) as fp32 scalar
// Encoding: [0x25][dim:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
// ================================================================
.L_op_cosine_dist_bf16:
    ldr     w22, [x19]             // dim (u32, bf16 element count)
    add     x19, x19, #4
    ldr     x8, [x19], #8         // a_ptr (bfloat16*)
    ldr     x11, [x19], #8        // b_ptr (bfloat16*)
    ldr     x12, [x19], #8        // out_ptr (float*)
    ptrue   p0.s
    // 6 accumulators: dot_lo/hi, norm_a_lo/hi, norm_b_lo/hi
    fmov    z4.s, #0.0             // dot_lo
    fmov    z5.s, #0.0             // dot_hi
    fmov    z6.s, #0.0             // norm_a_lo
    fmov    z7.s, #0.0             // norm_a_hi
    fmov    z8.s, #0.0             // norm_b_lo
    fmov    z9.s, #0.0             // norm_b_hi
    mov     x10, #0
    cbz     w22, .L_csbf16_reduce
    whilelt p1.h, x10, x22
.L_csbf16_loop:
    ld1h    {z0.h}, p1/z, [x8, x10, lsl #1]   // load bf16 a
    ld1h    {z1.h}, p1/z, [x11, x10, lsl #1]  // load bf16 b
    // Widen a: bf16 → fp32
    uunpklo z2.s, z0.h
    uunpkhi z3.s, z0.h
    lsl     z2.s, z2.s, #16        // a_lo fp32
    lsl     z3.s, z3.s, #16        // a_hi fp32
    // Widen b: bf16 → fp32
    uunpklo z10.s, z1.h
    uunpkhi z11.s, z1.h
    lsl     z10.s, z10.s, #16      // b_lo fp32
    lsl     z11.s, z11.s, #16      // b_hi fp32
    // Accumulate dot(a,b)
    fmla    z4.s, p0/m, z2.s, z10.s  // dot_lo += a_lo * b_lo
    fmla    z5.s, p0/m, z3.s, z11.s  // dot_hi += a_hi * b_hi
    // Accumulate ||a||^2
    fmla    z6.s, p0/m, z2.s, z2.s   // norm_a_lo += a_lo^2
    fmla    z7.s, p0/m, z3.s, z3.s   // norm_a_hi += a_hi^2
    // Accumulate ||b||^2
    fmla    z8.s, p0/m, z10.s, z10.s // norm_b_lo += b_lo^2
    fmla    z9.s, p0/m, z11.s, z11.s // norm_b_hi += b_hi^2
    inch    x10
    whilelt p1.h, x10, x22
    b.first .L_csbf16_loop
.L_csbf16_reduce:
    // Merge lo+hi accumulators
    fadd    z4.s, p0/m, z4.s, z5.s   // dot total
    fadd    z6.s, p0/m, z6.s, z7.s   // norm_a total
    fadd    z8.s, p0/m, z8.s, z9.s   // norm_b total
    faddv   s4, p0, z4.s              // dot scalar
    faddv   s6, p0, z6.s              // ||a||^2 scalar
    faddv   s8, p0, z8.s              // ||b||^2 scalar
    fsqrt   s6, s6                    // ||a||
    fsqrt   s8, s8                    // ||b||
    fmul    s6, s6, s8                // ||a|| * ||b||
    fdiv    s4, s4, s6                // dot / (||a|| * ||b||)
    fmov    s7, #1.0
    fsub    s4, s7, s4                // 1 - similarity
    str     s4, [x12]
    b       .L_dispatch
// ================================================================
// COSINE_DIST_F64 (0x26) — fused cosine distance for f64
// result = 1 - dot(a,b) / (||a|| * ||b||)
// Encoding: [0x26][dim:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
// ================================================================
.L_op_cosine_dist_f64:
    ldr     w22, [x19]             // dim (u32)
    add     x19, x19, #4
    ldr     x8, [x19], #8         // a_ptr
    ldr     x11, [x19], #8        // b_ptr
    ldr     x12, [x19], #8        // out_ptr
    ptrue   p0.d
    fmov    z4.d, #0.0             // dot(a,b) accumulator
    fmov    z5.d, #0.0             // ||a||^2 accumulator
    fmov    z6.d, #0.0             // ||b||^2 accumulator
    mov     x10, #0
    cbz     w22, .L_csf64_reduce
    whilelt p1.d, x10, x22
.L_csf64_loop:
    ld1d    {z0.d}, p1/z, [x8, x10, lsl #3]
    ld1d    {z1.d}, p1/z, [x11, x10, lsl #3]
    fmla    z4.d, p1/m, z0.d, z1.d // dot += a*b
    fmla    z5.d, p1/m, z0.d, z0.d // norm_a += a*a
    fmla    z6.d, p1/m, z1.d, z1.d // norm_b += b*b
    incd    x10
    whilelt p1.d, x10, x22
    b.first .L_csf64_loop
.L_csf64_reduce:
    faddv   d4, p0, z4.d           // dot scalar
    faddv   d5, p0, z5.d           // ||a||^2 scalar
    faddv   d6, p0, z6.d           // ||b||^2 scalar
    fsqrt   d5, d5                 // ||a||
    fsqrt   d6, d6                 // ||b||
    fmul    d5, d5, d6             // ||a|| * ||b||
    fdiv    d4, d4, d5             // dot / (||a|| * ||b||)
    fmov    d7, #1.0
    fsub    d4, d7, d4             // 1 - similarity
    str     d4, [x12]
    b       .L_dispatch
// ================================================================
// NORMALIZE_FP32 (0x27) — in-place unit normalize fp32 vector
// vec[i] /= ||vec||  (two-pass: compute norm, then multiply by 1/norm)
// Encoding: [0x27][dim:u32][vec_ptr:u64]
// ================================================================
.L_op_normalize_fp32:
    ldr     w22, [x19]             // dim (u32)
    add     x19, x19, #4
    ldr     x8, [x19], #8         // vec_ptr (float*, in-place)
    ptrue   p0.s
    fmov    z4.s, #0.0             // ||vec||^2 accumulator
    mov     x10, #0
    cbz     w22, .L_dispatch
    // ── Pass 1: compute ||vec||^2 ──
    whilelt p1.s, x10, x22
.L_nrm_pass1:
    ld1w    {z0.s}, p1/z, [x8, x10, lsl #2]
    fmla    z4.s, p1/m, z0.s, z0.s
    incw    x10
    whilelt p1.s, x10, x22
    b.first .L_nrm_pass1
    // Horizontal sum → scalar norm^2, then reciprocal sqrt
    faddv   s4, p0, z4.s           // ||vec||^2
    fsqrt   s4, s4                 // ||vec||
    fmov    s5, #1.0
    fdiv    s5, s5, s4             // inv_norm = 1.0 / ||vec||
    mov     z16.s, s5              // broadcast inv_norm
    // ── Pass 2: vec[i] *= inv_norm ──
    mov     x10, #0
    cntw    x9
.L_nrm_pass2:
    ld1w    {z0.s}, p0/z, [x8]
    fmul    z0.s, z0.s, z16.s
    st1w    {z0.s}, p0, [x8]
    add     x8, x8, x9, lsl #2
    sub     w22, w22, w9
    cbnz    w22, .L_nrm_pass2
    b       .L_dispatch
// ================================================================
// DCT2_FORWARD_FP32 (0x28)
// H.264 4-point integer butterfly DCT-II forward transform on
// groups of 4 float32 values. Converts fp32→int32, applies
// butterfly, converts int32→fp32.
// Encoding: [0x28][dim:u32][src_ptr:u64][dst_ptr:u64]
// dim = number of float32 elements (multiple of 4 and SVLs)
// Butterfly per group [x0,x1,x2,x3]:
//   s0=x0+x3  s1=x1+x2  d0=x0-x3  d1=x1-x2
//   out0=s0+s1  out1=d0+(d1>>1)  out2=s0-s1  out3=(d0>>1)-d1
// Uses LD4W/ST4W to deinterleave/interleave group positions,
// processing SVLs groups (16 on M4, = 64 elements) per iteration.
// ================================================================
.L_op_dct2_forward_fp32:
    ldr     w22, [x19]             // dim (element count)
    add     x19, x19, #4
    ldr     x8, [x19], #8         // src_ptr
    ldr     x11, [x19], #8        // dst_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    lsr     x23, x22, #2          // num_groups = dim / 4
    mov     x10, #0               // group index
    cntw    x9                    // groups per full iteration (SVLs = 16)
    whilelt p1.s, x10, x23
.L_dctf_loop:
    // ── Deinterleave: z0=pos0, z1=pos1, z2=pos2, z3=pos3 of each group ──
    ld4w    {z0.s-z3.s}, p1/z, [x8]
    // ── Convert fp32 → int32 for integer butterfly ──
    fcvtzs  z0.s, p0/m, z0.s
    fcvtzs  z1.s, p0/m, z1.s
    fcvtzs  z2.s, p0/m, z2.s
    fcvtzs  z3.s, p0/m, z3.s
    // ── Sums and differences ──
    add     z4.s, z0.s, z3.s      // s0 = x0 + x3
    add     z5.s, z1.s, z2.s      // s1 = x1 + x2
    sub     z6.s, z0.s, z3.s      // d0 = x0 - x3
    sub     z7.s, z1.s, z2.s      // d1 = x1 - x2
    // ── Butterfly outputs ──
    add     z0.s, z4.s, z5.s      // out0 = s0 + s1
    asr     z8.s, z7.s, #1        // d1 >> 1
    add     z1.s, z6.s, z8.s      // out1 = d0 + (d1 >> 1)
    sub     z2.s, z4.s, z5.s      // out2 = s0 - s1
    asr     z8.s, z6.s, #1        // d0 >> 1
    sub     z3.s, z8.s, z7.s      // out3 = (d0 >> 1) - d1
    // ── Convert int32 → fp32 ──
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s
    // ── Interleave and store ──
    st4w    {z0.s-z3.s}, p1, [x11]
    // ── Advance pointers and loop ──
    add     x8, x8, x9, lsl #4    // src += SVLs * 4 elems * 4 bytes = SVLs << 4
    add     x11, x11, x9, lsl #4   // dst += SVLs * 4 elems * 4 bytes
    add     x10, x10, x9          // group_index += SVLs
    whilelt p1.s, x10, x23
    b.first .L_dctf_loop
    b       .L_dispatch
// ================================================================
// DCT2_INVERSE_FP32 (0x29)
// H.264 4-point integer butterfly DCT-II inverse transform on
// groups of 4 float32 values. Converts fp32→int32, applies
// inverse butterfly, converts int32→fp32.
// Encoding: [0x29][dim:u32][src_ptr:u64][dst_ptr:u64]
// dim = number of float32 elements (multiple of 4 and SVLs)
// Inverse butterfly per group [y0,y1,y2,y3]:
//   s0=y0+y2  s1=y0-y2  d0=(y1>>1)-y3  d1=y1+(y3>>1)
//   out0=s0+d1  out1=s1+d0  out2=s1-d0  out3=s0-d1
// ================================================================
.L_op_dct2_inverse_fp32:
    ldr     w22, [x19]             // dim (element count)
    add     x19, x19, #4
    ldr     x8, [x19], #8         // src_ptr
    ldr     x11, [x19], #8        // dst_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    lsr     x23, x22, #2          // num_groups = dim / 4
    mov     x10, #0               // group index
    cntw    x9                    // groups per full iteration (SVLs = 16)
    whilelt p1.s, x10, x23
.L_dcti_loop:
    // ── Deinterleave: z0=pos0, z1=pos1, z2=pos2, z3=pos3 of each group ──
    ld4w    {z0.s-z3.s}, p1/z, [x8]
    // ── Convert fp32 → int32 for integer butterfly ──
    fcvtzs  z0.s, p0/m, z0.s      // y0
    fcvtzs  z1.s, p0/m, z1.s      // y1
    fcvtzs  z2.s, p0/m, z2.s      // y2
    fcvtzs  z3.s, p0/m, z3.s      // y3
    // ── Sums and differences ──
    add     z4.s, z0.s, z2.s      // s0 = y0 + y2
    sub     z5.s, z0.s, z2.s      // s1 = y0 - y2
    asr     z6.s, z1.s, #1        // y1 >> 1
    sub     z6.s, z6.s, z3.s      // d0 = (y1 >> 1) - y3
    asr     z7.s, z3.s, #1        // y3 >> 1
    add     z7.s, z1.s, z7.s      // d1 = y1 + (y3 >> 1)
    // ── Inverse butterfly outputs ──
    add     z0.s, z4.s, z7.s      // out0 = s0 + d1
    add     z1.s, z5.s, z6.s      // out1 = s1 + d0
    sub     z2.s, z5.s, z6.s      // out2 = s1 - d0
    sub     z3.s, z4.s, z7.s      // out3 = s0 - d1
    // ── Convert int32 → fp32 ──
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s
    // ── Interleave and store ──
    st4w    {z0.s-z3.s}, p1, [x11]
    // ── Advance pointers and loop ──
    add     x8, x8, x9, lsl #4    // src += SVLs * 4 * 4
    add     x11, x11, x9, lsl #4   // dst += SVLs * 4 * 4
    add     x10, x10, x9          // group_index += SVLs
    whilelt p1.s, x10, x23
    b.first .L_dcti_loop
    b       .L_dispatch
// ================================================================
// THRESHOLD_BITMAP_FP32 (0x2A)
// Compare float32 array against threshold, produce packed bitmap.
// Each bit in the output corresponds to one input float32 element:
// bit i is set iff src[i] > threshold.
// Encoding: [0x2A][dim:u32][threshold:f32][src_ptr:u64][bitmap_out:u64]
// dim = number of float32 elements
// Output: packed bits, byte 0 = bits 0-7, byte 1 = bits 8-15, etc.
// Output size = ceil(dim / 8) bytes.
//
// Algorithm per SVLs (16) elements:
// 1. fcmgt → predicate p1
// 2. Materialize as 0/1 int32, narrow to bytes via uzp1 chain
// 3. Multiply by bit-position mask {1,2,4,8,16,32,64,128} x2
// 4. Pairwise-add tree (3 rounds) packs 16 bytes → 2 bytes
// 5. Extract via fmov and store
// ================================================================
.L_op_threshold_bitmap_fp32:
    ldr     w22, [x19]             // dim (element count)
    ldr     s16, [x19, #4]        // threshold (f32)
    add     x19, x19, #8
    ldr     x8, [x19], #8         // src_ptr
    ldr     x11, [x19], #8        // bitmap_out
    cbz     w22, .L_dispatch
    ptrue   p0.s
    mov     z16.s, s16             // broadcast threshold to all .s lanes
    cntw    x9                    // elements per z-vector (16 on M4)
    // ── Build bit-position mask: {1,2,4,8,16,32,64,128} repeating every 8 bytes ──
    movz    x24, #0x0201
    movk    x24, #0x0804, lsl #16
    movk    x24, #0x2010, lsl #32
    movk    x24, #0x8040, lsl #48
    dup     z17.d, x24             // z17.b = {1,2,4,8,16,32,64,128, 1,2,4,8,...}
    ptrue   p2.b                   // byte-granularity predicate for mul/addp
    mov     x10, #0               // element index
.L_thr_loop:
    // ── Check remaining elements ──
    sub     x24, x22, x10          // remaining = dim - current
    cmp     x24, x9
    b.lt    .L_thr_tail
    // ── Full vector: 16 float32 elements → 2 bytes of bitmap ──
    ld1w    {z0.s}, p0/z, [x8]
    fcmgt   p1.s, p0/z, z0.s, z16.s
    // ── Materialize predicate as 0/1 int32 values ──
    mov     z1.d, #0
    mov     z1.s, p1/m, #1
    // ── Narrow int32 → int16 → int8 (16 elements → 16 bytes in low half) ──
    uzp1    z1.h, z1.h, z1.h      // 32→16 bit lanes
    uzp1    z1.b, z1.b, z1.b      // 16→8 bit lanes: bytes 0-15 = 0 or 1
    // ── Multiply by bit-position mask: byte i → 0 or 2^(i%8) ──
    mul     z1.b, p2/m, z1.b, z17.b
    // ── Pairwise-add tree: pack 16 bytes → 2 bytes ──
    addp    z1.b, p2/m, z1.b, z1.b
    addp    z1.b, p2/m, z1.b, z1.b
    addp    z1.b, p2/m, z1.b, z1.b
    // ── Extract and store 2 bytes of bitmap ──
    fmov    w24, s1                // low 32 bits of z1 → w24
    strh    w24, [x11]            // store 2 bytes of bitmap
    add     x11, x11, #2
    add     x8, x8, x9, lsl #2    // src += 16 * 4 bytes
    add     x10, x10, x9          // element_index += 16
    b       .L_thr_loop
.L_thr_tail:
    // ── Handle remaining 0-15 elements ──
    cbz     x24, .L_dispatch
    mov     x13, x24               // save remaining count before fmov clobbers x24
    whilelt p3.s, xzr, x24        // predicate for remaining elements
    ld1w    {z0.s}, p3/z, [x8]
    fcmgt   p1.s, p3/z, z0.s, z16.s
    // ── Materialize, narrow, pack (same as full path) ──
    mov     z1.d, #0
    mov     z1.s, p1/m, #1
    uzp1    z1.h, z1.h, z1.h
    uzp1    z1.b, z1.b, z1.b
    mul     z1.b, p2/m, z1.b, z17.b
    addp    z1.b, p2/m, z1.b, z1.b
    addp    z1.b, p2/m, z1.b, z1.b
    addp    z1.b, p2/m, z1.b, z1.b
    // ── Store ceil(remaining/8) bytes ──
    fmov    w24, s1                // extract packed bitmap
    cmp     x13, #8               // compare remaining count (not bitmap data)
    b.le    .L_thr_tail_1b
    strh    w24, [x11]            // 9-15 elements: store 2 bytes
    b       .L_dispatch
.L_thr_tail_1b:
    strb    w24, [x11]            // 1-8 elements: store 1 byte
    b       .L_dispatch
// ================================================================
// WELFORD_STATS_FP32 (0x2B) — online mean/stddev/maxabs/scale via Welford
// Processes n_vectors of dimension dim fp32 inputs, outputs per-dimension
// statistics as 4*dim doubles: mean[dim], stddev[dim], maxabs[dim], scale[dim]
// All accumulation in f64 for numerical stability.
//
// Bytecode: [0x2B][n_vectors:u32][dim:u32][src_ptr:u64][stats_out:u64]
// src_ptr  = n_vectors * dim float32 (row-major AoS)
// stats_out = 4 * dim doubles: mean[dim], stddev[dim], maxabs[dim], scale[dim]
//
// Outer loop: SVLd dimensions per chunk (8 on M4)
// Inner loop: n_vectors, loading one fp32 per f64 lane via ld1w {z.d}
//
// Stack frame: 48 bytes
//   [0]  src_ptr      [8]  stats_out
//   [16] n_vectors    [20] dim
//   [24] dim_offset   [32] row_stride (dim*4)
// ================================================================
.L_op_welford_stats_fp32:
    ldr     w22, [x19]             // n_vectors (u32)
    ldr     w23, [x19, #4]        // dim (u32)
    add     x19, x19, #8
    ldr     x8, [x19], #8         // src_ptr (float32*)
    ldr     x11, [x19], #8        // stats_out (double*)
    sub     sp, sp, #48
    str     x8, [sp, #0]
    str     x11, [sp, #8]
    str     w22, [sp, #16]
    str     w23, [sp, #20]
    lsl     w24, w23, #2           // row_stride = dim * 4 bytes
    str     w24, [sp, #32]
    ptrue   p0.d
    cntd    x9                     // SVLd = 8 on M4
    mov     x10, #0                // dim_offset = 0
.L_wf_dim_loop:
    cmp     w10, w23
    b.ge    .L_wf_done
    str     x10, [sp, #24]
    whilelt p1.d, x10, x23        // tail predicate for this dim chunk
    // ── Init accumulators: mean, M2, maxabs as f64 vectors ──
    fmov    z4.d, #0.0             // mean
    fmov    z5.d, #0.0             // M2
    fmov    z6.d, #0.0             // maxabs
    // ── Compute base ptr: &src[0 * dim + dim_offset] ──
    ldr     x8, [sp, #0]
    add     x8, x8, x10, lsl #2   // src + dim_offset * sizeof(float)
    ldr     w22, [sp, #16]        // n_vectors
    ldr     w24, [sp, #32]        // row_stride (bytes)
    mov     x12, #0                // count = 0
.L_wf_vec_loop:
    cmp     w12, w22
    b.ge    .L_wf_vec_done
    // ── Load SVLd fp32 values, widen to f64 ──
    ld1w    {z0.d}, p1/z, [x8]    // one fp32 per f64 lane (zero-ext upper 32b)
    fcvt    z0.d, p1/m, z0.s      // widen fp32 → fp64
    // ── maxabs = max(maxabs, |x|) ──
    fabs    z1.d, p1/m, z0.d
    fmax    z6.d, p1/m, z6.d, z1.d
    // ── Welford update ──
    add     x12, x12, #1
    ucvtf   d16, x12               // count → f64 scalar
    fmov    d17, #1.0
    fdiv    d16, d17, d16           // inv_count = 1.0 / count
    mov     z7.d, d16              // broadcast inv_count
    fsub    z1.d, z0.d, z4.d      // delta = x - mean
    fmla    z4.d, p1/m, z1.d, z7.d // mean += delta / count
    fsub    z2.d, z0.d, z4.d      // delta2 = x - mean (updated)
    fmla    z5.d, p1/m, z1.d, z2.d // M2 += delta * delta2
    add     x8, x8, x24           // advance src by row_stride
    b       .L_wf_vec_loop
.L_wf_vec_done:
    // ── Finalize: stddev = sqrt(M2 / count), scale = 7.0 / maxabs ──
    ldr     w22, [sp, #16]
    ucvtf   d16, w22               // count → f64
    fmov    d17, #1.0
    fdiv    d16, d17, d16          // 1/count
    mov     z7.d, d16
    mov     z3.d, z5.d
    fmul    z3.d, p1/m, z3.d, z7.d // variance = M2/count
    fsqrt   z3.d, p1/m, z3.d      // stddev
    movz    x13, #0x401C, lsl #48 // 7.0 as f64 = 0x401C_0000_0000_0000
    fmov    d16, x13
    mov     z7.d, d16              // broadcast 7.0
    fcmgt   p2.d, p1/z, z6.d, #0.0
    fmov    z2.d, #0.0             // default scale = 0
    mov     z1.d, z7.d             // z1 = 7.0
    fdiv    z1.d, p2/m, z1.d, z6.d // 7.0/maxabs where maxabs > 0
    sel     z2.d, p2, z1.d, z2.d
    // ── Store: mean, stddev, maxabs, scale ──
    ldr     x11, [sp, #8]
    ldr     x10, [sp, #24]
    ldr     w23, [sp, #20]
    st1d    {z4.d}, p1, [x11, x10, lsl #3]
    add     x13, x10, x23
    st1d    {z3.d}, p1, [x11, x13, lsl #3]
    add     x13, x13, x23
    st1d    {z6.d}, p1, [x11, x13, lsl #3]
    add     x13, x13, x23
    st1d    {z2.d}, p1, [x11, x13, lsl #3]
    add     x10, x10, x9
    b       .L_wf_dim_loop
.L_wf_done:
    add     sp, sp, #48
    b       .L_dispatch
// ================================================================
// QUANTIZE_PACK_4BIT_FP32 (0x2C) — quantize fp32 to signed 4-bit SoA packed
// Dual source: raw + DCT quantized simultaneously.
// Output: SoA layout, per-dim columns of ceil(n/2) packed bytes.
//
// Bytecode: [0x2C][n:u32][dim:u32][src_ptr:u64][stats_ptr:u64]
//           [raw_out:u64][dct_src:u64][dct_out:u64]
// stats_ptr = welford output: mean[dim](f64) at offset 0, scale[dim](f64) at 3*dim
//
// Per element: q = clamp(round((val - mean) * scale), -7, +7)
// Pack: low nibble = even vec idx, high nibble = odd vec idx
// SoA: output[d * packed_row + v/2] = packed pair for dim d
//
// Stack frame: 80 bytes
//   [0]  src_ptr      [8]  stats_ptr     [16] raw_out
//   [24] dct_src      [32] dct_out       [40] n  [44] dim
//   [48] dim_offset   [56] packed_row_bytes  [64] row_stride
// ================================================================
.L_op_quantize_pack_4bit_fp32:
    ldr     w22, [x19]             // n
    ldr     w23, [x19, #4]        // dim
    add     x19, x19, #8
    ldr     x8, [x19], #8         // src_ptr
    ldr     x11, [x19], #8        // stats_ptr
    ldr     x12, [x19], #8        // raw_out
    ldr     x13, [x19], #8        // dct_src
    ldr     x14, [x19], #8        // dct_out
    sub     sp, sp, #80
    str     x8, [sp, #0]
    str     x11, [sp, #8]
    str     x12, [sp, #16]
    str     x13, [sp, #24]
    str     x14, [sp, #32]
    str     w22, [sp, #40]
    str     w23, [sp, #44]
    add     w24, w22, #1
    lsr     w24, w24, #1           // packed_row_bytes = ceil(n/2)
    str     w24, [sp, #56]
    lsl     w24, w23, #2           // row_stride = dim*4
    str     w24, [sp, #64]
    mov     x10, #0                // dim_offset = 0
.L_qp_dim_loop:
    ldr     w23, [sp, #44]
    cmp     w10, w23
    b.ge    .L_qp_done
    str     x10, [sp, #48]
    // ── Load mean[d] and scale[d] from stats (f64), convert to fp32 ──
    ldr     x11, [sp, #8]
    ldr     d16, [x11, x10, lsl #3] // mean[d]
    add     x24, x10, x23
    add     x24, x24, x23
    add     x24, x24, x23          // offset = 3*dim + d
    ldr     d17, [x11, x24, lsl #3] // scale[d]
    fcvt    s16, d16
    fcvt    s17, d17
    // ── Output base for this dim ──
    ldr     x12, [sp, #16]
    ldr     x14, [sp, #32]
    ldr     w24, [sp, #56]
    mul     w15, w10, w24
    add     x12, x12, x15         // &raw_out[d * packed_row]
    add     x14, x14, x15         // &dct_out[d * packed_row]
    // ── Src bases for dim d ──
    ldr     x8, [sp, #0]
    add     x8, x8, x10, lsl #2   // &src[d]
    ldr     x13, [sp, #24]
    add     x13, x13, x10, lsl #2 // &dct[d]
    ldr     w22, [sp, #40]        // n
    ldr     w24, [sp, #64]        // row_stride bytes
    mov     x10, #0                // vec_idx
.L_qp_vec_pair:
    cmp     w10, w22
    b.ge    .L_qp_vec_end
    // ── Quantize raw src[vec_idx * dim + d] ──
    mul     w15, w10, w24
    ldr     s0, [x8, x15]
    fsub    s0, s0, s16
    fmul    s0, s0, s17
    frintn  s0, s0
    fcvtzs  w0, s0
    mov     w26, #-7
    cmp     w0, w26
    csel    w0, w26, w0, lt        // clamp low: max(val, -7)
    mov     w26, #7
    cmp     w0, w26
    csel    w0, w26, w0, gt        // clamp high: min(val, 7)
    and     w0, w0, #0x0F
    // ── Quantize dct[vec_idx * dim + d] ──
    ldr     s0, [x13, x15]
    fsub    s0, s0, s16
    fmul    s0, s0, s17
    frintn  s0, s0
    fcvtzs  w1, s0
    mov     w26, #-7
    cmp     w1, w26
    csel    w1, w26, w1, lt
    mov     w26, #7
    cmp     w1, w26
    csel    w1, w26, w1, gt
    and     w1, w1, #0x0F
    // ── Check for odd partner ──
    add     w10, w10, #1
    cmp     w10, w22
    b.ge    .L_qp_store_unpaired
    // ── Quantize raw src[vec_idx+1] ──
    mul     w15, w10, w24
    ldr     s0, [x8, x15]
    fsub    s0, s0, s16
    fmul    s0, s0, s17
    frintn  s0, s0
    fcvtzs  w2, s0
    mov     w26, #-7
    cmp     w2, w26
    csel    w2, w26, w2, lt
    mov     w26, #7
    cmp     w2, w26
    csel    w2, w26, w2, gt
    and     w2, w2, #0x0F
    orr     w0, w0, w2, lsl #4
    // ── Quantize dct[vec_idx+1] ──
    ldr     s0, [x13, x15]
    fsub    s0, s0, s16
    fmul    s0, s0, s17
    frintn  s0, s0
    fcvtzs  w2, s0
    mov     w26, #-7
    cmp     w2, w26
    csel    w2, w26, w2, lt
    mov     w26, #7
    cmp     w2, w26
    csel    w2, w26, w2, gt
    and     w2, w2, #0x0F
    orr     w1, w1, w2, lsl #4
    // ── Store packed byte ──
    sub     w15, w10, #1           // even index
    lsr     w15, w15, #1           // byte_idx
    strb    w0, [x12, x15]
    strb    w1, [x14, x15]
    add     w10, w10, #1
    b       .L_qp_vec_pair
.L_qp_store_unpaired:
    sub     w15, w10, #1
    lsr     w15, w15, #1
    strb    w0, [x12, x15]
    strb    w1, [x14, x15]
.L_qp_vec_end:
    ldr     x10, [sp, #48]
    add     x10, x10, #1
    b       .L_qp_dim_loop
.L_qp_done:
    add     sp, sp, #80
    b       .L_dispatch
// ================================================================
// THRESHOLD_8BIT (0x2D) — reconstruct 8-bit counters from 8 bitplanes,
// compare > threshold, produce bitmap output.
//
// Bytecode: [0x2D][n_bytes:u32][threshold:u8][src_ptr:u64][bitmap_out:u64]
// src_ptr = 8 contiguous bitplanes, each n_bytes long
// For each byte position j, bit position i: counter = sum(bp_k bit_i << k)
//   output bit i = (counter > threshold)
// ================================================================
.L_op_threshold_8bit:
    ldr     w22, [x19]             // n_bytes
    ldrb    w23, [x19, #4]        // threshold
    add     x19, x19, #5
    ldr     x8, [x19], #8         // src_ptr
    ldr     x11, [x19], #8        // bitmap_out
    // Bitplane base pointers: bp_k = src + k * n_bytes
    mov     x0, x8
    add     x1, x8, x22
    add     x2, x1, x22
    add     x3, x2, x22
    add     x4, x3, x22
    add     x5, x4, x22
    add     x6, x5, x22
    add     x7, x6, x22
    mov     x10, #0
.L_th8_byte:
    cmp     w10, w22
    b.ge    .L_th8_done
    ldrb    w12, [x0, x10]        // bp0[j]
    ldrb    w13, [x1, x10]        // bp1[j]
    ldrb    w14, [x2, x10]        // bp2[j]
    ldrb    w15, [x3, x10]        // bp3[j]
    ldrb    w16, [x4, x10]        // bp4[j]
    ldrb    w17, [x5, x10]        // bp5[j]
    ldrb    w18, [x6, x10]        // bp6[j]
    ldrb    w24, [x7, x10]        // bp7[j]
    mov     w9, #0                 // result byte
    // ── Bit 0: reconstruct counter from bit 0 of each bitplane ──
    and     w26, w12, #1
    ubfx    w20, w13, #0, #1
    orr     w26, w26, w20, lsl #1
    ubfx    w20, w14, #0, #1
    orr     w26, w26, w20, lsl #2
    ubfx    w20, w15, #0, #1
    orr     w26, w26, w20, lsl #3
    ubfx    w20, w16, #0, #1
    orr     w26, w26, w20, lsl #4
    ubfx    w20, w17, #0, #1
    orr     w26, w26, w20, lsl #5
    ubfx    w20, w18, #0, #1
    orr     w26, w26, w20, lsl #6
    ubfx    w20, w24, #0, #1
    orr     w26, w26, w20, lsl #7
    cmp     w26, w23
    cset    w20, hi
    orr     w9, w9, w20
    // ── Bit 1 ──
    ubfx    w26, w12, #1, #1
    ubfx    w20, w13, #1, #1
    orr     w26, w26, w20, lsl #1
    ubfx    w20, w14, #1, #1
    orr     w26, w26, w20, lsl #2
    ubfx    w20, w15, #1, #1
    orr     w26, w26, w20, lsl #3
    ubfx    w20, w16, #1, #1
    orr     w26, w26, w20, lsl #4
    ubfx    w20, w17, #1, #1
    orr     w26, w26, w20, lsl #5
    ubfx    w20, w18, #1, #1
    orr     w26, w26, w20, lsl #6
    ubfx    w20, w24, #1, #1
    orr     w26, w26, w20, lsl #7
    cmp     w26, w23
    cset    w20, hi
    orr     w9, w9, w20, lsl #1
    // ── Bit 2 ──
    ubfx    w26, w12, #2, #1
    ubfx    w20, w13, #2, #1
    orr     w26, w26, w20, lsl #1
    ubfx    w20, w14, #2, #1
    orr     w26, w26, w20, lsl #2
    ubfx    w20, w15, #2, #1
    orr     w26, w26, w20, lsl #3
    ubfx    w20, w16, #2, #1
    orr     w26, w26, w20, lsl #4
    ubfx    w20, w17, #2, #1
    orr     w26, w26, w20, lsl #5
    ubfx    w20, w18, #2, #1
    orr     w26, w26, w20, lsl #6
    ubfx    w20, w24, #2, #1
    orr     w26, w26, w20, lsl #7
    cmp     w26, w23
    cset    w20, hi
    orr     w9, w9, w20, lsl #2
    // ── Bit 3 ──
    ubfx    w26, w12, #3, #1
    ubfx    w20, w13, #3, #1
    orr     w26, w26, w20, lsl #1
    ubfx    w20, w14, #3, #1
    orr     w26, w26, w20, lsl #2
    ubfx    w20, w15, #3, #1
    orr     w26, w26, w20, lsl #3
    ubfx    w20, w16, #3, #1
    orr     w26, w26, w20, lsl #4
    ubfx    w20, w17, #3, #1
    orr     w26, w26, w20, lsl #5
    ubfx    w20, w18, #3, #1
    orr     w26, w26, w20, lsl #6
    ubfx    w20, w24, #3, #1
    orr     w26, w26, w20, lsl #7
    cmp     w26, w23
    cset    w20, hi
    orr     w9, w9, w20, lsl #3
    // ── Bit 4 ──
    ubfx    w26, w12, #4, #1
    ubfx    w20, w13, #4, #1
    orr     w26, w26, w20, lsl #1
    ubfx    w20, w14, #4, #1
    orr     w26, w26, w20, lsl #2
    ubfx    w20, w15, #4, #1
    orr     w26, w26, w20, lsl #3
    ubfx    w20, w16, #4, #1
    orr     w26, w26, w20, lsl #4
    ubfx    w20, w17, #4, #1
    orr     w26, w26, w20, lsl #5
    ubfx    w20, w18, #4, #1
    orr     w26, w26, w20, lsl #6
    ubfx    w20, w24, #4, #1
    orr     w26, w26, w20, lsl #7
    cmp     w26, w23
    cset    w20, hi
    orr     w9, w9, w20, lsl #4
    // ── Bit 5 ──
    ubfx    w26, w12, #5, #1
    ubfx    w20, w13, #5, #1
    orr     w26, w26, w20, lsl #1
    ubfx    w20, w14, #5, #1
    orr     w26, w26, w20, lsl #2
    ubfx    w20, w15, #5, #1
    orr     w26, w26, w20, lsl #3
    ubfx    w20, w16, #5, #1
    orr     w26, w26, w20, lsl #4
    ubfx    w20, w17, #5, #1
    orr     w26, w26, w20, lsl #5
    ubfx    w20, w18, #5, #1
    orr     w26, w26, w20, lsl #6
    ubfx    w20, w24, #5, #1
    orr     w26, w26, w20, lsl #7
    cmp     w26, w23
    cset    w20, hi
    orr     w9, w9, w20, lsl #5
    // ── Bit 6 ──
    ubfx    w26, w12, #6, #1
    ubfx    w20, w13, #6, #1
    orr     w26, w26, w20, lsl #1
    ubfx    w20, w14, #6, #1
    orr     w26, w26, w20, lsl #2
    ubfx    w20, w15, #6, #1
    orr     w26, w26, w20, lsl #3
    ubfx    w20, w16, #6, #1
    orr     w26, w26, w20, lsl #4
    ubfx    w20, w17, #6, #1
    orr     w26, w26, w20, lsl #5
    ubfx    w20, w18, #6, #1
    orr     w26, w26, w20, lsl #6
    ubfx    w20, w24, #6, #1
    orr     w26, w26, w20, lsl #7
    cmp     w26, w23
    cset    w20, hi
    orr     w9, w9, w20, lsl #6
    // ── Bit 7 ──
    ubfx    w26, w12, #7, #1
    ubfx    w20, w13, #7, #1
    orr     w26, w26, w20, lsl #1
    ubfx    w20, w14, #7, #1
    orr     w26, w26, w20, lsl #2
    ubfx    w20, w15, #7, #1
    orr     w26, w26, w20, lsl #3
    ubfx    w20, w16, #7, #1
    orr     w26, w26, w20, lsl #4
    ubfx    w20, w17, #7, #1
    orr     w26, w26, w20, lsl #5
    ubfx    w20, w18, #7, #1
    orr     w26, w26, w20, lsl #6
    ubfx    w20, w24, #7, #1
    orr     w26, w26, w20, lsl #7
    cmp     w26, w23
    cset    w20, hi
    orr     w9, w9, w20, lsl #7
    strb    w9, [x11, x10]
    add     x10, x10, #1
    b       .L_th8_byte
.L_th8_done:
    b       .L_dispatch
// ================================================================
// QUANTIZE_ACCUM_2BIT (0x2E)
// Ternary 2-bit decode {00=0, 01=+1, 11=-1}, bf16 scale, bf16 accum.
// Encoding: [0x2E][count:u32][packed_ptr:u64][scale_ptr:u64][accum_ptr:u64]
// Per iter: 16 elems (cntw) = 4 packed bytes. Decode via scalar GP into
// int32 scratch on stack, then ld1w → scvtf → fmul scale → fadd accum.
// bf16 ↔ fp32: ld1h into .s (zero-ext) + lsl #16; lsr #16 + st1h from .s.
// ================================================================
.L_op_quantize_accum_2bit:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8         // packed_ptr
    ldr     x11, [x19], #8        // scale_ptr (bf16*)
    ldr     x13, [x19], #8        // accum_ptr (bf16*)
    cbz     w22, .L_dispatch
    ptrue   p0.s
    sub     sp, sp, #64            // scratch: 16 x int32
    cntw    x9
    mov     x10, #0
    whilelt p1.s, x10, x22
.L_qa2b_loop:
    lsr     x14, x10, #2          // packed byte offset
    mov     x15, #0                // stack word offset (bytes)
    mov     w0, #4                 // 4 bytes to decode
.L_qa2b_byte:
    ldrb    w16, [x8, x14]
    add     x14, x14, #1
    and     w1, w16, #0x03         // crumb 0
    cmp     w1, #1
    cset    w2, eq
    cmp     w1, #3
    csinv   w2, w2, wzr, ne
    str     w2, [sp, x15]
    add     x15, x15, #4
    ubfx    w1, w16, #2, #2        // crumb 1
    cmp     w1, #1
    cset    w2, eq
    cmp     w1, #3
    csinv   w2, w2, wzr, ne
    str     w2, [sp, x15]
    add     x15, x15, #4
    ubfx    w1, w16, #4, #2        // crumb 2
    cmp     w1, #1
    cset    w2, eq
    cmp     w1, #3
    csinv   w2, w2, wzr, ne
    str     w2, [sp, x15]
    add     x15, x15, #4
    lsr     w1, w16, #6            // crumb 3
    cmp     w1, #1
    cset    w2, eq
    cmp     w1, #3
    csinv   w2, w2, wzr, ne
    str     w2, [sp, x15]
    add     x15, x15, #4
    subs    w0, w0, #1
    b.ne    .L_qa2b_byte
    ld1w    {z0.s}, p0/z, [sp]    // 16 decoded ternary as int32
    scvtf   z0.s, p0/m, z0.s      // → fp32
    ld1h    {z1.s}, p1/z, [x11, x10, lsl #1]  // bf16 scale → u32 (zero-ext)
    lsl     z1.s, z1.s, #16       // bf16 → fp32
    fmul    z0.s, p1/m, z0.s, z1.s
    ld1h    {z2.s}, p1/z, [x13, x10, lsl #1]  // bf16 accum → u32
    lsl     z2.s, z2.s, #16       // bf16 → fp32
    fadd    z2.s, p1/m, z2.s, z0.s
    lsr     z2.s, z2.s, #16       // fp32 → bf16
    st1h    {z2.s}, p1, [x13, x10, lsl #1]
    add     x10, x10, x9
    whilelt p1.s, x10, x22
    b.first .L_qa2b_loop
    add     sp, sp, #64
    b       .L_dispatch
// ================================================================
// ACCUM_8BIT (0x2F)
// INT8 signed data * bf16 per-coeff scale → bf16 accumulator.
// Encoding: [0x2F][count:u32][data_ptr:u64][scale_ptr:u64][accum_ptr:u64]
// Per iter: 16 elements. ld1sb into .s (sign-extend i8→i32), scvtf.
// ================================================================
.L_op_accum_8bit:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8         // data_ptr (int8_t*)
    ldr     x11, [x19], #8        // scale_ptr (bf16*)
    ldr     x13, [x19], #8        // accum_ptr (bf16*)
    cbz     w22, .L_dispatch
    ptrue   p0.s
    mov     x10, #0
    whilelt p1.s, x10, x22
.L_a8b_loop:
    ld1sb   {z0.s}, p1/z, [x8, x10]
    scvtf   z0.s, p0/m, z0.s
    ld1h    {z1.s}, p1/z, [x11, x10, lsl #1]
    lsl     z1.s, z1.s, #16
    fmul    z0.s, p1/m, z0.s, z1.s
    ld1h    {z2.s}, p1/z, [x13, x10, lsl #1]
    lsl     z2.s, z2.s, #16
    fadd    z2.s, p1/m, z2.s, z0.s
    lsr     z2.s, z2.s, #16
    st1h    {z2.s}, p1, [x13, x10, lsl #1]
    incw    x10
    whilelt p1.s, x10, x22
    b.first .L_a8b_loop
    b       .L_dispatch
// ================================================================
// SOA_SUB_SCALE_BF16 (0x30)
// SoA quantized L2 partial: accum[i] += bf16((src[i]*scale - scalar)^2)
// Encoding: [0x30][count:u32][src_ptr:u64][scalar:f32][scale:f32][accum_ptr:u64]
// Per iter: 16 elements. ld1b into .s (u8→u32), ucvtf, fmul scale, fsub scalar, square.
// ================================================================
.L_op_soa_sub_scale_bf16:
    ldr     w22, [x19]
    ldr     s16, [x19, #4]
    ldr     s17, [x19, #8]
    add     x19, x19, #12
    ldr     x8, [x19], #8         // src_ptr (uint8_t*)
    ldr     x13, [x19], #8        // accum_ptr (bf16*)
    cbz     w22, .L_dispatch
    ptrue   p0.s
    mov     z16.s, s16             // broadcast scalar
    mov     z17.s, s17             // broadcast scale
    mov     x10, #0
    whilelt p1.s, x10, x22
.L_soa_ss_loop:
    ld1b    {z0.s}, p1/z, [x8, x10]
    ucvtf   z0.s, p0/m, z0.s
    fmul    z0.s, p0/m, z0.s, z17.s
    fsub    z0.s, z0.s, z16.s
    fmul    z0.s, p0/m, z0.s, z0.s        // square
    ld1h    {z1.s}, p1/z, [x13, x10, lsl #1]
    lsl     z1.s, z1.s, #16
    fadd    z1.s, p1/m, z1.s, z0.s
    lsr     z1.s, z1.s, #16
    st1h    {z1.s}, p1, [x13, x10, lsl #1]
    incw    x10
    whilelt p1.s, x10, x22
    b.first .L_soa_ss_loop
    b       .L_dispatch
// ================================================================
// SOA_LUTI2_ACCUM (0x31)
// LUTI2 expand 2-bit packed indices via ZT0, accumulate bf16.
// Encoding: [0x31][count:u32][packed_ptr:u64][table_ptr:u64][accum_ptr:u64]
// count = number of 2-bit elements (4 per byte).
// luti2 z.h, zt0, z[seg]: 32 input bytes → 32 halfword outputs per segment.
// Segments [0]-[3] extract crumbs 0-3 from each byte (bits [1:0] thru [7:6]).
// Per z-load of 32 packed bytes: 4 segments x 32 = 128 elements.
// Accumulate by widening halfwords to fp32 via uunpklo/hi + lsl #16.
// ================================================================
.L_op_soa_luti2_accum:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8         // packed_ptr
    ldr     x11, [x19], #8        // table_ptr (64 bytes → ZT0)
    ldr     x13, [x19], #8        // accum_ptr (bf16*)
    cbz     w22, .L_dispatch
    ldr     zt0, [x11]
    ptrue   p0.s
    ptrue   p2.b
    cntw    x9                     // 16
    mov     x10, #0                // element offset
    mov     x15, #0                // packed byte offset
.L_soa_l2a_loop:
    cmp     x10, x22
    b.hs    .L_soa_l2a_done
    ld1b    {z0.b}, p2/z, [x8, x15]
    // Segment [0]: bits[1:0] → 32 halfwords
    luti2   z1.h, zt0, z0[0]
    // Low 16 halfwords → fp32, accumulate, store
    whilelt p1.s, x10, x22
    uunpklo z2.s, z1.h
    lsl     z2.s, z2.s, #16
    ld1h    {z3.s}, p1/z, [x13, x10, lsl #1]
    lsl     z3.s, z3.s, #16
    fadd    z3.s, p1/m, z3.s, z2.s
    lsr     z3.s, z3.s, #16
    st1h    {z3.s}, p1, [x13, x10, lsl #1]
    add     x10, x10, x9
    // High 16 halfwords
    whilelt p1.s, x10, x22
    b.none  .L_soa_l2a_done
    uunpkhi z2.s, z1.h
    lsl     z2.s, z2.s, #16
    ld1h    {z3.s}, p1/z, [x13, x10, lsl #1]
    lsl     z3.s, z3.s, #16
    fadd    z3.s, p1/m, z3.s, z2.s
    lsr     z3.s, z3.s, #16
    st1h    {z3.s}, p1, [x13, x10, lsl #1]
    add     x10, x10, x9
    // Segment [1]: bits[3:2]
    luti2   z1.h, zt0, z0[1]
    whilelt p1.s, x10, x22
    b.none  .L_soa_l2a_done
    uunpklo z2.s, z1.h
    lsl     z2.s, z2.s, #16
    ld1h    {z3.s}, p1/z, [x13, x10, lsl #1]
    lsl     z3.s, z3.s, #16
    fadd    z3.s, p1/m, z3.s, z2.s
    lsr     z3.s, z3.s, #16
    st1h    {z3.s}, p1, [x13, x10, lsl #1]
    add     x10, x10, x9
    whilelt p1.s, x10, x22
    b.none  .L_soa_l2a_done
    uunpkhi z2.s, z1.h
    lsl     z2.s, z2.s, #16
    ld1h    {z3.s}, p1/z, [x13, x10, lsl #1]
    lsl     z3.s, z3.s, #16
    fadd    z3.s, p1/m, z3.s, z2.s
    lsr     z3.s, z3.s, #16
    st1h    {z3.s}, p1, [x13, x10, lsl #1]
    add     x10, x10, x9
    // Segment [2]: bits[5:4]
    luti2   z1.h, zt0, z0[2]
    whilelt p1.s, x10, x22
    b.none  .L_soa_l2a_done
    uunpklo z2.s, z1.h
    lsl     z2.s, z2.s, #16
    ld1h    {z3.s}, p1/z, [x13, x10, lsl #1]
    lsl     z3.s, z3.s, #16
    fadd    z3.s, p1/m, z3.s, z2.s
    lsr     z3.s, z3.s, #16
    st1h    {z3.s}, p1, [x13, x10, lsl #1]
    add     x10, x10, x9
    whilelt p1.s, x10, x22
    b.none  .L_soa_l2a_done
    uunpkhi z2.s, z1.h
    lsl     z2.s, z2.s, #16
    ld1h    {z3.s}, p1/z, [x13, x10, lsl #1]
    lsl     z3.s, z3.s, #16
    fadd    z3.s, p1/m, z3.s, z2.s
    lsr     z3.s, z3.s, #16
    st1h    {z3.s}, p1, [x13, x10, lsl #1]
    add     x10, x10, x9
    // Segment [3]: bits[7:6]
    luti2   z1.h, zt0, z0[3]
    whilelt p1.s, x10, x22
    b.none  .L_soa_l2a_done
    uunpklo z2.s, z1.h
    lsl     z2.s, z2.s, #16
    ld1h    {z3.s}, p1/z, [x13, x10, lsl #1]
    lsl     z3.s, z3.s, #16
    fadd    z3.s, p1/m, z3.s, z2.s
    lsr     z3.s, z3.s, #16
    st1h    {z3.s}, p1, [x13, x10, lsl #1]
    add     x10, x10, x9
    whilelt p1.s, x10, x22
    b.none  .L_soa_l2a_done
    uunpkhi z2.s, z1.h
    lsl     z2.s, z2.s, #16
    ld1h    {z3.s}, p1/z, [x13, x10, lsl #1]
    lsl     z3.s, z3.s, #16
    fadd    z3.s, p1/m, z3.s, z2.s
    lsr     z3.s, z3.s, #16
    st1h    {z3.s}, p1, [x13, x10, lsl #1]
    add     x10, x10, x9
    add     x15, x15, #32         // 128 elements from 32 packed bytes
    b       .L_soa_l2a_loop
.L_soa_l2a_done:
    b       .L_dispatch
// ================================================================
// SOA_LUTI4_ACCUM (0x32)
// LUTI4 expand 4-bit packed indices via ZT0, accumulate bf16.
// Encoding: [0x32][count:u32][packed_ptr:u64][table_ptr:u64][accum_ptr:u64]
// count = number of 4-bit elements (2 per byte).
// luti4 z.h, zt0, z[seg]: seg[0]=low nibble, seg[1]=high nibble of 32 bytes.
// Per z-load of 32 packed bytes: 2 segments x 32 = 64 elements.
// ================================================================
.L_op_soa_luti4_accum:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8         // packed_ptr
    ldr     x11, [x19], #8        // table_ptr (64 bytes → ZT0)
    ldr     x13, [x19], #8        // accum_ptr (bf16*)
    cbz     w22, .L_dispatch
    ldr     zt0, [x11]
    ptrue   p0.s
    ptrue   p2.b
    cntw    x9                     // 16
    mov     x10, #0
    mov     x15, #0
.L_soa_l4a_loop:
    cmp     x10, x22
    b.hs    .L_soa_l4a_done
    ld1b    {z0.b}, p2/z, [x8, x15]
    // Segment [0]: low nibbles → 32 halfwords
    luti4   z1.h, zt0, z0[0]
    whilelt p1.s, x10, x22
    uunpklo z2.s, z1.h
    lsl     z2.s, z2.s, #16
    ld1h    {z3.s}, p1/z, [x13, x10, lsl #1]
    lsl     z3.s, z3.s, #16
    fadd    z3.s, p1/m, z3.s, z2.s
    lsr     z3.s, z3.s, #16
    st1h    {z3.s}, p1, [x13, x10, lsl #1]
    add     x10, x10, x9
    whilelt p1.s, x10, x22
    b.none  .L_soa_l4a_done
    uunpkhi z2.s, z1.h
    lsl     z2.s, z2.s, #16
    ld1h    {z3.s}, p1/z, [x13, x10, lsl #1]
    lsl     z3.s, z3.s, #16
    fadd    z3.s, p1/m, z3.s, z2.s
    lsr     z3.s, z3.s, #16
    st1h    {z3.s}, p1, [x13, x10, lsl #1]
    add     x10, x10, x9
    // Segment [1]: high nibbles → 32 halfwords
    luti4   z1.h, zt0, z0[1]
    whilelt p1.s, x10, x22
    b.none  .L_soa_l4a_done
    uunpklo z2.s, z1.h
    lsl     z2.s, z2.s, #16
    ld1h    {z3.s}, p1/z, [x13, x10, lsl #1]
    lsl     z3.s, z3.s, #16
    fadd    z3.s, p1/m, z3.s, z2.s
    lsr     z3.s, z3.s, #16
    st1h    {z3.s}, p1, [x13, x10, lsl #1]
    add     x10, x10, x9
    whilelt p1.s, x10, x22
    b.none  .L_soa_l4a_done
    uunpkhi z2.s, z1.h
    lsl     z2.s, z2.s, #16
    ld1h    {z3.s}, p1/z, [x13, x10, lsl #1]
    lsl     z3.s, z3.s, #16
    fadd    z3.s, p1/m, z3.s, z2.s
    lsr     z3.s, z3.s, #16
    st1h    {z3.s}, p1, [x13, x10, lsl #1]
    add     x10, x10, x9
    add     x15, x15, #32         // 64 elements from 32 packed bytes
    b       .L_soa_l4a_loop
.L_soa_l4a_done:
    b       .L_dispatch
// ================================================================
// BITMAP_SCORE_PIPELINE (0x33)
// Ripple-carry bitmap accumulate + threshold + extract candidate IDs.
// Encoding: [0x33][n_streams:u32][n_bytes:u32][n_vectors:u32]
//           [score_min:u32][max_candidates:u32]
//           [streams_ptr:u64][is_high_ptr:u64]
//           [candidates_out:u64][count_out:u64]
//
// Phase 1: For each stream, XOR with invert mask (if is_high==0), ripple-carry
//   add into 8 bitplanes on stack. SVE-vectorized 64 bytes/iter.
// Phase 2: Scalar reconstruct 8-bit counters per bit position, threshold,
//   emit qualifying vector_id = byte*8 + bit.
//
// Stack: [sp+0] frame_size, [sp+8] score_min, [sp+12] max_candidates,
//   [sp+16] candidates_out, [sp+24] count_out, [sp+32] streams_ptr,
//   [sp+40] is_high_ptr, [sp+48] n_streams, [sp+52] n_bytes,
//   [sp+56] n_vectors, [sp+64] bp_base, [sp+72] pad.
//   [sp+80 ..] 8 * n_bytes bitplane storage.
// ================================================================
.L_op_bitmap_score_pipeline:
    ldr     w22, [x19]             // n_streams
    ldr     w23, [x19, #4]        // n_bytes
    ldr     w24, [x19, #8]        // n_vectors
    ldr     w0, [x19, #12]        // score_min
    ldr     w1, [x19, #16]        // max_candidates
    add     x19, x19, #20
    ldr     x8, [x19], #8         // streams_ptr
    ldr     x11, [x19], #8        // is_high_ptr
    ldr     x13, [x19], #8        // candidates_out
    ldr     x12, [x19], #8        // count_out
    cbz     w22, .L_bsp_zero
    cbz     w23, .L_bsp_zero
    // Allocate stack: 80 + 8*n_bytes, 16-byte aligned
    mov     x2, x23
    lsl     x3, x2, #3
    add     x3, x3, #80
    add     x3, x3, #15
    and     x3, x3, #~15
    sub     sp, sp, x3
    str     x3, [sp, #0]
    str     w0, [sp, #8]
    str     w1, [sp, #12]
    str     x13, [sp, #16]
    str     x12, [sp, #24]
    str     x8, [sp, #32]
    str     x11, [sp, #40]
    str     w22, [sp, #48]
    str     w23, [sp, #52]
    str     w24, [sp, #56]
    add     x4, sp, #80
    str     x4, [sp, #64]
    // Zero all 8 bitplanes
    ptrue   p0.b
    cntb    x9
    mov     z0.d, #0
    mov     x5, x4
    lsl     x6, x2, #3
.L_bsp_zp:
    cbz     x6, .L_bsp_p1
    st1b    {z0.b}, p0, [x5]
    add     x5, x5, x9
    subs    x6, x6, x9
    b.hi    .L_bsp_zp
.L_bsp_p1:
    // ── Phase 1: ripple-carry ──
    ldr     x8, [sp, #32]
    ldr     x11, [sp, #40]
    mov     w3, #0
.L_bsp_stream:
    cmp     w3, w22
    b.hs    .L_bsp_p2
    ldr     x5, [x8, x3, lsl #3]
    ldrb    w6, [x11, x3]
    cmp     w6, #0
    mov     w7, #0xFF
    csel    w7, w7, wzr, eq
    dup     z16.b, w7
    ldr     x4, [sp, #64]
    ldr     w14, [sp, #52]
    mov     x15, #0
.L_bsp_rca:
    cmp     x15, x14
    b.hs    .L_bsp_ns
    whilelt p1.b, x15, x14
    ld1b    {z0.b}, p1/z, [x5, x15]
    eor     z0.b, z0.b, z16.b
    mov     x16, x4
    // bp[0]: XOR+AND ripple
    add     x6, x16, x15
    ld1b    {z1.b}, p1/z, [x6]
    mov     z2.d, z1.d
    eor     z1.b, z1.b, z0.b
    st1b    {z1.b}, p1, [x6]
    and     z0.b, z2.b, z0.b
    add     x16, x16, x2
    // bp[1]
    add     x6, x16, x15
    ld1b    {z1.b}, p1/z, [x6]
    mov     z2.d, z1.d
    eor     z1.b, z1.b, z0.b
    st1b    {z1.b}, p1, [x6]
    and     z0.b, z2.b, z0.b
    add     x16, x16, x2
    // bp[2]
    add     x6, x16, x15
    ld1b    {z1.b}, p1/z, [x6]
    mov     z2.d, z1.d
    eor     z1.b, z1.b, z0.b
    st1b    {z1.b}, p1, [x6]
    and     z0.b, z2.b, z0.b
    add     x16, x16, x2
    // bp[3]
    add     x6, x16, x15
    ld1b    {z1.b}, p1/z, [x6]
    mov     z2.d, z1.d
    eor     z1.b, z1.b, z0.b
    st1b    {z1.b}, p1, [x6]
    and     z0.b, z2.b, z0.b
    add     x16, x16, x2
    // bp[4]
    add     x6, x16, x15
    ld1b    {z1.b}, p1/z, [x6]
    mov     z2.d, z1.d
    eor     z1.b, z1.b, z0.b
    st1b    {z1.b}, p1, [x6]
    and     z0.b, z2.b, z0.b
    add     x16, x16, x2
    // bp[5]
    add     x6, x16, x15
    ld1b    {z1.b}, p1/z, [x6]
    mov     z2.d, z1.d
    eor     z1.b, z1.b, z0.b
    st1b    {z1.b}, p1, [x6]
    and     z0.b, z2.b, z0.b
    add     x16, x16, x2
    // bp[6]
    add     x6, x16, x15
    ld1b    {z1.b}, p1/z, [x6]
    mov     z2.d, z1.d
    eor     z1.b, z1.b, z0.b
    st1b    {z1.b}, p1, [x6]
    and     z0.b, z2.b, z0.b
    add     x16, x16, x2
    // bp[7] (terminal — no carry out)
    add     x6, x16, x15
    ld1b    {z1.b}, p1/z, [x6]
    eor     z1.b, z1.b, z0.b
    st1b    {z1.b}, p1, [x6]
    add     x15, x15, x9
    b       .L_bsp_rca
.L_bsp_ns:
    add     w3, w3, #1
    b       .L_bsp_stream
.L_bsp_p2:
    // ── Phase 2: threshold + extract ──
    ldr     x4, [sp, #64]         // bp_base
    ldr     w14, [sp, #52]        // n_bytes
    ldr     w0, [sp, #8]          // score_min
    ldr     w1, [sp, #12]         // max_candidates
    ldr     x13, [sp, #16]        // candidates_out
    ldr     x12, [sp, #24]        // count_out
    mov     w3, #0                 // candidate_count
    mov     x15, #0                // byte position j
.L_bsp_ex:
    cmp     x15, x14
    b.hs    .L_bsp_done
    // Load bp[0..7][j] into w-registers via computed offsets from bp_base
    ldr     x16, [sp, #64]
    ldrb    w4, [x16, x15]                           // bp[0]
    add     x17, x16, x2
    ldrb    w5, [x17, x15]                           // bp[1]
    add     x17, x17, x2
    ldrb    w6, [x17, x15]                           // bp[2]
    add     x17, x17, x2
    ldrb    w7, [x17, x15]                           // bp[3]
    add     x17, x17, x2
    ldrb    w16, [x17, x15]                          // bp[4]
    add     x17, x17, x2
    ldrb    w17, [x17, x15]                          // bp[5]
    ldr     x10, [sp, #64]
    mov     x23, x2
    lsl     x23, x23, #1
    add     x23, x23, x2, lsl #2          // 6 * n_bytes
    add     x23, x10, x23
    ldrb    w23, [x23, x15]                          // bp[6]
    ldr     x10, [sp, #64]
    mov     x24, x2
    lsl     x24, x24, #3
    sub     x24, x24, x2                  // 7 * n_bytes
    add     x24, x10, x24
    ldrb    w24, [x24, x15]                          // bp[7]
    mov     w8, #0                 // bit b
.L_bsp_bit:
    cmp     w8, #8
    b.hs    .L_bsp_nx
    cmp     w3, w1
    b.hs    .L_bsp_done
    // Reconstruct 8-bit counter from bitplanes
    lsr     w10, w4, w8
    and     w10, w10, #1
    lsr     w11, w5, w8
    and     w11, w11, #1
    orr     w10, w10, w11, lsl #1
    lsr     w11, w6, w8
    and     w11, w11, #1
    orr     w10, w10, w11, lsl #2
    lsr     w11, w7, w8
    and     w11, w11, #1
    orr     w10, w10, w11, lsl #3
    lsr     w11, w16, w8
    and     w11, w11, #1
    orr     w10, w10, w11, lsl #4
    lsr     w11, w17, w8
    and     w11, w11, #1
    orr     w10, w10, w11, lsl #5
    lsr     w11, w23, w8
    and     w11, w11, #1
    orr     w10, w10, w11, lsl #6
    lsr     w11, w24, w8
    and     w11, w11, #1
    orr     w10, w10, w11, lsl #7
    cmp     w10, w0
    b.lo    .L_bsp_sk
    lsl     w11, w15, #3
    add     w11, w11, w8
    str     w11, [x13, x3, lsl #2]
    add     w3, w3, #1
.L_bsp_sk:
    add     w8, w8, #1
    b       .L_bsp_bit
.L_bsp_nx:
    add     x15, x15, #1
    b       .L_bsp_ex
.L_bsp_done:
    str     w3, [x12]
    ldr     x3, [sp, #0]
    add     sp, sp, x3
    b       .L_dispatch
.L_bsp_zero:
    str     wzr, [x12]
    b       .L_dispatch
// ================================================================
// MOV_ZREG (0x34) — Move z{src} to z{dst}
// Encoding: [0x34][src:u8][dst:u8]
// Relays through a VL-sized stack slot. Two 32-entry branch tables
// handle the dynamic register selection. This is the only opcode
// that needs branch tables — all other kernels use fixed z-regs.
// ================================================================
.L_op_mov_zreg:
    ldrb    w9, [x19], #1          // src index (0-31)
    ldrb    w10, [x19], #1         // dst index (0-31)
    ptrue   p0.b
    addvl   sp, sp, #-1            // allocate VL bytes on stack
    // ── Phase 1: store z{src} to [sp] ──
    adr     x8, .L_mzr_st0
    add     x8, x8, x9, lsl #3    // each entry = 8 bytes (2 insns)
    br      x8
.L_mzr_phase2:
    // ── Phase 2: load [sp] into z{dst} ──
    adr     x8, .L_mzr_ld0
    add     x8, x8, x10, lsl #3
    br      x8
.L_mzr_done:
    addvl   sp, sp, #1             // restore stack
    b       .L_dispatch
// ── Store table: z{N} → [sp], 8 bytes per entry (str + b) ──
.L_mzr_st0:  str z0,  [sp]
    b .L_mzr_phase2
.L_mzr_st1:  str z1,  [sp]
    b .L_mzr_phase2
.L_mzr_st2:  str z2,  [sp]
    b .L_mzr_phase2
.L_mzr_st3:  str z3,  [sp]
    b .L_mzr_phase2
.L_mzr_st4:  str z4,  [sp]
    b .L_mzr_phase2
.L_mzr_st5:  str z5,  [sp]
    b .L_mzr_phase2
.L_mzr_st6:  str z6,  [sp]
    b .L_mzr_phase2
.L_mzr_st7:  str z7,  [sp]
    b .L_mzr_phase2
.L_mzr_st8:  str z8,  [sp]
    b .L_mzr_phase2
.L_mzr_st9:  str z9,  [sp]
    b .L_mzr_phase2
.L_mzr_st10: str z10, [sp]
    b .L_mzr_phase2
.L_mzr_st11: str z11, [sp]
    b .L_mzr_phase2
.L_mzr_st12: str z12, [sp]
    b .L_mzr_phase2
.L_mzr_st13: str z13, [sp]
    b .L_mzr_phase2
.L_mzr_st14: str z14, [sp]
    b .L_mzr_phase2
.L_mzr_st15: str z15, [sp]
    b .L_mzr_phase2
.L_mzr_st16: str z16, [sp]
    b .L_mzr_phase2
.L_mzr_st17: str z17, [sp]
    b .L_mzr_phase2
.L_mzr_st18: str z18, [sp]
    b .L_mzr_phase2
.L_mzr_st19: str z19, [sp]
    b .L_mzr_phase2
.L_mzr_st20: str z20, [sp]
    b .L_mzr_phase2
.L_mzr_st21: str z21, [sp]
    b .L_mzr_phase2
.L_mzr_st22: str z22, [sp]
    b .L_mzr_phase2
.L_mzr_st23: str z23, [sp]
    b .L_mzr_phase2
.L_mzr_st24: str z24, [sp]
    b .L_mzr_phase2
.L_mzr_st25: str z25, [sp]
    b .L_mzr_phase2
.L_mzr_st26: str z26, [sp]
    b .L_mzr_phase2
.L_mzr_st27: str z27, [sp]
    b .L_mzr_phase2
.L_mzr_st28: str z28, [sp]
    b .L_mzr_phase2
.L_mzr_st29: str z29, [sp]
    b .L_mzr_phase2
.L_mzr_st30: str z30, [sp]
    b .L_mzr_phase2
.L_mzr_st31: str z31, [sp]
    b .L_mzr_phase2
// ── Load table: [sp] → z{N}, 8 bytes per entry (ldr + b) ──
.L_mzr_ld0:  ldr z0,  [sp]
    b .L_mzr_done
.L_mzr_ld1:  ldr z1,  [sp]
    b .L_mzr_done
.L_mzr_ld2:  ldr z2,  [sp]
    b .L_mzr_done
.L_mzr_ld3:  ldr z3,  [sp]
    b .L_mzr_done
.L_mzr_ld4:  ldr z4,  [sp]
    b .L_mzr_done
.L_mzr_ld5:  ldr z5,  [sp]
    b .L_mzr_done
.L_mzr_ld6:  ldr z6,  [sp]
    b .L_mzr_done
.L_mzr_ld7:  ldr z7,  [sp]
    b .L_mzr_done
.L_mzr_ld8:  ldr z8,  [sp]
    b .L_mzr_done
.L_mzr_ld9:  ldr z9,  [sp]
    b .L_mzr_done
.L_mzr_ld10: ldr z10, [sp]
    b .L_mzr_done
.L_mzr_ld11: ldr z11, [sp]
    b .L_mzr_done
.L_mzr_ld12: ldr z12, [sp]
    b .L_mzr_done
.L_mzr_ld13: ldr z13, [sp]
    b .L_mzr_done
.L_mzr_ld14: ldr z14, [sp]
    b .L_mzr_done
.L_mzr_ld15: ldr z15, [sp]
    b .L_mzr_done
.L_mzr_ld16: ldr z16, [sp]
    b .L_mzr_done
.L_mzr_ld17: ldr z17, [sp]
    b .L_mzr_done
.L_mzr_ld18: ldr z18, [sp]
    b .L_mzr_done
.L_mzr_ld19: ldr z19, [sp]
    b .L_mzr_done
.L_mzr_ld20: ldr z20, [sp]
    b .L_mzr_done
.L_mzr_ld21: ldr z21, [sp]
    b .L_mzr_done
.L_mzr_ld22: ldr z22, [sp]
    b .L_mzr_done
.L_mzr_ld23: ldr z23, [sp]
    b .L_mzr_done
.L_mzr_ld24: ldr z24, [sp]
    b .L_mzr_done
.L_mzr_ld25: ldr z25, [sp]
    b .L_mzr_done
.L_mzr_ld26: ldr z26, [sp]
    b .L_mzr_done
.L_mzr_ld27: ldr z27, [sp]
    b .L_mzr_done
.L_mzr_ld28: ldr z28, [sp]
    b .L_mzr_done
.L_mzr_ld29: ldr z29, [sp]
    b .L_mzr_done
.L_mzr_ld30: ldr z30, [sp]
    b .L_mzr_done
.L_mzr_ld31: ldr z31, [sp]
    b .L_mzr_done
// ================================================================
// LOOP_BEGIN (0x35) — Set loop counter register
// Encoding: [0x35][count:u8]
// ================================================================
.L_op_loop_begin:
    ldrb    w20, [x19], #1         // count → x20
    b       .L_dispatch
// ================================================================
// LOOP_END (0x36) — Decrement counter, jump back by offset
// Encoding: [0x36][offset:u16]
// offset = bytes to rewind x19 (from the byte past this arg)
// ================================================================
.L_op_loop_end:
    ldrh    w22, [x19]             // offset
    add     x19, x19, #2
    sub     w20, w20, #1
    cbz     w20, .L_dispatch
    sub     x19, x19, x22
    b       .L_dispatch
// ================================================================
// SET_PARAM (0x37) — Set a param table entry to a pointer value
// Encoding: [0x37][idx:u8][ptr:u64]
// idx = param slot index (0-7), ptr = 64-bit pointer value
// ================================================================
.L_op_set_param:
    ldrb    w9, [x19], #1          // idx (0-7)
    ldr     x8, [x19], #8          // ptr value
    add     x10, sp, #128          // param table base
    str     x8, [x10, x9, lsl #3] // table[idx] = ptr
    b       .L_dispatch
// ================================================================
// LOAD_PARAM (0x38) — Load one z-vector from param[idx] into z0
// Encoding: [0x38][idx:u8]
// ================================================================
.L_op_load_param:
    ldrb    w9, [x19], #1          // idx
    add     x10, sp, #128
    ldr     x8, [x10, x9, lsl #3] // x8 = param[idx] pointer
    ptrue   p0.b
    ld1b    {z0.b}, p0/z, [x8]
    b       .L_dispatch
// ================================================================
// STORE_PARAM (0x39) — Store z0 to the pointer in param[idx]
// Encoding: [0x39][idx:u8]
// ================================================================
.L_op_store_param:
    ldrb    w9, [x19], #1
    add     x10, sp, #128
    ldr     x8, [x10, x9, lsl #3]
    ptrue   p0.b
    st1b    {z0.b}, p0, [x8]
    b       .L_dispatch
// ================================================================
// ADVANCE_PARAM (0x3A) — Advance param[idx] pointer by one VL
// Encoding: [0x3A][idx:u8]
// ================================================================
.L_op_advance_param:
    ldrb    w9, [x19], #1
    add     x10, sp, #128
    ldr     x8, [x10, x9, lsl #3]
    addvl   x8, x8, #1            // advance by VL bytes
    str     x8, [x10, x9, lsl #3]
    b       .L_dispatch
// ================================================================
// Shared trampoline: store z{w9} to [sp], return via x26
// Each entry is 8 bytes: str z{N}, [sp] + br x26
// ================================================================
.L_tramp_store:
    adr     x8, .L_ts0
    add     x8, x8, x9, lsl #3
    br      x8
.L_ts0:  str z0,  [sp]
    br x26
.L_ts1:  str z1,  [sp]
    br x26
.L_ts2:  str z2,  [sp]
    br x26
.L_ts3:  str z3,  [sp]
    br x26
.L_ts4:  str z4,  [sp]
    br x26
.L_ts5:  str z5,  [sp]
    br x26
.L_ts6:  str z6,  [sp]
    br x26
.L_ts7:  str z7,  [sp]
    br x26
.L_ts8:  str z8,  [sp]
    br x26
.L_ts9:  str z9,  [sp]
    br x26
.L_ts10: str z10, [sp]
    br x26
.L_ts11: str z11, [sp]
    br x26
.L_ts12: str z12, [sp]
    br x26
.L_ts13: str z13, [sp]
    br x26
.L_ts14: str z14, [sp]
    br x26
.L_ts15: str z15, [sp]
    br x26
.L_ts16: str z16, [sp]
    br x26
.L_ts17: str z17, [sp]
    br x26
.L_ts18: str z18, [sp]
    br x26
.L_ts19: str z19, [sp]
    br x26
.L_ts20: str z20, [sp]
    br x26
.L_ts21: str z21, [sp]
    br x26
.L_ts22: str z22, [sp]
    br x26
.L_ts23: str z23, [sp]
    br x26
.L_ts24: str z24, [sp]
    br x26
.L_ts25: str z25, [sp]
    br x26
.L_ts26: str z26, [sp]
    br x26
.L_ts27: str z27, [sp]
    br x26
.L_ts28: str z28, [sp]
    br x26
.L_ts29: str z29, [sp]
    br x26
.L_ts30: str z30, [sp]
    br x26
.L_ts31: str z31, [sp]
    br x26
// ================================================================
// Shared trampoline: load [sp] to z{w9}, return via x26
// Each entry is 8 bytes: ldr z{N}, [sp] + br x26
// ================================================================
.L_tramp_load:
    adr     x8, .L_tl0
    add     x8, x8, x9, lsl #3
    br      x8
.L_tl0:  ldr z0,  [sp]
    br x26
.L_tl1:  ldr z1,  [sp]
    br x26
.L_tl2:  ldr z2,  [sp]
    br x26
.L_tl3:  ldr z3,  [sp]
    br x26
.L_tl4:  ldr z4,  [sp]
    br x26
.L_tl5:  ldr z5,  [sp]
    br x26
.L_tl6:  ldr z6,  [sp]
    br x26
.L_tl7:  ldr z7,  [sp]
    br x26
.L_tl8:  ldr z8,  [sp]
    br x26
.L_tl9:  ldr z9,  [sp]
    br x26
.L_tl10: ldr z10, [sp]
    br x26
.L_tl11: ldr z11, [sp]
    br x26
.L_tl12: ldr z12, [sp]
    br x26
.L_tl13: ldr z13, [sp]
    br x26
.L_tl14: ldr z14, [sp]
    br x26
.L_tl15: ldr z15, [sp]
    br x26
.L_tl16: ldr z16, [sp]
    br x26
.L_tl17: ldr z17, [sp]
    br x26
.L_tl18: ldr z18, [sp]
    br x26
.L_tl19: ldr z19, [sp]
    br x26
.L_tl20: ldr z20, [sp]
    br x26
.L_tl21: ldr z21, [sp]
    br x26
.L_tl22: ldr z22, [sp]
    br x26
.L_tl23: ldr z23, [sp]
    br x26
.L_tl24: ldr z24, [sp]
    br x26
.L_tl25: ldr z25, [sp]
    br x26
.L_tl26: ldr z26, [sp]
    br x26
.L_tl27: ldr z27, [sp]
    br x26
.L_tl28: ldr z28, [sp]
    br x26
.L_tl29: ldr z29, [sp]
    br x26
.L_tl30: ldr z30, [sp]
    br x26
.L_tl31: ldr z31, [sp]
    br x26
// ================================================================
// FADD_ZREG (0x3B) — z{dst}.s = z{src1}.s + z{src2}.s
// Encoding: [0x3B][dst:u8][src1:u8][src2:u8]
// ================================================================
.L_op_fadd_zreg:
    ldrb    w22, [x19], #1         // dst index
    ldrb    w9, [x19], #1          // src1 index
    ldrb    w23, [x19], #1         // src2 index
    ptrue   p0.s
    addvl   sp, sp, #-1            // scratch slot
    adr     x26, .L_fazr_1
    b       .L_tramp_store         // store z{src1} to [sp]
.L_fazr_1:
    ldr     z0, [sp]               // z0 = src1
    mov     w9, w23                // src2 index
    adr     x26, .L_fazr_2
    b       .L_tramp_store         // store z{src2} to [sp]
.L_fazr_2:
    ldr     z1, [sp]               // z1 = src2
    fadd    z0.s, z0.s, z1.s
    str     z0, [sp]               // result to stack
    mov     w9, w22                // dst index
    adr     x26, .L_fazr_3
    b       .L_tramp_load          // load [sp] to z{dst}
.L_fazr_3:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// FSUB_ZREG (0x3C) — z{dst}.s = z{src1}.s - z{src2}.s
// Encoding: [0x3C][dst:u8][src1:u8][src2:u8]
// ================================================================
.L_op_fsub_zreg:
    ldrb    w22, [x19], #1         // dst index
    ldrb    w9, [x19], #1          // src1 index
    ldrb    w23, [x19], #1         // src2 index
    ptrue   p0.s
    addvl   sp, sp, #-1            // scratch slot
    adr     x26, .L_fszr_1
    b       .L_tramp_store         // store z{src1} to [sp]
.L_fszr_1:
    ldr     z0, [sp]               // z0 = src1
    mov     w9, w23                // src2 index
    adr     x26, .L_fszr_2
    b       .L_tramp_store         // store z{src2} to [sp]
.L_fszr_2:
    ldr     z1, [sp]               // z1 = src2
    fsub    z0.s, z0.s, z1.s
    str     z0, [sp]               // result to stack
    mov     w9, w22                // dst index
    adr     x26, .L_fszr_3
    b       .L_tramp_load          // load [sp] to z{dst}
.L_fszr_3:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// FMUL_ZREG (0x3D) — z{dst}.s = z{src1}.s * z{src2}.s
// Encoding: [0x3D][dst:u8][src1:u8][src2:u8]
// ================================================================
.L_op_fmul_zreg:
    ldrb    w22, [x19], #1         // dst index
    ldrb    w9, [x19], #1          // src1 index
    ldrb    w23, [x19], #1         // src2 index
    ptrue   p0.s
    addvl   sp, sp, #-1            // scratch slot
    adr     x26, .L_fmzr_1
    b       .L_tramp_store         // store z{src1} to [sp]
.L_fmzr_1:
    ldr     z0, [sp]               // z0 = src1
    mov     w9, w23                // src2 index
    adr     x26, .L_fmzr_2
    b       .L_tramp_store         // store z{src2} to [sp]
.L_fmzr_2:
    ldr     z1, [sp]               // z1 = src2
    fmul    z0.s, z0.s, z1.s
    str     z0, [sp]               // result to stack
    mov     w9, w22                // dst index
    adr     x26, .L_fmzr_3
    b       .L_tramp_load          // load [sp] to z{dst}
.L_fmzr_3:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// FMLA_ZREG (0x3E) — z{dst}.s += z{src1}.s * z{src2}.s
// Encoding: [0x3E][dst:u8][src1:u8][src2:u8]
// dst is both accumulator input and output.
// Register plan: w22=dst, w23=src1, w24=src2, w9=trampoline index
// ================================================================
.L_op_fmla_zreg:
    ldrb    w22, [x19], #1         // dst
    ldrb    w23, [x19], #1         // src1
    ldrb    w24, [x19], #1         // src2
    ptrue   p0.s
    addvl   sp, sp, #-1
    mov     w9, w22                // trampoline: store z{dst}
    adr     x26, .L_fma_1
    b       .L_tramp_store
.L_fma_1:
    ldr     z0, [sp]               // z0 = accumulator (dst)
    mov     w9, w23                // trampoline: store z{src1}
    adr     x26, .L_fma_2
    b       .L_tramp_store
.L_fma_2:
    ldr     z1, [sp]               // z1 = src1
    mov     w9, w24                // trampoline: store z{src2}
    adr     x26, .L_fma_3
    b       .L_tramp_store
.L_fma_3:
    ldr     z2, [sp]               // z2 = src2
    fmla    z0.s, p0/m, z1.s, z2.s // z0 += z1 * z2
    str     z0, [sp]
    mov     w9, w22                // trampoline: load into z{dst}
    adr     x26, .L_fma_4
    b       .L_tramp_load
.L_fma_4:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// AND_ZREG (0x3F) — z{dst} = z{src1} AND z{src2}
// Encoding: [0x3F][dst:u8][src1:u8][src2:u8]
// ================================================================
.L_op_and_zreg:
    ldrb    w22, [x19], #1
    ldrb    w9, [x19], #1
    ldrb    w23, [x19], #1
    addvl   sp, sp, #-1
    adr     x26, .L_and_1
    b       .L_tramp_store
.L_and_1:
    ldr     z0, [sp]
    mov     w9, w23
    adr     x26, .L_and_2
    b       .L_tramp_store
.L_and_2:
    ldr     z1, [sp]
    and     z0.d, z0.d, z1.d
    str     z0, [sp]
    mov     w9, w22
    adr     x26, .L_and_3
    b       .L_tramp_load
.L_and_3:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// ORR_ZREG (0x40) — z{dst} = z{src1} OR z{src2}
// Encoding: [0x40][dst:u8][src1:u8][src2:u8]
// ================================================================
.L_op_orr_zreg:
    ldrb    w22, [x19], #1
    ldrb    w9, [x19], #1
    ldrb    w23, [x19], #1
    addvl   sp, sp, #-1
    adr     x26, .L_orr_1
    b       .L_tramp_store
.L_orr_1:
    ldr     z0, [sp]
    mov     w9, w23
    adr     x26, .L_orr_2
    b       .L_tramp_store
.L_orr_2:
    ldr     z1, [sp]
    orr     z0.d, z0.d, z1.d
    str     z0, [sp]
    mov     w9, w22
    adr     x26, .L_orr_3
    b       .L_tramp_load
.L_orr_3:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// EOR_ZREG (0x41) — z{dst} = z{src1} XOR z{src2}
// Encoding: [0x41][dst:u8][src1:u8][src2:u8]
// ================================================================
.L_op_eor_zreg:
    ldrb    w22, [x19], #1
    ldrb    w9, [x19], #1
    ldrb    w23, [x19], #1
    addvl   sp, sp, #-1
    adr     x26, .L_eor_1
    b       .L_tramp_store
.L_eor_1:
    ldr     z0, [sp]
    mov     w9, w23
    adr     x26, .L_eor_2
    b       .L_tramp_store
.L_eor_2:
    ldr     z1, [sp]
    eor     z0.d, z0.d, z1.d
    str     z0, [sp]
    mov     w9, w22
    adr     x26, .L_eor_3
    b       .L_tramp_load
.L_eor_3:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// NOT_ZREG (0x42) — z{dst} = NOT z{src}
// Encoding: [0x42][dst:u8][src:u8]
// ================================================================
.L_op_not_zreg:
    ldrb    w22, [x19], #1         // dst
    ldrb    w9, [x19], #1          // src
    ptrue   p0.b
    addvl   sp, sp, #-1
    adr     x26, .L_not_1
    b       .L_tramp_store
.L_not_1:
    ldr     z0, [sp]
    not     z0.b, p0/m, z0.b
    str     z0, [sp]
    mov     w9, w22
    adr     x26, .L_not_2
    b       .L_tramp_load
.L_not_2:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// LSL_ZREG (0x43) — z{dst} = z{src} << amount (per 64-bit lane)
// Encoding: [0x43][dst:u8][src:u8][amount:u8]
// ================================================================
.L_op_lsl_zreg:
    ldrb    w22, [x19], #1         // dst index
    ldrb    w9, [x19], #1          // src index
    ldrb    w23, [x19], #1         // shift amount
    addvl   sp, sp, #-1
    adr     x26, .L_lsl_1
    b       .L_tramp_store         // store z{src} to [sp]
.L_lsl_1:
    ldr     z0, [sp]               // z0 = src
    mov     z1.d, x23              // broadcast shift amount to all .d lanes
    ptrue   p0.d
    lsl     z0.d, p0/m, z0.d, z1.d
    str     z0, [sp]
    mov     w9, w22                // dst index
    adr     x26, .L_lsl_2
    b       .L_tramp_load          // load [sp] to z{dst}
.L_lsl_2:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// LSR_ZREG (0x44) — z{dst} = z{src} >> amount (logical, per 64-bit lane)
// Encoding: [0x44][dst:u8][src:u8][amount:u8]
// ================================================================
.L_op_lsr_zreg:
    ldrb    w22, [x19], #1         // dst index
    ldrb    w9, [x19], #1          // src index
    ldrb    w23, [x19], #1         // shift amount
    addvl   sp, sp, #-1
    adr     x26, .L_lsr_1
    b       .L_tramp_store         // store z{src} to [sp]
.L_lsr_1:
    ldr     z0, [sp]               // z0 = src
    mov     z1.d, x23              // broadcast shift amount to all .d lanes
    ptrue   p0.d
    lsr     z0.d, p0/m, z0.d, z1.d
    str     z0, [sp]
    mov     w9, w22                // dst index
    adr     x26, .L_lsr_2
    b       .L_tramp_load          // load [sp] to z{dst}
.L_lsr_2:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// ASR_ZREG (0x45) — z{dst} = z{src} >> amount (arithmetic, per 64-bit lane)
// Encoding: [0x45][dst:u8][src:u8][amount:u8]
// ================================================================
.L_op_asr_zreg:
    ldrb    w22, [x19], #1         // dst index
    ldrb    w9, [x19], #1          // src index
    ldrb    w23, [x19], #1         // shift amount
    addvl   sp, sp, #-1
    adr     x26, .L_asr_1
    b       .L_tramp_store         // store z{src} to [sp]
.L_asr_1:
    ldr     z0, [sp]               // z0 = src
    mov     z1.d, x23              // broadcast shift amount to all .d lanes
    ptrue   p0.d
    asr     z0.d, p0/m, z0.d, z1.d
    str     z0, [sp]
    mov     w9, w22                // dst index
    adr     x26, .L_asr_2
    b       .L_tramp_load          // load [sp] to z{dst}
.L_asr_2:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// LOAD_ZT0 (0x46) — Load ZT0 table register from pointer
// Encoding: [0x46][ptr:u64]
// ================================================================
.L_op_load_zt0:
    ldr     x8, [x19], #8
    ldr     zt0, [x8]
    b       .L_dispatch
// ================================================================
// LUTI2_ZREG (0x47) — z{dst}.b = luti2(zt0, z{src}[0])
// Encoding: [0x47][dst:u8][src:u8]
// ================================================================
.L_op_luti2_zreg:
    ldrb    w22, [x19], #1         // dst index
    ldrb    w9, [x19], #1          // src index
    addvl   sp, sp, #-1
    adr     x26, .L_luti2_1
    b       .L_tramp_store         // store z{src} to [sp]
.L_luti2_1:
    ldr     z0, [sp]               // z0 = src (index vector)
    luti2   z1.b, zt0, z0[0]
    str     z1, [sp]
    mov     w9, w22                // dst index
    adr     x26, .L_luti2_2
    b       .L_tramp_load          // load [sp] to z{dst}
.L_luti2_2:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// LUTI4_ZREG (0x48) — z{dst}.b = luti4(zt0, z{src}[0])
// Encoding: [0x48][dst:u8][src:u8]
// ================================================================
.L_op_luti4_zreg:
    ldrb    w22, [x19], #1         // dst index
    ldrb    w9, [x19], #1          // src index
    addvl   sp, sp, #-1
    adr     x26, .L_luti4_1
    b       .L_tramp_store         // store z{src} to [sp]
.L_luti4_1:
    ldr     z0, [sp]               // z0 = src (index vector)
    luti4   z1.b, zt0, z0[0]
    str     z1, [sp]
    mov     w9, w22                // dst index
    adr     x26, .L_luti4_2
    b       .L_tramp_load          // load [sp] to z{dst}
.L_luti4_2:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// SMOPA_ZREG (0x49) — za{tile}.s += z{src1}.b outer_product z{src2}.b (signed)
// Encoding: [0x49][tile:u8][src1:u8][src2:u8]
// ================================================================
.L_op_smopa_zreg:
    ldrb    w22, [x19], #1         // tile index (0-3)
    ldrb    w9, [x19], #1          // src1 index
    ldrb    w23, [x19], #1         // src2 index
    ptrue   p0.b
    addvl   sp, sp, #-1
    adr     x26, .L_smzr_1
    b       .L_tramp_store         // store z{src1} to [sp]
.L_smzr_1:
    ldr     z0, [sp]               // z0 = src1
    mov     w9, w23                // src2 index
    adr     x26, .L_smzr_2
    b       .L_tramp_store         // store z{src2} to [sp]
.L_smzr_2:
    ldr     z1, [sp]               // z1 = src2
    addvl   sp, sp, #1
    cbz     w22, .L_smopa_t0
    cmp     w22, #1
    b.eq    .L_smopa_t1
    cmp     w22, #2
    b.eq    .L_smopa_t2
    smopa   za3.s, p0/m, p0/m, z0.b, z1.b
    b       .L_dispatch
.L_smopa_t0:
    smopa   za0.s, p0/m, p0/m, z0.b, z1.b
    b       .L_dispatch
.L_smopa_t1:
    smopa   za1.s, p0/m, p0/m, z0.b, z1.b
    b       .L_dispatch
.L_smopa_t2:
    smopa   za2.s, p0/m, p0/m, z0.b, z1.b
    b       .L_dispatch
// ================================================================
// UMOPA_ZREG (0x4A) — za{tile}.s += z{src1}.b outer_product z{src2}.b (unsigned)
// Encoding: [0x4A][tile:u8][src1:u8][src2:u8]
// ================================================================
.L_op_umopa_zreg:
    ldrb    w22, [x19], #1         // tile index (0-3)
    ldrb    w9, [x19], #1          // src1 index
    ldrb    w23, [x19], #1         // src2 index
    ptrue   p0.b
    addvl   sp, sp, #-1
    adr     x26, .L_umzr_1
    b       .L_tramp_store         // store z{src1} to [sp]
.L_umzr_1:
    ldr     z0, [sp]               // z0 = src1
    mov     w9, w23                // src2 index
    adr     x26, .L_umzr_2
    b       .L_tramp_store         // store z{src2} to [sp]
.L_umzr_2:
    ldr     z1, [sp]               // z1 = src2
    addvl   sp, sp, #1
    cbz     w22, .L_umopa_t0
    cmp     w22, #1
    b.eq    .L_umopa_t1
    cmp     w22, #2
    b.eq    .L_umopa_t2
    umopa   za3.s, p0/m, p0/m, z0.b, z1.b
    b       .L_dispatch
.L_umopa_t0:
    umopa   za0.s, p0/m, p0/m, z0.b, z1.b
    b       .L_dispatch
.L_umopa_t1:
    umopa   za1.s, p0/m, p0/m, z0.b, z1.b
    b       .L_dispatch
.L_umopa_t2:
    umopa   za2.s, p0/m, p0/m, z0.b, z1.b
    b       .L_dispatch
// ================================================================
// USMOPA_ZREG (0x4B) — za{tile}.s += z{src1}.b outer_product z{src2}.b (unsigned x signed)
// Encoding: [0x4B][tile:u8][src1:u8][src2:u8]
// ================================================================
.L_op_usmopa_zreg:
    ldrb    w22, [x19], #1         // tile index (0-3)
    ldrb    w9, [x19], #1          // src1 index
    ldrb    w23, [x19], #1         // src2 index
    ptrue   p0.b
    addvl   sp, sp, #-1
    adr     x26, .L_uszr_1
    b       .L_tramp_store         // store z{src1} to [sp]
.L_uszr_1:
    ldr     z0, [sp]               // z0 = src1
    mov     w9, w23                // src2 index
    adr     x26, .L_uszr_2
    b       .L_tramp_store         // store z{src2} to [sp]
.L_uszr_2:
    ldr     z1, [sp]               // z1 = src2
    addvl   sp, sp, #1
    cbz     w22, .L_usmopa_t0
    cmp     w22, #1
    b.eq    .L_usmopa_t1
    cmp     w22, #2
    b.eq    .L_usmopa_t2
    usmopa  za3.s, p0/m, p0/m, z0.b, z1.b
    b       .L_dispatch
.L_usmopa_t0:
    usmopa  za0.s, p0/m, p0/m, z0.b, z1.b
    b       .L_dispatch
.L_usmopa_t1:
    usmopa  za1.s, p0/m, p0/m, z0.b, z1.b
    b       .L_dispatch
.L_usmopa_t2:
    usmopa  za2.s, p0/m, p0/m, z0.b, z1.b
    b       .L_dispatch
// ================================================================
// FMOPA_ZREG (0x4C) — za{tile}.s += z{src1}.s outer_product z{src2}.s (float32)
// Encoding: [0x4C][tile:u8][src1:u8][src2:u8]
// ================================================================
.L_op_fmopa_zreg:
    ldrb    w22, [x19], #1         // tile index (0-3)
    ldrb    w9, [x19], #1          // src1 index
    ldrb    w23, [x19], #1         // src2 index
    ptrue   p0.s
    addvl   sp, sp, #-1
    adr     x26, .L_fozr_1
    b       .L_tramp_store         // store z{src1} to [sp]
.L_fozr_1:
    ldr     z0, [sp]               // z0 = src1
    mov     w9, w23                // src2 index
    adr     x26, .L_fozr_2
    b       .L_tramp_store         // store z{src2} to [sp]
.L_fozr_2:
    ldr     z1, [sp]               // z1 = src2
    addvl   sp, sp, #1
    cbz     w22, .L_fmopa_t0
    cmp     w22, #1
    b.eq    .L_fmopa_t1
    cmp     w22, #2
    b.eq    .L_fmopa_t2
    fmopa   za3.s, p0/m, p0/m, z0.s, z1.s
    b       .L_dispatch
.L_fmopa_t0:
    fmopa   za0.s, p0/m, p0/m, z0.s, z1.s
    b       .L_dispatch
.L_fmopa_t1:
    fmopa   za1.s, p0/m, p0/m, z0.s, z1.s
    b       .L_dispatch
.L_fmopa_t2:
    fmopa   za2.s, p0/m, p0/m, z0.s, z1.s
    b       .L_dispatch
// ================================================================
// CBLAS_SGEMM (0x4D) -- C = alpha*op(A)*op(B) + beta*C
//
// Encoding: [0x4D][flags:u8][M:u32][N:u32][K:u32]
//           [lda:u32][ldb:u32][ldc:u32]
//           [alpha:f32][beta:f32]
//           [A_ptr:u64][B_ptr:u64][C_ptr:u64]
//
// flags bit 0: transA (0=normal, 1=transpose)
// flags bit 1: transB (0=normal, 1=transpose)
// Total immediate payload after opcode: 57 bytes
//
// Tile geometry: 16 x 32 output (za0 = left 16 cols, za1 = right 16 cols)
// za2 = scratch for A transpose, za3 = scratch for transB
//
// Stack layout (128 bytes):
//   [sp+0]:   A_ptr           [sp+8]:   B_ptr
//   [sp+16]:  C_ptr           [sp+24]:  M (w)     [sp+28]: N (w)
//   [sp+32]:  K (w)           [sp+36]:  lda (w)   [sp+40]: ldb (w)
//   [sp+44]:  ldc (w)         [sp+48]:  flags (w)
//   [sp+52]:  k_blocks (w)    [sp+56]:  M_pad (w) [sp+60]: N_pad (w)
//   [sp+64]:  A_tile_base (x) [sp+72]:  lda*4 (x)
//   [sp+80]:  ti (w)          [sp+84]:  tj (w)
//   [sp+88]:  ldb*4 (x)       [sp+96]:  ldc*4 (x)
//   [sp+104]: beta_bits (w)
// z20 = alpha broadcast, z22 = beta broadcast
// ================================================================
.L_op_cblas_sgemm:
    // ── Parse bytecodes ──
    ldrb    w18, [x19], #1             // flags
    ldr     w0, [x19]                  // M
    ldr     w1, [x19, #4]             // N
    ldr     w2, [x19, #8]             // K
    ldr     w3, [x19, #12]            // lda
    ldr     w4, [x19, #16]            // ldb
    ldr     w5, [x19, #20]            // ldc
    ldr     s20, [x19, #24]           // alpha (f32)
    ldr     s22, [x19, #28]           // beta (f32)
    add     x19, x19, #32
    ldr     x6, [x19], #8             // A_ptr
    ldr     x7, [x19], #8             // B_ptr
    ldr     x8, [x19], #8             // C_ptr
    // ── Allocate stack frame ──
    sub     sp, sp, #128
    stp     x6, x7, [sp, #0]          // [0] A, [8] B
    str     x8, [sp, #16]             // [16] C
    stp     w0, w1, [sp, #24]         // [24] M, [28] N
    str     w2, [sp, #32]             // [32] K
    stp     w3, w4, [sp, #36]         // [36] lda, [40] ldb
    stp     w5, w18, [sp, #44]        // [44] ldc, [48] flags
    // ── Derived values ──
    ptrue   p0.s
    cntw    x9                         // SVLs = 16
    lsr     w15, w2, #4               // k_blocks = K / 16
    str     w15, [sp, #52]
    add     w10, w0, #15
    and     w10, w10, #0xFFFFFFF0      // M_pad = (M+15) & ~15
    str     w10, [sp, #56]
    add     w11, w1, #31
    and     w11, w11, #0xFFFFFFE0      // N_pad = (N+31) & ~31
    str     w11, [sp, #60]
    lsl     x13, x3, #2               // lda * 4
    str     x13, [sp, #72]
    lsl     x14, x4, #2               // ldb * 4
    str     x14, [sp, #88]
    lsl     x16, x5, #2               // ldc * 4
    str     x16, [sp, #96]
    // ── Broadcast alpha/beta, save beta bits ──
    mov     z20.s, s20
    mov     z22.s, s22
    fmov    w17, s22
    str     w17, [sp, #104]            // beta_bits for later zero-check
    // ── Tile row loop ──
    mov     w0, #0
.L_sg_tile_row:
    str     w0, [sp, #80]
    mov     w1, #0
.L_sg_tile_col:
    str     w1, [sp, #84]
    // ── Phase 1: init accumulators (beta*C or zero) ──
    ldr     w17, [sp, #104]
    cbnz    w17, .L_sg_load_beta
    zero    {za0.s, za1.s}
    b       .L_sg_beta_done
.L_sg_load_beta:
    zero    {za0.s, za1.s}
    ldr     x8, [sp, #16]             // C
    ldr     w0, [sp, #80]             // ti
    ldr     w1, [sp, #84]             // tj
    ldr     w14, [sp, #28]            // N
    ldr     w5, [sp, #44]             // ldc
    ldr     x16, [sp, #96]            // ldc*4
    ldr     w6, [sp, #24]             // M
    cntw    x9
    ptrue   p0.s
    mul     w10, w0, w5
    add     w10, w10, w1
    add     x8, x8, x10, lsl #2       // &C[ti][tj]
    // Column predicates
    sub     w3, w14, w1
    mov     w4, #32
    cmp     w3, w4
    csel    w3, w3, w4, lt
    whilelt p2.s, xzr, x3
    sub     w4, w3, #16
    cmp     w4, #0
    csel    w4, wzr, w4, lt
    whilelt p3.s, xzr, x4
    // Row count
    sub     w15, w6, w0
    mov     w3, #16
    cmp     w15, w3
    csel    w15, w15, w3, lt
    // Load C rows into za0/za1 scaled by beta, 4 at a time
    mov     w12, #0
.L_sg_beta_grp:
    cmp     w12, w15
    b.ge    .L_sg_beta_done
    // Load up to 4 rows, zero-fill any that are past valid row count
    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0
    // Row 0
    ld1w    {z0.s}, p2/z, [x8]
    ld1w    {z4.s}, p3/z, [x8, x9, lsl #2]
    fmul    z0.s, p0/m, z0.s, z22.s
    fmul    z4.s, p0/m, z4.s, z22.s
    add     w11, w12, #1
    cmp     w11, w15
    b.ge    .L_sg_beta_st
    add     x8, x8, x16
    // Row 1
    ld1w    {z1.s}, p2/z, [x8]
    ld1w    {z5.s}, p3/z, [x8, x9, lsl #2]
    fmul    z1.s, p0/m, z1.s, z22.s
    fmul    z5.s, p0/m, z5.s, z22.s
    add     w11, w12, #2
    cmp     w11, w15
    b.ge    .L_sg_beta_st
    add     x8, x8, x16
    // Row 2
    ld1w    {z2.s}, p2/z, [x8]
    ld1w    {z6.s}, p3/z, [x8, x9, lsl #2]
    fmul    z2.s, p0/m, z2.s, z22.s
    fmul    z6.s, p0/m, z6.s, z22.s
    add     w11, w12, #3
    cmp     w11, w15
    b.ge    .L_sg_beta_st
    add     x8, x8, x16
    // Row 3
    ld1w    {z3.s}, p2/z, [x8]
    ld1w    {z7.s}, p3/z, [x8, x9, lsl #2]
    fmul    z3.s, p0/m, z3.s, z22.s
    fmul    z7.s, p0/m, z7.s, z22.s
    add     x8, x8, x16
.L_sg_beta_st:
    mova    za0h.s[w12, 0:3], {z0.s-z3.s}
    mova    za1h.s[w12, 0:3], {z4.s-z7.s}
    add     w12, w12, #4
    b       .L_sg_beta_grp
.L_sg_beta_done:
    // ── Phase 2: K-block accumulation ──
    // Compute A_tile_base for this tile row
    ldr     w0, [sp, #80]             // ti
    ldr     w1, [sp, #84]             // tj
    ldr     x6, [sp, #0]              // A
    ldr     x7, [sp, #8]              // B
    ldr     w2, [sp, #32]             // K
    ldr     w18, [sp, #48]            // flags
    ldr     x17, [sp, #72]            // lda*4
    ldr     w3, [sp, #36]             // lda
    ldr     w4, [sp, #40]             // ldb
    ldr     x14, [sp, #88]            // ldb*4
    ldr     w15, [sp, #52]            // k_blocks
    ptrue   p0.s
    cntw    x9
    // transA=0: A_tile_base = A + ti * lda * 4
    // transA=1: A_tile_base = A + ti * 4  (column offset in physical A)
    tst     w18, #1
    b.ne    .L_sg_atbase_trans
    mul     w10, w0, w3
    add     x5, x6, x10, lsl #2       // A + ti*lda*4
    b       .L_sg_atbase_done
.L_sg_atbase_trans:
    add     x5, x6, x0, lsl #2        // A + ti*4
.L_sg_atbase_done:
    str     x5, [sp, #64]             // save A_tile_base
    mov     x13, xzr                   // k byte offset
    cbz     w15, .L_sg_kblock_done
.L_sg_kblock:
    // ── Load A tile (16 rows x 16 cols) into za2, then transpose via vertical extract ──
    zero    {za2.s}
    ldr     w18, [sp, #48]            // flags
    tst     w18, #1
    b.ne    .L_sg_load_a_trans
    // transA=0: load A[ti+r][k..k+15] for r in 0..15
    // x5 = A_tile_base (A + ti*lda*4), x13 = k byte offset
    // row r: [x5 + r*lda*4 + x13]
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
    b       .L_sg_a_loaded
.L_sg_load_a_trans:
    // transA=1: physical A is K x M. logical A^T[m][k] = physical A[k][m]
    // Load A[k+r][ti..ti+15] for r in 0..15
    // x5 = A + ti*4, each row offset = (k+r) * lda * 4
    lsr     x10, x13, #2              // k element index
    mul     x11, x10, x17             // k * lda * 4
    add     x8, x5, x11               // A + ti*4 + k*lda*4 = &A[k][ti]
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
    // za2 row r = A[k+r][ti..ti+15]
    // za2v col c = A[k..k+15][ti+c] = A^T[ti+c][k..k+15]
    // Vertical extract gives the same layout as the non-transpose path
.L_sg_a_loaded:
    // ── Load B rows and FMOPA ──
    // For each of 16 k-steps, extract A column from za2v, load B row, fmopa
    // transB=0: B[k+i][tj..tj+31] => contiguous, stride = ldb*4
    // transB=1: B^T[k+i][tj..tj+31] = B[tj..tj+31][k+i] => strided access
    ldr     w18, [sp, #48]
    tst     w18, #2
    b.ne    .L_sg_fmopa_transB
    // ── transB=0: B row (k+i) starts at B + (k+i)*ldb*4 + tj*4 ──
    lsr     x10, x13, #2              // k element index
    mul     x11, x10, x14             // k * ldb * 4
    add     x11, x7, x11              // B + k*ldb*4
    ldr     w1, [sp, #84]
    add     x11, x11, x1, lsl #2      // + tj*4
    mov     x3, x14                    // B row stride = ldb*4
    // Cols 0-3
    mov     w12, #0
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z0.s, z5.s
    add     x11, x11, x3
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
    add     x11, x11, x3
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z2.s, z5.s
    add     x11, x11, x3
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z3.s, z5.s
    add     x11, x11, x3
    // Cols 4-7
    mov     w12, #4
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z0.s, z5.s
    add     x11, x11, x3
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
    add     x11, x11, x3
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z2.s, z5.s
    add     x11, x11, x3
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z3.s, z5.s
    add     x11, x11, x3
    // Cols 8-11
    mov     w12, #8
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z0.s, z5.s
    add     x11, x11, x3
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
    add     x11, x11, x3
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z2.s, z5.s
    add     x11, x11, x3
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z3.s, z5.s
    add     x11, x11, x3
    // Cols 12-15
    mov     w12, #12
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z0.s, z5.s
    add     x11, x11, x3
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
    add     x11, x11, x3
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z2.s, z5.s
    add     x11, x11, x3
    ld1w    {z4.s}, p0/z, [x11]
    ld1w    {z5.s}, p0/z, [x11, x9, lsl #2]
    fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z3.s, z5.s
    b       .L_sg_kblock_advance
    // ── transB=1: B is N x K. B^T[k+i][j] = B[j][k+i] ──
    // For each k-step i, we need B^T row (k+i) = column (k+i) of physical B
    // B[j][k+i] for j in [tj..tj+31] is strided: stride = ldb*4 between j values
    // Strategy: load B[tj..tj+15][k..k+15] into za3, transpose via vertical extract
    // Then load B[tj+16..tj+31][k..k+15] into za3 for the right half
.L_sg_fmopa_transB:
    // ── Left half: B[tj..tj+15][k..k+15] into za3 ──
    zero    {za3.s}
    lsr     x10, x13, #2              // k element index
    ldr     w1, [sp, #84]             // tj
    // &B[tj][k] = B + tj*ldb*4 + k*4
    mul     x11, x1, x14              // tj * ldb * 4 (x1 zero-ext, x14 = ldb*4)
    add     x11, x7, x11              // B + tj*ldb*4
    add     x11, x11, x10, lsl #2     // + k*4
    mov     x3, x14                    // row stride = ldb*4
    mov     w12, #0
    ld1w    {z0.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z1.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z2.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z3.s}, p0/z, [x11]
    add     x11, x11, x3
    mova    za3h.s[w12, 0:3], {z0.s-z3.s}
    mov     w12, #4
    ld1w    {z0.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z1.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z2.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z3.s}, p0/z, [x11]
    add     x11, x11, x3
    mova    za3h.s[w12, 0:3], {z0.s-z3.s}
    mov     w12, #8
    ld1w    {z0.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z1.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z2.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z3.s}, p0/z, [x11]
    add     x11, x11, x3
    mova    za3h.s[w12, 0:3], {z0.s-z3.s}
    mov     w12, #12
    ld1w    {z0.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z1.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z2.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z3.s}, p0/z, [x11]
    mova    za3h.s[w12, 0:3], {z0.s-z3.s}
    // za3v col i = B[tj..tj+15][k+i] = B^T[k+i][tj..tj+15] (left half of B^T row)
    // FMOPA left half into za0
    mov     w12, #0
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za0.s, p0/m, p0/m, z1.s, z5.s
    fmopa   za0.s, p0/m, p0/m, z2.s, z6.s
    fmopa   za0.s, p0/m, p0/m, z3.s, z7.s
    mov     w12, #4
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za0.s, p0/m, p0/m, z1.s, z5.s
    fmopa   za0.s, p0/m, p0/m, z2.s, z6.s
    fmopa   za0.s, p0/m, p0/m, z3.s, z7.s
    mov     w12, #8
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za0.s, p0/m, p0/m, z1.s, z5.s
    fmopa   za0.s, p0/m, p0/m, z2.s, z6.s
    fmopa   za0.s, p0/m, p0/m, z3.s, z7.s
    mov     w12, #12
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za0.s, p0/m, p0/m, z1.s, z5.s
    fmopa   za0.s, p0/m, p0/m, z2.s, z6.s
    fmopa   za0.s, p0/m, p0/m, z3.s, z7.s
    // ── Right half: B[tj+16..tj+31][k..k+15] into za3 ──
    zero    {za3.s}
    ldr     w1, [sp, #84]             // tj
    add     w11, w1, #16              // tj+16
    mul     x11, x11, x14             // (tj+16) * ldb * 4
    add     x11, x7, x11              // B + (tj+16)*ldb*4
    add     x11, x11, x10, lsl #2     // + k*4
    mov     x3, x14
    mov     w12, #0
    ld1w    {z0.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z1.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z2.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z3.s}, p0/z, [x11]
    add     x11, x11, x3
    mova    za3h.s[w12, 0:3], {z0.s-z3.s}
    mov     w12, #4
    ld1w    {z0.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z1.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z2.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z3.s}, p0/z, [x11]
    add     x11, x11, x3
    mova    za3h.s[w12, 0:3], {z0.s-z3.s}
    mov     w12, #8
    ld1w    {z0.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z1.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z2.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z3.s}, p0/z, [x11]
    add     x11, x11, x3
    mova    za3h.s[w12, 0:3], {z0.s-z3.s}
    mov     w12, #12
    ld1w    {z0.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z1.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z2.s}, p0/z, [x11]
    add     x11, x11, x3
    ld1w    {z3.s}, p0/z, [x11]
    mova    za3h.s[w12, 0:3], {z0.s-z3.s}
    // FMOPA right half into za1
    mov     w12, #0
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    fmopa   za1.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
    fmopa   za1.s, p0/m, p0/m, z2.s, z6.s
    fmopa   za1.s, p0/m, p0/m, z3.s, z7.s
    mov     w12, #4
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    fmopa   za1.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
    fmopa   za1.s, p0/m, p0/m, z2.s, z6.s
    fmopa   za1.s, p0/m, p0/m, z3.s, z7.s
    mov     w12, #8
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    fmopa   za1.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
    fmopa   za1.s, p0/m, p0/m, z2.s, z6.s
    fmopa   za1.s, p0/m, p0/m, z3.s, z7.s
    mov     w12, #12
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    fmopa   za1.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
    fmopa   za1.s, p0/m, p0/m, z2.s, z6.s
    fmopa   za1.s, p0/m, p0/m, z3.s, z7.s
.L_sg_kblock_advance:
    add     x13, x13, #64             // k byte offset += 16 floats * 4
    subs    w15, w15, #1
    b.ne    .L_sg_kblock
.L_sg_kblock_done:
    // ── Phase 3: Store alpha * ZA to C ──
    ldr     x8, [sp, #16]             // C
    ldr     w0, [sp, #80]             // ti
    ldr     w1, [sp, #84]             // tj
    ldr     w14, [sp, #28]            // N
    ldr     w5, [sp, #44]             // ldc
    ldr     x10, [sp, #96]            // ldc*4
    ldr     w6, [sp, #24]             // M
    ptrue   p0.s
    cntw    x9
    mul     w11, w0, w5
    add     w11, w11, w1
    add     x8, x8, x11, lsl #2       // C + (ti*ldc + tj)*4
    // Column predicates
    sub     w3, w14, w1
    mov     w4, #32
    cmp     w3, w4
    csel    w3, w3, w4, lt
    whilelt p2.s, xzr, x3
    sub     w4, w3, #16
    cmp     w4, #0
    csel    w4, wzr, w4, lt
    whilelt p3.s, xzr, x4
    // Valid rows
    sub     w15, w6, w0
    mov     w3, #16
    cmp     w15, w3
    csel    w15, w15, w3, lt
    // Group 0 (rows 0-3)
    mov     w12, #0
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
    cmp     w15, #1
    b.lt    .L_sg_store_end
    st1w    {z0.s}, p2, [x8]
    st1w    {z4.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #2
    b.lt    .L_sg_store_end
    st1w    {z1.s}, p2, [x8]
    st1w    {z5.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #3
    b.lt    .L_sg_store_end
    st1w    {z2.s}, p2, [x8]
    st1w    {z6.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #4
    b.lt    .L_sg_store_end
    st1w    {z3.s}, p2, [x8]
    st1w    {z7.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    // Group 1 (rows 4-7)
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
    cmp     w15, #5
    b.lt    .L_sg_store_end
    st1w    {z0.s}, p2, [x8]
    st1w    {z4.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #6
    b.lt    .L_sg_store_end
    st1w    {z1.s}, p2, [x8]
    st1w    {z5.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #7
    b.lt    .L_sg_store_end
    st1w    {z2.s}, p2, [x8]
    st1w    {z6.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #8
    b.lt    .L_sg_store_end
    st1w    {z3.s}, p2, [x8]
    st1w    {z7.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    // Group 2 (rows 8-11)
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
    cmp     w15, #9
    b.lt    .L_sg_store_end
    st1w    {z0.s}, p2, [x8]
    st1w    {z4.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #10
    b.lt    .L_sg_store_end
    st1w    {z1.s}, p2, [x8]
    st1w    {z5.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #11
    b.lt    .L_sg_store_end
    st1w    {z2.s}, p2, [x8]
    st1w    {z6.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #12
    b.lt    .L_sg_store_end
    st1w    {z3.s}, p2, [x8]
    st1w    {z7.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    // Group 3 (rows 12-15)
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
    cmp     w15, #13
    b.lt    .L_sg_store_end
    st1w    {z0.s}, p2, [x8]
    st1w    {z4.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #14
    b.lt    .L_sg_store_end
    st1w    {z1.s}, p2, [x8]
    st1w    {z5.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #15
    b.lt    .L_sg_store_end
    st1w    {z2.s}, p2, [x8]
    st1w    {z6.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #16
    b.lt    .L_sg_store_end
    st1w    {z3.s}, p2, [x8]
    st1w    {z7.s}, p3, [x8, x9, lsl #2]
.L_sg_store_end:
    // ── Advance tile column ──
    ldr     w1, [sp, #84]
    ldr     w4, [sp, #60]             // N_pad
    add     w1, w1, #32
    cmp     w1, w4
    b.lt    .L_sg_tile_col
    // ── Advance tile row ──
    ldr     w0, [sp, #80]
    ldr     w3, [sp, #56]             // M_pad
    add     w0, w0, #16
    cmp     w0, w3
    b.lt    .L_sg_tile_row
    // ── Cleanup ──
    add     sp, sp, #128
    b       .L_dispatch
// ================================================================
// FCLAMP_ZREG (0x4E) — Clamp/max/min with scalar bounds
// Encoding: [0x4E][flags:u8][dst:u8][src:u8][lo:f32][hi:f32]
// flags bit 0: apply lower bound (max with lo)
// flags bit 1: apply upper bound (min with hi)
// flags=0x01: max, flags=0x02: min, flags=0x03: clamp
// ================================================================
.L_op_fclamp_zreg:
    ldrb    w22, [x19], #1         // flags
    ldrb    w23, [x19], #1         // dst
    ldrb    w9, [x19], #1          // src
    ldr     s16, [x19]             // lo (f32)
    ldr     s17, [x19, #4]         // hi (f32)
    add     x19, x19, #8           // advance past both f32 values
    ptrue   p0.s
    addvl   sp, sp, #-1
    // Load src into z0
    adr     x26, .L_fclamp_1
    b       .L_tramp_store
.L_fclamp_1:
    ldr     z0, [sp]
    // Apply lower bound if flags bit 0
    tbz     w22, #0, .L_fclamp_skip_lo
    mov     z1.s, s16              // broadcast lo
    fmaxnm  z0.s, p0/m, z0.s, z1.s
.L_fclamp_skip_lo:
    // Apply upper bound if flags bit 1
    tbz     w22, #1, .L_fclamp_skip_hi
    mov     z1.s, s17              // broadcast hi
    fminnm  z0.s, p0/m, z0.s, z1.s
.L_fclamp_skip_hi:
    // Store result to dst
    str     z0, [sp]
    mov     w9, w23
    adr     x26, .L_fclamp_done
    b       .L_tramp_load
.L_fclamp_done:
    addvl   sp, sp, #1
    b       .L_dispatch
