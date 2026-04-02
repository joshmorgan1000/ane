//----------------------------------------------------------------------------------------------------------- Bytecode Interpreter
// @file   bytecode_interpreter.s
// @brief  Handwritten AArch64 assembly for the bytecode interpreter's main dispatch loop and opcode handlers.
//         This keeps the stream mode active for up to 255 loops.
//
// Copyright (c) 2024 Josh Morgan. All rights reserved.
// Released under the MIT License
//
.text
.p2align 4
.global _stream_exec
_stream_exec:
    stp     x29, x30, [sp, #-192]!   // prologue: save callee-saved registers and make room on stack
    mov     x29, sp                  // set frame pointer (not strictly needed, but can help with debugging)
    stp     x19, x20, [sp, #16]      // save x19-x26 and d8-d13, which we use for general-purpose storage across opcode handlers
    stp     x21, x22, [sp, #32]      // These registers are not preserved across opcode handlers, but we save/restore them around the entire dispatch loop to avoid clobbering the caller's values.
    stp     x23, x24, [sp, #48]      // (x19-x26 are callee-saved, so we must preserve them across the entire call to _stream_exec)
    stp     x25, x26, [sp, #64]      // d8-d13 are also callee-saved, so we save/restore them around the entire dispatch loop as well
    stp     d8,  d9,  [sp, #80]      // (if opcode handlers want to use these registers, they can spill them to the stack or use other callee-saved registers as scratch)
    stp     d10, d11, [sp, #96]     
    stp     d12, d13, [sp, #112]   
    // ── x0 = data, x1 = size ──
    mov     x19, x0                // instruction pointer
    add     x21, x0, x1            // end = data + size
    // ── Enter streaming mode ──
    smstart
    // ── Load jump table base (must be after smstart, adrp is fine in streaming) ──
    adrp    x25, .L_jump_table@PAGE             // load page address of jump table
    add     x25, x25, .L_jump_table@PAGEOFF     // add page offset to get full address of jump table
    // ── Dispatch ──
.L_dispatch:
    cmp     x19, x21                 // IP >= bytecodes end?
    b.hs    .L_exit                  // done — no explicit halt needed
    ldrb    w9, [x19], #1            // fetch opcode, advance IP
    ldrsw   x10, [x25, x9, lsl #2]   // load relative offset (32-bit signed)
    add     x10, x25, x10            // absolute target = table_base + offset
    br      x10                      // jump to opcode handler 
// ================================================================
// Jump Table (PC-relative offsets, avoids text relocations on macOS)
// ================================================================
.p2align 2
.L_jump_table:
    .long   .L_exit          - .L_jump_table   // 0x00 reserved (exit)
    .long   .L_op_zero_za    - .L_jump_table   // 0x01 zero_za
    .long   .L_op_acc_smopa  - .L_jump_table   // 0x02 acc_smopa
    .long   .L_op_acc_umopa  - .L_jump_table   // 0x03 acc_umopa
    .long   .L_op_acc_usmopa - .L_jump_table   // 0x04 acc_usmopa
    .long   .L_op_acc_sumopa - .L_jump_table   // 0x05 acc_sumopa
    .long   .L_op_store_tiles - .L_jump_table  // 0x06 store_tiles
    .long   .L_op_smopa_2x2  - .L_jump_table   // 0x07 smopa_2x2
    .long   .L_op_umopa_2x2  - .L_jump_table   // 0x08 umopa_2x2
    .long   .L_op_usmopa_2x2 - .L_jump_table   // 0x09 usmopa_2x2
    .long   .L_op_load_bias  - .L_jump_table   // 0x0A load_bias
    .long   .L_op_scale_store - .L_jump_table  // 0x0B scale_store
    .long   .L_op_elementwise_add_fp32  - .L_jump_table       // 0x0C elementwise_add_fp32
    .long   .L_op_elementwise_scaled_add_fp32 - .L_jump_table // 0x0D elementwise_scaled_add_fp32
    .long   .L_op_elementwise_mul_fp32 - .L_jump_table        // 0x0E elementwise_mul_fp32
    .long   .L_op_relu_backward_fp32 - .L_jump_table    // 0x0F relu_backward_fp32
    .long   .L_op_scatter_tile_fp32 - .L_jump_table     // 0x10 scatter_tile_fp32
    .long   .L_op_transpose_fp32 - .L_jump_table        // 0x11 transpose_fp32
    .long   .L_op_softmax_argmax_fp32 - .L_jump_table   // 0x12 softmax_argmax_fp32
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
    .long   .L_op_faddv_zreg - .L_jump_table    // 0x4F faddv_zreg
    .long   .L_op_frsqrt_zreg - .L_jump_table   // 0x50 frsqrt_zreg
    .long   .L_op_rms_norm_fp32 - .L_jump_table // 0x51 rms_norm_fp32
    .long   .L_op_broadcast_scalar_zreg - .L_jump_table // 0x52 broadcast_scalar_zreg
    .long   .L_op_fscale_zreg - .L_jump_table  // 0x53 fscale_zreg
    .long   .L_op_silu_fp32 - .L_jump_table    // 0x54 silu_fp32
    .long   .L_op_rope_fp32 - .L_jump_table   // 0x55 rope_fp32
    .long   .L_op_softmax_fp32 - .L_jump_table // 0x56 softmax_fp32
    .long   .L_op_q8_0_gemv - .L_jump_table   // 0x57 q8_0_gemv
    .long   .L_op_q4_0_gemv - .L_jump_table   // 0x58 q4_0_gemv
    .long   .L_op_fdot_zreg - .L_jump_table   // 0x59 fdot_zreg
    .long   .L_op_fmla_wide - .L_jump_table   // 0x5A fmla_wide_zreg
    .long   .L_op_fadd_wide - .L_jump_table   // 0x5B fadd_wide_zreg
    .long   .L_op_fsub_wide - .L_jump_table   // 0x5C fsub_wide_zreg
    .long   .L_op_fmul_wide - .L_jump_table   // 0x5D fmul_wide_zreg
    .long   .L_op_load_wide_param - .L_jump_table  // 0x5E load_wide_param
    .long   .L_op_store_wide_param - .L_jump_table // 0x5F store_wide_param
    .long   .L_op_cblas_bfgemm - .L_jump_table    // 0x60 cblas_bfgemm
    .long   .L_op_cblas_igemm - .L_jump_table     // 0x61 cblas_igemm
    .long   .L_op_cblas_ugemm - .L_jump_table     // 0x62 cblas_ugemm
    .long   .L_op_cblas_usgemm - .L_jump_table    // 0x63 cblas_usgemm
    .long   .L_op_gemm_tile_fp32 - .L_jump_table  // 0x64 gemm_tile_fp32
    .long   .L_op_softmax_partial_fp32 - .L_jump_table // 0x65 softmax_partial_fp32
    .long   .L_op_softmax_correct_fp32 - .L_jump_table // 0x66 softmax_correct_fp32
    .long   .L_op_reduce_sum_sq_fp32 - .L_jump_table   // 0x67 reduce_sum_sq_fp32
    .long   .L_op_reduce_col_sum_fp32 - .L_jump_table // 0x68 reduce_col_sum_fp32
    .long   .L_op_silu_backward_fp32 - .L_jump_table  // 0x69 silu_backward_fp32
    .long   .L_op_softmax_backward_fp32 - .L_jump_table // 0x6A softmax_backward_fp32
    .long   .L_op_gelu_fp32 - .L_jump_table            // 0x6B gelu_fp32
    .long   .L_op_layer_norm_fp32 - .L_jump_table      // 0x6C layer_norm_fp32
    .long   .L_op_causal_mask_fp32 - .L_jump_table     // 0x6D causal_mask_fp32
    .long   .L_op_adam_step_fp32 - .L_jump_table       // 0x6E adam_step_fp32
    .long   .L_op_gelu_backward_fp32 - .L_jump_table  // 0x6F gelu_backward_fp32
    .long   .L_op_rms_norm_backward_fp32 - .L_jump_table // 0x70 rms_norm_backward_fp32
    .long   .L_op_layer_norm_backward_fp32 - .L_jump_table // 0x71 layer_norm_backward_fp32
    .long   .L_op_rope_backward_fp32 - .L_jump_table  // 0x72 rope_backward_fp32
    .long   .L_op_cross_entropy_fp32 - .L_jump_table  // 0x73 cross_entropy_fp32
    .long   .L_op_elementwise_sub_fp32 - .L_jump_table // 0x74 elementwise_sub_fp32
    .long   .L_op_q4_k_gemv - .L_jump_table           // 0x75 q4_k_gemv
    .long   .L_op_q2_k_gemv - .L_jump_table           // 0x76 q2_k_gemv
    .long   .L_op_q3_k_gemv - .L_jump_table           // 0x77 q3_k_gemv
    .long   .L_op_q5_k_gemv - .L_jump_table           // 0x78 q5_k_gemv
    .long   .L_op_q6_k_gemv - .L_jump_table           // 0x79 q6_k_gemv
    .long   .L_op_flash_attention_fp32 - .L_jump_table // 0x7A flash_attention_fp32
    .long   .L_op_get_rows_fp32 - .L_jump_table       // 0x7B get_rows_fp32
    .long   .L_op_get_rows_q8_0 - .L_jump_table       // 0x7C get_rows_q8_0
    .long   .L_op_get_rows_q4_0 - .L_jump_table       // 0x7D get_rows_q4_0
    .long   .L_op_dense_strided_fp32 - .L_jump_table  // 0x7E dense_strided_fp32
    .long   .L_op_advance_param_stride - .L_jump_table // 0x7F advance_param_stride
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
// DENSE_STRIDED_FP32 (0x7E)
// Fused matmul+bias+ReLU with explicit strides (lda, ldb, ldc).
// Same as dense_fp32 but A stride = lda*4, B stride = ldb*4, C stride = ldc*4.
// Immediates: M:u32, N:u32, K:u32, lda:u32, ldb:u32, ldc:u32, scale:f32, flags:u8
// Operands:   A:u64, B:u64, bias:u64, C:u64
// 16×32 output tiles: za0+za1 accumulate, za2 inline transpose.
//   Caller must pad output buffer rows to ((M+31)&~31).
//   Caller must pad A columns (K dim) to ((K+15)&~15) with zeros.
// Stack layout (144 bytes):
//   [0]  A, [8] B, [16] C
//   [24] M, [28] N, [32] K
//   [40] N_pad, [44] M_pad
//   [48] k_blocks
//   [56] B_stride (ldb*4), [64] A_row_stride (lda*4)
//   [72] bias
//   [80] ti, [84] tj
//   [88] C_stride (ldc*4)
// ================================================================
.L_op_dense_strided_fp32:
    // ── Parse immediates + operands ──
    ldr     w0, [x19]              // M
    ldr     w1, [x19, #4]          // N
    ldr     w2, [x19, #8]          // K
    ldr     w11, [x19, #12]        // lda
    ldr     w12, [x19, #16]        // ldb
    ldr     w13, [x19, #20]        // ldc
    ldr     s20, [x19, #24]        // scale (f32)
    ldrb    w18, [x19, #28]        // flags (bit 0 = relu)
    add     x19, x19, #29
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
    lsl     x17, x11, #2           // lda * 4 = A row stride in bytes
    // ── Branchless ReLU threshold: z21 = relu ? 0.0 : -FLT_MAX ──
    mov     z20.s, s20             // broadcast scale
    and     w18, w18, #1
    movz    w10, #0xFF7F, lsl #16
    movk    w10, #0xFFFF           // -FLT_MAX
    cmp     w18, #0
    csel    w10, w10, wzr, eq      // relu=0 → -FLT_MAX; relu=1 → 0.0
    fmov    s21, w10
    mov     z21.s, s21
    // ── Fixed 144-byte stack frame (context only, no scratch) ──
    sub     sp, sp, #144
    stp     x5, x6, [sp, #0]      // [0] A, [8] B
    str     x7, [sp, #16]         // [16] C
    stp     w0, w1, [sp, #24]     // [24] M, [28] N
    str     w2, [sp, #32]         // [32] K
    stp     w4, w3, [sp, #40]     // [40] N_pad, [44] M_pad  (store before x3 is reused)
    str     w15, [sp, #48]        // [48] k_blocks
    lsl     x3, x12, #2           // ldb * 4 = B row stride in bytes
    stp     x3, x17, [sp, #56]    // [56] B_stride, [64] A_row_stride
    str     x8, [sp, #72]         // [72] bias
    lsl     x14, x13, #2          // ldc * 4 = C row stride in bytes
    str     x14, [sp, #88]        // [88] C_stride
    // ── Tile row loop: ti = 0, 16, ... ──
    mov     w0, #0                 // ti = 0
.L_strided_tile_row:
    str     w0, [sp, #80]          // save ti
    // A_tile_base = A + ti * lda * 4
    ldr     x5, [sp, #0]
    ldr     x17, [sp, #64]         // A_row_stride = lda*4
    mul     x10, x0, x17           // ti * lda * 4
    add     x5, x5, x10           // x5 = A + ti * lda * 4
    // ── Tile column loop: tj = 0, 32, ... ──
    mov     w1, #0                 // tj = 0
.L_strided_tile_col:
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
    mova    za0h.s[w12, 0:3], {z0.s-z3.s}
    mov     w12, #4
    mova    za0h.s[w12, 0:3], {z0.s-z3.s}
    mov     w12, #8
    mova    za0h.s[w12, 0:3], {z0.s-z3.s}
    mov     w12, #12
    mova    za0h.s[w12, 0:3], {z0.s-z3.s}
    // Broadcast bias[tj+16..tj+31] into all 16 rows of za1
    mov     z5.d, z4.d
    mov     z6.d, z4.d
    mov     z7.d, z4.d
    mov     w12, #0
    mova    za1h.s[w12, 0:3], {z4.s-z7.s}
    mov     w12, #4
    mova    za1h.s[w12, 0:3], {z4.s-z7.s}
    mov     w12, #8
    mova    za1h.s[w12, 0:3], {z4.s-z7.s}
    mov     w12, #12
    mova    za1h.s[w12, 0:3], {z4.s-z7.s}
    // B_tile = B + tj * 4
    ldr     x6, [sp, #8]
    add     x6, x6, x1, lsl #2
    ldr     x3, [sp, #56]          // B_stride = ldb*4
    ldr     x17, [sp, #64]         // A_row_stride = lda*4
    ldr     w15, [sp, #48]         // k_blocks
    ptrue   p0.s
    cntw    x9
    // x13 = ck byte offset into A rows (advances by 64 per k-block)
    mov     x13, xzr
    // ── K-block loop: process 16 columns of A per iteration ──
    cbz     w15, .L_strided_kblock_done
.L_strided_kblock:
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
    b.ne    .L_strided_kblock
.L_strided_kblock_done:
    // ── Store: extract za0+za1 → scale → clamp → C ──
    // Use ldc-based stride for output, NOT N*4
    ldr     x7, [sp, #16]         // C
    ldr     w0, [sp, #80]         // ti
    ldr     w1, [sp, #84]         // tj
    ldr     w14, [sp, #28]        // N (column count for predicates)
    ldr     x10, [sp, #88]        // C_stride = ldc*4
    // C_tile = C + ti * ldc * 4 + tj * 4
    mul     x8, x0, x10           // ti * ldc * 4
    add     x8, x8, x1, lsl #2   // + tj * 4
    add     x7, x7, x8           // C_tile = C + ti*ldc*4 + tj*4
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
    b.lt    .L_strided_tile_col
    // ── Advance tile row ──
    ldr     w0, [sp, #80]          // ti
    ldr     w3, [sp, #44]          // M_pad
    add     w0, w0, #16
    cmp     w0, w3
    b.lt    .L_strided_tile_row
.L_strided_tile_done:
    add     sp, sp, #144
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
// ADVANCE_PARAM_STRIDE (0x7F) — Advance param[idx] by stride bytes
// Encoding: [0x7F][idx:u8][stride:u32]
// ================================================================
.L_op_advance_param_stride:
    ldrb    w9, [x19], #1          // idx
    ldr     w10, [x19], #4         // stride (u32, byte count)
    add     x11, sp, #128          // param table base
    ldr     x8, [x11, x9, lsl #3] // current pointer
    add     x8, x8, x10           // advance by stride bytes
    str     x8, [x11, x9, lsl #3]
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
// ================================================================
// FADDV_ZREG (0x4F) — z{dst}.s = broadcast(faddv(z{src}.s))
// Encoding: [0x4F][dst:u8][src:u8]
// Horizontal FP32 sum across all lanes, broadcast result to all lanes
// ================================================================
.L_op_faddv_zreg:
    ldrb    w22, [x19], #1         // dst index
    ldrb    w9, [x19], #1          // src index
    ptrue   p0.s
    addvl   sp, sp, #-1
    adr     x26, .L_faddv_1
    b       .L_tramp_store         // store z{src} to [sp]
.L_faddv_1:
    ldr     z0, [sp]               // z0 = src
    faddv   s0, p0, z0.s           // horizontal sum -> s0
    mov     z0.s, s0               // broadcast scalar to all lanes
    str     z0, [sp]
    mov     w9, w22                // dst index
    adr     x26, .L_faddv_2
    b       .L_tramp_load          // load [sp] to z{dst}
.L_faddv_2:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// FRSQRT_ZREG (0x50) — z{dst}.s = 1/sqrt(z{src}.s) per element
// Encoding: [0x50][dst:u8][src:u8]
// Two Newton-Raphson refinement steps for accuracy
// ================================================================
.L_op_frsqrt_zreg:
    ldrb    w22, [x19], #1         // dst index
    ldrb    w9, [x19], #1          // src index
    ptrue   p0.s
    addvl   sp, sp, #-1
    adr     x26, .L_frsqrt_1
    b       .L_tramp_store         // store z{src} to [sp]
.L_frsqrt_1:
    ldr     z0, [sp]               // z0 = src (x)
    frsqrte z1.s, z0.s             // z1 = initial estimate of 1/sqrt(x)
    fmul    z2.s, z1.s, z1.s       // z2 = est^2
    frsqrts z2.s, z0.s, z2.s       // z2 = (3 - x * est^2) / 2
    fmul    z1.s, p0/m, z1.s, z2.s // z1 = refined estimate
    fmul    z2.s, z1.s, z1.s       // z2 = est^2 (second pass)
    frsqrts z2.s, z0.s, z2.s       // z2 = refinement factor
    fmul    z1.s, p0/m, z1.s, z2.s // z1 = final result
    mov     z0.d, z1.d             // z0 = final result
    str     z0, [sp]
    mov     w9, w22                // dst index
    adr     x26, .L_frsqrt_2
    b       .L_tramp_load          // load [sp] to z{dst}
.L_frsqrt_2:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// RMS_NORM_FP32 (0x51) — Fused RMS normalization (memory-to-memory)
// out[i] = in[i] * rsqrt(mean(in[i]^2) + eps) * weight[i]
// Encoding: [0x51][dim:u32][eps:f32][input_ptr:u64][weight_ptr:u64][output_ptr:u64]
// ================================================================
.L_op_rms_norm_fp32:
    ldr     w22, [x19]             // dim (u32)
    ldr     s16, [x19, #4]         // eps (f32)
    ldr     x8, [x19, #8]         // input_ptr
    ldr     x10, [x19, #16]        // weight_ptr
    ldr     x11, [x19, #24]        // output_ptr
    add     x19, x19, #32          // advance IP past all operands
    ptrue   p0.s
    cntw    x9                     // SVLs = 16 on M4
    // Save input_ptr to stack for pass 2
    str     x8, [sp, #128]
    // ── Pass 1: compute sum of squares ──
    mov     z4.d, #0               // accumulator
    mov     x12, x8                // x12 = current input ptr
    mov     w13, w22               // w13 = remaining elements
.L_rms_pass1:
    cbz     w13, .L_rms_pass1_done
    ld1w    {z0.s}, p0/z, [x12]
    fmla    z4.s, p0/m, z0.s, z0.s // accumulate x^2
    add     x12, x12, x9, lsl #2  // advance by SVLs * 4 bytes
    sub     w13, w13, w9           // decrement remaining
    b       .L_rms_pass1
.L_rms_pass1_done:
    // Horizontal sum of squared values
    faddv   s4, p0, z4.s           // s4 = sum of all x^2
    // Compute mean = sum / dim
    ucvtf   s5, w22                // s5 = (float)dim
    fdiv    s4, s4, s5             // s4 = mean(x^2)
    fadd    s4, s4, s16            // s4 = mean(x^2) + eps
    // Compute rsqrt(mean + eps) with two Newton-Raphson steps
    frsqrte s7, s4                 // initial estimate
    fmul    s8, s7, s7             // est^2
    frsqrts s8, s4, s8             // refinement factor
    fmul    s7, s7, s8             // refined estimate
    fmul    s8, s7, s7             // est^2 (second pass)
    frsqrts s8, s4, s8             // refinement factor
    fmul    s7, s7, s8             // final rsqrt
    // Broadcast rsqrt scalar to all lanes
    mov     z16.s, s7
    // ── Pass 2: normalize and scale ──
    ldr     x12, [sp, #128]        // reload input_ptr
    mov     x14, x10               // x14 = weight_ptr
    mov     x15, x11               // x15 = output_ptr
    mov     w13, w22               // w13 = remaining elements
.L_rms_pass2:
    cbz     w13, .L_rms_done
    ld1w    {z0.s}, p0/z, [x12]   // load input
    ld1w    {z1.s}, p0/z, [x14]   // load weight
    fmul    z0.s, p0/m, z0.s, z16.s // x * rsqrt(mean_sq + eps)
    fmul    z0.s, p0/m, z0.s, z1.s  // * weight
    st1w    {z0.s}, p0, [x15]
    add     x12, x12, x9, lsl #2  // advance input
    add     x14, x14, x9, lsl #2  // advance weight
    add     x15, x15, x9, lsl #2  // advance output
    sub     w13, w13, w9           // decrement remaining
    b       .L_rms_pass2
.L_rms_done:
    b       .L_dispatch
// ================================================================
// BROADCAST_SCALAR_ZREG (0x52) — z{dst}.s = broadcast(value)
// Encoding: [0x52][dst:u8][value:f32]
// Fills all 16 FP32 lanes of z{dst} with the immediate float value.
// ================================================================
.L_op_broadcast_scalar_zreg:
    ldrb    w22, [x19], #1         // dst index
    ldr     s16, [x19]             // value (f32 immediate)
    add     x19, x19, #4           // advance IP past f32
    ptrue   p0.s
    mov     z0.s, s16              // broadcast scalar to all 16 lanes
    addvl   sp, sp, #-1
    str     z0, [sp]
    mov     w9, w22                // dst index for trampoline
    adr     x26, .L_bcast_done
    b       .L_tramp_load          // load [sp] to z{dst}
.L_bcast_done:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// FSCALE_ZREG (0x53) — z{dst}.s = z{src}.s * scalar
// Encoding: [0x53][dst:u8][src:u8][scalar:f32]
// Element-wise multiply of z{src} by a broadcast scalar, result in z{dst}.
// ================================================================
.L_op_fscale_zreg:
    ldrb    w22, [x19], #1         // dst index
    ldrb    w9, [x19], #1          // src index
    ldr     s16, [x19]             // scalar (f32 immediate)
    add     x19, x19, #4           // advance IP past f32
    ptrue   p0.s
    addvl   sp, sp, #-1
    // Store z{src} to stack via trampoline
    adr     x26, .L_fscale_1
    b       .L_tramp_store         // store z{src} to [sp]
.L_fscale_1:
    ldr     z0, [sp]               // z0 = src data
    mov     z1.s, s16              // z1 = broadcast scalar
    fmul    z0.s, z0.s, z1.s       // z0 = src * scalar
    str     z0, [sp]
    mov     w9, w22                // dst index for trampoline
    adr     x26, .L_fscale_done
    b       .L_tramp_load          // load [sp] to z{dst}
.L_fscale_done:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// SILU_FP32 (0x54) — SiLU activation (memory-to-memory)
// out[i] = in[i] * sigmoid(in[i]) where sigmoid(x) = 1/(1+exp(-x))
// Encoding: [0x54][count:u32][input_ptr:u64][output_ptr:u64]
// Uses the same exp polynomial as softmax (range reduction + Horner).
// ================================================================
.L_op_silu_fp32:
    ldr     w22, [x19]             // count (u32)
    add     x19, x19, #4
    ldr     x8, [x19], #8         // input_ptr
    ldr     x11, [x19], #8        // output_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9                     // SVLs = 16 on M4
    // Load exp constants (same polynomial as softmax)
    // log2(e) = 1.4426950216
    movz    w4, #0x3FB8, lsl #16
    movk    w4, #0xAA3B
    fmov    s28, w4
    mov     z28.s, s28              // log2(e)
    // c4 = 0.009607379
    movz    w4, #0x3C1D, lsl #16
    movk    w4, #0x955A
    fmov    s29, w4
    mov     z29.s, s29              // c4
    // c3 = 0.05550348
    movz    w4, #0x3D63, lsl #16
    movk    w4, #0x5847
    fmov    s30, w4
    mov     z30.s, s30              // c3
    // c2 = 0.24022651
    movz    w4, #0x3E75, lsl #16
    movk    w4, #0xFDF0
    fmov    s31, w4
    mov     z31.s, s31              // c2
    // c1 = ln(2) = 0.6931471825
    movz    w4, #0x3F31, lsl #16
    movk    w4, #0x7218
    fmov    s27, w4
    mov     z27.s, s27              // c1 = ln(2)
    fmov    z26.s, #1.0             // c0 = 1.0
.L_silu_loop:
    cbz     w22, .L_silu_done
    ld1w    {z0.s}, p0/z, [x8]    // z0 = x (input)
    // Save original x for final multiply
    mov     z7.d, z0.d              // z7 = x (preserved)
    // Negate: compute exp(-x) for sigmoid
    fneg    z0.s, p0/m, z0.s       // z0 = -x
    // Range reduction: t = -x * log2(e)
    fmul    z1.s, z0.s, z28.s      // z1 = -x * log2(e)
    frintm  z2.s, p0/m, z1.s       // z2 = floor(t) = n
    fsub    z3.s, z1.s, z2.s        // z3 = f = t - n (fractional, [0,1))
    // Horner polynomial: 2^f = c4*f + c3, *f + c2, *f + c1, *f + c0
    fmul    z4.s, z29.s, z3.s       // c4 * f
    fadd    z4.s, z4.s, z30.s       // + c3
    fmul    z4.s, z4.s, z3.s        // * f
    fadd    z4.s, z4.s, z31.s       // + c2
    fmul    z4.s, z4.s, z3.s        // * f
    fadd    z4.s, z4.s, z27.s       // + c1
    fmul    z4.s, z4.s, z3.s        // * f
    fadd    z4.s, z4.s, z26.s       // + c0 (1.0)
    // Reconstruct 2^n via integer exponent bit manipulation
    fcvtzs  z5.s, p0/m, z2.s       // n as integer
    mov     z6.s, #-127
    smax    z5.s, p0/m, z5.s, z6.s // clamp exponent to >= -127
    add     z5.s, z5.s, #127       // bias exponent
    lsl     z5.s, z5.s, #23        // shift into IEEE754 exponent field
    // exp(-x) = poly * 2^n
    fmul    z4.s, z4.s, z5.s        // z4 = exp(-x)
    // sigmoid(x) = 1 / (1 + exp(-x))
    fadd    z4.s, z4.s, z26.s       // z4 = 1 + exp(-x)
    // Compute reciprocal via frecpe + Newton-Raphson
    frecpe  z5.s, z4.s              // z5 = ~1/z4
    frecps  z6.s, z4.s, z5.s        // refinement step
    fmul    z5.s, p0/m, z5.s, z6.s  // z5 = refined 1/(1+exp(-x))
    frecps  z6.s, z4.s, z5.s        // second refinement
    fmul    z5.s, p0/m, z5.s, z6.s  // z5 = sigmoid(x)
    // SiLU = x * sigmoid(x)
    fmul    z0.s, z7.s, z5.s        // z0 = x * sigmoid(x)
    st1w    {z0.s}, p0, [x11]
    add     x8, x8, x9, lsl #2    // advance input ptr
    add     x11, x11, x9, lsl #2   // advance output ptr
    sub     w22, w22, w9            // decrement remaining
    b       .L_silu_loop
.L_silu_done:
    b       .L_dispatch
// ================================================================
// ROPE_FP32 (0x55)
// Rotary Position Embedding for LLM attention heads.
// Applies position-dependent rotation to each pair of elements:
//   out[2i]   = in[2i]*cos(freq) - in[2i+1]*sin(freq)
//   out[2i+1] = in[2i]*sin(freq) + in[2i+1]*cos(freq)
// where freq = pos / theta^(2i/dim)
//
// Frequency generation uses repeated multiplication:
//   ratio = theta^(2/dim) computed via ln+exp polynomial chain
//   power[k+1] = power[k] * ratio, freq[k] = pos / power[k]
//
// Sin/cos via degree-5/4 Taylor with 2pi range reduction.
// Rotation vectorized via LD2W/ST2W deinterleave/reinterleave.
//
// Bytecode: [0x55][dim:u32][pos:u32][theta:f32][input_ptr:u64][output_ptr:u64]
// ================================================================
.L_op_rope_fp32:
    ldr     w22, [x19]             // dim (number of floats, must be even)
    ldr     w23, [x19, #4]         // pos (token position)
    ldr     w24, [x19, #8]         // theta bits (f32)
    add     x19, x19, #12          // advance IP past first 3 operands
    ldr     x8, [x19], #8          // input_ptr
    ldr     x11, [x19], #8         // output_ptr
    lsr     w26, w22, #1           // dim_pairs = dim / 2
    cbz     w26, .L_rope_done      // early exit if no pairs
    ptrue   p0.s                   // predicate for full vectors
    cntw    x9                     // SVLs = 16
    // ── Compute ratio = theta^(2/dim) via scalar ln+exp ──
    fmov    s0, w24                // s0 = theta
    fmov    w4, s0                 // w4 = theta bits
    ubfx    w5, w4, #23, #8        // extract biased exponent
    sub     w5, w5, #127           // e = unbiased exponent
    scvtf   s1, w5                 // s1 = (float)e
    mov     w6, #127               // biased exponent for 1.0
    bfi     w4, w6, #23, #8       // set exponent to 127 → m in [1,2)
    fmov    s2, w4                 // s2 = m
    // ln(m) for m in [1,2): minimax cubic on t = m-1
    fmov    s3, #1.0
    fsub    s2, s2, s3             // t = m - 1
    movz    w6, #0x3E94, lsl #16
    movk    w6, #0x3014
    fmov    s4, w6                 // a3 = 0.28947478
    movz    w6, #0xBEFB, lsl #16
    movk    w6, #0xD464
    fmov    s5, w6                 // a2 = -0.49190896
    movz    w6, #0x3F7F, lsl #16
    movk    w6, #0xF972
    fmov    s6, w6                 // a1 = 0.99949556
    fmul    s4, s4, s2             // a3*t
    fadd    s4, s4, s5             // a3*t + a2
    fmul    s4, s4, s2             // (a3*t + a2)*t
    fadd    s4, s4, s6             // (a3*t + a2)*t + a1
    fmul    s4, s4, s2             // ln(m)
    movz    w6, #0x3F31, lsl #16
    movk    w6, #0x7218
    fmov    s5, w6                 // ln(2) = 0.693147
    fmul    s1, s1, s5             // e * ln(2)
    fadd    s1, s1, s4             // s1 = ln(theta)
    // exponent_step = (2/dim) * ln(theta)
    fmov    s6, #2.0
    ucvtf   s7, w22                // (float)dim
    fdiv    s6, s6, s7             // 2/dim
    fmul    s6, s6, s1             // s6 = exponent_step
    // exp(exponent_step) via degree-4 polynomial
    movz    w6, #0x3FB8, lsl #16
    movk    w6, #0xAA3B
    fmov    s10, w6                // log2(e)
    fmul    s7, s6, s10            // x/ln(2)
    frintm  s8, s7                 // n = floor
    fsub    s9, s7, s8             // frac
    movz    w6, #0x3C1D, lsl #16
    movk    w6, #0x955A
    fmov    s12, w6                // c4
    movz    w6, #0x3D63, lsl #16
    movk    w6, #0x5847
    fmov    s13, w6                // c3
    movz    w6, #0x3E75, lsl #16
    movk    w6, #0xFDF0
    fmov    s14, w6                // c2
    fmul    s15, s12, s9
    fadd    s15, s15, s13
    fmul    s15, s15, s9
    fadd    s15, s15, s14
    fmul    s15, s15, s9
    movz    w6, #0x3F31, lsl #16
    movk    w6, #0x7218
    fmov    s14, w6                // c1 = ln(2)
    fadd    s15, s15, s14
    fmul    s15, s15, s9
    fmov    s14, #1.0
    fadd    s15, s15, s14          // poly
    fcvtzs  w6, s8
    add     w6, w6, #127
    lsl     w6, w6, #23
    fmov    s14, w6                // 2^n as float bits
    fmul    s0, s15, s14           // s0 = ratio = theta^(2/dim)
    // ── Build power vector [1, ratio, ratio^2, ..., ratio^(SVLs-1)] on stack ──
    add     x14, sp, #128          // reuse param area (64 bytes, not needed for pointers)
    fmov    s2, #1.0               // power = ratio^0
    mov     w12, #0
.L_rope_pw:
    cmp     w12, w9                // w9 = SVLs = 16
    b.ge    .L_rope_pw_done
    str     s2, [x14, w12, uxtw #2]
    fmul    s2, s2, s0             // power *= ratio
    add     w12, w12, #1
    b       .L_rope_pw
.L_rope_pw_done:
    ld1w    {z29.s}, p0/z, [x14]  // z29 = [1, ratio, ratio^2, ..., ratio^15]
    mov     z30.s, s2              // z30 = broadcast(ratio^SVLs) for chunk stepping
    ucvtf   s3, w23               // s3 = (float)pos
    mov     z31.s, s3              // z31 = broadcast(pos)
    // ── Load trig constants ──
    movz    w4, #0x4049, lsl #16
    movk    w4, #0x0FDB
    fmov    s16, w4
    mov     z20.s, s16             // pi = 3.14159 = 0x40490FDB
    movz    w4, #0x3EA2, lsl #16
    movk    w4, #0xF983
    fmov    s16, w4
    mov     z21.s, s16             // 1/pi = 0x3EA2F983
    movz    w4, #0xBE2A, lsl #16
    movk    w4, #0xAAAB
    fmov    s16, w4
    mov     z22.s, s16             // -1/6
    movz    w4, #0x3C08, lsl #16
    movk    w4, #0x8889
    fmov    s16, w4
    mov     z23.s, s16             // 1/120
    movz    w4, #0xB950, lsl #16
    movk    w4, #0x0D01
    fmov    s16, w4
    mov     z27.s, s16             // -1/5040
    movz    w4, #0xBF00, lsl #16
    movk    w4, #0x0000
    fmov    s16, w4
    mov     z24.s, s16             // -0.5
    movz    w4, #0x3D2A, lsl #16
    movk    w4, #0xAAAB
    fmov    s16, w4
    mov     z25.s, s16             // 1/24
    movz    w4, #0xBAB6, lsl #16
    movk    w4, #0x0B61
    fmov    s16, w4
    mov     z28.s, s16             // -1/720
    fmov    z26.s, #1.0
    // ── Vectorized rotation loop ──
    mov     w12, #0                // pair index offset
    whilelt p1.s, w12, w26
.L_rope_vec:
    b.none  .L_rope_done
    // Compute freq[k] = pos / power[k] via vector divide
    movprfx z0, z31
    fdiv    z0.s, p1/m, z0.s, z29.s // z0 = pos / power (only active lanes)
    // Range reduce to [-pi/2, pi/2]: r = x - round(x/pi)*pi
    fmul    z1.s, z0.s, z21.s     // x / pi
    frintn  z1.s, p0/m, z1.s      // n = round(x/pi)
    fcvtzs  z7.s, p0/m, z1.s      // n as integer (for parity check)
    fmls    z0.s, p0/m, z1.s, z20.s // r = x - n*pi, in [-pi/2, pi/2]
    // Sign flip mask: if n is odd, negate sin & cos
    and     z7.s, z7.s, #1        // parity bit
    lsl     z7.s, z7.s, #31       // 0x80000000 if odd, 0 if even
    // r^2
    fmul    z1.s, z0.s, z0.s      // z1 = r^2
    // sin(r) via 7th-order Horner: r*(1 + r^2*(-1/6 + r^2*(1/120 + r^2*(-1/5040))))
    mov     z2.d, z27.d            // z2 = -1/5040
    fmad    z2.s, p0/m, z1.s, z23.s // z2 = z2*r^2 + 1/120
    fmad    z2.s, p0/m, z1.s, z22.s // z2 = z2*r^2 + (-1/6)
    fmad    z2.s, p0/m, z1.s, z26.s // z2 = z2*r^2 + 1
    fmul    z2.s, p0/m, z2.s, z0.s  // z2 = z2*r = sin(r)
    // cos(r) via 6th-order Horner: 1 + r^2*(-1/2 + r^2*(1/24 + r^2*(-1/720)))
    mov     z3.d, z28.d            // z3 = -1/720
    fmad    z3.s, p0/m, z1.s, z25.s // z3 = z3*r^2 + 1/24
    fmad    z3.s, p0/m, z1.s, z24.s // z3 = z3*r^2 + (-1/2)
    fmad    z3.s, p0/m, z1.s, z26.s // z3 = z3*r^2 + 1 = cos(r)
    // Apply sign correction for odd n
    eor     z2.d, z2.d, z7.d      // flip sin sign if n odd
    eor     z3.d, z3.d, z7.d      // flip cos sign if n odd
    // LD2W deinterleaves: z4 = even (x[0],x[2],...), z5 = odd (x[1],x[3],...)
    ld2w    {z4.s, z5.s}, p1/z, [x8]
    // Rotation: out_even = in_even*cos - in_odd*sin
    //           out_odd  = in_even*sin + in_odd*cos
    mov     z6.d, z4.d             // save in_even
    fmul    z4.s, p0/m, z4.s, z3.s // in_even * cos
    fmls    z4.s, p0/m, z5.s, z2.s // - in_odd * sin
    fmul    z5.s, p0/m, z5.s, z3.s // in_odd * cos
    fmla    z5.s, p0/m, z6.s, z2.s // + in_even * sin
    // ST2W reinterleaves pairs
    st2w    {z4.s, z5.s}, p1, [x11]
    // Advance pointers: each chunk = SVLs pairs = 2*SVLs floats
    add     x8, x8, x9, lsl #3    // input += SVLs * 8
    add     x11, x11, x9, lsl #3  // output += SVLs * 8
    // Advance power vector: multiply by ratio^SVLs for next chunk
    fmul    z29.s, p0/m, z29.s, z30.s
    add     w12, w12, w9
    whilelt p1.s, w12, w26
    b       .L_rope_vec
.L_rope_done:
    b       .L_dispatch
// ================================================================
// SOFTMAX_FP32 (0x56)
// Standalone softmax over a 1D fp32 array:
//   output[i] = exp(input[i] - max) / sum(exp(input[j] - max))
//
// Three-pass numerically stable algorithm:
//   Pass 1: find max via fmax + fmaxv horizontal reduction
//   Pass 2: compute exp(x-max) via degree-4 minimax polynomial,
//           store intermediate, accumulate sum
//   Pass 3: multiply each stored exp by 1/sum
//
// Exp reconstruction uses integer exponent injection (no fscale):
//   2^n = (n+127) << 23 reinterpret as float
//
// Bytecode: [0x56][dim:u32][input_ptr:u64][output_ptr:u64]
// ================================================================
.L_op_softmax_fp32:
    ldr     w22, [x19]             // dim
    add     x19, x19, #4
    ldr     x8, [x19], #8         // input_ptr
    ldr     x11, [x19], #8        // output_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9                     // SVLs = 16
    // ── Load exp polynomial constants ──
    movz    w4, #0x3FB8, lsl #16
    movk    w4, #0xAA3B
    fmov    s28, w4
    mov     z28.s, s28             // log2(e)
    movz    w4, #0x3C1D, lsl #16
    movk    w4, #0x955A
    fmov    s29, w4
    mov     z29.s, s29             // c4
    movz    w4, #0x3D63, lsl #16
    movk    w4, #0x5847
    fmov    s30, w4
    mov     z30.s, s30             // c3
    movz    w4, #0x3E75, lsl #16
    movk    w4, #0xFDF0
    fmov    s31, w4
    mov     z31.s, s31             // c2
    movz    w4, #0x3F31, lsl #16
    movk    w4, #0x7218
    fmov    s27, w4
    mov     z27.s, s27             // c1 = ln(2)
    fmov    z26.s, #1.0            // c0 = 1.0
    mov     x14, x8               // save input_ptr
    mov     x15, x11              // save output_ptr
    // ── Pass 1: find max ──
    movz    w4, #0xFF80, lsl #16   // -inf
    fmov    s16, w4
    mov     z16.s, s16
    mov     w12, w22
.L_soft_max:
    cbz     w12, .L_soft_max_done
    ld1w    {z0.s}, p0/z, [x8]
    fmax    z16.s, p0/m, z16.s, z0.s
    add     x8, x8, x9, lsl #2
    sub     w12, w12, w9
    cbnz    w12, .L_soft_max
.L_soft_max_done:
    fmaxv   s16, p0, z16.s
    mov     z16.s, s16
    // ── Pass 2: exp(x - max) + accumulate sum ──
    fmov    z17.s, #0.0
    mov     x8, x14
    mov     x11, x15
    mov     w12, w22
.L_soft_exp:
    cbz     w12, .L_soft_exp_done
    ld1w    {z0.s}, p0/z, [x8]
    fsub    z0.s, z0.s, z16.s
    fmul    z1.s, z0.s, z28.s     // x * log2(e)
    frintm  z2.s, p0/m, z1.s      // n = floor
    fsub    z3.s, z1.s, z2.s      // frac
    fmul    z4.s, z29.s, z3.s     // c4*f
    fadd    z4.s, z4.s, z30.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z31.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z27.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z26.s     // poly(frac)
    fcvtzs  z5.s, p0/m, z2.s
    mov     z6.s, #-127
    smax    z5.s, p0/m, z5.s, z6.s
    add     z5.s, z5.s, #127
    lsl     z5.s, z5.s, #23       // 2^n as IEEE bits
    fmul    z4.s, z4.s, z5.s      // exp(x - max)
    st1w    {z4.s}, p0, [x11]
    fadd    z17.s, z17.s, z4.s
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    sub     w12, w12, w9
    b       .L_soft_exp
.L_soft_exp_done:
    // ── Pass 3: normalize ──
    faddv   s17, p0, z17.s
    fmov    s18, #1.0
    fdiv    s17, s18, s17          // 1/sum
    mov     z17.s, s17
    mov     x11, x15
    mov     w12, w22
.L_soft_div:
    cbz     w12, .L_soft_done
    ld1w    {z0.s}, p0/z, [x11]
    fmul    z0.s, z0.s, z17.s
    st1w    {z0.s}, p0, [x11]
    add     x11, x11, x9, lsl #2
    sub     w12, w12, w9
    b       .L_soft_div
.L_soft_done:
    b       .L_dispatch
// ================================================================
// Q8_0_GEMV (0x57) — Quantized GEMV for llama.cpp Q8_0 format
// Encoding: [0x57][M:u32][K:u32][input_ptr:u64][weights_ptr:u64][output_ptr:u64]
// Each Q8_0 block: 2-byte fp16 scale + 32 int8 quants = 34 bytes per 32 elements.
// Computes output[m] = sum_k dequant(W[m,k]) * input[k] for all M rows.
// ================================================================
.L_op_q8_0_gemv:
    ldr     w22, [x19]             // M (number of output rows)
    add     x19, x19, #4
    ldr     w23, [x19]             // K (input dimension, multiple of 32)
    add     x19, x19, #4
    ldr     x8, [x19], #8         // input_ptr (fp32, K elements)
    ldr     x11, [x19], #8        // weights_ptr (Q8_0 blocks, M rows)
    ldr     x13, [x19], #8        // output_ptr (fp32, M elements)
    cbz     w22, .L_dispatch       // early exit if M == 0
    ptrue   p0.s
    lsr     w24, w23, #5           // num_blocks = K / 32
    mov     x26, x8               // save input base for resetting each row
.L_q8_gemv_row:
    cbz     w22, .L_dispatch       // all rows done
    mov     z4.d, #0               // zero accumulator for this row
    mov     x8, x26               // reset input ptr to start of vector
    mov     w10, w24               // block counter = num_blocks
.L_q8_gemv_block:
    cbz     w10, .L_q8_gemv_store
    // Load fp16 scale from block header, convert to fp32, broadcast
    ldr     h0, [x11]             // fp16 scale (2 bytes)
    fcvt    s0, h0                // fp16 -> fp32
    mov     z16.s, s0             // broadcast scale to all 16 lanes
    // First half: 16 int8 quants at [block + 2]
    add     x9, x11, #2           // x9 = &qs[0]
    ld1sb   {z0.s}, p0/z, [x9]   // load 16 signed int8, sign-extend to int32
    scvtf   z0.s, p0/m, z0.s     // int32 -> fp32
    fmul    z0.s, z0.s, z16.s    // dequantize: qs[i] * scale
    ld1w    {z1.s}, p0/z, [x8]   // load 16 fp32 input values
    fmla    z4.s, p0/m, z0.s, z1.s // acc += dequant * input
    add     x8, x8, #64           // input ptr += 16 floats (64 bytes)
    // Second half: next 16 int8 quants at [block + 18]
    add     x9, x9, #16           // x9 = &qs[16]
    ld1sb   {z0.s}, p0/z, [x9]   // load 16 signed int8, sign-extend to int32
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z16.s
    ld1w    {z1.s}, p0/z, [x8]
    fmla    z4.s, p0/m, z0.s, z1.s
    add     x8, x8, #64           // input ptr += 16 floats
    // Advance to next Q8_0 block (34 bytes total)
    add     x11, x11, #34
    sub     w10, w10, #1
    b       .L_q8_gemv_block
.L_q8_gemv_store:
    faddv   s4, p0, z4.s          // horizontal sum -> scalar result
    str     s4, [x13], #4         // store output[m], advance output ptr
    sub     w22, w22, #1
    b       .L_q8_gemv_row
// ================================================================
// Q4_0_GEMV (0x58) — Quantized GEMV for llama.cpp Q4_0 format
// Encoding: [0x58][M:u32][K:u32][input_ptr:u64][weights_ptr:u64][output_ptr:u64]
// Each Q4_0 block: 2-byte fp16 scale + 16 packed bytes (32 4-bit values) = 18 bytes.
// Low nibble of byte i = element i (0..15), high nibble = element i+16 (16..31).
// Dequant: float_val = (nibble - 8) * fp16_to_fp32(d)
// ================================================================
.L_op_q4_0_gemv:
    ldr     w22, [x19]             // M (number of output rows)
    add     x19, x19, #4
    ldr     w23, [x19]             // K (input dimension, multiple of 32)
    add     x19, x19, #4
    ldr     x8, [x19], #8         // input_ptr (fp32, K elements)
    ldr     x11, [x19], #8        // weights_ptr (Q4_0 blocks, M rows)
    ldr     x13, [x19], #8        // output_ptr (fp32, M elements)
    cbz     w22, .L_dispatch       // early exit if M == 0
    ptrue   p0.s
    lsr     w24, w23, #5           // num_blocks = K / 32
    mov     x26, x8               // save input base for resetting each row
    // Prepare constant: 8 in every 32-bit lane for unsigned->signed offset
    mov     w9, #8
    dup     z17.s, w9              // z17 = {8, 8, ..., 8} for subtracting bias
.L_q4_gemv_row:
    cbz     w22, .L_dispatch       // all rows done
    mov     z4.d, #0               // zero accumulator for this row
    mov     x8, x26               // reset input ptr to start of vector
    mov     w10, w24               // block counter = num_blocks
.L_q4_gemv_block:
    cbz     w10, .L_q4_gemv_store
    // Load fp16 scale from block header, convert to fp32, broadcast
    ldr     h0, [x11]             // fp16 scale (2 bytes)
    fcvt    s0, h0                // fp16 -> fp32
    mov     z16.s, s0             // broadcast scale to all 16 lanes
    // Load 16 packed bytes (32 nibbles) from [block + 2]
    add     x9, x11, #2           // x9 = &qs[0]
    ld1b    {z0.s}, p0/z, [x9]   // load 16 bytes, zero-extend each to 32-bit lane
    // Extract LOW nibbles (elements 0..15): val = (byte & 0x0F) - 8
    mov     z2.d, z0.d             // copy before destructive AND
    and     z2.s, z2.s, #0x0F     // isolate low nibble in each lane
    sub     z2.s, z2.s, z17.s     // subtract 8: unsigned [0,15] -> signed [-8,+7]
    scvtf   z2.s, p0/m, z2.s     // int32 -> fp32
    fmul    z2.s, z2.s, z16.s    // dequantize: signed_val * scale
    ld1w    {z1.s}, p0/z, [x8]   // load 16 fp32 inputs (elements 0..15)
    fmla    z4.s, p0/m, z2.s, z1.s // acc += dequant * input
    add     x8, x8, #64           // input ptr += 16 floats (64 bytes)
    // Extract HIGH nibbles (elements 16..31): val = (byte >> 4) - 8
    lsr     z3.s, z0.s, #4        // shift right 4 bits
    and     z3.s, z3.s, #0x0F     // mask nibble (top bits already zero from zero-extend+shift)
    sub     z3.s, z3.s, z17.s     // subtract 8
    scvtf   z3.s, p0/m, z3.s
    fmul    z3.s, z3.s, z16.s
    ld1w    {z1.s}, p0/z, [x8]   // load 16 fp32 inputs (elements 16..31)
    fmla    z4.s, p0/m, z3.s, z1.s
    add     x8, x8, #64           // input ptr += 16 floats
    // Advance to next Q4_0 block (18 bytes total)
    add     x11, x11, #18
    sub     w10, w10, #1
    b       .L_q4_gemv_block
.L_q4_gemv_store:
    faddv   s4, p0, z4.s          // horizontal sum -> scalar result
    str     s4, [x13], #4         // store output[m], advance output ptr
    sub     w22, w22, #1
    b       .L_q4_gemv_row
// ================================================================
// FDOT_ZREG (0x59)
// Dot product with 1/2/4-wide jump points.
// Bytecode: [0x59][width:u8]
//   width=1: z0 = broadcast(dot(z0, z1))         — 16 products → scalar
//   width=2: z0 = broadcast(dot(z0:z1, z2:z3))   — 32 products → scalar
//   width=4: z0 = broadcast(dot(z0:z3, z4:z7))   — 64 products → scalar
// ================================================================
.L_op_fdot_zreg:
    ldrb    w22, [x19], #1         // width (1, 2, or 4)
    ptrue   p0.s
    cmp     w22, #4
    b.eq    .L_fdot_4
    cmp     w22, #2
    b.eq    .L_fdot_2
    // ── Width 1: dot(z0, z1) → z0 ──
.L_fdot_1:
    fmul    z0.s, z0.s, z1.s      // z0[i] *= z1[i]
    faddv   s0, p0, z0.s          // s0 = horizontal sum
    mov     z0.s, s0               // broadcast scalar to all lanes
    b       .L_dispatch
    // ── Width 2: dot(z0:z1, z2:z3) → z0 ──
.L_fdot_2:
    fmul    z0.s, z0.s, z2.s      // pair 0: z0[i] *= z2[i]
    fmla    z0.s, p0/m, z1.s, z3.s // pair 1: z0[i] += z1[i]*z3[i]
    faddv   s0, p0, z0.s          // horizontal sum of 32 products
    mov     z0.s, s0               // broadcast
    b       .L_dispatch
    // ── Width 4: dot(z0:z3, z4:z7) → z0 ──
.L_fdot_4:
    fmul    z0.s, z0.s, z4.s      // pair 0
    fmla    z0.s, p0/m, z1.s, z5.s // pair 1
    fmla    z0.s, p0/m, z2.s, z6.s // pair 2
    fmla    z0.s, p0/m, z3.s, z7.s // pair 3
    faddv   s0, p0, z0.s          // horizontal sum of 64 products
    mov     z0.s, s0               // broadcast
    b       .L_dispatch
// ================================================================
// FMLA_WIDE_ZREG (0x5A)
// Wide fused multiply-accumulate: multiple products into z0 accumulator.
// Bytecode: [0x5A][width:u8]
//   width=1: z0 += z1 * z2
//   width=2: z0 += z1*z3 + z2*z4           (a=z1:z2, b=z3:z4)
//   width=4: z0 += z1*z5 + z2*z6 + z3*z7 + z4*z8  (a=z1:z4, b=z5:z8)
// All implicit fixed registers. User arranges via mov_zreg.
// ================================================================
.L_op_fmla_wide:
    ldrb    w22, [x19], #1         // width (1, 2, or 4)
    ptrue   p0.s
    cmp     w22, #4
    b.eq    .L_fmla_w4
    cmp     w22, #2
    b.eq    .L_fmla_w2
    // ── Width 1: z0 += z1 * z2 ──
    fmla    z0.s, p0/m, z1.s, z2.s
    b       .L_dispatch
    // ── Width 2: z0 += z1*z3 + z2*z4 ──
.L_fmla_w2:
    fmla    z0.s, p0/m, z1.s, z3.s
    fmla    z0.s, p0/m, z2.s, z4.s
    b       .L_dispatch
    // ── Width 4: z0 += z1*z5 + z2*z6 + z3*z7 + z4*z8 ──
.L_fmla_w4:
    fmla    z0.s, p0/m, z1.s, z5.s
    fmla    z0.s, p0/m, z2.s, z6.s
    fmla    z0.s, p0/m, z3.s, z7.s
    fmla    z0.s, p0/m, z4.s, z8.s
    b       .L_dispatch
// ================================================================
// WIDE ARITHMETIC (0x5B-0x5D)
// fadd_wide / fsub_wide / fmul_wide
// Bytecode: [op][width:u8][dst:u8][src1:u8][src2:u8]
//   width=1: z[dst] = z[src1] op z[src2]
//   width=2: z[dst+i] = z[src1+i] op z[src2+i] for i=0..1
//   width=4: z[dst+i] = z[src1+i] op z[src2+i] for i=0..3
// Each entry point sets w24 = op selector then falls into shared loop.
// ================================================================
.L_op_fadd_wide:
    mov     w24, #0
    b       .L_wide_arith
.L_op_fsub_wide:
    mov     w24, #1
    b       .L_wide_arith
.L_op_fmul_wide:
    mov     w24, #2
.L_wide_arith:
    ldrb    w3, [x19], #1          // width (1, 2, or 4) — w25/x25 holds jump table, DO NOT clobber
    ldrb    w22, [x19], #1         // dst base index
    ldrb    w10, [x19], #1         // src1 base index
    ldrb    w23, [x19], #1         // src2 base index
    ptrue   p0.s
    addvl   sp, sp, #-1            // scratch slot
    mov     w12, #0                // iteration counter
.L_wide_arith_loop:
    cmp     w12, w3
    b.ge    .L_wide_arith_done
    add     w9, w10, w12
    adr     x26, .L_waz_1
    b       .L_tramp_store         // z[src1+i] → stack
.L_waz_1:
    ldr     z0, [sp]               // z0 = z[src1+i]
    add     w9, w23, w12
    adr     x26, .L_waz_2
    b       .L_tramp_store         // z[src2+i] → stack
.L_waz_2:
    ldr     z1, [sp]               // z1 = z[src2+i]
    cbz     w24, .L_waz_add
    cmp     w24, #1
    b.eq    .L_waz_sub
    fmul    z0.s, z0.s, z1.s
    b       .L_waz_store
.L_waz_add:
    fadd    z0.s, z0.s, z1.s
    b       .L_waz_store
.L_waz_sub:
    fsub    z0.s, z0.s, z1.s
.L_waz_store:
    str     z0, [sp]               // result → scratch
    add     w9, w22, w12           // dst+i
    adr     x26, .L_waz_3
    b       .L_tramp_load          // scratch → z[dst+i]
.L_waz_3:
    add     w12, w12, #1
    b       .L_wide_arith_loop
.L_wide_arith_done:
    addvl   sp, sp, #1
    b       .L_dispatch
// ================================================================
// LOAD_WIDE_PARAM (0x5E)
// Multi-vector contiguous load from param pointer directly into
// destination registers. Uses SME2 ld1w multi-vector form for
// width=2 (128 bytes) and width=4 (256 bytes). Advances param.
//
// Bytecode: [0x5E][width:u8][param_idx:u8][dst_base:u8]
//   dst_base must be aligned: width=2 → even, width=4 → multiple of 4
//
// width=1: ld1w into z[dst], advance param by 64
// width=2: ld1w {z[dst]:z[dst+1]}, advance param by 128
// width=4: ld1w {z[dst]:z[dst+3]}, advance param by 256
// ================================================================
.L_op_load_wide_param:
    ldrb    w22, [x19], #1         // width
    ldrb    w23, [x19], #1         // param index
    ldrb    w24, [x19], #1         // dst base register
    add     x4, sp, #128
    ldr     x8, [x4, w23, uxtw #3]
    ptrue   p0.s
    ptrue   pn8.s
    cmp     w22, #4
    b.eq    .L_lwp_4
    cmp     w22, #2
    b.eq    .L_lwp_2
    // ── Width 1: single ld1w + trampoline ──
    ld1w    {z0.s}, p0/z, [x8]
    addvl   sp, sp, #-1
    str     z0, [sp]
    mov     w9, w24
    adr     x26, .L_lwp_1_done
    b       .L_tramp_load
.L_lwp_1_done:
    addvl   sp, sp, #1
    b       .L_dispatch
    // ── Width 2: branch table for dst pairs ──
.L_lwp_2:
    adr     x5, .L_lwp2_table
    add     x5, x5, x24, lsl #2   // dst/2 * 8 bytes per entry (2 insns)
    br      x5
.L_lwp2_table:
    ld1w    {z0.s, z1.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z2.s, z3.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z4.s, z5.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z6.s, z7.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z8.s, z9.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z10.s, z11.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z12.s, z13.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z14.s, z15.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z16.s, z17.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z18.s, z19.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z20.s, z21.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z22.s, z23.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z24.s, z25.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z26.s, z27.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z28.s, z29.s}, pn8/z, [x8]
    b       .L_lwp_2_done
    ld1w    {z30.s, z31.s}, pn8/z, [x8]
    b       .L_lwp_2_done
.L_lwp_2_done:
    b       .L_dispatch
    // ── Width 4: branch table for dst quads ──
.L_lwp_4:
    adr     x5, .L_lwp4_table
    add     x5, x5, x24, lsl #1   // dst/4 * 8 bytes per entry (2 insns)
    br      x5
.L_lwp4_table:
    ld1w    {z0.s - z3.s}, pn8/z, [x8]
    b       .L_lwp_4_done
    ld1w    {z4.s - z7.s}, pn8/z, [x8]
    b       .L_lwp_4_done
    ld1w    {z8.s - z11.s}, pn8/z, [x8]
    b       .L_lwp_4_done
    ld1w    {z12.s - z15.s}, pn8/z, [x8]
    b       .L_lwp_4_done
    ld1w    {z16.s - z19.s}, pn8/z, [x8]
    b       .L_lwp_4_done
    ld1w    {z20.s - z23.s}, pn8/z, [x8]
    b       .L_lwp_4_done
    ld1w    {z24.s - z27.s}, pn8/z, [x8]
    b       .L_lwp_4_done
    ld1w    {z28.s - z31.s}, pn8/z, [x8]
    b       .L_lwp_4_done
.L_lwp_4_done:
    b       .L_dispatch
// ================================================================
// STORE_WIDE_PARAM (0x5F)
// Multi-vector contiguous store — mirrors load_wide_param.
// Bytecode: [0x5F][width:u8][param_idx:u8][src_base:u8]
// ================================================================
.L_op_store_wide_param:
    ldrb    w22, [x19], #1         // width
    ldrb    w23, [x19], #1         // param index
    ldrb    w24, [x19], #1         // src base register
    add     x4, sp, #128
    ldr     x11, [x4, w23, uxtw #3]
    ptrue   p0.s
    ptrue   pn8.s
    cmp     w22, #4
    b.eq    .L_swp_4
    cmp     w22, #2
    b.eq    .L_swp_2
    // ── Width 1: trampoline store ──
    addvl   sp, sp, #-1
    mov     w9, w24
    adr     x26, .L_swp_1_done
    b       .L_tramp_store
.L_swp_1_done:
    ldr     z0, [sp]
    st1w    {z0.s}, p0, [x11]
    addvl   sp, sp, #1
    b       .L_dispatch
    // ── Width 2: branch table ──
.L_swp_2:
    adr     x5, .L_swp2_table
    add     x5, x5, x24, lsl #2
    br      x5
.L_swp2_table:
    st1w    {z0.s, z1.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z2.s, z3.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z4.s, z5.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z6.s, z7.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z8.s, z9.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z10.s, z11.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z12.s, z13.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z14.s, z15.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z16.s, z17.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z18.s, z19.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z20.s, z21.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z22.s, z23.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z24.s, z25.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z26.s, z27.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z28.s, z29.s}, pn8, [x11]
    b       .L_swp_2_done
    st1w    {z30.s, z31.s}, pn8, [x11]
    b       .L_swp_2_done
.L_swp_2_done:
    b       .L_dispatch
    // ── Width 4: branch table ──
.L_swp_4:
    adr     x5, .L_swp4_table
    add     x5, x5, x24, lsl #1
    br      x5
.L_swp4_table:
    st1w    {z0.s - z3.s}, pn8, [x11]
    b       .L_swp_4_done
    st1w    {z4.s - z7.s}, pn8, [x11]
    b       .L_swp_4_done
    st1w    {z8.s - z11.s}, pn8, [x11]
    b       .L_swp_4_done
    st1w    {z12.s - z15.s}, pn8, [x11]
    b       .L_swp_4_done
    st1w    {z16.s - z19.s}, pn8, [x11]
    b       .L_swp_4_done
    st1w    {z20.s - z23.s}, pn8, [x11]
    b       .L_swp_4_done
    st1w    {z24.s - z27.s}, pn8, [x11]
    b       .L_swp_4_done
    st1w    {z28.s - z31.s}, pn8, [x11]
    b       .L_swp_4_done
.L_swp_4_done:
    b       .L_dispatch
// ================================================================
// CBLAS_BFGEMM (0x60) -- C = alpha*op(A_bf16)*op(B_bf16) + beta*C
//
// Encoding: [0x60][flags:u8][M:u32][N:u32][K:u32]
//           [lda:u32][ldb:u32][ldc:u32]
//           [alpha:f32][beta:f32]
//           [A_ptr:u64][B_ptr:u64][C_ptr:u64]
//
// flags bit 0: transA (0=normal, 1=transpose)
// flags bit 1: transB (0=normal, 1=transpose)
// A, B are bf16; C is fp32. lda/ldb in bf16 elements, ldc in fp32 elements.
// K must be a multiple of 32 (bf16 elements).
// Total immediate payload after opcode: 57 bytes
//
// Tile geometry: 16 x 32 output (za0 = left 16 cols, za1 = right 16 cols)
// za2 = scratch for A tile, za3 = scratch for transB
// BFMOPA 2:1 widening: each instruction processes 2 K-elements per output position
//
// Stack layout (128 bytes): identical to cblas_sgemm
//   [sp+0]:   A_ptr           [sp+8]:   B_ptr
//   [sp+16]:  C_ptr           [sp+24]:  M (w)     [sp+28]: N (w)
//   [sp+32]:  K (w)           [sp+36]:  lda (w)   [sp+40]: ldb (w)
//   [sp+44]:  ldc (w)         [sp+48]:  flags (w)
//   [sp+52]:  k_blocks (w)    [sp+56]:  M_pad (w) [sp+60]: N_pad (w)
//   [sp+64]:  A_tile_base (x) [sp+72]:  lda*2 (x)
//   [sp+80]:  ti (w)          [sp+84]:  tj (w)
//   [sp+88]:  ldb*2 (x)       [sp+96]:  ldc*4 (x)
//   [sp+104]: beta_bits (w)   [sp+112]: B_col_stride (x)
// z20 = alpha broadcast, z22 = beta broadcast
// ================================================================
.L_op_cblas_bfgemm:
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
    lsr     w15, w2, #5               // k_blocks = K / 32
    str     w15, [sp, #52]
    add     w10, w0, #15
    and     w10, w10, #0xFFFFFFF0      // M_pad = (M+15) & ~15
    str     w10, [sp, #56]
    add     w11, w1, #31
    and     w11, w11, #0xFFFFFFE0      // N_pad = (N+31) & ~31
    str     w11, [sp, #60]
    lsl     x13, x3, #1               // lda * 2 (bf16 = 2 bytes)
    str     x13, [sp, #72]
    lsl     x14, x4, #1               // ldb * 2
    str     x14, [sp, #88]
    lsl     x16, x5, #2               // ldc * 4 (fp32 output)
    str     x16, [sp, #96]
    // ── Broadcast alpha/beta, save beta bits ──
    mov     z20.s, s20
    mov     z22.s, s22
    fmov    w17, s22
    str     w17, [sp, #104]            // beta_bits for later zero-check
    // ── Tile row loop ──
    mov     w0, #0
.L_bf_tile_row:
    str     w0, [sp, #80]
    mov     w1, #0
.L_bf_tile_col:
    str     w1, [sp, #84]
    // ── Phase 1: init accumulators (beta*C or zero) ──
    ldr     w17, [sp, #104]
    cbnz    w17, .L_bf_load_beta
    zero    {za0.s, za1.s}
    b       .L_bf_beta_done
.L_bf_load_beta:
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
    sub     w3, w14, w1
    mov     w4, #32
    cmp     w3, w4
    csel    w3, w3, w4, lt
    whilelt p2.s, xzr, x3
    sub     w4, w3, #16
    cmp     w4, #0
    csel    w4, wzr, w4, lt
    whilelt p3.s, xzr, x4
    sub     w15, w6, w0
    mov     w3, #16
    cmp     w15, w3
    csel    w15, w15, w3, lt
    mov     w12, #0
.L_bf_beta_grp:
    cmp     w12, w15
    b.ge    .L_bf_beta_done
    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0
    ld1w    {z0.s}, p2/z, [x8]
    ld1w    {z4.s}, p3/z, [x8, x9, lsl #2]
    fmul    z0.s, p0/m, z0.s, z22.s
    fmul    z4.s, p0/m, z4.s, z22.s
    add     w11, w12, #1
    cmp     w11, w15
    b.ge    .L_bf_beta_st
    add     x8, x8, x16
    ld1w    {z1.s}, p2/z, [x8]
    ld1w    {z5.s}, p3/z, [x8, x9, lsl #2]
    fmul    z1.s, p0/m, z1.s, z22.s
    fmul    z5.s, p0/m, z5.s, z22.s
    add     w11, w12, #2
    cmp     w11, w15
    b.ge    .L_bf_beta_st
    add     x8, x8, x16
    ld1w    {z2.s}, p2/z, [x8]
    ld1w    {z6.s}, p3/z, [x8, x9, lsl #2]
    fmul    z2.s, p0/m, z2.s, z22.s
    fmul    z6.s, p0/m, z6.s, z22.s
    add     w11, w12, #3
    cmp     w11, w15
    b.ge    .L_bf_beta_st
    add     x8, x8, x16
    ld1w    {z3.s}, p2/z, [x8]
    ld1w    {z7.s}, p3/z, [x8, x9, lsl #2]
    fmul    z3.s, p0/m, z3.s, z22.s
    fmul    z7.s, p0/m, z7.s, z22.s
    add     x8, x8, x16
.L_bf_beta_st:
    mova    za0h.s[w12, 0:3], {z0.s-z3.s}
    mova    za1h.s[w12, 0:3], {z4.s-z7.s}
    add     w12, w12, #4
    b       .L_bf_beta_grp
.L_bf_beta_done:
    // ── Phase 2: K-block accumulation ──
    ldr     w0, [sp, #80]             // ti
    ldr     w1, [sp, #84]             // tj
    ldr     x6, [sp, #0]              // A
    ldr     x7, [sp, #8]              // B
    ldr     w2, [sp, #32]             // K
    ldr     w18, [sp, #48]            // flags
    ldr     x17, [sp, #72]            // lda*2
    ldr     w3, [sp, #36]             // lda
    ldr     w4, [sp, #40]             // ldb
    ldr     x14, [sp, #88]            // ldb*2
    ldr     w15, [sp, #52]            // k_blocks
    ptrue   p0.s
    cntw    x9
    // transA=0: A_tile_base = A + ti * lda * 2
    // transA=1: A_tile_base = A + ti * 2
    tst     w18, #1
    b.ne    .L_bf_atbase_trans
    mul     w10, w0, w3
    add     x5, x6, x10, lsl #1       // A + ti*lda*2
    b       .L_bf_atbase_done
.L_bf_atbase_trans:
    add     x5, x6, x0, lsl #1        // A + ti*2
.L_bf_atbase_done:
    str     x5, [sp, #64]             // save A_tile_base
    mov     x13, xzr                   // k byte offset
    cbz     w15, .L_bf_kblock_done
.L_bf_kblock:
    // ── Load A tile (16 rows x 16 words = 32 bf16) into za2 ──
    zero    {za2.s}
    ldr     w18, [sp, #48]
    tst     w18, #1
    b.ne    .L_bf_load_a_trans
    // transA=0: row r: A[ti+r][k..k+31] via ld1w (16 words = 64 bytes = 32 bf16)
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
    b       .L_bf_a_loaded
.L_bf_load_a_trans:
    // transA=1: A is K x M. Load A[k+r][ti..ti+15] for r in 0..15
    lsr     x10, x13, #1              // k element index (bf16 = 2 bytes)
    mul     x11, x10, x17             // k * lda * 2
    add     x8, x5, x11               // A + ti*2 + k*lda*2 = &A[k][ti]
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
.L_bf_a_loaded:
    // ── Load B and BFMOPA ──
    // za2v column c gives 16 words = 32 bf16: pair r = (A[row_r][2c], A[row_r][2c+1])
    // BFMOPA 2:1 widening: za[i][j] += z_a[2i]*z_b[2j] + z_a[2i+1]*z_b[2j+1]
    ldr     w18, [sp, #48]
    tst     w18, #2
    b.ne    .L_bf_fmopa_transB
    // ── transB=0: B is K x N row-major bf16 ──
    // z_b pair j = (B[k_base+2c][tj+j], B[k_base+2c+1][tj+j])
    // Load 2 B rows, zip to interleave K-pairs
    lsr     x10, x13, #1              // k element index
    ldr     w1, [sp, #84]             // tj
    // Set up p4 = predicate for 16 bf16 elements (half of cnth=32)
    mov     x16, #16
    whilelt p4.h, xzr, x16
    // Cols 0-3
    mov     w12, #0
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    // z0 covers K-pair at column 0: A rows use k_base = k + 2*0 = k
    // B row addresses: B + (k+2*col)*ldb*2 + tj*2
    add     x11, x10, #0              // k + 2*0
    mul     x11, x11, x14             // * ldb*2
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1     // + tj*2
    mov     x3, x14                    // B row stride = ldb*2
    // z0: k_pair (k+0, k+1)
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z0.h, z4.h
    add     x8, x11, #32              // right 16 cols: tj+16
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z0.h, z5.h
    // z1: k_pair (k+2, k+3)
    add     x11, x10, #2
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z1.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z1.h, z5.h
    // z2: k_pair (k+4, k+5)
    add     x11, x10, #4
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z2.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z2.h, z5.h
    // z3: k_pair (k+6, k+7)
    add     x11, x10, #6
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z3.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z3.h, z5.h
    // Cols 4-7
    mov     w12, #4
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    add     x11, x10, #8
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z0.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z0.h, z5.h
    add     x11, x10, #10
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z1.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z1.h, z5.h
    add     x11, x10, #12
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z2.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z2.h, z5.h
    add     x11, x10, #14
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z3.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z3.h, z5.h
    // Cols 8-11
    mov     w12, #8
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    add     x11, x10, #16
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z0.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z0.h, z5.h
    add     x11, x10, #18
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z1.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z1.h, z5.h
    add     x11, x10, #20
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z2.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z2.h, z5.h
    add     x11, x10, #22
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z3.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z3.h, z5.h
    // Cols 12-15
    mov     w12, #12
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    add     x11, x10, #24
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z0.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z0.h, z5.h
    add     x11, x10, #26
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z1.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z1.h, z5.h
    add     x11, x10, #28
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z2.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z2.h, z5.h
    add     x11, x10, #30
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x1, lsl #1
    ld1h    {z8.h}, p4/z, [x11]
    add     x8, x11, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z4.h, z8.h, z9.h
    bfmopa  za0.s, p0/m, p0/m, z3.h, z4.h
    add     x8, x11, #32
    ld1h    {z8.h}, p4/z, [x8]
    add     x8, x8, x3
    ld1h    {z9.h}, p4/z, [x8]
    zip1    z5.h, z8.h, z9.h
    bfmopa  za1.s, p0/m, p0/m, z3.h, z5.h
    b       .L_bf_kblock_advance
    // ── transB=1: B is N x K bf16. B^T[k][j] = B[j][k] ──
    // Load B[tj..tj+15][k..k+31] into za3, transpose via vertical extract
.L_bf_fmopa_transB:
    // ── Left half: B[tj..tj+15][k..k+31] into za3 ──
    zero    {za3.s}
    lsr     x10, x13, #1              // k element index
    ldr     w1, [sp, #84]             // tj
    mul     x11, x1, x14              // tj * ldb * 2
    add     x11, x7, x11              // B + tj*ldb*2
    add     x11, x11, x10, lsl #1     // + k*2
    mov     x3, x14                    // row stride = ldb*2
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
    // BFMOPA left half into za0
    mov     w12, #0
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    bfmopa  za0.s, p0/m, p0/m, z0.h, z4.h
    bfmopa  za0.s, p0/m, p0/m, z1.h, z5.h
    bfmopa  za0.s, p0/m, p0/m, z2.h, z6.h
    bfmopa  za0.s, p0/m, p0/m, z3.h, z7.h
    mov     w12, #4
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    bfmopa  za0.s, p0/m, p0/m, z0.h, z4.h
    bfmopa  za0.s, p0/m, p0/m, z1.h, z5.h
    bfmopa  za0.s, p0/m, p0/m, z2.h, z6.h
    bfmopa  za0.s, p0/m, p0/m, z3.h, z7.h
    mov     w12, #8
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    bfmopa  za0.s, p0/m, p0/m, z0.h, z4.h
    bfmopa  za0.s, p0/m, p0/m, z1.h, z5.h
    bfmopa  za0.s, p0/m, p0/m, z2.h, z6.h
    bfmopa  za0.s, p0/m, p0/m, z3.h, z7.h
    mov     w12, #12
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    bfmopa  za0.s, p0/m, p0/m, z0.h, z4.h
    bfmopa  za0.s, p0/m, p0/m, z1.h, z5.h
    bfmopa  za0.s, p0/m, p0/m, z2.h, z6.h
    bfmopa  za0.s, p0/m, p0/m, z3.h, z7.h
    // ── Right half: B[tj+16..tj+31][k..k+31] into za3 ──
    zero    {za3.s}
    ldr     w1, [sp, #84]             // tj
    add     w11, w1, #16              // tj+16
    mul     x11, x11, x14             // (tj+16) * ldb * 2
    add     x11, x7, x11              // B + (tj+16)*ldb*2
    add     x11, x11, x10, lsl #1     // + k*2
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
    // BFMOPA right half into za1
    mov     w12, #0
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    bfmopa  za1.s, p0/m, p0/m, z0.h, z4.h
    bfmopa  za1.s, p0/m, p0/m, z1.h, z5.h
    bfmopa  za1.s, p0/m, p0/m, z2.h, z6.h
    bfmopa  za1.s, p0/m, p0/m, z3.h, z7.h
    mov     w12, #4
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    bfmopa  za1.s, p0/m, p0/m, z0.h, z4.h
    bfmopa  za1.s, p0/m, p0/m, z1.h, z5.h
    bfmopa  za1.s, p0/m, p0/m, z2.h, z6.h
    bfmopa  za1.s, p0/m, p0/m, z3.h, z7.h
    mov     w12, #8
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    bfmopa  za1.s, p0/m, p0/m, z0.h, z4.h
    bfmopa  za1.s, p0/m, p0/m, z1.h, z5.h
    bfmopa  za1.s, p0/m, p0/m, z2.h, z6.h
    bfmopa  za1.s, p0/m, p0/m, z3.h, z7.h
    mov     w12, #12
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    bfmopa  za1.s, p0/m, p0/m, z0.h, z4.h
    bfmopa  za1.s, p0/m, p0/m, z1.h, z5.h
    bfmopa  za1.s, p0/m, p0/m, z2.h, z6.h
    bfmopa  za1.s, p0/m, p0/m, z3.h, z7.h
.L_bf_kblock_advance:
    add     x13, x13, #64             // k byte offset += 32 bf16 * 2
    subs    w15, w15, #1
    b.ne    .L_bf_kblock
.L_bf_kblock_done:
    // ── Phase 3: Store alpha * ZA to C (identical to sgemm) ──
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
    sub     w3, w14, w1
    mov     w4, #32
    cmp     w3, w4
    csel    w3, w3, w4, lt
    whilelt p2.s, xzr, x3
    sub     w4, w3, #16
    cmp     w4, #0
    csel    w4, wzr, w4, lt
    whilelt p3.s, xzr, x4
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
    b.lt    .L_bf_store_end
    st1w    {z0.s}, p2, [x8]
    st1w    {z4.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #2
    b.lt    .L_bf_store_end
    st1w    {z1.s}, p2, [x8]
    st1w    {z5.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #3
    b.lt    .L_bf_store_end
    st1w    {z2.s}, p2, [x8]
    st1w    {z6.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #4
    b.lt    .L_bf_store_end
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
    b.lt    .L_bf_store_end
    st1w    {z0.s}, p2, [x8]
    st1w    {z4.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #6
    b.lt    .L_bf_store_end
    st1w    {z1.s}, p2, [x8]
    st1w    {z5.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #7
    b.lt    .L_bf_store_end
    st1w    {z2.s}, p2, [x8]
    st1w    {z6.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #8
    b.lt    .L_bf_store_end
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
    b.lt    .L_bf_store_end
    st1w    {z0.s}, p2, [x8]
    st1w    {z4.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #10
    b.lt    .L_bf_store_end
    st1w    {z1.s}, p2, [x8]
    st1w    {z5.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #11
    b.lt    .L_bf_store_end
    st1w    {z2.s}, p2, [x8]
    st1w    {z6.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #12
    b.lt    .L_bf_store_end
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
    b.lt    .L_bf_store_end
    st1w    {z0.s}, p2, [x8]
    st1w    {z4.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #14
    b.lt    .L_bf_store_end
    st1w    {z1.s}, p2, [x8]
    st1w    {z5.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #15
    b.lt    .L_bf_store_end
    st1w    {z2.s}, p2, [x8]
    st1w    {z6.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #16
    b.lt    .L_bf_store_end
    st1w    {z3.s}, p2, [x8]
    st1w    {z7.s}, p3, [x8, x9, lsl #2]
.L_bf_store_end:
    // ── Advance tile column ──
    ldr     w1, [sp, #84]
    ldr     w4, [sp, #60]             // N_pad
    add     w1, w1, #32
    cmp     w1, w4
    b.lt    .L_bf_tile_col
    // ── Advance tile row ──
    ldr     w0, [sp, #80]
    ldr     w3, [sp, #56]             // M_pad
    add     w0, w0, #16
    cmp     w0, w3
    b.lt    .L_bf_tile_row
    // ── Cleanup ──
    add     sp, sp, #128
    b       .L_dispatch
// ================================================================
// INTEGER GEMM MACRO — generates cblas_igemm / cblas_ugemm / cblas_usgemm
//
// C = alpha * cvtf(op(A_i8) @ op(B_i8)) + beta * C
//
// Encoding: [opcode][flags:u8][M:u32][N:u32][K:u32]
//           [lda:u32][ldb:u32][ldc:u32]
//           [alpha:f32][beta:f32]
//           [A_ptr:u64][B_ptr:u64][C_ptr:u64]
//
// flags bit 0: transA    flags bit 1: transB
// A, B are i8/u8; C is fp32. lda/ldb in byte elements, ldc in fp32 elements.
// K must be a multiple of 64 (byte elements).
//
// Tile geometry: 16 x 32 output (za0 = left 16 cols, za1 = right 16 cols)
// za2 = scratch for A tile, za3 = scratch for transB
// SMOPA/UMOPA/USMOPA 4:1 widening: 4 K-elements per output position per instruction
//
// Stack layout (128 bytes): identical to cblas_sgemm
//   [sp+0]:   A_ptr           [sp+8]:   B_ptr
//   [sp+16]:  C_ptr           [sp+24]:  M (w)     [sp+28]: N (w)
//   [sp+32]:  K (w)           [sp+36]:  lda (w)   [sp+40]: ldb (w)
//   [sp+44]:  ldc (w)         [sp+48]:  flags (w)
//   [sp+52]:  k_blocks (w)    [sp+56]:  M_pad (w) [sp+60]: N_pad (w)
//   [sp+64]:  A_tile_base (x) [sp+72]:  lda (x, byte stride)
//   [sp+80]:  ti (w)          [sp+84]:  tj (w)
//   [sp+88]:  ldb (x, byte stride) [sp+96]:  ldc*4 (x)
//   [sp+104]: beta_bits (w)
// z20 = alpha broadcast, z22 = beta broadcast
// ================================================================
.macro CBLAS_INTEGER_GEMM lbl, mopa_inst
.L_op_\lbl:
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
    lsr     w15, w2, #6               // k_blocks = K / 64
    str     w15, [sp, #52]
    add     w10, w0, #15
    and     w10, w10, #0xFFFFFFF0      // M_pad = (M+15) & ~15
    str     w10, [sp, #56]
    add     w11, w1, #31
    and     w11, w11, #0xFFFFFFE0      // N_pad = (N+31) & ~31
    str     w11, [sp, #60]
    sxtw    x13, w3                    // lda byte stride (i8 = 1 byte)
    str     x13, [sp, #72]
    sxtw    x14, w4                    // ldb byte stride
    str     x14, [sp, #88]
    lsl     x16, x5, #2               // ldc * 4 (fp32 output)
    str     x16, [sp, #96]
    // ── Broadcast alpha/beta, save beta bits ──
    mov     z20.s, s20
    mov     z22.s, s22
    fmov    w17, s22
    str     w17, [sp, #104]
    // ── Tile row loop ──
    mov     w0, #0
.L_\lbl\()_tile_row:
    str     w0, [sp, #80]
    mov     w1, #0
.L_\lbl\()_tile_col:
    str     w1, [sp, #84]
    // ── Phase 1: init accumulators (beta*C or zero) ──
    ldr     w17, [sp, #104]
    cbnz    w17, .L_\lbl\()_load_beta
    zero    {za0.s, za1.s}
    b       .L_\lbl\()_beta_done
.L_\lbl\()_load_beta:
    zero    {za0.s, za1.s}
    ldr     x8, [sp, #16]
    ldr     w0, [sp, #80]
    ldr     w1, [sp, #84]
    ldr     w14, [sp, #28]
    ldr     w5, [sp, #44]
    ldr     x16, [sp, #96]
    ldr     w6, [sp, #24]
    cntw    x9
    ptrue   p0.s
    mul     w10, w0, w5
    add     w10, w10, w1
    add     x8, x8, x10, lsl #2
    sub     w3, w14, w1
    mov     w4, #32
    cmp     w3, w4
    csel    w3, w3, w4, lt
    whilelt p2.s, xzr, x3
    sub     w4, w3, #16
    cmp     w4, #0
    csel    w4, wzr, w4, lt
    whilelt p3.s, xzr, x4
    sub     w15, w6, w0
    mov     w3, #16
    cmp     w15, w3
    csel    w15, w15, w3, lt
    mov     w12, #0
.L_\lbl\()_beta_grp:
    cmp     w12, w15
    b.ge    .L_\lbl\()_beta_done
    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0
    ld1w    {z0.s}, p2/z, [x8]
    ld1w    {z4.s}, p3/z, [x8, x9, lsl #2]
    fmul    z0.s, p0/m, z0.s, z22.s
    fmul    z4.s, p0/m, z4.s, z22.s
    add     w11, w12, #1
    cmp     w11, w15
    b.ge    .L_\lbl\()_beta_st
    add     x8, x8, x16
    ld1w    {z1.s}, p2/z, [x8]
    ld1w    {z5.s}, p3/z, [x8, x9, lsl #2]
    fmul    z1.s, p0/m, z1.s, z22.s
    fmul    z5.s, p0/m, z5.s, z22.s
    add     w11, w12, #2
    cmp     w11, w15
    b.ge    .L_\lbl\()_beta_st
    add     x8, x8, x16
    ld1w    {z2.s}, p2/z, [x8]
    ld1w    {z6.s}, p3/z, [x8, x9, lsl #2]
    fmul    z2.s, p0/m, z2.s, z22.s
    fmul    z6.s, p0/m, z6.s, z22.s
    add     w11, w12, #3
    cmp     w11, w15
    b.ge    .L_\lbl\()_beta_st
    add     x8, x8, x16
    ld1w    {z3.s}, p2/z, [x8]
    ld1w    {z7.s}, p3/z, [x8, x9, lsl #2]
    fmul    z3.s, p0/m, z3.s, z22.s
    fmul    z7.s, p0/m, z7.s, z22.s
    add     x8, x8, x16
.L_\lbl\()_beta_st:
    mova    za0h.s[w12, 0:3], {z0.s-z3.s}
    mova    za1h.s[w12, 0:3], {z4.s-z7.s}
    add     w12, w12, #4
    b       .L_\lbl\()_beta_grp
.L_\lbl\()_beta_done:
    // ── Phase 2: K-block accumulation ──
    ldr     w0, [sp, #80]
    ldr     w1, [sp, #84]
    ldr     x6, [sp, #0]
    ldr     x7, [sp, #8]
    ldr     w2, [sp, #32]
    ldr     w18, [sp, #48]
    ldr     x17, [sp, #72]            // lda (byte stride)
    ldr     w3, [sp, #36]
    ldr     w4, [sp, #40]
    ldr     x14, [sp, #88]            // ldb (byte stride)
    ldr     w15, [sp, #52]
    ptrue   p0.s
    cntw    x9
    // transA=0: A_tile_base = A + ti * lda
    // transA=1: A_tile_base = A + ti  (byte offset)
    tst     w18, #1
    b.ne    .L_\lbl\()_atbase_trans
    mul     w10, w0, w3
    sxtw    x10, w10
    add     x5, x6, x10               // A + ti*lda
    b       .L_\lbl\()_atbase_done
.L_\lbl\()_atbase_trans:
    add     x5, x6, x0                // A + ti (byte offset)
.L_\lbl\()_atbase_done:
    str     x5, [sp, #64]
    mov     x13, xzr                   // k byte offset
    cbz     w15, .L_\lbl\()_kblock_done
.L_\lbl\()_kblock:
    // ── Load A tile (16 rows x 16 words = 64 i8) into za2 ──
    zero    {za2.s}
    ldr     w18, [sp, #48]
    tst     w18, #1
    b.ne    .L_\lbl\()_load_a_trans
    // transA=0: row r: A[ti+r][k..k+63] via ld1w (16 words = 64 bytes)
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
    b       .L_\lbl\()_a_loaded
.L_\lbl\()_load_a_trans:
    // transA=1: A is K x M. Load A[k+r][ti..ti+15] for r in 0..15
    // Each row = 16 bytes starting at A + (k+r)*lda + ti
    // ld1w loads 64 bytes = 16 words; but we want 16 bytes spread across columns.
    // Since element size is 1 byte, ti offsets by 1 byte per column.
    // We need to load 64 bytes per za2 row, so we load A[k+r][ti..ti+63].
    // But for transA, the logical A^T has M columns, so this loads beyond M if M<64.
    // The za2v column extraction will only use the first 16 rows of output anyway.
    // Load: &A[k][ti] = A + k*lda + ti, stride = lda
    add     x8, x5, x13               // A + ti + k (k is byte offset, ti is byte offset)
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
.L_\lbl\()_a_loaded:
    // ── Load B and MOPA ──
    // za2v column c gives 16 words = 64 i8: group r = (A[row_r][4c..4c+3])
    // SMOPA/UMOPA/USMOPA 4:1 widening: za[i][j] += sum(d=0..3) z_a[4i+d]*z_b[4j+d]
    ldr     w18, [sp, #48]
    tst     w18, #2
    b.ne    .L_\lbl\()_mopa_transB
    // ── transB=0: B is K x N row-major i8 ──
    // M4 SMOPA workaround: SMOPA .b only uses the low byte of each .s group (d=0).
    // We compensate by calling SMOPA 4x per za2v column, shifting A data by d*8 bits
    // and loading each B row individually via ld1b {z.s} (byte-to-word widening).
    mov     x3, x14                    // B row stride = ldb
    mov     x16, #16                   // right-half column offset
.macro IGEMM_NOTRANSB_COL4 col_base
    mov     w12, #\col_base
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    // Process z0 (za2v column col_base, k_offset = col_base * 4)
    ldr     x11, [sp, #8]             // B base
    add     x10, x13, #(\col_base * 4)
    madd    x11, x10, x14, x11        // B + (k + col_base*4) * ldb
    ldr     w1, [sp, #84]
    add     x11, x11, x1              // + tj → B[k+col_base*4][tj]
    // d=0: z_a = z0 as-is (low byte = A[i][4*col_base+0])
    ld1b    {z4.s}, p0/z, [x11]       // left 16 cols, byte→word
    \mopa_inst za0.s, p0/m, p0/m, z0.b, z4.b
    add     x8, x11, x16
    ld1b    {z4.s}, p0/z, [x8]        // right 16 cols
    \mopa_inst za1.s, p0/m, p0/m, z0.b, z4.b
    // d=1: shift z0 right by 8 to expose byte 1
    lsr     z5.s, z0.s, #8
    add     x8, x11, x3
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za0.s, p0/m, p0/m, z5.b, z4.b
    add     x8, x8, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z5.b, z4.b
    // d=2
    lsr     z5.s, z0.s, #16
    add     x8, x11, x3, lsl #1       // x11 + 2*ldb
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za0.s, p0/m, p0/m, z5.b, z4.b
    add     x8, x8, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z5.b, z4.b
    // d=3
    lsr     z5.s, z0.s, #24
    add     x8, x11, x3, lsl #1       // x11 + 2*ldb
    add     x8, x8, x3                // x11 + 3*ldb
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za0.s, p0/m, p0/m, z5.b, z4.b
    add     x8, x8, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z5.b, z4.b
    // Process z1 (za2v column col_base+1, k_offset = col_base*4 + 4)
    ldr     x11, [sp, #8]
    add     x10, x13, #(\col_base * 4 + 4)
    madd    x11, x10, x14, x11
    ldr     w1, [sp, #84]
    add     x11, x11, x1
    ld1b    {z4.s}, p0/z, [x11]
    \mopa_inst za0.s, p0/m, p0/m, z1.b, z4.b
    add     x8, x11, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z1.b, z4.b
    lsr     z5.s, z1.s, #8
    add     x8, x11, x3
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za0.s, p0/m, p0/m, z5.b, z4.b
    add     x8, x8, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z5.b, z4.b
    lsr     z5.s, z1.s, #16
    add     x8, x11, x3, lsl #1
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za0.s, p0/m, p0/m, z5.b, z4.b
    add     x8, x8, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z5.b, z4.b
    lsr     z5.s, z1.s, #24
    add     x8, x11, x3, lsl #1
    add     x8, x8, x3
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za0.s, p0/m, p0/m, z5.b, z4.b
    add     x8, x8, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z5.b, z4.b
    // Process z2 (za2v column col_base+2, k_offset = col_base*4 + 8)
    ldr     x11, [sp, #8]
    add     x10, x13, #(\col_base * 4 + 8)
    madd    x11, x10, x14, x11
    ldr     w1, [sp, #84]
    add     x11, x11, x1
    ld1b    {z4.s}, p0/z, [x11]
    \mopa_inst za0.s, p0/m, p0/m, z2.b, z4.b
    add     x8, x11, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z2.b, z4.b
    lsr     z5.s, z2.s, #8
    add     x8, x11, x3
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za0.s, p0/m, p0/m, z5.b, z4.b
    add     x8, x8, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z5.b, z4.b
    lsr     z5.s, z2.s, #16
    add     x8, x11, x3, lsl #1
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za0.s, p0/m, p0/m, z5.b, z4.b
    add     x8, x8, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z5.b, z4.b
    lsr     z5.s, z2.s, #24
    add     x8, x11, x3, lsl #1
    add     x8, x8, x3
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za0.s, p0/m, p0/m, z5.b, z4.b
    add     x8, x8, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z5.b, z4.b
    // Process z3 (za2v column col_base+3, k_offset = col_base*4 + 12)
    ldr     x11, [sp, #8]
    add     x10, x13, #(\col_base * 4 + 12)
    madd    x11, x10, x14, x11
    ldr     w1, [sp, #84]
    add     x11, x11, x1
    ld1b    {z4.s}, p0/z, [x11]
    \mopa_inst za0.s, p0/m, p0/m, z3.b, z4.b
    add     x8, x11, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z3.b, z4.b
    lsr     z5.s, z3.s, #8
    add     x8, x11, x3
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za0.s, p0/m, p0/m, z5.b, z4.b
    add     x8, x8, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z5.b, z4.b
    lsr     z5.s, z3.s, #16
    add     x8, x11, x3, lsl #1
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za0.s, p0/m, p0/m, z5.b, z4.b
    add     x8, x8, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z5.b, z4.b
    lsr     z5.s, z3.s, #24
    add     x8, x11, x3, lsl #1
    add     x8, x8, x3
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za0.s, p0/m, p0/m, z5.b, z4.b
    add     x8, x8, x16
    ld1b    {z4.s}, p0/z, [x8]
    \mopa_inst za1.s, p0/m, p0/m, z5.b, z4.b
.endm
    IGEMM_NOTRANSB_COL4 0
    IGEMM_NOTRANSB_COL4 4
    IGEMM_NOTRANSB_COL4 8
    IGEMM_NOTRANSB_COL4 12
.purgem IGEMM_NOTRANSB_COL4
    b       .L_\lbl\()_kblock_advance
    // ── transB=1: B is N x K i8. B^T[k][j] = B[j][k] ──
    // Load B[tj..tj+15][k..k+63] into za3, transpose via vertical extract
.L_\lbl\()_mopa_transB:
    // ── Left half: B[tj..tj+15][k..k+63] into za3 ──
    zero    {za3.s}
    ldr     w1, [sp, #84]             // tj
    mul     x11, x1, x14              // tj * ldb
    add     x11, x7, x11              // B + tj*ldb
    add     x11, x11, x13             // + k byte offset
    mov     x3, x14                    // row stride = ldb
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
    // MOPA left half into za0
    mov     w12, #0
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    \mopa_inst za0.s, p0/m, p0/m, z0.b, z4.b
    \mopa_inst za0.s, p0/m, p0/m, z1.b, z5.b
    \mopa_inst za0.s, p0/m, p0/m, z2.b, z6.b
    \mopa_inst za0.s, p0/m, p0/m, z3.b, z7.b
    mov     w12, #4
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    \mopa_inst za0.s, p0/m, p0/m, z0.b, z4.b
    \mopa_inst za0.s, p0/m, p0/m, z1.b, z5.b
    \mopa_inst za0.s, p0/m, p0/m, z2.b, z6.b
    \mopa_inst za0.s, p0/m, p0/m, z3.b, z7.b
    mov     w12, #8
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    \mopa_inst za0.s, p0/m, p0/m, z0.b, z4.b
    \mopa_inst za0.s, p0/m, p0/m, z1.b, z5.b
    \mopa_inst za0.s, p0/m, p0/m, z2.b, z6.b
    \mopa_inst za0.s, p0/m, p0/m, z3.b, z7.b
    mov     w12, #12
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    \mopa_inst za0.s, p0/m, p0/m, z0.b, z4.b
    \mopa_inst za0.s, p0/m, p0/m, z1.b, z5.b
    \mopa_inst za0.s, p0/m, p0/m, z2.b, z6.b
    \mopa_inst za0.s, p0/m, p0/m, z3.b, z7.b
    // ── Right half: B[tj+16..tj+31][k..k+63] into za3 ──
    zero    {za3.s}
    ldr     w1, [sp, #84]
    add     w11, w1, #16
    sxtw    x11, w11
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x13
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
    // MOPA right half into za1
    mov     w12, #0
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    \mopa_inst za1.s, p0/m, p0/m, z0.b, z4.b
    \mopa_inst za1.s, p0/m, p0/m, z1.b, z5.b
    \mopa_inst za1.s, p0/m, p0/m, z2.b, z6.b
    \mopa_inst za1.s, p0/m, p0/m, z3.b, z7.b
    mov     w12, #4
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    \mopa_inst za1.s, p0/m, p0/m, z0.b, z4.b
    \mopa_inst za1.s, p0/m, p0/m, z1.b, z5.b
    \mopa_inst za1.s, p0/m, p0/m, z2.b, z6.b
    \mopa_inst za1.s, p0/m, p0/m, z3.b, z7.b
    mov     w12, #8
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    \mopa_inst za1.s, p0/m, p0/m, z0.b, z4.b
    \mopa_inst za1.s, p0/m, p0/m, z1.b, z5.b
    \mopa_inst za1.s, p0/m, p0/m, z2.b, z6.b
    \mopa_inst za1.s, p0/m, p0/m, z3.b, z7.b
    mov     w12, #12
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    \mopa_inst za1.s, p0/m, p0/m, z0.b, z4.b
    \mopa_inst za1.s, p0/m, p0/m, z1.b, z5.b
    \mopa_inst za1.s, p0/m, p0/m, z2.b, z6.b
    \mopa_inst za1.s, p0/m, p0/m, z3.b, z7.b
.L_\lbl\()_kblock_advance:
    add     x13, x13, #64             // k byte offset += 64 i8
    subs    w15, w15, #1
    b.ne    .L_\lbl\()_kblock
.L_\lbl\()_kblock_done:
    // ── Phase 3: Store scvtf(accum) * alpha + (beta already folded into za) to C ──
    ldr     x8, [sp, #16]
    ldr     w0, [sp, #80]
    ldr     w1, [sp, #84]
    ldr     w14, [sp, #28]
    ldr     w5, [sp, #44]
    ldr     x10, [sp, #96]
    ldr     w6, [sp, #24]
    ptrue   p0.s
    cntw    x9
    mul     w11, w0, w5
    add     w11, w11, w1
    add     x8, x8, x11, lsl #2
    sub     w3, w14, w1
    mov     w4, #32
    cmp     w3, w4
    csel    w3, w3, w4, lt
    whilelt p2.s, xzr, x3
    sub     w4, w3, #16
    cmp     w4, #0
    csel    w4, wzr, w4, lt
    whilelt p3.s, xzr, x4
    sub     w15, w6, w0
    mov     w3, #16
    cmp     w15, w3
    csel    w15, w15, w3, lt
    // Group 0 (rows 0-3): extract int32, scvtf, fmul alpha
    mov     w12, #0
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s
    scvtf   z4.s, p0/m, z4.s
    scvtf   z5.s, p0/m, z5.s
    scvtf   z6.s, p0/m, z6.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z0.s, z0.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z3.s, z3.s, z20.s
    fmul    z4.s, z4.s, z20.s
    fmul    z5.s, z5.s, z20.s
    fmul    z6.s, z6.s, z20.s
    fmul    z7.s, z7.s, z20.s
    cmp     w15, #1
    b.lt    .L_\lbl\()_store_end
    st1w    {z0.s}, p2, [x8]
    st1w    {z4.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #2
    b.lt    .L_\lbl\()_store_end
    st1w    {z1.s}, p2, [x8]
    st1w    {z5.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #3
    b.lt    .L_\lbl\()_store_end
    st1w    {z2.s}, p2, [x8]
    st1w    {z6.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #4
    b.lt    .L_\lbl\()_store_end
    st1w    {z3.s}, p2, [x8]
    st1w    {z7.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    // Group 1 (rows 4-7)
    mov     w12, #4
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s
    scvtf   z4.s, p0/m, z4.s
    scvtf   z5.s, p0/m, z5.s
    scvtf   z6.s, p0/m, z6.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z0.s, z0.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z3.s, z3.s, z20.s
    fmul    z4.s, z4.s, z20.s
    fmul    z5.s, z5.s, z20.s
    fmul    z6.s, z6.s, z20.s
    fmul    z7.s, z7.s, z20.s
    cmp     w15, #5
    b.lt    .L_\lbl\()_store_end
    st1w    {z0.s}, p2, [x8]
    st1w    {z4.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #6
    b.lt    .L_\lbl\()_store_end
    st1w    {z1.s}, p2, [x8]
    st1w    {z5.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #7
    b.lt    .L_\lbl\()_store_end
    st1w    {z2.s}, p2, [x8]
    st1w    {z6.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #8
    b.lt    .L_\lbl\()_store_end
    st1w    {z3.s}, p2, [x8]
    st1w    {z7.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    // Group 2 (rows 8-11)
    mov     w12, #8
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s
    scvtf   z4.s, p0/m, z4.s
    scvtf   z5.s, p0/m, z5.s
    scvtf   z6.s, p0/m, z6.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z0.s, z0.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z3.s, z3.s, z20.s
    fmul    z4.s, z4.s, z20.s
    fmul    z5.s, z5.s, z20.s
    fmul    z6.s, z6.s, z20.s
    fmul    z7.s, z7.s, z20.s
    cmp     w15, #9
    b.lt    .L_\lbl\()_store_end
    st1w    {z0.s}, p2, [x8]
    st1w    {z4.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #10
    b.lt    .L_\lbl\()_store_end
    st1w    {z1.s}, p2, [x8]
    st1w    {z5.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #11
    b.lt    .L_\lbl\()_store_end
    st1w    {z2.s}, p2, [x8]
    st1w    {z6.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #12
    b.lt    .L_\lbl\()_store_end
    st1w    {z3.s}, p2, [x8]
    st1w    {z7.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    // Group 3 (rows 12-15)
    mov     w12, #12
    mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s
    scvtf   z4.s, p0/m, z4.s
    scvtf   z5.s, p0/m, z5.s
    scvtf   z6.s, p0/m, z6.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z0.s, z0.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z3.s, z3.s, z20.s
    fmul    z4.s, z4.s, z20.s
    fmul    z5.s, z5.s, z20.s
    fmul    z6.s, z6.s, z20.s
    fmul    z7.s, z7.s, z20.s
    cmp     w15, #13
    b.lt    .L_\lbl\()_store_end
    st1w    {z0.s}, p2, [x8]
    st1w    {z4.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #14
    b.lt    .L_\lbl\()_store_end
    st1w    {z1.s}, p2, [x8]
    st1w    {z5.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #15
    b.lt    .L_\lbl\()_store_end
    st1w    {z2.s}, p2, [x8]
    st1w    {z6.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #16
    b.lt    .L_\lbl\()_store_end
    st1w    {z3.s}, p2, [x8]
    st1w    {z7.s}, p3, [x8, x9, lsl #2]
.L_\lbl\()_store_end:
    // ── Advance tile column ──
    ldr     w1, [sp, #84]
    ldr     w4, [sp, #60]
    add     w1, w1, #32
    cmp     w1, w4
    b.lt    .L_\lbl\()_tile_col
    // ── Advance tile row ──
    ldr     w0, [sp, #80]
    ldr     w3, [sp, #56]
    add     w0, w0, #16
    cmp     w0, w3
    b.lt    .L_\lbl\()_tile_row
    // ── Cleanup ──
    add     sp, sp, #128
    b       .L_dispatch
.endm
// ── Instantiate the three integer GEMM variants ──
    CBLAS_INTEGER_GEMM cblas_igemm, smopa
    CBLAS_INTEGER_GEMM cblas_ugemm, umopa
    CBLAS_INTEGER_GEMM cblas_usgemm, usmopa
// ================================================================
// GEMM_TILE_FP32 (0x64) -- Tile-range fp32 GEMM via FMOPA
//
// Same inner kernel as cblas_sgemm but only computes output tiles in
// the range [ti_start..ti_start+ti_count, tj_start..tj_start+tj_count].
// This enables multi-threaded (assign tile ranges to threads) and
// multi-node (assign tile ranges matching parameter shards) execution.
//
// Encoding: [0x64][flags:u8][M:u32][N:u32][K:u32]
//           [lda:u32][ldb:u32][ldc:u32]
//           [alpha:f32][beta:f32]
//           [ti_start:u32][tj_start:u32][ti_count:u32][tj_count:u32]
//           [A_ptr:u64][B_ptr:u64][C_ptr:u64]
//
// flags bit 0: transA    flags bit 1: transB
// M, N are the FULL matrix dimensions (used for edge predication).
// ti_start, tj_start: starting row/column (should be multiples of 16/32).
// ti_count, tj_count: number of rows/columns to process from the start.
// Total immediate payload after opcode: 73 bytes
//
// Tile geometry: identical to cblas_sgemm (16 x 32 output tiles)
// Stack layout (128 bytes):
//   [sp+0]:   A_ptr           [sp+8]:   B_ptr
//   [sp+16]:  C_ptr           [sp+24]:  M (w)     [sp+28]: N (w)
//   [sp+32]:  K (w)           [sp+36]:  lda (w)   [sp+40]: ldb (w)
//   [sp+44]:  ldc (w)         [sp+48]:  flags (w)
//   [sp+52]:  k_blocks (w)    [sp+56]:  ti_end (w) [sp+60]: tj_end (w)
//   [sp+64]:  A_tile_base (x) [sp+72]:  lda*4 (x)
//   [sp+80]:  ti (w)          [sp+84]:  tj (w)
//   [sp+88]:  ldb*4 (x)       [sp+96]:  ldc*4 (x)
//   [sp+104]: beta_bits (w)   [sp+108]: tj_start (w)
// z20 = alpha broadcast, z22 = beta broadcast
// ================================================================
.L_op_gemm_tile_fp32:
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
    ldr     w20, [x19, #32]           // ti_start
    ldr     w21, [x19, #36]           // tj_start
    ldr     w22, [x19, #40]           // ti_count
    ldr     w23, [x19, #44]           // tj_count
    add     x19, x19, #48
    ldr     x6, [x19], #8             // A_ptr
    ldr     x7, [x19], #8             // B_ptr
    ldr     x8, [x19], #8             // C_ptr
    // ── Allocate stack frame ──
    sub     sp, sp, #128
    stp     x6, x7, [sp, #0]
    str     x8, [sp, #16]
    stp     w0, w1, [sp, #24]
    str     w2, [sp, #32]
    stp     w3, w4, [sp, #36]
    stp     w5, w18, [sp, #44]
    str     w21, [sp, #108]            // save tj_start
    // ── Derived values ──
    ptrue   p0.s
    cntw    x9
    lsr     w15, w2, #4               // k_blocks = K / 16
    str     w15, [sp, #52]
    // ti_end = ti_start + ((ti_count + 15) & ~15)
    add     w10, w22, #15
    and     w10, w10, #0xFFFFFFF0
    add     w10, w20, w10              // ti_end = ti_start + padded ti_count
    str     w10, [sp, #56]
    // tj_end = tj_start + ((tj_count + 31) & ~31)
    add     w11, w23, #31
    and     w11, w11, #0xFFFFFFE0
    add     w11, w21, w11              // tj_end = tj_start + padded tj_count
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
    str     w17, [sp, #104]
    // ── Tile row loop (starts at ti_start) ──
    mov     w0, w20                    // w0 = ti_start
.L_gt_tile_row:
    str     w0, [sp, #80]
    ldr     w1, [sp, #108]             // w1 = tj_start
.L_gt_tile_col:
    str     w1, [sp, #84]
    // ── Phase 1: init accumulators (beta*C or zero) ──
    ldr     w17, [sp, #104]
    cbnz    w17, .L_gt_load_beta
    zero    {za0.s, za1.s}
    b       .L_gt_beta_done
.L_gt_load_beta:
    zero    {za0.s, za1.s}
    ldr     x8, [sp, #16]
    ldr     w0, [sp, #80]
    ldr     w1, [sp, #84]
    ldr     w14, [sp, #28]
    ldr     w5, [sp, #44]
    ldr     x16, [sp, #96]
    ldr     w6, [sp, #24]
    cntw    x9
    ptrue   p0.s
    mul     w10, w0, w5
    add     w10, w10, w1
    add     x8, x8, x10, lsl #2
    sub     w3, w14, w1
    mov     w4, #32
    cmp     w3, w4
    csel    w3, w3, w4, lt
    whilelt p2.s, xzr, x3
    sub     w4, w3, #16
    cmp     w4, #0
    csel    w4, wzr, w4, lt
    whilelt p3.s, xzr, x4
    sub     w15, w6, w0
    mov     w3, #16
    cmp     w15, w3
    csel    w15, w15, w3, lt
    mov     w12, #0
.L_gt_beta_grp:
    cmp     w12, w15
    b.ge    .L_gt_beta_done
    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0
    ld1w    {z0.s}, p2/z, [x8]
    ld1w    {z4.s}, p3/z, [x8, x9, lsl #2]
    fmul    z0.s, p0/m, z0.s, z22.s
    fmul    z4.s, p0/m, z4.s, z22.s
    add     w11, w12, #1
    cmp     w11, w15
    b.ge    .L_gt_beta_st
    add     x8, x8, x16
    ld1w    {z1.s}, p2/z, [x8]
    ld1w    {z5.s}, p3/z, [x8, x9, lsl #2]
    fmul    z1.s, p0/m, z1.s, z22.s
    fmul    z5.s, p0/m, z5.s, z22.s
    add     w11, w12, #2
    cmp     w11, w15
    b.ge    .L_gt_beta_st
    add     x8, x8, x16
    ld1w    {z2.s}, p2/z, [x8]
    ld1w    {z6.s}, p3/z, [x8, x9, lsl #2]
    fmul    z2.s, p0/m, z2.s, z22.s
    fmul    z6.s, p0/m, z6.s, z22.s
    add     w11, w12, #3
    cmp     w11, w15
    b.ge    .L_gt_beta_st
    add     x8, x8, x16
    ld1w    {z3.s}, p2/z, [x8]
    ld1w    {z7.s}, p3/z, [x8, x9, lsl #2]
    fmul    z3.s, p0/m, z3.s, z22.s
    fmul    z7.s, p0/m, z7.s, z22.s
    add     x8, x8, x16
.L_gt_beta_st:
    mova    za0h.s[w12, 0:3], {z0.s-z3.s}
    mova    za1h.s[w12, 0:3], {z4.s-z7.s}
    add     w12, w12, #4
    b       .L_gt_beta_grp
.L_gt_beta_done:
    // ── Phase 2: K-block accumulation (identical to sgemm) ──
    ldr     w0, [sp, #80]
    ldr     w1, [sp, #84]
    ldr     x6, [sp, #0]
    ldr     x7, [sp, #8]
    ldr     w2, [sp, #32]
    ldr     w18, [sp, #48]
    ldr     x17, [sp, #72]
    ldr     w3, [sp, #36]
    ldr     w4, [sp, #40]
    ldr     x14, [sp, #88]
    ldr     w15, [sp, #52]
    ptrue   p0.s
    cntw    x9
    tst     w18, #1
    b.ne    .L_gt_atbase_trans
    mul     w10, w0, w3
    add     x5, x6, x10, lsl #2
    b       .L_gt_atbase_done
.L_gt_atbase_trans:
    add     x5, x6, x0, lsl #2
.L_gt_atbase_done:
    str     x5, [sp, #64]
    mov     x13, xzr
    cbz     w15, .L_gt_kblock_done
.L_gt_kblock:
    // ── Load A tile into za2 ──
    zero    {za2.s}
    ldr     w18, [sp, #48]
    tst     w18, #1
    b.ne    .L_gt_load_a_trans
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
    b       .L_gt_a_loaded
.L_gt_load_a_trans:
    lsr     x10, x13, #2
    mul     x11, x10, x17
    add     x8, x5, x11
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
.L_gt_a_loaded:
    // ── Load B rows and FMOPA ──
    ldr     w18, [sp, #48]
    tst     w18, #2
    b.ne    .L_gt_fmopa_transB
    // ── transB=0 ──
    lsr     x10, x13, #2
    mul     x11, x10, x14
    add     x11, x7, x11
    ldr     w1, [sp, #84]
    add     x11, x11, x1, lsl #2
    mov     x3, x14
.macro GT_FMOPA_NOTRANSB_COL4 col_base
    mov     w12, #\col_base
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
.endm
    GT_FMOPA_NOTRANSB_COL4 0
    GT_FMOPA_NOTRANSB_COL4 4
    GT_FMOPA_NOTRANSB_COL4 8
    GT_FMOPA_NOTRANSB_COL4 12
.purgem GT_FMOPA_NOTRANSB_COL4
    b       .L_gt_kblock_advance
    // ── transB=1 ──
.L_gt_fmopa_transB:
    // Left half: B[tj..tj+15][k..k+15] into za3
    zero    {za3.s}
    lsr     x10, x13, #2
    ldr     w1, [sp, #84]
    mul     x11, x1, x14
    add     x11, x7, x11
    add     x11, x11, x10, lsl #2
    mov     x3, x14
.macro GT_LOAD_ZA3_16ROWS
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
.endm
    GT_LOAD_ZA3_16ROWS
.macro GT_FMOPA_TRANSB_HALF za_dst
    mov     w12, #0
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    fmopa   \za_dst\().s, p0/m, p0/m, z0.s, z4.s
    fmopa   \za_dst\().s, p0/m, p0/m, z1.s, z5.s
    fmopa   \za_dst\().s, p0/m, p0/m, z2.s, z6.s
    fmopa   \za_dst\().s, p0/m, p0/m, z3.s, z7.s
    mov     w12, #4
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    fmopa   \za_dst\().s, p0/m, p0/m, z0.s, z4.s
    fmopa   \za_dst\().s, p0/m, p0/m, z1.s, z5.s
    fmopa   \za_dst\().s, p0/m, p0/m, z2.s, z6.s
    fmopa   \za_dst\().s, p0/m, p0/m, z3.s, z7.s
    mov     w12, #8
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    fmopa   \za_dst\().s, p0/m, p0/m, z0.s, z4.s
    fmopa   \za_dst\().s, p0/m, p0/m, z1.s, z5.s
    fmopa   \za_dst\().s, p0/m, p0/m, z2.s, z6.s
    fmopa   \za_dst\().s, p0/m, p0/m, z3.s, z7.s
    mov     w12, #12
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za3v.s[w12, 0:3]
    fmopa   \za_dst\().s, p0/m, p0/m, z0.s, z4.s
    fmopa   \za_dst\().s, p0/m, p0/m, z1.s, z5.s
    fmopa   \za_dst\().s, p0/m, p0/m, z2.s, z6.s
    fmopa   \za_dst\().s, p0/m, p0/m, z3.s, z7.s
.endm
    GT_FMOPA_TRANSB_HALF za0
    // Right half: B[tj+16..tj+31][k..k+15] into za3
    zero    {za3.s}
    ldr     w1, [sp, #84]
    add     w11, w1, #16
    mul     x11, x11, x14
    add     x11, x7, x11
    add     x11, x11, x10, lsl #2
    mov     x3, x14
    GT_LOAD_ZA3_16ROWS
    GT_FMOPA_TRANSB_HALF za1
.purgem GT_LOAD_ZA3_16ROWS
.purgem GT_FMOPA_TRANSB_HALF
.L_gt_kblock_advance:
    add     x13, x13, #64
    subs    w15, w15, #1
    b.ne    .L_gt_kblock
.L_gt_kblock_done:
    // ── Phase 3: Store alpha * ZA to C ──
    ldr     x8, [sp, #16]
    ldr     w0, [sp, #80]
    ldr     w1, [sp, #84]
    ldr     w14, [sp, #28]
    ldr     w5, [sp, #44]
    ldr     x10, [sp, #96]
    ldr     w6, [sp, #24]
    ptrue   p0.s
    cntw    x9
    mul     w11, w0, w5
    add     w11, w11, w1
    add     x8, x8, x11, lsl #2
    sub     w3, w14, w1
    mov     w4, #32
    cmp     w3, w4
    csel    w3, w3, w4, lt
    whilelt p2.s, xzr, x3
    sub     w4, w3, #16
    cmp     w4, #0
    csel    w4, wzr, w4, lt
    whilelt p3.s, xzr, x4
    sub     w15, w6, w0
    mov     w3, #16
    cmp     w15, w3
    csel    w15, w15, w3, lt
.macro GT_STORE_GROUP grp_base
    mov     w12, #\grp_base
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
    cmp     w15, #(\grp_base + 1)
    b.lt    .L_gt_store_end
    st1w    {z0.s}, p2, [x8]
    st1w    {z4.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #(\grp_base + 2)
    b.lt    .L_gt_store_end
    st1w    {z1.s}, p2, [x8]
    st1w    {z5.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #(\grp_base + 3)
    b.lt    .L_gt_store_end
    st1w    {z2.s}, p2, [x8]
    st1w    {z6.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
    cmp     w15, #(\grp_base + 4)
    b.lt    .L_gt_store_end
    st1w    {z3.s}, p2, [x8]
    st1w    {z7.s}, p3, [x8, x9, lsl #2]
    add     x8, x8, x10
.endm
    GT_STORE_GROUP 0
    GT_STORE_GROUP 4
    GT_STORE_GROUP 8
    GT_STORE_GROUP 12
.purgem GT_STORE_GROUP
.L_gt_store_end:
    // ── Advance tile column ──
    ldr     w1, [sp, #84]
    ldr     w4, [sp, #60]             // tj_end
    add     w1, w1, #32
    cmp     w1, w4
    b.lt    .L_gt_tile_col
    // ── Advance tile row ──
    ldr     w0, [sp, #80]
    ldr     w3, [sp, #56]             // ti_end
    add     w0, w0, #16
    cmp     w0, w3
    b.lt    .L_gt_tile_row
    // ── Cleanup ──
    add     sp, sp, #128
    b       .L_dispatch
// ================================================================
// SOFTMAX_PARTIAL_FP32 (0x65) — Partial softmax for cross-shard merging
//
// Encoding: [0x65][dim:u32][in_ptr:u64][out_ptr:u64][max_ptr:u64][sum_ptr:u64]
//
// Pass 1: find local_max = max(in[0..dim-1])
// Pass 2: out[i] = exp(in[i] - local_max), local_sum = sum(out[0..dim-1])
// Stores local_max to *max_ptr and local_sum to *sum_ptr.
// dim must be a multiple of cntw (16).
// ================================================================
.L_op_softmax_partial_fp32:
    ldr     w22, [x19]             // dim
    add     x19, x19, #4
    ldr     x8, [x19], #8         // input_ptr
    ldr     x11, [x19], #8        // output_ptr
    ldr     x12, [x19], #8        // max_ptr
    ldr     x13, [x19], #8        // sum_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    // ── Load exp polynomial constants ──
    movz    w4, #0x3FB8, lsl #16
    movk    w4, #0xAA3B
    fmov    s28, w4
    mov     z28.s, s28             // log2(e)
    movz    w4, #0x3C1D, lsl #16
    movk    w4, #0x955A
    fmov    s29, w4
    mov     z29.s, s29             // c4
    movz    w4, #0x3D63, lsl #16
    movk    w4, #0x5847
    fmov    s30, w4
    mov     z30.s, s30             // c3
    movz    w4, #0x3E75, lsl #16
    movk    w4, #0xFDF0
    fmov    s31, w4
    mov     z31.s, s31             // c2
    movz    w4, #0x3F31, lsl #16
    movk    w4, #0x7218
    fmov    s27, w4
    mov     z27.s, s27             // c1 = ln(2)
    fmov    z26.s, #1.0            // c0 = 1.0
    mov     x14, x8               // save input_ptr
    mov     x15, x11              // save output_ptr
    // ── Pass 1: find max ──
    movz    w4, #0xFF80, lsl #16   // -inf
    fmov    s16, w4
    mov     z16.s, s16
    mov     w10, w22
.L_sp_max:
    ld1w    {z0.s}, p0/z, [x8]
    fmax    z16.s, p0/m, z16.s, z0.s
    add     x8, x8, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_sp_max
    fmaxv   s16, p0, z16.s
    mov     z16.s, s16
    // ── Store max ──
    str     s16, [x12]
    // ── Pass 2: exp(x - max) + accumulate sum ──
    fmov    z17.s, #0.0
    mov     x8, x14
    mov     x11, x15
    mov     w10, w22
.L_sp_exp:
    ld1w    {z0.s}, p0/z, [x8]
    fsub    z0.s, z0.s, z16.s
    fmul    z1.s, z0.s, z28.s     // x * log2(e)
    frintm  z2.s, p0/m, z1.s      // n = floor
    fsub    z3.s, z1.s, z2.s      // frac
    fmul    z4.s, z29.s, z3.s     // Horner: c4*f
    fadd    z4.s, z4.s, z30.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z31.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z27.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z26.s     // poly(frac)
    fcvtzs  z5.s, p0/m, z2.s
    mov     z6.s, #-127
    smax    z5.s, p0/m, z5.s, z6.s
    add     z5.s, z5.s, #127
    lsl     z5.s, z5.s, #23       // 2^n as IEEE bits
    fmul    z4.s, z4.s, z5.s      // exp(x - max)
    st1w    {z4.s}, p0, [x11]
    fadd    z17.s, z17.s, z4.s
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_sp_exp
    // ── Store sum ──
    faddv   s17, p0, z17.s
    str     s17, [x13]
    b       .L_dispatch
// ================================================================
// SOFTMAX_CORRECT_FP32 (0x66) — Apply max-correction to partial softmax
//
// Encoding: [0x66][dim:u32][local_max:f32][global_max:f32]
//           [inout_ptr:u64][sum_ptr:u64]
//
// correction = exp(local_max - global_max)
// For each i: inout[i] *= correction
// *sum_ptr *= correction
// dim must be a multiple of cntw (16).
// ================================================================
.L_op_softmax_correct_fp32:
    ldr     w22, [x19]             // dim
    ldr     s0, [x19, #4]         // local_max
    ldr     s1, [x19, #8]         // global_max
    add     x19, x19, #12
    ldr     x8, [x19], #8         // inout_ptr
    ldr     x11, [x19], #8        // sum_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    // ── Compute correction = exp(local_max - global_max) ──
    fsub    s2, s0, s1             // local_max - global_max
    // Scalar exp via polynomial
    movz    w4, #0x3FB8, lsl #16
    movk    w4, #0xAA3B
    fmov    s28, w4                // log2(e)
    fmul    s3, s2, s28            // x * log2(e)
    frintm  s4, s3                 // n = floor
    fsub    s5, s3, s4             // frac
    movz    w4, #0x3C1D, lsl #16
    movk    w4, #0x955A
    fmov    s6, w4                 // c4
    fmul    s6, s6, s5             // c4*f
    movz    w4, #0x3D63, lsl #16
    movk    w4, #0x5847
    fmov    s7, w4
    fadd    s6, s6, s7             // +c3
    fmul    s6, s6, s5
    movz    w4, #0x3E75, lsl #16
    movk    w4, #0xFDF0
    fmov    s7, w4
    fadd    s6, s6, s7             // +c2
    fmul    s6, s6, s5
    movz    w4, #0x3F31, lsl #16
    movk    w4, #0x7218
    fmov    s7, w4
    fadd    s6, s6, s7             // +c1
    fmul    s6, s6, s5
    fmov    s7, #1.0
    fadd    s6, s6, s7             // +c0 = poly(frac)
    fcvtzs  w4, s4                 // n as int
    mov     w5, #-127
    cmp     w4, w5
    csel    w4, w5, w4, lt
    add     w4, w4, #127
    lsl     w4, w4, #23            // 2^n as IEEE bits
    fmov    s7, w4
    fmul    s6, s6, s7             // correction = exp(local_max - global_max)
    // ── Broadcast correction and apply ──
    mov     z16.s, s6
    mov     w10, w22
.L_sc_loop:
    ld1w    {z0.s}, p0/z, [x8]
    fmul    z0.s, z0.s, z16.s
    st1w    {z0.s}, p0, [x8]
    add     x8, x8, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_sc_loop
    // ── Correct sum ──
    ldr     s0, [x11]
    fmul    s0, s0, s6
    str     s0, [x11]
    b       .L_dispatch
// ================================================================
// REDUCE_SUM_SQ_FP32 (0x67) — Partial sum of squares
//
// Encoding: [0x67][dim:u32][in_ptr:u64][out_ptr:u64]
//
// result = sum(in[i]^2) for i in 0..dim-1. Stores scalar fp32 to *out_ptr.
// dim must be a multiple of cntw (16).
// Building block for decomposed RMS norm: each shard computes partial sum_sq,
// merge across shards, then apply rsqrt(sum_sq/total_dim + eps) * x.
// ================================================================
.L_op_reduce_sum_sq_fp32:
    ldr     w22, [x19]             // dim
    add     x19, x19, #4
    ldr     x8, [x19], #8         // in_ptr
    ldr     x11, [x19], #8        // out_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    fmov    z0.s, #0.0             // accumulator
    mov     w10, w22
.L_rss_loop:
    ld1w    {z1.s}, p0/z, [x8]
    fmla    z0.s, p0/m, z1.s, z1.s // accum += x[i]^2
    add     x8, x8, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_rss_loop
    faddv   s0, p0, z0.s
    str     s0, [x11]
    b       .L_dispatch
// ================================================================
// REDUCE_COL_SUM_FP32 (0x68) — Column-wise sum (bias gradient)
//
// dst[j] = sum(src[i * stride + j]) for i in 0..M-1, j in 0..N-1
// Processes 16 columns per outer iteration. Inner loop streams down
// M rows with stride between them — sequential access per column group.
//
// Encoding: [0x68][M:u32][N:u32][stride:u32][src_ptr:u64][dst_ptr:u64]
//
// M = number of rows to sum
// N = number of columns in the output (must be multiple of 16)
// stride = row stride in fp32 elements (typically N for contiguous matrix)
// ================================================================
.L_op_reduce_col_sum_fp32:
    ldr     w22, [x19]             // M (rows)
    ldr     w23, [x19, #4]        // N (columns)
    ldr     w24, [x19, #8]        // stride (fp32 elements)
    add     x19, x19, #12
    ldr     x8, [x19], #8         // src_ptr
    ldr     x11, [x19], #8        // dst_ptr
    cbz     w22, .L_dispatch
    cbz     w23, .L_dispatch
    ptrue   p0.s
    cntw    x9                     // 16
    lsl     x16, x24, #2           // stride in bytes (stride * 4)
    mov     x17, x8                // save src base
    // ── Outer loop: 16 columns at a time ──
    mov     w10, #0                // col offset
.L_rcs_col:
    cmp     w10, w23
    b.ge    .L_rcs_done
    // Zero accumulator
    fmov    z0.s, #0.0
    // Compute column predicate for edge case (last group may have < 16 cols)
    sub     w12, w23, w10          // remaining cols
    whilelt p1.s, xzr, x12
    // Row base = src + col_offset * 4
    add     x8, x17, x10, lsl #2
    mov     w12, w22               // row counter
    // ── Inner loop: sum down M rows ──
.L_rcs_row:
    ld1w    {z1.s}, p1/z, [x8]
    fadd    z0.s, p1/m, z0.s, z1.s
    add     x8, x8, x16           // advance by stride bytes to next row
    subs    w12, w12, #1
    b.ne    .L_rcs_row
    // Store 16-column partial sum
    st1w    {z0.s}, p1, [x11]
    add     x11, x11, x9, lsl #2  // dst += 16 floats
    add     w10, w10, w9           // col += 16
    b       .L_rcs_col
.L_rcs_done:
    b       .L_dispatch
// ================================================================
// EXP POLYNOMIAL MACRO — shared by silu_backward, gelu, adam, etc.
// Expects input in z0, clobbers z1-z6. Result in z4.
// Polynomial constants must be pre-loaded:
//   z28 = log2(e), z29 = c4, z30 = c3, z31 = c2, z27 = c1=ln(2), z26 = 1.0
// ================================================================
.macro EXP_POLY_Z0_TO_Z4
    fmul    z1.s, z0.s, z28.s      // x * log2(e)
    frintm  z2.s, p0/m, z1.s       // n = floor
    fsub    z3.s, z1.s, z2.s        // frac
    fmul    z4.s, z29.s, z3.s       // c4*f
    fadd    z4.s, z4.s, z30.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z31.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z27.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z26.s       // poly(frac)
    fcvtzs  z5.s, p0/m, z2.s
    mov     z6.s, #-127
    smax    z5.s, p0/m, z5.s, z6.s
    add     z5.s, z5.s, #127
    lsl     z5.s, z5.s, #23         // 2^n as IEEE bits
    fmul    z4.s, z4.s, z5.s        // z4 = exp(z0)
.endm
// ================================================================
// SIGMOID MACRO — computes sigmoid(z7) into z5, clobbers z0-z6.
// Requires exp polynomial constants in z26-z31, z27, z28.
// ================================================================
.macro SIGMOID_Z7_TO_Z5
    fneg    z0.s, p0/m, z7.s       // -x
    EXP_POLY_Z0_TO_Z4              // z4 = exp(-x)
    fadd    z4.s, z4.s, z26.s       // 1 + exp(-x)
    frecpe  z5.s, z4.s
    frecps  z6.s, z4.s, z5.s
    fmul    z5.s, p0/m, z5.s, z6.s
    frecps  z6.s, z4.s, z5.s
    fmul    z5.s, p0/m, z5.s, z6.s  // z5 = sigmoid(x) = 1/(1+exp(-x))
.endm
// ================================================================
// LOAD_EXP_CONSTANTS MACRO — loads polynomial constants into z26-z31, z27, z28
// ================================================================
.macro LOAD_EXP_CONSTANTS
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
    fmov    z26.s, #1.0             // c0
.endm
// ================================================================
// SILU_BACKWARD_FP32 (0x69) — SiLU backward pass
// dx = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
//    = dy * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
//
// Encoding: [0x69][dim:u32][x_ptr:u64][dy_ptr:u64][dx_ptr:u64]
// dim must be a multiple of 16.
// ================================================================
.L_op_silu_backward_fp32:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8         // x_ptr (forward input)
    ldr     x11, [x19], #8        // dy_ptr (upstream gradient)
    ldr     x13, [x19], #8        // dx_ptr (output gradient)
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    LOAD_EXP_CONSTANTS
    mov     w10, w22
.L_silub_loop:
    ld1w    {z7.s}, p0/z, [x8]    // z7 = x
    SIGMOID_Z7_TO_Z5               // z5 = sigmoid(x)
    // z5 = sigmoid(x), z7 = x
    // dx = dy * (sigmoid + x * sigmoid * (1 - sigmoid))
    mov     z0.d, z26.d             // z0 = 1.0
    fsub    z0.s, z0.s, z5.s       // z0 = 1 - sigmoid
    fmul    z0.s, z0.s, z5.s       // z0 = sigmoid * (1 - sigmoid)
    fmul    z0.s, z0.s, z7.s       // z0 = x * sigmoid * (1 - sigmoid)
    fadd    z0.s, z0.s, z5.s       // z0 = sigmoid + x * sigmoid * (1 - sigmoid)
    ld1w    {z1.s}, p0/z, [x11]   // z1 = dy
    fmul    z0.s, z0.s, z1.s       // z0 = dy * derivative
    st1w    {z0.s}, p0, [x13]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x13, x13, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_silub_loop
    b       .L_dispatch
// ================================================================
// SOFTMAX_BACKWARD_FP32 (0x6A) — Softmax backward pass
// dx[i] = s[i] * (dy[i] - sum(s[j]*dy[j]))
//
// Two-pass:
//   Pass 1: dot = sum(s[i] * dy[i])
//   Pass 2: dx[i] = s[i] * (dy[i] - dot)
//
// Encoding: [0x6A][dim:u32][s_ptr:u64][dy_ptr:u64][dx_ptr:u64]
// dim must be a multiple of 16.
// ================================================================
.L_op_softmax_backward_fp32:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8         // s_ptr (softmax output from forward)
    ldr     x11, [x19], #8        // dy_ptr (upstream gradient)
    ldr     x13, [x19], #8        // dx_ptr (output gradient)
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    // ── Pass 1: dot = sum(s * dy) ──
    fmov    z16.s, #0.0            // accumulator
    mov     x14, x8                // save s_ptr
    mov     x15, x11               // save dy_ptr
    mov     w10, w22
.L_smb_dot:
    ld1w    {z0.s}, p0/z, [x8]
    ld1w    {z1.s}, p0/z, [x11]
    fmla    z16.s, p0/m, z0.s, z1.s
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_smb_dot
    faddv   s16, p0, z16.s
    mov     z16.s, s16              // broadcast dot product
    // ── Pass 2: dx = s * (dy - dot) ──
    mov     x8, x14
    mov     x11, x15
    mov     w10, w22
.L_smb_grad:
    ld1w    {z0.s}, p0/z, [x8]    // s
    ld1w    {z1.s}, p0/z, [x11]   // dy
    fsub    z1.s, z1.s, z16.s      // dy - dot
    fmul    z0.s, z0.s, z1.s       // s * (dy - dot)
    st1w    {z0.s}, p0, [x13]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x13, x13, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_smb_grad
    b       .L_dispatch
// ================================================================
// GELU_FP32 (0x6B) — GeLU activation (tanh approximation)
// out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// tanh(z) = 1 - 2/(1 + exp(2z)), computed via exp polynomial + reciprocal.
//
// Encoding: [0x6B][count:u32][in_ptr:u64][out_ptr:u64]
// count must be a multiple of 16.
// ================================================================
.L_op_gelu_fp32:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8         // input_ptr
    ldr     x11, [x19], #8        // output_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    LOAD_EXP_CONSTANTS
    // GeLU constants
    movz    w4, #0x3F4C, lsl #16
    movk    w4, #0x422A
    fmov    s16, w4
    mov     z16.s, s16              // sqrt(2/pi) = 0.7978845608
    movz    w4, #0x3D37, lsl #16
    movk    w4, #0x2713
    fmov    s17, w4
    mov     z17.s, s17              // 0.044715
    fmov    z18.s, #0.5             // 0.5
    fmov    z19.s, #2.0             // 2.0
    mov     w10, w22
.L_gelu_loop:
    ld1w    {z7.s}, p0/z, [x8]    // z7 = x
    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    fmul    z0.s, z7.s, z7.s       // x^2
    fmul    z0.s, z0.s, z7.s       // x^3
    fmul    z0.s, z0.s, z17.s      // 0.044715 * x^3
    fadd    z0.s, z0.s, z7.s       // x + 0.044715 * x^3
    fmul    z0.s, z0.s, z16.s      // inner = sqrt(2/pi) * (...)
    // tanh(inner) = 1 - 2/(1 + exp(2*inner))
    fmul    z0.s, z0.s, z19.s      // 2 * inner
    EXP_POLY_Z0_TO_Z4              // z4 = exp(2*inner)
    fadd    z4.s, z4.s, z26.s       // 1 + exp(2*inner)
    frecpe  z5.s, z4.s
    frecps  z6.s, z4.s, z5.s
    fmul    z5.s, p0/m, z5.s, z6.s
    frecps  z6.s, z4.s, z5.s
    fmul    z5.s, p0/m, z5.s, z6.s  // z5 = 1/(1+exp(2*inner))
    fmul    z5.s, z5.s, z19.s      // 2/(1+exp(2*inner))
    fsub    z5.s, z26.s, z5.s      // tanh = 1 - 2/(1+exp(2*inner))
    // out = 0.5 * x * (1 + tanh)
    fadd    z5.s, z5.s, z26.s       // 1 + tanh
    fmul    z5.s, z5.s, z7.s       // x * (1 + tanh)
    fmul    z5.s, z5.s, z18.s      // 0.5 * x * (1 + tanh)
    st1w    {z5.s}, p0, [x11]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_gelu_loop
    b       .L_dispatch
// ================================================================
// LAYER_NORM_FP32 (0x6C) — Full layer normalization
// out[i] = ((x[i] - mean) / sqrt(var + eps)) * gamma[i] + beta[i]
//
// Two-pass:
//   Pass 1: compute mean and variance (Welford online)
//   Pass 2: normalize, scale by gamma, add beta
//
// Encoding: [0x6C][dim:u32][eps:f32][in_ptr:u64][gamma_ptr:u64][beta_ptr:u64][out_ptr:u64]
// dim must be a multiple of 16.
// ================================================================
.L_op_layer_norm_fp32:
    ldr     w22, [x19]             // dim
    ldr     s20, [x19, #4]        // eps
    add     x19, x19, #8
    ldr     x8, [x19], #8         // in_ptr
    ldr     x11, [x19], #8        // gamma_ptr
    ldr     x12, [x19], #8        // beta_ptr
    ldr     x13, [x19], #8        // out_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    mov     x14, x8                // save in_ptr
    // ── Pass 1: compute mean ──
    fmov    z16.s, #0.0            // sum accumulator
    mov     w10, w22
.L_ln_sum:
    ld1w    {z0.s}, p0/z, [x8]
    fadd    z16.s, z16.s, z0.s
    add     x8, x8, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_ln_sum
    faddv   s16, p0, z16.s         // total sum
    ucvtf   s17, w22               // (float)dim
    fdiv    s16, s16, s17           // mean = sum / dim
    mov     z16.s, s16              // broadcast mean
    // ── Pass 1b: compute variance ──
    fmov    z17.s, #0.0            // var accumulator
    mov     x8, x14
    mov     w10, w22
.L_ln_var:
    ld1w    {z0.s}, p0/z, [x8]
    fsub    z0.s, z0.s, z16.s      // x - mean
    fmla    z17.s, p0/m, z0.s, z0.s // var += (x - mean)^2
    add     x8, x8, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_ln_var
    faddv   s17, p0, z17.s         // horizontal sum → var_sum scalar
    ucvtf   s18, w22
    fdiv    s17, s17, s18           // var = var_sum / dim
    fadd    s17, s17, s20           // var + eps
    fsqrt   s17, s17                // sqrt(var + eps)
    fmov    s18, #1.0
    fdiv    s17, s18, s17           // inv_std = 1 / sqrt(var + eps)
    mov     z17.s, s17              // broadcast inv_std
    // ── Pass 2: normalize, scale, shift ──
    mov     x8, x14
    mov     w10, w22
.L_ln_norm:
    ld1w    {z0.s}, p0/z, [x8]    // x
    ld1w    {z1.s}, p0/z, [x11]   // gamma
    ld1w    {z2.s}, p0/z, [x12]   // beta
    fsub    z0.s, z0.s, z16.s      // x - mean
    fmul    z0.s, z0.s, z17.s      // (x - mean) * inv_std
    fmul    z0.s, z0.s, z1.s       // * gamma
    fadd    z0.s, z0.s, z2.s       // + beta
    st1w    {z0.s}, p0, [x13]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x12, x12, x9, lsl #2
    add     x13, x13, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_ln_norm
    b       .L_dispatch
// ================================================================
// CAUSAL_MASK_FP32 (0x6D) — Apply causal (lower-triangular) attention mask
// For each position (i, j) where j > i: scores[i * stride + j] = -inf
//
// Encoding: [0x6D][dim:u32][stride:u32][ptr:u64]
// dim = number of rows = number of columns (square attention matrix)
// stride = row stride in fp32 elements
// ================================================================
.L_op_causal_mask_fp32:
    ldr     w22, [x19]             // dim
    ldr     w23, [x19, #4]        // stride
    add     x19, x19, #8
    ldr     x8, [x19], #8         // ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    lsl     x16, x23, #2           // stride in bytes
    movz    w4, #0xFF80, lsl #16   // -inf (0xFF800000)
    fmov    s16, w4
    mov     z16.s, s16              // z16 = broadcast(-inf)
    // For row i: mask columns j where j > i
    // i.e., starting from column i+1 through dim-1
    mov     w10, #0                // row i
.L_cm_row:
    cmp     w10, w22
    b.ge    .L_cm_done
    add     w12, w10, #1           // first masked column = i + 1
    cmp     w12, w22
    b.ge    .L_cm_next_row         // row i has no columns to mask (last row)
    // Address of scores[i][i+1]
    mul     w14, w10, w23          // i * stride
    add     w14, w14, w12          // + (i+1)
    add     x11, x8, x14, lsl #2  // byte address
    sub     w15, w22, w12          // count = dim - (i+1)
    // Fill with -inf, 16 at a time
.L_cm_fill:
    cmp     w15, w9
    b.lt    .L_cm_fill_tail
    st1w    {z16.s}, p0, [x11]
    add     x11, x11, x9, lsl #2
    sub     w15, w15, w9
    b       .L_cm_fill
.L_cm_fill_tail:
    cbz     w15, .L_cm_next_row
    whilelt p1.s, xzr, x15
    st1w    {z16.s}, p1, [x11]
.L_cm_next_row:
    add     w10, w10, #1
    b       .L_cm_row
.L_cm_done:
    b       .L_dispatch
// ================================================================
// ADAM_STEP_FP32 (0x6E) — Fused Adam optimizer step
// Single pass over params, grads, m (1st moment), v (2nd moment):
//   m[i] = beta1 * m[i] + (1-beta1) * g[i]
//   v[i] = beta2 * v[i] + (1-beta2) * g[i]^2
//   m_hat = m[i] / (1 - beta1^t)
//   v_hat = v[i] / (1 - beta2^t)
//   params[i] -= lr * m_hat / (sqrt(v_hat) + eps)
//
// Encoding: [0x6E][count:u32][lr:f32][beta1:f32][beta2:f32][eps:f32][t:u32]
//           [params_ptr:u64][grads_ptr:u64][m_ptr:u64][v_ptr:u64]
// count must be a multiple of 16. t = current timestep (1-based).
// ================================================================
.L_op_adam_step_fp32:
    ldr     w22, [x19]             // count
    ldr     s20, [x19, #4]        // lr
    ldr     s21, [x19, #8]        // beta1
    ldr     s22, [x19, #12]       // beta2
    ldr     s23, [x19, #16]       // eps
    ldr     w24, [x19, #20]       // t (timestep)
    add     x19, x19, #24
    ldr     x8, [x19], #8         // params_ptr
    ldr     x11, [x19], #8        // grads_ptr
    ldr     x12, [x19], #8        // m_ptr
    ldr     x13, [x19], #8        // v_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    // ── Pre-compute bias-corrected lr scalars via exp(t*ln(beta)) ──
    // Both beta1^t and beta2^t computed in parallel via z-vectors
    LOAD_EXP_CONSTANTS
    ucvtf   s1, w24                 // s1 = (float)t
    // ln(beta) via vector pipeline: broadcast scalar, extract exponent, minimax cubic
    // Process both beta1 and beta2 in z-vector lane 0 and lane 1
    mov     z0.s, s21               // broadcast beta1 into z0 (only lane 0 matters)
    mov     z8.s, s22               // broadcast beta2 into z8
    // ln(z0) → z0, ln(z8) → z8 via shared LN sequence
    // z0 = beta1, z8 = beta2. Compute ln of each using lane 0.
    // Extract exponent: reinterpret as int, extract bits 23..30
    // Use integer operations on z-vector lanes
    lsr     z1.s, z0.s, #23        // shift exponent to low bits
    and     z1.s, z1.s, #0xFF      // mask 8-bit exponent
    mov     z2.s, #127
    sub     z1.s, z1.s, z2.s       // unbiased exponent
    scvtf   z1.s, p0/m, z1.s       // e as float
    // Set exponent to 127 (mantissa in [1,2))
    mov     z3.d, z0.d
    and     z3.s, z3.s, #0x007FFFFF // extract mantissa bits
    orr     z3.s, z3.s, #0x3F800000 // set exponent = 127 → value in [1,2)
    fsub    z3.s, z3.s, z26.s      // t = m - 1
    // Minimax cubic: ln(1+t) ≈ a3*t^3 + a2*t^2 + a1*t
    movz    w4, #0x3E94, lsl #16
    movk    w4, #0x3014
    fmov    s9, w4
    mov     z9.s, s9                // a3 = 0.28947478
    movz    w4, #0xBEFB, lsl #16
    movk    w4, #0xD464
    fmov    s10, w4
    mov     z10.s, s10              // a2 = -0.49190896
    movz    w4, #0x3F7F, lsl #16
    movk    w4, #0xF972
    fmov    s11, w4
    mov     z11.s, s11              // a1 = 0.99949556
    fmul    z4.s, z9.s, z3.s       // a3*t
    fadd    z4.s, z4.s, z10.s      // + a2
    fmul    z4.s, z4.s, z3.s       // (a3*t+a2)*t
    fadd    z4.s, z4.s, z11.s      // + a1
    fmul    z4.s, z4.s, z3.s       // ln(m)
    fmul    z1.s, z1.s, z27.s      // e * ln(2) (z27 = ln(2) from LOAD_EXP_CONSTANTS)
    fadd    z0.s, z1.s, z4.s       // z0 = ln(beta1)
    // Same for beta2
    lsr     z1.s, z8.s, #23
    and     z1.s, z1.s, #0xFF
    sub     z1.s, z1.s, z2.s
    scvtf   z1.s, p0/m, z1.s
    mov     z3.d, z8.d
    and     z3.s, z3.s, #0x007FFFFF
    orr     z3.s, z3.s, #0x3F800000
    fsub    z3.s, z3.s, z26.s
    fmul    z4.s, z9.s, z3.s
    fadd    z4.s, z4.s, z10.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z11.s
    fmul    z4.s, z4.s, z3.s
    fmul    z1.s, z1.s, z27.s
    fadd    z8.s, z1.s, z4.s       // z8 = ln(beta2)
    // t * ln(beta) → exp
    mov     z1.s, s1               // broadcast t
    fmul    z0.s, z0.s, z1.s       // t * ln(beta1)
    EXP_POLY_Z0_TO_Z4              // z4 = beta1^t
    fmov    s3, s4                  // s3 = beta1^t (lane 0)
    mov     z0.d, z8.d             // z0 = t * ln(beta2)
    fmul    z0.s, z0.s, z1.s
    EXP_POLY_Z0_TO_Z4              // z4 = beta2^t
    // s4 already has beta2^t in lane 0
    fmov    s0, #1.0
    fsub    s3, s0, s3             // bc1 = 1 - beta1^t
    fsub    s4, s0, s4             // bc2 = 1 - beta2^t
    fdiv    s5, s20, s3            // lr_bc = lr / bc1
    fsqrt   s6, s4                 // sqrt(bc2)  (for v_hat correction)
    // Broadcast constants to z-vectors
    mov     z16.s, s21              // beta1
    fmov    s7, #1.0
    fsub    s7, s7, s21
    mov     z17.s, s7               // 1 - beta1
    mov     z18.s, s22              // beta2
    fmov    s7, #1.0
    fsub    s7, s7, s22
    mov     z19.s, s7               // 1 - beta2
    mov     z20.s, s5               // lr / (1 - beta1^t)
    mov     z21.s, s23              // eps
    fdiv    s6, s0, s6             // 1/sqrt(bc2)
    mov     z22.s, s6               // 1/sqrt(1 - beta2^t) for v_hat correction
    mov     w10, w22
.L_adam_loop:
    ld1w    {z0.s}, p0/z, [x12]   // m
    ld1w    {z1.s}, p0/z, [x13]   // v
    ld1w    {z2.s}, p0/z, [x11]   // g
    ld1w    {z3.s}, p0/z, [x8]    // params
    // m = beta1 * m + (1 - beta1) * g
    fmul    z0.s, z0.s, z16.s      // beta1 * m
    fmla    z0.s, p0/m, z17.s, z2.s // + (1-beta1) * g
    // v = beta2 * v + (1 - beta2) * g^2
    fmul    z1.s, z1.s, z18.s      // beta2 * v
    fmul    z4.s, z2.s, z2.s       // g^2
    fmla    z1.s, p0/m, z19.s, z4.s // + (1-beta2) * g^2
    // Store updated m, v
    st1w    {z0.s}, p0, [x12]
    st1w    {z1.s}, p0, [x13]
    // v_hat = v / (1 - beta2^t) → multiply by 1/(1-beta2^t) pre-corrected via sqrt
    // update = lr_bc * m / (sqrt(v_hat) + eps)
    //        = (lr/(1-b1^t)) * m / (sqrt(v/(1-b2^t)) + eps)
    fmul    z4.s, z1.s, z22.s      // v * (1/sqrt(1-b2^t))^2 = v/(1-b2^t) ... no
    // Simpler: sqrt(v) * (1/sqrt(1-b2^t)) = sqrt(v/(1-b2^t)) = sqrt(v_hat)
    fsqrt   z4.s, p0/m, z1.s       // sqrt(v)
    fmul    z4.s, z4.s, z22.s      // sqrt(v) / sqrt(1-b2^t) = sqrt(v_hat)
    fadd    z4.s, z4.s, z21.s      // sqrt(v_hat) + eps
    frecpe  z5.s, z4.s
    frecps  z6.s, z4.s, z5.s
    fmul    z5.s, p0/m, z5.s, z6.s
    frecps  z6.s, z4.s, z5.s
    fmul    z5.s, p0/m, z5.s, z6.s  // 1 / (sqrt(v_hat) + eps)
    fmul    z5.s, z5.s, z0.s       // m / (sqrt(v_hat) + eps)
    fmul    z5.s, z5.s, z20.s      // lr_bc * m / (sqrt(v_hat) + eps)
    fsub    z3.s, z3.s, z5.s       // params -= update
    st1w    {z3.s}, p0, [x8]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x12, x12, x9, lsl #2
    add     x13, x13, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_adam_loop
    b       .L_dispatch
// ================================================================
// GELU_BACKWARD_FP32 (0x6F) — GeLU backward pass
// gelu(x) = 0.5*x*(1+tanh(inner)) where inner = sqrt(2/pi)*(x+0.044715*x^3)
// gelu'(x) = 0.5*(1+tanh) + 0.5*x*(1-tanh^2)*sqrt(2/pi)*(1+3*0.044715*x^2)
// dx = dy * gelu'(x)
//
// Encoding: [0x6F][dim:u32][x_ptr:u64][dy_ptr:u64][dx_ptr:u64]
// dim must be a multiple of 16.
// ================================================================
.L_op_gelu_backward_fp32:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8         // x_ptr
    ldr     x11, [x19], #8        // dy_ptr
    ldr     x13, [x19], #8        // dx_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    LOAD_EXP_CONSTANTS
    movz    w4, #0x3F4C, lsl #16
    movk    w4, #0x422A
    fmov    s16, w4
    mov     z16.s, s16              // sqrt(2/pi)
    movz    w4, #0x3D37, lsl #16
    movk    w4, #0x2713
    fmov    s17, w4
    mov     z17.s, s17              // 0.044715
    fmov    z18.s, #0.5
    fmov    z19.s, #2.0
    // 3 * 0.044715 = 0.134145
    movz    w4, #0x3E09, lsl #16
    movk    w4, #0x7B42
    fmov    s20, w4
    mov     z20.s, s20              // 0.134145
    mov     w10, w22
.L_gelub_loop:
    ld1w    {z7.s}, p0/z, [x8]    // z7 = x
    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    fmul    z0.s, z7.s, z7.s       // x^2
    fmul    z1.s, z0.s, z7.s       // x^3
    fmul    z1.s, z1.s, z17.s      // 0.044715 * x^3
    fadd    z1.s, z1.s, z7.s       // x + 0.044715 * x^3
    fmul    z1.s, z1.s, z16.s      // inner
    // tanh(inner) = 1 - 2/(1 + exp(2*inner))
    fmul    z2.s, z1.s, z19.s      // 2 * inner
    mov     z0.d, z2.d
    EXP_POLY_Z0_TO_Z4              // z4 = exp(2*inner)
    fadd    z4.s, z4.s, z26.s       // 1 + exp(2*inner)
    frecpe  z5.s, z4.s
    frecps  z6.s, z4.s, z5.s
    fmul    z5.s, p0/m, z5.s, z6.s
    frecps  z6.s, z4.s, z5.s
    fmul    z5.s, p0/m, z5.s, z6.s  // z5 = 1/(1+exp(2*inner))
    fmul    z5.s, z5.s, z19.s      // 2/(1+exp(2*inner))
    fsub    z3.s, z26.s, z5.s      // z3 = tanh(inner)
    // gelu'(x) = 0.5*(1+tanh) + 0.5*x*(1-tanh^2)*sqrt(2/pi)*(1+0.134145*x^2)
    fmul    z4.s, z3.s, z3.s       // tanh^2
    fsub    z4.s, z26.s, z4.s      // 1 - tanh^2 = sech^2
    fmul    z5.s, z7.s, z7.s       // x^2
    fmul    z5.s, z5.s, z20.s      // 0.134145 * x^2
    fadd    z5.s, z5.s, z26.s      // 1 + 0.134145 * x^2
    fmul    z5.s, z5.s, z16.s      // sqrt(2/pi) * (1 + 0.134145*x^2)
    fmul    z5.s, z5.s, z4.s       // sech^2 * sqrt(2/pi) * (1+...)
    fmul    z5.s, z5.s, z7.s       // x * sech^2 * sqrt(2/pi) * (1+...)
    fmul    z5.s, z5.s, z18.s      // 0.5 * x * sech^2 * ...
    fadd    z4.s, z3.s, z26.s      // 1 + tanh
    fmul    z4.s, z4.s, z18.s      // 0.5 * (1 + tanh)
    fadd    z4.s, z4.s, z5.s       // gelu'(x) = term1 + term2
    ld1w    {z1.s}, p0/z, [x11]   // dy
    fmul    z4.s, z4.s, z1.s       // dx = dy * gelu'(x)
    st1w    {z4.s}, p0, [x13]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x13, x13, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_gelub_loop
    b       .L_dispatch
// ================================================================
// RMS_NORM_BACKWARD_FP32 (0x70) — RMS norm backward pass
// Forward: y = x * inv_rms * w, where inv_rms = 1/sqrt(sum(x^2)/dim + eps)
//
// Gradients:
//   dot = sum(dy * w * x)
//   dx = inv_rms * (dy*w - x * dot * inv_rms^2 / dim)
//   dw = dy * x * inv_rms
//
// Three passes: (1) sum(x^2), (2) dot=sum(dy*w*x), (3) dx and dw
//
// Encoding: [0x70][dim:u32][eps:f32][x:u64][w:u64][dy:u64][dx:u64][dw:u64]
// dim must be a multiple of 16.
// ================================================================
.L_op_rms_norm_backward_fp32:
    ldr     w22, [x19]             // dim
    ldr     s20, [x19, #4]        // eps
    add     x19, x19, #8
    ldr     x8, [x19], #8         // x_ptr
    ldr     x11, [x19], #8        // w_ptr
    ldr     x12, [x19], #8        // dy_ptr
    ldr     x13, [x19], #8        // dx_ptr
    ldr     x14, [x19], #8        // dw_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    mov     x15, x8                // save x_ptr
    mov     x24, x11               // save w_ptr
    mov     x16, x12               // save dy_ptr
    // ── Pass 1: sum_sq = sum(x^2) ──
    fmov    z16.s, #0.0
    mov     w10, w22
.L_rmsb_sumsq:
    ld1w    {z0.s}, p0/z, [x8]
    fmla    z16.s, p0/m, z0.s, z0.s
    add     x8, x8, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_rmsb_sumsq
    faddv   s16, p0, z16.s         // sum_sq scalar
    ucvtf   s17, w22
    fdiv    s18, s16, s17           // sum_sq / dim
    fadd    s18, s18, s20           // + eps
    fsqrt   s18, s18                // sqrt(sum_sq/dim + eps) = rms
    fmov    s19, #1.0
    fdiv    s19, s19, s18           // inv_rms
    // inv_rms^2 / dim = 1/(rms^2 * dim) = 1/((sum_sq/dim+eps)*dim)
    fmul    s21, s18, s18           // rms^2
    fmul    s21, s21, s17           // rms^2 * dim
    fmov    s22, #1.0
    fdiv    s21, s22, s21           // inv_rms^2 / dim
    // ── Pass 2: dot = sum(dy * w * x) ──
    fmov    z17.s, #0.0
    mov     x8, x15
    mov     x11, x24
    mov     x12, x16
    mov     w10, w22
.L_rmsb_dot:
    ld1w    {z0.s}, p0/z, [x12]   // dy
    ld1w    {z1.s}, p0/z, [x11]   // w
    ld1w    {z2.s}, p0/z, [x8]    // x
    fmul    z0.s, z0.s, z1.s       // dy * w
    fmla    z17.s, p0/m, z0.s, z2.s // += dy*w*x
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x12, x12, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_rmsb_dot
    faddv   s17, p0, z17.s         // dot scalar
    fmul    s17, s17, s21           // dot * inv_rms^2 / dim
    mov     z17.s, s17              // broadcast scale
    mov     z19.s, s19              // broadcast inv_rms
    // ── Pass 3: dx = inv_rms * (dy*w - x * dot_scale), dw = dy * x * inv_rms ──
    mov     x8, x15
    mov     x11, x24
    mov     x12, x16
    mov     w10, w22
.L_rmsb_grad:
    ld1w    {z0.s}, p0/z, [x12]   // dy
    ld1w    {z1.s}, p0/z, [x11]   // w
    ld1w    {z2.s}, p0/z, [x8]    // x
    // dw = dy * x * inv_rms
    fmul    z3.s, z0.s, z2.s       // dy * x
    fmul    z3.s, z3.s, z19.s      // * inv_rms
    st1w    {z3.s}, p0, [x14]
    // dx = inv_rms * (dy*w - x * dot_scale)
    fmul    z4.s, z0.s, z1.s       // dy * w
    fmul    z5.s, z2.s, z17.s      // x * dot_scale
    fsub    z4.s, z4.s, z5.s       // dy*w - x*dot_scale
    fmul    z4.s, z4.s, z19.s      // * inv_rms
    st1w    {z4.s}, p0, [x13]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x12, x12, x9, lsl #2
    add     x13, x13, x9, lsl #2
    add     x14, x14, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_rmsb_grad
    b       .L_dispatch
// ================================================================
// LAYER_NORM_BACKWARD_FP32 (0x71) — Layer norm backward pass
// Forward: y = (x - mean) * inv_std * gamma + beta
//
// Gradients:
//   dgamma[i] = dy[i] * x_hat[i]    where x_hat = (x - mean) * inv_std
//   dbeta[i]  = dy[i]
//   ds = sum(dy * gamma * x_hat) / dim
//   dm = sum(dy * gamma) / dim
//   dx[i] = inv_std * (dy[i] * gamma[i] - dm - x_hat[i] * ds)
//
// Four passes: (1) mean+var, (2) dgamma+dbeta+accum ds/dm, (3) dx
//
// Encoding: [0x71][dim:u32][eps:f32][x:u64][gamma:u64][dy:u64][dx:u64][dgamma:u64][dbeta:u64]
// dim must be a multiple of 16.
// ================================================================
.L_op_layer_norm_backward_fp32:
    ldr     w22, [x19]
    ldr     s20, [x19, #4]        // eps
    add     x19, x19, #8
    ldr     x8, [x19], #8         // x
    ldr     x11, [x19], #8        // gamma
    ldr     x12, [x19], #8        // dy
    ldr     x13, [x19], #8        // dx
    ldr     x14, [x19], #8        // dgamma
    ldr     x15, [x19], #8        // dbeta
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    mov     x24, x8                // save x
    mov     x16, x11               // save gamma
    mov     x17, x12               // save dy
    // ── Pass 1a: mean ──
    fmov    z16.s, #0.0
    mov     w10, w22
.L_lnb_sum:
    ld1w    {z0.s}, p0/z, [x8]
    fadd    z16.s, z16.s, z0.s
    add     x8, x8, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_lnb_sum
    faddv   s16, p0, z16.s
    ucvtf   s17, w22
    fdiv    s16, s16, s17           // mean
    mov     z16.s, s16
    // ── Pass 1b: var ──
    fmov    z17.s, #0.0
    mov     x8, x24
    mov     w10, w22
.L_lnb_var:
    ld1w    {z0.s}, p0/z, [x8]
    fsub    z0.s, z0.s, z16.s
    fmla    z17.s, p0/m, z0.s, z0.s
    add     x8, x8, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_lnb_var
    faddv   s17, p0, z17.s
    ucvtf   s18, w22
    fdiv    s17, s17, s18           // var
    fadd    s17, s17, s20           // var + eps
    fsqrt   s17, s17
    fmov    s18, #1.0
    fdiv    s17, s18, s17           // inv_std
    mov     z17.s, s17              // broadcast inv_std
    // ── Pass 2: dgamma, dbeta, accumulate ds and dm ──
    fmov    z18.s, #0.0            // ds accumulator
    fmov    z19.s, #0.0            // dm accumulator
    mov     x8, x24
    mov     x11, x16
    mov     x12, x17
    mov     w10, w22
.L_lnb_dg:
    ld1w    {z0.s}, p0/z, [x8]    // x
    ld1w    {z1.s}, p0/z, [x11]   // gamma
    ld1w    {z2.s}, p0/z, [x12]   // dy
    // x_hat = (x - mean) * inv_std
    fsub    z3.s, z0.s, z16.s
    fmul    z3.s, z3.s, z17.s      // x_hat
    // dgamma = dy * x_hat
    fmul    z4.s, z2.s, z3.s
    st1w    {z4.s}, p0, [x14]
    // dbeta = dy
    st1w    {z2.s}, p0, [x15]
    // ds += dy * gamma * x_hat
    fmul    z4.s, z2.s, z1.s       // dy * gamma
    fmla    z18.s, p0/m, z4.s, z3.s // += dy*gamma*x_hat
    // dm += dy * gamma
    fadd    z19.s, z19.s, z4.s
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x12, x12, x9, lsl #2
    add     x14, x14, x9, lsl #2
    add     x15, x15, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_lnb_dg
    faddv   s18, p0, z18.s         // ds total
    ucvtf   s0, w22
    fdiv    s18, s18, s0            // ds / dim
    mov     z18.s, s18
    faddv   s19, p0, z19.s         // dm total
    fdiv    s19, s19, s0            // dm / dim
    mov     z19.s, s19
    // ── Pass 3: dx = inv_std * (dy*gamma - dm - x_hat*ds) ──
    mov     x8, x24
    mov     x11, x16
    mov     x12, x17
    mov     w10, w22
.L_lnb_dx:
    ld1w    {z0.s}, p0/z, [x8]    // x
    ld1w    {z1.s}, p0/z, [x11]   // gamma
    ld1w    {z2.s}, p0/z, [x12]   // dy
    fsub    z3.s, z0.s, z16.s
    fmul    z3.s, z3.s, z17.s      // x_hat
    fmul    z4.s, z2.s, z1.s       // dy * gamma
    fsub    z4.s, z4.s, z19.s      // - dm
    fmul    z5.s, z3.s, z18.s      // x_hat * ds
    fsub    z4.s, z4.s, z5.s       // dy*gamma - dm - x_hat*ds
    fmul    z4.s, z4.s, z17.s      // * inv_std
    st1w    {z4.s}, p0, [x13]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x12, x12, x9, lsl #2
    add     x13, x13, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_lnb_dx
    b       .L_dispatch
// ================================================================
// ROPE_BACKWARD_FP32 (0x72) — RoPE inverse rotation
// Same frequency computation as forward, but applies inverse rotation:
//   dx_even = dy_even*cos + dy_odd*sin
//   dx_odd  = -dy_even*sin + dy_odd*cos
// Implemented by negating sin after computation.
//
// Encoding: [0x72][dim:u32][pos:u32][theta:f32][dy_ptr:u64][dx_ptr:u64]
// ================================================================
.L_op_rope_backward_fp32:
    ldr     w22, [x19]
    ldr     w23, [x19, #4]
    ldr     w24, [x19, #8]
    add     x19, x19, #12
    ldr     x8, [x19], #8
    ldr     x11, [x19], #8
    lsr     w26, w22, #1
    cbz     w26, .L_ropeb_done
    ptrue   p0.s
    cntw    x9
    // ── Compute ratio = theta^(2/dim) (same as forward) ──
    fmov    s0, w24
    fmov    w4, s0
    ubfx    w5, w4, #23, #8
    sub     w5, w5, #127
    scvtf   s1, w5
    mov     w6, #127
    bfi     w4, w6, #23, #8
    fmov    s2, w4
    fmov    s3, #1.0
    fsub    s2, s2, s3
    movz    w6, #0x3E94, lsl #16
    movk    w6, #0x3014
    fmov    s4, w6
    movz    w6, #0xBEFB, lsl #16
    movk    w6, #0xD464
    fmov    s5, w6
    movz    w6, #0x3F7F, lsl #16
    movk    w6, #0xF972
    fmov    s6, w6
    fmul    s4, s4, s2
    fadd    s4, s4, s5
    fmul    s4, s4, s2
    fadd    s4, s4, s6
    fmul    s4, s4, s2
    movz    w6, #0x3F31, lsl #16
    movk    w6, #0x7218
    fmov    s5, w6
    fmul    s1, s1, s5
    fadd    s1, s1, s4
    fmov    s6, #2.0
    ucvtf   s7, w22
    fdiv    s6, s6, s7
    fmul    s6, s6, s1
    movz    w6, #0x3FB8, lsl #16
    movk    w6, #0xAA3B
    fmov    s10, w6
    fmul    s7, s6, s10
    frintm  s8, s7
    fsub    s9, s7, s8
    movz    w6, #0x3C1D, lsl #16
    movk    w6, #0x955A
    fmov    s12, w6
    movz    w6, #0x3D63, lsl #16
    movk    w6, #0x5847
    fmov    s13, w6
    movz    w6, #0x3E75, lsl #16
    movk    w6, #0xFDF0
    fmov    s14, w6
    fmul    s15, s12, s9
    fadd    s15, s15, s13
    fmul    s15, s15, s9
    fadd    s15, s15, s14
    fmul    s15, s15, s9
    movz    w6, #0x3F31, lsl #16
    movk    w6, #0x7218
    fmov    s14, w6
    fadd    s15, s15, s14
    fmul    s15, s15, s9
    fmov    s14, #1.0
    fadd    s15, s15, s14
    fcvtzs  w6, s8
    add     w6, w6, #127
    lsl     w6, w6, #23
    fmov    s14, w6
    fmul    s0, s15, s14
    // ── Build power vector ──
    add     x14, sp, #128
    fmov    s2, #1.0
    mov     w12, #0
.L_ropeb_pw:
    cmp     w12, w9
    b.ge    .L_ropeb_pw_done
    str     s2, [x14, w12, uxtw #2]
    fmul    s2, s2, s0
    add     w12, w12, #1
    b       .L_ropeb_pw
.L_ropeb_pw_done:
    ld1w    {z29.s}, p0/z, [x14]
    mov     z30.s, s2
    ucvtf   s3, w23
    mov     z31.s, s3
    // ── Trig constants ──
    movz    w4, #0x4049, lsl #16
    movk    w4, #0x0FDB
    fmov    s16, w4
    mov     z20.s, s16
    movz    w4, #0x3EA2, lsl #16
    movk    w4, #0xF983
    fmov    s16, w4
    mov     z21.s, s16
    movz    w4, #0xBE2A, lsl #16
    movk    w4, #0xAAAB
    fmov    s16, w4
    mov     z22.s, s16
    movz    w4, #0x3C08, lsl #16
    movk    w4, #0x8889
    fmov    s16, w4
    mov     z23.s, s16
    movz    w4, #0xB950, lsl #16
    movk    w4, #0x0D01
    fmov    s16, w4
    mov     z27.s, s16
    movz    w4, #0xBF00, lsl #16
    movk    w4, #0x0000
    fmov    s16, w4
    mov     z24.s, s16
    movz    w4, #0x3D2A, lsl #16
    movk    w4, #0xAAAB
    fmov    s16, w4
    mov     z25.s, s16
    movz    w4, #0xBAB6, lsl #16
    movk    w4, #0x0B61
    fmov    s16, w4
    mov     z28.s, s16
    fmov    z26.s, #1.0
    // ── Vectorized inverse rotation loop ──
    mov     w12, #0
    whilelt p1.s, w12, w26
.L_ropeb_vec:
    b.none  .L_ropeb_done
    movprfx z0, z31
    fdiv    z0.s, p1/m, z0.s, z29.s
    fmul    z1.s, z0.s, z21.s
    frintn  z1.s, p0/m, z1.s
    fcvtzs  z7.s, p0/m, z1.s
    fmls    z0.s, p0/m, z1.s, z20.s
    and     z7.s, z7.s, #1
    lsl     z7.s, z7.s, #31
    fmul    z1.s, z0.s, z0.s
    mov     z2.d, z27.d
    fmad    z2.s, p0/m, z1.s, z23.s
    fmad    z2.s, p0/m, z1.s, z22.s
    fmad    z2.s, p0/m, z1.s, z26.s
    fmul    z2.s, p0/m, z2.s, z0.s  // sin(r)
    mov     z3.d, z28.d
    fmad    z3.s, p0/m, z1.s, z25.s
    fmad    z3.s, p0/m, z1.s, z24.s
    fmad    z3.s, p0/m, z1.s, z26.s  // cos(r)
    eor     z2.d, z2.d, z7.d
    eor     z3.d, z3.d, z7.d
    // ── BACKWARD: negate sin for inverse rotation ──
    fneg    z2.s, p0/m, z2.s
    // LD2W deinterleaves
    ld2w    {z4.s, z5.s}, p1/z, [x8]
    mov     z6.d, z4.d
    fmul    z4.s, p0/m, z4.s, z3.s
    fmls    z4.s, p0/m, z5.s, z2.s
    fmul    z5.s, p0/m, z5.s, z3.s
    fmla    z5.s, p0/m, z6.s, z2.s
    st2w    {z4.s, z5.s}, p1, [x11]
    add     x8, x8, x9, lsl #3
    add     x11, x11, x9, lsl #3
    fmul    z29.s, p0/m, z29.s, z30.s
    add     w12, w12, w9
    whilelt p1.s, w12, w26
    b       .L_ropeb_vec
.L_ropeb_done:
    b       .L_dispatch
// ================================================================
// CROSS_ENTROPY_FP32 (0x73) — Cross-entropy loss + gradient
// loss = -log(softmax(logits)[label])
// grad[i] = softmax[i] - (i == label ? 1 : 0)
//
// Three passes: (1) find max, (2) exp + sum, (3) grad + loss
//
// Encoding: [0x73][dim:u32][label:u32][logits:u64][grad_out:u64][loss_out:u64]
// dim must be a multiple of 16.
// ================================================================
.L_op_cross_entropy_fp32:
    ldr     w22, [x19]             // dim
    ldr     w23, [x19, #4]        // label (correct class index)
    add     x19, x19, #8
    ldr     x8, [x19], #8         // logits
    ldr     x11, [x19], #8        // grad_out
    ldr     x12, [x19], #8        // loss_out
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    LOAD_EXP_CONSTANTS
    mov     x14, x8                // save logits
    mov     x15, x11               // save grad_out
    // ── Pass 1: find max ──
    movz    w4, #0xFF80, lsl #16
    fmov    s16, w4
    mov     z16.s, s16
    mov     w10, w22
.L_ce_max:
    ld1w    {z0.s}, p0/z, [x8]
    fmax    z16.s, p0/m, z16.s, z0.s
    add     x8, x8, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_ce_max
    fmaxv   s16, p0, z16.s
    mov     z16.s, s16              // broadcast max
    // ── Pass 2: exp(logits - max), accumulate sum, store exp values ──
    fmov    z17.s, #0.0            // sum accumulator
    mov     x8, x14
    mov     x11, x15
    mov     w10, w22
.L_ce_exp:
    ld1w    {z0.s}, p0/z, [x8]
    fsub    z0.s, z0.s, z16.s
    EXP_POLY_Z0_TO_Z4
    st1w    {z4.s}, p0, [x11]     // store exp for gradient
    fadd    z17.s, z17.s, z4.s
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_ce_exp
    faddv   s17, p0, z17.s         // total sum
    // ── Compute loss = -log(exp[label]/sum) = log(sum) - (logits[label] - max) ──
    // log(sum) via z-vector pipeline (lane 0 only)
    mov     z0.s, s17              // broadcast sum into z0
    lsr     z1.s, z0.s, #23
    and     z1.s, z1.s, #0xFF
    mov     z2.s, #127
    sub     z1.s, z1.s, z2.s
    scvtf   z1.s, p0/m, z1.s       // exponent as float
    mov     z3.d, z0.d
    and     z3.s, z3.s, #0x007FFFFF
    orr     z3.s, z3.s, #0x3F800000
    fsub    z3.s, z3.s, z26.s      // t = m - 1
    movz    w4, #0x3E94, lsl #16
    movk    w4, #0x3014
    fmov    s8, w4
    mov     z8.s, s8               // a3
    movz    w4, #0xBEFB, lsl #16
    movk    w4, #0xD464
    fmov    s9, w4
    mov     z9.s, s9               // a2
    movz    w4, #0x3F7F, lsl #16
    movk    w4, #0xF972
    fmov    s10, w4
    mov     z10.s, s10             // a1
    fmul    z4.s, z8.s, z3.s
    fadd    z4.s, z4.s, z9.s
    fmul    z4.s, z4.s, z3.s
    fadd    z4.s, z4.s, z10.s
    fmul    z4.s, z4.s, z3.s       // ln(m)
    fmul    z1.s, z1.s, z27.s      // e * ln(2)
    fadd    z0.s, z1.s, z4.s       // log(sum)
    fmov    s18, s0                // extract lane 0 → s18
    // logits[label] - max
    ldr     s19, [x14, w23, uxtw #2]  // logits[label]
    fmov    w4, s16                 // max (from z16 lane 0)
    fmov    s0, w4
    fsub    s19, s19, s0            // logits[label] - max
    fsub    s18, s18, s19           // loss = log(sum) - (logits[label] - max)
    str     s18, [x12]             // store loss scalar
    // ── Pass 3: grad = softmax - one_hot = exp/sum - delta ──
    fmov    s18, #1.0
    fdiv    s17, s18, s17           // 1/sum
    mov     z17.s, s17              // broadcast 1/sum
    mov     z18.s, w23              // broadcast label index
    fmov    z19.s, #1.0
    mov     x11, x15
    mov     w10, #0                // element index
.L_ce_grad:
    cmp     w10, w22
    b.ge    .L_ce_done
    ld1w    {z0.s}, p0/z, [x11]   // stored exp values
    fmul    z0.s, z0.s, z17.s      // softmax = exp / sum
    // Subtract 1.0 at the label position using index compare
    index   z1.s, w10, #1         // z1 = [w10, w10+1, ..., w10+15]
    cmpeq   p1.s, p0/z, z1.s, z18.s // p1 set only at label lane
    fsub    z0.s, p1/m, z0.s, z19.s // softmax[label] -= 1.0
    st1w    {z0.s}, p0, [x11]
    add     x11, x11, x9, lsl #2
    add     w10, w10, w9
    b       .L_ce_grad
.L_ce_done:
    b       .L_dispatch
// ================================================================
// ELEMENTWISE_SUB_FP32 (0x74) — out[i] = a[i] - b[i]
//
// Encoding: [0x74][count:u32][a_ptr:u64][b_ptr:u64][out_ptr:u64]
// count must be a multiple of 16.
// ================================================================
.L_op_elementwise_sub_fp32:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8         // a_ptr
    ldr     x11, [x19], #8        // b_ptr
    ldr     x13, [x19], #8        // out_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    mov     w10, w22
.L_esub_loop:
    ld1w    {z0.s}, p0/z, [x8]
    ld1w    {z1.s}, p0/z, [x11]
    fsub    z0.s, z0.s, z1.s
    st1w    {z0.s}, p0, [x13]
    add     x8, x8, x9, lsl #2
    add     x11, x11, x9, lsl #2
    add     x13, x13, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_esub_loop
    b       .L_dispatch
// ================================================================
// Q4_K_GEMV (0x75) — Quantized GEMV for ggml Q4_K format
//
// Super-block: 256 values in 144 bytes:
//   [d:fp16][dmin:fp16][scales:12 bytes][qs:128 bytes (4-bit packed)]
// 8 sub-blocks of 32 values each. Each sub-block has a 6-bit scale and 6-bit min
// packed into the 12-byte scales array.
// Dequant: val = d * sc * (nibble) - dmin * m
//
// Encoding: [0x75][M:u32][K:u32][in:u64][W:u64][out:u64]
// K must be a multiple of 256.
// ================================================================
.L_op_q4_k_gemv:
    ldr     w22, [x19]             // M
    add     x19, x19, #4
    ldr     w23, [x19]             // K
    add     x19, x19, #4
    ldr     x8, [x19], #8         // input_ptr (fp32)
    ldr     x11, [x19], #8        // weights_ptr (Q4_K blocks)
    ldr     x13, [x19], #8        // output_ptr (fp32)
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9                     // 16
    lsr     w24, w23, #8           // num_super_blocks = K / 256
    mov     x17, x8                // save input base
.L_q4k_row:
    cbz     w22, .L_dispatch
    mov     z4.d, #0               // accumulator
    mov     x8, x17                // reset input ptr
    mov     w10, w24               // super-block counter
.L_q4k_sblock:
    cbz     w10, .L_q4k_store
    // Parse super-block header
    ldr     h0, [x11]             // d (fp16)
    ldr     h1, [x11, #2]         // dmin (fp16)
    fcvt    s0, h0                // d as fp32
    fcvt    s1, h1                // dmin as fp32
    fneg    s1, s1                // -dmin (pre-negate for fmla)
    add     x14, x11, #4          // &scales[0] (12 bytes)
    add     x15, x11, #16         // &qs[0] (128 bytes of 4-bit packed)
    // Process 8 sub-blocks of 32 values each
    mov     w12, #0               // sub-block index
.L_q4k_sub:
    cmp     w12, #8
    b.ge    .L_q4k_sblock_next
    // Extract 6-bit scale and min for this sub-block from the packed 12-byte array
    // scales layout: sc[0..3] in low 6 bits of bytes 0..3, m[0..3] in low 6 bits of bytes 4..7
    // sc[4..7] in low 4 bits of bytes 8..11, m[4..7] in high 4 bits of bytes 8..11
    // combined with high 2 bits from bytes 0..7
    // Simplified: for sub-block j<4: sc = scales[j] & 0x3F, m = scales[j+4] & 0x3F
    //             for sub-block j>=4: sc = (scales[j+4]&0xF) | ((scales[j-4]>>6)<<4)
    //                                  m = (scales[j+4]>>4)  | ((scales[j]>>6)<<4)
    cmp     w12, #4
    b.ge    .L_q4k_scale_hi
    // Low 4 sub-blocks: simple 6-bit extraction
    ldrb    w3, [x14, w12, uxtw]       // scales[j]
    and     w3, w3, #0x3F              // sc = low 6 bits
    add     w5, w12, #4
    ldrb    w6, [x14, w5, uxtw]        // scales[j+4]
    and     w6, w6, #0x3F              // m = low 6 bits
    b       .L_q4k_scale_done
.L_q4k_scale_hi:
    // High 4 sub-blocks: reconstruct from split fields
    sub     w5, w12, #4                 // j-4
    add     w7, w12, #4                 // j+4 (index 8..11)
    ldrb    w3, [x14, w7, uxtw]        // scales[j+4]
    ldrb    w6, [x14, w5, uxtw]        // scales[j-4]
    and     w16, w3, #0x0F             // low 4 bits of scales[j+4]
    lsr     w6, w6, #6                 // high 2 bits of scales[j-4]
    orr     w3, w16, w6, lsl #4        // sc = low4 | (high2 << 4)
    ldrb    w6, [x14, w12, uxtw]       // scales[j]
    lsr     w16, w3, #4                // high 4 bits of scales[j+4] (wait, reread)
    // Actually: m = (scales[j+4]>>4) | ((scales[j]>>6)<<4)
    ldrb    w16, [x14, w7, uxtw]       // re-read scales[j+4]
    lsr     w16, w16, #4               // high nibble
    lsr     w6, w6, #6                 // high 2 bits of scales[j]
    orr     w6, w16, w6, lsl #4        // m = high4 | (high2 << 4)
.L_q4k_scale_done:
    // w3 = sc (6-bit scale), w6 = m (6-bit min)
    // Compute d_sc = d * sc, dmin_m = -dmin * m
    ucvtf   s2, w3                 // (float)sc
    fmul    s2, s0, s2             // d * sc
    mov     z16.s, s2              // broadcast d*sc
    ucvtf   s3, w6                 // (float)m
    fmul    s3, s1, s3             // -dmin * m
    mov     z17.s, s3              // broadcast -dmin*m
    // Load 32 packed 4-bit quants (16 bytes) for this sub-block
    // Each byte has low nibble = element 2i, high nibble = element 2i+1
    // Sub-block j starts at qs[j*16]
    lsl     w5, w12, #4            // j * 16
    add     x16, x15, x5           // &qs[j*16]
    // First 16 values: low nibbles
    ld1b    {z0.s}, p0/z, [x16]   // load 16 bytes, zero-extend to 32-bit
    mov     z2.d, z0.d
    and     z2.s, z2.s, #0x0F     // low nibble
    ucvtf   z2.s, p0/m, z2.s      // to float
    fmul    z2.s, z2.s, z16.s     // * d*sc
    fadd    z2.s, z2.s, z17.s     // + (-dmin*m)
    ld1w    {z1.s}, p0/z, [x8]
    fmla    z4.s, p0/m, z2.s, z1.s
    add     x8, x8, #64
    // Next 16 values: high nibbles
    lsr     z3.s, z0.s, #4
    and     z3.s, z3.s, #0x0F
    ucvtf   z3.s, p0/m, z3.s
    fmul    z3.s, z3.s, z16.s
    fadd    z3.s, z3.s, z17.s
    ld1w    {z1.s}, p0/z, [x8]
    fmla    z4.s, p0/m, z3.s, z1.s
    add     x8, x8, #64
    add     w12, w12, #1
    b       .L_q4k_sub
.L_q4k_sblock_next:
    add     x11, x11, #144        // advance to next super-block
    sub     w10, w10, #1
    b       .L_q4k_sblock
.L_q4k_store:
    faddv   s4, p0, z4.s
    str     s4, [x13], #4
    sub     w22, w22, #1
    b       .L_q4k_row
// ================================================================
// Q2_K_GEMV (0x76) — Quantized GEMV for ggml Q2_K format
//
// Super-block: 256 values in 84 bytes:
//   [scales:16 bytes (4-bit sc + 4-bit m per sub-block)]
//   [qs:64 bytes (2-bit packed, 4 per byte)]
//   [d:fp16][dmin:fp16]
// 16 sub-blocks of 16 values each.
// Dequant: val = d * (sc & 0xF) * q - dmin * (sc >> 4)
//
// Encoding: [0x76][M:u32][K:u32][in:u64][W:u64][out:u64]
// K must be a multiple of 256.
// ================================================================
.L_op_q2_k_gemv:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     w23, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8         // input_ptr
    ldr     x11, [x19], #8        // weights_ptr
    ldr     x13, [x19], #8        // output_ptr
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    lsr     w24, w23, #8           // num_super_blocks = K / 256
    mov     x17, x8
.L_q2k_row:
    cbz     w22, .L_dispatch
    mov     z4.d, #0
    mov     x8, x17
    mov     w10, w24
.L_q2k_sblock:
    cbz     w10, .L_q2k_store
    // Q2_K layout: scales[16], qs[64], d(fp16), dmin(fp16)
    add     x14, x11, #0          // &scales[0]
    add     x15, x11, #16         // &qs[0]
    ldr     h0, [x11, #80]        // d
    ldr     h1, [x11, #82]        // dmin
    fcvt    s0, h0
    fcvt    s1, h1
    fneg    s1, s1
    // 16 sub-blocks of 16 values, each uses 4 bytes of qs (4 values per byte × 4 bytes)
    mov     w12, #0
.L_q2k_sub:
    cmp     w12, #16
    b.ge    .L_q2k_sblock_next
    // Extract scale and min from scales[j]: low nibble = sc, high nibble = m
    ldrb    w3, [x14, w12, uxtw]
    and     w5, w3, #0x0F          // sc
    lsr     w6, w3, #4             // m
    ucvtf   s2, w5
    fmul    s2, s0, s2             // d * sc
    mov     z16.s, s2
    ucvtf   s3, w6
    fmul    s3, s1, s3             // -dmin * m
    mov     z17.s, s3
    // Load 4 bytes of 2-bit quants (16 values) for this sub-block
    lsl     w5, w12, #2            // j * 4
    add     x16, x15, x5
    // Load 4 bytes into a GP register, extract 16 × 2-bit values
    ldr     w3, [x16]              // 32 bits = 16 × 2-bit values
    // Expand to 16 fp32 values on stack, then load as z-vector
    // Use shifts to extract each pair of bits into z-vector lanes
    // Strategy: load 4 bytes as z-vector, shift/mask to extract 2-bit fields
    // Load the 4 bytes zero-extended into z0 lanes
    ld1b    {z0.s}, p0/z, [x16]   // loads 16 bytes but only 4 are valid; 1 byte per lane
    // But we need 16 values from 4 bytes (4 per byte). Different approach:
    // Replicate each byte 4 times, then shift by {0,2,4,6} and mask
    // Simpler: broadcast the 32-bit word, shift by lane-dependent amounts, mask
    mov     z0.s, w3               // broadcast the 32-bit packed word
    // Lane i extracts bits (2*i)..(2*i+1): shift right by 2*i, mask with 0x3
    index   z1.s, #0, #2          // z1 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    lsr     z0.s, p0/m, z0.s, z1.s // shift each lane by its index*2
    and     z0.s, z0.s, #0x3       // mask to 2 bits
    ucvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z16.s      // * d*sc
    fadd    z0.s, z0.s, z17.s      // + (-dmin*m)
    ld1w    {z1.s}, p0/z, [x8]
    fmla    z4.s, p0/m, z0.s, z1.s
    add     x8, x8, #64
    add     w12, w12, #1
    b       .L_q2k_sub
.L_q2k_sblock_next:
    add     x11, x11, #84
    sub     w10, w10, #1
    b       .L_q2k_sblock
.L_q2k_store:
    faddv   s4, p0, z4.s
    str     s4, [x13], #4
    sub     w22, w22, #1
    b       .L_q2k_row
// ================================================================
// Q3_K_GEMV (0x77) — Quantized GEMV for ggml Q3_K format
//
// Super-block: 256 values in 110 bytes:
//   [hmask:32 bytes (high bits)][qs:64 bytes (low 2 bits packed)]
//   [scales:12 bytes][d:fp16]
// 3-bit quants: low 2 bits in qs, high 1 bit in hmask.
// val = d * sc * (q3 - 4) where q3 = (qs_2bit | (hbit << 2)), range [0,7], centered at 4.
//
// Encoding: [0x77][M:u32][K:u32][in:u64][W:u64][out:u64]
// K must be a multiple of 256.
// ================================================================
.L_op_q3_k_gemv:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     w23, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8
    ldr     x11, [x19], #8
    ldr     x13, [x19], #8
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    lsr     w24, w23, #8
    mov     x17, x8
.L_q3k_row:
    cbz     w22, .L_dispatch
    mov     z4.d, #0
    mov     x8, x17
    mov     w10, w24
.L_q3k_sblock:
    cbz     w10, .L_q3k_store
    // Q3_K layout: hmask[32], qs[64], scales[12], d(fp16)
    add     x14, x11, #0          // &hmask[0]
    add     x15, x11, #32         // &qs[0]
    add     x16, x11, #96         // &scales[0]
    ldr     h0, [x11, #108]       // d
    fcvt    s0, h0
    // 8 sub-blocks of 32 values each
    mov     w12, #0
.L_q3k_sub:
    cmp     w12, #8
    b.ge    .L_q3k_sblock_next
    // Extract scale for this sub-block (same packed 6-bit format as Q4_K but simpler)
    // For Q3_K, scales are stored as 6-bit signed values
    // Sub-blocks 0-3: scales[j] & 0x3F, adjusted by high bits
    // The scale extraction is complex; use simplified approach:
    // scales are 32 values packed into 12 bytes, but for GEMV we treat them
    // as 8 6-bit values. Lower 4: scales[j]&0x3F. Upper 4: reconstructed.
    cmp     w12, #4
    b.ge    .L_q3k_scale_hi
    ldrb    w3, [x16, w12, uxtw]
    and     w3, w3, #0x3F
    b       .L_q3k_scale_done2
.L_q3k_scale_hi:
    sub     w5, w12, #4
    add     w7, w12, #4
    ldrb    w3, [x16, w7, uxtw]
    and     w3, w3, #0x0F
    ldrb    w6, [x16, w5, uxtw]
    lsr     w6, w6, #6
    orr     w3, w3, w6, lsl #4
.L_q3k_scale_done2:
    // Q3_K scales are centered: subtract 32 to get signed scale
    sub     w3, w3, #32
    scvtf   s2, w3
    fmul    s2, s0, s2             // d * signed_scale
    mov     z16.s, s2
    // Load 2-bit quants: 8 bytes for 32 values (4 per byte)
    lsl     w5, w12, #3            // j * 8
    add     x3, x15, x5
    // Process in two halves of 16
    // First 16: bytes 0-3 of sub-block qs, each byte has 4×2-bit
    ldr     w5, [x3]              // 4 bytes = 16 × 2-bit
    mov     z0.s, w5
    index   z1.s, #0, #2
    lsr     z0.s, p0/m, z0.s, z1.s
    and     z0.s, z0.s, #0x3
    // Load high bits from hmask
    // hmask is 32 bytes = 256 bits. Bit (j*32 + i) is the high bit for element j*32+i.
    // For sub-block j, elements j*32..j*32+15: bits are at hmask byte positions
    lsl     w5, w12, #2            // j * 4 (byte offset for first 16 bits in hmask, 2 bits per byte × 4 = 8... no)
    // hmask bit layout: bit i of hmask corresponds to element i (sequential)
    // sub-block j covers elements j*32..j*32+31
    // hmask byte (j*32+i)/8, bit (j*32+i)%8
    // For first 16 elements of sub-block j: elements j*32..j*32+15
    // These span hmask bytes j*4..j*4+1 (16 bits = 2 bytes)
    lsl     w5, w12, #2            // j * 4
    ldrh    w6, [x14, w5, uxtw]   // 16 bits of hmask for first 16 elements
    mov     z2.s, w6
    index   z3.s, #0, #1
    lsr     z2.s, p0/m, z2.s, z3.s
    and     z2.s, z2.s, #0x1      // high bit per element
    lsl     z2.s, z2.s, #2        // shift to bit position 2
    orr     z0.s, z0.s, z2.s      // combine: q3 = low2 | (high1 << 2)
    mov     z2.s, #4
    sub     z0.s, z0.s, z2.s      // center: q3 - 4, range [-4, +3]
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z16.s
    ld1w    {z1.s}, p0/z, [x8]
    fmla    z4.s, p0/m, z0.s, z1.s
    add     x8, x8, #64
    // Second 16 elements
    ldr     w5, [x3, #4]          // next 4 bytes
    mov     z0.s, w5
    lsr     z0.s, p0/m, z0.s, z1.s // reuse z1 index? No, z1 was clobbered by input load
    index   z1.s, #0, #2
    lsr     z0.s, p0/m, z0.s, z1.s
    and     z0.s, z0.s, #0x3
    lsl     w5, w12, #2
    add     w5, w5, #2             // offset +2 bytes for next 16 bits
    ldrh    w6, [x14, w5, uxtw]
    mov     z2.s, w6
    index   z3.s, #0, #1
    lsr     z2.s, p0/m, z2.s, z3.s
    and     z2.s, z2.s, #0x1
    lsl     z2.s, z2.s, #2
    orr     z0.s, z0.s, z2.s
    mov     z2.s, #4
    sub     z0.s, z0.s, z2.s
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z16.s
    ld1w    {z1.s}, p0/z, [x8]
    fmla    z4.s, p0/m, z0.s, z1.s
    add     x8, x8, #64
    add     w12, w12, #1
    b       .L_q3k_sub
.L_q3k_sblock_next:
    add     x11, x11, #110
    sub     w10, w10, #1
    b       .L_q3k_sblock
.L_q3k_store:
    faddv   s4, p0, z4.s
    str     s4, [x13], #4
    sub     w22, w22, #1
    b       .L_q3k_row
// ================================================================
// Q5_K_GEMV (0x78) — Quantized GEMV for ggml Q5_K format
//
// Super-block: 256 values in 176 bytes:
//   [d:fp16][dmin:fp16][scales:12 bytes][qh:32 bytes (high bits)][qs:128 bytes (low 4 bits)]
// 5-bit quants: low 4 bits in qs, high 1 bit in qh.
// 8 sub-blocks of 32 values each.
// Dequant: val = d * sc * q5 - dmin * m, where q5 = low4 | (hbit << 4)
//
// Encoding: [0x78][M:u32][K:u32][in:u64][W:u64][out:u64]
// K must be a multiple of 256.
// ================================================================
.L_op_q5_k_gemv:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     w23, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8
    ldr     x11, [x19], #8
    ldr     x13, [x19], #8
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    lsr     w24, w23, #8
    mov     x17, x8
.L_q5k_row:
    cbz     w22, .L_dispatch
    mov     z4.d, #0
    mov     x8, x17
    mov     w10, w24
.L_q5k_sblock:
    cbz     w10, .L_q5k_store
    // Q5_K layout: d(fp16), dmin(fp16), scales[12], qh[32], qs[128]
    ldr     h0, [x11]
    ldr     h1, [x11, #2]
    fcvt    s0, h0
    fcvt    s1, h1
    fneg    s1, s1
    add     x14, x11, #4          // &scales[0]
    add     x15, x11, #16         // &qh[0] (32 bytes of high bits)
    add     x16, x11, #48         // &qs[0] (128 bytes of low 4-bit)
    mov     w12, #0
.L_q5k_sub:
    cmp     w12, #8
    b.ge    .L_q5k_sblock_next
    // Extract 6-bit scale/min (same as Q4_K)
    cmp     w12, #4
    b.ge    .L_q5k_scale_hi
    ldrb    w3, [x14, w12, uxtw]
    and     w3, w3, #0x3F
    add     w5, w12, #4
    ldrb    w6, [x14, w5, uxtw]
    and     w6, w6, #0x3F
    b       .L_q5k_scale_done
.L_q5k_scale_hi:
    sub     w5, w12, #4
    add     w7, w12, #4
    ldrb    w3, [x14, w7, uxtw]
    ldrb    w6, [x14, w5, uxtw]
    and     w16, w3, #0x0F
    lsr     w6, w6, #6
    orr     w3, w16, w6, lsl #4
    ldrb    w16, [x14, w7, uxtw]
    ldrb    w6, [x14, w12, uxtw]
    lsr     w16, w16, #4
    lsr     w6, w6, #6
    orr     w6, w16, w6, lsl #4
.L_q5k_scale_done:
    ucvtf   s2, w3
    fmul    s2, s0, s2
    mov     z16.s, s2              // d * sc
    ucvtf   s3, w6
    fmul    s3, s1, s3
    mov     z17.s, s3              // -dmin * m
    // Load low 4-bit quants (16 bytes = 32 nibbles for this sub-block)
    lsl     w5, w12, #4
    add     x3, x16, x5
    ld1b    {z0.s}, p0/z, [x3]    // 16 bytes → 16 lanes
    mov     z2.d, z0.d
    and     z2.s, z2.s, #0x0F     // low nibble (first 16 elements)
    // Load high bits from qh
    // qh has 32 bytes = 256 bits, one per element
    // sub-block j: elements j*32..j*32+31
    lsl     w5, w12, #2
    ldr     w6, [x15, w5, uxtw]   // 32 bits of qh for this sub-block
    // First 16 elements: bits 0..15
    and     w7, w6, #0xFFFF
    mov     z5.s, w7
    index   z6.s, #0, #1
    lsr     z5.s, p0/m, z5.s, z6.s
    and     z5.s, z5.s, #0x1
    lsl     z5.s, z5.s, #4         // shift to bit 4
    orr     z2.s, z2.s, z5.s       // q5 = low4 | (hbit << 4)
    ucvtf   z2.s, p0/m, z2.s
    fmul    z2.s, z2.s, z16.s
    fadd    z2.s, z2.s, z17.s
    ld1w    {z1.s}, p0/z, [x8]
    fmla    z4.s, p0/m, z2.s, z1.s
    add     x8, x8, #64
    // Second 16 elements: high nibbles + qh bits 16..31
    lsr     z3.s, z0.s, #4
    and     z3.s, z3.s, #0x0F
    lsr     w7, w6, #16            // bits 16..31
    mov     z5.s, w7
    lsr     z5.s, p0/m, z5.s, z6.s
    and     z5.s, z5.s, #0x1
    lsl     z5.s, z5.s, #4
    orr     z3.s, z3.s, z5.s
    ucvtf   z3.s, p0/m, z3.s
    fmul    z3.s, z3.s, z16.s
    fadd    z3.s, z3.s, z17.s
    ld1w    {z1.s}, p0/z, [x8]
    fmla    z4.s, p0/m, z3.s, z1.s
    add     x8, x8, #64
    add     w12, w12, #1
    b       .L_q5k_sub
.L_q5k_sblock_next:
    add     x11, x11, #176
    sub     w10, w10, #1
    b       .L_q5k_sblock
.L_q5k_store:
    faddv   s4, p0, z4.s
    str     s4, [x13], #4
    sub     w22, w22, #1
    b       .L_q5k_row
// ================================================================
// Q6_K_GEMV (0x79) — Quantized GEMV for ggml Q6_K format
//
// Super-block: 256 values in 210 bytes:
//   [ql:128 bytes (low 4 bits)][qh:64 bytes (high 2 bits)]
//   [scales:16 bytes (int8 per sub-block)][d:fp16]
// 6-bit quants: low 4 bits in ql, high 2 bits in qh.
// 16 sub-blocks of 16 values each.
// Dequant: val = d * sc * (q6 - 32), q6 = low4 | (high2 << 4), range [0,63] centered at 32.
//
// Encoding: [0x79][M:u32][K:u32][in:u64][W:u64][out:u64]
// K must be a multiple of 256.
// ================================================================
.L_op_q6_k_gemv:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     w23, [x19]
    add     x19, x19, #4
    ldr     x8, [x19], #8
    ldr     x11, [x19], #8
    ldr     x13, [x19], #8
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    lsr     w24, w23, #8
    mov     x17, x8
.L_q6k_row:
    cbz     w22, .L_dispatch
    mov     z4.d, #0
    mov     x8, x17
    mov     w10, w24
.L_q6k_sblock:
    cbz     w10, .L_q6k_store
    // Q6_K layout: ql[128], qh[64], scales[16], d(fp16)
    add     x14, x11, #0          // &ql[0]
    add     x15, x11, #128        // &qh[0]
    add     x16, x11, #192        // &scales[0]
    ldr     h0, [x11, #208]       // d
    fcvt    s0, h0
    // 16 sub-blocks of 16 values each
    mov     w12, #0
.L_q6k_sub:
    cmp     w12, #16
    b.ge    .L_q6k_sblock_next
    // Scale: int8 from scales[j]
    ldrsb   w3, [x16, w12, uxtw]  // signed int8 scale
    scvtf   s2, w3
    fmul    s2, s0, s2             // d * scale
    mov     z16.s, s2
    // Load low 4 bits: ql has 128 bytes, 8 bytes per sub-block (but layout is interleaved)
    // Q6_K ql layout: for sub-block j, the low 4 bits are at ql[j*8..j*8+7]
    // But ql packing: byte i has two 4-bit values (low nibble = element 2i, high = element 2i+1)
    // So 8 bytes = 16 values (one sub-block's worth)
    lsl     w5, w12, #3            // j * 8
    add     x3, x14, x5
    ld1b    {z0.s}, p0/z, [x3]    // 16 bytes but only 8 valid for this sub-block
    // Actually ql has 128 bytes for 256 elements: 128/256 = 0.5 bytes per element = 4 bits each
    // ql is packed as nibbles, 2 per byte. Sub-block j's 16 elements use 8 bytes.
    // Load 8 bytes → 8 lanes. Need to split into 16 values.
    // Use whilelt for 8 elements, then expand nibbles
    mov     x5, #8
    whilelt p1.s, xzr, x5
    ld1b    {z0.s}, p1/z, [x3]    // 8 bytes into lanes 0..7
    // Low nibbles → elements 0,2,4,...,14 (even indices)
    // High nibbles → elements 1,3,5,...,15 (odd indices)
    // Strategy: expand each byte into 2 values
    mov     z2.d, z0.d
    and     z2.s, z2.s, #0x0F     // low nibbles (8 values in lanes 0..7)
    lsr     z3.s, z0.s, #4
    and     z3.s, z3.s, #0x0F     // high nibbles (8 values in lanes 0..7)
    // Load high 2 bits from qh
    // qh has 64 bytes for 256 elements: 64/256 = 0.25 bytes = 2 bits each
    // Sub-block j's 16 elements use 4 bytes of qh
    lsl     w5, w12, #2            // j * 4
    add     x3, x15, x5
    ldr     w6, [x3]              // 32 bits = 16 × 2-bit high values
    // Extract 2 bits per element using broadcast + shift + mask
    // Even elements (0,2,4..14) use bits 0,2,4,...,14 → need bits at positions 0,4,8,...,28
    // Wait, qh packing: 2 bits per element, sequential. Element i uses bits 2i..2i+1.
    // For 16 elements: bits 0..31 of the 32-bit word.
    // Even element i (0..7): combine qh bits with ql low nibble → q6 = low4 | (high2 << 4)
    // For the first 8 elements (from low nibbles):
    // qh bits for element 2k: bits at position 2*(2k) = 4k, 4k+1
    // Actually simpler: element i has qh bits at position 2*i, 2*i+1
    // First 8 values (even indices in the sub-block, from low nibbles):
    // Elements 0,1,...,7 in the sub-block
    // But our split: z2 = low nibble values (sub-block elements 0,2,4,...,14)
    //                z3 = high nibble values (sub-block elements 1,3,5,...,15)
    // Actually ql packing for Q6_K is different — it's sequential nibbles, not interleaved.
    // ql byte i: low nibble = element 2i, high nibble = element 2i+1
    // So for sub-block j (16 elements): 8 bytes, element j*16+2k in low nibble of byte k,
    //                                            element j*16+2k+1 in high nibble of byte k
    // qh for these: element j*16+n has 2-bit high at qh bit position j*16+n (but packed differently)
    // qh is packed: 4 bytes per sub-block, element n within sub-block uses bits 2n..2n+1
    // So for even elements (n=0,2,4,...,14 in sub-block): qh bit positions 0,4,8,...,28
    // For odd elements (n=1,3,5,...,15): qh bit positions 2,6,10,...,30
    // Elements from z2 (low nibble, n=0,2,...,14): qh_even
    // Elements from z3 (high nibble, n=1,3,...,15): qh_odd
    // Extract even-indexed 2-bit fields: shift by [0,4,8,...,28], mask 0x3
    mov     z5.s, w6               // broadcast qh word
    index   z6.s, #0, #4          // [0, 4, 8, 12, 16, 20, 24, 28] for even elements
    lsr     z5.s, p1/m, z5.s, z6.s
    and     z5.s, z5.s, #0x3
    lsl     z5.s, z5.s, #4
    orr     z2.s, z2.s, z5.s       // q6_even = low4 | (high2 << 4)
    // Odd elements: shift by [2,6,10,...,30]
    mov     z5.s, w6
    index   z6.s, #2, #4          // [2, 6, 10, 14, 18, 22, 26, 30]
    lsr     z5.s, p1/m, z5.s, z6.s
    and     z5.s, z5.s, #0x3
    lsl     z5.s, z5.s, #4
    orr     z3.s, z3.s, z5.s       // q6_odd = low4 | (high2 << 4)
    // Center: subtract 32
    mov     z5.s, #32
    sub     z2.s, z2.s, z5.s       // q6_even - 32
    sub     z3.s, z3.s, z5.s       // q6_odd - 32
    // Convert and accumulate even elements (first 8 values)
    scvtf   z2.s, p1/m, z2.s
    fmul    z2.s, z2.s, z16.s
    // We have 8 values in z2 lanes 0..7 and 8 values in z3 lanes 0..7
    // But input vector has 16 contiguous elements. Need to interleave.
    // Load input, split into even/odd, accumulate separately
    ld1w    {z7.s}, p0/z, [x8]    // 16 input values
    // Interleave z2 and z3 into 16 values: z2[0], z3[0], z2[1], z3[1], ...
    zip1    z0.s, z2.s, z3.s      // interleave low halves → 16 values
    scvtf   z3.s, p1/m, z3.s      // convert odd values (not yet done above)
    fmul    z3.s, z3.s, z16.s
    zip1    z0.s, z2.s, z3.s      // re-interleave after both are dequantized
    fmla    z4.s, p0/m, z0.s, z7.s
    add     x8, x8, #64
    add     w12, w12, #1
    b       .L_q6k_sub
.L_q6k_sblock_next:
    add     x11, x11, #210
    sub     w10, w10, #1
    b       .L_q6k_sblock
.L_q6k_store:
    faddv   s4, p0, z4.s
    str     s4, [x13], #4
    sub     w22, w22, #1
    b       .L_q6k_row
// ================================================================
// FLASH_ATTENTION_FP32 (0x7A) — Fused tiled flash attention (single head)
//
// Computes: out = softmax(Q @ K^T / sqrt(d)) @ V
// Uses online softmax (FlashAttention algorithm) to avoid materializing
// the full N×N score matrix. Processes 16×16 score tiles.
//
// Encoding: [0x7A][N:u32][d:u32][flags:u8][Q:u64][K:u64][V:u64][out:u64]
// N = sequence length (must be multiple of 16)
// d = head dimension (must be multiple of 16)
// flags bit 0: causal mask (mask where key position > query position)
// Q, K, V: N×d fp32 row-major matrices
// out: N×d fp32 output
//
// Tile geometry:
//   za2 = 16×16 score tile (Q_block @ K_block^T), then P after softmax
//   za3 = Q column chunk (16×16 load, vertical extract for Q columns)
//   za1 = K column chunk (same pattern for K columns)
//   za0 = output tile (P @ V_chunk accumulation)
//
// Stack layout (256 bytes):
//   [sp+0]:    m[16] (per-row max, 64 bytes)
//   [sp+64]:   l[16] (per-row sum, 64 bytes)
//   [sp+128]:  Q_ptr (x)      [sp+136]:  K_ptr (x)
//   [sp+144]:  V_ptr (x)      [sp+152]:  out_ptr (x)
//   [sp+160]:  N (w)          [sp+164]:  d (w)
//   [sp+168]:  flags (w)      [sp+172]:  d_blocks (w) = d/16
//   [sp+176]:  qi (w)         [sp+180]:  kj (w)
//   [sp+184]:  d_stride (x) = d * 4
//   [sp+192]:  rsqrt_d (f32)
// ================================================================
.L_op_flash_attention_fp32:
    ldr     w22, [x19]             // N
    ldr     w23, [x19, #4]        // d
    ldrb    w24, [x19, #8]        // flags
    add     x19, x19, #9
    ldr     x6, [x19], #8         // Q
    ldr     x7, [x19], #8         // K
    ldr     x8, [x19], #8         // V
    ldr     x11, [x19], #8        // out
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9                     // 16
    sub     sp, sp, #320
    str     x6, [sp, #128]
    str     x7, [sp, #136]
    str     x8, [sp, #144]
    str     x11, [sp, #152]
    stp     w22, w23, [sp, #160]
    str     w24, [sp, #168]
    lsr     w10, w23, #4           // d_blocks = d / 16
    str     w10, [sp, #172]
    lsl     x10, x23, #2           // d_stride = d * 4
    str     x10, [sp, #184]
    // Compute rsqrt(d) via frsqrte + Newton
    ucvtf   s0, w23
    frsqrte s1, s0
    fmul    s2, s1, s1
    fmul    s2, s0, s2
    frsqrts s2, s0, s2
    fmul    s1, s1, s2             // refined rsqrt(d)
    str     s1, [sp, #192]
    // ── Zero the output buffer: N × d floats ──
    mul     w10, w22, w23          // N * d
    mov     z0.d, #0
.L_fa_zero:
    cbz     w10, .L_fa_zero_done
    st1w    {z0.s}, p0, [x11]
    add     x11, x11, x9, lsl #2
    subs    w10, w10, w9
    b.ne    .L_fa_zero
.L_fa_zero_done:
    // ── Query block loop ──
    mov     w0, #0                 // qi = 0
.L_fa_qi:
    str     w0, [sp, #176]
    ldr     w22, [sp, #160]
    cmp     w0, w22
    b.ge    .L_fa_done
    // Initialize m = -inf, l = 0 for this query block
    movz    w4, #0xFF80, lsl #16
    fmov    s0, w4
    mov     z0.s, s0               // -inf
    st1w    {z0.s}, p0, [sp]      // m[0..15] = -inf
    fmov    z0.s, #0.0
    add     x16, sp, #64
    st1w    {z0.s}, p0, [x16]    // l[0..15] = 0
    // ── Key block loop ──
    mov     w1, #0                 // kj = 0
.L_fa_kj:
    str     w1, [sp, #180]
    ldr     w22, [sp, #160]
    cmp     w1, w22
    b.ge    .L_fa_kj_done
    // Causal check: if causal and kj > qi+15, skip (all masked)
    ldr     w24, [sp, #168]
    tst     w24, #1
    b.eq    .L_fa_no_skip
    ldr     w0, [sp, #176]
    add     w0, w0, #15
    cmp     w1, w0
    b.gt    .L_fa_kj_done          // all keys in this block are future → done
.L_fa_no_skip:
    // ── Phase 1: Score = Q_block @ K_block^T → za2 (16×16) ──
    zero    {za2.s}
    ldr     x6, [sp, #128]         // Q
    ldr     x7, [sp, #136]         // K
    ldr     x10, [sp, #184]        // d_stride
    ldr     w0, [sp, #176]         // qi
    ldr     w1, [sp, #180]         // kj
    ldr     w23, [sp, #164]        // d
    // Q_base = Q + qi * d * 4
    mul     w3, w0, w23
    add     x6, x6, x3, lsl #2    // Q_block start
    // K_base = K + kj * d * 4
    mul     w3, w1, w23
    add     x7, x7, x3, lsl #2    // K_block start
    ldr     w12, [sp, #172]        // d_blocks = d / 16
    mov     w14, #0                // k_chunk offset (in elements)
.L_fa_score_k:
    cbz     w12, .L_fa_score_done
    // Load Q[qi:qi+16, k:k+16] into za3
    zero    {za3.s}
    add     x3, x6, x14, lsl #2   // Q + qi*d*4 + k*4
    mov     w15, #0
    ld1w    {z0.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z1.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z2.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z3.s}, p0/z, [x3]
    add     x3, x3, x10
    mova    za3h.s[w15, 0:3], {z0.s-z3.s}
    mov     w15, #4
    ld1w    {z0.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z1.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z2.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z3.s}, p0/z, [x3]
    add     x3, x3, x10
    mova    za3h.s[w15, 0:3], {z0.s-z3.s}
    mov     w15, #8
    ld1w    {z0.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z1.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z2.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z3.s}, p0/z, [x3]
    add     x3, x3, x10
    mova    za3h.s[w15, 0:3], {z0.s-z3.s}
    mov     w15, #12
    ld1w    {z0.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z1.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z2.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z3.s}, p0/z, [x3]
    mova    za3h.s[w15, 0:3], {z0.s-z3.s}
    // Load K[kj:kj+16, k:k+16] into za1
    zero    {za1.s}
    add     x3, x7, x14, lsl #2
    mov     w15, #0
    ld1w    {z0.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z1.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z2.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z3.s}, p0/z, [x3]
    add     x3, x3, x10
    mova    za1h.s[w15, 0:3], {z0.s-z3.s}
    mov     w15, #4
    ld1w    {z0.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z1.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z2.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z3.s}, p0/z, [x3]
    add     x3, x3, x10
    mova    za1h.s[w15, 0:3], {z0.s-z3.s}
    mov     w15, #8
    ld1w    {z0.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z1.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z2.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z3.s}, p0/z, [x3]
    add     x3, x3, x10
    mova    za1h.s[w15, 0:3], {z0.s-z3.s}
    mov     w15, #12
    ld1w    {z0.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z1.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z2.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z3.s}, p0/z, [x3]
    mova    za1h.s[w15, 0:3], {z0.s-z3.s}
    // FMOPA: za2 += Q_col × K_col for 16 k-steps
    mov     w15, #0
    mova    {z0.s-z3.s}, za3v.s[w15, 0:3]
    mova    {z4.s-z7.s}, za1v.s[w15, 0:3]
    fmopa   za2.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za2.s, p0/m, p0/m, z1.s, z5.s
    fmopa   za2.s, p0/m, p0/m, z2.s, z6.s
    fmopa   za2.s, p0/m, p0/m, z3.s, z7.s
    mov     w15, #4
    mova    {z0.s-z3.s}, za3v.s[w15, 0:3]
    mova    {z4.s-z7.s}, za1v.s[w15, 0:3]
    fmopa   za2.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za2.s, p0/m, p0/m, z1.s, z5.s
    fmopa   za2.s, p0/m, p0/m, z2.s, z6.s
    fmopa   za2.s, p0/m, p0/m, z3.s, z7.s
    mov     w15, #8
    mova    {z0.s-z3.s}, za3v.s[w15, 0:3]
    mova    {z4.s-z7.s}, za1v.s[w15, 0:3]
    fmopa   za2.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za2.s, p0/m, p0/m, z1.s, z5.s
    fmopa   za2.s, p0/m, p0/m, z2.s, z6.s
    fmopa   za2.s, p0/m, p0/m, z3.s, z7.s
    mov     w15, #12
    mova    {z0.s-z3.s}, za3v.s[w15, 0:3]
    mova    {z4.s-z7.s}, za1v.s[w15, 0:3]
    fmopa   za2.s, p0/m, p0/m, z0.s, z4.s
    fmopa   za2.s, p0/m, p0/m, z1.s, z5.s
    fmopa   za2.s, p0/m, p0/m, z2.s, z6.s
    fmopa   za2.s, p0/m, p0/m, z3.s, z7.s
    add     w14, w14, #16          // k_chunk += 16
    sub     w12, w12, #1
    b       .L_fa_score_k
.L_fa_score_done:
    // za2 now has the 16×16 raw score tile
    // ── Phase 2: Scale by rsqrt(d), apply causal mask, online softmax ──
    LOAD_EXP_CONSTANTS
    ldr     s16, [sp, #192]        // rsqrt_d
    mov     z16.s, s16
    movz    w4, #0xFF80, lsl #16
    fmov    s17, w4
    mov     z17.s, s17              // -inf
    ld1w    {z18.s}, p0/z, [sp]    // m_old[16]
    add     x16, sp, #64
    ld1w    {z19.s}, p0/z, [x16]   // l_old[16]
    ldr     w0, [sp, #176]         // qi
    ldr     w1, [sp, #180]         // kj
    ldr     w24, [sp, #168]        // flags
    // Process 4 rows at a time (extract 4, scale, mask, find max, write back)
    mov     w12, #0
.L_fa_softmax_grp:
    cmp     w12, #16
    b.ge    .L_fa_softmax_done
    mova    {z0.s-z3.s}, za2h.s[w12, 0:3]
    // Scale by rsqrt_d
    fmul    z0.s, z0.s, z16.s
    fmul    z1.s, z1.s, z16.s
    fmul    z2.s, z2.s, z16.s
    fmul    z3.s, z3.s, z16.s
    // Causal mask: for row qi+w12+r (r=0..3), mask columns where kj+col > qi+w12+r
    tst     w24, #1
    b.eq    .L_fa_no_mask
    add     w3, w0, w12            // qi + row_base
    // Row 0: mask where kj + col > qi + row_base
    index   z8.s, w1, #1          // z8 = [kj, kj+1, ..., kj+15]
    mov     z9.s, w3               // qi + row_base
    cmpgt   p1.s, p0/z, z8.s, z9.s // p1 set where kj+col > qi+row
    mov     z0.s, p1/m, z17.s     // masked positions → -inf
    add     w3, w3, #1
    mov     z9.s, w3
    cmpgt   p1.s, p0/z, z8.s, z9.s
    mov     z1.s, p1/m, z17.s
    add     w3, w3, #1
    mov     z9.s, w3
    cmpgt   p1.s, p0/z, z8.s, z9.s
    mov     z2.s, p1/m, z17.s
    add     w3, w3, #1
    mov     z9.s, w3
    cmpgt   p1.s, p0/z, z8.s, z9.s
    mov     z3.s, p1/m, z17.s
.L_fa_no_mask:
    // Write scaled+masked scores back to za2
    mova    za2h.s[w12, 0:3], {z0.s-z3.s}
    // Find per-row max (horizontal max of each row)
    fmaxv   s8, p0, z0.s          // max of row w12+0
    fmaxv   s9, p0, z1.s          // max of row w12+1
    fmaxv   s10, p0, z2.s         // max of row w12+2
    fmaxv   s11, p0, z3.s         // max of row w12+3
    // Store row maxima into a temporary z-vector
    // We'll build the full m_new[16] vector across 4 groups
    // Store to stack and load at the end
    add     x3, sp, #196          // temp area at end of frame
    str     s8, [x3, w12, uxtw #2]
    add     w4, w12, #1
    str     s9, [x3, w4, uxtw #2]
    add     w4, w12, #2
    str     s10, [x3, w4, uxtw #2]
    add     w4, w12, #3
    str     s11, [x3, w4, uxtw #2]
    add     w12, w12, #4
    b       .L_fa_softmax_grp
.L_fa_softmax_done:
    // Load m_new[16] from temp area
    add     x3, sp, #196
    ld1w    {z20.s}, p0/z, [x3]   // z20 = per-row max of current score tile
    // m_combined = max(m_old, m_new)
    mov     z21.d, z18.d
    fmax    z21.s, p0/m, z21.s, z20.s // z21 = max(m_old, m_new)
    // Correction factor for old accumulator: exp(m_old - m_combined)
    movprfx z22, z18
    fsub    z22.s, p0/m, z22.s, z21.s // m_old - m_combined (≤ 0)
    // exp(z22) → z22 via per-lane exp
    // Process 16 lanes: broadcast is already in z22, use EXP_POLY
    mov     z0.d, z22.d
    EXP_POLY_Z0_TO_Z4
    mov     z22.d, z4.d            // z22 = exp(m_old - m_combined) = alpha
    // Correction for new scores: exp(m_new - m_combined)
    movprfx z23, z20
    fsub    z23.s, p0/m, z23.s, z21.s // m_new - m_combined (≤ 0)
    mov     z0.d, z23.d
    EXP_POLY_Z0_TO_Z4
    mov     z23.d, z4.d            // z23 = exp(m_new - m_combined) = beta
    // l_new = alpha * l_old + beta * rowsum(P)
    // First compute P = exp(score - m_combined_per_row) in za2
    // Process each row group: subtract m_combined (per-row), exp, sum
    fmov    z24.s, #0.0            // will accumulate per-row sums
    LOAD_EXP_CONSTANTS             // reload exp constants (clobbered by EXP_POLY above)
    mov     w12, #0
.L_fa_exp_grp:
    cmp     w12, #16
    b.ge    .L_fa_exp_done
    mova    {z0.s-z3.s}, za2h.s[w12, 0:3]
    // Need to subtract m_combined for each row. m_combined is in z21.
    // Extract the scalar for each row from z21 and broadcast
    // Row w12+0: z21[w12+0], Row w12+1: z21[w12+1], etc.
    // Use lastb/clastb or index extraction
    // Simpler: store z21 to stack, load scalars
    add     x3, sp, #196
    st1w    {z21.s}, p0, [x3]
    ldr     s8, [x3, w12, uxtw #2]
    mov     z8.s, s8               // broadcast m_combined for row w12+0
    fsub    z0.s, z0.s, z8.s
    add     w4, w12, #1
    ldr     s8, [x3, w4, uxtw #2]
    mov     z8.s, s8
    fsub    z1.s, z1.s, z8.s
    add     w4, w12, #2
    ldr     s8, [x3, w4, uxtw #2]
    mov     z8.s, s8
    fsub    z2.s, z2.s, z8.s
    add     w4, w12, #3
    ldr     s8, [x3, w4, uxtw #2]
    mov     z8.s, s8
    fsub    z3.s, z3.s, z8.s
    // Exp each row (4 rows, need to call EXP_POLY 4 times)
    // Save z1-z3, process z0
    add     x16, sp, #196
    st1w    {z1.s}, p0, [x16]    // temp reuse (we already stored z21 but don't need it anymore after loading)
    // Actually this temp area is too small for multiple vectors. Let me use a different approach.
    // Process one row at a time through exp, accumulate sum, write back to za2
    // Row 0:
    EXP_POLY_Z0_TO_Z4             // z4 = exp(score[row0] - m)
    faddv   s8, p0, z4.s          // rowsum
    add     x3, sp, #196
    str     s8, [x3, w12, uxtw #2] // store rowsum for row w12+0
    // Write P row back to za2 — but we can only write 4 rows at once with mova
    // Store z4 temporarily and batch-write later
    mov     z8.d, z4.d             // save row 0 P
    // Row 1:
    mov     z0.d, z1.d
    EXP_POLY_Z0_TO_Z4
    faddv   s9, p0, z4.s
    add     w4, w12, #1
    str     s9, [x3, w4, uxtw #2]
    mov     z9.d, z4.d             // save row 1 P
    // Row 2:
    mov     z0.d, z2.d
    EXP_POLY_Z0_TO_Z4
    faddv   s10, p0, z4.s
    add     w4, w12, #2
    str     s10, [x3, w4, uxtw #2]
    mov     z10.d, z4.d            // save row 2 P
    // Row 3:
    mov     z0.d, z3.d
    EXP_POLY_Z0_TO_Z4
    faddv   s11, p0, z4.s
    add     w4, w12, #3
    str     s11, [x3, w4, uxtw #2]
    // Write 4 P rows back to za2
    mov     z0.d, z8.d
    mov     z1.d, z9.d
    mov     z2.d, z10.d
    mov     z3.d, z4.d             // row 3 is still in z4
    mova    za2h.s[w12, 0:3], {z0.s-z3.s}
    LOAD_EXP_CONSTANTS             // reload (clobbered by EXP_POLY)
    add     w12, w12, #4
    b       .L_fa_exp_grp
.L_fa_exp_done:
    // Load per-row sums from temp area
    add     x3, sp, #196
    ld1w    {z24.s}, p0/z, [x3]   // z24 = rowsum(P) for each row
    // l_new = alpha * l_old + beta * rowsum(P)
    fmul    z19.s, z19.s, z22.s    // alpha * l_old
    fmul    z24.s, z24.s, z23.s    // beta * rowsum(P)
    fadd    z19.s, z19.s, z24.s    // l_new
    // Save updated m and l
    st1w    {z21.s}, p0, [sp]      // m = m_combined
    add     x16, sp, #64
    st1w    {z19.s}, p0, [x16]   // l = l_new
    // ── Phase 3: Rescale existing output and accumulate P @ V ──
    // For each d_chunk of 16 V-columns:
    //   Load O[qi:qi+16, dc:dc+16] (16×16 from output buffer)
    //   Scale each row by alpha (correction for old max)
    //   Compute P @ V[kj:kj+16, dc:dc+16] → add to O
    //   Store updated O
    ldr     x11, [sp, #152]        // out
    ldr     x8, [sp, #144]         // V
    ldr     x10, [sp, #184]        // d_stride
    ldr     w0, [sp, #176]         // qi
    ldr     w1, [sp, #180]         // kj
    ldr     w23, [sp, #164]        // d
    mul     w3, w0, w23
    add     x11, x11, x3, lsl #2  // out + qi * d * 4
    mul     w3, w1, w23
    add     x8, x8, x3, lsl #2    // V + kj * d * 4
    ldr     w12, [sp, #172]        // d_blocks
    mov     w14, #0                // dc = 0 (d-column offset)
.L_fa_v_chunk:
    cbz     w12, .L_fa_kj_advance
    // Load O[qi:qi+16, dc:dc+16] into za0, scale by alpha
    zero    {za0.s}
    add     x3, x11, x14, lsl #2  // &out[qi][dc]
    mov     w15, #0
    ld1w    {z0.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z1.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z2.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z3.s}, p0/z, [x3]
    add     x3, x3, x10
    // Scale rows by alpha (z22 has per-row alpha, need per-row broadcast)
    // z22[i] = alpha for row i. For row 0: multiply z0 by z22[0] broadcast
    // Use fmul with lane-indexed scalar: not directly available in SVE
    // Simpler: z22 is a vector, but we need to multiply each row vector
    // by a DIFFERENT scalar (the alpha for that row).
    // Use the stored alpha vector and extract scalars
    add     x5, sp, #196
    st1w    {z22.s}, p0, [x5]     // store alpha vector
    ldr     s8, [x5]              // alpha[0]
    mov     z8.s, s8
    fmul    z0.s, z0.s, z8.s
    ldr     s8, [x5, #4]          // alpha[1]
    mov     z8.s, s8
    fmul    z1.s, z1.s, z8.s
    ldr     s8, [x5, #8]
    mov     z8.s, s8
    fmul    z2.s, z2.s, z8.s
    ldr     s8, [x5, #12]
    mov     z8.s, s8
    fmul    z3.s, z3.s, z8.s
    mova    za0h.s[w15, 0:3], {z0.s-z3.s}
    // Rows 4-7
    mov     w15, #4
    ld1w    {z0.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z1.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z2.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z3.s}, p0/z, [x3]
    add     x3, x3, x10
    ldr     s8, [x5, #16]
    mov     z8.s, s8
    fmul    z0.s, z0.s, z8.s
    ldr     s8, [x5, #20]
    mov     z8.s, s8
    fmul    z1.s, z1.s, z8.s
    ldr     s8, [x5, #24]
    mov     z8.s, s8
    fmul    z2.s, z2.s, z8.s
    ldr     s8, [x5, #28]
    mov     z8.s, s8
    fmul    z3.s, z3.s, z8.s
    mova    za0h.s[w15, 0:3], {z0.s-z3.s}
    // Rows 8-11
    mov     w15, #8
    ld1w    {z0.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z1.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z2.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z3.s}, p0/z, [x3]
    add     x3, x3, x10
    ldr     s8, [x5, #32]
    mov     z8.s, s8
    fmul    z0.s, z0.s, z8.s
    ldr     s8, [x5, #36]
    mov     z8.s, s8
    fmul    z1.s, z1.s, z8.s
    ldr     s8, [x5, #40]
    mov     z8.s, s8
    fmul    z2.s, z2.s, z8.s
    ldr     s8, [x5, #44]
    mov     z8.s, s8
    fmul    z3.s, z3.s, z8.s
    mova    za0h.s[w15, 0:3], {z0.s-z3.s}
    // Rows 12-15
    mov     w15, #12
    ld1w    {z0.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z1.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z2.s}, p0/z, [x3]
    add     x3, x3, x10
    ld1w    {z3.s}, p0/z, [x3]
    ldr     s8, [x5, #48]
    mov     z8.s, s8
    fmul    z0.s, z0.s, z8.s
    ldr     s8, [x5, #52]
    mov     z8.s, s8
    fmul    z1.s, z1.s, z8.s
    ldr     s8, [x5, #56]
    mov     z8.s, s8
    fmul    z2.s, z2.s, z8.s
    ldr     s8, [x5, #60]
    mov     z8.s, s8
    fmul    z3.s, z3.s, z8.s
    mova    za0h.s[w15, 0:3], {z0.s-z3.s}
    // za0 now has alpha-scaled old output for this d-chunk
    // Accumulate P @ V_chunk: for each k in 0..15, za0 += P_col[k] × V_row[k]
    // P is in za2, V rows loaded from memory
    add     x3, x8, x14, lsl #2   // V + kj*d*4 + dc*4
    mov     w15, #0
    mova    {z0.s-z3.s}, za2v.s[w15, 0:3]
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
    mov     w15, #4
    mova    {z0.s-z3.s}, za2v.s[w15, 0:3]
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
    mov     w15, #8
    mova    {z0.s-z3.s}, za2v.s[w15, 0:3]
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
    mov     w15, #12
    mova    {z0.s-z3.s}, za2v.s[w15, 0:3]
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
    ld1w    {z4.s}, p0/z, [x3]
    add     x3, x3, x10
    fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
    ld1w    {z4.s}, p0/z, [x3]
    fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
    // Store za0 to output[qi:qi+16, dc:dc+16]
    add     x3, x11, x14, lsl #2
    mov     w15, #0
    mova    {z0.s-z3.s}, za0h.s[w15, 0:3]
    st1w    {z0.s}, p0, [x3]
    add     x3, x3, x10
    st1w    {z1.s}, p0, [x3]
    add     x3, x3, x10
    st1w    {z2.s}, p0, [x3]
    add     x3, x3, x10
    st1w    {z3.s}, p0, [x3]
    add     x3, x3, x10
    mov     w15, #4
    mova    {z0.s-z3.s}, za0h.s[w15, 0:3]
    st1w    {z0.s}, p0, [x3]
    add     x3, x3, x10
    st1w    {z1.s}, p0, [x3]
    add     x3, x3, x10
    st1w    {z2.s}, p0, [x3]
    add     x3, x3, x10
    st1w    {z3.s}, p0, [x3]
    add     x3, x3, x10
    mov     w15, #8
    mova    {z0.s-z3.s}, za0h.s[w15, 0:3]
    st1w    {z0.s}, p0, [x3]
    add     x3, x3, x10
    st1w    {z1.s}, p0, [x3]
    add     x3, x3, x10
    st1w    {z2.s}, p0, [x3]
    add     x3, x3, x10
    st1w    {z3.s}, p0, [x3]
    add     x3, x3, x10
    mov     w15, #12
    mova    {z0.s-z3.s}, za0h.s[w15, 0:3]
    st1w    {z0.s}, p0, [x3]
    add     x3, x3, x10
    st1w    {z1.s}, p0, [x3]
    add     x3, x3, x10
    st1w    {z2.s}, p0, [x3]
    add     x3, x3, x10
    st1w    {z3.s}, p0, [x3]
    add     w14, w14, #16
    sub     w12, w12, #1
    ldr     w12, [sp, #172]        // reload d_blocks (w12 was reused)
    sub     w12, w12, #1           // hmm, need proper d_chunk counter
    // Actually: w14 tracks d-column offset, compare against d
    ldr     w23, [sp, #164]
    cmp     w14, w23
    b.lt    .L_fa_v_chunk
.L_fa_kj_advance:
    ldr     w1, [sp, #180]
    add     w1, w1, #16
    b       .L_fa_kj
.L_fa_kj_done:
    // ── Phase 4: Final normalization: O[qi] /= l ──
    ldr     x11, [sp, #152]
    ldr     x10, [sp, #184]
    ldr     w0, [sp, #176]
    ldr     w23, [sp, #164]
    mul     w3, w0, w23
    add     x11, x11, x3, lsl #2  // &out[qi][0]
    add     x16, sp, #64
    ld1w    {z19.s}, p0/z, [x16]   // l[16]
    // Compute 1/l per row
    frecpe  z20.s, z19.s
    frecps  z21.s, z19.s, z20.s
    fmul    z20.s, p0/m, z20.s, z21.s
    frecps  z21.s, z19.s, z20.s
    fmul    z20.s, p0/m, z20.s, z21.s // z20 = 1/l[16]
    // For each row i, multiply entire output row by 1/l[i]
    // Store 1/l to temp, extract per row
    add     x3, sp, #196
    st1w    {z20.s}, p0, [x3]
    mov     w12, #0
.L_fa_norm:
    cmp     w12, #16
    b.ge    .L_fa_qi_advance
    ldr     s8, [x3, w12, uxtw #2]
    mov     z8.s, s8               // broadcast 1/l[row]
    mov     x5, x11               // row start
    ldr     w4, [sp, #172]         // d_blocks
.L_fa_norm_d:
    cbz     w4, .L_fa_norm_next
    ld1w    {z0.s}, p0/z, [x5]
    fmul    z0.s, z0.s, z8.s
    st1w    {z0.s}, p0, [x5]
    add     x5, x5, x9, lsl #2
    sub     w4, w4, #1
    b       .L_fa_norm_d
.L_fa_norm_next:
    add     x11, x11, x10         // next row
    add     w12, w12, #1
    b       .L_fa_norm
.L_fa_qi_advance:
    ldr     w0, [sp, #176]
    add     w0, w0, #16
    b       .L_fa_qi
.L_fa_done:
    add     sp, sp, #320
    b       .L_dispatch
// ================================================================
// GET_ROWS_FP32 (0x7B) — Embedding lookup from fp32 table
//
// For each index in indices[0..n_rows-1], copies the corresponding row
// from the embedding table to the output. No dequantization needed.
//
// Encoding: [0x7B][n_rows:u32][dim:u32][table:u64][indices:u64][out:u64]
// dim must be a multiple of 16. indices are uint32_t.
// ================================================================
.L_op_get_rows_fp32:
    ldr     w22, [x19]             // n_rows
    ldr     w23, [x19, #4]        // dim
    add     x19, x19, #8
    ldr     x8, [x19], #8         // table (fp32, vocab_size × dim)
    ldr     x11, [x19], #8        // indices (uint32_t array)
    ldr     x13, [x19], #8        // out (fp32, n_rows × dim)
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    lsl     x10, x23, #2           // dim_bytes = dim * 4
.L_gr32_row:
    ldr     w3, [x11], #4         // idx = indices[i]
    mul     x3, x3, x10            // idx * dim_bytes
    add     x14, x8, x3            // &table[idx][0]
    mov     w12, w23               // dim counter
.L_gr32_copy:
    ld1w    {z0.s}, p0/z, [x14]
    st1w    {z0.s}, p0, [x13]
    add     x14, x14, x9, lsl #2
    add     x13, x13, x9, lsl #2
    subs    w12, w12, w9
    b.ne    .L_gr32_copy
    subs    w22, w22, #1
    b.ne    .L_gr32_row
    b       .L_dispatch
// ================================================================
// GET_ROWS_Q8_0 (0x7C) — Embedding lookup + dequant from Q8_0 table
//
// Each table row is stored as Q8_0 blocks: {fp16 scale, 32×int8} = 34 bytes per 32 elements.
// Dequantizes to fp32 on output.
//
// Encoding: [0x7C][n_rows:u32][dim:u32][table:u64][indices:u64][out:u64]
// dim must be a multiple of 32.
// ================================================================
.L_op_get_rows_q8_0:
    ldr     w22, [x19]
    ldr     w23, [x19, #4]        // dim
    add     x19, x19, #8
    ldr     x8, [x19], #8         // table (Q8_0 blocks)
    ldr     x11, [x19], #8        // indices
    ldr     x13, [x19], #8        // out (fp32)
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    lsr     w24, w23, #5           // blocks_per_row = dim / 32
    // row_bytes = blocks_per_row * 34
    mov     w10, #34
    mul     w10, w24, w10
    sxtw    x10, w10               // row_bytes
.L_grq8_row:
    ldr     w3, [x11], #4         // idx
    mul     x3, x3, x10            // idx * row_bytes
    add     x14, x8, x3            // &table[idx] (Q8_0 row start)
    mov     w12, w24               // block counter
.L_grq8_block:
    cbz     w12, .L_grq8_next
    ldr     h0, [x14]             // fp16 scale
    fcvt    s0, h0
    mov     z16.s, s0              // broadcast scale
    add     x16, x14, #2           // &qs[0]
    // First 16 quants
    ld1sb   {z0.s}, p0/z, [x16]
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z16.s
    st1w    {z0.s}, p0, [x13]
    add     x13, x13, x9, lsl #2
    // Second 16 quants
    add     x16, x16, #16
    ld1sb   {z0.s}, p0/z, [x16]
    scvtf   z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z16.s
    st1w    {z0.s}, p0, [x13]
    add     x13, x13, x9, lsl #2
    add     x14, x14, #34          // next block
    sub     w12, w12, #1
    b       .L_grq8_block
.L_grq8_next:
    subs    w22, w22, #1
    b.ne    .L_grq8_row
    b       .L_dispatch
// ================================================================
// GET_ROWS_Q4_0 (0x7D) — Embedding lookup + dequant from Q4_0 table
//
// Each table row is stored as Q4_0 blocks: {fp16 scale, 16 packed bytes} = 18 bytes per 32 elements.
// Low nibble = element i (0..15), high nibble = element i+16 (16..31).
// Dequant: (nibble - 8) * scale
//
// Encoding: [0x7D][n_rows:u32][dim:u32][table:u64][indices:u64][out:u64]
// dim must be a multiple of 32.
// ================================================================
.L_op_get_rows_q4_0:
    ldr     w22, [x19]
    ldr     w23, [x19, #4]
    add     x19, x19, #8
    ldr     x8, [x19], #8         // table (Q4_0 blocks)
    ldr     x11, [x19], #8        // indices
    ldr     x13, [x19], #8        // out (fp32)
    cbz     w22, .L_dispatch
    ptrue   p0.s
    cntw    x9
    lsr     w24, w23, #5           // blocks_per_row = dim / 32
    mov     w10, #18
    mul     w10, w24, w10
    sxtw    x10, w10               // row_bytes
    mov     w4, #8
    dup     z17.s, w4              // bias = 8
.L_grq4_row:
    ldr     w3, [x11], #4
    mul     x3, x3, x10
    add     x14, x8, x3
    mov     w12, w24
.L_grq4_block:
    cbz     w12, .L_grq4_next
    ldr     h0, [x14]
    fcvt    s0, h0
    mov     z16.s, s0              // broadcast scale
    add     x16, x14, #2           // &qs[0]
    ld1b    {z0.s}, p0/z, [x16]   // 16 bytes → 16 lanes
    // Low nibbles (elements 0..15)
    mov     z2.d, z0.d
    and     z2.s, z2.s, #0x0F
    sub     z2.s, z2.s, z17.s      // - 8
    scvtf   z2.s, p0/m, z2.s
    fmul    z2.s, z2.s, z16.s
    st1w    {z2.s}, p0, [x13]
    add     x13, x13, x9, lsl #2
    // High nibbles (elements 16..31)
    lsr     z3.s, z0.s, #4
    and     z3.s, z3.s, #0x0F
    sub     z3.s, z3.s, z17.s
    scvtf   z3.s, p0/m, z3.s
    fmul    z3.s, z3.s, z16.s
    st1w    {z3.s}, p0, [x13]
    add     x13, x13, x9, lsl #2
    add     x14, x14, #18
    sub     w12, w12, #1
    b       .L_grq4_block
.L_grq4_next:
    subs    w22, w22, #1
    b.ne    .L_grq4_row
    b       .L_dispatch
