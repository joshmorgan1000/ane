// quantize_act_fp32_int8.s — Quantize FP32 activation vector to INT8 with per-block scales
//
// void quantize_act_fp32_int8(
//     const float* input,          // x0: [K] FP32 input
//     int8_t*      output,         // x1: [K] int8 output
//     float*       block_scales,   // x2: [K/32] fp32 per-block scales
//     long         K               // x3: total elements (multiple of 32)
// )
//
// For each block of 32 elements:
//   1. absmax = max(|input[b*32 .. b*32+31]|)
//   2. block_scales[b] = absmax / 127.0f
//   3. inv_scale = (absmax != 0) ? 127.0f / absmax : 0.0f
//   4. output[b*32+k] = round(input[b*32+k] * inv_scale) clamped to [-127, 127]
//
// Uses streaming SVE: fabs, fmaxv, fmul, fcvtzs, sqxtnb (narrow)

.section __TEXT,__text,regular,pure_instructions
.global _quantize_act_fp32_int8
.p2align 4

_quantize_act_fp32_int8:
    stp     x29, x30, [sp, #-128]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10,  d11,  [sp, #64]
    stp     d12,  d13,  [sp, #80]
    stp     d14,  d15,  [sp, #96]
    // Stack scratch space: offset #112-127 available for temporaries

    cbz     x3, .Lqa_done

    mov     x19, x0                 // input
    mov     x20, x1                 // output
    mov     x21, x2                 // block_scales
    mov     x22, x3                 // K

    // Number of blocks
    lsr     x8, x22, #5            // n_blocks = K / 32

    smstart sm
    ptrue   p0.s                    // 16 .s lanes

    // Load constants
    adr     x9, .Lqa_const
    ld1rw   {z16.s}, p0/z, [x9]       // z16 = 127.0f
    ld1rw   {z17.s}, p0/z, [x9, #4]   // z17 = 1/127.0f

    // Process blocks
    mov     x10, #0                 // block counter
    mov     x11, #0                 // element offset

.Lqa_block_loop:
    cmp     x10, x8
    b.ge    .Lqa_end

    // Load 32 FP32 values (2 z-register loads of 16 each)
    add     x12, x19, x11, lsl #2  // &input[offset]
    ld1w    {z0.s}, p0/z, [x12]          // first 16 elements
    ld1w    {z1.s}, p0/z, [x12, #1, mul vl]  // next 16 elements

    // Compute |x| for both halves
    fabs    z2.s, p0/m, z0.s
    fabs    z3.s, p0/m, z1.s

    // Max across both halves
    fmax    z2.s, p0/m, z2.s, z3.s // element-wise max of abs values
    fmaxv   s4, p0, z2.s           // horizontal max → s4 = absmax

    // Save absmax to stack for use after check
    str     s4, [sp, #112]         // use dedicated scratch slot (not d8-d15 slots)

    // Compute block_scale = absmax / 127.0f
    fmul    s5, s4, s17            // s5 = absmax / 127.0 = absmax * (1/127)
    // Store block_scale
    str     s5, [x21, x10, lsl #2]

    // Compute inv_scale = 127.0f / absmax (or 0 if absmax == 0)
    fcmp    s4, #0.0
    b.eq    .Lqa_zero_block

    fdiv    s6, s16, s4            // s6 = 127.0f / absmax

    // Broadcast inv_scale to all lanes
    mov     z6.s, s6               // z6 = inv_scale broadcast

    // Quantize: round(input * inv_scale), clamp to [-127, 127]
    fmul    z0.s, p0/m, z0.s, z6.s
    fmul    z1.s, p0/m, z1.s, z6.s

    // Round to nearest integer
    frintn  z0.s, p0/m, z0.s
    frintn  z1.s, p0/m, z1.s

    // FP32 → INT32
    fcvtzs  z0.s, p0/m, z0.s
    fcvtzs  z1.s, p0/m, z1.s

    // Clamp to [-127, 127] (already naturally within range given scale,
    // but clamp for safety with rounding)
    mov     w12, #127
    dup     z7.s, w12
    mov     w12, #-127
    dup     z8.s, w12
    smax    z0.s, p0/m, z0.s, z8.s
    smin    z0.s, p0/m, z0.s, z7.s
    smax    z1.s, p0/m, z1.s, z8.s
    smin    z1.s, p0/m, z1.s, z7.s

    // Narrow INT32 → INT8 and store
    // SVE: st1b stores the bottom byte of each .s element
    add     x12, x20, x11          // &output[offset]
    st1b    {z0.s}, p0, [x12]           // store 16 int8 values (bottom bytes of .s)
    st1b    {z1.s}, p0, [x12, #1, mul vl]  // next 16

    b       .Lqa_next_block

.Lqa_zero_block:
    // absmax == 0: output all zeros
    mov     z0.d, #0
    add     x12, x20, x11
    st1b    {z0.s}, p0, [x12]
    st1b    {z0.s}, p0, [x12, #1, mul vl]

.Lqa_next_block:
    add     x10, x10, #1
    add     x11, x11, #32
    b       .Lqa_block_loop

.Lqa_end:
    smstop

.Lqa_done:
    ldp     d8,  d9,  [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     d10,  d11,  [sp, #64]
    ldp     d12,  d13,  [sp, #80]
    ldp     d14,  d15,  [sp, #96]
    ldp     x29, x30, [sp], #128
    ret

.p2align 2
.Lqa_const:
    .long   0x42FE0000              // 127.0f
    .long   0x3C010204              // 1.0f/127.0f = 0.007874016f
