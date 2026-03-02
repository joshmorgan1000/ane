// quantize_fp32_int8.s — FP32 to INT8 quantization via SME2 streaming SVE
//
// void quantize_fp32_int8(const float *input, int8_t *output, float scale, long n)
//
// Computes output[i] = clamp(round(input[i] * scale), -128, 127)
// AAPCS: x0=input, x1=output, s0=scale, x2=n
//
// Optimized 4-vector loop: processes 64 floats (256 bytes) per iteration
// using SME2 vlx4 operations for maximum throughput.

.section __TEXT,__text,regular,pure_instructions
.global _quantize_fp32_int8
.p2align 4

_quantize_fp32_int8:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]
    str     s0, [sp, #80]

    cbz     x2, .Ldone

    smstart sm

    ptrue   p0.s
    ptrue   p1.b
    ptrue   pn8.s                   // predicate-as-counter for 4-vector ops

    // Broadcast scale to all 4 working vectors
    ld1rw   {z16.s}, p0/z, [sp, #80]

    // Saturation bounds (GPR→dup pattern to avoid SIGILL on M4)
    mov     w12, #127
    dup     z17.s, w12
    mov     w13, #-128
    dup     z18.s, w13

    // Calculate number of full 4-vector iterations (64 floats each)
    cntw    x10                     // SVL in words (16 on M4)
    lsl     x11, x10, #2            // 4 * SVL = 64 floats per iteration
    mov     x8, #0                  // input index
    mov     x9, #0                  // output index

    // Main loop: process 64 floats (4 vectors) per iteration
.Lloop_main:
    sub     x12, x2, x8             // remaining elements
    cmp     x12, x11
    b.lt    .Lloop_tail             // < 64 remaining, go to tail

    // Load 4 vectors (64 floats = 256 bytes)
    ld1w    {z0.s - z3.s}, pn8/z, [x0, x8, lsl #2]

    // Scale all 4 vectors
    fmul    z0.s, p0/m, z0.s, z16.s
    fmul    z1.s, p0/m, z1.s, z16.s
    fmul    z2.s, p0/m, z2.s, z16.s
    fmul    z3.s, p0/m, z3.s, z16.s

    // Round to nearest
    frintn  z0.s, p0/m, z0.s
    frintn  z1.s, p0/m, z1.s
    frintn  z2.s, p0/m, z2.s
    frintn  z3.s, p0/m, z3.s

    // Convert to int32
    fcvtzs  z0.s, p0/m, z0.s
    fcvtzs  z1.s, p0/m, z1.s
    fcvtzs  z2.s, p0/m, z2.s
    fcvtzs  z3.s, p0/m, z3.s

    // Saturate to [-128, 127]
    smax    z0.s, p0/m, z0.s, z18.s
    smax    z1.s, p0/m, z1.s, z18.s
    smax    z2.s, p0/m, z2.s, z18.s
    smax    z3.s, p0/m, z3.s, z18.s
    smin    z0.s, p0/m, z0.s, z17.s
    smin    z1.s, p0/m, z1.s, z17.s
    smin    z2.s, p0/m, z2.s, z17.s
    smin    z3.s, p0/m, z3.s, z17.s

    // Narrow int32 → int16: pairs of vectors
    uzp1    z4.h, z0.h, z1.h        // z0,z1 → z4 (32 int16s)
    uzp1    z5.h, z2.h, z3.h        // z2,z3 → z5 (32 int16s)

    // Narrow int16 → int8: combine into single vector
    uzp1    z6.b, z4.b, z5.b        // z4,z5 → z6 (64 int8s)

    // Store 64 bytes
    st1b    {z6.b}, p1, [x1, x9]

    add     x8, x8, x11             // advance by 64 floats
    add     x9, x9, x11             // advance by 64 bytes
    b       .Lloop_main

    // Tail loop: process remaining elements with single-vector predicated ops
.Lloop_tail:
    whilelt p2.s, x8, x2
    b.none  .Lexit

    ld1w    {z0.s}, p2/z, [x0, x8, lsl #2]

    fmul    z0.s, p0/m, z0.s, z16.s
    frintn  z0.s, p0/m, z0.s
    fcvtzs  z0.s, p0/m, z0.s
    smax    z0.s, p0/m, z0.s, z18.s
    smin    z0.s, p0/m, z0.s, z17.s

    uzp1    z1.h, z0.h, z0.h
    uzp1    z2.b, z1.b, z1.b

    whilelt p3.b, x9, x2
    st1b    {z2.b}, p3, [x1, x9]

    incw    x8
    cntw    x10
    add     x9, x9, x10
    b       .Lloop_tail

.Lexit:
    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #96
    ret
