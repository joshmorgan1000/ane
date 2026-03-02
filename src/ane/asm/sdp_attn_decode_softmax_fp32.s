// sdp_attn_decode_softmax_fp32.s -- Phase 3: Softmax (decode, single-threaded)
//
// void sdp_attn_decode_softmax_fp32(
//     const float* scores_global,  // x0 - input scores [num_heads, cache_len]
//     float* attn,                 // x1 - output [num_heads, cache_len]
//     long num_heads,              // x2
//     long cache_len               // x3
// );
//
// Layout: scores_global[head][position] in row-major (head-contiguous)
// Computation per head:
//   1. Find max over cache_len
//   2. Subtract max, multiply by log2(e)
//   3. Compute exp via polynomial approximation
//   4. Sum all exps
//   5. Divide by sum (normalize to probabilities)

.section __TEXT,__text,regular,pure_instructions
.global _sdp_attn_decode_softmax_fp32
.p2align 4

_sdp_attn_decode_softmax_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     d8,  d9,  [sp, #64]
    stp     d10, d11, [sp, #80]

    // Early exit checks
    cbz     x2, .Ldec_done              // num_heads == 0
    cbz     x3, .Ldec_done              // cache_len == 0

    // Save arguments
    mov     x19, x0                     // scores_global
    mov     x20, x1                     // attn output
    mov     x21, x2                     // num_heads
    mov     x22, x3                     // cache_len

    smstart sm

    ptrue   p0.s

    // Load constants for exp polynomial
    adr     x11, .Ldec_consts
    ld1rw   {z21.s}, p0/z, [x11]        // log2(e)
    ld1rw   {z22.s}, p0/z, [x11, #4]
    ld1rw   {z23.s}, p0/z, [x11, #8]
    ld1rw   {z24.s}, p0/z, [x11, #12]
    ld1rw   {z25.s}, p0/z, [x11, #16]
    ld1rw   {z26.s}, p0/z, [x11, #20]
    ld1rw   {z27.s}, p0/z, [x11, #24]
    ld1rw   {z29.s}, p0/z, [x11, #28]   // -inf for max init
    fmov    z28.s, #1.0                 // 1.0 for exp final term

    // Head loop: iterate over num_heads
    mov     x12, #0

.Ldec_head_loop:
    cmp     x12, x21
    b.ge    .Ldec_exit

    mul     x14, x12, x22               // head offset: head * cache_len
    add     x14, x19, x14, lsl #2       // pointer to scores for this head

    // === FIND MAX ===
    mov     z4.d, z29.d                 // init to -inf
    mov     z5.d, z29.d
    mov     z6.d, z29.d
    mov     z7.d, z29.d
    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

.Ldec_find_max:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    fmax    z4.s, p0/m, z4.s, z0.s
    fmax    z5.s, p0/m, z5.s, z1.s
    fmax    z6.s, p0/m, z6.s, z2.s
    fmax    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Ldec_find_max

    fmax    z4.s, p0/m, z4.s, z5.s
    fmax    z6.s, p0/m, z6.s, z7.s
    fmax    z4.s, p0/m, z4.s, z6.s
    fmaxv   s20, p0, z4.s
    mov     z20.s, s20

    // === EXP AND SUBTRACT MAX ===
    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

.Ldec_exp_subtract:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    fsub    z0.s, p0/m, z0.s, z20.s
    fsub    z1.s, p0/m, z1.s, z20.s
    fsub    z2.s, p0/m, z2.s, z20.s
    fsub    z3.s, p0/m, z3.s, z20.s
    fmul    z0.s, p0/m, z0.s, z21.s
    fmul    z1.s, p0/m, z1.s, z21.s
    fmul    z2.s, p0/m, z2.s, z21.s
    fmul    z3.s, p0/m, z3.s, z21.s

    // Round exponent
    movprfx z8, z0
    frintn  z8.s, p0/m, z0.s
    movprfx z9, z1
    frintn  z9.s, p0/m, z1.s
    movprfx z10, z2
    frintn  z10.s, p0/m, z2.s
    movprfx z11, z3
    frintn  z11.s, p0/m, z3.s

    // Fractional part
    fsub    z0.s, p0/m, z0.s, z8.s
    fsub    z1.s, p0/m, z1.s, z9.s
    fsub    z2.s, p0/m, z2.s, z10.s
    fsub    z3.s, p0/m, z3.s, z11.s

    // Convert exponent to int
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s

    // Polynomial approximation: Horner's method
    mov     z16.d, z22.d
    mov     z17.d, z22.d
    mov     z18.d, z22.d
    mov     z19.d, z22.d
    fmad    z16.s, p0/m, z0.s, z23.s
    fmad    z17.s, p0/m, z1.s, z23.s
    fmad    z18.s, p0/m, z2.s, z23.s
    fmad    z19.s, p0/m, z3.s, z23.s
    fmad    z16.s, p0/m, z0.s, z24.s
    fmad    z17.s, p0/m, z1.s, z24.s
    fmad    z18.s, p0/m, z2.s, z24.s
    fmad    z19.s, p0/m, z3.s, z24.s
    fmad    z16.s, p0/m, z0.s, z25.s
    fmad    z17.s, p0/m, z1.s, z25.s
    fmad    z18.s, p0/m, z2.s, z25.s
    fmad    z19.s, p0/m, z3.s, z25.s
    fmad    z16.s, p0/m, z0.s, z26.s
    fmad    z17.s, p0/m, z1.s, z26.s
    fmad    z18.s, p0/m, z2.s, z26.s
    fmad    z19.s, p0/m, z3.s, z26.s
    fmad    z16.s, p0/m, z0.s, z27.s
    fmad    z17.s, p0/m, z1.s, z27.s
    fmad    z18.s, p0/m, z2.s, z27.s
    fmad    z19.s, p0/m, z3.s, z27.s
    fmad    z16.s, p0/m, z0.s, z28.s
    fmad    z17.s, p0/m, z1.s, z28.s
    fmad    z18.s, p0/m, z2.s, z28.s
    fmad    z19.s, p0/m, z3.s, z28.s

    // Scale by 2^exponent
    fscale  z16.s, p0/m, z16.s, z8.s
    fscale  z17.s, p0/m, z17.s, z9.s
    fscale  z18.s, p0/m, z18.s, z10.s
    fscale  z19.s, p0/m, z19.s, z11.s

    st1w    {z16.s-z19.s}, pn9, [x14, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Ldec_exp_subtract

    // === SUM ALL EXPS ===
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0
    mov     x8, #0
    whilelt pn8.s, x8, x22, vlx4

.Ldec_sum_exp:
    ld1w    {z0.s-z3.s}, pn8/z, [x14, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn8.s, x8, x22, vlx4
    b.first .Ldec_sum_exp

    fadd    z4.s, p0/m, z4.s, z5.s
    fadd    z6.s, p0/m, z6.s, z7.s
    fadd    z4.s, p0/m, z4.s, z6.s
    faddv   s4, p0, z4.s
    mov     z4.s, s4
    fmov    z5.s, #1.0
    fdiv    z5.s, p0/m, z5.s, z4.s
    mov     z30.d, z5.d

    // === NORMALIZE ===
    mul     x15, x12, x22               // head * cache_len
    add     x15, x20, x15, lsl #2       // output ptr for this head
    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

.Ldec_normalize:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z30.s
    fmul    z1.s, p0/m, z1.s, z30.s
    fmul    z2.s, p0/m, z2.s, z30.s
    fmul    z3.s, p0/m, z3.s, z30.s
    st1w    {z0.s-z3.s}, pn9, [x15, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Ldec_normalize

    add     x12, x12, #1
    b       .Ldec_head_loop

.Ldec_exit:
    smstop

.Ldec_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     d8,  d9,  [sp, #64]
    ldp     d10, d11, [sp, #80]
    ldp     x29, x30, [sp], #96
    ret

.p2align 2
.Ldec_consts:
    .float 1.4426950408889634        // log2(e)
    .float 0.00015403530393381609
    .float 0.0013333558146428443
    .float 0.009618129107628477
    .float 0.05550410866482158
    .float 0.24022650695910072
    .float 0.6931471805599453
    .long  0xFF800000               // -inf
