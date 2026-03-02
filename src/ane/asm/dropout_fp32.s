// dropout_fp32.s — Fused mask-apply + scale via SME2 streaming SVE
//
// void dropout_fp32(const float* input, const float* mask, float* output,
//                   float inv_keep_prob, long n)
//
// AAPCS: x0=input, x1=mask, x2=output, s0=inv_keep_prob, x3=n
//
// Computes output[i] = input[i] * mask[i] * inv_keep_prob
// where mask[i] is 0.0f (drop) or 1.0f (keep), caller-generated.
//
// inv_keep_prob is saved before smstart (which zeroes float registers),
// then broadcast into z28 for use in the inner loop.
//
// Processes 64 floats (256 bytes) per iteration on M4 via vlx4.

.section __TEXT,__text,regular,pure_instructions
.global _dropout_fp32
.p2align 4

_dropout_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]
    // Save inv_keep_prob (s0) before smstart zeroes float registers
    str     s0, [sp, #80]

    cbz     x3, .Ldone

    smstart sm

    ptrue   p0.s
    ld1rw   {z28.s}, p0/z, [sp, #80]   // broadcast inv_keep_prob to all lanes

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]   // load input
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]   // load mask
    // Step 1: input * mask (zeroes dropped elements)
    fmul    z0.s, p0/m, z0.s, z4.s
    fmul    z1.s, p0/m, z1.s, z5.s
    fmul    z2.s, p0/m, z2.s, z6.s
    fmul    z3.s, p0/m, z3.s, z7.s
    // Step 2: scale by inv_keep_prob
    fmul    z0.s, p0/m, z0.s, z28.s
    fmul    z1.s, p0/m, z1.s, z28.s
    fmul    z2.s, p0/m, z2.s, z28.s
    fmul    z3.s, p0/m, z3.s, z28.s
    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]      // store output
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #96
    ret
