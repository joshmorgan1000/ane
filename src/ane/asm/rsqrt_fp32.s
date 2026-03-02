// rsqrt_fp32.s — Element-wise FP32 reciprocal square root via SME2 streaming SVE
//
// void rsqrt_fp32(const float *input, float *output, long n)
//
// Computes output[i] = 1/sqrt(input[i]) for i in [0, n).
// Uses frsqrte + 2 Newton-Raphson steps for full float precision.

.section __TEXT,__text,regular,pure_instructions
.global _rsqrt_fp32
.p2align 4

_rsqrt_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x2, .Ldone

    smstart sm

    ptrue   p0.s
    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]

    // Estimate: e ≈ 1/sqrt(x)
    frsqrte z4.s, z0.s
    frsqrte z5.s, z1.s
    frsqrte z6.s, z2.s
    frsqrte z7.s, z3.s

    // Newton-Raphson step 1: e *= (3 - x*e²) / 2  via frsqrts
    fmul    z8.s,  z4.s, z4.s              // e²
    fmul    z9.s,  z5.s, z5.s
    fmul    z10.s, z6.s, z6.s
    fmul    z11.s, z7.s, z7.s
    frsqrts z8.s,  z0.s, z8.s              // (3 - x*e²) / 2
    frsqrts z9.s,  z1.s, z9.s
    frsqrts z10.s, z2.s, z10.s
    frsqrts z11.s, z3.s, z11.s
    fmul    z4.s, z4.s, z8.s               // e *= step
    fmul    z5.s, z5.s, z9.s
    fmul    z6.s, z6.s, z10.s
    fmul    z7.s, z7.s, z11.s

    // Newton-Raphson step 2
    fmul    z8.s,  z4.s, z4.s
    fmul    z9.s,  z5.s, z5.s
    fmul    z10.s, z6.s, z6.s
    fmul    z11.s, z7.s, z7.s
    frsqrts z8.s,  z0.s, z8.s
    frsqrts z9.s,  z1.s, z9.s
    frsqrts z10.s, z2.s, z10.s
    frsqrts z11.s, z3.s, z11.s
    fmul    z4.s, z4.s, z8.s
    fmul    z5.s, z5.s, z9.s
    fmul    z6.s, z6.s, z10.s
    fmul    z7.s, z7.s, z11.s

    st1w    {z4.s-z7.s}, pn9, [x1, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x2, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
