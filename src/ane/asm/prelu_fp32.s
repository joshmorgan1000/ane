// prelu_fp32.s — FP32 Parametric ReLU via SME2 streaming SVE
//
// void prelu_fp32(const float *input, const float *alpha, float *output, long n)
// AAPCS: x0=input, x1=alpha, x2=output, x3=n
//
// Computes output[i] = input[i] >= 0 ? input[i] : alpha[i] * input[i]
// Unlike leaky_relu, alpha is per-element (loaded per iteration, not scalar).
// Uses predication: compare with zero, then blend alpha*input for negative lanes.
//
// Processing: 4 vectors (64 floats) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _prelu_fp32
.p2align 4

_prelu_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x3, .Ldone

    smstart sm

    ptrue   p0.s
    mov     z16.d, #0             // zero for comparison

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lloop:
    // Load input
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]

    // Load alpha (per-element)
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]

    // Compute alpha * input (using fmla with zero accumulator equivalent: move alpha to result, then fmul by input)
    mov     z8.d, z4.d
    fmul    z8.s, p0/m, z8.s, z0.s
    mov     z9.d, z5.d
    fmul    z9.s, p0/m, z9.s, z1.s
    mov     z10.d, z6.d
    fmul    z10.s, p0/m, z10.s, z2.s
    mov     z11.d, z7.d
    fmul    z11.s, p0/m, z11.s, z3.s

    // input >= 0 predicates
    fcmge   p1.s, p0/z, z0.s, z16.s
    fcmge   p2.s, p0/z, z1.s, z16.s
    fcmge   p3.s, p0/z, z2.s, z16.s
    fcmge   p4.s, p0/z, z3.s, z16.s

    // sel: positive → input, negative → alpha*input
    sel     z8.s, p1, z0.s, z8.s
    sel     z9.s, p2, z1.s, z9.s
    sel     z10.s, p3, z2.s, z10.s
    sel     z11.s, p4, z3.s, z11.s

    st1w    {z8.s-z11.s}, pn9, [x2, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
