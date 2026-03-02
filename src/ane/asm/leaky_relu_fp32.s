// leaky_relu_fp32.s — FP32 Leaky ReLU via SME2 streaming SVE
//
// void leaky_relu_fp32(const float *input, float *output, float alpha, long n)
// AAPCS: x0=input, x1=output, s0=alpha, x2=n
//
// Computes output[i] = x >= 0 ? x : alpha * x
// Uses predication: compare with zero, then blend alpha*x for negative lanes.
//
// Processing: 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _leaky_relu_fp32
.p2align 4

_leaky_relu_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x2, .Llrelu_done

    // Preserve alpha (s0) before smstart zeroes it
    fmov    w8, s0

    smstart sm

    // Restore alpha into z8 as broadcast
    fmov    s0, w8
    ptrue   p0.s
    mov     z8.s, s0              // z8 = alpha broadcast

    mov     z9.s, #0              // zero for comparison

    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Llrelu_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]

    // Compute alpha * x in z4-z7
    fmul    z4.s, z0.s, z8.s
    fmul    z5.s, z1.s, z8.s
    fmul    z6.s, z2.s, z8.s
    fmul    z7.s, z3.s, z8.s

    // For x >= 0, keep x; for x < 0, use alpha*x
    // Use fcmge to get predicate: p1 = (x >= 0)
    fcmge   p1.s, p0/z, z0.s, z9.s
    fcmge   p2.s, p0/z, z1.s, z9.s
    fcmge   p3.s, p0/z, z2.s, z9.s
    fcmge   p4.s, p0/z, z3.s, z9.s

    // Merge: result = x >= 0 ? x : alpha*x
    // Start with alpha*x, then overwrite positive lanes with x
    sel     z4.s, p1, z0.s, z4.s
    sel     z5.s, p2, z1.s, z5.s
    sel     z6.s, p3, z2.s, z6.s
    sel     z7.s, p4, z3.s, z7.s

    st1w    {z4.s-z7.s}, pn9, [x1, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x2, vlx4
    b.first .Llrelu_loop

    smstop

.Llrelu_done:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
