// prelu_backward_fp32.s — PReLU backward pass via SME2 streaming SVE
//
// void prelu_backward_fp32(const float* dy, const float* x, const float* slope,
//                           float* dx, float* dslope, long n)
// AAPCS: x0=dy, x1=x, x2=slope, x3=dx, x4=dslope, x5=n
//
// Math:
//   dx[i]     = dy[i] * (x[i] > 0 ? 1.0 : slope[i])
//   dslope[i] = dy[i] * min(x[i], 0.0)   = (x[i] < 0) ? dy[i]*x[i] : 0.0
//
// Strategy:
//   1. Load dy (z0-z3), x (z4-z7), slope (z8-z11)
//   2. Compute alpha*dy for negative branch (z12-z15 = slope*dy)
//   3. Compute fcmge predicate: x >= 0
//   4. dx: sel(p_pos, dy, slope*dy)
//   5. dslope: fmul(dy, x), then zero out positive lanes via predicated sel
//
// Uses smstart sm (no ZA needed).
// Processes 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _prelu_backward_fp32
.p2align 4

_prelu_backward_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x5, .Lpreluback_done

    smstart sm

    ptrue   p0.s
    mov     z30.d, #0               // zero vector for comparison and zeroing

    mov     x8, #0
    whilelt pn9.s, x8, x5, vlx4

.Lpreluback_loop:
    // Load dy → z0-z3
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    // Load x → z4-z7
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]
    // Load slope → z8-z11
    ld1w    {z8.s-z11.s}, pn9/z, [x2, x8, lsl #2]

    // Compute slope * dy → z12-z15 (used for negative branch of dx)
    mov     z12.d, z8.d
    fmul    z12.s, p0/m, z12.s, z0.s
    mov     z13.d, z9.d
    fmul    z13.s, p0/m, z13.s, z1.s
    mov     z14.d, z10.d
    fmul    z14.s, p0/m, z14.s, z2.s
    mov     z15.d, z11.d
    fmul    z15.s, p0/m, z15.s, z3.s

    // x >= 0 predicates (p1-p4)
    fcmge   p1.s, p0/z, z4.s, z30.s
    fcmge   p2.s, p0/z, z5.s, z30.s
    fcmge   p3.s, p0/z, z6.s, z30.s
    fcmge   p4.s, p0/z, z7.s, z30.s

    // dx: positive → dy, negative → slope*dy
    sel     z16.s, p1, z0.s, z12.s
    sel     z17.s, p2, z1.s, z13.s
    sel     z18.s, p3, z2.s, z14.s
    sel     z19.s, p4, z3.s, z15.s

    // Store dx
    st1w    {z16.s-z19.s}, pn9, [x3, x8, lsl #2]

    // dslope = dy * x, zeroed where x >= 0
    // Compute dy * x for all lanes → z20-z23
    mov     z20.d, z0.d
    fmul    z20.s, p0/m, z20.s, z4.s
    mov     z21.d, z1.d
    fmul    z21.s, p0/m, z21.s, z5.s
    mov     z22.d, z2.d
    fmul    z22.s, p0/m, z22.s, z6.s
    mov     z23.d, z3.d
    fmul    z23.s, p0/m, z23.s, z7.s

    // Where x >= 0, dslope = 0; where x < 0, dslope = dy*x
    // p1-p4 are (x >= 0); use sel: false-lane (x < 0) → dy*x, true-lane (x >= 0) → 0
    sel     z20.s, p1, z30.s, z20.s
    sel     z21.s, p2, z30.s, z21.s
    sel     z22.s, p3, z30.s, z22.s
    sel     z23.s, p4, z30.s, z23.s

    // Store dslope
    st1w    {z20.s-z23.s}, pn9, [x4, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x5, vlx4
    b.first .Lpreluback_loop

    smstop

.Lpreluback_done:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
