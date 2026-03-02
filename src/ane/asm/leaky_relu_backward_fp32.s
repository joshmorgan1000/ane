// leaky_relu_backward_fp32.s — Leaky ReLU backward pass via SME2 streaming SVE
//
// void leaky_relu_backward_fp32(const float* dy, const float* x, float* dx, long n, float alpha)
// AAPCS: x0=dy (upstream gradient), x1=x (forward input), x2=dx (output gradient),
//        x3=n, s0=alpha
//
// Computes: dx[i] = dy[i] * (x[i] > 0 ? 1.0 : alpha)
// When x > 0: pass gradient straight through (multiply by 1.0 = keep dy).
// When x <= 0: scale gradient by alpha.
//
// Processing: 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _leaky_relu_backward_fp32
.p2align 4

_leaky_relu_backward_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x3, .Llrb_done

    // Save alpha (s0) before smstart zeroes vector registers
    fmov    w9, s0

    smstart sm

    // Restore alpha and broadcast into z16 (all lanes)
    fmov    s0, w9
    ptrue   p0.s
    mov     z16.s, s0               // z16 = alpha broadcast across all lanes

    // Zero vector for compare baseline
    mov     z9.d, #0

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Llrb_loop:
    // Load dy into z0-z3 and x into z4-z7
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]   // dy
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]   // x

    // Compare x > 0 to build masks
    fcmgt   p1.s, p0/z, z4.s, z9.s
    fcmgt   p2.s, p0/z, z5.s, z9.s
    fcmgt   p3.s, p0/z, z6.s, z9.s
    fcmgt   p4.s, p0/z, z7.s, z9.s

    // For lanes where x <= 0: scale dy by alpha
    // Use working copies z17-z20 = alpha * dy  (the "false" branch result)
    mov     z17.d, z16.d
    mov     z18.d, z16.d
    mov     z19.d, z16.d
    mov     z20.d, z16.d
    fmul    z17.s, p0/m, z17.s, z0.s
    fmul    z18.s, p0/m, z18.s, z1.s
    fmul    z19.s, p0/m, z19.s, z2.s
    fmul    z20.s, p0/m, z20.s, z3.s

    // Select: p_true (x>0) → dy unchanged, p_false (x<=0) → alpha*dy
    sel     z0.s, p1, z0.s, z17.s
    sel     z1.s, p2, z1.s, z18.s
    sel     z2.s, p3, z2.s, z19.s
    sel     z3.s, p4, z3.s, z20.s

    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Llrb_loop

    smstop

.Llrb_done:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #96
    ret
