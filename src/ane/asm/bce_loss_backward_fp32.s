// bce_loss_backward_fp32.s — BCE loss backward pass via SME2 streaming SVE
//
// void bce_loss_backward_fp32(const float* pred, const float* target, float* grad, long n)
// AAPCS: x0=pred, x1=target, x2=grad, x3=n
//
// Math: grad[i] = -(target/max(pred,eps) - (1-target)/max(1-pred,eps)) / n
//              = ((1-target)/max(1-pred,eps) - target/max(pred,eps)) / n
//
// Per block:
//   p_clamp = max(pred, eps)
//   q_clamp = max(1 - pred, eps)
//   rp = 1/p_clamp  (frecpe + 2x NR)
//   rq = 1/q_clamp  (frecpe + 2x NR)
//   term1 = target * rp
//   term2 = (1 - target) * rq
//   grad  = (term2 - term1) * inv_n
//
// Register constants after smstart:
//   z16 = eps = 1e-7
//   z17 = 1.0
//   z18 = inv_n
//
// Processing: 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _bce_loss_backward_fp32
.p2align 4

_bce_loss_backward_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    // sp+88: scratch slot for inv_n scalar

    cbz     x3, .Lbceb_done

    // Compute inv_n = 1/n before smstart.
    scvtf   s0, x3
    adr     x9, .Lbceb_const
    ldr     s1, [x9]                    // 1.0
    fdiv    s0, s1, s0                  // s0 = 1/n
    str     s0, [sp, #88]

    smstart sm

    ptrue   p0.s

    // Load constant vectors.
    adr     x9, .Lbceb_const
    ld1rw   {z16.s}, p0/z, [x9, #4]    // z16 = eps = 1e-7
    ld1rw   {z17.s}, p0/z, [x9]        // z17 = 1.0
    ld1rw   {z18.s}, p0/z, [sp, #88]   // z18 = inv_n

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lbceb_loop:
    // Load pred into z0-z3. Save copies for computing 1-pred.
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]

    // p_clamp = max(pred, eps) → z4-z7.
    movprfx z4, z0
    fmax    z4.s, p0/m, z4.s, z16.s
    movprfx z5, z1
    fmax    z5.s, p0/m, z5.s, z16.s
    movprfx z6, z2
    fmax    z6.s, p0/m, z6.s, z16.s
    movprfx z7, z3
    fmax    z7.s, p0/m, z7.s, z16.s
    // z4-z7 = p_clamp

    // Compute 1 - pred → z20-z23, then clamp to eps → q_clamp.
    // Use movprfx + fsub: z20 = z17 (1.0), then z20 -= z0 (pred).
    movprfx z20, z17
    fsub    z20.s, p0/m, z20.s, z0.s
    movprfx z21, z17
    fsub    z21.s, p0/m, z21.s, z1.s
    movprfx z22, z17
    fsub    z22.s, p0/m, z22.s, z2.s
    movprfx z23, z17
    fsub    z23.s, p0/m, z23.s, z3.s
    fmax    z20.s, p0/m, z20.s, z16.s
    fmax    z21.s, p0/m, z21.s, z16.s
    fmax    z22.s, p0/m, z22.s, z16.s
    fmax    z23.s, p0/m, z23.s, z16.s
    // z20-z23 = q_clamp = max(1-pred, eps)

    // rp = 1/p_clamp (z4-z7) → result in z8-z11.
    frecpe  z8.s,  z4.s
    frecpe  z9.s,  z5.s
    frecpe  z10.s, z6.s
    frecpe  z11.s, z7.s
    // NR step 1
    frecps  z24.s, z4.s, z8.s
    frecps  z25.s, z5.s, z9.s
    frecps  z26.s, z6.s, z10.s
    frecps  z27.s, z7.s, z11.s
    fmul    z8.s,  p0/m, z8.s,  z24.s
    fmul    z9.s,  p0/m, z9.s,  z25.s
    fmul    z10.s, p0/m, z10.s, z26.s
    fmul    z11.s, p0/m, z11.s, z27.s
    // NR step 2
    frecps  z24.s, z4.s, z8.s
    frecps  z25.s, z5.s, z9.s
    frecps  z26.s, z6.s, z10.s
    frecps  z27.s, z7.s, z11.s
    fmul    z8.s,  p0/m, z8.s,  z24.s
    fmul    z9.s,  p0/m, z9.s,  z25.s
    fmul    z10.s, p0/m, z10.s, z26.s
    fmul    z11.s, p0/m, z11.s, z27.s
    // z8-z11 = rp

    // rq = 1/q_clamp (z20-z23) → result in z4-z7.
    frecpe  z4.s,  z20.s
    frecpe  z5.s,  z21.s
    frecpe  z6.s,  z22.s
    frecpe  z7.s,  z23.s
    // NR step 1
    frecps  z24.s, z20.s, z4.s
    frecps  z25.s, z21.s, z5.s
    frecps  z26.s, z22.s, z6.s
    frecps  z27.s, z23.s, z7.s
    fmul    z4.s,  p0/m, z4.s,  z24.s
    fmul    z5.s,  p0/m, z5.s,  z25.s
    fmul    z6.s,  p0/m, z6.s,  z26.s
    fmul    z7.s,  p0/m, z7.s,  z27.s
    // NR step 2
    frecps  z24.s, z20.s, z4.s
    frecps  z25.s, z21.s, z5.s
    frecps  z26.s, z22.s, z6.s
    frecps  z27.s, z23.s, z7.s
    fmul    z4.s,  p0/m, z4.s,  z24.s
    fmul    z5.s,  p0/m, z5.s,  z25.s
    fmul    z6.s,  p0/m, z6.s,  z26.s
    fmul    z7.s,  p0/m, z7.s,  z27.s
    // z4-z7 = rq

    // Load target into z0-z3.
    ld1w    {z0.s-z3.s}, pn9/z, [x1, x8, lsl #2]

    // term1 = target * rp → store back in z8-z11.
    fmul    z8.s,  p0/m, z8.s,  z0.s
    fmul    z9.s,  p0/m, z9.s,  z1.s
    fmul    z10.s, p0/m, z10.s, z2.s
    fmul    z11.s, p0/m, z11.s, z3.s

    // 1 - target → z20-z23.
    movprfx z20, z17
    fsub    z20.s, p0/m, z20.s, z0.s
    movprfx z21, z17
    fsub    z21.s, p0/m, z21.s, z1.s
    movprfx z22, z17
    fsub    z22.s, p0/m, z22.s, z2.s
    movprfx z23, z17
    fsub    z23.s, p0/m, z23.s, z3.s

    // term2 = (1-target) * rq → z4-z7.
    fmul    z4.s,  p0/m, z4.s,  z20.s
    fmul    z5.s,  p0/m, z5.s,  z21.s
    fmul    z6.s,  p0/m, z6.s,  z22.s
    fmul    z7.s,  p0/m, z7.s,  z23.s

    // grad_unscaled = term2 - term1 = (1-t)/q - t/p
    fsub    z4.s, p0/m, z4.s, z8.s
    fsub    z5.s, p0/m, z5.s, z9.s
    fsub    z6.s, p0/m, z6.s, z10.s
    fsub    z7.s, p0/m, z7.s, z11.s

    // grad = grad_unscaled * inv_n
    fmul    z4.s, p0/m, z4.s, z18.s
    fmul    z5.s, p0/m, z5.s, z18.s
    fmul    z6.s, p0/m, z6.s, z18.s
    fmul    z7.s, p0/m, z7.s, z18.s

    st1w    {z4.s-z7.s}, pn9, [x2, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lbceb_loop

    smstop

.Lbceb_done:
    ldp     d8,  d9,  [sp, #48]
    ldp     d10, d11, [sp, #64]
    ldp     d12, d13, [sp, #80]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #96
    ret

.p2align 2
.Lbceb_const:
    .long   0x3F800000  // 1.0f      (offset 0)
    .long   0x33D6BF95  // 1e-7f eps (offset 4)
