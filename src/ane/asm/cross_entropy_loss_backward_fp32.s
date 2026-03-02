// cross_entropy_loss_backward_fp32.s — Cross-entropy loss backward via SME2 streaming SVE
//
// void cross_entropy_loss_backward_fp32(const float* pred, const float* target, float* grad, long n)
// AAPCS: x0=pred, x1=target, x2=grad, x3=n
//
// Math: grad[i] = -target[i] / (max(pred[i], eps) * n)
//              = -target[i] * inv_n / max(pred[i], eps)
//              = -target[i] * inv_n * recip(max(pred[i], eps))
//
// Strategy:
//   1. Compute scalar inv_n = 1/n before smstart; store to stack.
//   2. Broadcast eps into z16, inv_n into z17.
//   3. Per block:
//      a. Load pred into z0-z3, clamp each to eps using fmax.
//      b. Compute per-lane reciprocal via frecpe + frecps + fmul (2 NR steps).
//      c. Load target into z4-z7.
//      d. grad = -target * inv_n * recip(pred_clamped)
//         = fneg(target) * inv_n * recip  → use fmul chain.
//
// eps = 1e-7f = 0x33D6BF95
// Processing: 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _cross_entropy_loss_backward_fp32
.p2align 4

_cross_entropy_loss_backward_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    // sp+88: scratch slot

    cbz     x3, .Lceb_done

    // Compute inv_n = 1/n in scalar before smstart.
    scvtf   s0, x3
    adr     x9, .Lceb_const
    ldr     s1, [x9]                    // 1.0
    fdiv    s0, s1, s0                  // inv_n = 1.0/n
    str     s0, [sp, #88]

    smstart sm

    ptrue   p0.s

    // Load constant vectors.
    adr     x9, .Lceb_const
    ld1rw   {z16.s}, p0/z, [x9, #4]    // z16 = eps = 1e-7
    ld1rw   {z17.s}, p0/z, [sp, #88]   // z17 = inv_n (broadcast from stack)

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lceb_loop:
    // Load pred, clamp to eps.
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    fmax    z0.s, p0/m, z0.s, z16.s
    fmax    z1.s, p0/m, z1.s, z16.s
    fmax    z2.s, p0/m, z2.s, z16.s
    fmax    z3.s, p0/m, z3.s, z16.s

    // Reciprocal of clamped pred: frecpe + 2x NR (frecps + fmul).
    frecpe  z4.s, z0.s
    frecpe  z5.s, z1.s
    frecpe  z6.s, z2.s
    frecpe  z7.s, z3.s

    // NR step 1
    frecps  z8.s,  z0.s, z4.s
    frecps  z9.s,  z1.s, z5.s
    frecps  z10.s, z2.s, z6.s
    frecps  z11.s, z3.s, z7.s
    fmul    z4.s, p0/m, z4.s, z8.s
    fmul    z5.s, p0/m, z5.s, z9.s
    fmul    z6.s, p0/m, z6.s, z10.s
    fmul    z7.s, p0/m, z7.s, z11.s

    // NR step 2
    frecps  z8.s,  z0.s, z4.s
    frecps  z9.s,  z1.s, z5.s
    frecps  z10.s, z2.s, z6.s
    frecps  z11.s, z3.s, z7.s
    fmul    z4.s, p0/m, z4.s, z8.s
    fmul    z5.s, p0/m, z5.s, z9.s
    fmul    z6.s, p0/m, z6.s, z10.s
    fmul    z7.s, p0/m, z7.s, z11.s
    // z4-z7 = 1/max(pred, eps)

    // Load target.
    ld1w    {z0.s-z3.s}, pn9/z, [x1, x8, lsl #2]

    // grad = -target * inv_n * recip(pred_clamped)
    // Compute: z_out = target * inv_n  (then multiply by recip, then negate)
    fmul    z0.s, p0/m, z0.s, z17.s    // target * inv_n
    fmul    z1.s, p0/m, z1.s, z17.s
    fmul    z2.s, p0/m, z2.s, z17.s
    fmul    z3.s, p0/m, z3.s, z17.s

    fmul    z0.s, p0/m, z0.s, z4.s     // * recip
    fmul    z1.s, p0/m, z1.s, z5.s
    fmul    z2.s, p0/m, z2.s, z6.s
    fmul    z3.s, p0/m, z3.s, z7.s

    fneg    z0.s, p0/m, z0.s            // negate: -target*inv_n*recip
    fneg    z1.s, p0/m, z1.s
    fneg    z2.s, p0/m, z2.s
    fneg    z3.s, p0/m, z3.s

    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lceb_loop

    smstop

.Lceb_done:
    ldp     d8,  d9,  [sp, #48]
    ldp     d10, d11, [sp, #64]
    ldp     d12, d13, [sp, #80]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #96
    ret

.p2align 2
.Lceb_const:
    .long   0x3F800000  // 1.0f      (offset 0)
    .long   0x33D6BF95  // 1e-7f eps (offset 4)
