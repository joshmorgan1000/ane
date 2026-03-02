// mae_loss_backward_fp32.s — MAE loss backward pass via SME2 streaming SVE
//
// void mae_loss_backward_fp32(const float* pred, const float* target, float* grad, long n)
// AAPCS: x0=pred, x1=target, x2=grad, x3=n
//
// Math: grad[i] = sign(pred[i] - target[i]) / n
//   = +1/n when pred > target
//   = -1/n when pred < target
//   =  0   when pred == target
//
// Strategy:
//   1. Compute scalar inv_n = 1/n before smstart; store to stack.
//   2. After smstart, broadcast +inv_n into z8 and -inv_n into z9.
//   3. Load 4 vectors of pred into z0-z3, 4 of target into z4-z7.
//   4. Compute d = pred - target; hold in z4-z7 (reuse target slots).
//   5. Use z16 as zero constant.
//   6. For each of the 4 diff vectors:
//      - Make p_pos (d > 0) and p_neg (d < 0).
//      - Build result in z0-z3: start from z16 (zero), sel +inv_n where pos,
//        then sel -inv_n where neg. Reuse the same output register.
//   7. Store z0-z3 (multiple of 4, valid for vlx4 st1w).
//
// Processing: 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _mae_loss_backward_fp32
.p2align 4

_mae_loss_backward_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    // sp+88: scratch slot for scalar

    cbz     x3, .Lmaeb_done

    // Compute inv_n = 1/n in scalar before smstart.
    scvtf   s0, x3                      // s0 = (float)n
    adr     x9, .Lmaeb_const
    ldr     s1, [x9]                    // s1 = 1.0
    fdiv    s0, s1, s0                  // s0 = 1.0/n
    str     s0, [sp, #88]               // save for broadcast after smstart

    smstart sm

    ptrue   p0.s

    // Broadcast +inv_n into z8, -inv_n into z9.
    ld1rw   {z8.s}, p0/z, [sp, #88]    // z8 = +inv_n (all lanes)
    fneg    z9.s, p0/m, z8.s            // z9 = -inv_n (all lanes)

    // Zero register for sel fallback. Use z16 (caller-saved, not d16).
    mov     z16.d, #0                   // z16 = 0.0

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lmaeb_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]   // pred
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]   // target

    // d = pred - target, stored in z0-z3.
    // Standard destructive SVE: fsub Zdn, Pg/M, Zdn, Zm → Zdn = Zdn - Zm
    // z0 holds pred, z4 holds target → z0 = pred - target.
    fsub    z0.s, p0/m, z0.s, z4.s
    fsub    z1.s, p0/m, z1.s, z5.s
    fsub    z2.s, p0/m, z2.s, z6.s
    fsub    z3.s, p0/m, z3.s, z7.s
    // z0-z3 = diff; z4-z7 are now free.

    // For d0 (z0): p_pos = d>0, p_neg = d<0. Build result in z4.
    fcmgt   p1.s, p0/z, z0.s, z16.s
    fcmlt   p2.s, p0/z, z0.s, z16.s
    mov     z4.d, z16.d
    sel     z4.s, p1, z8.s, z4.s
    sel     z4.s, p2, z9.s, z4.s

    // For d1 (z1) → z5.
    fcmgt   p3.s, p0/z, z1.s, z16.s
    fcmlt   p4.s, p0/z, z1.s, z16.s
    mov     z5.d, z16.d
    sel     z5.s, p3, z8.s, z5.s
    sel     z5.s, p4, z9.s, z5.s

    // For d2 (z2) → z6.
    fcmgt   p5.s, p0/z, z2.s, z16.s
    fcmlt   p6.s, p0/z, z2.s, z16.s
    mov     z6.d, z16.d
    sel     z6.s, p5, z8.s, z6.s
    sel     z6.s, p6, z9.s, z6.s

    // For d3 (z3) → z7. Reuse p1/p2.
    fcmgt   p1.s, p0/z, z3.s, z16.s
    fcmlt   p2.s, p0/z, z3.s, z16.s
    mov     z7.d, z16.d
    sel     z7.s, p1, z8.s, z7.s
    sel     z7.s, p2, z9.s, z7.s

    st1w    {z4.s-z7.s}, pn9, [x2, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lmaeb_loop

    smstop

.Lmaeb_done:
    ldp     d8,  d9,  [sp, #48]
    ldp     d10, d11, [sp, #64]
    ldp     d12, d13, [sp, #80]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #96
    ret

.p2align 2
.Lmaeb_const:
    .long   0x3F800000  // 1.0f
