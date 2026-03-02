// mse_loss_backward_fp32.s — MSE loss backward pass via SME2 streaming SVE
//
// void mse_loss_backward_fp32(const float* pred, const float* target, float* grad, long n)
// AAPCS: x0=pred, x1=target, x2=grad, x3=n
//
// Math: grad[i] = 2 * (pred[i] - target[i]) / n
//
// Strategy:
//   1. Convert n to float, compute inv_n = 1/n via scalar fdiv.
//   2. Precompute scalar = 2 * inv_n, broadcast to vector register z8.
//   3. For each 64-element block: load pred, load target, subtract, multiply by z8, store.
//
// Processing: 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _mse_loss_backward_fp32
.p2align 4

_mse_loss_backward_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    // sp+88 = 8 scratch bytes (scalar result slot, 16-byte aligned pair used for d12/d13)

    cbz     x3, .Lmb_done

    // Compute 2/n in scalar before smstart (no float args to preserve).
    // x3 = n (long integer). Convert to float, compute 2.0/n, store to stack.
    scvtf   s0, x3                      // s0 = (float)n
    adr     x9, .Lmb_const
    ldr     s1, [x9]                    // s1 = 2.0
    fdiv    s0, s1, s0                  // s0 = 2.0 / n
    str     s0, [sp, #88]               // save scalar to stack

    smstart sm

    ptrue   p0.s

    // Broadcast scalar 2/n into z8 (all lanes).
    ld1rw   {z8.s}, p0/z, [sp, #88]

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lmb_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]   // pred
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]   // target

    // diff = pred - target
    fsub    z0.s, p0/m, z0.s, z4.s
    fsub    z1.s, p0/m, z1.s, z5.s
    fsub    z2.s, p0/m, z2.s, z6.s
    fsub    z3.s, p0/m, z3.s, z7.s

    // grad = diff * (2/n)
    fmul    z0.s, p0/m, z0.s, z8.s
    fmul    z1.s, p0/m, z1.s, z8.s
    fmul    z2.s, p0/m, z2.s, z8.s
    fmul    z3.s, p0/m, z3.s, z8.s

    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lmb_loop

    smstop

.Lmb_done:
    ldp     d8,  d9,  [sp, #48]
    ldp     d10, d11, [sp, #64]
    ldp     d12, d13, [sp, #80]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #96
    ret

.p2align 2
.Lmb_const:
    .long   0x40000000  // 2.0f
