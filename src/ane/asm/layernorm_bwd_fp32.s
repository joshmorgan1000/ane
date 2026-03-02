// layernorm_bwd_fp32.s — Layer Normalization backward pass via SME2 streaming SVE
//
// void layernorm_bwd_fp32(const float *dy, const float *x, const float *gamma,
//                          float mean, float inv_std,
//                          float *dx, float *dgamma, float *dbeta, long n)
//
// AAPCS: x0=dy, x1=x, x2=gamma, s0=mean, s1=inv_std,
//        x3=dx, x4=dgamma, x5=dbeta, x6=n
//
// Computes LayerNorm backward pass:
//   x_hat[i] = (x[i] - mean) * inv_std
//   c1 = sum(dy[i] * gamma[i])
//   c2 = sum(dy[i] * gamma[i] * x_hat[i])
//   dx[i]     = inv_std * (dy[i]*gamma[i] - c1/n - x_hat[i] * c2/n)
//   dgamma[i] = dy[i] * x_hat[i]
//   dbeta[i]  = dy[i]
//
// Two-phase algorithm, fully vectorized:
//   Phase 1: Compute c1 = sum(dy*gamma) and c2 = sum(dy*gamma*x_hat)
//            x_hat recomputed on-the-fly from x, mean, inv_std
//   Interlude: faddv → broadcast → vectorized frecpe NR for 1/n
//   Phase 2: Compute dx, dgamma, dbeta using broadcasted c1/n and c2/n
//
// Processes 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _layernorm_bwd_fp32
.p2align 4

_layernorm_bwd_fp32:
    stp     x29, x30, [sp, #-144]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     d8,  d9,  [sp, #80]
    stp     d10, d11, [sp, #96]
    stp     d12, d13, [sp, #112]
    stp     d14, d15, [sp, #128]

    cbz     x6, .Llnb_done

    // Save arguments in callee-saved registers
    mov     x19, x0                 // dy
    mov     x20, x1                 // x
    mov     x21, x2                 // gamma
    mov     x22, x3                 // dx
    mov     x23, x4                 // dgamma
    mov     x24, x5                 // dbeta
    mov     x25, x6                 // n

    // Save float args via bitwise copy to GP registers before smstart
    fmov    w9, s0                  // mean bits → w9
    fmov    w10, s1                 // inv_std bits → w10

    smstart sm

    ptrue   p0.s

    // Restore and broadcast mean and inv_std from GP registers
    fmov    s0, w9
    fmov    s1, w10
    mov     z30.s, s0               // z30 = mean (broadcast)
    mov     z31.s, s1               // z31 = inv_std (broadcast)

    // ═════════════════════════════════════════════════════════════
    // Phase 1: Compute c1 = sum(dy*gamma) and c2 = sum(dy*gamma*x_hat)
    //   z8-z11  = 4 accumulators for c1
    //   z12-z15 = 4 accumulators for c2
    // ═════════════════════════════════════════════════════════════

    mov     z8.d,  #0
    mov     z9.d,  #0
    mov     z10.d, #0
    mov     z11.d, #0
    mov     z12.d, #0
    mov     z13.d, #0
    mov     z14.d, #0
    mov     z15.d, #0

    mov     x8, #0
    whilelt pn9.s, x8, x25, vlx4

.Llnb_sum_loop:
    // Load dy
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]
    // Load gamma
    ld1w    {z4.s-z7.s}, pn9/z, [x21, x8, lsl #2]

    // dy_gamma = dy * gamma → z4-z7 (reuse gamma slots)
    fmul    z4.s, z0.s, z4.s
    fmul    z5.s, z1.s, z5.s
    fmul    z6.s, z2.s, z6.s
    fmul    z7.s, z3.s, z7.s

    // Accumulate c1 += dy_gamma
    fadd    z8.s,  p0/m, z8.s,  z4.s
    fadd    z9.s,  p0/m, z9.s,  z5.s
    fadd    z10.s, p0/m, z10.s, z6.s
    fadd    z11.s, p0/m, z11.s, z7.s

    // Load x, compute x_hat = (x - mean) * inv_std → z0-z3 (reuse dy slots)
    ld1w    {z0.s-z3.s}, pn9/z, [x20, x8, lsl #2]
    fsub    z0.s, p0/m, z0.s, z30.s
    fsub    z1.s, p0/m, z1.s, z30.s
    fsub    z2.s, p0/m, z2.s, z30.s
    fsub    z3.s, p0/m, z3.s, z30.s
    fmul    z0.s, p0/m, z0.s, z31.s
    fmul    z1.s, p0/m, z1.s, z31.s
    fmul    z2.s, p0/m, z2.s, z31.s
    fmul    z3.s, p0/m, z3.s, z31.s

    // Accumulate c2 += dy_gamma * x_hat
    fmla    z12.s, p0/m, z4.s, z0.s
    fmla    z13.s, p0/m, z5.s, z1.s
    fmla    z14.s, p0/m, z6.s, z2.s
    fmla    z15.s, p0/m, z7.s, z3.s

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x25, vlx4
    b.first .Llnb_sum_loop

    // Tree-reduce c1: z8-z11 → scalar
    fadd    z8.s,  p0/m, z8.s,  z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s,  p0/m, z8.s,  z10.s
    faddv   s8, p0, z8.s            // s8 = c1

    // Tree-reduce c2: z12-z15 → scalar
    fadd    z12.s, p0/m, z12.s, z13.s
    fadd    z14.s, p0/m, z14.s, z15.s
    fadd    z12.s, p0/m, z12.s, z14.s
    faddv   s12, p0, z12.s          // s12 = c2

    // ═════════════════════════════════════════════════════════════
    // Vectorized interlude: compute c1/n and c2/n via frecpe NR
    // ═════════════════════════════════════════════════════════════

    // Broadcast n to vector for vectorized reciprocal
    scvtf   s16, x25                // s16 = (float)n
    mov     z16.s, s16              // z16 = n (broadcast)

    // inv_n = 1/n via vectorized frecpe + 2 Newton-Raphson steps
    frecpe  z17.s, z16.s
    frecps  z18.s, z16.s, z17.s
    fmul    z17.s, z17.s, z18.s
    frecps  z18.s, z16.s, z17.s
    fmul    z17.s, z17.s, z18.s     // z17 = 1/n (broadcast)

    // c1/n and c2/n as broadcast vectors
    mov     z28.s, s8               // z28 = c1 (broadcast)
    fmul    z28.s, z28.s, z17.s     // z28 = c1/n (broadcast)
    mov     z29.s, s12              // z29 = c2 (broadcast)
    fmul    z29.s, z29.s, z17.s     // z29 = c2/n (broadcast)

    // ═════════════════════════════════════════════════════════════
    // Phase 2: Compute dx, dgamma, dbeta
    //   x_hat[i] = (x[i] - mean) * inv_std
    //   dx[i]     = inv_std * (dy[i]*gamma[i] - c1/n - x_hat[i]*c2/n)
    //   dgamma[i] = dy[i] * x_hat[i]
    //   dbeta[i]  = dy[i]
    // ═════════════════════════════════════════════════════════════

    mov     x8, #0
    whilelt pn9.s, x8, x25, vlx4

.Llnb_grad_loop:
    // Load dy → z0-z3
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Store dbeta = dy (before we modify z0-z3)
    st1w    {z0.s-z3.s}, pn9, [x24, x8, lsl #2]

    // Load x, compute x_hat = (x - mean) * inv_std → z4-z7
    ld1w    {z4.s-z7.s}, pn9/z, [x20, x8, lsl #2]
    fsub    z4.s, p0/m, z4.s, z30.s
    fsub    z5.s, p0/m, z5.s, z30.s
    fsub    z6.s, p0/m, z6.s, z30.s
    fsub    z7.s, p0/m, z7.s, z30.s
    fmul    z4.s, p0/m, z4.s, z31.s
    fmul    z5.s, p0/m, z5.s, z31.s
    fmul    z6.s, p0/m, z6.s, z31.s
    fmul    z7.s, p0/m, z7.s, z31.s

    // Compute dgamma = dy * x_hat → z16-z19 (clobbers z16, which is fine)
    fmul    z16.s, z0.s, z4.s
    fmul    z17.s, z1.s, z5.s
    fmul    z18.s, z2.s, z6.s
    fmul    z19.s, z3.s, z7.s
    // Store dgamma
    st1w    {z16.s-z19.s}, pn9, [x23, x8, lsl #2]

    // Load gamma → z16-z19
    ld1w    {z16.s-z19.s}, pn9/z, [x21, x8, lsl #2]

    // Compute dy_gamma = dy * gamma → z0-z3
    fmul    z0.s, p0/m, z0.s, z16.s
    fmul    z1.s, p0/m, z1.s, z17.s
    fmul    z2.s, p0/m, z2.s, z18.s
    fmul    z3.s, p0/m, z3.s, z19.s

    // Compute dy_gamma - c1/n → z0-z3
    fsub    z0.s, p0/m, z0.s, z28.s
    fsub    z1.s, p0/m, z1.s, z28.s
    fsub    z2.s, p0/m, z2.s, z28.s
    fsub    z3.s, p0/m, z3.s, z28.s

    // Compute dy_gamma - c1/n - x_hat * c2/n → z0-z3
    // Using fmls: z0 = z0 - z4 * z29
    fmls    z0.s, p0/m, z4.s, z29.s
    fmls    z1.s, p0/m, z5.s, z29.s
    fmls    z2.s, p0/m, z6.s, z29.s
    fmls    z3.s, p0/m, z7.s, z29.s

    // dx = inv_std * (dy_gamma - c1/n - x_hat*c2/n)
    fmul    z0.s, p0/m, z0.s, z31.s
    fmul    z1.s, p0/m, z1.s, z31.s
    fmul    z2.s, p0/m, z2.s, z31.s
    fmul    z3.s, p0/m, z3.s, z31.s

    // Store dx
    st1w    {z0.s-z3.s}, pn9, [x22, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x25, vlx4
    b.first .Llnb_grad_loop

    smstop

.Llnb_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     d8,  d9,  [sp, #80]
    ldp     d10, d11, [sp, #96]
    ldp     d12, d13, [sp, #112]
    ldp     d14, d15, [sp, #128]
    ldp     x29, x30, [sp], #144
    ret
