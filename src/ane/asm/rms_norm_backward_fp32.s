// rms_norm_backward_fp32.s — Single-threaded RMSNorm backward pass via SME2
//
// void rms_norm_backward_fp32(const float *dy, const float *x,
//                              const float *gamma, float *dx, float *dgamma,
//                              long n, float eps);
//
// AAPCS64: x0=dy, x1=x, x2=gamma, x3=dx, x4=dgamma, x5=n, s0=eps
//
// This is a per-row normalization backward (single normalization instance).
// n is the row width.
//
// Math:
//   rms     = sqrt( (1/n)*sum(x[j]^2) + eps )
//   inv_rms = 1/rms
//   x_norm[j] = x[j] * inv_rms
//   c1        = (1/n) * sum_j( gamma[j] * dy[j] * x_norm[j] )
//   dx[j]     = ( gamma[j]*dy[j] - c1*x_norm[j] ) * inv_rms
//   dgamma[j] = dy[j] * x_norm[j]
//
// Note: dbeta is not part of RMSNorm (no bias term), so it is omitted.
//
// Algorithm (3 passes over the row):
//   Pass 1: sum_sq = sum_j( x[j]^2 )  → compute inv_rms
//   Pass 2: c1 = (1/n)*sum_j( gamma[j]*dy[j]*(x[j]*inv_rms) )
//           simultaneously write dgamma[j] = dy[j]*(x[j]*inv_rms)
//   Pass 3: dx[j] = (gamma[j]*dy[j] - c1*(x[j]*inv_rms)) * inv_rms
//
// inv_rms computed via frecpe+frecps Newton step (2 iterations) over
// the scalar rms value, then broadcast to all vector lanes.
//
// Frame layout (128 bytes, 16-byte aligned):
//   [sp+  0] x29, x30
//   [sp+ 16] x19, x20
//   [sp+ 32] x21, x22
//   [sp+ 48] x23, x24
//   [sp+ 64] d8,  d9
//   [sp+ 80] d10, d11
//   [sp+ 96] d12, d13
//   [sp+112] d14, d15

.section __TEXT,__text,regular,pure_instructions
.global _rms_norm_backward_fp32
.p2align 4

_rms_norm_backward_fp32:
    stp     x29, x30, [sp, #-128]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     d8,  d9,  [sp, #64]
    stp     d10, d11, [sp, #80]
    stp     d12, d13, [sp, #96]
    stp     d14, d15, [sp, #112]

    // Early exit on n == 0
    cbz     x5, .Lrmsbk_done

    // Save arguments to callee-saved registers
    mov     x19, x0                 // dy
    mov     x20, x1                 // x
    mov     x21, x2                 // gamma
    mov     x22, x3                 // dx
    mov     x23, x4                 // dgamma
    mov     x24, x5                 // n

    // Save eps (float) before smstart (smstart zeroes s0)
    fmov    w9, s0                  // eps bits → w9

    smstart sm

    ptrue   p0.s

    // Restore eps and broadcast
    fmov    s0, w9
    mov     z29.s, s0               // z29 = eps (broadcast)

    // ═════════════════════════════════════════════════════════════
    // Pass 1: Compute sum_sq = sum_j( x[j]^2 )
    // ═════════════════════════════════════════════════════════════

    mov     z8.d,  #0               // acc0
    mov     z9.d,  #0               // acc1
    mov     z10.d, #0               // acc2
    mov     z11.d, #0               // acc3

    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lrmsbk_sumsq_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x20, x8, lsl #2]  // x

    fmla    z8.s,  p0/m, z0.s, z0.s
    fmla    z9.s,  p0/m, z1.s, z1.s
    fmla    z10.s, p0/m, z2.s, z2.s
    fmla    z11.s, p0/m, z3.s, z3.s

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lrmsbk_sumsq_loop

    // Tree-reduce 4 accumulators → 1 → scalar
    fadd    z8.s,  p0/m, z8.s,  z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s,  p0/m, z8.s,  z10.s
    faddv   s8, p0, z8.s            // s8 = sum_sq

    // Compute inv_rms = 1 / sqrt( sum_sq/n + eps )
    //   mean_sq = sum_sq / n
    //   rms²    = mean_sq + eps
    //   inv_rms = 1 / sqrt(rms²)
    scvtf   s9, x24                 // s9 = (float)n
    fdiv    s8, s8, s9              // s8 = sum_sq / n  (mean_sq)
    fadd    s8, s8, s0              // s8 = mean_sq + eps  (s0 = eps still valid)
    // fsqrt then reciprocal
    fsqrt   s8, s8                  // s8 = rms
    // Newton-Raphson reciprocal: 1/rms via frecpe+frecps (2 iterations for single precision)
    frecpe  s10, s8                 // initial estimate
    frecps  s11, s8, s10            // refinement factor
    fmul    s10, s10, s11           // refined
    frecps  s11, s8, s10            // second refinement
    fmul    s10, s10, s11           // s10 = inv_rms (full precision)

    // Broadcast inv_rms into z28 for vectorised use
    mov     z28.s, s10              // z28 = inv_rms

    // ═════════════════════════════════════════════════════════════
    // Pass 2: Compute c1 and write dgamma[j] = dy[j] * x_norm[j]
    //   x_norm[j] = x[j] * inv_rms
    //   c1_acc   += gamma[j] * dy[j] * x_norm[j]
    // ═════════════════════════════════════════════════════════════

    mov     z8.d,  #0               // c1_acc0
    mov     z9.d,  #0               // c1_acc1
    mov     z10.d, #0               // c1_acc2
    mov     z11.d, #0               // c1_acc3

    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lrmsbk_c1_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x20, x8, lsl #2]      // x
    ld1w    {z4.s-z7.s}, pn9/z, [x19, x8, lsl #2]      // dy
    ld1w    {z16.s-z19.s}, pn9/z, [x21, x8, lsl #2]    // gamma

    // x_norm = x * inv_rms
    fmul    z0.s, p0/m, z0.s, z28.s
    fmul    z1.s, p0/m, z1.s, z28.s
    fmul    z2.s, p0/m, z2.s, z28.s
    fmul    z3.s, p0/m, z3.s, z28.s

    // dgamma[j] = dy[j] * x_norm[j]
    fmul    z20.s, z4.s, z0.s
    fmul    z21.s, z5.s, z1.s
    fmul    z22.s, z6.s, z2.s
    fmul    z23.s, z7.s, z3.s
    st1w    {z20.s-z23.s}, pn9, [x23, x8, lsl #2]

    // c1_acc += gamma[j] * dy[j] * x_norm[j]
    //        = gamma[j] * dgamma[j]
    fmla    z8.s,  p0/m, z16.s, z20.s
    fmla    z9.s,  p0/m, z17.s, z21.s
    fmla    z10.s, p0/m, z18.s, z22.s
    fmla    z11.s, p0/m, z19.s, z23.s

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lrmsbk_c1_loop

    // Reduce c1
    fadd    z8.s,  p0/m, z8.s,  z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s,  p0/m, z8.s,  z10.s
    faddv   s8, p0, z8.s            // s8 = sum(gamma*dgamma)

    // c1 = sum / n  →  broadcast into z27
    fdiv    s8, s8, s9              // s9 still holds (float)n
    mov     z27.s, s8               // z27 = c1

    // ═════════════════════════════════════════════════════════════
    // Pass 3: dx[j] = ( gamma[j]*dy[j] - c1*x_norm[j] ) * inv_rms
    // ═════════════════════════════════════════════════════════════

    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lrmsbk_dx_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x20, x8, lsl #2]      // x
    ld1w    {z4.s-z7.s}, pn9/z, [x19, x8, lsl #2]      // dy
    ld1w    {z16.s-z19.s}, pn9/z, [x21, x8, lsl #2]    // gamma

    // x_norm = x * inv_rms
    fmul    z0.s, p0/m, z0.s, z28.s
    fmul    z1.s, p0/m, z1.s, z28.s
    fmul    z2.s, p0/m, z2.s, z28.s
    fmul    z3.s, p0/m, z3.s, z28.s

    // gamma_dy = gamma * dy
    fmul    z16.s, p0/m, z16.s, z4.s
    fmul    z17.s, p0/m, z17.s, z5.s
    fmul    z18.s, p0/m, z18.s, z6.s
    fmul    z19.s, p0/m, z19.s, z7.s

    // gamma_dy - c1 * x_norm  (fmls: z16 -= z0*z27)
    fmls    z16.s, p0/m, z0.s, z27.s
    fmls    z17.s, p0/m, z1.s, z27.s
    fmls    z18.s, p0/m, z2.s, z27.s
    fmls    z19.s, p0/m, z3.s, z27.s

    // * inv_rms
    fmul    z16.s, p0/m, z16.s, z28.s
    fmul    z17.s, p0/m, z17.s, z28.s
    fmul    z18.s, p0/m, z18.s, z28.s
    fmul    z19.s, p0/m, z19.s, z28.s

    st1w    {z16.s-z19.s}, pn9, [x22, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lrmsbk_dx_loop

    smstop

.Lrmsbk_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     d8,  d9,  [sp, #64]
    ldp     d10, d11, [sp, #80]
    ldp     d12, d13, [sp, #96]
    ldp     d14, d15, [sp, #112]
    ldp     x29, x30, [sp], #128
    ret
