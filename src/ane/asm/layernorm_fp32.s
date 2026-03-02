// layernorm_fp32.s — Fused Layer Normalization (forward) via SME2 streaming SVE
//
// void layernorm_fp32(const float *x, const float *gamma, const float *beta,
//                     float *out, float *mean_out, float *inv_std_out,
//                     float eps, long n)
//
// AAPCS: x0=x, x1=gamma, x2=beta, x3=out, x4=mean_out, x5=inv_std_out,
//        s0=eps, x6=n
//
// Computes:
//   mean = (1/n) * sum(x[i])
//   var  = (1/n) * sum(x[i]^2) - mean^2
//   inv_std = 1/sqrt(var + eps)
//   out[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i]
//
// Outputs mean and inv_std for use in backward pass.
//
// Two-phase algorithm, fully vectorized:
//   Phase 1: Single pass computing sum(x) and sum(x^2) simultaneously
//            using 8 independent vector accumulators (4 for sum, 4 for sumsqr)
//   Interlude: faddv → broadcast → vectorized frecpe/frsqrte + Newton-Raphson
//   Phase 2: Normalize and apply affine: out = gamma * (x - mean) * inv_std + beta
//
// Processes 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _layernorm_fp32
.p2align 4

_layernorm_fp32:
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

    cbz     x6, .Lln_done

    // Save arguments in callee-saved registers
    mov     x19, x0                 // x
    mov     x20, x1                 // gamma
    mov     x21, x2                 // beta
    mov     x22, x3                 // out
    mov     x23, x4                 // mean_out
    mov     x24, x5                 // inv_std_out

    // Save eps to GP register before smstart (smstart zeroes s0)
    fmov    w9, s0                  // eps bits → w9

    // Save n to callee-saved
    mov     x25, x6                 // n

    smstart sm

    ptrue   p0.s

    // Restore eps: broadcast from GP → vector
    fmov    s0, w9
    mov     z16.s, s0               // z16 = eps (broadcast)

    // ═════════════════════════════════════════════════════════════
    // Phase 1: Compute sum(x) and sum(x^2) in a single pass
    //   z8-z11  = 4 accumulators for sum(x)
    //   z12-z15 = 4 accumulators for sum(x^2)
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

.Lln_stat_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Accumulate sum(x)
    fadd    z8.s,  p0/m, z8.s,  z0.s
    fadd    z9.s,  p0/m, z9.s,  z1.s
    fadd    z10.s, p0/m, z10.s, z2.s
    fadd    z11.s, p0/m, z11.s, z3.s

    // Accumulate sum(x^2) via fmla: acc += x * x
    fmla    z12.s, p0/m, z0.s, z0.s
    fmla    z13.s, p0/m, z1.s, z1.s
    fmla    z14.s, p0/m, z2.s, z2.s
    fmla    z15.s, p0/m, z3.s, z3.s

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x25, vlx4
    b.first .Lln_stat_loop

    // Tree-reduce sum(x): z8-z11 → z8 → scalar
    fadd    z8.s,  p0/m, z8.s,  z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s,  p0/m, z8.s,  z10.s
    faddv   s8, p0, z8.s            // s8 = sum_x

    // Tree-reduce sum(x^2): z12-z15 → z12 → scalar
    fadd    z12.s, p0/m, z12.s, z13.s
    fadd    z14.s, p0/m, z14.s, z15.s
    fadd    z12.s, p0/m, z12.s, z14.s
    faddv   s12, p0, z12.s          // s12 = sum_x2

    // ═════════════════════════════════════════════════════════════
    // Vectorized interlude: compute mean, var, inv_std
    //   All Newton-Raphson done on broadcast vectors, not scalars.
    // ═════════════════════════════════════════════════════════════

    // Broadcast n to a vector for vectorized reciprocal
    scvtf   s17, x25                // s17 = (float)n
    mov     z17.s, s17              // z17 = n (broadcast)

    // inv_n = 1/n via vectorized frecpe + 2 Newton-Raphson steps
    frecpe  z18.s, z17.s            // z18 ≈ 1/n
    frecps  z19.s, z17.s, z18.s     // refinement
    fmul    z18.s, z18.s, z19.s
    frecps  z19.s, z17.s, z18.s
    fmul    z18.s, z18.s, z19.s     // z18 = 1/n (precise, broadcast)

    // mean = sum_x * inv_n (broadcast scalar × broadcast vector)
    mov     z20.s, s8               // z20 = sum_x (broadcast)
    fmul    z20.s, z20.s, z18.s     // z20 = mean (broadcast)

    // var = sum_x2 * inv_n - mean^2
    mov     z21.s, s12              // z21 = sum_x2 (broadcast)
    fmul    z21.s, z21.s, z18.s     // z21 = sum_x2 / n
    fmls    z21.s, p0/m, z20.s, z20.s  // z21 = sum_x2/n - mean^2 = var

    // var + eps
    fadd    z21.s, p0/m, z21.s, z16.s  // z21 = var + eps

    // inv_std = 1/sqrt(var + eps) via vectorized frsqrte + 2 Newton-Raphson steps
    frsqrte z22.s, z21.s            // z22 ≈ 1/sqrt(var+eps)
    fmul    z23.s, z22.s, z22.s     // e^2
    frsqrts z23.s, z21.s, z23.s     // (3 - x*e^2)/2
    fmul    z22.s, z22.s, z23.s     // refined
    fmul    z23.s, z22.s, z22.s     // e^2 (step 2)
    frsqrts z23.s, z21.s, z23.s
    fmul    z22.s, z22.s, z23.s     // z22 = inv_std (precise, broadcast)

    // z20 = mean (broadcast), z22 = inv_std (broadcast)
    // Save mean and inv_std to stack for output after smstop
    // Use faddv identity (already scalar in lane 0) or just extract
    // Since z20/z22 are broadcast, lane 0 has the scalar we need
    str     s20, [sp, #-16]!        // [sp] = mean
    str     s22, [sp, #4]           // [sp+4] = inv_std

    // Alias for readability in phase 2
    // z30 = mean, z31 = inv_std
    mov     z30.d, z20.d
    mov     z31.d, z22.d

    // ═════════════════════════════════════════════════════════════
    // Phase 2: Normalize and apply affine transform
    //   out[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i]
    // ═════════════════════════════════════════════════════════════

    mov     x8, #0
    whilelt pn9.s, x8, x25, vlx4

.Lln_norm_loop:
    // Load x
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // x_hat = (x - mean) * inv_std
    fsub    z0.s, p0/m, z0.s, z30.s
    fsub    z1.s, p0/m, z1.s, z30.s
    fsub    z2.s, p0/m, z2.s, z30.s
    fsub    z3.s, p0/m, z3.s, z30.s
    fmul    z0.s, p0/m, z0.s, z31.s
    fmul    z1.s, p0/m, z1.s, z31.s
    fmul    z2.s, p0/m, z2.s, z31.s
    fmul    z3.s, p0/m, z3.s, z31.s

    // Load gamma, multiply: gamma * x_hat
    ld1w    {z4.s-z7.s}, pn9/z, [x20, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z4.s
    fmul    z1.s, p0/m, z1.s, z5.s
    fmul    z2.s, p0/m, z2.s, z6.s
    fmul    z3.s, p0/m, z3.s, z7.s

    // Load beta, add: gamma * x_hat + beta
    ld1w    {z4.s-z7.s}, pn9/z, [x21, x8, lsl #2]
    fadd    z0.s, p0/m, z0.s, z4.s
    fadd    z1.s, p0/m, z1.s, z5.s
    fadd    z2.s, p0/m, z2.s, z6.s
    fadd    z3.s, p0/m, z3.s, z7.s

    // Store output
    st1w    {z0.s-z3.s}, pn9, [x22, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x25, vlx4
    b.first .Lln_norm_loop

    smstop

    // Write mean and inv_std outputs (reload from stack after smstop)
    ldr     s0, [sp]
    ldr     s1, [sp, #4]
    str     s0, [x23]               // *mean_out = mean
    str     s1, [x24]               // *inv_std_out = inv_std

    add     sp, sp, #16             // pop scratch

.Lln_done:
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
