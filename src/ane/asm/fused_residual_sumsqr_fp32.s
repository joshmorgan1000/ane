// fused_residual_sumsqr_fp32.s — Fused residual add + sum-of-squares
//
// void fused_residual_sumsqr_fp32(const float* residual, const float* x,
//                                  float* hidden, float* ss_out, long n)
//
// AAPCS: x0=residual, x1=x, x2=hidden, x3=ss_out, x4=n
//
// This kernel fuses:
//   hidden[i] = residual[i] + x[i]     (element-wise add)
//   ss = sum(hidden[i]^2)              (sum of squares for RMS norm)
//
// Saves one memory round-trip compared to separate add + sumsqr.
// Used at the start of every RMS normalization in transformer layers.

.section __TEXT,__text,regular,pure_instructions
.global _fused_residual_sumsqr_fp32
.p2align 4

_fused_residual_sumsqr_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8,  d9,  [sp, #32]
    stp     d10, d11, [sp, #48]
    stp     d12, d13, [sp, #64]
    stp     d14, d15, [sp, #80]

    // Return 0 if n=0
    cbz     x4, .Lzero

    // Save ss_out pointer
    mov     x19, x3

    smstart sm

    ptrue   p0.s

    // Initialize 4 independent accumulators for sum-of-squares
    // Using 4 accumulators hides fadd latency
    mov     z8.d,  #0
    mov     z9.d,  #0
    mov     z10.d, #0
    mov     z11.d, #0

    mov     x8, #0
    whilelt pn9.s, x8, x4, vlx4

.Lloop:
    // Load residual and x
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]

    // hidden = residual + x
    fadd    z0.s, p0/m, z0.s, z4.s
    fadd    z1.s, p0/m, z1.s, z5.s
    fadd    z2.s, p0/m, z2.s, z6.s
    fadd    z3.s, p0/m, z3.s, z7.s

    // Store hidden result
    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]

    // Compute squares: z4-z7 = hidden^2
    fmul    z4.s, z0.s, z0.s
    fmul    z5.s, z1.s, z1.s
    fmul    z6.s, z2.s, z2.s
    fmul    z7.s, z3.s, z3.s

    // Accumulate sum of squares
    fadd    z8.s,  p0/m, z8.s,  z4.s
    fadd    z9.s,  p0/m, z9.s,  z5.s
    fadd    z10.s, p0/m, z10.s, z6.s
    fadd    z11.s, p0/m, z11.s, z7.s

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x4, vlx4
    b.first .Lloop

    // Tree-reduce 4 accumulators → 1
    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s

    // Horizontal sum → scalar
    faddv   s0, p0, z8.s

    // Save result to stack before smstop
    str     s0, [sp, #96]

    smstop

    // Reload result after smstop and store to output
    ldr     s0, [sp, #96]
    str     s0, [x19]
    b       .Ldone

.Lzero:
    fmov    s0, wzr
    str     s0, [x3]

.Ldone:
    ldp     x19, x20, [sp, #16]
    ldp     d8,  d9,  [sp, #32]
    ldp     d10, d11, [sp, #48]
    ldp     d12, d13, [sp, #64]
    ldp     d14, d15, [sp, #80]
    ldp     x29, x30, [sp], #112
    ret
