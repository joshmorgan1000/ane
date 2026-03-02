// fused_batchnorm_relu_fp32.s -- Fused batch normalization + ReLU
//
// Computes: out[i] = max(0, gamma[i] * (x[i] - mean) * inv_std + beta[i])
//
// Where mean and inv_std = 1/sqrt(var + eps) are pre-computed by caller.
//
// void fused_batchnorm_relu_fp32(const float* x, long n,
//                                 float mean, float inv_std,
//                                 const float* gamma, const float* beta,
//                                 float* out)
//
// AAPCS64: x0=x, x1=n, s0=mean, s1=inv_std, x2=gamma, x3=beta, x4=out

.section __TEXT,__text,regular,pure_instructions
.global _fused_batchnorm_relu_fp32
.p2align 4

_fused_batchnorm_relu_fp32:
    stp     x29, x30, [sp, #-128]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     d8,  d9,  [sp, #64]
    stp     d10, d11, [sp, #80]
    stp     d12, d13, [sp, #96]
    stp     d14, d15, [sp, #112]

    // Validate
    cbz     x1, .Lfbnr_done

    // Save pointers
    mov     x19, x0                 // x
    mov     x20, x1                 // n
    mov     x21, x2                 // gamma
    mov     x22, x3                 // beta
    mov     x23, x4                 // out

    // Save float args before smstart
    fmov    w9, s0                  // mean bits
    fmov    w10, s1                 // inv_std bits

    smstart sm

    ptrue   p0.s

    // Restore and broadcast mean and inv_std
    fmov    s0, w9
    fmov    s1, w10
    mov     z28.s, s0               // broadcast mean
    mov     z29.s, s1               // broadcast inv_std
    mov     z30.s, #0               // broadcast zero for ReLU

    mov     x8, #0
    whilelt pn9.s, x8, x20, vlx4

.Lfbnr_loop:
    // Load input
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Normalize: (x - mean) * inv_std
    fsub    z0.s, p0/m, z0.s, z28.s
    fsub    z1.s, p0/m, z1.s, z28.s
    fsub    z2.s, p0/m, z2.s, z28.s
    fsub    z3.s, p0/m, z3.s, z28.s
    fmul    z0.s, p0/m, z0.s, z29.s
    fmul    z1.s, p0/m, z1.s, z29.s
    fmul    z2.s, p0/m, z2.s, z29.s
    fmul    z3.s, p0/m, z3.s, z29.s

    // Scale: gamma * normalized
    ld1w    {z4.s-z7.s}, pn9/z, [x21, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z4.s
    fmul    z1.s, p0/m, z1.s, z5.s
    fmul    z2.s, p0/m, z2.s, z6.s
    fmul    z3.s, p0/m, z3.s, z7.s

    // Shift: + beta
    ld1w    {z4.s-z7.s}, pn9/z, [x22, x8, lsl #2]
    fadd    z0.s, p0/m, z0.s, z4.s
    fadd    z1.s, p0/m, z1.s, z5.s
    fadd    z2.s, p0/m, z2.s, z6.s
    fadd    z3.s, p0/m, z3.s, z7.s

    // ReLU: max(0, result)
    fmax    z0.s, p0/m, z0.s, z30.s
    fmax    z1.s, p0/m, z1.s, z30.s
    fmax    z2.s, p0/m, z2.s, z30.s
    fmax    z3.s, p0/m, z3.s, z30.s

    // Store
    st1w    {z0.s-z3.s}, pn9, [x23, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x20, vlx4
    b.first .Lfbnr_loop

    smstop

.Lfbnr_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     d8,  d9,  [sp, #64]
    ldp     d10, d11, [sp, #80]
    ldp     d12, d13, [sp, #96]
    ldp     d14, d15, [sp, #112]
    ldp     x29, x30, [sp], #128
    ret
