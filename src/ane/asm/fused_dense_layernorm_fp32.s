// fused_dense_layernorm_fp32.s -- Fused dense (matvec) + layernorm + scale + bias
//
// Computes: temp = W @ x  (m elements), then layernorm(temp, gamma, beta, eps)
//   mean = sum(temp) / m
//   var  = sum((temp - mean)^2) / m
//   out[i] = gamma[i] * (temp[i] - mean) / sqrt(var + eps) + beta[i]
//
// void fused_dense_layernorm_fp32(const float* W, int m, int n,
//                                  const float* x, const float* gamma,
//                                  const float* beta, float eps, float* out)
//
// AAPCS64: x0=W, x1=m, x2=n, x3=x, x4=gamma, x5=beta, s0=eps, x6=out
//
// This kernel avoids materializing the intermediate matmul result to memory.
// Phase 1: Compute temp = W @ x using dot products row-by-row
// Phase 2: Compute mean and variance from temp
// Phase 3: Normalize, scale, and shift: out[i] = gamma[i] * (temp[i] - mean) * inv_std + beta[i]
//
// Since layernorm requires global statistics, we store temp in a stack-allocated
// scratch buffer, then normalize in a second pass.

.section __TEXT,__text,regular,pure_instructions
.global _fused_dense_layernorm_fp32
.p2align 4

_fused_dense_layernorm_fp32:
    stp     x29, x30, [sp, #-160]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8,  d9,  [sp, #96]
    stp     d10, d11, [sp, #112]
    stp     d12, d13, [sp, #128]
    stp     d14, d15, [sp, #144]

    // Early exit
    cbz     x1, .Lfdn_done         // m == 0
    cbz     x2, .Lfdn_done         // n == 0

    // Save arguments
    mov     x19, x0                 // W
    mov     x20, x1                 // m
    mov     x21, x2                 // n
    mov     x22, x3                 // x (input vector)
    mov     x23, x4                 // gamma
    mov     x24, x5                 // beta
    mov     x25, x6                 // out
    fmov    w26, s0                 // eps (saved as int bits)

    // Allocate temp buffer: m * 4 bytes on heap
    lsl     x0, x20, #2
    bl      _malloc
    mov     x27, x0                 // temp buffer

    // ================================================================
    // Phase 1: Compute temp = W @ x using dot products
    // For each row i in [0, m): temp[i] = dot(W[i*n : i*n+n], x)
    // ================================================================

    smstart sm
    ptrue   p0.s

    mov     x8, #0                  // row index
    lsl     x9, x21, #2            // n * 4 (row stride in W)

.Lfdn_matvec_loop:
    cmp     x8, x20
    b.ge    .Lfdn_matvec_done

    // Compute dot product of W[row] and x
    // w_ptr = W + row * n * 4
    mul     x10, x8, x9
    add     x10, x19, x10          // w_ptr

    // Accumulate dot product
    mov     z28.s, #0               // accumulator

    mov     x11, #0                 // inner index
    whilelt p1.s, x11, x21

.Lfdn_dot_loop:
    ld1w    {z0.s}, p1/z, [x10, x11, lsl #2]
    ld1w    {z1.s}, p1/z, [x22, x11, lsl #2]
    fmla    z28.s, p1/m, z0.s, z1.s

    incw    x11
    whilelt p1.s, x11, x21
    b.first .Lfdn_dot_loop

    // Horizontal sum of z28
    faddv   s2, p0, z28.s

    // Store temp[row]
    str     s2, [x27, x8, lsl #2]

    add     x8, x8, #1
    b       .Lfdn_matvec_loop

.Lfdn_matvec_done:
    // ================================================================
    // Phase 2: Compute mean and variance
    // mean = sum(temp) / m
    // var  = sum((temp - mean)^2) / m
    // ================================================================

    // Compute sum for mean
    mov     z28.s, #0               // sum accumulator
    mov     x8, #0
    whilelt p1.s, x8, x20

.Lfdn_sum_loop:
    ld1w    {z0.s}, p1/z, [x27, x8, lsl #2]
    fadd    z28.s, p1/m, z28.s, z0.s
    incw    x8
    whilelt p1.s, x8, x20
    b.first .Lfdn_sum_loop

    faddv   s2, p0, z28.s          // total sum
    scvtf   s3, x20                // (float)m
    fdiv    s4, s2, s3              // mean = sum / m

    // Compute variance
    mov     z29.s, s4               // broadcast mean
    mov     z28.s, #0               // var accumulator
    mov     x8, #0
    whilelt p1.s, x8, x20

.Lfdn_var_loop:
    ld1w    {z0.s}, p1/z, [x27, x8, lsl #2]
    fsub    z0.s, p1/m, z0.s, z29.s  // temp - mean
    fmla    z28.s, p1/m, z0.s, z0.s  // += (temp - mean)^2
    incw    x8
    whilelt p1.s, x8, x20
    b.first .Lfdn_var_loop

    faddv   s5, p0, z28.s          // total var sum
    fdiv    s5, s5, s3              // var = sum / m

    // inv_std = 1.0 / sqrt(var + eps)
    fmov    w10, s5
    fmov    s6, w26                 // eps
    fadd    s5, s5, s6              // var + eps
    fsqrt   s5, s5                  // sqrt(var + eps)
    fmov    s7, #1.0
    fdiv    s5, s7, s5              // inv_std

    // ================================================================
    // Phase 3: Normalize, scale, shift
    // out[i] = gamma[i] * (temp[i] - mean) * inv_std + beta[i]
    // ================================================================

    mov     z30.s, s4               // broadcast mean
    mov     z31.s, s5               // broadcast inv_std

    mov     x8, #0
    whilelt pn9.s, x8, x20, vlx4

.Lfdn_norm_loop:
    // Load temp
    ld1w    {z0.s-z3.s}, pn9/z, [x27, x8, lsl #2]

    // Subtract mean
    fsub    z0.s, p0/m, z0.s, z30.s
    fsub    z1.s, p0/m, z1.s, z30.s
    fsub    z2.s, p0/m, z2.s, z30.s
    fsub    z3.s, p0/m, z3.s, z30.s

    // Multiply by inv_std
    fmul    z0.s, p0/m, z0.s, z31.s
    fmul    z1.s, p0/m, z1.s, z31.s
    fmul    z2.s, p0/m, z2.s, z31.s
    fmul    z3.s, p0/m, z3.s, z31.s

    // Load gamma, multiply
    ld1w    {z4.s-z7.s}, pn9/z, [x23, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z4.s
    fmul    z1.s, p0/m, z1.s, z5.s
    fmul    z2.s, p0/m, z2.s, z6.s
    fmul    z3.s, p0/m, z3.s, z7.s

    // Load beta, add
    ld1w    {z4.s-z7.s}, pn9/z, [x24, x8, lsl #2]
    fadd    z0.s, p0/m, z0.s, z4.s
    fadd    z1.s, p0/m, z1.s, z5.s
    fadd    z2.s, p0/m, z2.s, z6.s
    fadd    z3.s, p0/m, z3.s, z7.s

    // Store output
    st1w    {z0.s-z3.s}, pn9, [x25, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x20, vlx4
    b.first .Lfdn_norm_loop

    smstop

    // Free temp buffer
    mov     x0, x27
    bl      _free

.Lfdn_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     d8,  d9,  [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    ldp     x29, x30, [sp], #160
    ret
