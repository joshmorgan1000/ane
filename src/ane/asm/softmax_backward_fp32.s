// softmax_backward_fp32.s — Softmax gradient via SME2 streaming SVE
//
// void softmax_backward_fp32(const float *dy, const float *y, float *dx, long n);
//
// AAPCS64: x0=dy (upstream gradient), x1=y (softmax output), x2=dx (output gradient), x3=n
//
// Computes softmax backward pass gradient:
//   dx[i] = y[i] * (dy[i] - dot(dy, y))
//
// where dot(dy, y) = sum_i(dy[i] * y[i])
//
// Algorithm:
//   1) Compute dot product: sum = sum_i(dy[i] * y[i]) using vlx4
//   2) Broadcast sum to vector register
//   3) For each element: dx[i] = y[i] * (dy[i] - sum)

.section __TEXT,__text,regular,pure_instructions
.global _softmax_backward_fp32
.p2align 4

_softmax_backward_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    stp     d14,  d15,  [sp, #96]

    // Early exit on n == 0
    cbz     x3, .Lsbk_done

    // Save arguments in callee-saved registers
    mov     x19, x0                 // dy
    mov     x20, x1                 // y
    mov     x21, x2                 // dx
    mov     x22, x3                 // n

    smstart sm

    ptrue   p0.s

    // ═════════════════════════════════════════════════════════════
    // Phase 1: Compute dot product: sum = sum_i(dy[i] * y[i])
    // ═════════════════════════════════════════════════════════════

    // Initialize 4 independent accumulators
    mov     z8.d, #0                // acc0
    mov     z9.d, #0                // acc1
    mov     z10.d, #0               // acc2
    mov     z11.d, #0               // acc3

    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

.Lsbk_dot_loop:
    // Load dy and y
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]  // dy
    ld1w    {z4.s-z7.s}, pn9/z, [x20, x8, lsl #2]  // y

    // Accumulate products: acc += dy * y
    fmla    z8.s, p0/m, z0.s, z4.s
    fmla    z9.s, p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lsbk_dot_loop

    // Tree-reduce 4 accumulators → 1 → scalar
    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s

    // Horizontal sum
    faddv   s8, p0, z8.s

    // Broadcast scalar sum to all lanes of z30
    mov     z30.s, s8

    // ═════════════════════════════════════════════════════════════
    // Phase 2: Compute dx[i] = y[i] * (dy[i] - sum)
    // ═════════════════════════════════════════════════════════════

    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

.Lsbk_gradient_loop:
    // Load dy and y
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]  // dy
    ld1w    {z4.s-z7.s}, pn9/z, [x20, x8, lsl #2]  // y

    // Compute dy - sum
    fsub    z0.s, p0/m, z0.s, z30.s
    fsub    z1.s, p0/m, z1.s, z30.s
    fsub    z2.s, p0/m, z2.s, z30.s
    fsub    z3.s, p0/m, z3.s, z30.s

    // Compute dx = y * (dy - sum)
    fmul    z0.s, p0/m, z0.s, z4.s
    fmul    z1.s, p0/m, z1.s, z5.s
    fmul    z2.s, p0/m, z2.s, z6.s
    fmul    z3.s, p0/m, z3.s, z7.s

    // Store dx
    st1w    {z0.s-z3.s}, pn9, [x21, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lsbk_gradient_loop

    smstop

.Lsbk_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     d8,  d9,  [sp, #48]
    ldp     d10, d11, [sp, #64]
    ldp     d12, d13, [sp, #80]
    ldp     d14,  d15,  [sp, #96]
    ldp     x29, x30, [sp], #112
    ret
