// gelu_backward_fp32.s — GELU backward pass via SME2 streaming SVE with exp polynomial
//
// void gelu_backward_fp32(const float* dy, const float* x, float* dx, long n)
// AAPCS: x0=dy (upstream gradient), x1=x (forward input), x2=dx (output gradient), x3=n
//
// Computes: GELU'(x) ≈ 0.5 * (1 + erf(x/√2)) + x * phi(x) * sqrt(2/π)
//           where phi(x) = exp(-x^2/2) / sqrt(2π)
//
// Using approximation: GELU(x) ≈ 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x^3)))
// Derivative: GELU'(x) = 0.5 + [components of tanh derivative]
//
// Simplified approach: use exp(-x^2/2) polynomial to approximate the Gaussian part.
//
// Processing: 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _gelu_backward_fp32
.p2align 4

_gelu_backward_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8,  d9,  [sp, #32]
    stp     d10, d11, [sp, #48]
    stp     d12, d13, [sp, #64]
    stp     d14, d15, [sp, #80]

    cbz     x3, .Lgb_done

    mov     x19, x0                 // dy
    mov     x20, x1                 // x

    smstart

    ptrue   p0.s
    mov     w11, #0

    // Load constants
    adr     x9, .Lgb_const
    ld1rw   {z8.s}, p0/z, [x9]      // 0.5
    ld1rw   {z9.s}, p0/z, [x9, #4]  // 1.0
    ld1rw   {z10.s}, p0/z, [x9, #8] // sqrt(2/pi) ≈ 0.797885
    ld1rw   {z11.s}, p0/z, [x9, #12] // -0.5 (for -x^2/2)
    ld1rw   {z12.s}, p0/z, [x9, #16] // exp polynomial c1 = 1.0
    ld1rw   {z13.s}, p0/z, [x9, #20] // exp polynomial c2 = 0.5
    ld1rw   {z14.s}, p0/z, [x9, #24] // exp polynomial c3 = 1/6
    ld1rw   {z15.s}, p0/z, [x9, #28] // exp polynomial c4 = 1/24
    ld1rw   {z26.s}, p0/z, [x9, #32] // 0.044715 (coefficient for cubic term in tanh approx)

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lgb_loop:
    // Load x and dy
    ld1w    {z0.s-z3.s}, pn9/z, [x20, x8, lsl #2]  // x
    ld1w    {z4.s-z7.s}, pn9/z, [x19, x8, lsl #2]  // dy

    // Save original x for later use
    mov     z16.d, z0.d
    mov     z17.d, z1.d
    mov     z18.d, z2.d
    mov     z19.d, z3.d

    // Compute x^3 for tanh approximation: x^3 = x * x * x
    fmul    z20.s, z0.s, z0.s      // x^2
    fmul    z21.s, z1.s, z1.s
    fmul    z22.s, z2.s, z2.s
    fmul    z23.s, z3.s, z3.s

    fmul    z20.s, p0/m, z20.s, z0.s  // x^3
    fmul    z21.s, p0/m, z21.s, z1.s
    fmul    z22.s, p0/m, z22.s, z2.s
    fmul    z23.s, p0/m, z23.s, z3.s

    // Compute 0.044715*x^3
    fmul    z20.s, p0/m, z20.s, z26.s
    fmul    z21.s, p0/m, z21.s, z26.s
    fmul    z22.s, p0/m, z22.s, z26.s
    fmul    z23.s, p0/m, z23.s, z26.s

    // Compute x + 0.044715*x^3
    fadd    z20.s, p0/m, z20.s, z16.s
    fadd    z21.s, p0/m, z21.s, z17.s
    fadd    z22.s, p0/m, z22.s, z18.s
    fadd    z23.s, p0/m, z23.s, z19.s

    // Compute sqrt(2/pi) * (x + 0.044715*x^3)
    fmul    z20.s, p0/m, z20.s, z10.s
    fmul    z21.s, p0/m, z21.s, z10.s
    fmul    z22.s, p0/m, z22.s, z10.s
    fmul    z23.s, p0/m, z23.s, z10.s

    // Now use fast tanh approximation via sign(x) - x (simplified).
    // For GELU gradient, we use: 0.5 * (1 + tanh(...))
    // tanh(x) ≈ sign(x) for |x| > large threshold, else polynomial.
    // Simplified: compute sigmoid of 2*arg directly via exp
    // Result: GELU'(x) ≈ 0.5 + 0.5*tanh(arg) ≈ 0.5 + 0.5*sigmoid(2*arg) - 0.5
    //        = 0.5*sigmoid(2*arg)

    // Actually, simpler approach: GELU'(x) ≈ 0.5*(1 + x*phi(x)*c) for Gaussian approximation
    // where phi(x) ≈ exp(-x^2/2) polynomial
    // Compute -x^2/2 for Gaussian PDF
    fmul    z20.s, z16.s, z16.s    // x^2
    fmul    z21.s, z17.s, z17.s
    fmul    z22.s, z18.s, z18.s
    fmul    z23.s, z19.s, z19.s

    fmul    z20.s, p0/m, z20.s, z11.s  // -x^2/2
    fmul    z21.s, p0/m, z21.s, z11.s
    fmul    z22.s, p0/m, z22.s, z11.s
    fmul    z23.s, p0/m, z23.s, z11.s

    // Compute exp(-x^2/2) via polynomial (simplified 3-term version for speed)
    // exp(r) ≈ 1 + r + r^2/2 + r^3/6
    zero    {za}

    // Term c1*r
    fmla    za.s[w11, 0, vgx4], {z20.s, z21.s, z22.s, z23.s}, z12.s

    // r^2
    fmul    z0.s, z20.s, z20.s
    fmul    z1.s, z21.s, z21.s
    fmul    z2.s, z22.s, z22.s
    fmul    z3.s, z23.s, z23.s
    // Term c2*r^2
    fmla    za.s[w11, 0, vgx4], {z0.s, z1.s, z2.s, z3.s}, z13.s

    // r^3
    fmul    z0.s, p0/m, z0.s, z20.s
    fmul    z1.s, p0/m, z1.s, z21.s
    fmul    z2.s, p0/m, z2.s, z22.s
    fmul    z3.s, p0/m, z3.s, z23.s
    // Term c3*r^3
    fmla    za.s[w11, 0, vgx4], {z0.s, z1.s, z2.s, z3.s}, z14.s

    // Extract result
    mova    {z0.s, z1.s, z2.s, z3.s}, za.s[w11, 0, vgx4]

    // Add c0 = 1.0 to get exp(-x^2/2)
    fadd    z0.s, z0.s, z9.s
    fadd    z1.s, z1.s, z9.s
    fadd    z2.s, z2.s, z9.s
    fadd    z3.s, z3.s, z9.s

    // Compute final gradient: dx = dy * 0.5 * (1 + x * exp(-x^2/2) * sqrt(2/pi))
    // = dy * 0.5 * (1 + x * phi(x))
    fmul    z0.s, p0/m, z0.s, z16.s    // x * exp(-x^2/2)
    fmul    z1.s, p0/m, z1.s, z17.s
    fmul    z2.s, p0/m, z2.s, z18.s
    fmul    z3.s, p0/m, z3.s, z19.s

    fmul    z0.s, p0/m, z0.s, z10.s    // * sqrt(2/pi)
    fmul    z1.s, p0/m, z1.s, z10.s
    fmul    z2.s, p0/m, z2.s, z10.s
    fmul    z3.s, p0/m, z3.s, z10.s

    // 1 + x*phi(x)*sqrt(2/pi)
    fadd    z0.s, z0.s, z9.s
    fadd    z1.s, z1.s, z9.s
    fadd    z2.s, z2.s, z9.s
    fadd    z3.s, z3.s, z9.s

    // 0.5 * (1 + x*phi(x)*sqrt(2/pi))
    fmul    z0.s, p0/m, z0.s, z8.s
    fmul    z1.s, p0/m, z1.s, z8.s
    fmul    z2.s, p0/m, z2.s, z8.s
    fmul    z3.s, p0/m, z3.s, z8.s

    // Reload dy and multiply: dx = dy * GELU'(x)
    ld1w    {z4.s-z7.s}, pn9/z, [x19, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z4.s
    fmul    z1.s, p0/m, z1.s, z5.s
    fmul    z2.s, p0/m, z2.s, z6.s
    fmul    z3.s, p0/m, z3.s, z7.s

    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lgb_loop

    smstop

.Lgb_done:
    ldp     d14, d15, [sp, #80]
    ldp     d12, d13, [sp, #64]
    ldp     d10, d11, [sp, #48]
    ldp     d8,  d9,  [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #112
    ret

.p2align 2
.Lgb_const:
    .long   0x3F000000  // 0.5
    .long   0x3F800000  // 1.0
    .long   0x3F4C8E0C  // sqrt(2/pi) ≈ 0.797885
    .long   0xBF000000  // -0.5
    .long   0x3F800000  // 1.0 (exp polynomial c1)
    .long   0x3F000000  // 0.5 (exp polynomial c2)
    .long   0x3E2AAAAB  // 1/6 (exp polynomial c3)
    .long   0x3D2AAAAB  // 1/24 (exp polynomial c4)
    .long   0x3D368BC0  // 0.044715
