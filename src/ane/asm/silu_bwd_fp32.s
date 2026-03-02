// silu_bwd_fp32.s — SiLU (Swish) backward pass via SME2 with FMLA VGx4 ZA accumulation
//
// Computes: dx[i] = dy[i] * (sigma(x[i]) + x[i] * sigma(x[i]) * (1 - sigma(x[i])))
// where sigma(x) = 1 / (1 + exp(-x))
//
// Equivalently: dx = dy * sigma * (1 + x * (1 - sigma))
//             = dy * sigma * (1 + x - x*sigma)
//
// void silu_bwd_fp32(const float *x, const float *dy, float *dx, long n)
// AAPCS: x0=x (input), x1=dy (upstream gradient), x2=dx (output gradient), x3=n
//
// Strategy:
//   1. Compute sigma = sigmoid(x) via exp(-x) power series + ZA accumulation
//   2. Compute (1 - sigma) via subtraction
//   3. Compute x * (1 - sigma) via multiply
//   4. Add 1 to get (1 + x*(1 - sigma))
//   5. Multiply by sigma to get sigma * (1 + x*(1 - sigma))
//   6. Multiply by dy to get final gradient
//
// Keeps x in z4-z7 across the sigmoid computation.
// dy loaded at end into z24-z27 (multiple-of-4 aligned for vlx4).
//
// Processes 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _silu_bwd_fp32
.p2align 4

_silu_bwd_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    stp     d14, d15, [sp, #96]

    // Early exit on n=0
    cbz     x3, .Lsb_done

    // Save pointers to callee-saved registers
    mov     x19, x0                 // x (input)
    mov     x20, x1                 // dy (upstream gradient)
    mov     x21, x2                 // dx (output gradient)

    // Enable streaming SVE mode + ZA access
    smstart

    ptrue   p0.s

    // Load constants (all in z0-z15 range for fmla broadcast compatibility)
    adr     x9, .Lsb_const
    ld1rw   {z16.s}, p0/z, [x9]        // log2(e)
    ld1rw   {z17.s}, p0/z, [x9, #4]    // ln(2)
    ld1rw   {z8.s}, p0/z, [x9, #8]     // 1.0
    ld1rw   {z9.s}, p0/z, [x9, #12]    // 0.5 = 1/2!
    ld1rw   {z10.s}, p0/z, [x9, #16]   // 1/6 = 1/3!
    ld1rw   {z11.s}, p0/z, [x9, #20]   // 1/24 = 1/4!
    ld1rw   {z12.s}, p0/z, [x9, #24]   // 1/120 = 1/5!
    ld1rw   {z13.s}, p0/z, [x9, #28]   // 1/720 = 1/6!
    ld1rw   {z14.s}, p0/z, [x9, #32]   // 1/5040 = 1/7!
    ld1rw   {z15.s}, p0/z, [x9, #36]   // 88.0 (clamp_hi)
    ld1rw   {z26.s}, p0/z, [x9, #40]   // -88.0 (clamp_lo)

    // ZA vector select register
    mov     w11, #0

    // Main loop setup
    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lsb_loop:
    // Load input x into z0-z3
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Save original x into z4-z7 for later use in gradient formula
    mov     z4.d, z0.d
    mov     z5.d, z1.d
    mov     z6.d, z2.d
    mov     z7.d, z3.d

    // Negate: -x for exp(-x)
    fneg    z0.s, p0/m, z0.s
    fneg    z1.s, p0/m, z1.s
    fneg    z2.s, p0/m, z2.s
    fneg    z3.s, p0/m, z3.s

    // Clamp -x to [-88, 88] to prevent overflow
    fclamp  {z0.s-z3.s}, z26.s, z15.s

    // Range reduction: n_val = round(-x * log2(e))
    fmul    z28.s, z0.s, z16.s
    fmul    z29.s, z1.s, z16.s
    fmul    z30.s, z2.s, z16.s
    fmul    z31.s, z3.s, z16.s
    frintn  z28.s, p0/m, z28.s
    frintn  z29.s, p0/m, z29.s
    frintn  z30.s, p0/m, z30.s
    frintn  z31.s, p0/m, z31.s

    // r = -x - n_val * ln(2) -> z0-z3
    fmls    z0.s, p0/m, z28.s, z17.s
    fmls    z1.s, p0/m, z29.s, z17.s
    fmls    z2.s, p0/m, z30.s, z17.s
    fmls    z3.s, p0/m, z31.s, z17.s

    // --- Polynomial exp(r) via FMLA VGx4 with ZA accumulation ---
    zero    {za}

    // Term c1*r (c1 = 1.0 = z8)
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z8.s

    // r^2 -> z18-z21
    fmul    z18.s, z0.s, z0.s
    fmul    z19.s, z1.s, z1.s
    fmul    z20.s, z2.s, z2.s
    fmul    z21.s, z3.s, z3.s
    // Term c2*r^2 (c2 = 0.5)
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z9.s

    // r^3 = r^2 * r -> z22-z25
    fmul    z22.s, z18.s, z0.s
    fmul    z23.s, z19.s, z1.s
    fmul    z24.s, z20.s, z2.s
    fmul    z25.s, z21.s, z3.s
    // Term c3*r^3 (c3 = 1/6)
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z10.s

    // r^4 = r^2 * r^2 -> z22-z25
    fmul    z22.s, z18.s, z18.s
    fmul    z23.s, z19.s, z19.s
    fmul    z24.s, z20.s, z20.s
    fmul    z25.s, z21.s, z21.s
    // Term c4*r^4 (c4 = 1/24)
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z11.s

    // r^6 = r^4 * r^2 -> z18-z21 (clobbers r^2)
    fmul    z18.s, z22.s, z18.s
    fmul    z19.s, z23.s, z19.s
    fmul    z20.s, z24.s, z20.s
    fmul    z21.s, z25.s, z21.s

    // r^5 = r^4 * r -> z22-z25 (clobbers r^4)
    fmul    z22.s, z22.s, z0.s
    fmul    z23.s, z23.s, z1.s
    fmul    z24.s, z24.s, z2.s
    fmul    z25.s, z25.s, z3.s
    // Term c5*r^5 (c5 = 1/120)
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z12.s

    // Term c6*r^6 (c6 = 1/720) - r^6 in z18-z21
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z13.s

    // r^7 = r^6 * r -> z18-z21
    fmul    z18.s, z18.s, z0.s
    fmul    z19.s, z19.s, z1.s
    fmul    z20.s, z20.s, z2.s
    fmul    z21.s, z21.s, z3.s
    // Term c7*r^7 (c7 = 1/5040 = z14)
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z14.s

    // Extract accumulated polynomial result and add c0 = 1.0
    mova    {z0.s-z3.s}, za.s[w11, 0, vgx4]
    fadd    z0.s, z0.s, z8.s
    fadd    z1.s, z1.s, z8.s
    fadd    z2.s, z2.s, z8.s
    fadd    z3.s, z3.s, z8.s

    // Scale by 2^n_val to get exp(-x)
    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s
    fscale  z0.s, p0/m, z0.s, z28.s
    fscale  z1.s, p0/m, z1.s, z29.s
    fscale  z2.s, p0/m, z2.s, z30.s
    fscale  z3.s, p0/m, z3.s, z31.s

    // Now z0-z3 = exp(-x)
    // Compute 1 + exp(-x)
    fadd    z0.s, z0.s, z8.s
    fadd    z1.s, z1.s, z8.s
    fadd    z2.s, z2.s, z8.s
    fadd    z3.s, z3.s, z8.s

    // Compute sigma = 1 / (1 + exp(-x))
    // z18-z21 = sigma
    movprfx z18, z8
    fdiv    z18.s, p0/m, z18.s, z0.s
    movprfx z19, z8
    fdiv    z19.s, p0/m, z19.s, z1.s
    movprfx z20, z8
    fdiv    z20.s, p0/m, z20.s, z2.s
    movprfx z21, z8
    fdiv    z21.s, p0/m, z21.s, z3.s

    // Compute (1 - sigma) -> z0-z3
    movprfx z0, z8
    fsub    z0.s, p0/m, z0.s, z18.s
    movprfx z1, z8
    fsub    z1.s, p0/m, z1.s, z19.s
    movprfx z2, z8
    fsub    z2.s, p0/m, z2.s, z20.s
    movprfx z3, z8
    fsub    z3.s, p0/m, z3.s, z21.s

    // Compute x * (1 - sigma) -> z0-z3
    // z4-z7 = original x
    fmul    z0.s, p0/m, z0.s, z4.s
    fmul    z1.s, p0/m, z1.s, z5.s
    fmul    z2.s, p0/m, z2.s, z6.s
    fmul    z3.s, p0/m, z3.s, z7.s

    // Compute 1 + x*(1 - sigma) -> z0-z3
    fadd    z0.s, z0.s, z8.s
    fadd    z1.s, z1.s, z8.s
    fadd    z2.s, z2.s, z8.s
    fadd    z3.s, z3.s, z8.s

    // Compute sigma * (1 + x*(1 - sigma)) -> z0-z3
    fmul    z0.s, p0/m, z0.s, z18.s
    fmul    z1.s, p0/m, z1.s, z19.s
    fmul    z2.s, p0/m, z2.s, z20.s
    fmul    z3.s, p0/m, z3.s, z21.s

    // Load dy into z4-z7 (x is no longer needed; z4 is multiple-of-4 for vlx4)
    // and multiply: dx = dy * sigma * (1 + x*(1 - sigma))
    ld1w    {z4.s-z7.s}, pn9/z, [x20, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z4.s
    fmul    z1.s, p0/m, z1.s, z5.s
    fmul    z2.s, p0/m, z2.s, z6.s
    fmul    z3.s, p0/m, z3.s, z7.s

    // Store result
    st1w    {z0.s-z3.s}, pn9, [x21, x8, lsl #2]

    // Increment and loop
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lsb_loop

    smstop

.Lsb_done:
    // Epilogue: restore callee-saved registers
    ldp     d14, d15, [sp, #96]
    ldp     d12, d13, [sp, #80]
    ldp     d10, d11, [sp, #64]
    ldp     d8,  d9,  [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #112
    ret

// Constant pool
.p2align 2
.Lsb_const:
    .long   0x3FB8AA3B  // log2(e) = 1.44269504
    .long   0x3F317218  // ln(2)   = 0.69314718
    .long   0x3F800000  // 1.0
    .long   0x3F000000  // 0.5     = 1/2!
    .long   0x3E2AAAAB  // 1/6     = 1/3!
    .long   0x3D2AAAAB  // 1/24    = 1/4!
    .long   0x3C088889  // 1/120   = 1/5!
    .long   0x3AB60B61  // 1/720   = 1/6!
    .long   0x39500D01  // 1/5040  = 1/7!
    .long   0x42B00000  // 88.0
    .long   0xC2B00000  // -88.0
