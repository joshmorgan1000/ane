// mish_fp32.s — Element-wise Mish activation via SME2 with ZA accumulation
//
// Computes: output[i] = x * tanh(softplus(x))
//                      = x * tanh(log(1 + exp(x)))
//
// Uses: tanh(softplus(x)) = (s²-1)/(s²+1) where s = 1+exp(x)
// Clamped to [-88, 20] to prevent s² float overflow.
//
// void mish_fp32(const float *input, float *output, long n)
// AAPCS: x0=input, x1=output, x2=n
//
// Processes 4 vectors (64 floats on M4) per iteration.
// Uses ZA accumulation for the final x * tanh multiplication.

.section __TEXT,__text,regular,pure_instructions
.global _mish_fp32
.p2align 4

_mish_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8,  d9,  [sp, #32]
    stp     d10, d11, [sp, #48]
    stp     d12, d13, [sp, #64]
    stp     d14, d15, [sp, #80]

    // Early exit on n=0
    cbz     x2, .Lmish_done

    // Save pointers
    mov     x19, x0                 // input
    mov     x20, x1                 // output

    // Enable streaming mode with ZA access
    smstart

    ptrue   p0.s

    // Load constants
    adr     x9, .Lmish_const
    ld1rw   {z16.s}, p0/z, [x9]        // log2(e)
    ld1rw   {z17.s}, p0/z, [x9, #4]    // ln(2)
    ld1rw   {z8.s}, p0/z, [x9, #8]     // 1.0
    ld1rw   {z9.s}, p0/z, [x9, #12]    // 0.5
    ld1rw   {z10.s}, p0/z, [x9, #16]   // 1/6
    ld1rw   {z11.s}, p0/z, [x9, #20]   // 1/24
    ld1rw   {z12.s}, p0/z, [x9, #24]   // 1/120
    ld1rw   {z13.s}, p0/z, [x9, #28]   // 1/720
    ld1rw   {z14.s}, p0/z, [x9, #32]   // 1/5040
    ld1rw   {z15.s}, p0/z, [x9, #36]   // 20.0 (upper clamp)
    ld1rw   {z26.s}, p0/z, [x9, #40]   // -88.0

    // ZA vector select register
    mov     w11, #0

    // Main loop setup
    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lmish_loop:
    // Load input x
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Save x for final multiply
    mov     z4.d, z0.d
    mov     z5.d, z1.d
    mov     z6.d, z2.d
    mov     z7.d, z3.d

    // Clamp to [-88, 88] to prevent overflow in exp(x)
    fclamp  {z0.s-z3.s}, z26.s, z15.s

    // Range reduction: n = round(x * log2(e))
    fmul    z28.s, z0.s, z16.s
    fmul    z29.s, z1.s, z16.s
    fmul    z30.s, z2.s, z16.s
    fmul    z31.s, z3.s, z16.s
    frintn  z28.s, p0/m, z28.s
    frintn  z29.s, p0/m, z29.s
    frintn  z30.s, p0/m, z30.s
    frintn  z31.s, p0/m, z31.s

    // r = x - n * ln(2)
    fmls    z0.s, p0/m, z28.s, z17.s
    fmls    z1.s, p0/m, z29.s, z17.s
    fmls    z2.s, p0/m, z30.s, z17.s
    fmls    z3.s, p0/m, z31.s, z17.s

    // --- Polynomial exp(r) via ZA accumulation ---
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

    // r^5 = r^4 * r
    fmul    z22.s, z22.s, z0.s
    fmul    z23.s, z23.s, z1.s
    fmul    z24.s, z24.s, z2.s
    fmul    z25.s, z25.s, z3.s
    // Term c5*r^5 (c5 = 1/120)
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z12.s

    // r^6 = r^4 * r^2 (reuse r^2 slots)
    fmul    z18.s, z22.s, z18.s
    fmul    z19.s, z23.s, z19.s
    fmul    z20.s, z24.s, z20.s
    fmul    z21.s, z25.s, z21.s
    // Term c6*r^6 (c6 = 1/720)
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z13.s

    // r^7 = r^6 * r
    fmul    z18.s, z18.s, z0.s
    fmul    z19.s, z19.s, z1.s
    fmul    z20.s, z20.s, z2.s
    fmul    z21.s, z21.s, z3.s
    // Term c7*r^7 (c7 = 1/5040)
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z14.s

    // Extract accumulated polynomial + 1.0
    mova    {z0.s-z3.s}, za.s[w11, 0, vgx4]
    fadd    z0.s, z0.s, z8.s
    fadd    z1.s, z1.s, z8.s
    fadd    z2.s, z2.s, z8.s
    fadd    z3.s, z3.s, z8.s

    // Scale by 2^n to get exp(x)
    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s
    fscale  z0.s, p0/m, z0.s, z28.s
    fscale  z1.s, p0/m, z1.s, z29.s
    fscale  z2.s, p0/m, z2.s, z30.s
    fscale  z3.s, p0/m, z3.s, z31.s

    // Now z0-z3 = exp(x)
    // Correct: tanh(softplus(x)) = (s²-1)/(s²+1) where s = 1 + exp(x)

    // s = 1 + exp(x) -> z18-z21
    fadd    z18.s, z0.s, z8.s
    fadd    z19.s, z1.s, z8.s
    fadd    z20.s, z2.s, z8.s
    fadd    z21.s, z3.s, z8.s

    // s² -> z22-z25
    fmul    z22.s, z18.s, z18.s
    fmul    z23.s, z19.s, z19.s
    fmul    z24.s, z20.s, z20.s
    fmul    z25.s, z21.s, z21.s

    // s²-1 -> z18-z21
    fsub    z18.s, z22.s, z8.s
    fsub    z19.s, z23.s, z8.s
    fsub    z20.s, z24.s, z8.s
    fsub    z21.s, z25.s, z8.s

    // s²+1 -> z22-z25
    fadd    z22.s, z22.s, z8.s
    fadd    z23.s, z23.s, z8.s
    fadd    z24.s, z24.s, z8.s
    fadd    z25.s, z25.s, z8.s

    // tanh(softplus(x)) = (s²-1)/(s²+1) -> z18-z21
    fdiv    z18.s, p0/m, z18.s, z22.s
    fdiv    z19.s, p0/m, z19.s, z23.s
    fdiv    z20.s, p0/m, z20.s, z24.s
    fdiv    z21.s, p0/m, z21.s, z25.s

    // x * tanh(softplus(x)) using element-wise multiply
    fmul    z0.s, z4.s, z18.s
    fmul    z1.s, z5.s, z19.s
    fmul    z2.s, z6.s, z20.s
    fmul    z3.s, z7.s, z21.s

    // Store result
    st1w    {z0.s-z3.s}, pn9, [x20, x8, lsl #2]

    // Increment and loop
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x2, vlx4
    b.first .Lmish_loop

    smstop

.Lmish_done:
    // Epilogue
    ldp     d14, d15, [sp, #80]
    ldp     d12, d13, [sp, #64]
    ldp     d10, d11, [sp, #48]
    ldp     d8,  d9,  [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret

// Constant pool
.p2align 2
.Lmish_const:
    .long   0x3FB8AA3B  // log2(e) = 1.44269504
    .long   0x3F317218  // ln(2)   = 0.69314718
    .long   0x3F800000  // 1.0
    .long   0x3F000000  // 0.5     = 1/2!
    .long   0x3E2AAAAB  // 1/6     = 1/3!
    .long   0x3D2AAAAB  // 1/24    = 1/4!
    .long   0x3C088889  // 1/120   = 1/5!
    .long   0x3AB60B61  // 1/720   = 1/6!
    .long   0x39500D01  // 1/5040  = 1/7!
    .long   0x41A00000  // 20.0 (upper clamp to prevent s² overflow)
    .long   0xC2B00000  // -88.0
