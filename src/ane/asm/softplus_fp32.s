// softplus_fp32.s — Element-wise Softplus activation via SME2 streaming SVE
//
// Computes: output[i] = log(1 + exp(x))
//
// For numerical stability, uses: softplus(x) = max(x, 0) + log(1 + exp(-|x|))
// This avoids overflow for large positive x.
//
// Strategy:
// 1. Load x
// 2. Compute u = exp(-|x|) using range-reduced polynomial (similar to silu_fp32.s)
// 3. Compute log(1 + u) using Taylor series for log1p
// 4. Result = max(x, 0) + log(1 + u)
//
// void softplus_fp32(const float *input, float *output, long n)
// AAPCS: x0=input, x1=output, x2=n
//
// Processes 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _softplus_fp32
.p2align 4

_softplus_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8,  d9,  [sp, #32]
    stp     d10, d11, [sp, #48]
    stp     d12, d13, [sp, #64]
    stp     d14, d15, [sp, #80]

    // Early exit on n=0
    cbz     x2, .Lsoftplus_done

    // Save pointers to callee-saved registers
    mov     x19, x0                 // input
    mov     x20, x1                 // output

    // Enable streaming SVE mode (no ZA needed)
    smstart sm

    ptrue   p0.s

    // Load constants
    adr     x9, .Lsoftplus_const
    ld1rw   {z16.s}, p0/z, [x9]        // log2(e)
    ld1rw   {z17.s}, p0/z, [x9, #4]    // ln(2)
    ld1rw   {z18.s}, p0/z, [x9, #8]    // 0.0
    ld1rw   {z19.s}, p0/z, [x9, #12]   // 1.0
    ld1rw   {z20.s}, p0/z, [x9, #16]   // 0.5 = 1/2!
    ld1rw   {z21.s}, p0/z, [x9, #20]   // 1/6 = 1/3!
    ld1rw   {z22.s}, p0/z, [x9, #24]   // 1/24 = 1/4!
    ld1rw   {z23.s}, p0/z, [x9, #28]   // 1/120 = 1/5!
    ld1rw   {z24.s}, p0/z, [x9, #32]   // 1/720 = 1/6!
    ld1rw   {z25.s}, p0/z, [x9, #36]   // 1/5040 = 1/7!
    ld1rw   {z26.s}, p0/z, [x9, #40]   // 88.0 (clamp_hi)
    ld1rw   {z27.s}, p0/z, [x9, #44]   // -88.0 (clamp_lo)
    // Log1p Taylor coefficients
    ld1rw   {z28.s}, p0/z, [x9, #48]   // 1.0 (log1p c1 in Horner)
    ld1rw   {z29.s}, p0/z, [x9, #52]   // -0.5 (log1p c2)
    ld1rw   {z30.s}, p0/z, [x9, #56]   // 1/3 (log1p c3)
    ld1rw   {z31.s}, p0/z, [x9, #60]   // -1/4 (log1p c4)

    // Main loop setup
    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lsoftplus_loop:
    // Load input x into z0-z3
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Save original x into z4-z7 for final max(x, 0)
    mov     z4.d, z0.d
    mov     z5.d, z1.d
    mov     z6.d, z2.d
    mov     z7.d, z3.d

    // Compute |x|
    fabs    z0.s, p0/m, z0.s
    fabs    z1.s, p0/m, z1.s
    fabs    z2.s, p0/m, z2.s
    fabs    z3.s, p0/m, z3.s

    // Negate: -|x| for exp(-|x|)
    fneg    z0.s, p0/m, z0.s
    fneg    z1.s, p0/m, z1.s
    fneg    z2.s, p0/m, z2.s
    fneg    z3.s, p0/m, z3.s

    // Clamp -|x| to [-88, 88] to prevent overflow
    fclamp  {z0.s-z3.s}, z27.s, z26.s

    // Range reduction: n_val = round(-|x| * log2(e))
    fmul    z8.s, z0.s, z16.s
    fmul    z9.s, z1.s, z16.s
    fmul    z10.s, z2.s, z16.s
    fmul    z11.s, z3.s, z16.s
    frintn  z8.s, p0/m, z8.s
    frintn  z9.s, p0/m, z9.s
    frintn  z10.s, p0/m, z10.s
    frintn  z11.s, p0/m, z11.s

    // r = -|x| - n_val * ln(2)  ->  z0-z3
    fmls    z0.s, p0/m, z8.s, z17.s
    fmls    z1.s, p0/m, z9.s, z17.s
    fmls    z2.s, p0/m, z10.s, z17.s
    fmls    z3.s, p0/m, z11.s, z17.s

    // --- Polynomial exp(r) via nested multiplication ---
    // exp(r) ≈ 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5! + r⁶/6! + r⁷/7!
    // Horner: p = c7; p = p*r + c6; p = p*r + c5; ... p = p*r + 1

    // Compute powers of r
    fmul    z12.s, z0.s, z0.s      // r²
    fmul    z13.s, z1.s, z1.s
    fmul    z14.s, z2.s, z2.s
    fmul    z15.s, z3.s, z3.s

    // Accumulator starts at c7 = 1/5040
    mov     z8.d, z25.d             // p = c7
    mov     z9.d, z25.d
    mov     z10.d, z25.d
    mov     z11.d, z25.d

    // p = p*r + c6
    fmad    z8.s, p0/m, z0.s, z24.s
    fmad    z9.s, p0/m, z1.s, z24.s
    fmad    z10.s, p0/m, z2.s, z24.s
    fmad    z11.s, p0/m, z3.s, z24.s

    // p = p*r + c5
    fmad    z8.s, p0/m, z0.s, z23.s
    fmad    z9.s, p0/m, z1.s, z23.s
    fmad    z10.s, p0/m, z2.s, z23.s
    fmad    z11.s, p0/m, z3.s, z23.s

    // p = p*r + c4
    fmad    z8.s, p0/m, z0.s, z22.s
    fmad    z9.s, p0/m, z1.s, z22.s
    fmad    z10.s, p0/m, z2.s, z22.s
    fmad    z11.s, p0/m, z3.s, z22.s

    // p = p*r + c3
    fmad    z8.s, p0/m, z0.s, z21.s
    fmad    z9.s, p0/m, z1.s, z21.s
    fmad    z10.s, p0/m, z2.s, z21.s
    fmad    z11.s, p0/m, z3.s, z21.s

    // p = p*r + c2
    fmad    z8.s, p0/m, z0.s, z20.s
    fmad    z9.s, p0/m, z1.s, z20.s
    fmad    z10.s, p0/m, z2.s, z20.s
    fmad    z11.s, p0/m, z3.s, z20.s

    // p = p*r + c1
    fmad    z8.s, p0/m, z0.s, z19.s
    fmad    z9.s, p0/m, z1.s, z19.s
    fmad    z10.s, p0/m, z2.s, z19.s
    fmad    z11.s, p0/m, z3.s, z19.s

    // p = p*r + 1.0 (c0)
    fmad    z8.s, p0/m, z0.s, z19.s
    fmad    z9.s, p0/m, z1.s, z19.s
    fmad    z10.s, p0/m, z2.s, z19.s
    fmad    z11.s, p0/m, z3.s, z19.s

    // Scale by 2^n to get exp(-|x|)
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s

    // Wait, we overwrote the exponent values. Let me fix: save exponents first
    // Restart from range reduction

    // Load input x
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Save original x
    mov     z4.d, z0.d
    mov     z5.d, z1.d
    mov     z6.d, z2.d
    mov     z7.d, z3.d

    // Abs and negate
    fabs    z0.s, p0/m, z0.s
    fabs    z1.s, p0/m, z1.s
    fabs    z2.s, p0/m, z2.s
    fabs    z3.s, p0/m, z3.s
    fneg    z0.s, p0/m, z0.s
    fneg    z1.s, p0/m, z1.s
    fneg    z2.s, p0/m, z2.s
    fneg    z3.s, p0/m, z3.s

    // Clamp
    fclamp  {z0.s-z3.s}, z27.s, z26.s

    // Range reduce and save exponents in temp regs
    fmul    z12.s, z0.s, z16.s
    fmul    z13.s, z1.s, z16.s
    fmul    z14.s, z2.s, z16.s
    fmul    z15.s, z3.s, z16.s
    frintn  z12.s, p0/m, z12.s
    frintn  z13.s, p0/m, z13.s
    frintn  z14.s, p0/m, z14.s
    frintn  z15.s, p0/m, z15.s

    // r = x - n * ln(2)
    fmls    z0.s, p0/m, z12.s, z17.s
    fmls    z1.s, p0/m, z13.s, z17.s
    fmls    z2.s, p0/m, z14.s, z17.s
    fmls    z3.s, p0/m, z15.s, z17.s

    // Polynomial expansion
    mov     z8.d, z25.d
    fmad    z8.s, p0/m, z0.s, z24.s
    fmad    z8.s, p0/m, z0.s, z23.s
    fmad    z8.s, p0/m, z0.s, z22.s
    fmad    z8.s, p0/m, z0.s, z21.s
    fmad    z8.s, p0/m, z0.s, z20.s
    fmad    z8.s, p0/m, z0.s, z19.s
    fmad    z8.s, p0/m, z0.s, z19.s

    mov     z9.d, z25.d
    fmad    z9.s, p0/m, z1.s, z24.s
    fmad    z9.s, p0/m, z1.s, z23.s
    fmad    z9.s, p0/m, z1.s, z22.s
    fmad    z9.s, p0/m, z1.s, z21.s
    fmad    z9.s, p0/m, z1.s, z20.s
    fmad    z9.s, p0/m, z1.s, z19.s
    fmad    z9.s, p0/m, z1.s, z19.s

    mov     z10.d, z25.d
    fmad    z10.s, p0/m, z2.s, z24.s
    fmad    z10.s, p0/m, z2.s, z23.s
    fmad    z10.s, p0/m, z2.s, z22.s
    fmad    z10.s, p0/m, z2.s, z21.s
    fmad    z10.s, p0/m, z2.s, z20.s
    fmad    z10.s, p0/m, z2.s, z19.s
    fmad    z10.s, p0/m, z2.s, z19.s

    mov     z11.d, z25.d
    fmad    z11.s, p0/m, z3.s, z24.s
    fmad    z11.s, p0/m, z3.s, z23.s
    fmad    z11.s, p0/m, z3.s, z22.s
    fmad    z11.s, p0/m, z3.s, z21.s
    fmad    z11.s, p0/m, z3.s, z20.s
    fmad    z11.s, p0/m, z3.s, z19.s
    fmad    z11.s, p0/m, z3.s, z19.s

    // Scale by 2^n
    fcvtzs  z12.s, p0/m, z12.s
    fcvtzs  z13.s, p0/m, z13.s
    fcvtzs  z14.s, p0/m, z14.s
    fcvtzs  z15.s, p0/m, z15.s
    fscale  z8.s, p0/m, z8.s, z12.s
    fscale  z9.s, p0/m, z9.s, z13.s
    fscale  z10.s, p0/m, z10.s, z14.s
    fscale  z11.s, p0/m, z11.s, z15.s

    // Now z8-z11 = exp(-|x|)
    // We need log(1 + exp(-|x|))
    // Use log1p approximation on z8-z11

    // Horner from inside out: p = c4; p = p*u + c3; p = p*u + c2; p = p*u + c1; result = p*u
    mov     z0.d, z31.d             // p = c4 = -0.25
    fmad    z0.s, p0/m, z8.s, z30.s // p = p*u + c3 (1/3)
    fmad    z0.s, p0/m, z8.s, z29.s // p = p*u + c2 (-0.5)
    fmad    z0.s, p0/m, z8.s, z28.s // p = p*u + c1 (1.0)
    fmul    z0.s, p0/m, z0.s, z8.s  // result = p*u

    mov     z1.d, z31.d
    fmad    z1.s, p0/m, z9.s, z30.s
    fmad    z1.s, p0/m, z9.s, z29.s
    fmad    z1.s, p0/m, z9.s, z28.s
    fmul    z1.s, p0/m, z1.s, z9.s

    mov     z2.d, z31.d
    fmad    z2.s, p0/m, z10.s, z30.s
    fmad    z2.s, p0/m, z10.s, z29.s
    fmad    z2.s, p0/m, z10.s, z28.s
    fmul    z2.s, p0/m, z2.s, z10.s

    mov     z3.d, z31.d
    fmad    z3.s, p0/m, z11.s, z30.s
    fmad    z3.s, p0/m, z11.s, z29.s
    fmad    z3.s, p0/m, z11.s, z28.s
    fmul    z3.s, p0/m, z3.s, z11.s

    // Compute max(x, 0) and add log1p result
    fmax    z4.s, p0/m, z4.s, z18.s  // max(x, 0) where z18 = 0.0
    fmax    z5.s, p0/m, z5.s, z18.s
    fmax    z6.s, p0/m, z6.s, z18.s
    fmax    z7.s, p0/m, z7.s, z18.s

    fadd    z4.s, p0/m, z4.s, z0.s  // result = max(x, 0) + log1p(exp(-|x|))
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s

    // Store result
    st1w    {z4.s-z7.s}, pn9, [x20, x8, lsl #2]

    // Increment and loop
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x2, vlx4
    b.first .Lsoftplus_loop

    smstop

.Lsoftplus_done:
    // Epilogue: restore callee-saved registers
    ldp     d14, d15, [sp, #80]
    ldp     d12, d13, [sp, #64]
    ldp     d10, d11, [sp, #48]
    ldp     d8,  d9,  [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret

// Constant pool
.p2align 2
.Lsoftplus_const:
    .long   0x3FB8AA3B  // log2(e) = 1.44269504
    .long   0x3F317218  // ln(2)   = 0.69314718
    .long   0x00000000  // 0.0
    .long   0x3F800000  // 1.0
    .long   0x3F000000  // 0.5     = 1/2!
    .long   0x3E2AAAAB  // 1/6     = 1/3!
    .long   0x3D2AAAAB  // 1/24    = 1/4!
    .long   0x3C088889  // 1/120   = 1/5!
    .long   0x3AB60B61  // 1/720   = 1/6!
    .long   0x39500D01  // 1/5040  = 1/7!
    .long   0x42B00000  // 88.0
    .long   0xC2B00000  // -88.0
    .long   0x3F800000  // 1.0 (log1p c1)
    .long   0xBF000000  // -0.5 (log1p c2)
    .long   0x3EAAAAAB  // 1/3 (log1p c3)
    .long   0xBE000000  // -0.25 (log1p c4)
