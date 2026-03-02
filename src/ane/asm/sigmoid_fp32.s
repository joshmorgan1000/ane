// sigmoid_fp32.s
// Single-threaded sigmoid using FMLA VGx4 with ZA accumulation
// Computes σ(x) = 1/(1+exp(-x))
//
// Signature: void sigmoid_fp32(const float *input, float *output, long n)
// AAPCS: x0=input, x1=output, x2=n

.section __TEXT,__text,regular,pure_instructions
.global _sigmoid_fp32
.p2align 4

_sigmoid_fp32:
    // Prologue: save callee-saved registers
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8,  d9,  [sp, #32]
    stp     d10, d11, [sp, #48]
    stp     d12, d13, [sp, #64]
    stp     d14,  d15,  [sp, #80]

    // Early exit on n=0
    cbz     x2, .Lsig_done

    // Save input/output pointers to callee-saved registers
    mov     x19, x0                 // input
    mov     x20, x1                 // output

    // Enable streaming SVE mode + ZA access
    smstart

    ptrue   p0.s

    // Load constants
    adr     x9, .Lsig_const
    ld1rw   {z16.s}, p0/z, [x9]        // log2(e)
    ld1rw   {z17.s}, p0/z, [x9, #4]    // ln(2)
    ld1rw   {z8.s}, p0/z, [x9, #8]     // 1.0
    ld1rw   {z9.s}, p0/z, [x9, #12]    // 0.5
    ld1rw   {z10.s}, p0/z, [x9, #16]   // 1/6
    ld1rw   {z11.s}, p0/z, [x9, #20]   // 1/24
    ld1rw   {z12.s}, p0/z, [x9, #24]   // 1/120
    ld1rw   {z13.s}, p0/z, [x9, #28]   // 1/720
    ld1rw   {z14.s}, p0/z, [x9, #32]   // 1/5040
    ld1rw   {z15.s}, p0/z, [x9, #36]   // clamp_hi (88.0)
    ld1rw   {z26.s}, p0/z, [x9, #40]   // clamp_lo (-88.0)

    // ZA vector select register
    mov     w11, #0

    // Main loop setup
    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lsig_loop:
    // Load input
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Negate: -x
    fneg    z0.s, p0/m, z0.s
    fneg    z1.s, p0/m, z1.s
    fneg    z2.s, p0/m, z2.s
    fneg    z3.s, p0/m, z3.s

    // Clamp -x to [-88, 88]
    fmin    z0.s, p0/m, z0.s, z15.s
    fmax    z0.s, p0/m, z0.s, z26.s
    fmin    z1.s, p0/m, z1.s, z15.s
    fmax    z1.s, p0/m, z1.s, z26.s
    fmin    z2.s, p0/m, z2.s, z15.s
    fmax    z2.s, p0/m, z2.s, z26.s
    fmin    z3.s, p0/m, z3.s, z15.s
    fmax    z3.s, p0/m, z3.s, z26.s

    // Range reduction: n = round(-x * log2(e))
    fmul    z28.s, z0.s, z16.s
    fmul    z29.s, z1.s, z16.s
    fmul    z30.s, z2.s, z16.s
    fmul    z31.s, z3.s, z16.s

    frintn  z28.s, p0/m, z28.s
    frintn  z29.s, p0/m, z29.s
    frintn  z30.s, p0/m, z30.s
    frintn  z31.s, p0/m, z31.s

    // r = -x - n*ln(2) (using fmls for fused multiply-subtract)
    mov     z4.d, z0.d
    mov     z5.d, z1.d
    mov     z6.d, z2.d
    mov     z7.d, z3.d

    fmls    z4.s, p0/m, z28.s, z17.s
    fmls    z5.s, p0/m, z29.s, z17.s
    fmls    z6.s, p0/m, z30.s, z17.s
    fmls    z7.s, p0/m, z31.s, z17.s

    // Power series polynomial via FMLA VGx4
    zero    {za}

    // c1*r (coefficient is 1.0 = z8)
    fmla    za.s[w11, 0, vgx4], {z4.s-z7.s}, z8.s

    // r²
    fmul    z0.s, z4.s, z4.s
    fmul    z1.s, z5.s, z5.s
    fmul    z2.s, z6.s, z6.s
    fmul    z3.s, z7.s, z7.s

    // c2*r² (c2 = 0.5 = z9)
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z9.s

    // r³
    fmul    z18.s, z0.s, z4.s
    fmul    z19.s, z1.s, z5.s
    fmul    z20.s, z2.s, z6.s
    fmul    z21.s, z3.s, z7.s

    // c3*r³ (c3 = 1/6 = z10)
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z10.s

    // r⁴
    fmul    z22.s, z0.s, z0.s
    fmul    z23.s, z1.s, z1.s
    fmul    z24.s, z2.s, z2.s
    fmul    z25.s, z3.s, z3.s

    // c4*r⁴ (c4 = 1/24 = z11)
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z11.s

    // r⁵
    fmul    z18.s, z22.s, z4.s
    fmul    z19.s, z23.s, z5.s
    fmul    z20.s, z24.s, z6.s
    fmul    z21.s, z25.s, z7.s

    // c5*r⁵ (c5 = 1/120 = z12)
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z12.s

    // r⁶
    fmul    z22.s, z22.s, z0.s
    fmul    z23.s, z23.s, z1.s
    fmul    z24.s, z24.s, z2.s
    fmul    z25.s, z25.s, z3.s

    // c6*r⁶ (c6 = 1/720 = z13)
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z13.s

    // r⁷
    fmul    z18.s, z22.s, z4.s
    fmul    z19.s, z23.s, z5.s
    fmul    z20.s, z24.s, z6.s
    fmul    z21.s, z25.s, z7.s

    // c7*r⁷ (c7 = 1/5040 = z14)
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z14.s

    // Extract accumulated polynomial result
    mova    {z0.s-z3.s}, za.s[w11, 0, vgx4]

    // Add 1.0 to get 1 + r + r²/2 + ... (the exp(r) polynomial)
    fadd    z0.s, z0.s, z8.s
    fadd    z1.s, z1.s, z8.s
    fadd    z2.s, z2.s, z8.s
    fadd    z3.s, z3.s, z8.s

    // Convert n to integer BEFORE fscale (fscale expects integer exponent)
    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s

    // Reconstruct: exp(r) * 2^n
    fscale  z0.s, p0/m, z0.s, z28.s
    fscale  z1.s, p0/m, z1.s, z29.s
    fscale  z2.s, p0/m, z2.s, z30.s
    fscale  z3.s, p0/m, z3.s, z31.s

    // Now z0-z3 contain exp(-x)
    // Compute 1 + exp(-x)
    fadd    z0.s, z0.s, z8.s
    fadd    z1.s, z1.s, z8.s
    fadd    z2.s, z2.s, z8.s
    fadd    z3.s, z3.s, z8.s

    // Compute 1.0 / (1 + exp(-x)) = sigmoid
    movprfx z4, z8
    fdiv    z4.s, p0/m, z4.s, z0.s
    movprfx z5, z8
    fdiv    z5.s, p0/m, z5.s, z1.s
    movprfx z6, z8
    fdiv    z6.s, p0/m, z6.s, z2.s
    movprfx z7, z8
    fdiv    z7.s, p0/m, z7.s, z3.s

    // Store result
    st1w    {z4.s-z7.s}, pn9, [x20, x8, lsl #2]

    // Increment and loop
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x2, vlx4
    b.first .Lsig_loop

    smstop

.Lsig_done:
    // Epilogue: restore callee-saved registers
    ldp     d12, d13, [sp, #64]
    ldp     d10, d11, [sp, #48]
    ldp     d8,  d9,  [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     d14,  d15,  [sp, #80]
    ldp     x29, x30, [sp], #96
    ret

// Constant pool
.p2align 2
.Lsig_const:
    .long   0x3FB8AA3B  // log2(e)
    .long   0x3F317218  // ln(2)
    .long   0x3F800000  // 1.0
    .long   0x3F000000  // 0.5
    .long   0x3E2AAAAB  // 1/6
    .long   0x3D2AAAAB  // 1/24
    .long   0x3C088889  // 1/120
    .long   0x3AB60B61  // 1/720
    .long   0x39500D01  // 1/5040
    .long   0x42B00000  // 88.0
    .long   0xC2B00000  // -88.0
