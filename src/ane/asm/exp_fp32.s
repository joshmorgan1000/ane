// exp_fp32.s — Element-wise e^x via SME2 with FMLA VGx4 ZA accumulation
//
// void exp_fp32(const float *input, float *output, long n)
//
// Range reduction: x = n*ln2 + r, exp(x) = 2^n * exp(r)
// Power series exp(r) evaluated via FMLA VGx4 into ZA tile rows.
// Each FMLA VGx4 replaces 4 scalar fmad instructions.
//
// Processes 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _exp_fp32
.p2align 4

_exp_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x2, .Ldone

    smstart                     // streaming SVE + ZA

    ptrue   p0.s

    // Load constants
    adr     x9, .Lconst
    ld1rw   {z16.s}, p0/z, [x9]        // log2(e)
    ld1rw   {z17.s}, p0/z, [x9, #4]    // ln(2)
    ld1rw   {z8.s}, p0/z, [x9, #8]     // 1.0  (c0 = c1)
    ld1rw   {z9.s}, p0/z, [x9, #12]    // 0.5  (c2)
    ld1rw   {z10.s}, p0/z, [x9, #16]   // 1/6  (c3)
    ld1rw   {z11.s}, p0/z, [x9, #20]   // 1/24 (c4)
    ld1rw   {z12.s}, p0/z, [x9, #24]   // 1/120 (c5)
    ld1rw   {z13.s}, p0/z, [x9, #28]   // 1/720 (c6)
    ld1rw   {z14.s}, p0/z, [x9, #32]   // 1/5040 (c7)
    ld1rw   {z15.s}, p0/z, [x9, #36]   // 88.0  (clamp hi)
    ld1rw   {z26.s}, p0/z, [x9, #40]   // -88.0 (clamp lo)

    mov     w11, #0             // ZA vector select register

    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    // ── Load 4 vectors ──
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]

    // ── Clamp to [-88, 88] ──
    fmin    z0.s, p0/m, z0.s, z15.s
    fmin    z1.s, p0/m, z1.s, z15.s
    fmin    z2.s, p0/m, z2.s, z15.s
    fmin    z3.s, p0/m, z3.s, z15.s
    fmax    z0.s, p0/m, z0.s, z26.s
    fmax    z1.s, p0/m, z1.s, z26.s
    fmax    z2.s, p0/m, z2.s, z26.s
    fmax    z3.s, p0/m, z3.s, z26.s

    // ── Range reduction ──
    // n = round(x * log2e) → z28-z31 (preserved for fscale)
    fmul    z28.s, z0.s, z16.s
    fmul    z29.s, z1.s, z16.s
    fmul    z30.s, z2.s, z16.s
    fmul    z31.s, z3.s, z16.s
    frintn  z28.s, p0/m, z28.s
    frintn  z29.s, p0/m, z29.s
    frintn  z30.s, p0/m, z30.s
    frintn  z31.s, p0/m, z31.s

    // r = x - n * ln2 → z4-z7
    movprfx z4, z0
    fmls    z4.s, p0/m, z28.s, z17.s
    movprfx z5, z1
    fmls    z5.s, p0/m, z29.s, z17.s
    movprfx z6, z2
    fmls    z6.s, p0/m, z30.s, z17.s
    movprfx z7, z3
    fmls    z7.s, p0/m, z31.s, z17.s

    // ── Polynomial: exp(r) ≈ 1 + r + r²/2 + r³/6 + ... + r⁷/5040 ──
    // Power series via FMLA VGx4 into ZA rows 0-3
    zero    {za}

    // Term c1*r  (c1 = 1.0 = z8)
    fmla    za.s[w11, 0, vgx4], {z4.s-z7.s}, z8.s

    // r² → z0-z3, then term c2*r²
    fmul    z0.s, z4.s, z4.s
    fmul    z1.s, z5.s, z5.s
    fmul    z2.s, z6.s, z6.s
    fmul    z3.s, z7.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z9.s

    // r³ → z18-z21, then term c3*r³
    fmul    z18.s, z0.s, z4.s
    fmul    z19.s, z1.s, z5.s
    fmul    z20.s, z2.s, z6.s
    fmul    z21.s, z3.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z10.s

    // r⁴ → z22-z25, then term c4*r⁴
    fmul    z22.s, z0.s, z0.s
    fmul    z23.s, z1.s, z1.s
    fmul    z24.s, z2.s, z2.s
    fmul    z25.s, z3.s, z3.s
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z11.s

    // r⁵ = r⁴ * r → z18-z21, then term c5*r⁵
    fmul    z18.s, z22.s, z4.s
    fmul    z19.s, z23.s, z5.s
    fmul    z20.s, z24.s, z6.s
    fmul    z21.s, z25.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z12.s

    // r⁶ = r⁴ * r² → z22-z25, then term c6*r⁶
    fmul    z22.s, z22.s, z0.s
    fmul    z23.s, z23.s, z1.s
    fmul    z24.s, z24.s, z2.s
    fmul    z25.s, z25.s, z3.s
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z13.s

    // r⁷ = r⁶ * r → z18-z21, then term c7*r⁷
    fmul    z18.s, z22.s, z4.s
    fmul    z19.s, z23.s, z5.s
    fmul    z20.s, z24.s, z6.s
    fmul    z21.s, z25.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z14.s

    // Extract polynomial sum from ZA, add c0 = 1.0
    mova    {z0.s-z3.s}, za.s[w11, 0, vgx4]
    fadd    z0.s, z0.s, z8.s
    fadd    z1.s, z1.s, z8.s
    fadd    z2.s, z2.s, z8.s
    fadd    z3.s, z3.s, z8.s

    // ── Scale by 2^n ──
    // Convert n from float to integer (fscale expects integer exponent)
    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s

    fscale  z0.s, p0/m, z0.s, z28.s
    fscale  z1.s, p0/m, z1.s, z29.s
    fscale  z2.s, p0/m, z2.s, z30.s
    fscale  z3.s, p0/m, z3.s, z31.s

    // ── Store ──
    st1w    {z0.s-z3.s}, pn9, [x1, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x2, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret

.p2align 2
.Lconst:
    .long   0x3FB8AA3B  // log2(e) = 1.44269504
    .long   0x3F317218  // ln(2)   = 0.69314718
    .long   0x3F800000  // 1.0     (c0 = c1)
    .long   0x3F000000  // 0.5     (c2)
    .long   0x3E2AAAAB  // 1/6     (c3)
    .long   0x3D2AAAAB  // 1/24    (c4)
    .long   0x3C088889  // 1/120   (c5)
    .long   0x3AB60B61  // 1/720   (c6)
    .long   0x39500D01  // 1/5040  (c7)
    .long   0x42B00000  // 88.0
    .long   0xC2B00000  // -88.0
