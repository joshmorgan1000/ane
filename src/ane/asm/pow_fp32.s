// pow_fp32.s — Element-wise power function via SME2 streaming SVE
//
// void pow_fp32(const float *base, const float *exponent, float *output, long n)
//
// Computes output[i] = base[i] ^ exponent[i] via exp(exponent[i] * log(base[i]))
//
// Edge cases:
//   - base[i] <= 0: set output[i] = 0.0
//   - exponent[i] == 0: set output[i] = 1.0
//
// Processes 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _pow_fp32
.p2align 4

_pow_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    stp     d14, d15, [sp, #96]

    cbz     x3, .Lpow_done

    smstart sm

    ptrue   p0.s

    // Load constants for log
    adr     x9, .Lpow_const
    ld1rw   {z16.s}, p0/z, [x9]        // ln(2)
    ld1rw   {z17.s}, p0/z, [x9, #4]    // 1.0
    ld1rw   {z18.s}, p0/z, [x9, #8]    // 2.0
    ld1rw   {z19.s}, p0/z, [x9, #12]   // 1/3
    ld1rw   {z20.s}, p0/z, [x9, #16]   // 1/5
    ld1rw   {z21.s}, p0/z, [x9, #20]   // 1/7
    ld1rw   {z22.s}, p0/z, [x9, #24]   // 1/9
    ld1rw   {z23.s}, p0/z, [x9, #28]   // 1/11
    ld1rw   {z24.s}, p0/z, [x9, #32]   // sqrt(2)
    ld1rw   {z25.s}, p0/z, [x9, #36]   // 0.5
    ld1rw   {z26.s}, p0/z, [x9, #40]   // log2(e)
    ld1rw   {z27.s}, p0/z, [x9, #44]   // 88.0 (clamp_hi)
    ld1rw   {z28.s}, p0/z, [x9, #48]   // -88.0 (clamp_lo)
    ld1rw   {z29.s}, p0/z, [x9, #52]   // 1/6 (exp c3)
    ld1rw   {z30.s}, p0/z, [x9, #56]   // 1/24 (exp c4)
    ld1rw   {z31.s}, p0/z, [x9, #60]   // 0.0

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lpow_loop:
    // ──────────────────────────────────────────────────────────────
    // Load base and exponent
    // ──────────────────────────────────────────────────────────────
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]     // base
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]     // exponent

    // ──────────────────────────────────────────────────────────────
    // LOG(base): Extract exponent and mantissa
    // ──────────────────────────────────────────────────────────────
    // Save base to z8-z11 for later use
    mov     z8.d, z0.d
    mov     z9.d, z1.d
    mov     z10.d, z2.d
    mov     z11.d, z3.d

    // Extract exponent: e = (bits >> 23) - 127
    mov     w19, #127
    dup     z12.s, w19
    lsr     z13.s, z0.s, #23
    lsr     z14.s, z1.s, #23
    lsr     z15.s, z2.s, #23
    mov     z0.d, z3.d
    lsr     z0.s, z0.s, #23
    sub     z13.s, z13.s, z12.s
    sub     z14.s, z14.s, z12.s
    sub     z15.s, z15.s, z12.s
    sub     z0.s, z0.s, z12.s

    // Set exponent bits to 127 → m in [1.0, 2.0)
    mov     w19, #0x007FFFFF
    dup     z12.s, w19
    and     z8.s, z8.s, z12.s
    and     z9.s, z9.s, z12.s
    and     z10.s, z10.s, z12.s
    and     z11.s, z11.s, z12.s
    mov     w19, #0x3F800000
    dup     z12.s, w19
    orr     z8.s, z8.s, z12.s
    orr     z9.s, z9.s, z12.s
    orr     z10.s, z10.s, z12.s
    orr     z11.s, z11.s, z12.s

    // Normalize: if m > sqrt(2), halve m and increment e
    fcmgt   p1.s, p0/z, z8.s, z24.s
    fcmgt   p2.s, p0/z, z9.s, z24.s
    fcmgt   p3.s, p0/z, z10.s, z24.s
    fcmgt   p4.s, p0/z, z11.s, z24.s
    fmul    z8.s, p1/m, z8.s, z25.s
    fmul    z9.s, p2/m, z9.s, z25.s
    fmul    z10.s, p3/m, z10.s, z25.s
    fmul    z11.s, p4/m, z11.s, z25.s
    mov     w19, #1
    dup     z12.s, w19
    add     z13.s, p1/m, z13.s, z12.s
    add     z14.s, p2/m, z14.s, z12.s
    add     z15.s, p3/m, z15.s, z12.s
    add     z0.s, p4/m, z0.s, z12.s

    // Convert e to float
    scvtf   z13.s, p0/m, z13.s
    scvtf   z14.s, p0/m, z14.s
    scvtf   z15.s, p0/m, z15.s
    scvtf   z0.s, p0/m, z0.s

    // ──────────────────────────────────────────────────────────────
    // LOG(base): Compute s = (m-1) / (m+1)
    // ──────────────────────────────────────────────────────────────
    // f = m - 1.0
    fsub    z8.s, z8.s, z17.s
    fsub    z9.s, z9.s, z17.s
    fsub    z10.s, z10.s, z17.s
    fsub    z11.s, z11.s, z17.s

    // denominator = f + 2.0 = m + 1
    fadd    z12.s, z8.s, z18.s
    mov     z1.d, z9.d
    fadd    z1.s, z1.s, z18.s
    mov     z2.d, z10.d
    fadd    z2.s, z2.s, z18.s
    mov     z3.d, z11.d
    fadd    z3.s, z3.s, z18.s

    // s = f / (f + 2.0)
    fdiv    z8.s, p0/m, z8.s, z12.s
    fdiv    z9.s, p0/m, z9.s, z1.s
    fdiv    z10.s, p0/m, z10.s, z2.s
    fdiv    z11.s, p0/m, z11.s, z3.s

    // s2 = s * s
    fmul    z12.s, z8.s, z8.s
    fmul    z1.s, z9.s, z9.s
    fmul    z2.s, z10.s, z10.s
    fmul    z3.s, z11.s, z11.s

    // ──────────────────────────────────────────────────────────────
    // LOG(base): Polynomial evaluation
    // ──────────────────────────────────────────────────────────────
    // p = 1/11
    mov     z4.d, z23.d
    mov     z5.d, z23.d
    mov     z6.d, z23.d
    mov     z7.d, z23.d

    // p = p*s2 + 1/9
    fmad    z4.s, p0/m, z12.s, z22.s
    fmad    z5.s, p0/m, z1.s, z22.s
    fmad    z6.s, p0/m, z2.s, z22.s
    fmad    z7.s, p0/m, z3.s, z22.s

    // p = p*s2 + 1/7
    fmad    z4.s, p0/m, z12.s, z21.s
    fmad    z5.s, p0/m, z1.s, z21.s
    fmad    z6.s, p0/m, z2.s, z21.s
    fmad    z7.s, p0/m, z3.s, z21.s

    // p = p*s2 + 1/5
    fmad    z4.s, p0/m, z12.s, z20.s
    fmad    z5.s, p0/m, z1.s, z20.s
    fmad    z6.s, p0/m, z2.s, z20.s
    fmad    z7.s, p0/m, z3.s, z20.s

    // p = p*s2 + 1/3
    fmad    z4.s, p0/m, z12.s, z19.s
    fmad    z5.s, p0/m, z1.s, z19.s
    fmad    z6.s, p0/m, z2.s, z19.s
    fmad    z7.s, p0/m, z3.s, z19.s

    // p = p*s2 + 1.0
    fmad    z4.s, p0/m, z12.s, z17.s
    fmad    z5.s, p0/m, z1.s, z17.s
    fmad    z6.s, p0/m, z2.s, z17.s
    fmad    z7.s, p0/m, z3.s, z17.s

    // ln(m) = 2 * s * poly
    fmul    z4.s, z4.s, z8.s
    fmul    z5.s, z5.s, z9.s
    fmul    z6.s, z6.s, z10.s
    fmul    z7.s, z7.s, z11.s
    fmul    z4.s, z4.s, z18.s
    fmul    z5.s, z5.s, z18.s
    fmul    z6.s, z6.s, z18.s
    fmul    z7.s, z7.s, z18.s

    // result = e * ln2 + ln(m)
    fmla    z4.s, p0/m, z13.s, z16.s
    fmla    z5.s, p0/m, z14.s, z16.s
    fmla    z6.s, p0/m, z15.s, z16.s
    fmla    z7.s, p0/m, z0.s, z16.s

    // ──────────────────────────────────────────────────────────────
    // Multiply by exponent: product = log(base) * exponent
    // ──────────────────────────────────────────────────────────────
    // Load exponent values (saved earlier in z4-z7 from input)
    ld1w    {z0.s-z3.s}, pn9/z, [x1, x8, lsl #2]
    fmul    z4.s, z4.s, z0.s
    fmul    z5.s, z5.s, z1.s
    fmul    z6.s, z6.s, z2.s
    fmul    z7.s, z7.s, z3.s

    // ──────────────────────────────────────────────────────────────
    // EXP(product): Clamp to [-88, 88]
    // ──────────────────────────────────────────────────────────────
    fmin    z4.s, p0/m, z4.s, z27.s
    fmin    z5.s, p0/m, z5.s, z27.s
    fmin    z6.s, p0/m, z6.s, z27.s
    fmin    z7.s, p0/m, z7.s, z27.s
    fmax    z4.s, p0/m, z4.s, z28.s
    fmax    z5.s, p0/m, z5.s, z28.s
    fmax    z6.s, p0/m, z6.s, z28.s
    fmax    z7.s, p0/m, z7.s, z28.s

    // ──────────────────────────────────────────────────────────────
    // EXP(product): Range reduction x = n*ln2 + r
    // ──────────────────────────────────────────────────────────────
    // n = round(x * log2e)
    fmul    z8.s, z4.s, z26.s
    fmul    z9.s, z5.s, z26.s
    fmul    z10.s, z6.s, z26.s
    fmul    z11.s, z7.s, z26.s
    frintn  z8.s, p0/m, z8.s
    frintn  z9.s, p0/m, z9.s
    frintn  z10.s, p0/m, z10.s
    frintn  z11.s, p0/m, z11.s

    // r = x - n * ln2
    movprfx z0, z4
    fmls    z0.s, p0/m, z8.s, z16.s
    movprfx z1, z5
    fmls    z1.s, p0/m, z9.s, z16.s
    movprfx z2, z6
    fmls    z2.s, p0/m, z10.s, z16.s
    movprfx z3, z7
    fmls    z3.s, p0/m, z11.s, z16.s

    // ──────────────────────────────────────────────────────────────
    // EXP(product): Polynomial exp(r) ≈ 1 + r + r²/2 + ... + r⁷/5040
    // ──────────────────────────────────────────────────────────────
    // Load remaining exp coefficients
    adr     x9, .Lpow_const_exp
    ld1rw   {z12.s}, p0/z, [x9]        // 1/120 (c5)
    ld1rw   {z13.s}, p0/z, [x9, #4]    // 1/720 (c6)
    ld1rw   {z14.s}, p0/z, [x9, #8]    // 1/5040 (c7)

    // Start with c7
    mov     z4.d, z14.d
    mov     z5.d, z14.d
    mov     z6.d, z14.d
    mov     z7.d, z14.d

    // p = p*r + c6
    fmad    z4.s, p0/m, z0.s, z13.s
    fmad    z5.s, p0/m, z1.s, z13.s
    fmad    z6.s, p0/m, z2.s, z13.s
    fmad    z7.s, p0/m, z3.s, z13.s

    // p = p*r + c5
    fmad    z4.s, p0/m, z0.s, z12.s
    fmad    z5.s, p0/m, z1.s, z12.s
    fmad    z6.s, p0/m, z2.s, z12.s
    fmad    z7.s, p0/m, z3.s, z12.s

    // p = p*r + c4 (1/24)
    fmad    z4.s, p0/m, z0.s, z30.s
    fmad    z5.s, p0/m, z1.s, z30.s
    fmad    z6.s, p0/m, z2.s, z30.s
    fmad    z7.s, p0/m, z3.s, z30.s

    // p = p*r + c3 (1/6)
    fmad    z4.s, p0/m, z0.s, z29.s
    fmad    z5.s, p0/m, z1.s, z29.s
    fmad    z6.s, p0/m, z2.s, z29.s
    fmad    z7.s, p0/m, z3.s, z29.s

    // p = p*r + c2 (0.5)
    fmad    z4.s, p0/m, z0.s, z25.s
    fmad    z5.s, p0/m, z1.s, z25.s
    fmad    z6.s, p0/m, z2.s, z25.s
    fmad    z7.s, p0/m, z3.s, z25.s

    // p = p*r + c1 (1.0)
    fmad    z4.s, p0/m, z0.s, z17.s
    fmad    z5.s, p0/m, z1.s, z17.s
    fmad    z6.s, p0/m, z2.s, z17.s
    fmad    z7.s, p0/m, z3.s, z17.s

    // p = p*r + c0 (1.0)
    fmad    z4.s, p0/m, z0.s, z17.s
    fmad    z5.s, p0/m, z1.s, z17.s
    fmad    z6.s, p0/m, z2.s, z17.s
    fmad    z7.s, p0/m, z3.s, z17.s

    // ──────────────────────────────────────────────────────────────
    // EXP(product): Scale by 2^n
    // ──────────────────────────────────────────────────────────────
    // Convert n to integer for fscale
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s

    fscale  z4.s, p0/m, z4.s, z8.s
    fscale  z5.s, p0/m, z5.s, z9.s
    fscale  z6.s, p0/m, z6.s, z10.s
    fscale  z7.s, p0/m, z7.s, z11.s

    // ──────────────────────────────────────────────────────────────
    // Recompute edge case predicates (they were clobbered during computation)
    // ──────────────────────────────────────────────────────────────
    ld1w    {z12.s-z15.s}, pn9/z, [x0, x8, lsl #2]   // reload base vectors
    fcmle   p1.s, p0/z, z12.s, z31.s                 // base[0] <= 0
    fcmle   p2.s, p0/z, z13.s, z31.s                 // base[1] <= 0
    fcmle   p3.s, p0/z, z14.s, z31.s                 // base[2] <= 0
    fcmle   p4.s, p0/z, z15.s, z31.s                 // base[3] <= 0

    ld1w    {z0.s-z3.s}, pn9/z, [x1, x8, lsl #2]     // reload exponent vectors
    fcmeq   p5.s, p0/z, z0.s, z31.s                  // exp[0] == 0
    fcmeq   p6.s, p0/z, z1.s, z31.s                  // exp[1] == 0
    fcmeq   p7.s, p0/z, z2.s, z31.s                  // exp[2] == 0

    // ──────────────────────────────────────────────────────────────
    // Apply edge cases using sel (first arg selected when predicate TRUE)
    // ──────────────────────────────────────────────────────────────
    // If base <= 0, set result to 0.0
    sel     z4.s, p1, z31.s, z4.s                    // p1 TRUE → z31 (0.0)
    sel     z5.s, p2, z31.s, z5.s
    sel     z6.s, p3, z31.s, z6.s
    sel     z7.s, p4, z31.s, z7.s

    // If exponent == 0, set result to 1.0
    sel     z4.s, p5, z17.s, z4.s                    // p5 TRUE → z17 (1.0)
    sel     z5.s, p6, z17.s, z5.s
    sel     z6.s, p7, z17.s, z6.s
    // For z7, check exp[3] == 0 using fresh comparison
    fcmeq   p1.s, p0/z, z3.s, z31.s
    sel     z7.s, p1, z17.s, z7.s

    // ──────────────────────────────────────────────────────────────
    // Store result
    // ──────────────────────────────────────────────────────────────
    st1w    {z4.s-z7.s}, pn9, [x2, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lpow_loop

    smstop

.Lpow_done:
    ldp     d8,  d9,  [sp, #48]
    ldp     d10, d11, [sp, #64]
    ldp     d12, d13, [sp, #80]
    ldp     d14, d15, [sp, #96]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #112
    ret

.p2align 2
.Lpow_const:
    .long   0x3F317218  // ln(2)
    .long   0x3F800000  // 1.0
    .long   0x40000000  // 2.0
    .long   0x3EAAAAAB  // 1/3
    .long   0x3E4CCCCD  // 1/5
    .long   0x3E124925  // 1/7
    .long   0x3DE38E39  // 1/9
    .long   0x3DBA2E8C  // 1/11
    .long   0x3FB504F3  // sqrt(2)
    .long   0x3F000000  // 0.5
    .long   0x3FB8AA3B  // log2(e)
    .long   0x42B00000  // 88.0
    .long   0xC2B00000  // -88.0
    .long   0x3E2AAAAB  // 1/6 (exp c3)
    .long   0x3D2AAAAB  // 1/24 (exp c4)
    .long   0x00000000  // 0.0

.Lpow_const_exp:
    .long   0x3C088889  // 1/120 (c5)
    .long   0x3AB60B61  // 1/720 (c6)
    .long   0x39500D01  // 1/5040 (c7)
