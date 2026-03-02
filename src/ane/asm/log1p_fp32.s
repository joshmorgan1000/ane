// log1p_fp32.s — Element-wise ln(1 + x) via SME2 streaming SVE
//
// void log1p_fp32(const float *input, float *output, long n)
// AAPCS: x0=input, x1=output, x2=n
//
// Computes ln(1 + x) for each element.  Adds 1.0 to each input, clamps
// to [FLT_MIN, +inf) to guard log(0) / log(negative), then applies the
// same IEEE-754 range-reduction + 6-term Horner polynomial used in
// log_fp32.s.
//
// Algorithm:
//   y = clamp(x + 1.0, FLT_MIN, +inf)
//   y = 2^e * m,  normalise m to [1, sqrt(2)]
//   s = (m-1)/(m+1)
//   poly = 2*s*(1 + s^2/3 + s^4/5 + s^6/7 + s^8/9 + s^10/11)   (Horner)
//   result = e*ln(2) + poly
//
// Processes 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _log1p_fp32
.p2align 4

_log1p_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x2, .Ldone

    smstart sm

    ptrue   p0.s

    adr     x9, .Lconst
    ld1rw   {z16.s}, p0/z, [x9]        // ln(2)   = 0x3F317218
    ld1rw   {z17.s}, p0/z, [x9, #4]    // 1.0     = 0x3F800000
    ld1rw   {z18.s}, p0/z, [x9, #8]    // 2.0     = 0x40000000
    ld1rw   {z19.s}, p0/z, [x9, #12]   // 1/3     = 0x3EAAAAAB
    ld1rw   {z20.s}, p0/z, [x9, #16]   // 1/5     = 0x3E4CCCCD
    ld1rw   {z21.s}, p0/z, [x9, #20]   // 1/7     = 0x3E124925
    ld1rw   {z22.s}, p0/z, [x9, #24]   // 1/9     = 0x3DE38E39
    ld1rw   {z23.s}, p0/z, [x9, #28]   // 1/11    = 0x3DBA2E8C
    ld1rw   {z25.s}, p0/z, [x9, #32]   // sqrt(2) = 0x3FB504F3
    ld1rw   {z26.s}, p0/z, [x9, #36]   // 0.5     = 0x3F000000
    ld1rw   {z28.s}, p0/z, [x9, #40]   // FLT_MIN = 0x00800000
    mov     w10, #127
    dup     z24.s, w10                  // exponent bias
    mov     w11, #1
    dup     z27.s, w11                  // integer 1

    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]

    // y = x + 1.0
    fadd    z0.s, p0/m, z0.s, z17.s
    fadd    z1.s, p0/m, z1.s, z17.s
    fadd    z2.s, p0/m, z2.s, z17.s
    fadd    z3.s, p0/m, z3.s, z17.s

    // clamp y to [FLT_MIN, +inf)
    fmax    z0.s, p0/m, z0.s, z28.s
    fmax    z1.s, p0/m, z1.s, z28.s
    fmax    z2.s, p0/m, z2.s, z28.s
    fmax    z3.s, p0/m, z3.s, z28.s

    // Extract exponent: e = (bits >> 23) - 127  ->  z4-z7
    lsr     z4.s, z0.s, #23
    lsr     z5.s, z1.s, #23
    lsr     z6.s, z2.s, #23
    lsr     z7.s, z3.s, #23
    sub     z4.s, z4.s, z24.s
    sub     z5.s, z5.s, z24.s
    sub     z6.s, z6.s, z24.s
    sub     z7.s, z7.s, z24.s

    // Set exponent to 127 -> m in [1.0, 2.0)
    and     z0.s, z0.s, #0x007FFFFF
    and     z1.s, z1.s, #0x007FFFFF
    and     z2.s, z2.s, #0x007FFFFF
    and     z3.s, z3.s, #0x007FFFFF
    orr     z0.s, z0.s, #0x3F800000
    orr     z1.s, z1.s, #0x3F800000
    orr     z2.s, z2.s, #0x3F800000
    orr     z3.s, z3.s, #0x3F800000

    // Normalize: if m > sqrt(2), halve m and increment e
    fcmgt   p1.s, p0/z, z0.s, z25.s
    fcmgt   p2.s, p0/z, z1.s, z25.s
    fcmgt   p3.s, p0/z, z2.s, z25.s
    fcmgt   p4.s, p0/z, z3.s, z25.s
    fmul    z0.s, p1/m, z0.s, z26.s
    fmul    z1.s, p2/m, z1.s, z26.s
    fmul    z2.s, p3/m, z2.s, z26.s
    fmul    z3.s, p4/m, z3.s, z26.s
    add     z4.s, p1/m, z4.s, z27.s
    add     z5.s, p2/m, z5.s, z27.s
    add     z6.s, p3/m, z6.s, z27.s
    add     z7.s, p4/m, z7.s, z27.s

    // Convert exponent to float
    scvtf   z4.s, p0/m, z4.s
    scvtf   z5.s, p0/m, z5.s
    scvtf   z6.s, p0/m, z6.s
    scvtf   z7.s, p0/m, z7.s

    // f = m - 1.0  ->  z8-z11
    mov     z8.d,  z0.d
    mov     z9.d,  z1.d
    mov     z10.d, z2.d
    mov     z11.d, z3.d
    fsub    z8.s,  p0/m, z8.s,  z17.s
    fsub    z9.s,  p0/m, z9.s,  z17.s
    fsub    z10.s, p0/m, z10.s, z17.s
    fsub    z11.s, p0/m, z11.s, z17.s

    // f + 2.0 = m + 1  ->  z0-z3 (reuse)
    mov     z0.d, z8.d
    mov     z1.d, z9.d
    mov     z2.d, z10.d
    mov     z3.d, z11.d
    fadd    z0.s, p0/m, z0.s, z18.s
    fadd    z1.s, p0/m, z1.s, z18.s
    fadd    z2.s, p0/m, z2.s, z18.s
    fadd    z3.s, p0/m, z3.s, z18.s

    // s = f / (f+2) = (m-1)/(m+1)  ->  z8-z11
    fdiv    z8.s,  p0/m, z8.s,  z0.s
    fdiv    z9.s,  p0/m, z9.s,  z1.s
    fdiv    z10.s, p0/m, z10.s, z2.s
    fdiv    z11.s, p0/m, z11.s, z3.s

    // s2 = s * s  ->  z0-z3 (reuse)
    fmul    z0.s, z8.s,  z8.s
    fmul    z1.s, z9.s,  z9.s
    fmul    z2.s, z10.s, z10.s
    fmul    z3.s, z11.s, z11.s

    // Horner: start with p = 1/11
    mov     z12.d, z23.d
    mov     z13.d, z23.d
    mov     z14.d, z23.d
    mov     z15.d, z23.d

    fmad    z12.s, p0/m, z0.s, z22.s   // p = p*s2 + 1/9
    fmad    z13.s, p0/m, z1.s, z22.s
    fmad    z14.s, p0/m, z2.s, z22.s
    fmad    z15.s, p0/m, z3.s, z22.s

    fmad    z12.s, p0/m, z0.s, z21.s   // p = p*s2 + 1/7
    fmad    z13.s, p0/m, z1.s, z21.s
    fmad    z14.s, p0/m, z2.s, z21.s
    fmad    z15.s, p0/m, z3.s, z21.s

    fmad    z12.s, p0/m, z0.s, z20.s   // p = p*s2 + 1/5
    fmad    z13.s, p0/m, z1.s, z20.s
    fmad    z14.s, p0/m, z2.s, z20.s
    fmad    z15.s, p0/m, z3.s, z20.s

    fmad    z12.s, p0/m, z0.s, z19.s   // p = p*s2 + 1/3
    fmad    z13.s, p0/m, z1.s, z19.s
    fmad    z14.s, p0/m, z2.s, z19.s
    fmad    z15.s, p0/m, z3.s, z19.s

    fmad    z12.s, p0/m, z0.s, z17.s   // p = p*s2 + 1.0
    fmad    z13.s, p0/m, z1.s, z17.s
    fmad    z14.s, p0/m, z2.s, z17.s
    fmad    z15.s, p0/m, z3.s, z17.s

    // ln(m) = 2 * s * p
    fmul    z12.s, p0/m, z12.s, z8.s
    fmul    z13.s, p0/m, z13.s, z9.s
    fmul    z14.s, p0/m, z14.s, z10.s
    fmul    z15.s, p0/m, z15.s, z11.s
    fmul    z12.s, p0/m, z12.s, z18.s
    fmul    z13.s, p0/m, z13.s, z18.s
    fmul    z14.s, p0/m, z14.s, z18.s
    fmul    z15.s, p0/m, z15.s, z18.s

    // result = e*ln(2) + ln(m)
    fmla    z12.s, p0/m, z4.s, z16.s
    fmla    z13.s, p0/m, z5.s, z16.s
    fmla    z14.s, p0/m, z6.s, z16.s
    fmla    z15.s, p0/m, z7.s, z16.s

    st1w    {z12.s-z15.s}, pn9, [x1, x8, lsl #2]

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
    .long   0x00800000  // FLT_MIN
