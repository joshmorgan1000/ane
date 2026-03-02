// selu_backward_fp32.s — SELU backward pass via SME2 streaming SVE
//
// void selu_backward_fp32(const float* dy, const float* x, float* dx, long n)
// AAPCS: x0=dy (upstream gradient), x1=x (forward input), x2=dx (output gradient), x3=n
//
// Computes: dx[i] = lambda * dy[i] * (x[i] > 0 ? 1.0 : alpha * exp(x[i]))
//
// Hardcoded SELU constants:
//   lambda = 1.0507009873554805  (IEEE-754 SP: 0x3F867D5F)
//   alpha  = 1.6732631921814685  (IEEE-754 SP: 0x3FD62D7D)
// - alpha*lambda is precomputed as a single constant = 1.7580993175506592 (0x3FE10966)
//
// Strategy:
// - For x > 0:  result = lambda * dy
// - For x <= 0: result = (alpha*lambda) * exp(x) * dy
// - alpha*lambda precomputed to save one multiply on the x<=0 path.
// - 7-term Taylor polynomial for exp via range-reduction + ZA VGx4 FMLA + fscale.
// - Clamp x to [-88, 0] before exp.
//
// Register layout:
//   z0-z3:   dy  (preserved until final sel and final multiply)
//   z4-z7:   original x → clamped x → r → exp(x) → exp(x)*alpha*lambda*dy
//   z8:      1.0
//   z9-z14:  polynomial coefficients
//   z15:     0.0 (upper clamp)
//   z16:     log2(e)
//   z17:     ln(2)
//   z18:     lambda (for x>0 branch)
//   z19:     alpha*lambda (for x<=0 branch)
//   z20-z23: power-of-r temporaries (20%4=0, valid for ZA VGx4)
//   z24:     zero for fcmgt (24%4=0)
//   z26:     -88.0 (lower clamp)
//   z28-z31: n_val (28%4=0)
//
// NOTE: ZA VGx4 vector lists MUST start at a register index that is a multiple of 4.
//       We use {z4-z7} for r input and {z20-z23} for powers.
//
// Uses full smstart (ZA needed for FMLA VGx4).
// Processing: 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _selu_backward_fp32
.p2align 4

_selu_backward_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x3, .Lsebw_done

    // Enter streaming mode with ZA access
    smstart

    ptrue   p0.s
    mov     w11, #0                 // ZA slice index

    // Load constants
    adr     x9, .Lsebw_const
    ld1rw   {z16.s}, p0/z, [x9]         // log2(e)
    ld1rw   {z17.s}, p0/z, [x9, #4]     // ln(2)
    ld1rw   {z8.s},  p0/z, [x9, #8]     // 1.0
    ld1rw   {z9.s},  p0/z, [x9, #12]    // 0.5 (1/2!)
    ld1rw   {z10.s}, p0/z, [x9, #16]    // 1/6 (1/3!)
    ld1rw   {z11.s}, p0/z, [x9, #20]    // 1/24 (1/4!)
    ld1rw   {z12.s}, p0/z, [x9, #24]    // 1/120 (1/5!)
    ld1rw   {z13.s}, p0/z, [x9, #28]    // 1/720 (1/6!)
    ld1rw   {z14.s}, p0/z, [x9, #32]    // 1/5040 (1/7!)
    ld1rw   {z15.s}, p0/z, [x9, #36]    // 0.0 (upper clamp)
    ld1rw   {z26.s}, p0/z, [x9, #40]    // -88.0 (lower clamp)
    ld1rw   {z18.s}, p0/z, [x9, #44]    // lambda
    ld1rw   {z19.s}, p0/z, [x9, #48]    // alpha*lambda

    // Zero for comparisons
    mov     z24.d, #0

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lsebw_loop:
    // Load dy and x
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]   // dy
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]   // x

    // Build x>0 mask
    fcmgt   p1.s, p0/z, z4.s, z24.s
    fcmgt   p2.s, p0/z, z5.s, z24.s
    fcmgt   p3.s, p0/z, z6.s, z24.s
    fcmgt   p4.s, p0/z, z7.s, z24.s

    // Clamp x in-place to [-88, 0] for exp
    fmax    z4.s, p0/m, z4.s, z26.s
    fmin    z4.s, p0/m, z4.s, z15.s
    fmax    z5.s, p0/m, z5.s, z26.s
    fmin    z5.s, p0/m, z5.s, z15.s
    fmax    z6.s, p0/m, z6.s, z26.s
    fmin    z6.s, p0/m, z6.s, z15.s
    fmax    z7.s, p0/m, z7.s, z26.s
    fmin    z7.s, p0/m, z7.s, z15.s

    // Range reduction: n_val = round(x_clamped * log2(e)) → z28-z31
    fmul    z28.s, z4.s, z16.s
    fmul    z29.s, z5.s, z16.s
    fmul    z30.s, z6.s, z16.s
    fmul    z31.s, z7.s, z16.s
    frintn  z28.s, p0/m, z28.s
    frintn  z29.s, p0/m, z29.s
    frintn  z30.s, p0/m, z30.s
    frintn  z31.s, p0/m, z31.s

    // r = x_clamped - n_val * ln(2)
    fmls    z4.s, p0/m, z28.s, z17.s
    fmls    z5.s, p0/m, z29.s, z17.s
    fmls    z6.s, p0/m, z30.s, z17.s
    fmls    z7.s, p0/m, z31.s, z17.s

    // --- 7-term Taylor polynomial for exp(r) via ZA VGx4 ---
    // r in {z4-z7} (4%4=0), powers in {z20-z23} (20%4=0).
    zero    {za}

    // c1*r
    fmla    za.s[w11, 0, vgx4], {z4.s-z7.s}, z8.s

    // r^2 → {z20-z23}
    fmul    z20.s, z4.s, z4.s
    fmul    z21.s, z5.s, z5.s
    fmul    z22.s, z6.s, z6.s
    fmul    z23.s, z7.s, z7.s
    // c2*r^2
    fmla    za.s[w11, 0, vgx4], {z20.s-z23.s}, z9.s

    // r^3 = r^2 * r
    fmul    z20.s, z20.s, z4.s
    fmul    z21.s, z21.s, z5.s
    fmul    z22.s, z22.s, z6.s
    fmul    z23.s, z23.s, z7.s
    // c3*r^3
    fmla    za.s[w11, 0, vgx4], {z20.s-z23.s}, z10.s

    // r^4 = r^2 * r^2
    fmul    z20.s, z4.s, z4.s
    fmul    z21.s, z5.s, z5.s
    fmul    z22.s, z6.s, z6.s
    fmul    z23.s, z7.s, z7.s
    fmul    z20.s, z20.s, z20.s
    fmul    z21.s, z21.s, z21.s
    fmul    z22.s, z22.s, z22.s
    fmul    z23.s, z23.s, z23.s
    // c4*r^4
    fmla    za.s[w11, 0, vgx4], {z20.s-z23.s}, z11.s

    // r^5 = r^4 * r
    fmul    z20.s, z20.s, z4.s
    fmul    z21.s, z21.s, z5.s
    fmul    z22.s, z22.s, z6.s
    fmul    z23.s, z23.s, z7.s
    // c5*r^5
    fmla    za.s[w11, 0, vgx4], {z20.s-z23.s}, z12.s

    // r^6 = r^3 * r^3
    fmul    z20.s, z4.s, z4.s
    fmul    z21.s, z5.s, z5.s
    fmul    z22.s, z6.s, z6.s
    fmul    z23.s, z7.s, z7.s
    fmul    z20.s, z20.s, z4.s
    fmul    z21.s, z21.s, z5.s
    fmul    z22.s, z22.s, z6.s
    fmul    z23.s, z23.s, z7.s
    fmul    z20.s, z20.s, z20.s
    fmul    z21.s, z21.s, z21.s
    fmul    z22.s, z22.s, z22.s
    fmul    z23.s, z23.s, z23.s
    // c6*r^6
    fmla    za.s[w11, 0, vgx4], {z20.s-z23.s}, z13.s

    // r^7 = r^6 * r
    fmul    z20.s, z20.s, z4.s
    fmul    z21.s, z21.s, z5.s
    fmul    z22.s, z22.s, z6.s
    fmul    z23.s, z23.s, z7.s
    // c7*r^7
    fmla    za.s[w11, 0, vgx4], {z20.s-z23.s}, z14.s

    // Extract into {z4-z7} and add c0 = 1.0
    mova    {z4.s-z7.s}, za.s[w11, 0, vgx4]
    fadd    z4.s, z4.s, z8.s
    fadd    z5.s, z5.s, z8.s
    fadd    z6.s, z6.s, z8.s
    fadd    z7.s, z7.s, z8.s

    // Scale by 2^n_val to get exp(x)
    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s
    fscale  z4.s, p0/m, z4.s, z28.s
    fscale  z5.s, p0/m, z5.s, z29.s
    fscale  z6.s, p0/m, z6.s, z30.s
    fscale  z7.s, p0/m, z7.s, z31.s

    // alpha*lambda * exp(x) for x<=0 branch → z4-z7
    fmul    z4.s, p0/m, z4.s, z19.s
    fmul    z5.s, p0/m, z5.s, z19.s
    fmul    z6.s, p0/m, z6.s, z19.s
    fmul    z7.s, p0/m, z7.s, z19.s

    // Multiply x<=0 result by dy
    fmul    z4.s, p0/m, z4.s, z0.s
    fmul    z5.s, p0/m, z5.s, z1.s
    fmul    z6.s, p0/m, z6.s, z2.s
    fmul    z7.s, p0/m, z7.s, z3.s

    // lambda * dy for x>0 branch → z0-z3 (in-place)
    fmul    z0.s, p0/m, z0.s, z18.s
    fmul    z1.s, p0/m, z1.s, z18.s
    fmul    z2.s, p0/m, z2.s, z18.s
    fmul    z3.s, p0/m, z3.s, z18.s

    // Select: x>0 → lambda*dy (z0-z3), x<=0 → lambda*alpha*exp(x)*dy (z4-z7)
    sel     z0.s, p1, z0.s, z4.s
    sel     z1.s, p2, z1.s, z5.s
    sel     z2.s, p3, z2.s, z6.s
    sel     z3.s, p4, z3.s, z7.s

    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lsebw_loop

    smstop

.Lsebw_done:
    ldp     d14, d15, [sp, #64]
    ldp     d12, d13, [sp, #48]
    ldp     d10, d11, [sp, #32]
    ldp     d8,  d9,  [sp, #16]
    ldp     x29, x30, [sp], #80
    ret

// Constant pool
.p2align 2
.Lsebw_const:
    .long   0x3FB8AA3B  // log2(e)                  = 1.44269504
    .long   0x3F317218  // ln(2)                     = 0.69314718
    .long   0x3F800000  // 1.0
    .long   0x3F000000  // 0.5                       = 1/2!
    .long   0x3E2AAAAB  // 1/6                       = 1/3!
    .long   0x3D2AAAAB  // 1/24                      = 1/4!
    .long   0x3C088889  // 1/120                     = 1/5!
    .long   0x3AB60B61  // 1/720                     = 1/6!
    .long   0x39500D01  // 1/5040                    = 1/7!
    .long   0x00000000  // 0.0                       (upper clamp)
    .long   0xC2B00000  // -88.0                     (lower clamp)
    .long   0x3F867D5F  // lambda = 1.0507009873554805
    .long   0x3FE10966  // alpha*lambda = 1.7580993175506592
