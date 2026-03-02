// softplus_backward_fp32.s — Softplus backward pass via SME2 with ZA accumulation
//
// void softplus_backward_fp32(const float* dy, const float* x, float* dx, long n)
// AAPCS: x0=dy (upstream gradient), x1=x (forward input), x2=dx (output), x3=n
//
// Math: dx[i] = dy[i] * sigmoid(x[i])
//             = dy[i] / (1 + exp(-x[i]))
//
// Strategy: compute exp(-x) via 7-term Taylor with range reduction (same pattern as
// silu_backward), derive sigma = 1/(1+exp(-x)) via fdiv, then dx = dy * sigma.
//
// Uses smstart (full ZA) for the fmla vgx4 polynomial accumulation.
// Processes 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _softplus_backward_fp32
.p2align 4

_softplus_backward_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]
    stp     x19, x20, [sp, #80]

    cbz     x3, .Lspb_done

    // Save pointers before smstart
    mov     x19, x0                 // dy
    mov     x20, x1                 // x (input)

    // Enter streaming mode with ZA access (needed for fmla vgx4)
    smstart

    ptrue   p0.s
    mov     w11, #0                 // ZA vector select register

    // Load constants
    adr     x9, .Lspb_const
    ld1rw   {z16.s}, p0/z, [x9]        // log2(e)
    ld1rw   {z17.s}, p0/z, [x9, #4]    // ln(2)
    ld1rw   {z8.s},  p0/z, [x9, #8]    // 1.0
    ld1rw   {z9.s},  p0/z, [x9, #12]   // 0.5
    ld1rw   {z10.s}, p0/z, [x9, #16]   // 1/6
    ld1rw   {z11.s}, p0/z, [x9, #20]   // 1/24
    ld1rw   {z12.s}, p0/z, [x9, #24]   // 1/120
    ld1rw   {z13.s}, p0/z, [x9, #28]   // 1/720
    ld1rw   {z14.s}, p0/z, [x9, #32]   // 1/5040
    ld1rw   {z15.s}, p0/z, [x9, #36]   // 88.0 (clamp_hi)
    ld1rw   {z26.s}, p0/z, [x9, #40]   // -88.0 (clamp_lo)

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lspb_loop:
    // Load x into z0-z3
    ld1w    {z0.s-z3.s}, pn9/z, [x20, x8, lsl #2]

    // Negate x → compute exp(-x)
    fneg    z0.s, p0/m, z0.s
    fneg    z1.s, p0/m, z1.s
    fneg    z2.s, p0/m, z2.s
    fneg    z3.s, p0/m, z3.s

    // Clamp -x to [-88, 88]
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

    // r = (-x) - n_val * ln(2) — result in z0-z3 (fmls: z -= a*b)
    fmls    z0.s, p0/m, z28.s, z17.s
    fmls    z1.s, p0/m, z29.s, z17.s
    fmls    z2.s, p0/m, z30.s, z17.s
    fmls    z3.s, p0/m, z31.s, z17.s

    // --- 7-term Taylor polynomial exp(r) via ZA accumulation ---
    // exp(r) = 1 + r + r^2/2! + r^3/3! + ... + r^7/7!
    zero    {za}

    // c1 * r  (c1 = 1.0)
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z8.s

    // r^2 → z18-z21
    fmul    z18.s, z0.s, z0.s
    fmul    z19.s, z1.s, z1.s
    fmul    z20.s, z2.s, z2.s
    fmul    z21.s, z3.s, z3.s
    // c2 * r^2  (c2 = 0.5)
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z9.s

    // r^3 = r^2 * r → z22-z25
    fmul    z22.s, z18.s, z0.s
    fmul    z23.s, z19.s, z1.s
    fmul    z24.s, z20.s, z2.s
    fmul    z25.s, z21.s, z3.s
    // c3 * r^3  (c3 = 1/6)
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z10.s

    // r^4 = r^2 * r^2 → z22-z25
    fmul    z22.s, z18.s, z18.s
    fmul    z23.s, z19.s, z19.s
    fmul    z24.s, z20.s, z20.s
    fmul    z25.s, z21.s, z21.s
    // c4 * r^4  (c4 = 1/24)
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z11.s

    // r^5 = r^4 * r → z22-z25
    fmul    z22.s, z22.s, z0.s
    fmul    z23.s, z23.s, z1.s
    fmul    z24.s, z24.s, z2.s
    fmul    z25.s, z25.s, z3.s
    // c5 * r^5  (c5 = 1/120)
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z12.s

    // r^6 = r^5 * r → z18-z21 (clobbers r^2 slots — no longer needed)
    fmul    z18.s, z22.s, z18.s
    fmul    z19.s, z23.s, z19.s
    fmul    z20.s, z24.s, z20.s
    fmul    z21.s, z25.s, z21.s
    // c6 * r^6  (c6 = 1/720)
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z13.s

    // r^7 = r^6 * r → z18-z21
    fmul    z18.s, z18.s, z0.s
    fmul    z19.s, z19.s, z1.s
    fmul    z20.s, z20.s, z2.s
    fmul    z21.s, z21.s, z3.s
    // c7 * r^7  (c7 = 1/5040)
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z14.s

    // Extract polynomial sum, add c0 = 1.0
    mova    {z0.s-z3.s}, za.s[w11, 0, vgx4]
    fadd    z0.s, z0.s, z8.s
    fadd    z1.s, z1.s, z8.s
    fadd    z2.s, z2.s, z8.s
    fadd    z3.s, z3.s, z8.s

    // Scale by 2^n_val to reconstruct exp(-x)
    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s
    fscale  z0.s, p0/m, z0.s, z28.s
    fscale  z1.s, p0/m, z1.s, z29.s
    fscale  z2.s, p0/m, z2.s, z30.s
    fscale  z3.s, p0/m, z3.s, z31.s

    // z0-z3 = exp(-x)
    // 1 + exp(-x) → z0-z3
    fadd    z0.s, z0.s, z8.s
    fadd    z1.s, z1.s, z8.s
    fadd    z2.s, z2.s, z8.s
    fadd    z3.s, z3.s, z8.s

    // sigma = 1 / (1 + exp(-x)) via fdiv (full precision for backward)
    // z18-z21 = sigma
    movprfx z18, z8
    fdiv    z18.s, p0/m, z18.s, z0.s
    movprfx z19, z8
    fdiv    z19.s, p0/m, z19.s, z1.s
    movprfx z20, z8
    fdiv    z20.s, p0/m, z20.s, z2.s
    movprfx z21, z8
    fdiv    z21.s, p0/m, z21.s, z3.s

    // Load dy, compute dx = dy * sigma
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z18.s
    fmul    z1.s, p0/m, z1.s, z19.s
    fmul    z2.s, p0/m, z2.s, z20.s
    fmul    z3.s, p0/m, z3.s, z21.s

    // Store dx
    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lspb_loop

    smstop

.Lspb_done:
    ldp     x19, x20, [sp, #80]
    ldp     d14, d15, [sp, #64]
    ldp     d12, d13, [sp, #48]
    ldp     d10, d11, [sp, #32]
    ldp     d8,  d9,  [sp, #16]
    ldp     x29, x30, [sp], #96
    ret

// Constant pool
.p2align 2
.Lspb_const:
    .long   0x3FB8AA3B  // log2(e)
    .long   0x3F317218  // ln(2)
    .long   0x3F800000  // 1.0
    .long   0x3F000000  // 0.5    = 1/2!
    .long   0x3E2AAAAB  // 1/6    = 1/3!
    .long   0x3D2AAAAB  // 1/24   = 1/4!
    .long   0x3C088889  // 1/120  = 1/5!
    .long   0x3AB60B61  // 1/720  = 1/6!
    .long   0x39500D01  // 1/5040 = 1/7!
    .long   0x42B00000  // 88.0   (clamp_hi)
    .long   0xC2B00000  // -88.0  (clamp_lo)
