// mish_backward_fp32.s — Mish backward pass via SME2 with ZA accumulation
//
// void mish_backward_fp32(const float* dy, const float* x, float* dx, long n)
// AAPCS: x0=dy, x1=x, x2=dx, x3=n
//
// Math:
//   sp  = softplus(x)   = log(1 + exp(x))   [clamped: if x > 20 then sp = x]
//   t   = tanh(sp)
//   sig = sigmoid(x)    = 1 / (1 + exp(-x))
//   dx  = dy * (t + x * (1 - t*t) * sig)
//
// Key insight: we only need to compute exp(x) once.
//   - From exp(x): s = 1 + exp(x)
//   - tanh(sp) = (s²-1)/(s²+1)        [same formula as mish_fp32.s forward]
//   - sig = exp(x) / (1 + exp(x))     = exp(x) / s  [no second exp needed]
//   - sech²(sp) = 1 - t²              [tanh derivative]
//
// Register map during computation:
//   z0-z3   : r (reduced exponent argument), then dx accumulation
//   z4-z7   : saved x (original input)
//   z8      : 1.0 constant
//   z9      : 0.5 constant
//   z10-z14 : Taylor coefficients (1/6, 1/24, 1/120, 1/720, 1/5040)
//   z15     : 20.0 clamp upper bound
//   z16     : log2(e)
//   z17     : ln(2)
//   z18-z21 : s = 1 + exp(x), then sig, then reused
//   z22-z25 : s², then tanh(sp) = t, then reused
//   z26     : -88.0 clamp lower bound
//   z27     : scratch
//   z28-z31 : n_val (exponent for fscale)
//
// Uses smstart (full ZA) for fmla vgx4 polynomial accumulation.
// Processes 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _mish_backward_fp32
.p2align 4

_mish_backward_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]
    stp     x19, x20, [sp, #80]

    cbz     x3, .Lmishb_done

    // Save pointers before smstart
    mov     x19, x0                 // dy
    mov     x20, x1                 // x

    // Enter streaming mode with ZA access
    smstart

    ptrue   p0.s
    mov     w11, #0                 // ZA vector select register

    // Load constants
    adr     x9, .Lmishb_const
    ld1rw   {z16.s}, p0/z, [x9]        // log2(e)
    ld1rw   {z17.s}, p0/z, [x9, #4]    // ln(2)
    ld1rw   {z8.s},  p0/z, [x9, #8]    // 1.0
    ld1rw   {z9.s},  p0/z, [x9, #12]   // 0.5
    ld1rw   {z10.s}, p0/z, [x9, #16]   // 1/6
    ld1rw   {z11.s}, p0/z, [x9, #20]   // 1/24
    ld1rw   {z12.s}, p0/z, [x9, #24]   // 1/120
    ld1rw   {z13.s}, p0/z, [x9, #28]   // 1/720
    ld1rw   {z14.s}, p0/z, [x9, #32]   // 1/5040
    ld1rw   {z15.s}, p0/z, [x9, #36]   // 20.0  (softplus clamp — above this sp~x)
    ld1rw   {z26.s}, p0/z, [x9, #40]   // -88.0 (exp underflow guard)

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lmishb_loop:
    // Load x into z0-z3; save copy in z4-z7 for final formula
    ld1w    {z0.s-z3.s}, pn9/z, [x20, x8, lsl #2]
    mov     z4.d, z0.d
    mov     z5.d, z1.d
    mov     z6.d, z2.d
    mov     z7.d, z3.d

    // Clamp x to [-88, 20] for safe exp(x) (upper 20 prevents s² overflow)
    fclamp  {z0.s-z3.s}, z26.s, z15.s

    // Range reduction: n_val = round(x * log2(e))
    fmul    z28.s, z0.s, z16.s
    fmul    z29.s, z1.s, z16.s
    fmul    z30.s, z2.s, z16.s
    fmul    z31.s, z3.s, z16.s
    frintn  z28.s, p0/m, z28.s
    frintn  z29.s, p0/m, z29.s
    frintn  z30.s, p0/m, z30.s
    frintn  z31.s, p0/m, z31.s

    // r = x - n_val * ln(2)
    fmls    z0.s, p0/m, z28.s, z17.s
    fmls    z1.s, p0/m, z29.s, z17.s
    fmls    z2.s, p0/m, z30.s, z17.s
    fmls    z3.s, p0/m, z31.s, z17.s

    // --- 7-term Taylor polynomial exp(r) via ZA accumulation ---
    zero    {za}

    // c1 * r  (c1 = 1.0)
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z8.s

    // r^2 → z18-z21
    fmul    z18.s, z0.s, z0.s
    fmul    z19.s, z1.s, z1.s
    fmul    z20.s, z2.s, z2.s
    fmul    z21.s, z3.s, z3.s
    // c2 * r^2
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z9.s

    // r^3 = r^2 * r → z22-z25
    fmul    z22.s, z18.s, z0.s
    fmul    z23.s, z19.s, z1.s
    fmul    z24.s, z20.s, z2.s
    fmul    z25.s, z21.s, z3.s
    // c3 * r^3
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z10.s

    // r^4 = r^2 * r^2 → z22-z25
    fmul    z22.s, z18.s, z18.s
    fmul    z23.s, z19.s, z19.s
    fmul    z24.s, z20.s, z20.s
    fmul    z25.s, z21.s, z21.s
    // c4 * r^4
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z11.s

    // r^5 = r^4 * r → z22-z25
    fmul    z22.s, z22.s, z0.s
    fmul    z23.s, z23.s, z1.s
    fmul    z24.s, z24.s, z2.s
    fmul    z25.s, z25.s, z3.s
    // c5 * r^5
    fmla    za.s[w11, 0, vgx4], {z22.s-z25.s}, z12.s

    // r^6 = r^5 * r → z18-z21 (clobbers r^2)
    fmul    z18.s, z22.s, z18.s
    fmul    z19.s, z23.s, z19.s
    fmul    z20.s, z24.s, z20.s
    fmul    z21.s, z25.s, z21.s
    // c6 * r^6
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z13.s

    // r^7 = r^6 * r → z18-z21
    fmul    z18.s, z18.s, z0.s
    fmul    z19.s, z19.s, z1.s
    fmul    z20.s, z20.s, z2.s
    fmul    z21.s, z21.s, z3.s
    // c7 * r^7
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z14.s

    // Extract polynomial, add c0 = 1.0 → exp(r)
    mova    {z0.s-z3.s}, za.s[w11, 0, vgx4]
    fadd    z0.s, z0.s, z8.s
    fadd    z1.s, z1.s, z8.s
    fadd    z2.s, z2.s, z8.s
    fadd    z3.s, z3.s, z8.s

    // Scale by 2^n_val → exp(x)
    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s
    fscale  z0.s, p0/m, z0.s, z28.s
    fscale  z1.s, p0/m, z1.s, z29.s
    fscale  z2.s, p0/m, z2.s, z30.s
    fscale  z3.s, p0/m, z3.s, z31.s

    // z0-z3 = exp(x)
    // s = 1 + exp(x) → z18-z21
    fadd    z18.s, z0.s, z8.s
    fadd    z19.s, z1.s, z8.s
    fadd    z20.s, z2.s, z8.s
    fadd    z21.s, z3.s, z8.s

    // sig = exp(x) / s = exp(x) / (1 + exp(x)) → z0-z3 (reuse, via fdiv)
    fdiv    z0.s, p0/m, z0.s, z18.s
    fdiv    z1.s, p0/m, z1.s, z19.s
    fdiv    z2.s, p0/m, z2.s, z20.s
    fdiv    z3.s, p0/m, z3.s, z21.s
    // z0-z3 = sig

    // s² → z22-z25
    fmul    z22.s, z18.s, z18.s
    fmul    z23.s, z19.s, z19.s
    fmul    z24.s, z20.s, z20.s
    fmul    z25.s, z21.s, z21.s

    // t = tanh(sp) = (s²-1)/(s²+1) → z18-z21
    // Numerator: s²-1
    fsub    z18.s, z22.s, z8.s
    fsub    z19.s, z23.s, z8.s
    fsub    z20.s, z24.s, z8.s
    fsub    z21.s, z25.s, z8.s
    // Denominator: s²+1
    fadd    z22.s, z22.s, z8.s
    fadd    z23.s, z23.s, z8.s
    fadd    z24.s, z24.s, z8.s
    fadd    z25.s, z25.s, z8.s
    // t = (s²-1)/(s²+1)
    fdiv    z18.s, p0/m, z18.s, z22.s
    fdiv    z19.s, p0/m, z19.s, z23.s
    fdiv    z20.s, p0/m, z20.s, z24.s
    fdiv    z21.s, p0/m, z21.s, z25.s
    // z18-z21 = t = tanh(sp)

    // sech²(sp) = 1 - t² → z22-z25
    // z18-z21 = t; compute t² → negate → add 1.0 to get (1 - t²)
    fmul    z22.s, z18.s, z18.s     // t0²
    fmul    z23.s, z19.s, z19.s     // t1²
    fmul    z24.s, z20.s, z20.s     // t2²
    fmul    z25.s, z21.s, z21.s     // t3²
    fneg    z22.s, p0/m, z22.s      // -t0²
    fneg    z23.s, p0/m, z23.s      // -t1²
    fneg    z24.s, p0/m, z24.s      // -t2²
    fneg    z25.s, p0/m, z25.s      // -t3²
    fadd    z22.s, z22.s, z8.s      // 1 - t0²
    fadd    z23.s, z23.s, z8.s      // 1 - t1²
    fadd    z24.s, z24.s, z8.s      // 1 - t2²
    fadd    z25.s, z25.s, z8.s      // 1 - t3²
    // z22-z25 = sech²(sp) = 1 - t²

    // Compute: x * sech²(sp) * sig → z22-z25
    // Step 1: sech² * sig
    fmul    z22.s, p0/m, z22.s, z0.s
    fmul    z23.s, p0/m, z23.s, z1.s
    fmul    z24.s, p0/m, z24.s, z2.s
    fmul    z25.s, p0/m, z25.s, z3.s
    // Step 2: x * (sech² * sig) — x is in z4-z7
    fmul    z22.s, p0/m, z22.s, z4.s
    fmul    z23.s, p0/m, z23.s, z5.s
    fmul    z24.s, p0/m, z24.s, z6.s
    fmul    z25.s, p0/m, z25.s, z7.s

    // t + x * (1 - t²) * sig → z18-z21
    fadd    z18.s, z18.s, z22.s
    fadd    z19.s, z19.s, z23.s
    fadd    z20.s, z20.s, z24.s
    fadd    z21.s, z21.s, z25.s

    // Load dy and compute dx = dy * (t + x*(1-t²)*sig)
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z18.s
    fmul    z1.s, p0/m, z1.s, z19.s
    fmul    z2.s, p0/m, z2.s, z20.s
    fmul    z3.s, p0/m, z3.s, z21.s

    // Store dx
    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lmishb_loop

    smstop

.Lmishb_done:
    ldp     x19, x20, [sp, #80]
    ldp     d14, d15, [sp, #64]
    ldp     d12, d13, [sp, #48]
    ldp     d10, d11, [sp, #32]
    ldp     d8,  d9,  [sp, #16]
    ldp     x29, x30, [sp], #96
    ret

// Constant pool
.p2align 2
.Lmishb_const:
    .long   0x3FB8AA3B  // log2(e)
    .long   0x3F317218  // ln(2)
    .long   0x3F800000  // 1.0
    .long   0x3F000000  // 0.5    = 1/2!
    .long   0x3E2AAAAB  // 1/6    = 1/3!
    .long   0x3D2AAAAB  // 1/24   = 1/4!
    .long   0x3C088889  // 1/120  = 1/5!
    .long   0x3AB60B61  // 1/720  = 1/6!
    .long   0x39500D01  // 1/5040 = 1/7!
    .long   0x41A00000  // 20.0   (softplus upper clamp, prevents s² overflow)
    .long   0xC2B00000  // -88.0  (exp underflow guard)
