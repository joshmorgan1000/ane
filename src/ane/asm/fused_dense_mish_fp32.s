// fused_dense_mish_fp32.s -- Fused dense (matvec) + bias + Mish activation
//
// Computes: temp = W @ x + bias, then mish(temp)
//   mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
//
// For numerical stability:
//   softplus(x) = x >= 0 ? x + ln(1 + exp(-x)) : ln(1 + exp(x))
//   But for SME2, we use: softplus(x) = max(x, 0) + ln(1 + exp(-|x|))
//   which is equivalent and avoids branch.
//   Then tanh(s) = (exp(2s)-1)/(exp(2s)+1) = 1 - 2/(exp(2s)+1)
//   Mish = x * tanh(softplus(x))
//
// void fused_dense_mish_fp32(const float* W, int m, int n,
//                             const float* x, const float* bias, float* out)
//
// AAPCS64: x0=W, x1=m, x2=n, x3=x, x4=bias, x5=out

.section __TEXT,__text,regular,pure_instructions
.global _fused_dense_mish_fp32
.p2align 4

_fused_dense_mish_fp32:
    stp     x29, x30, [sp, #-160]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8,  d9,  [sp, #96]
    stp     d10, d11, [sp, #112]
    stp     d12, d13, [sp, #128]
    stp     d14, d15, [sp, #144]

    cbz     x1, .Lfdm_done
    cbz     x2, .Lfdm_done

    // Save arguments
    mov     x19, x0                 // W
    mov     x20, x1                 // m
    mov     x21, x2                 // n
    mov     x22, x3                 // x (input vector)
    mov     x23, x4                 // bias
    mov     x24, x5                 // out

    // Allocate temp buffer: m * 4 bytes
    lsl     x0, x20, #2
    bl      _malloc
    mov     x25, x0                 // temp buffer

    // ================================================================
    // Phase 1: Compute temp = W @ x + bias
    // ================================================================

    smstart                         // streaming SVE + ZA (needed for polynomial exp)
    ptrue   p0.s

    mov     x8, #0
    lsl     x9, x21, #2

.Lfdm_matvec_loop:
    cmp     x8, x20
    b.ge    .Lfdm_matvec_done

    mul     x10, x8, x9
    add     x10, x19, x10

    mov     z28.s, #0

    mov     x11, #0
    whilelt p1.s, x11, x21

.Lfdm_dot_loop:
    ld1w    {z0.s}, p1/z, [x10, x11, lsl #2]
    ld1w    {z1.s}, p1/z, [x22, x11, lsl #2]
    fmla    z28.s, p1/m, z0.s, z1.s

    incw    x11
    whilelt p1.s, x11, x21
    b.first .Lfdm_dot_loop

    faddv   s2, p0, z28.s

    // Add bias
    ldr     s3, [x23, x8, lsl #2]
    fadd    s2, s2, s3

    str     s2, [x25, x8, lsl #2]

    add     x8, x8, #1
    b       .Lfdm_matvec_loop

.Lfdm_matvec_done:
    // ================================================================
    // Phase 2: Apply Mish = x * tanh(softplus(x))
    // We compute softplus then tanh then multiply by x.
    //
    // Approach: Use sigmoid-based identity:
    //   mish(x) = x * tanh(softplus(x))
    //           = x * tanh(ln(1 + exp(x)))
    //
    // For efficiency, compute exp(-|x|) using the same polynomial
    // as sigmoid, then:
    //   softplus(x) = max(x,0) + ln(1 + exp(-|x|))
    //   We approximate ln(1+y) ~ y - y^2/2 + y^3/3 for small y,
    //   but it's simpler to compute tanh(softplus) directly via:
    //   tanh(s) = 1 - 2/(1+exp(2s))
    //
    // Simplest correct approach for SME2:
    //   1. Compute exp(-|x|) via standard exp polynomial
    //   2. softplus = max(x,0) + ln(1 + exp(-|x|))
    //      But ln(1+y) for y = exp(-|x|) is expensive.
    //
    // Alternative: Use the identity mish(x) = x * (1 - 2/(1+exp(2*softplus(x))))
    //   softplus(x) ~ x for large x, softplus(x) ~ exp(x) for very negative x
    //
    // Pragmatic approach: Compute sigmoid(x) and use:
    //   mish(x) = x * tanh(softplus(x))
    //
    // We use: omega = 4*sigmoid(x)*(x+1) + exp(x)*(exp(x)-4) all / (...)
    // Actually, simplest: mish(x) = x * (exp(2*softplus(x))-1)/(exp(2*softplus(x))+1)
    //
    // MOST PRACTICAL: Use the relation mish(x) = x * tanh(softplus(x))
    //   where softplus(x) = log(1+exp(x))
    //   and we approximate both via polynomial exp.
    //
    // Implementation:
    //   1. Compute exp(x) via standard polynomial (clamp x to [-88,88])
    //   2. softplus = ln(1 + exp(x)) = ln(exp(max(x,0)) * (1 + exp(-|x|)))
    //      = max(x,0) + ln(1+exp(-|x|))
    //   3. For ln(1+y) where y=exp(-|x|) in [0,1]:
    //      Use: ln(1+y) = y - y^2/2 + y^3/3 - y^4/4 + y^5/5
    //      This converges well for y in [0,1].
    //   4. tanh(s) = 2*sigmoid(2s) - 1
    //   5. mish = x * tanh(s)
    //
    // For simplicity and correctness, use a two-pass approach:
    //   Pass A: Compute softplus via exp then log1p
    //   Pass B: Compute tanh(softplus) via exp/sigmoid, multiply by x
    //
    // Even simpler: Use mish(x) = x * tanh(softplus(x))
    //   = x * tanh(ln(1+exp(x)))
    //   = x * (2*sigmoid(2*ln(1+exp(x))) - 1)
    //
    // But 2*ln(1+exp(x)) = ln((1+exp(x))^2)... getting complex.
    //
    // FINAL SIMPLE APPROACH:
    //   We compute sigmoid(x), then:
    //   mish(x) = x * sigmoid(x) * (1 + x*(1-sigmoid(x)))
    //   This is NOT the exact mish formula but a known approximation.
    //
    // ACTUALLY correct formula uses: mish(x) = x * tanh(softplus(x))
    //   = x * (exp(x)+1)^2-1) / ((exp(x)+1)^2+1)
    //   Numerator: (1+exp(x))^2 - 1 = exp(2x) + 2*exp(x)
    //   Denominator: (1+exp(x))^2 + 1 = exp(2x) + 2*exp(x) + 2
    //   So mish(x) = x * (exp(2x) + 2*exp(x)) / (exp(2x) + 2*exp(x) + 2)
    //   Let e = exp(x), then:
    //   mish(x) = x * e*(e+2) / (e*(e+2) + 2)
    //           = x * e*(e+2) / (e^2 + 2*e + 2)
    //
    // This is clean! We only need exp(x), then simple arithmetic.
    // ================================================================

    // Load exp constants
    adr     x9, .Lfdm_const
    ld1rw   {z8.s}, p0/z, [x9]         // log2(e)
    ld1rw   {z9.s}, p0/z, [x9, #4]     // ln(2)
    ld1rw   {z10.s}, p0/z, [x9, #8]    // 1.0
    ld1rw   {z11.s}, p0/z, [x9, #12]   // 0.5 = 1/2!
    ld1rw   {z12.s}, p0/z, [x9, #16]   // 1/6 = 1/3!
    ld1rw   {z13.s}, p0/z, [x9, #20]   // 1/24 = 1/4!
    ld1rw   {z14.s}, p0/z, [x9, #24]   // 1/120 = 1/5!
    ld1rw   {z15.s}, p0/z, [x9, #28]   // 1/720 = 1/6!
    ld1rw   {z27.s}, p0/z, [x9, #32]   // 88.0
    ld1rw   {z26.s}, p0/z, [x9, #36]   // -88.0

    mov     w11, #0                 // ZA row select

    mov     x8, #0
    whilelt pn9.s, x8, x20, vlx4

.Lfdm_mish_loop:
    // Load temp (x values)
    ld1w    {z0.s-z3.s}, pn9/z, [x25, x8, lsl #2]

    // Save x for final multiply
    mov     z16.d, z0.d
    mov     z17.d, z1.d
    mov     z18.d, z2.d
    mov     z19.d, z3.d

    // Clamp x to [-88, 88] for exp
    fmin    z0.s, p0/m, z0.s, z27.s
    fmax    z0.s, p0/m, z0.s, z26.s
    fmin    z1.s, p0/m, z1.s, z27.s
    fmax    z1.s, p0/m, z1.s, z26.s
    fmin    z2.s, p0/m, z2.s, z27.s
    fmax    z2.s, p0/m, z2.s, z26.s
    fmin    z3.s, p0/m, z3.s, z27.s
    fmax    z3.s, p0/m, z3.s, z26.s

    // Range reduce: n_val = round(x * log2(e))
    fmul    z20.s, z0.s, z8.s
    fmul    z21.s, z1.s, z8.s
    fmul    z22.s, z2.s, z8.s
    fmul    z23.s, z3.s, z8.s
    frintn  z20.s, p0/m, z20.s
    frintn  z21.s, p0/m, z21.s
    frintn  z22.s, p0/m, z22.s
    frintn  z23.s, p0/m, z23.s

    // r = x - n_val * ln(2)
    movprfx z4, z0
    fmls    z4.s, p0/m, z20.s, z9.s
    movprfx z5, z1
    fmls    z5.s, p0/m, z21.s, z9.s
    movprfx z6, z2
    fmls    z6.s, p0/m, z22.s, z9.s
    movprfx z7, z3
    fmls    z7.s, p0/m, z23.s, z9.s

    // Polynomial exp(r) via FMLA VGx4 with ZA accumulation
    zero    {za}

    // c1*r
    fmla    za.s[w11, 0, vgx4], {z4.s-z7.s}, z10.s

    // r^2
    fmul    z0.s, z4.s, z4.s
    fmul    z1.s, z5.s, z5.s
    fmul    z2.s, z6.s, z6.s
    fmul    z3.s, z7.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z11.s

    // r^3
    fmul    z0.s, z0.s, z4.s
    fmul    z1.s, z1.s, z5.s
    fmul    z2.s, z2.s, z6.s
    fmul    z3.s, z3.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z12.s

    // r^4
    fmul    z0.s, z0.s, z4.s
    fmul    z1.s, z1.s, z5.s
    fmul    z2.s, z2.s, z6.s
    fmul    z3.s, z3.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z13.s

    // r^5
    fmul    z0.s, z0.s, z4.s
    fmul    z1.s, z1.s, z5.s
    fmul    z2.s, z2.s, z6.s
    fmul    z3.s, z3.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z14.s

    // r^6
    fmul    z0.s, z0.s, z4.s
    fmul    z1.s, z1.s, z5.s
    fmul    z2.s, z2.s, z6.s
    fmul    z3.s, z3.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z15.s

    // Extract and add c0=1.0
    mova    {z0.s-z3.s}, za.s[w11, 0, vgx4]
    fadd    z0.s, z0.s, z10.s
    fadd    z1.s, z1.s, z10.s
    fadd    z2.s, z2.s, z10.s
    fadd    z3.s, z3.s, z10.s

    // Scale by 2^n_val
    fcvtzs  z20.s, p0/m, z20.s
    fcvtzs  z21.s, p0/m, z21.s
    fcvtzs  z22.s, p0/m, z22.s
    fcvtzs  z23.s, p0/m, z23.s
    fscale  z0.s, p0/m, z0.s, z20.s
    fscale  z1.s, p0/m, z1.s, z21.s
    fscale  z2.s, p0/m, z2.s, z22.s
    fscale  z3.s, p0/m, z3.s, z23.s

    // z0-z3 = exp(x) = e
    // mish(x) = x * e*(e+2) / (e^2 + 2*e + 2)
    //         = x * e*(e+2) / (e*(e+2) + 2)

    // Compute e + 2
    adr     x10, .Lfdm_two
    ld1rw   {z24.s}, p0/z, [x10]       // 2.0

    fadd    z4.s, z0.s, z24.s           // e+2 [0]
    fadd    z5.s, z1.s, z24.s           // e+2 [1]
    fadd    z6.s, z2.s, z24.s           // e+2 [2]
    fadd    z7.s, z3.s, z24.s           // e+2 [3]

    // Numerator: e * (e+2)
    fmul    z20.s, z0.s, z4.s
    fmul    z21.s, z1.s, z5.s
    fmul    z22.s, z2.s, z6.s
    fmul    z23.s, z3.s, z7.s

    // Denominator: e*(e+2) + 2
    fadd    z4.s, z20.s, z24.s
    fadd    z5.s, z21.s, z24.s
    fadd    z6.s, z22.s, z24.s
    fadd    z7.s, z23.s, z24.s

    // tanh(softplus) = num / den
    // fdiv requires dest == first source for predicated form, use movprfx
    movprfx z0, z20
    fdiv    z0.s, p0/m, z0.s, z4.s
    movprfx z1, z21
    fdiv    z1.s, p0/m, z1.s, z5.s
    movprfx z2, z22
    fdiv    z2.s, p0/m, z2.s, z6.s
    movprfx z3, z23
    fdiv    z3.s, p0/m, z3.s, z7.s

    // mish = x * tanh(softplus(x))
    fmul    z0.s, p0/m, z0.s, z16.s
    fmul    z1.s, p0/m, z1.s, z17.s
    fmul    z2.s, p0/m, z2.s, z18.s
    fmul    z3.s, p0/m, z3.s, z19.s

    // Store output
    st1w    {z0.s-z3.s}, pn9, [x24, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x20, vlx4
    b.first .Lfdm_mish_loop

    smstop

    // Free temp buffer
    mov     x0, x25
    bl      _free

.Lfdm_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     d8,  d9,  [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    ldp     x29, x30, [sp], #160
    ret

// Constant pool
.p2align 2
.Lfdm_const:
    .long   0x3FB8AA3B  // log2(e) = 1.44269504
    .long   0x3F317218  // ln(2)   = 0.69314718
    .long   0x3F800000  // 1.0
    .long   0x3F000000  // 0.5     = 1/2!
    .long   0x3E2AAAAB  // 1/6     = 1/3!
    .long   0x3D2AAAAB  // 1/24    = 1/4!
    .long   0x3C088889  // 1/120   = 1/5!
    .long   0x3AB60B61  // 1/720   = 1/6!
    .long   0x42B00000  // 88.0
    .long   0xC2B00000  // -88.0

.Lfdm_two:
    .long   0x40000000  // 2.0
