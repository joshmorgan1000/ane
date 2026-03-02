// tanh_fp32.s — Element-wise tanh via SME2 streaming SVE
//
// void tanh_fp32(const float *input, float *output, long n)
//
// For |x| >= 0.625: tanh(x) = 2*σ(2x) - 1
// For |x| < 0.625:  tanh(x) ≈ x*(1 - x²*(1/3 - x²*(2/15 - x²*17/315)))
// Blended via predicate to avoid cancellation near zero.
//
// 4-vector parallel processing with single-vector tail for correctness.

.section __TEXT,__text,regular,pure_instructions
.global _tanh_fp32
.p2align 4

_tanh_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    stp     d14,  d15,  [sp, #96]

    cbz     x2, .Ldone

    mov     x19, x0             // save input
    mov     x20, x1             // save output
    mov     x21, x2             // save n

    smstart sm

    ptrue   p0.s

    // Load constants
    adr     x9, .Lconst
    ld1rw   {z16.s}, p0/z, [x9]        // log2(e)
    ld1rw   {z17.s}, p0/z, [x9, #4]    // ln(2)
    ld1rw   {z18.s}, p0/z, [x9, #8]    // 1.0
    ld1rw   {z19.s}, p0/z, [x9, #12]   // c2 = 0.5
    ld1rw   {z20.s}, p0/z, [x9, #16]   // c3 = 1/6
    ld1rw   {z21.s}, p0/z, [x9, #20]   // c4 = 1/24
    ld1rw   {z22.s}, p0/z, [x9, #24]   // c5 = 1/120
    ld1rw   {z23.s}, p0/z, [x9, #28]   // c6 = 1/720
    ld1rw   {z24.s}, p0/z, [x9, #32]   // c7 = 1/5040
    ld1rw   {z25.s}, p0/z, [x9, #36]   // clamp_hi = 88.0
    ld1rw   {z26.s}, p0/z, [x9, #40]   // clamp_lo = -88.0
    ld1rw   {z27.s}, p0/z, [x9, #44]   // 2.0
    ld1rw   {z28.s}, p0/z, [x9, #48]   // threshold = 0.625
    ld1rw   {z29.s}, p0/z, [x9, #52]   // 1/3 (Taylor)
    ld1rw   {z30.s}, p0/z, [x9, #56]   // 2/15 (Taylor)
    ld1rw   {z31.s}, p0/z, [x9, #60]   // 17/315 (Taylor)

    // Compute aligned count for 4-vector processing
    cntw    x10                 // VL (elements per vector)
    lsl     x11, x10, #2        // 4*VL
    udiv    x12, x21, x11
    mul     x22, x12, x11       // aligned count

    mov     x8, #0
    cmp     x8, x22
    b.ge    .Ltail

    ptrue   pn9.s               // predicate for 4-vector loads

    //==========================================================================
    // Main loop: process 4 vectors at a time
    //==========================================================================
.Lloop:
    // Load 4 vectors
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Save original x for Taylor path and blend check
    mov     z4.d, z0.d
    mov     z5.d, z1.d
    mov     z6.d, z2.d
    mov     z7.d, z3.d

    //--------------------------------------------------------------------------
    // Taylor path: tanh(x) ≈ x*(1 - x²*(1/3 - x²*(2/15 - x²*17/315)))
    // Computed in z8-z11
    //--------------------------------------------------------------------------
    // x² in z8-z11
    fmul    z8.s, z4.s, z4.s
    fmul    z9.s, z5.s, z5.s
    fmul    z10.s, z6.s, z6.s
    fmul    z11.s, z7.s, z7.s

    // Step 1: tmp = 17/315 * x²
    fmul    z12.s, z31.s, z8.s
    fmul    z13.s, z31.s, z9.s
    fmul    z14.s, z31.s, z10.s
    fmul    z15.s, z31.s, z11.s

    // Step 2: tmp = 2/15 - tmp
    fsub    z12.s, z30.s, z12.s
    fsub    z13.s, z30.s, z13.s
    fsub    z14.s, z30.s, z14.s
    fsub    z15.s, z30.s, z15.s

    // Step 3: tmp = tmp * x²
    fmul    z12.s, z12.s, z8.s
    fmul    z13.s, z13.s, z9.s
    fmul    z14.s, z14.s, z10.s
    fmul    z15.s, z15.s, z11.s

    // Step 4: tmp = 1/3 - tmp
    fsub    z12.s, z29.s, z12.s
    fsub    z13.s, z29.s, z13.s
    fsub    z14.s, z29.s, z14.s
    fsub    z15.s, z29.s, z15.s

    // Step 5: tmp = tmp * x²
    fmul    z12.s, z12.s, z8.s
    fmul    z13.s, z13.s, z9.s
    fmul    z14.s, z14.s, z10.s
    fmul    z15.s, z15.s, z11.s

    // Step 6: tmp = 1 - tmp
    fsub    z12.s, z18.s, z12.s
    fsub    z13.s, z18.s, z13.s
    fsub    z14.s, z18.s, z14.s
    fsub    z15.s, z18.s, z15.s

    // Step 7: taylor = x * tmp (stored in z8-z11, reusing x² space)
    fmul    z8.s, z4.s, z12.s
    fmul    z9.s, z5.s, z13.s
    fmul    z10.s, z6.s, z14.s
    fmul    z11.s, z7.s, z15.s

    //--------------------------------------------------------------------------
    // Exp path: tanh(x) = 2*σ(2x) - 1
    // σ(2x) = 1/(1 + exp(-2x))
    //--------------------------------------------------------------------------
    // Compute -2x (clamped)
    fmul    z0.s, z4.s, z27.s          // 2x
    fmul    z1.s, z5.s, z27.s
    fmul    z2.s, z6.s, z27.s
    fmul    z3.s, z7.s, z27.s

    fneg    z0.s, p0/m, z0.s           // -2x
    fneg    z1.s, p0/m, z1.s
    fneg    z2.s, p0/m, z2.s
    fneg    z3.s, p0/m, z3.s

    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s
    fmax    z0.s, p0/m, z0.s, z26.s
    fmax    z1.s, p0/m, z1.s, z26.s
    fmax    z2.s, p0/m, z2.s, z26.s
    fmax    z3.s, p0/m, z3.s, z26.s

    // Range reduce: n = round(x * log2(e))
    fmul    z12.s, z0.s, z16.s
    fmul    z13.s, z1.s, z16.s
    fmul    z14.s, z2.s, z16.s
    fmul    z15.s, z3.s, z16.s
    frintn  z12.s, p0/m, z12.s
    frintn  z13.s, p0/m, z13.s
    frintn  z14.s, p0/m, z14.s
    frintn  z15.s, p0/m, z15.s

    // r = x - n * ln(2)
    movprfx z0, z0
    fmls    z0.s, p0/m, z12.s, z17.s
    movprfx z1, z1
    fmls    z1.s, p0/m, z13.s, z17.s
    movprfx z2, z2
    fmls    z2.s, p0/m, z14.s, z17.s
    movprfx z3, z3
    fmls    z3.s, p0/m, z15.s, z17.s

    // Swap: save r to stack-like registers, use z0-z3 for polynomial
    // Actually, let me use different registers for the polynomial
    // Save r in some temp space... we're running low on registers

    // Polynomial evaluation using Horner (need to be clever with registers)
    // p = c7; for each ci: p = p*r + ci

    // Save r values (currently in z0-z3)
    // We need: r (z0-z3), n (z12-z15), and result space
    // Let's compute polynomial in place

    // Initialize polynomial with c7
    mov     z4.d, z24.d         // p0 = c7
    mov     z5.d, z24.d         // p1 = c7
    mov     z6.d, z24.d         // p2 = c7
    mov     z7.d, z24.d         // p3 = c7

    // p = p*r + c6
    fmad    z4.s, p0/m, z0.s, z23.s
    fmad    z5.s, p0/m, z1.s, z23.s
    fmad    z6.s, p0/m, z2.s, z23.s
    fmad    z7.s, p0/m, z3.s, z23.s

    // p = p*r + c5
    fmad    z4.s, p0/m, z0.s, z22.s
    fmad    z5.s, p0/m, z1.s, z22.s
    fmad    z6.s, p0/m, z2.s, z22.s
    fmad    z7.s, p0/m, z3.s, z22.s

    // p = p*r + c4
    fmad    z4.s, p0/m, z0.s, z21.s
    fmad    z5.s, p0/m, z1.s, z21.s
    fmad    z6.s, p0/m, z2.s, z21.s
    fmad    z7.s, p0/m, z3.s, z21.s

    // p = p*r + c3
    fmad    z4.s, p0/m, z0.s, z20.s
    fmad    z5.s, p0/m, z1.s, z20.s
    fmad    z6.s, p0/m, z2.s, z20.s
    fmad    z7.s, p0/m, z3.s, z20.s

    // p = p*r + c2
    fmad    z4.s, p0/m, z0.s, z19.s
    fmad    z5.s, p0/m, z1.s, z19.s
    fmad    z6.s, p0/m, z2.s, z19.s
    fmad    z7.s, p0/m, z3.s, z19.s

    // p = p*r + 1.0
    fmad    z4.s, p0/m, z0.s, z18.s
    fmad    z5.s, p0/m, z1.s, z18.s
    fmad    z6.s, p0/m, z2.s, z18.s
    fmad    z7.s, p0/m, z3.s, z18.s

    // p = p*r + 1.0
    fmad    z4.s, p0/m, z0.s, z18.s
    fmad    z5.s, p0/m, z1.s, z18.s
    fmad    z6.s, p0/m, z2.s, z18.s
    fmad    z7.s, p0/m, z3.s, z18.s

    // Scale by 2^n
    fcvtzs  z12.s, p0/m, z12.s
    fcvtzs  z13.s, p0/m, z13.s
    fcvtzs  z14.s, p0/m, z14.s
    fcvtzs  z15.s, p0/m, z15.s
    fscale  z4.s, p0/m, z4.s, z12.s
    fscale  z5.s, p0/m, z5.s, z13.s
    fscale  z6.s, p0/m, z6.s, z14.s
    fscale  z7.s, p0/m, z7.s, z15.s

    // Now z4-z7 = exp(-2x)
    // σ(2x) = 1 / (1 + exp(-2x))
    fadd    z4.s, z4.s, z18.s
    fadd    z5.s, z5.s, z18.s
    fadd    z6.s, z6.s, z18.s
    fadd    z7.s, z7.s, z18.s

    movprfx z0, z18
    fdiv    z0.s, p0/m, z0.s, z4.s
    movprfx z1, z18
    fdiv    z1.s, p0/m, z1.s, z5.s
    movprfx z2, z18
    fdiv    z2.s, p0/m, z2.s, z6.s
    movprfx z3, z18
    fdiv    z3.s, p0/m, z3.s, z7.s

    // tanh = 2*σ(2x) - 1
    fmul    z0.s, z0.s, z27.s
    fmul    z1.s, z1.s, z27.s
    fmul    z2.s, z2.s, z27.s
    fmul    z3.s, z3.s, z27.s
    fsub    z0.s, z0.s, z18.s
    fsub    z1.s, z1.s, z18.s
    fsub    z2.s, z2.s, z18.s
    fsub    z3.s, z3.s, z18.s

    //--------------------------------------------------------------------------
    // Blend: use Taylor where |x| < threshold
    // Original x was saved but we overwrote it... need to reload
    //--------------------------------------------------------------------------
    // Reload original x to compute |x| for blend
    ld1w    {z4.s-z7.s}, pn9/z, [x19, x8, lsl #2]

    fabs    z4.s, p0/m, z4.s
    fabs    z5.s, p0/m, z5.s
    fabs    z6.s, p0/m, z6.s
    fabs    z7.s, p0/m, z7.s

    // |x| < threshold?
    fcmlt   p1.s, p0/z, z4.s, z28.s
    fcmlt   p2.s, p0/z, z5.s, z28.s
    fcmlt   p3.s, p0/z, z6.s, z28.s
    fcmlt   p4.s, p0/z, z7.s, z28.s

    // Blend: use Taylor (z8-z11) where predicate is true, else exp (z0-z3)
    mov     z0.s, p1/m, z8.s
    mov     z1.s, p2/m, z9.s
    mov     z2.s, p3/m, z10.s
    mov     z3.s, p4/m, z11.s

    // Store results
    st1w    {z0.s-z3.s}, pn9, [x20, x8, lsl #2]

    incw    x8, all, mul #4
    cmp     x8, x22
    b.lt    .Lloop

    //==========================================================================
    // Tail: process remaining elements one vector at a time
    //==========================================================================
.Ltail:
    whilelt p1.s, x8, x21
    b.none  .Lend

.Ltail_loop:
    ld1w    {z0.s}, p1/z, [x19, x8, lsl #2]
    mov     z5.d, z0.d                  // save x

    // --- Taylor path ---
    fmul    z8.s, z0.s, z0.s           // x²

    fmul    z9.s, z31.s, z8.s          // 17/315 * x²
    fsub    z9.s, z30.s, z9.s          // 2/15 - ...
    fmul    z9.s, z9.s, z8.s           // * x²
    fsub    z9.s, z29.s, z9.s          // 1/3 - ...
    fmul    z9.s, z9.s, z8.s           // * x²
    fsub    z9.s, z18.s, z9.s          // 1 - ...
    fmul    z9.s, z0.s, z9.s           // x * ...  = Taylor tanh

    // --- Exp path ---
    fmul    z0.s, z0.s, z27.s          // 2x
    fneg    z0.s, p0/m, z0.s           // -2x
    fmin    z0.s, p0/m, z0.s, z25.s
    fmax    z0.s, p0/m, z0.s, z26.s

    fmul    z1.s, z0.s, z16.s
    frintn  z1.s, p0/m, z1.s
    movprfx z2, z0
    fmls    z2.s, p0/m, z1.s, z17.s

    mov     z3.d, z24.d
    fmad    z3.s, p0/m, z2.s, z23.s
    fmad    z3.s, p0/m, z2.s, z22.s
    fmad    z3.s, p0/m, z2.s, z21.s
    fmad    z3.s, p0/m, z2.s, z20.s
    fmad    z3.s, p0/m, z2.s, z19.s
    fmad    z3.s, p0/m, z2.s, z18.s
    fmad    z3.s, p0/m, z2.s, z18.s

    fcvtzs  z1.s, p0/m, z1.s
    fscale  z3.s, p0/m, z3.s, z1.s

    fadd    z3.s, z3.s, z18.s
    movprfx z4, z18
    fdiv    z4.s, p0/m, z4.s, z3.s

    fmul    z4.s, z4.s, z27.s
    fsub    z4.s, z4.s, z18.s

    // Blend
    mov     z6.d, z5.d
    fabs    z6.s, p0/m, z6.s
    fcmlt   p2.s, p0/z, z6.s, z28.s
    mov     z4.s, p2/m, z9.s

    st1w    {z4.s}, p1, [x20, x8, lsl #2]
    incw    x8
    whilelt p1.s, x8, x21
    b.first .Ltail_loop

.Lend:
    smstop

.Ldone:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     d8,  d9,  [sp, #48]
    ldp     d10, d11, [sp, #64]
    ldp     d12, d13, [sp, #80]
    ldp     d14,  d15,  [sp, #96]
    ldp     x29, x30, [sp], #112
    ret

.p2align 2
.Lconst:
    .long   0x3FB8AA3B  // log2(e)
    .long   0x3F317218  // ln(2)
    .long   0x3F800000  // 1.0
    .long   0x3F000000  // 0.5          (c2)
    .long   0x3E2AAAAB  // 1/6          (c3)
    .long   0x3D2AAAAB  // 1/24         (c4)
    .long   0x3C088889  // 1/120        (c5)
    .long   0x3AB60B61  // 1/720        (c6)
    .long   0x39500D01  // 1/5040       (c7)
    .long   0x42B00000  // 88.0
    .long   0xC2B00000  // -88.0
    .long   0x40000000  // 2.0
    .long   0x3E800000  // 0.25 (threshold)
    .float  0.33333334  // 1/3  (Taylor)
    .float  0.13333334  // 2/15 (Taylor)
    .float  0.05396825  // 17/315 (Taylor)
