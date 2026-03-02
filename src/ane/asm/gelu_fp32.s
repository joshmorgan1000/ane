// gelu_fp32.s — Element-wise GELU via SME2 with ZA tiles
//
// void gelu_fp32(const float *input, float *output, long n)
//
// Fast GELU approximation: gelu(x) = x * σ(1.702*x)
// where σ(t) = 1/(1+exp(-t))
//
// Uses ZA tiles for the final x * σ(x) element-wise multiply via:
//   FMLA ZA.S[Wv, offs, VGx4], {Zn.S-Zn+3.S}, {Zm.S-Zm+3.S}
//
// Processing: 4 vectors (64 floats on M4) per iteration through ZA accumulate.

.section __TEXT,__text,regular,pure_instructions
.global _gelu_fp32
.p2align 4

_gelu_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8,  d9,  [sp, #32]
    stp     d10, d11, [sp, #48]
    stp     d12, d13, [sp, #64]
    stp     d14,  d15,  [sp, #80]

    cbz     x2, .Ldone

    mov     x19, x0             // save input
    mov     x20, x1             // save output

    smstart                     // Enable BOTH streaming mode AND ZA

    ptrue   p0.s

    adr     x9, .Lconst
    ld1rw   {z16.s}, p0/z, [x9]        // log2(e)
    ld1rw   {z17.s}, p0/z, [x9, #4]    // ln(2)
    ld1rw   {z18.s}, p0/z, [x9, #8]    // 1.0
    ld1rw   {z19.s}, p0/z, [x9, #12]   // c2 = 0.5
    ld1rw   {z20.s}, p0/z, [x9, #16]   // c3
    ld1rw   {z21.s}, p0/z, [x9, #20]   // c4
    ld1rw   {z22.s}, p0/z, [x9, #24]   // c5
    ld1rw   {z23.s}, p0/z, [x9, #28]   // c6
    ld1rw   {z24.s}, p0/z, [x9, #32]   // c7
    ld1rw   {z25.s}, p0/z, [x9, #36]   // clamp_hi
    ld1rw   {z26.s}, p0/z, [x9, #40]   // clamp_lo
    ld1rw   {z27.s}, p0/z, [x9, #44]   // -1.702

    // Compute aligned count for 4-vector processing
    cntw    x10                 // VL (elements per vector)
    lsl     x11, x10, #2        // 4*VL
    udiv    x12, x2, x11
    mul     x13, x12, x11       // aligned count

    mov     x8, #0
    cmp     x8, x13
    b.ge    .Ltail

    ptrue   pn9.s               // predicate for 4-vector loads

    //==========================================================================
    // Main loop: process 4 vectors at a time using ZA for final multiply
    //==========================================================================
.Lloop:
    // Load 4 vectors of input x
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Save original x in z4-z7 for final multiply
    mov     z4.d, z0.d
    mov     z5.d, z1.d
    mov     z6.d, z2.d
    mov     z7.d, z3.d

    // Compute exp(-1.702*x): multiply by -1.702
    fmul    z0.s, z0.s, z27.s
    fmul    z1.s, z1.s, z27.s
    fmul    z2.s, z2.s, z27.s
    fmul    z3.s, z3.s, z27.s

    // Clamp to [-88, 88]
    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s
    fmax    z0.s, p0/m, z0.s, z26.s
    fmax    z1.s, p0/m, z1.s, z26.s
    fmax    z2.s, p0/m, z2.s, z26.s
    fmax    z3.s, p0/m, z3.s, z26.s

    // Range reduce: n = round(x * log2(e))
    fmul    z8.s, z0.s, z16.s
    fmul    z9.s, z1.s, z16.s
    fmul    z10.s, z2.s, z16.s
    fmul    z11.s, z3.s, z16.s
    frintn  z8.s, p0/m, z8.s
    frintn  z9.s, p0/m, z9.s
    frintn  z10.s, p0/m, z10.s
    frintn  z11.s, p0/m, z11.s

    // r = x - n * ln(2) in z12-z15
    movprfx z12, z0
    fmls    z12.s, p0/m, z8.s, z17.s
    movprfx z13, z1
    fmls    z13.s, p0/m, z9.s, z17.s
    movprfx z14, z2
    fmls    z14.s, p0/m, z10.s, z17.s
    movprfx z15, z3
    fmls    z15.s, p0/m, z11.s, z17.s

    // Horner polynomial: p = c7
    mov     z0.d, z24.d
    mov     z1.d, z24.d
    mov     z2.d, z24.d
    mov     z3.d, z24.d

    // p = p*r + c6
    fmad    z0.s, p0/m, z12.s, z23.s
    fmad    z1.s, p0/m, z13.s, z23.s
    fmad    z2.s, p0/m, z14.s, z23.s
    fmad    z3.s, p0/m, z15.s, z23.s

    // p = p*r + c5
    fmad    z0.s, p0/m, z12.s, z22.s
    fmad    z1.s, p0/m, z13.s, z22.s
    fmad    z2.s, p0/m, z14.s, z22.s
    fmad    z3.s, p0/m, z15.s, z22.s

    // p = p*r + c4
    fmad    z0.s, p0/m, z12.s, z21.s
    fmad    z1.s, p0/m, z13.s, z21.s
    fmad    z2.s, p0/m, z14.s, z21.s
    fmad    z3.s, p0/m, z15.s, z21.s

    // p = p*r + c3
    fmad    z0.s, p0/m, z12.s, z20.s
    fmad    z1.s, p0/m, z13.s, z20.s
    fmad    z2.s, p0/m, z14.s, z20.s
    fmad    z3.s, p0/m, z15.s, z20.s

    // p = p*r + c2
    fmad    z0.s, p0/m, z12.s, z19.s
    fmad    z1.s, p0/m, z13.s, z19.s
    fmad    z2.s, p0/m, z14.s, z19.s
    fmad    z3.s, p0/m, z15.s, z19.s

    // p = p*r + 1.0
    fmad    z0.s, p0/m, z12.s, z18.s
    fmad    z1.s, p0/m, z13.s, z18.s
    fmad    z2.s, p0/m, z14.s, z18.s
    fmad    z3.s, p0/m, z15.s, z18.s

    // p = p*r + 1.0 (final term)
    fmad    z0.s, p0/m, z12.s, z18.s
    fmad    z1.s, p0/m, z13.s, z18.s
    fmad    z2.s, p0/m, z14.s, z18.s
    fmad    z3.s, p0/m, z15.s, z18.s

    // Convert n to integer and scale
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s
    fscale  z0.s, p0/m, z0.s, z8.s
    fscale  z1.s, p0/m, z1.s, z9.s
    fscale  z2.s, p0/m, z2.s, z10.s
    fscale  z3.s, p0/m, z3.s, z11.s

    // Now z0-z3 = exp(-1.702*x)
    // σ(1.702*x) = 1 / (1 + exp(-1.702*x))
    fadd    z0.s, z0.s, z18.s
    fadd    z1.s, z1.s, z18.s
    fadd    z2.s, z2.s, z18.s
    fadd    z3.s, z3.s, z18.s

    // Division: σ = 1 / (1 + exp(-1.702*x)) into z8-z11
    movprfx z8, z18
    fdiv    z8.s, p0/m, z8.s, z0.s
    movprfx z9, z18
    fdiv    z9.s, p0/m, z9.s, z1.s
    movprfx z10, z18
    fdiv    z10.s, p0/m, z10.s, z2.s
    movprfx z11, z18
    fdiv    z11.s, p0/m, z11.s, z3.s

    // Now z4-z7 = original x, z8-z11 = σ(1.702*x)
    // Use ZA for element-wise multiply: gelu = x * σ

    // Zero the ZA rows we'll use
    zero    {za}

    // FMLA ZA.S[Wv, offs, VGx4], {z4-z7}, {z8-z11}
    // This does: ZA[row+0] += z4 * z8, ZA[row+16] += z5 * z9, etc. (element-wise)
    // Wv must be W8-W11; use w10 since w8/w9 may conflict with x8 loop counter
    mov     w10, #0
    fmla    za.s[w10, 0, vgx4], {z4.s-z7.s}, {z8.s-z11.s}

    // Move results from ZA back to Z registers
    mova    {z0.s-z3.s}, za.s[w10, 0, vgx4]

    // Store 4 vectors
    st1w    {z0.s-z3.s}, pn9, [x20, x8, lsl #2]

    incw    x8, all, mul #4
    cmp     x8, x13
    b.lt    .Lloop

    //==========================================================================
    // Tail: process remaining elements one vector at a time
    //==========================================================================
.Ltail:
    whilelt p1.s, x8, x2
    b.none  .Lend

.Ltail_loop:
    ld1w    {z0.s}, p1/z, [x19, x8, lsl #2]
    mov     z5.d, z0.d                  // save x

    // Compute exp(-1.702*x)
    fmul    z0.s, z0.s, z27.s           // -1.702*x
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
    fscale  z3.s, p0/m, z3.s, z1.s     // z3 = exp(-1.702*x)

    // σ(1.702*x) = 1 / (1 + exp(-1.702*x))
    fadd    z3.s, z3.s, z18.s
    movprfx z4, z18
    fdiv    z4.s, p0/m, z4.s, z3.s

    // gelu = x * σ(1.702*x) - use regular fmul for tail
    fmul    z4.s, z5.s, z4.s

    st1w    {z4.s}, p1, [x20, x8, lsl #2]
    incw    x8
    whilelt p1.s, x8, x2
    b.first .Ltail_loop

.Lend:
    smstop

.Ldone:
    ldp     x19, x20, [sp, #16]
    ldp     d8,  d9,  [sp, #32]
    ldp     d10, d11, [sp, #48]
    ldp     d12, d13, [sp, #64]
    ldp     d14,  d15,  [sp, #80]
    ldp     x29, x30, [sp], #96
    ret

.p2align 2
.Lconst:
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
    .float  -1.702      // -1.702
