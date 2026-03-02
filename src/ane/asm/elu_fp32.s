// elu_fp32.s — Element-wise ELU (Exponential Linear Unit) activation via SME2 streaming SVE
//
// void elu_fp32(const float *input, float *output, float alpha, long n)
// AAPCS: x0=input, x1=output, s0=alpha, x2=n
//
// Computes: output[i] = x >= 0 ? x : alpha * (exp(x) - 1)
//
// Strategy:
// 1. Load 4 vectors of input
// 2. For lanes where x >= 0: result = x (identity)
// 3. For lanes where x < 0: compute exp(x) - 1, multiply by alpha
// 4. Use sel to merge results based on fcmge predicate
// 5. Clamp negative inputs to [-88, 0] to prevent exp overflow
//
// Processing: 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _elu_fp32
.p2align 4

_elu_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    // Early exit on n=0
    cbz     x2, .Lelu_done

    // Save alpha (s0) before smstart zeroes it
    fmov    w8, s0

    // Enable streaming SVE mode
    smstart sm

    // Restore alpha and broadcast into z25
    fmov    s0, w8
    ptrue   p0.s
    mov     z25.s, s0               // z25 = alpha broadcast

    // Load constants for exp computation
    adr     x9, .Lelu_const
    ld1rw   {z16.s}, p0/z, [x9]        // log2(e)
    ld1rw   {z17.s}, p0/z, [x9, #4]    // ln(2)
    ld1rw   {z8.s}, p0/z, [x9, #8]     // 1.0
    ld1rw   {z9.s}, p0/z, [x9, #12]    // 0.5 = 1/2!
    ld1rw   {z10.s}, p0/z, [x9, #16]   // 1/6 = 1/3!
    ld1rw   {z11.s}, p0/z, [x9, #20]   // 1/24 = 1/4!
    ld1rw   {z12.s}, p0/z, [x9, #24]   // 1/120 = 1/5!
    ld1rw   {z13.s}, p0/z, [x9, #28]   // 1/720 = 1/6!
    ld1rw   {z14.s}, p0/z, [x9, #32]   // 1/5040 = 1/7!
    ld1rw   {z15.s}, p0/z, [x9, #36]   // 0.0 (clamp_lo for negatives)
    ld1rw   {z26.s}, p0/z, [x9, #40]   // -88.0 (clamp_lo for exp)

    // Zero for comparisons
    mov     z24.d, #0

    // Main loop setup
    mov     x7, #0
    whilelt pn9.s, x7, x2, vlx4

.Lelu_loop:
    // Load input x into z0-z3
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x7, lsl #2]

    // Save original x into z4-z7 for final blending
    mov     z4.d, z0.d
    mov     z5.d, z1.d
    mov     z6.d, z2.d
    mov     z7.d, z3.d

    // Clamp x to [-88, 0] for exp(x) computation on negative lanes
    fmax    z0.s, p0/m, z0.s, z26.s    // clamp to -88 (prevent underflow)
    fmin    z0.s, p0/m, z0.s, z15.s    // clamp to 0 (only for exp on negatives)
    fmax    z1.s, p0/m, z1.s, z26.s
    fmin    z1.s, p0/m, z1.s, z15.s
    fmax    z2.s, p0/m, z2.s, z26.s
    fmin    z2.s, p0/m, z2.s, z15.s
    fmax    z3.s, p0/m, z3.s, z26.s
    fmin    z3.s, p0/m, z3.s, z15.s

    // Range reduction: n_val = round(x * log2(e))
    fmul    z28.s, z0.s, z16.s
    fmul    z29.s, z1.s, z16.s
    fmul    z30.s, z2.s, z16.s
    fmul    z31.s, z3.s, z16.s
    frintn  z28.s, p0/m, z28.s
    frintn  z29.s, p0/m, z29.s
    frintn  z30.s, p0/m, z30.s
    frintn  z31.s, p0/m, z31.s

    // r = x - n_val * ln(2) using fmls
    fmls    z0.s, p0/m, z28.s, z17.s
    fmls    z1.s, p0/m, z29.s, z17.s
    fmls    z2.s, p0/m, z30.s, z17.s
    fmls    z3.s, p0/m, z31.s, z17.s

    // --- Polynomial exp(r) computation: p(r) = 1 + r + r^2/2! + ... + r^7/7! ---
    // Start with c0=1.0 in z18-z21, then accumulate terms
    mov     z18.d, z8.d              // z18-z21 = 1.0
    mov     z19.d, z8.d
    mov     z20.d, z8.d
    mov     z21.d, z8.d

    // Add c1*r (c1 = 1.0)
    fadd    z18.s, z18.s, z0.s
    fadd    z19.s, z19.s, z1.s
    fadd    z20.s, z20.s, z2.s
    fadd    z21.s, z21.s, z3.s

    // r^2 -> z22-z25
    fmul    z22.s, z0.s, z0.s
    fmul    z23.s, z1.s, z1.s
    fmul    z24.s, z2.s, z2.s
    fmul    z27.s, z3.s, z3.s        // using z27 for r^2[3]
    // Add c2*r^2 (c2 = 0.5)
    fmla    z18.s, p0/m, z22.s, z9.s
    fmla    z19.s, p0/m, z23.s, z9.s
    fmla    z20.s, p0/m, z24.s, z9.s
    fmla    z21.s, p0/m, z27.s, z9.s

    // r^3 = r^2 * r
    fmul    z22.s, z22.s, z0.s
    fmul    z23.s, z23.s, z1.s
    fmul    z24.s, z24.s, z2.s
    fmul    z27.s, z27.s, z3.s
    // Add c3*r^3 (c3 = 1/6)
    fmla    z18.s, p0/m, z22.s, z10.s
    fmla    z19.s, p0/m, z23.s, z10.s
    fmla    z20.s, p0/m, z24.s, z10.s
    fmla    z21.s, p0/m, z27.s, z10.s

    // r^4 = r^2 * r^2 (reload r^2)
    fmul    z22.s, z0.s, z0.s
    fmul    z23.s, z1.s, z1.s
    fmul    z24.s, z2.s, z2.s
    fmul    z27.s, z3.s, z3.s
    fmul    z22.s, z22.s, z22.s
    fmul    z23.s, z23.s, z23.s
    fmul    z24.s, z24.s, z24.s
    fmul    z27.s, z27.s, z27.s
    // Add c4*r^4 (c4 = 1/24)
    fmla    z18.s, p0/m, z22.s, z11.s
    fmla    z19.s, p0/m, z23.s, z11.s
    fmla    z20.s, p0/m, z24.s, z11.s
    fmla    z21.s, p0/m, z27.s, z11.s

    // r^2 again for r^5 and r^6
    fmul    z22.s, z0.s, z0.s
    fmul    z23.s, z1.s, z1.s
    fmul    z24.s, z2.s, z2.s
    fmul    z27.s, z3.s, z3.s

    // r^5 = r^4 * r (r^4 was clobbered, reload)
    fmul    z22.s, z0.s, z0.s
    fmul    z23.s, z1.s, z1.s
    fmul    z24.s, z2.s, z2.s
    fmul    z27.s, z3.s, z3.s
    fmul    z22.s, z22.s, z0.s
    fmul    z23.s, z23.s, z1.s
    fmul    z24.s, z24.s, z2.s
    fmul    z27.s, z27.s, z3.s
    // Add c5*r^5 (c5 = 1/120)
    fmla    z18.s, p0/m, z22.s, z12.s
    fmla    z19.s, p0/m, z23.s, z12.s
    fmla    z20.s, p0/m, z24.s, z12.s
    fmla    z21.s, p0/m, z27.s, z12.s

    // r^6 = r^3 * r^3 (reload r^3)
    fmul    z22.s, z0.s, z0.s
    fmul    z23.s, z1.s, z1.s
    fmul    z24.s, z2.s, z2.s
    fmul    z27.s, z3.s, z3.s
    fmul    z22.s, z22.s, z0.s
    fmul    z23.s, z23.s, z1.s
    fmul    z24.s, z24.s, z2.s
    fmul    z27.s, z27.s, z3.s
    fmul    z22.s, z22.s, z22.s
    fmul    z23.s, z23.s, z23.s
    fmul    z24.s, z24.s, z24.s
    fmul    z27.s, z27.s, z27.s
    // Add c6*r^6 (c6 = 1/720)
    fmla    z18.s, p0/m, z22.s, z13.s
    fmla    z19.s, p0/m, z23.s, z13.s
    fmla    z20.s, p0/m, z24.s, z13.s
    fmla    z21.s, p0/m, z27.s, z13.s

    // r^7 = r^6 * r
    fmul    z22.s, z22.s, z0.s
    fmul    z23.s, z23.s, z1.s
    fmul    z24.s, z24.s, z2.s
    fmul    z27.s, z27.s, z3.s
    // Add c7*r^7 (c7 = 1/5040)
    fmla    z18.s, p0/m, z22.s, z14.s
    fmla    z19.s, p0/m, z23.s, z14.s
    fmla    z20.s, p0/m, z24.s, z14.s
    fmla    z21.s, p0/m, z27.s, z14.s

    // Scale by 2^n_val to get exp(x)
    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s
    fscale  z18.s, p0/m, z18.s, z28.s
    fscale  z19.s, p0/m, z19.s, z29.s
    fscale  z20.s, p0/m, z20.s, z30.s
    fscale  z21.s, p0/m, z21.s, z31.s

    // Compute exp(x) - 1
    fsub    z18.s, z18.s, z8.s
    fsub    z19.s, z19.s, z8.s
    fsub    z20.s, z20.s, z8.s
    fsub    z21.s, z21.s, z8.s

    // Compute alpha * (exp(x) - 1)
    fmul    z18.s, z18.s, z25.s
    fmul    z19.s, z19.s, z25.s
    fmul    z20.s, z20.s, z25.s
    fmul    z21.s, z21.s, z25.s

    // Compare original input with 0: x >= 0 ?
    fcmge   p1.s, p0/z, z4.s, z24.s
    fcmge   p2.s, p0/z, z5.s, z24.s
    fcmge   p3.s, p0/z, z6.s, z24.s
    fcmge   p4.s, p0/z, z7.s, z24.s

    // Merge: result = x >= 0 ? x : alpha * (exp(x) - 1)
    // Start with alpha*(exp(x)-1), then overwrite non-negative lanes with x
    sel     z18.s, p1, z4.s, z18.s
    sel     z19.s, p2, z5.s, z19.s
    sel     z20.s, p3, z6.s, z20.s
    sel     z21.s, p4, z7.s, z21.s

    // Move results to z0-z3 for storage
    mov     z0.d, z18.d
    mov     z1.d, z19.d
    mov     z2.d, z20.d
    mov     z3.d, z21.d

    // Store result
    st1w    {z0.s-z3.s}, pn9, [x1, x7, lsl #2]

    // Increment and loop
    incw    x7, all, mul #4
    whilelt pn9.s, x7, x2, vlx4
    b.first .Lelu_loop

    smstop

.Lelu_done:
    // Epilogue: restore callee-saved registers
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #96
    ret

// Constant pool
.p2align 2
.Lelu_const:
    .long   0x3FB8AA3B  // log2(e) = 1.44269504
    .long   0x3F317218  // ln(2)   = 0.69314718
    .long   0x3F800000  // 1.0
    .long   0x3F000000  // 0.5     = 1/2!
    .long   0x3E2AAAAB  // 1/6     = 1/3!
    .long   0x3D2AAAAB  // 1/24    = 1/4!
    .long   0x3C088889  // 1/120   = 1/5!
    .long   0x3AB60B61  // 1/720   = 1/6!
    .long   0x39500D01  // 1/5040  = 1/7!
    .long   0x00000000  // 0.0 (clamp_lo for negatives)
    .long   0xC2B00000  // -88.0 (clamp_lo for exp)
