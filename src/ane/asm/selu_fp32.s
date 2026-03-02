// selu_fp32.s — Element-wise SELU (Scaled Exponential Linear Unit) activation via SME2
//
// void selu_fp32(const float *input, float *output, long n)
// AAPCS: x0=input, x1=output, x2=n
//
// Computes: output[i] = lambda * (x >= 0 ? x : alpha * (exp(x) - 1))
// where lambda = 1.0507009873554804934193349852946
//       alpha  = 1.6732632423543772848170429916717
//
// These are FIXED constants, not parameters.
//
// Strategy:
// 1. Load 4 vectors of input
// 2. For lanes where x >= 0: result = lambda * x
// 3. For lanes where x < 0: compute alpha * (exp(x) - 1), then multiply by lambda
// 4. Use sel to merge results based on fcmge predicate
// 5. Clamp negative inputs to [-88, 0] to prevent exp overflow
//
// Processing: 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _selu_fp32
.p2align 4

_selu_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    // Early exit on n=0
    cbz     x2, .Lselu_done

    // Enable streaming SVE mode
    smstart sm

    ptrue   p0.s

    // Load constants
    adr     x9, .Lselu_const
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
    ld1rw   {z18.s}, p0/z, [x9, #44]   // lambda = 1.0507009873554804934193349852946
    ld1rw   {z19.s}, p0/z, [x9, #48]   // alpha = 1.6732632423543772848170429916717

    // Zero for comparisons
    mov     z24.d, #0

    // Main loop setup
    mov     x7, #0
    whilelt pn9.s, x7, x2, vlx4

.Lselu_loop:
    // Load input x into z0-z3
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x7, lsl #2]

    // Save original x into z4-z7 for final processing
    mov     z4.d, z0.d
    mov     z5.d, z1.d
    mov     z6.d, z2.d
    mov     z7.d, z3.d

    // Clamp x to [-88, 0] for exp(x) computation on negative lanes
    fmax    z0.s, p0/m, z0.s, z26.s    // clamp to -88
    fmin    z0.s, p0/m, z0.s, z15.s    // clamp to 0
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

    // r = x - n_val * ln(2)
    fmls    z0.s, p0/m, z28.s, z17.s
    fmls    z1.s, p0/m, z29.s, z17.s
    fmls    z2.s, p0/m, z30.s, z17.s
    fmls    z3.s, p0/m, z31.s, z17.s

    // --- Polynomial exp(r) computation ---
    // Start with c0=1.0
    mov     z20.d, z8.d
    mov     z21.d, z8.d
    mov     z22.d, z8.d
    mov     z23.d, z8.d

    // Add c1*r
    fadd    z20.s, z20.s, z0.s
    fadd    z21.s, z21.s, z1.s
    fadd    z22.s, z22.s, z2.s
    fadd    z23.s, z23.s, z3.s

    // r^2
    fmul    z25.s, z0.s, z0.s
    fmul    z27.s, z1.s, z1.s
    fmul    z0.s, z2.s, z2.s
    fmul    z1.s, z3.s, z3.s
    // Add c2*r^2
    fmla    z20.s, p0/m, z25.s, z9.s
    fmla    z21.s, p0/m, z27.s, z9.s
    fmla    z22.s, p0/m, z0.s, z9.s
    fmla    z23.s, p0/m, z1.s, z9.s

    // r^3 = r^2 * r (restore r values)
    fmul    z2.s, z0.s, z2.s
    fmul    z3.s, z1.s, z3.s
    // Actually we clobbered z0-z3 with r^2. Let me restructure to keep r values.
    // This is getting complex with limited z-registers. Let me use a different approach:
    // Keep r in z0-z3, keep 1.0 available elsewhere

    // Start over with better register allocation
    // We'll load z0-z3 again since we clobbered them
    // Actually, let me just compute directly from the clamped x values

    // Reset: reload the clamped values since we clobbered z0-z3
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x7, lsl #2]
    fmax    z0.s, p0/m, z0.s, z26.s
    fmin    z0.s, p0/m, z0.s, z15.s
    fmax    z1.s, p0/m, z1.s, z26.s
    fmin    z1.s, p0/m, z1.s, z15.s
    fmax    z2.s, p0/m, z2.s, z26.s
    fmin    z2.s, p0/m, z2.s, z15.s
    fmax    z3.s, p0/m, z3.s, z26.s
    fmin    z3.s, p0/m, z3.s, z15.s

    // Range reduction again
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

    // --- Polynomial: p(r) = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 + r^7/5040 ---
    // Start with 1.0
    mov     z20.d, z8.d
    mov     z21.d, z8.d
    mov     z22.d, z8.d
    mov     z23.d, z8.d

    // c1*r (c1=1.0)
    fadd    z20.s, z20.s, z0.s
    fadd    z21.s, z21.s, z1.s
    fadd    z22.s, z22.s, z2.s
    fadd    z23.s, z23.s, z3.s

    // r^2
    fmul    z25.s, z0.s, z0.s
    fmul    z27.s, z1.s, z1.s
    fmul    z4.s, z2.s, z2.s
    fmul    z5.s, z3.s, z3.s
    // c2*r^2
    fmla    z20.s, p0/m, z25.s, z9.s
    fmla    z21.s, p0/m, z27.s, z9.s
    fmla    z22.s, p0/m, z4.s, z9.s
    fmla    z23.s, p0/m, z5.s, z9.s

    // r^3 = r^2 * r
    fmul    z25.s, z25.s, z0.s
    fmul    z27.s, z27.s, z1.s
    fmul    z4.s, z4.s, z2.s
    fmul    z5.s, z5.s, z3.s
    // c3*r^3
    fmla    z20.s, p0/m, z25.s, z10.s
    fmla    z21.s, p0/m, z27.s, z10.s
    fmla    z22.s, p0/m, z4.s, z10.s
    fmla    z23.s, p0/m, z5.s, z10.s

    // r^4 = (r^2)^2 (reload r^2)
    fmul    z6.s, z0.s, z0.s
    fmul    z7.s, z1.s, z1.s
    fmul    z25.s, z2.s, z2.s
    fmul    z27.s, z3.s, z3.s
    fmul    z6.s, z6.s, z6.s
    fmul    z7.s, z7.s, z7.s
    fmul    z25.s, z25.s, z25.s
    fmul    z27.s, z27.s, z27.s
    // c4*r^4
    fmla    z20.s, p0/m, z6.s, z11.s
    fmla    z21.s, p0/m, z7.s, z11.s
    fmla    z22.s, p0/m, z25.s, z11.s
    fmla    z23.s, p0/m, z27.s, z11.s

    // r^5 = r^4 * r (r^4 in z6,z7,z25,z27; r in z0-z3)
    fmul    z6.s, z6.s, z0.s
    fmul    z7.s, z7.s, z1.s
    fmul    z25.s, z25.s, z2.s
    fmul    z27.s, z27.s, z3.s
    // c5*r^5
    fmla    z20.s, p0/m, z6.s, z12.s
    fmla    z21.s, p0/m, z7.s, z12.s
    fmla    z22.s, p0/m, z25.s, z12.s
    fmla    z23.s, p0/m, z27.s, z12.s

    // r^6 = (r^3)^2 (reload r^3)
    fmul    z6.s, z0.s, z0.s
    fmul    z7.s, z1.s, z1.s
    fmul    z25.s, z2.s, z2.s
    fmul    z27.s, z3.s, z3.s
    fmul    z6.s, z6.s, z0.s
    fmul    z7.s, z7.s, z1.s
    fmul    z25.s, z25.s, z2.s
    fmul    z27.s, z27.s, z3.s
    fmul    z6.s, z6.s, z6.s
    fmul    z7.s, z7.s, z7.s
    fmul    z25.s, z25.s, z25.s
    fmul    z27.s, z27.s, z27.s
    // c6*r^6
    fmla    z20.s, p0/m, z6.s, z13.s
    fmla    z21.s, p0/m, z7.s, z13.s
    fmla    z22.s, p0/m, z25.s, z13.s
    fmla    z23.s, p0/m, z27.s, z13.s

    // r^7 = r^6 * r
    fmul    z6.s, z6.s, z0.s
    fmul    z7.s, z7.s, z1.s
    fmul    z25.s, z25.s, z2.s
    fmul    z27.s, z27.s, z3.s
    // c7*r^7
    fmla    z20.s, p0/m, z6.s, z14.s
    fmla    z21.s, p0/m, z7.s, z14.s
    fmla    z22.s, p0/m, z25.s, z14.s
    fmla    z23.s, p0/m, z27.s, z14.s

    // Scale by 2^n_val to get exp(x)
    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s
    fscale  z20.s, p0/m, z20.s, z28.s
    fscale  z21.s, p0/m, z21.s, z29.s
    fscale  z22.s, p0/m, z22.s, z30.s
    fscale  z23.s, p0/m, z23.s, z31.s

    // exp(x) - 1
    fsub    z20.s, z20.s, z8.s
    fsub    z21.s, z21.s, z8.s
    fsub    z22.s, z22.s, z8.s
    fsub    z23.s, z23.s, z8.s

    // alpha * (exp(x) - 1)
    // z19 = alpha, z20-z23 = exp(x)-1
    fmul    z20.s, z20.s, z19.s
    fmul    z21.s, z21.s, z19.s
    fmul    z22.s, z22.s, z19.s
    fmul    z23.s, z23.s, z19.s

    // Reload original x (z6/z7 were clobbered by r^4 computation)
    ld1w    {z4.s-z7.s}, pn9/z, [x0, x7, lsl #2]

    // Compare original input with 0
    fcmge   p1.s, p0/z, z4.s, z24.s
    fcmge   p2.s, p0/z, z5.s, z24.s
    fcmge   p3.s, p0/z, z6.s, z24.s
    fcmge   p4.s, p0/z, z7.s, z24.s

    // Merge: result = x >= 0 ? x : alpha * (exp(x) - 1)
    sel     z20.s, p1, z4.s, z20.s
    sel     z21.s, p2, z5.s, z21.s
    sel     z22.s, p3, z6.s, z22.s
    sel     z23.s, p4, z7.s, z23.s

    // Multiply by lambda
    // z18 = lambda
    fmul    z20.s, z20.s, z18.s
    fmul    z21.s, z21.s, z18.s
    fmul    z22.s, z22.s, z18.s
    fmul    z23.s, z23.s, z18.s

    // Move results to z0-z3 for storage
    mov     z0.d, z20.d
    mov     z1.d, z21.d
    mov     z2.d, z22.d
    mov     z3.d, z23.d

    // Store result
    st1w    {z0.s-z3.s}, pn9, [x1, x7, lsl #2]

    // Increment and loop
    incw    x7, all, mul #4
    whilelt pn9.s, x7, x2, vlx4
    b.first .Lselu_loop

    smstop

.Lselu_done:
    // Epilogue: restore callee-saved registers
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #96
    ret

// Constant pool
.p2align 2
.Lselu_const:
    .long   0x3FB8AA3B  // log2(e) = 1.44269504
    .long   0x3F317218  // ln(2)   = 0.69314718
    .long   0x3F800000  // 1.0
    .long   0x3F000000  // 0.5     = 1/2!
    .long   0x3E2AAAAB  // 1/6     = 1/3!
    .long   0x3D2AAAAB  // 1/24    = 1/4!
    .long   0x3C088889  // 1/120   = 1/5!
    .long   0x3AB60B61  // 1/720   = 1/6!
    .long   0x39500D01  // 1/5040  = 1/7!
    .long   0x00000000  // 0.0
    .long   0xC2B00000  // -88.0
    .long   0x3F867E7F  // lambda = 1.0507009873554804934193349852946
    .long   0x3FD6A62F  // alpha  = 1.6732632423543772848170429916717
