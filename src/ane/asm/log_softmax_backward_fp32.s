// log_softmax_backward_fp32.s — Log-softmax backward pass via SME2 streaming SVE
//
// void log_softmax_backward_fp32(const float *dy, const float *y, float *dx, long n);
//
// AAPCS64: x0=dy (upstream gradient), x1=y (saved log_softmax output),
//          x2=dx (output gradient), x3=n
//
// Math:  dx[i] = dy[i] - exp(y[i]) * sum(dy)
//
// Since y is the log_softmax output, y[i] <= 0, so exp(y[i]) in (0,1].
// exp(y[i]) = softmax(x[i]) (the original softmax probabilities).
//
// Algorithm:
//   Phase 1: sum_dy = sum_i(dy[i])  using 4 accumulators + tree reduce + faddv
//   Phase 2: for each i:
//              ey = exp(y[i])  via range-reduction + 6-term Horner polynomial
//              dx[i] = dy[i] - ey * sum_dy
//
// exp uses the same decomposition as log_softmax_fp32.s:
//   scaled = y * log2(e)
//   n_int  = round(scaled)
//   f      = scaled - n_int
//   2^f    = Horner(f, c0..c6)
//   exp(y) = 2^f * fscale(n_int)
//
// No input clamping is needed for y <= 0: exp(y) in (0,1], no overflow.

.section __TEXT,__text,regular,pure_instructions
.global _log_softmax_backward_fp32
.p2align 4

_log_softmax_backward_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    stp     d14, d15, [sp, #96]

    // Early exit
    cbz     x3, .Llsbk_done

    // Save arguments in callee-saved registers
    mov     x19, x0                 // dy
    mov     x20, x1                 // y  (log_softmax output)
    mov     x21, x2                 // dx
    mov     x22, x3                 // n

    smstart sm

    ptrue   p0.s

    // ═════════════════════════════════════════════════════════════
    // Phase 1: Compute sum_dy = sum_i(dy[i])
    // ═════════════════════════════════════════════════════════════

    mov     z8.d,  #0               // acc0
    mov     z9.d,  #0               // acc1
    mov     z10.d, #0               // acc2
    mov     z11.d, #0               // acc3

    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

.Llsbk_sum_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]  // dy

    fadd    z8.s,  p0/m, z8.s,  z0.s
    fadd    z9.s,  p0/m, z9.s,  z1.s
    fadd    z10.s, p0/m, z10.s, z2.s
    fadd    z11.s, p0/m, z11.s, z3.s

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Llsbk_sum_loop

    // Tree-reduce 4 accumulators → 1 → scalar
    fadd    z8.s,  p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s,  p0/m, z8.s, z10.s
    faddv   s8, p0, z8.s            // s8 = sum_dy (scalar)

    // Broadcast sum_dy into z30 for use in Phase 2
    mov     z30.s, s8               // z30 = sum_dy (all lanes)

    // ═════════════════════════════════════════════════════════════
    // Load exp constants
    // ═════════════════════════════════════════════════════════════
    adr     x9, .Llsbk_const
    ld1rw   {z16.s}, p0/z, [x9]        // log2(e) = 1.4426950...
    ld1rw   {z17.s}, p0/z, [x9, #4]    // c6
    ld1rw   {z18.s}, p0/z, [x9, #8]    // c5
    ld1rw   {z19.s}, p0/z, [x9, #12]   // c4
    ld1rw   {z20.s}, p0/z, [x9, #16]   // c3
    ld1rw   {z21.s}, p0/z, [x9, #20]   // c2
    ld1rw   {z22.s}, p0/z, [x9, #24]   // c1 = ln(2)
    // c0 = 1.0 — load via fmov (not dup #imm)
    fmov    z23.s, #1.0             // z23 = 1.0 (c0)

    // ═════════════════════════════════════════════════════════════
    // Phase 2: dx[i] = dy[i] - exp(y[i]) * sum_dy
    // ═════════════════════════════════════════════════════════════

    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

.Llsbk_grad_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x20, x8, lsl #2]  // y (log_softmax output)

    // scaled = y * log2(e)
    fmul    z0.s, p0/m, z0.s, z16.s
    fmul    z1.s, p0/m, z1.s, z16.s
    fmul    z2.s, p0/m, z2.s, z16.s
    fmul    z3.s, p0/m, z3.s, z16.s

    // n_int = round(scaled) → z8-z11
    movprfx z8,  z0
    frintn  z8.s,  p0/m, z0.s
    movprfx z9,  z1
    frintn  z9.s,  p0/m, z1.s
    movprfx z10, z2
    frintn  z10.s, p0/m, z2.s
    movprfx z11, z3
    frintn  z11.s, p0/m, z3.s

    // f = scaled - n_int
    fsub    z0.s, p0/m, z0.s, z8.s
    fsub    z1.s, p0/m, z1.s, z9.s
    fsub    z2.s, p0/m, z2.s, z10.s
    fsub    z3.s, p0/m, z3.s, z11.s

    // Convert n_int to integer for fscale
    fcvtzs  z8.s,  p0/m, z8.s
    fcvtzs  z9.s,  p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s

    // Horner: 2^f ≈ c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*(c5 + f*c6)))))
    // Start with p = c6, in-place in z24-z27
    mov     z24.d, z17.d            // p0_vec = c6
    mov     z25.d, z17.d
    mov     z26.d, z17.d
    mov     z27.d, z17.d

    fmad    z24.s, p0/m, z0.s, z18.s   // p = p*f + c5
    fmad    z25.s, p0/m, z1.s, z18.s
    fmad    z26.s, p0/m, z2.s, z18.s
    fmad    z27.s, p0/m, z3.s, z18.s

    fmad    z24.s, p0/m, z0.s, z19.s   // p = p*f + c4
    fmad    z25.s, p0/m, z1.s, z19.s
    fmad    z26.s, p0/m, z2.s, z19.s
    fmad    z27.s, p0/m, z3.s, z19.s

    fmad    z24.s, p0/m, z0.s, z20.s   // p = p*f + c3
    fmad    z25.s, p0/m, z1.s, z20.s
    fmad    z26.s, p0/m, z2.s, z20.s
    fmad    z27.s, p0/m, z3.s, z20.s

    fmad    z24.s, p0/m, z0.s, z21.s   // p = p*f + c2
    fmad    z25.s, p0/m, z1.s, z21.s
    fmad    z26.s, p0/m, z2.s, z21.s
    fmad    z27.s, p0/m, z3.s, z21.s

    fmad    z24.s, p0/m, z0.s, z22.s   // p = p*f + c1
    fmad    z25.s, p0/m, z1.s, z22.s
    fmad    z26.s, p0/m, z2.s, z22.s
    fmad    z27.s, p0/m, z3.s, z22.s

    fmad    z24.s, p0/m, z0.s, z23.s   // p = p*f + c0  (c0=1.0)
    fmad    z25.s, p0/m, z1.s, z23.s
    fmad    z26.s, p0/m, z2.s, z23.s
    fmad    z27.s, p0/m, z3.s, z23.s

    // exp(y) = p * 2^n_int
    fscale  z24.s, p0/m, z24.s, z8.s
    fscale  z25.s, p0/m, z25.s, z9.s
    fscale  z26.s, p0/m, z26.s, z10.s
    fscale  z27.s, p0/m, z27.s, z11.s

    // ey_sum = exp(y[i]) * sum_dy
    fmul    z24.s, p0/m, z24.s, z30.s
    fmul    z25.s, p0/m, z25.s, z30.s
    fmul    z26.s, p0/m, z26.s, z30.s
    fmul    z27.s, p0/m, z27.s, z30.s

    // Load dy
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // dx[i] = dy[i] - exp(y[i]) * sum_dy
    fsub    z0.s, p0/m, z0.s, z24.s
    fsub    z1.s, p0/m, z1.s, z25.s
    fsub    z2.s, p0/m, z2.s, z26.s
    fsub    z3.s, p0/m, z3.s, z27.s

    st1w    {z0.s-z3.s}, pn9, [x21, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Llsbk_grad_loop

    smstop

.Llsbk_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     d8,  d9,  [sp, #48]
    ldp     d10, d11, [sp, #64]
    ldp     d12, d13, [sp, #80]
    ldp     d14, d15, [sp, #96]
    ldp     x29, x30, [sp], #112
    ret

// Constant pool — IEEE-754 values for exp via 2^f Horner decomposition
// Same coefficients as log_softmax_fp32.s
.p2align 2
.Llsbk_const:
    .float  1.4426950408889634      // log2(e)
    .float  0.00015403530393381609  // c6 = ln2^6/720
    .float  0.0013333558146428443   // c5 = ln2^5/120
    .float  0.009618129107628477    // c4 = ln2^4/24
    .float  0.05550410866482158     // c3 = ln2^3/6
    .float  0.24022650695910072     // c2 = ln2^2/2
    .float  0.6931471805599453      // c1 = ln2
