// softmax_fp32.s — Softmax (1D) via SME2 streaming SVE
//
// void softmax_fp32(const float *input, float *output, long n);
//
// Four-phase algorithm:
//   1) Find max (single-vector, p-predicated — correct for all n)
//   2) Compute exp(input[i] - max), store to output (vlx4, fast)
//   3) Sum the exp values (single-vector, p-predicated — correct for all n)
//   4) Normalize: output *= 1/sum (vlx4, fast)
//
// Degree-6 Taylor polynomial for 2^f, fscale for 2^n reconstruction.
// Phases 2 & 4 process 64 floats per iteration on M4.

.section __TEXT,__text,regular,pure_instructions
.global _softmax_fp32
.p2align 4

_softmax_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    stp     d14, d15, [sp, #96]

    cbz     x2, .Ldone

    mov     x19, x0
    mov     x20, x1
    mov     x21, x2

    smstart sm

    ptrue   p0.s

    // Load constants
    adr     x9, .Lconsts
    ld1rw   {z21.s}, p0/z, [x9]        // log2e
    ld1rw   {z22.s}, p0/z, [x9, #4]    // c6
    ld1rw   {z23.s}, p0/z, [x9, #8]    // c5
    ld1rw   {z24.s}, p0/z, [x9, #12]   // c4
    ld1rw   {z25.s}, p0/z, [x9, #16]   // c3
    ld1rw   {z26.s}, p0/z, [x9, #20]   // c2
    ld1rw   {z27.s}, p0/z, [x9, #24]   // c1 = ln2
    ld1rw   {z29.s}, p0/z, [x9, #28]   // -inf
    fmov    z28.s, #1.0                  // c0

    // ═════════════════════════════════════════════════════════════
    // Phase 1: find max (4-vector main + single-vector tail)
    // ═════════════════════════════════════════════════════════════
    mov     z4.d, z29.d         // acc0 = -inf
    mov     z5.d, z29.d         // acc1 = -inf
    mov     z6.d, z29.d         // acc2 = -inf
    mov     z7.d, z29.d         // acc3 = -inf

    // Compute aligned count: floor(n / (4*VL)) * (4*VL)
    cntw    x10                 // x10 = VL (elements per vector)
    lsl     x11, x10, #2        // x11 = 4*VL
    udiv    x12, x21, x11       // x12 = n / (4*VL)
    mul     x22, x12, x11       // x22 = aligned count

    mov     x8, #0
    cmp     x8, x22
    b.ge    .Lmax_tail

    // Set up all-true predicate for aligned loop
    ptrue   pn9.s

.Lmax_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]
    fmax    z4.s, p0/m, z4.s, z0.s
    fmax    z5.s, p0/m, z5.s, z1.s
    fmax    z6.s, p0/m, z6.s, z2.s
    fmax    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    cmp     x8, x22
    b.lt    .Lmax_loop

.Lmax_tail:
    // Single-vector cleanup for remaining elements
    whilelt p1.s, x8, x21
    b.none  .Lmax_reduce

.Lmax_tail_loop:
    ld1w    {z0.s}, p1/z, [x19, x8, lsl #2]
    fmax    z4.s, p1/m, z4.s, z0.s
    incw    x8
    whilelt p1.s, x8, x21
    b.first .Lmax_tail_loop

.Lmax_reduce:
    // Tree-reduce 4 accumulators → 1
    fmax    z4.s, p0/m, z4.s, z5.s
    fmax    z6.s, p0/m, z6.s, z7.s
    fmax    z4.s, p0/m, z4.s, z6.s
    fmaxv   s20, p0, z4.s
    mov     z20.s, s20

    // ═════════════════════════════════════════════════════════════
    // Phase 2: exp(input - max) → output (vlx4)
    // ═════════════════════════════════════════════════════════════
    mov     x8, #0
    whilelt pn9.s, x8, x21, vlx4

.Lexp_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // x = input - max
    fsub    z0.s, p0/m, z0.s, z20.s
    fsub    z1.s, p0/m, z1.s, z20.s
    fsub    z2.s, p0/m, z2.s, z20.s
    fsub    z3.s, p0/m, z3.s, z20.s

    // z_scaled = x * log2e
    fmul    z0.s, p0/m, z0.s, z21.s
    fmul    z1.s, p0/m, z1.s, z21.s
    fmul    z2.s, p0/m, z2.s, z21.s
    fmul    z3.s, p0/m, z3.s, z21.s

    // n = round(z_scaled) → z8-z11
    movprfx z8, z0
    frintn  z8.s, p0/m, z0.s
    movprfx z9, z1
    frintn  z9.s, p0/m, z1.s
    movprfx z10, z2
    frintn  z10.s, p0/m, z2.s
    movprfx z11, z3
    frintn  z11.s, p0/m, z3.s

    // f = z_scaled - n → z0-z3
    fsub    z0.s, p0/m, z0.s, z8.s
    fsub    z1.s, p0/m, z1.s, z9.s
    fsub    z2.s, p0/m, z2.s, z10.s
    fsub    z3.s, p0/m, z3.s, z11.s

    // Convert n to integer for fscale
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s

    // Horner: 2^f = c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*(c5 + f*c6)))))
    mov     z16.d, z22.d        // p = c6
    mov     z17.d, z22.d
    mov     z18.d, z22.d
    mov     z19.d, z22.d

    fmad    z16.s, p0/m, z0.s, z23.s   // p = p*f + c5
    fmad    z17.s, p0/m, z1.s, z23.s
    fmad    z18.s, p0/m, z2.s, z23.s
    fmad    z19.s, p0/m, z3.s, z23.s

    fmad    z16.s, p0/m, z0.s, z24.s   // p = p*f + c4
    fmad    z17.s, p0/m, z1.s, z24.s
    fmad    z18.s, p0/m, z2.s, z24.s
    fmad    z19.s, p0/m, z3.s, z24.s

    fmad    z16.s, p0/m, z0.s, z25.s   // p = p*f + c3
    fmad    z17.s, p0/m, z1.s, z25.s
    fmad    z18.s, p0/m, z2.s, z25.s
    fmad    z19.s, p0/m, z3.s, z25.s

    fmad    z16.s, p0/m, z0.s, z26.s   // p = p*f + c2
    fmad    z17.s, p0/m, z1.s, z26.s
    fmad    z18.s, p0/m, z2.s, z26.s
    fmad    z19.s, p0/m, z3.s, z26.s

    fmad    z16.s, p0/m, z0.s, z27.s   // p = p*f + c1
    fmad    z17.s, p0/m, z1.s, z27.s
    fmad    z18.s, p0/m, z2.s, z27.s
    fmad    z19.s, p0/m, z3.s, z27.s

    fmad    z16.s, p0/m, z0.s, z28.s   // p = p*f + c0
    fmad    z17.s, p0/m, z1.s, z28.s
    fmad    z18.s, p0/m, z2.s, z28.s
    fmad    z19.s, p0/m, z3.s, z28.s

    // result = p * 2^n
    fscale  z16.s, p0/m, z16.s, z8.s
    fscale  z17.s, p0/m, z17.s, z9.s
    fscale  z18.s, p0/m, z18.s, z10.s
    fscale  z19.s, p0/m, z19.s, z11.s

    // Store (pn9 masks inactive lanes)
    st1w    {z16.s-z19.s}, pn9, [x20, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x21, vlx4
    b.first .Lexp_loop

    // ═════════════════════════════════════════════════════════════
    // Phase 3: sum exp values (4-vector, 4 accumulators)
    // ═════════════════════════════════════════════════════════════
    mov     z4.d, #0            // acc0
    mov     z5.d, #0            // acc1
    mov     z6.d, #0            // acc2
    mov     z7.d, #0            // acc3

    mov     x8, #0
    whilelt pn8.s, x8, x21, vlx4

.Lsum_loop:
    ld1w    {z0.s-z3.s}, pn8/z, [x20, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn8.s, x8, x21, vlx4
    b.first .Lsum_loop

    // Tree-reduce 4 accumulators → 1 → scalar
    fadd    z4.s, p0/m, z4.s, z5.s
    fadd    z6.s, p0/m, z6.s, z7.s
    fadd    z4.s, p0/m, z4.s, z6.s
    faddv   s4, p0, z4.s
    mov     z4.s, s4

    // 1/sum via frecpe + 2 Newton-Raphson steps
    frecpe  z5.s, z4.s
    frecps  z6.s, z4.s, z5.s
    fmul    z5.s, p0/m, z5.s, z6.s
    frecps  z6.s, z4.s, z5.s
    fmul    z5.s, p0/m, z5.s, z6.s
    mov     z30.d, z5.d

    // ═════════════════════════════════════════════════════════════
    // Phase 4: normalize output *= 1/sum (vlx4)
    // ═════════════════════════════════════════════════════════════
    mov     x8, #0
    whilelt pn9.s, x8, x21, vlx4

.Lnorm_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x20, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z30.s
    fmul    z1.s, p0/m, z1.s, z30.s
    fmul    z2.s, p0/m, z2.s, z30.s
    fmul    z3.s, p0/m, z3.s, z30.s
    st1w    {z0.s-z3.s}, pn9, [x20, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x21, vlx4
    b.first .Lnorm_loop

.Lexit:
    smstop

.Ldone:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     d8,  d9,  [sp, #48]
    ldp     d10, d11, [sp, #64]
    ldp     d12, d13, [sp, #80]
    ldp     d14, d15, [sp, #96]
    ldp     x29, x30, [sp], #112
    ret

// ── Literal pool ────────────────────────────────────────────────
.p2align 2
.Lconsts:
    .float 1.4426950408889634       // log2e
    .float 0.00015403530393381609   // c6 = ln2^6/720
    .float 0.0013333558146428443    // c5 = ln2^5/120
    .float 0.009618129107628477     // c4 = ln2^4/24
    .float 0.05550410866482158      // c3 = ln2^3/6
    .float 0.24022650695910072      // c2 = ln2^2/2
    .float 0.6931471805599453       // c1 = ln2
    .long  0xFF800000               // -inf
