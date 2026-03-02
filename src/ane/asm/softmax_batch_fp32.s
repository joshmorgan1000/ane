// softmax_batch_fp32.s — Batched Softmax (2D) via SME2 streaming SVE
//
// void softmax_batch_fp32(const float *input, float *output, long batch_size, long seq_len);
//
// AAPCS64: x0=input, x1=output, x2=batch_size, x3=seq_len
//
// Processes each row (sequence) independently using the standard 4-phase softmax:
//   1) Find max value in row
//   2) Compute exp(x[i] - max) for each element
//   3) Sum all exp values
//   4) Normalize: output[i] = exp[i] / sum
//
// Outer loop iterates over batch_size, inner processing over seq_len.
// Phases 2 & 4 use vlx4 (64 floats/iteration), phases 1 & 3 use single-vector.

.section __TEXT,__text,regular,pure_instructions
.global _softmax_batch_fp32
.p2align 4

_softmax_batch_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    stp     d14, d15, [sp, #96]

    // Early exit checks
    cbz     x2, .Lsbf_done          // batch_size == 0
    cbz     x3, .Lsbf_done          // seq_len == 0

    // Save arguments in callee-saved registers
    mov     x19, x0                 // input base
    mov     x20, x1                 // output base
    mov     x21, x2                 // batch_size
    mov     x22, x3                 // seq_len

    smstart sm

    ptrue   p0.s

    // ── Load constants ──
    adr     x9, .Lsbf_consts
    ld1rw   {z21.s}, p0/z, [x9]        // log2e
    ld1rw   {z22.s}, p0/z, [x9, #4]    // c6
    ld1rw   {z23.s}, p0/z, [x9, #8]    // c5
    ld1rw   {z24.s}, p0/z, [x9, #12]   // c4
    ld1rw   {z25.s}, p0/z, [x9, #16]   // c3
    ld1rw   {z26.s}, p0/z, [x9, #20]   // c2
    ld1rw   {z27.s}, p0/z, [x9, #24]   // c1 = ln2
    ld1rw   {z29.s}, p0/z, [x9, #28]   // -inf
    fmov    z28.s, #1.0                  // c0

    // Compute row size in bytes for pointer updates
    lsl     x15, x22, #2            // x15 = seq_len * 4 (bytes)

    // ── Outer loop over batch_size ──
    mov     x14, #0                 // batch index

.Lsbf_batch_loop:
    // Current row pointers
    mul     x16, x14, x15           // byte offset = batch_idx * row_bytes
    add     x10, x19, x16           // input_row = input + offset
    add     x11, x20, x16           // output_row = output + offset

    // ═════════════════════════════════════════════════════════════
    // Phase 1: find max in current row
    // ═════════════════════════════════════════════════════════════
    mov     z4.d, z29.d             // acc0 = -inf
    mov     z5.d, z29.d             // acc1 = -inf
    mov     z6.d, z29.d             // acc2 = -inf
    mov     z7.d, z29.d             // acc3 = -inf

    // Compute aligned count: floor(seq_len / (4*VL)) * (4*VL)
    cntw    x12                     // x12 = VL (elements per vector)
    lsl     x13, x12, #2            // x13 = 4*VL
    udiv    x16, x22, x13           // x16 = seq_len / (4*VL)
    mul     x17, x16, x13           // x17 = aligned count

    mov     x8, #0
    cmp     x8, x17
    b.ge    .Lsbf_max_tail

    // Set up all-true predicate for aligned loop
    ptrue   pn9.s

.Lsbf_max_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x10, x8, lsl #2]
    fmax    z4.s, p0/m, z4.s, z0.s
    fmax    z5.s, p0/m, z5.s, z1.s
    fmax    z6.s, p0/m, z6.s, z2.s
    fmax    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    cmp     x8, x17
    b.lt    .Lsbf_max_loop

.Lsbf_max_tail:
    // Single-vector cleanup for remaining elements
    whilelt p1.s, x8, x22
    b.none  .Lsbf_max_reduce

.Lsbf_max_tail_loop:
    ld1w    {z0.s}, p1/z, [x10, x8, lsl #2]
    fmax    z4.s, p1/m, z4.s, z0.s
    incw    x8
    whilelt p1.s, x8, x22
    b.first .Lsbf_max_tail_loop

.Lsbf_max_reduce:
    // Tree-reduce 4 accumulators → 1 → scalar
    fmax    z4.s, p0/m, z4.s, z5.s
    fmax    z6.s, p0/m, z6.s, z7.s
    fmax    z4.s, p0/m, z4.s, z6.s
    fmaxv   s20, p0, z4.s
    mov     z20.s, s20              // broadcast max to all lanes

    // ═════════════════════════════════════════════════════════════
    // Phase 2: exp(input - max) → output (vlx4)
    // ═════════════════════════════════════════════════════════════
    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

.Lsbf_exp_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x10, x8, lsl #2]

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

    // Horner polynomial: 2^f = c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*(c5 + f*c6)))))
    mov     z16.d, z22.d            // p = c6
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
    st1w    {z16.s-z19.s}, pn9, [x11, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lsbf_exp_loop

    // ═════════════════════════════════════════════════════════════
    // Phase 3: sum exp values (vlx4 with 4 accumulators)
    // ═════════════════════════════════════════════════════════════
    mov     z4.d, #0                // acc0
    mov     z5.d, #0                // acc1
    mov     z6.d, #0                // acc2
    mov     z7.d, #0                // acc3

    mov     x8, #0
    whilelt pn8.s, x8, x22, vlx4

.Lsbf_sum_loop:
    ld1w    {z0.s-z3.s}, pn8/z, [x11, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn8.s, x8, x22, vlx4
    b.first .Lsbf_sum_loop

    // Tree-reduce 4 accumulators → 1 → scalar
    fadd    z4.s, p0/m, z4.s, z5.s
    fadd    z6.s, p0/m, z6.s, z7.s
    fadd    z4.s, p0/m, z4.s, z6.s
    faddv   s4, p0, z4.s
    mov     z4.s, s4

    // Compute 1/sum via frecpe + 2 Newton-Raphson steps
    frecpe  z5.s, z4.s
    frecps  z6.s, z4.s, z5.s
    fmul    z5.s, p0/m, z5.s, z6.s
    frecps  z6.s, z4.s, z5.s
    fmul    z5.s, p0/m, z5.s, z6.s
    mov     z30.d, z5.d             // z30 = 1/sum

    // ═════════════════════════════════════════════════════════════
    // Phase 4: normalize output *= 1/sum (vlx4)
    // ═════════════════════════════════════════════════════════════
    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

.Lsbf_norm_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x11, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z30.s
    fmul    z1.s, p0/m, z1.s, z30.s
    fmul    z2.s, p0/m, z2.s, z30.s
    fmul    z3.s, p0/m, z3.s, z30.s
    st1w    {z0.s-z3.s}, pn9, [x11, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lsbf_norm_loop

    // ── Next batch row ──
    add     x14, x14, #1
    cmp     x14, x21
    b.lt    .Lsbf_batch_loop

    smstop

.Lsbf_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     d8,  d9,  [sp, #48]
    ldp     d10, d11, [sp, #64]
    ldp     d12, d13, [sp, #80]
    ldp     d14, d15, [sp, #96]
    ldp     x29, x30, [sp], #112
    ret

// ── Constant pool ────────────────────────────────────────────────
.p2align 2
.Lsbf_consts:
    .float 1.4426950408889634       // log2e
    .float 0.00015403530393381609   // c6 = ln2^6/720
    .float 0.0013333558146428443    // c5 = ln2^5/120
    .float 0.009618129107628477     // c4 = ln2^4/24
    .float 0.05550410866482158      // c3 = ln2^3/6
    .float 0.24022650695910072      // c2 = ln2^2/2
    .float 0.6931471805599453       // c1 = ln2
    .long  0xFF800000               // -inf
