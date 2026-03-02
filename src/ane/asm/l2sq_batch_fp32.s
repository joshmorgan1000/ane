// l2sq_batch_fp32.s — Batched L2-squared distance via SME2
//
// void l2sq_batch_fp32(const float* query,    // x0: dim floats
//                      const float* data,     // x1: n_vectors × dim floats
//                      float* output,         // x2: n_vectors output distances
//                      long dim,              // x3
//                      long n_vectors)        // x4
//
// Computes output[j] = sum_i (query[i] - data[j*dim + i])^2 for j in [0, n_vectors).
//
// Uses the algebraic identity: (q - d)^2 = q^2 + d^2 - 2*q*d
// This maps to: FMLA(q,q) + FMLA(d,d) - FMLS(q,d) - FMLS(q,d) per ZA row group.
//
// Batch-of-4 path: processes 4 database vectors in parallel using ZA rows 0-15
// (4 rows per vector × 4 vectors = 16 ZA rows).
// Tail path: processes remaining 1-3 vectors using z-reg accumulators.
//
// Register allocation:
//   w8=0, w9=4, w10=8, w11=12: ZA row bases for vec 0-3
//   x19: query, x20: data cursor, x21: output cursor
//   x22: dim, x23: n_vectors remaining, x24: dim_bytes
//   x25,x26,x14,x15: data pointers for batch-of-4
//   x13: inner loop offset

.section __TEXT,__text,regular,pure_instructions
.global _l2sq_batch_fp32
.p2align 4

_l2sq_batch_fp32:
    // Prologue: save callee-saved registers
    stp     x29, x30, [sp, #-144]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     d8,  d9,  [sp, #80]
    stp     d10, d11, [sp, #96]
    stp     d12, d13, [sp, #112]
    stp     d14,  d15,  [sp, #128]

    // Early exit: dim <= 0 → fill output with 0.0
    cmp     x3, #0
    b.le    .Lzero_fill

    // Early exit: no vectors
    cbz     x4, .Ldone

    // Save args into callee-saved registers
    mov     x19, x0             // query
    mov     x20, x1             // data cursor
    mov     x21, x2             // output cursor
    mov     x22, x3             // dim
    mov     x23, x4             // n_vectors remaining
    lsl     x24, x3, #2         // dim_bytes = dim * 4

    smstart                     // full streaming mode (ZA + SVE)
    ptrue   p0.s

    // ================================================================
    // Batch-of-4 path: 4 vectors at a time using ZA
    // ================================================================
.Lbatch4:
    cmp     x23, #4
    b.lt    .Ltail

    // Zero all ZA tiles
    zero    {za}

    // Setup ZA row bases in w8-w11 (required range for ZA indexing)
    mov     w8,  #0
    mov     w9,  #4
    mov     w10, #8
    mov     w11, #12

    // Setup data pointers for 4 concurrent vectors
    // Use x25,x26 (callee-saved) and x14,x15 (caller-saved scratch)
    mov     x25, x20                 // vec 0
    add     x26, x20, x24           // vec 1 = data + dim_bytes
    add     x14, x26, x24           // vec 2 = data + 2*dim_bytes
    add     x15, x14, x24           // vec 3 = data + 3*dim_bytes

    // Inner loop offset
    mov     x13, #0
    whilelt pn9.s, x13, x22, vlx4

.Linner4:
    // Load query chunk (shared across all 4 vectors)
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x13, lsl #2]

    // ---- Vector 0 → ZA rows 0-3 (w8=0) ----
    ld1w    {z4.s-z7.s}, pn9/z, [x25, x13, lsl #2]
    fmla    za.s[w8, 0, vgx4], {z0.s-z3.s}, {z0.s-z3.s}   // += q^2
    fmla    za.s[w8, 0, vgx4], {z4.s-z7.s}, {z4.s-z7.s}   // += d0^2
    fmls    za.s[w8, 0, vgx4], {z0.s-z3.s}, {z4.s-z7.s}   // -= q*d0
    fmls    za.s[w8, 0, vgx4], {z0.s-z3.s}, {z4.s-z7.s}   // -= q*d0

    // ---- Vector 1 → ZA rows 4-7 (w9=4) ----
    ld1w    {z4.s-z7.s}, pn9/z, [x26, x13, lsl #2]
    fmla    za.s[w9, 0, vgx4], {z0.s-z3.s}, {z0.s-z3.s}   // += q^2
    fmla    za.s[w9, 0, vgx4], {z4.s-z7.s}, {z4.s-z7.s}   // += d1^2
    fmls    za.s[w9, 0, vgx4], {z0.s-z3.s}, {z4.s-z7.s}   // -= q*d1
    fmls    za.s[w9, 0, vgx4], {z0.s-z3.s}, {z4.s-z7.s}   // -= q*d1

    // ---- Vector 2 → ZA rows 8-11 (w10=8) ----
    ld1w    {z4.s-z7.s}, pn9/z, [x14, x13, lsl #2]
    fmla    za.s[w10, 0, vgx4], {z0.s-z3.s}, {z0.s-z3.s}  // += q^2
    fmla    za.s[w10, 0, vgx4], {z4.s-z7.s}, {z4.s-z7.s}  // += d2^2
    fmls    za.s[w10, 0, vgx4], {z0.s-z3.s}, {z4.s-z7.s}  // -= q*d2
    fmls    za.s[w10, 0, vgx4], {z0.s-z3.s}, {z4.s-z7.s}  // -= q*d2

    // ---- Vector 3 → ZA rows 12-15 (w11=12) ----
    ld1w    {z4.s-z7.s}, pn9/z, [x15, x13, lsl #2]
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, {z0.s-z3.s}  // += q^2
    fmla    za.s[w11, 0, vgx4], {z4.s-z7.s}, {z4.s-z7.s}  // += d3^2
    fmls    za.s[w11, 0, vgx4], {z0.s-z3.s}, {z4.s-z7.s}  // -= q*d3
    fmls    za.s[w11, 0, vgx4], {z0.s-z3.s}, {z4.s-z7.s}  // -= q*d3

    incw    x13, all, mul #4
    whilelt pn9.s, x13, x22, vlx4
    b.first .Linner4

    // ================================================================
    // ZA Extraction & Reduction
    // ================================================================
    // ZA sub-tile mapping (FP32, SVLs=16):
    //   Full ZA row R → sub-tile za(R%4), sub-row (R/4)
    //   za0h.s rows 0-3 = full ZA rows 0, 4, 8, 12
    //   za1h.s rows 0-3 = full ZA rows 1, 5, 9, 13
    //   za2h.s rows 0-3 = full ZA rows 2, 6, 10, 14
    //   za3h.s rows 0-3 = full ZA rows 3, 7, 11, 15
    //
    // So extracting za0h.s[w12, 0:3] gives us one element from each of
    // the 4 vectors (rows 0, 4, 8, 12), and similarly for za1h-za3h.

    // Extract all 16 ZA rows using za.s[w_reg, 0, vgx4] form
    // Reuse w8-w11 which still hold 0,4,8,12 from the inner loop
    mova    {z0.s-z3.s},   za.s[w8,  0, vgx4]    // full ZA rows 0-3  (vec 0)
    mova    {z4.s-z7.s},   za.s[w9,  0, vgx4]    // full ZA rows 4-7  (vec 1)
    mova    {z8.s-z11.s},  za.s[w10, 0, vgx4]    // full ZA rows 8-11 (vec 2)
    mova    {z12.s-z15.s}, za.s[w11, 0, vgx4]    // full ZA rows 12-15 (vec 3)

    // Tree-reduce 4 rows → 1 per vector (interleaved for ILP)
    fadd    z0.s, p0/m, z0.s, z1.s     // vec0: r0+r1
    fadd    z4.s, p0/m, z4.s, z5.s     // vec1: r4+r5
    fadd    z8.s, p0/m, z8.s, z9.s     // vec2: r8+r9
    fadd    z12.s, p0/m, z12.s, z13.s  // vec3: r12+r13

    fadd    z2.s, p0/m, z2.s, z3.s     // vec0: r2+r3
    fadd    z6.s, p0/m, z6.s, z7.s     // vec1: r6+r7
    fadd    z10.s, p0/m, z10.s, z11.s  // vec2: r10+r11
    fadd    z14.s, p0/m, z14.s, z15.s  // vec3: r14+r15

    fadd    z0.s, p0/m, z0.s, z2.s     // vec0 final
    fadd    z4.s, p0/m, z4.s, z6.s     // vec1 final
    fadd    z8.s, p0/m, z8.s, z10.s    // vec2 final
    fadd    z12.s, p0/m, z12.s, z14.s  // vec3 final

    // Horizontal sum to scalars
    faddv   s16, p0, z0.s              // vec 0 distance
    faddv   s17, p0, z4.s              // vec 1 distance
    faddv   s18, p0, z8.s              // vec 2 distance
    faddv   s19, p0, z12.s             // vec 3 distance

    // Store 4 results
    str     s16, [x21]
    str     s17, [x21, #4]
    str     s18, [x21, #8]
    str     s19, [x21, #12]
    add     x21, x21, #16

    // Advance data cursor by 4 vectors
    add     x20, x20, x24, lsl #2  // data += 4 * dim_bytes
    sub     x23, x23, #4
    b       .Lbatch4

    // ================================================================
    // Tail path: 1-3 remaining vectors using z-reg accumulators
    // ================================================================
.Ltail:
    cbz     x23, .Lstop

.Ltail1:
    // Zero 4 accumulator z-regs
    mov     z8.d,  #0
    mov     z9.d,  #0
    mov     z10.d, #0
    mov     z11.d, #0

    mov     x13, #0
    whilelt pn9.s, x13, x22, vlx4

.Ltail_inner:
    // Load query chunk
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x13, lsl #2]
    // Load data chunk
    ld1w    {z4.s-z7.s}, pn9/z, [x20, x13, lsl #2]

    // diff = query - data (destructive: overwrites query regs, fine since
    // we reload from memory each iteration)
    fsub    z0.s, p0/m, z0.s, z4.s
    fsub    z1.s, p0/m, z1.s, z5.s
    fsub    z2.s, p0/m, z2.s, z6.s
    fsub    z3.s, p0/m, z3.s, z7.s

    // Accumulate diff^2
    fmla    z8.s,  p0/m, z0.s, z0.s
    fmla    z9.s,  p0/m, z1.s, z1.s
    fmla    z10.s, p0/m, z2.s, z2.s
    fmla    z11.s, p0/m, z3.s, z3.s

    incw    x13, all, mul #4
    whilelt pn9.s, x13, x22, vlx4
    b.first .Ltail_inner

    // Tree-reduce 4 accumulators → 1
    fadd    z8.s,  p0/m, z8.s,  z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s,  p0/m, z8.s,  z10.s

    // Horizontal sum → scalar
    faddv   s0, p0, z8.s
    str     s0, [x21]
    add     x21, x21, #4

    // Advance data cursor by 1 vector
    add     x20, x20, x24
    subs    x23, x23, #1
    b.gt    .Ltail1

.Lstop:
    smstop
    b       .Ldone

    // ================================================================
    // Zero-fill: dim <= 0, fill output with 0.0
    // ================================================================
.Lzero_fill:
    cbz     x4, .Ldone
    // Use scalar stores (dim=0 is edge case, perf not critical)
    mov     w5, #0                  // 0.0f bit pattern
    mov     x6, x4                  // count
.Lzero_loop:
    str     w5, [x2], #4
    subs    x6, x6, #1
    b.gt    .Lzero_loop

.Ldone:
    ldp     d12, d13, [sp, #112]
    ldp     d10, d11, [sp, #96]
    ldp     d8,  d9,  [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     d14,  d15,  [sp, #128]
    ldp     x29, x30, [sp], #144
    ret
