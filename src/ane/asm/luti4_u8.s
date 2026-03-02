// luti4_u8.s — SME2 4-bit LUT expand via ZT0 (multi-vec pipelined)
//
// void luti4_u8(const uint8_t *lut64, const uint8_t *packed_indices,
//               uint8_t *output, long n);
//
// Uses multi-vec LUTI4: luti4 {zN-zN+1}, zt0, zM[0] (2 outputs per op)
// Pipeline 4 LUTI4 ops = 8 output vectors per batch = 512 output bytes
// With 4 Z vectors/cycle throughput on M4 Max

.section __TEXT,__text,regular,pure_instructions
.global _luti4_u8
.p2align 4

_luti4_u8:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]

    cbz     x3, .Ldone

    smstart

    ldr     zt0, [x0]
    ptrue   p0.b
    ptrue   pn8.b                       // predicate-as-counter for multi-vec

    // ════════════════════════════════════════════════════════════════
    // Main loop: 2048 output bytes per iteration
    // Software pipelined: interleave load/compute/store for latency hiding
    // ════════════════════════════════════════════════════════════════
.Lloop2048:
    cmp     x3, #2048
    b.lt    .Lloop512

    // Wave 1: Load 4 → LUTI4 4 → Store 8
    ld1b    {z0.b-z3.b}, pn8/z, [x1]
    luti4   {z16.b, z17.b}, zt0, z0[0]
    luti4   {z18.b, z19.b}, zt0, z1[0]
    luti4   {z20.b, z21.b}, zt0, z2[0]
    luti4   {z22.b, z23.b}, zt0, z3[0]

    // Wave 2: Load 4 → LUTI4 4 → Store 8 (overlapped)
    ld1b    {z4.b-z7.b}, pn8/z, [x1, #4, mul vl]
    st1b    {z16.b-z19.b}, pn8, [x2]
    luti4   {z24.b, z25.b}, zt0, z4[0]
    luti4   {z26.b, z27.b}, zt0, z5[0]
    st1b    {z20.b-z23.b}, pn8, [x2, #4, mul vl]
    luti4   {z28.b, z29.b}, zt0, z6[0]
    luti4   {z30.b, z31.b}, zt0, z7[0]

    // Wave 3: Load 4 → LUTI4 4
    ld1b    {z8.b-z11.b}, pn8/z, [x1, #8, mul vl]
    st1b    {z24.b-z27.b}, pn8, [x2, #8, mul vl]
    luti4   {z16.b, z17.b}, zt0, z8[0]
    luti4   {z18.b, z19.b}, zt0, z9[0]
    st1b    {z28.b-z31.b}, pn8, [x2, #12, mul vl]
    luti4   {z20.b, z21.b}, zt0, z10[0]
    luti4   {z22.b, z23.b}, zt0, z11[0]

    // Wave 4: Load 4 → LUTI4 4 → Store remaining
    ld1b    {z12.b-z15.b}, pn8/z, [x1, #12, mul vl]
    st1b    {z16.b-z19.b}, pn8, [x2, #16, mul vl]
    luti4   {z24.b, z25.b}, zt0, z12[0]
    luti4   {z26.b, z27.b}, zt0, z13[0]
    st1b    {z20.b-z23.b}, pn8, [x2, #20, mul vl]
    luti4   {z28.b, z29.b}, zt0, z14[0]
    luti4   {z30.b, z31.b}, zt0, z15[0]

    // Final stores
    st1b    {z24.b-z27.b}, pn8, [x2, #24, mul vl]
    st1b    {z28.b-z31.b}, pn8, [x2, #28, mul vl]

    addvl   x1, x1, #16
    addvl   x2, x2, #16
    addvl   x2, x2, #16
    sub     x3, x3, #2048
    b       .Lloop2048

    // ════════════════════════════════════════════════════════════════
    // Medium: 512 output bytes (4 input vecs → 8 output vecs)
    // ════════════════════════════════════════════════════════════════
.Lloop512:
    cmp     x3, #512
    b.lt    .Lloop128

    ld1b    {z0.b-z3.b}, pn8/z, [x1]

    luti4   {z16.b, z17.b}, zt0, z0[0]
    luti4   {z18.b, z19.b}, zt0, z1[0]
    luti4   {z20.b, z21.b}, zt0, z2[0]
    luti4   {z22.b, z23.b}, zt0, z3[0]

    st1b    {z16.b-z19.b}, pn8, [x2]
    st1b    {z20.b-z23.b}, pn8, [x2, #4, mul vl]

    addvl   x1, x1, #4
    addvl   x2, x2, #8
    sub     x3, x3, #512
    b       .Lloop512

    // ════════════════════════════════════════════════════════════════
    // Small: 128 output bytes
    // ════════════════════════════════════════════════════════════════
.Lloop128:
    cmp     x3, #128
    b.lt    .Ltail

    ld1b    {z0.b}, p0/z, [x1]
    luti4   {z2.b, z3.b}, zt0, z0[0]
    st1b    {z2.b}, p0, [x2]
    st1b    {z3.b}, p0, [x2, #1, mul vl]
    add     x1, x1, #64
    addvl   x2, x2, #2
    sub     x3, x3, #128
    b       .Lloop128

    // ════════════════════════════════════════════════════════════════
    // Tail: predicated
    // ════════════════════════════════════════════════════════════════
.Ltail:
    cbz     x3, .Lstop

    lsr     x7, x3, #1
    whilelt p1.b, xzr, x7
    whilelt p2.b, xzr, x3

    ld1b    {z0.b}, p1/z, [x1]
    luti4   {z2.b, z3.b}, zt0, z0[0]

    cmp     x3, #64
    b.le    .Ltail_small
    st1b    {z2.b}, p0, [x2]
    sub     x8, x3, #64
    whilelt p3.b, xzr, x8
    st1b    {z3.b}, p3, [x2, #1, mul vl]
    b       .Lstop

.Ltail_small:
    st1b    {z2.b}, p2, [x2]

.Lstop:
    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10,  d11,  [sp, #32]
    ldp     d12,  d13,  [sp, #48]
    ldp     d14,  d15,  [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
