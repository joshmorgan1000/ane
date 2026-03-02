// tbl2_u8.s — SVE2 byte-level table lookup (two sources)
//
// void tbl2_u8(const uint8_t *table, const uint8_t *indices,
//              uint8_t *output, long n_table, long n);
//
// table:   up to 2×SVLb (128) entries loaded into two z registers
// indices: n byte indices (0..127, out-of-range yields 0)
// Uses TBL z_out.b, {z_tbl0.b, z_tbl1.b}, z_idx.b (SVE2 two-source form)
// vlx4 loop over indices
//
// Register allocation:
//   x0 = table ptr
//   x1 = indices ptr
//   x2 = output ptr
//   x3 = n_table
//   x4 = n (byte count)
//   x8 = loop counter
//   x9 = SVLb (bytes per vector)
//   z16 = first table vector (entries 0..63)
//   z17 = second table vector (entries 64..127)
//   p1 = first table load predicate
//   p2 = second table load predicate
//   pn9 = loop predicate

.section __TEXT,__text,regular,pure_instructions
.global _tbl2_u8
.p2align 4

_tbl2_u8:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8, d9, [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x4, .Ldone

    smstart sm

    // Load first vector of table (up to SVLb entries)
    cntb    x9                          // x9 = SVLb (64 on M4)
    // First chunk: min(n_table, SVLb) entries
    cmp     x3, x9
    csel    x10, x3, x9, lo            // x10 = min(n_table, SVLb)
    whilelt p1.b, xzr, x10
    ld1b    {z16.b}, p1/z, [x0]

    // Second chunk: remaining entries (n_table - SVLb), clamped to >= 0
    subs    x10, x3, x9
    b.le    .Lno_second
    whilelt p2.b, xzr, x10
    ld1b    {z17.b}, p2/z, [x0, x9]
    b       .Ltable_done

.Lno_second:
    mov     z17.b, #0

.Ltable_done:
    mov     x8, #0
    whilelt pn9.b, x8, x4, vlx4

.Lloop:
    ld1b    {z0.b-z3.b}, pn9/z, [x1, x8]
    tbl     z0.b, {z16.b, z17.b}, z0.b
    tbl     z1.b, {z16.b, z17.b}, z1.b
    tbl     z2.b, {z16.b, z17.b}, z2.b
    tbl     z3.b, {z16.b, z17.b}, z3.b
    st1b    {z0.b-z3.b}, pn9, [x2, x8]
    incb    x8, all, mul #4
    whilelt pn9.b, x8, x4, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d14, d15, [sp, #64]
    ldp     d12, d13, [sp, #48]
    ldp     d10, d11, [sp, #32]
    ldp     d8, d9, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret
