// tbl_u8.s — SVE byte-level table lookup (single source)
//
// void tbl_u8(const uint8_t *table, const uint8_t *indices,
//             uint8_t *output, long n_table, long n);
//
// table:   up to SVLb (64) entries loaded into one z register
// indices: n byte indices (0..63, out-of-range yields 0)
// Uses TBL z_out.b, {z_table.b}, z_idx.b
// vlx4 loop over indices
//
// Register allocation:
//   x0 = table ptr
//   x1 = indices ptr
//   x2 = output ptr
//   x3 = n_table
//   x4 = n (byte count)
//   x8 = loop counter
//   z16 = table vector
//   p1 = table load predicate
//   pn9 = loop predicate

.section __TEXT,__text,regular,pure_instructions
.global _tbl_u8
.p2align 4

_tbl_u8:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8, d9, [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x4, .Ldone

    smstart sm

    // Create predicate for partial table load
    whilelt p1.b, xzr, x3          // p1 active for [0, n_table)
    ld1b    {z16.b}, p1/z, [x0]    // load table, zero-fill beyond n_table

    mov     x8, #0
    whilelt pn9.b, x8, x4, vlx4

.Lloop:
    ld1b    {z0.b-z3.b}, pn9/z, [x1, x8]
    tbl     z0.b, {z16.b}, z0.b
    tbl     z1.b, {z16.b}, z1.b
    tbl     z2.b, {z16.b}, z2.b
    tbl     z3.b, {z16.b}, z3.b
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
