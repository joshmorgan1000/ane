// tbx_u8.s — SVE2 byte-level table lookup with merge (TBX)
//
// void tbx_u8(const uint8_t *table, const uint8_t *indices,
//             const uint8_t *fallback, uint8_t *output,
//             long n_table, long n);
//
// Like TBL but out-of-range indices (>= n_table) preserve the corresponding
// fallback value instead of zeroing.
//
// Implementation: Do TBL, then compare indices with n_table and use predicated
// selection to restore fallback for out-of-range indices.
//
// Register allocation:
//   x0 = table ptr
//   x1 = indices ptr
//   x2 = fallback ptr
//   x3 = output ptr
//   x4 = n_table
//   x5 = n (byte count)
//   x8 = loop counter
//   z16 = table vector
//   z17 = n_table broadcast
//   p1 = table load predicate
//   p2-p5 = index comparison predicates
//   pn9 = loop predicate

.section __TEXT,__text,regular,pure_instructions
.global _tbx_u8
.p2align 4

_tbx_u8:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8, d9, [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x5, .Ldone

    smstart sm

    // Create predicate for partial table load
    whilelt p1.b, xzr, x4          // p1 active for [0, n_table)
    ld1b    {z16.b}, p1/z, [x0]    // load table, zero-fill beyond n_table

    // Broadcast n_table for comparisons
    mov     z17.b, w4

    ptrue   p0.b

    mov     x8, #0
    whilelt pn9.b, x8, x5, vlx4

.Lloop:
    ld1b    {z0.b-z3.b}, pn9/z, [x2, x8]   // load fallback into z0-z3
    ld1b    {z4.b-z7.b}, pn9/z, [x1, x8]   // load indices into z4-z7

    // Do TBL lookup into z8-z11 (preserves fallback in z0-z3)
    tbl     z8.b, {z16.b}, z4.b
    tbl     z9.b, {z16.b}, z5.b
    tbl     z10.b, {z16.b}, z6.b
    tbl     z11.b, {z16.b}, z7.b

    // Compare: p2/p3/p4/p5 = (index < n_table) — where TBL result is valid
    cmplo   p2.b, p0/z, z4.b, z17.b
    cmplo   p3.b, p0/z, z5.b, z17.b
    cmplo   p4.b, p0/z, z6.b, z17.b
    cmplo   p5.b, p0/z, z7.b, z17.b

    // Merge: where index < n_table, use TBL result; else keep fallback
    sel     z0.b, p2, z8.b, z0.b
    sel     z1.b, p3, z9.b, z1.b
    sel     z2.b, p4, z10.b, z2.b
    sel     z3.b, p5, z11.b, z3.b

    st1b    {z0.b-z3.b}, pn9, [x3, x8]     // store
    incb    x8, all, mul #4
    whilelt pn9.b, x8, x5, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d14, d15, [sp, #64]
    ldp     d12, d13, [sp, #48]
    ldp     d10, d11, [sp, #32]
    ldp     d8, d9, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret
