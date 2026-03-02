// where_fp32.s — FP32 ternary select via SME2 streaming SVE
//
// void where_fp32(const uint32_t *cond, const float *a, const float *b, float *out, long n)
// AAPCS: x0=cond, x1=a, x2=b, x3=out, x4=n
//
// Computes out[i] = cond[i] != 0 ? a[i] : b[i]
// Uses integer comparison on uint32_t condition array to generate predicate,
// then sel to blend between a[i] and b[i].
//
// Processing: 4 vectors (64 elements) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _where_fp32
.p2align 4

_where_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x4, .Ldone

    smstart sm

    ptrue   p0.s
    mov     z9.d, #0              // zero for integer comparison

    mov     x8, #0
    whilelt pn9.s, x8, x4, vlx4

.Lloop:
    // Load condition as uint32
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]

    // Load a and b
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]
    ld1w    {z12.s-z15.s}, pn9/z, [x2, x8, lsl #2]

    // Compare condition != 0 using integer semantics
    // cmpne gives predicate: true where cond != 0
    cmpne   p1.s, p0/z, z0.s, z9.s
    cmpne   p2.s, p0/z, z1.s, z9.s
    cmpne   p3.s, p0/z, z2.s, z9.s
    cmpne   p4.s, p0/z, z3.s, z9.s

    // sel: p_true → a[i], p_false → b[i]
    // sel z_dst, p_pred, z_true, z_false
    // Where predicate is true, dst = true; where false, dst = false
    sel     z0.s, p1, z4.s, z12.s
    sel     z1.s, p2, z5.s, z13.s
    sel     z2.s, p3, z6.s, z14.s
    sel     z3.s, p4, z7.s, z15.s

    st1w    {z0.s-z3.s}, pn9, [x3, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x4, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
