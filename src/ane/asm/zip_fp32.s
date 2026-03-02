// zip_fp32.s — Interleave two FP32 arrays via SME2 streaming SVE
//
// void zip_fp32(const float *a, const float *b, float *out, long n)
//
// Produces out[2i] = a[i], out[2i+1] = b[i] for i in [0, n).
// Output length is 2*n.
//
// Uses zip1/zip2 to interleave vector halves. Each iteration
// processes SVLs input elements (16 on M4) producing 2*SVLs output elements.

.section __TEXT,__text,regular,pure_instructions
.global _zip_fp32
.p2align 4

_zip_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x3, .Ldone

    smstart sm

    cntw    x8                  // SVLs (16 on M4)
    lsl     x9, x3, #1         // 2*n = total output elements
    mov     x10, #0             // input index
    mov     x11, #0             // output index

.Lloop:
    whilelt p1.s, x10, x3
    b.none  .Lexit

    ld1w    {z0.s}, p1/z, [x0, x10, lsl #2]    // a
    ld1w    {z1.s}, p1/z, [x1, x10, lsl #2]    // b

    zip1    z2.s, z0.s, z1.s       // [a0,b0,a1,b1,...,a7,b7]
    zip2    z3.s, z0.s, z1.s       // [a8,b8,a9,b9,...,a15,b15]

    // Store with output predicates (handles tail correctly)
    whilelt p2.s, x11, x9
    st1w    {z2.s}, p2, [x2, x11, lsl #2]
    add     x11, x11, x8

    whilelt p3.s, x11, x9
    st1w    {z3.s}, p3, [x2, x11, lsl #2]
    add     x11, x11, x8

    incw    x10
    b       .Lloop

.Lexit:
    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
