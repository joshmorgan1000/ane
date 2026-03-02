// fma_fp32.s — Element-wise FP32 fused multiply-add via SME2 streaming SVE
//
// void fma_fp32(const float *a, const float *b, const float *c, float *out, long n)
//
// Computes out[i] = a[i] * b[i] + c[i] for i in [0, n).

.section __TEXT,__text,regular,pure_instructions
.global _fma_fp32
.p2align 4

_fma_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]

    cbz     x4, .Ldone

    smstart sm

    ptrue   p0.s
    mov     x8, #0
    whilelt pn9.s, x8, x4, vlx4

.Lloop:
    ld1w    {z0.s-z3.s},   pn9/z, [x0, x8, lsl #2]    // a
    ld1w    {z4.s-z7.s},   pn9/z, [x1, x8, lsl #2]    // b
    ld1w    {z8.s-z11.s},  pn9/z, [x2, x8, lsl #2]    // c

    // out = a*b + c  →  c += a*b  (fmla: addend = z8..z11)
    fmla    z8.s,  p0/m, z0.s, z4.s
    fmla    z9.s,  p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s

    st1w    {z8.s-z11.s}, pn9, [x3, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x4, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10,  d11,  [sp, #32]
    ldp     d12,  d13,  [sp, #48]
    ldp     d14,  d15,  [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
