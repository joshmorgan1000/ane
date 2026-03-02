// mul_fp32.s — Element-wise FP32 multiplication via SME2 streaming SVE
//
// void mul_fp32(const float *a, const float *b, float *c, long n)
//
// Computes c[i] = a[i] * b[i] for i in [0, n).
//
// Loads both inputs as 4-vector groups (64 floats / 256 bytes per iteration
// on M4), performs 4 predicated fmul instructions, stores result.

.section __TEXT,__text,regular,pure_instructions
.global _mul_fp32
.p2align 4

_mul_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]

    cbz     x3, .Ldone

    smstart sm

    ptrue   p0.s
    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z4.s
    fmul    z1.s, p0/m, z1.s, z5.s
    fmul    z2.s, p0/m, z2.s, z6.s
    fmul    z3.s, p0/m, z3.s, z7.s
    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10,  d11,  [sp, #32]
    ldp     d12,  d13,  [sp, #48]
    ldp     d14,  d15,  [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
