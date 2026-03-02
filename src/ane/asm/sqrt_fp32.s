// sqrt_fp32.s — Element-wise FP32 square root via SME2 streaming SVE
//
// void sqrt_fp32(const float *input, float *output, long n)
//
// Computes output[i] = sqrt(input[i]) for i in [0, n).

.section __TEXT,__text,regular,pure_instructions
.global _sqrt_fp32
.p2align 4

_sqrt_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]

    cbz     x2, .Ldone

    smstart sm

    ptrue   p0.s
    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    fsqrt   z0.s, p0/m, z0.s
    fsqrt   z1.s, p0/m, z1.s
    fsqrt   z2.s, p0/m, z2.s
    fsqrt   z3.s, p0/m, z3.s
    st1w    {z0.s-z3.s}, pn9, [x1, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x2, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10,  d11,  [sp, #32]
    ldp     d12,  d13,  [sp, #48]
    ldp     d14,  d15,  [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
