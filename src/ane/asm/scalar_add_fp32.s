// scalar_add_fp32.s — Add scalar to vector via SME2 streaming SVE
//
// void scalar_add_fp32(const float *input, float scalar, float *output, long n)
//
// Computes output[i] = input[i] + scalar for i in [0, n).
// AAPCS: x0=input, s0=scalar, x1=output, x2=n

.section __TEXT,__text,regular,pure_instructions
.global _scalar_add_fp32
.p2align 4

_scalar_add_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]
    str     s0, [sp, #80]

    cbz     x2, .Ldone

    smstart sm

    ptrue   p0.s
    ld1rw   {z8.s}, p0/z, [sp, #80]

    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    fadd    z0.s, p0/m, z0.s, z8.s
    fadd    z1.s, p0/m, z1.s, z8.s
    fadd    z2.s, p0/m, z2.s, z8.s
    fadd    z3.s, p0/m, z3.s, z8.s
    st1w    {z0.s-z3.s}, pn9, [x1, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x2, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #96
    ret
