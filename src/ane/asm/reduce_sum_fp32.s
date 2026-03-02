// reduce_sum_fp32.s — Sum reduction via SME2 streaming SVE
//
// float reduce_sum_fp32(const float *input, long n)
//
// Returns sum(input[i]) for i in [0, n).

.section __TEXT,__text,regular,pure_instructions
.global _reduce_sum_fp32
.p2align 4

_reduce_sum_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cmp     x1, #0
    b.le    .Lzero

    smstart sm

    ptrue   p0.s
    mov     z8.d, #0
    mov     z9.d, #0
    mov     z10.d, #0
    mov     z11.d, #0

    mov     x8, #0
    whilelt pn9.s, x8, x1, vlx4

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    fadd    z8.s,  p0/m, z8.s,  z0.s
    fadd    z9.s,  p0/m, z9.s,  z1.s
    fadd    z10.s, p0/m, z10.s, z2.s
    fadd    z11.s, p0/m, z11.s, z3.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x1, vlx4
    b.first .Lloop

    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s
    faddv   s0, p0, z8.s
    str     s0, [sp, #80]

    smstop

    ldr     s0, [sp, #80]
    b       .Ldone

.Lzero:
    fmov    s0, wzr

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #96
    ret

