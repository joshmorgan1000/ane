// fill_fp32.s — Broadcast scalar to FP32 array via SME2 streaming SVE
//
// void fill_fp32(float *output, float value, long n)
//
// Sets output[i] = value for i in [0, n).
// AAPCS: x0=output, s0=value, x1=n

.section __TEXT,__text,regular,pure_instructions
.global _fill_fp32
.p2align 4

_fill_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]
    str     s0, [sp, #80]

    cbz     x1, .Ldone

    smstart sm

    ptrue   p0.s
    ld1rw   {z0.s}, p0/z, [sp, #80]
    mov     z1.d, z0.d
    mov     z2.d, z0.d
    mov     z3.d, z0.d

    mov     x8, #0
    whilelt pn9.s, x8, x1, vlx4

.Lloop:
    st1w    {z0.s-z3.s}, pn9, [x0, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x1, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #96
    ret
