// reshape_fp32.s — FP32 reshape (contiguous copy) via SME2 streaming SVE
//
// void reshape_fp32(const float *src, float *dst, long n)
//
// Copies n floats from src to dst. Semantically represents a tensor reshape
// operation where the data layout is contiguous and only the logical shape
// changes. Provides a named kernel for compute graph tracking separate from
// copy_fp32.
//
// AAPCS: x0=src, x1=dst, x2=n

.section __TEXT,__text,regular,pure_instructions
.global _reshape_fp32
.p2align 4

_reshape_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x2, .Ldone

    smstart sm

    mov     x8, #0                 // element counter
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
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
    ldp     x29, x30, [sp], #80
    ret
