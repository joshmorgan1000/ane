// clamp_fp32.s — Clamp FP32 vector to [lo, hi] via SME2 streaming SVE
//
// void clamp_fp32(const float *input, float lo, float hi, float *output, long n)
//
// Computes output[i] = min(max(input[i], lo), hi) for i in [0, n).
// AAPCS: x0=input, s0=lo, s1=hi, x1=output, x2=n

.section __TEXT,__text,regular,pure_instructions
.global _clamp_fp32
.p2align 4

_clamp_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]
    str     s0, [sp, #80]           // lo
    str     s1, [sp, #84]           // hi

    cbz     x2, .Ldone

    smstart sm

    ptrue   p0.s
    ld1rw   {z8.s}, p0/z, [sp, #80]    // lo broadcast
    ld1rw   {z9.s}, p0/z, [sp, #84]    // hi broadcast

    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    fclamp  {z0.s-z3.s}, z8.s, z9.s    // clamp to [lo, hi] in single instruction
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
