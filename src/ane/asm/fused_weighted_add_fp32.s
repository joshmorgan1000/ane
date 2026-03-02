// fused_weighted_add_fp32.s — Fused weighted add: out[i] = acc[i] + w * expert_out[i]
//
// void fused_weighted_add_fp32(const float* acc, const float* expert_out,
//                               float* out, float w, long n)
// AAPCS: x0=acc, x1=expert_out, x2=out, s0=w, x3=n

.section __TEXT,__text,regular,pure_instructions
.global _fused_weighted_add_fp32
.p2align 4

_fused_weighted_add_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]
    str     s0, [sp, #80]

    cbz     x3, .Lwa_done

    smstart sm
    ptrue   p0.s
    ld1rw   {z8.s}, p0/z, [sp, #80]

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lwa_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x1, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z8.s
    fmul    z1.s, p0/m, z1.s, z8.s
    fmul    z2.s, p0/m, z2.s, z8.s
    fmul    z3.s, p0/m, z3.s, z8.s
    ld1w    {z4.s-z7.s}, pn9/z, [x0, x8, lsl #2]
    fadd    z0.s, p0/m, z0.s, z4.s
    fadd    z1.s, p0/m, z1.s, z5.s
    fadd    z2.s, p0/m, z2.s, z6.s
    fadd    z3.s, p0/m, z3.s, z7.s
    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lwa_loop

    smstop

.Lwa_done:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #96
    ret
