// fused_rms_norm_scale_fp32.s — Fused RMSNorm scale: out[i] = input[i] * inv_rms * weight[i]
//
// void fused_rms_norm_scale_fp32(const float* input, const float* weight,
//                                 float* out, float inv_rms, long n)
// AAPCS: x0=input, x1=weight, x2=out, s0=inv_rms, x3=n

.section __TEXT,__text,regular,pure_instructions
.global _fused_rms_norm_scale_fp32
.p2align 4

_fused_rms_norm_scale_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]
    str     s0, [sp, #80]

    cbz     x3, .Lrns_done

    smstart sm

    ptrue   p0.s
    ld1rw   {z8.s}, p0/z, [sp, #80]

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lrns_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z8.s
    fmul    z1.s, p0/m, z1.s, z8.s
    fmul    z2.s, p0/m, z2.s, z8.s
    fmul    z3.s, p0/m, z3.s, z8.s
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z4.s
    fmul    z1.s, p0/m, z1.s, z5.s
    fmul    z2.s, p0/m, z2.s, z6.s
    fmul    z3.s, p0/m, z3.s, z7.s
    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lrns_loop

    smstop

.Lrns_done:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #96
    ret
