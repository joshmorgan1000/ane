// tanh_backward_fp32.s — Tanh backward pass via SME2 streaming SVE
//
// void tanh_backward_fp32(const float* dy, const float* y, float* dx, long n)
// AAPCS: x0=dy (upstream gradient), x1=y (tanh output), x2=dx (output gradient), x3=n
//
// Computes: dx[i] = dy[i] * (1 - y[i]^2)
//
// Processing: 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _tanh_backward_fp32
.p2align 4

_tanh_backward_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x3, .Ltb_done

    smstart sm

    ptrue   p0.s
    adr     x9, .Ltb_const
    ld1rw   {z8.s}, p0/z, [x9]     // 1.0

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Ltb_loop:
    // Load dy and y
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]   // dy
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]   // y

    // Compute y^2 in z9-z12
    fmul    z9.s, z4.s, z4.s
    fmul    z10.s, z5.s, z5.s
    fmul    z11.s, z6.s, z6.s
    fmul    z12.s, z7.s, z7.s

    // Compute (1 - y^2) in z9-z12
    movprfx z9, z8
    fsub    z9.s, p0/m, z9.s, z4.s
    fmul    z9.s, p0/m, z9.s, z4.s  // Actually: 1 - y, then y, then (1-y)*y. Let's redo.
    // Actually we want 1 - y^2. Let me recalculate:
    // z9 starts with 1.0 from z8
    // We need: z9 = 1 - (y*y)
    // So: z9 = z8 - z4*z4

    movprfx z9, z8
    fmls    z9.s, p0/m, z4.s, z4.s  // z9 = z8 - z4*z4 = 1 - y^2
    movprfx z10, z8
    fmls    z10.s, p0/m, z5.s, z5.s
    movprfx z11, z8
    fmls    z11.s, p0/m, z6.s, z6.s
    movprfx z12, z8
    fmls    z12.s, p0/m, z7.s, z7.s

    // Compute dy * (1 - y^2) in z0-z3
    fmul    z0.s, p0/m, z0.s, z9.s
    fmul    z1.s, p0/m, z1.s, z10.s
    fmul    z2.s, p0/m, z2.s, z11.s
    fmul    z3.s, p0/m, z3.s, z12.s

    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Ltb_loop

    smstop

.Ltb_done:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret

.p2align 2
.Ltb_const:
    .long   0x3F800000  // 1.0
