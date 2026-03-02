// relu_backward_fp32.s — ReLU backward pass via SME2 streaming SVE
//
// void relu_backward_fp32(const float* dy, const float* x, float* dx, long n)
// AAPCS: x0=dy (upstream gradient), x1=x (forward input), x2=dx (output gradient), x3=n
//
// Computes: dx[i] = x[i] > 0 ? dy[i] : 0
// Simple predicated multiply: if x > 0, keep dy; else zero.
//
// Processing: 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _relu_backward_fp32
.p2align 4

_relu_backward_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x3, .Lrb_done

    smstart sm

    ptrue   p0.s
    mov     z9.s, #0              // zero for comparison

    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lrb_loop:
    // Load dy into z0-z3 and x into z4-z7
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]   // dy
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]   // x

    // Compare x > 0: fcmgt creates predicate (x > 0)
    fcmgt   p1.s, p0/z, z4.s, z9.s
    fcmgt   p2.s, p0/z, z5.s, z9.s
    fcmgt   p3.s, p0/z, z6.s, z9.s
    fcmgt   p4.s, p0/z, z7.s, z9.s

    // Selective move: dx = (x > 0) ? dy : 0
    // Use sel: if predicate true, keep dy; else use zero (already in z0-z3 as zeroed by pn9)
    mov     z10.d, z9.d
    mov     z11.d, z9.d
    mov     z12.d, z9.d
    mov     z13.d, z9.d
    sel     z0.s, p1, z0.s, z10.s
    sel     z1.s, p2, z1.s, z11.s
    sel     z2.s, p3, z2.s, z12.s
    sel     z3.s, p4, z3.s, z13.s

    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lrb_loop

    smstop

.Lrb_done:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
