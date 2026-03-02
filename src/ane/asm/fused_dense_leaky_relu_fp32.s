// fused_dense_leaky_relu_fp32.s -- Fused dense (matvec) + bias + leaky ReLU
//
// Computes: temp = W @ x + bias, then leaky_relu(temp, alpha)
//   out[i] = temp[i] >= 0 ? temp[i] : alpha * temp[i]
//
// void fused_dense_leaky_relu_fp32(const float* W, int m, int n,
//                                   const float* x, const float* bias,
//                                   float alpha, float* out)
//
// AAPCS64: x0=W, x1=m, x2=n, x3=x, x4=bias, s0=alpha, x5=out
//
// Phase 1: Compute temp = W @ x + bias using dot products row-by-row
// Phase 2: Apply leaky ReLU element-wise using predicated select

.section __TEXT,__text,regular,pure_instructions
.global _fused_dense_leaky_relu_fp32
.p2align 4

_fused_dense_leaky_relu_fp32:
    stp     x29, x30, [sp, #-160]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8,  d9,  [sp, #96]
    stp     d10, d11, [sp, #112]
    stp     d12, d13, [sp, #128]
    stp     d14, d15, [sp, #144]

    // Early exit
    cbz     x1, .Lfdlr_done        // m == 0
    cbz     x2, .Lfdlr_done        // n == 0

    // Save arguments
    mov     x19, x0                 // W
    mov     x20, x1                 // m
    mov     x21, x2                 // n
    mov     x22, x3                 // x (input vector)
    mov     x23, x4                 // bias
    mov     x24, x5                 // out
    fmov    w25, s0                 // alpha (saved as int bits)

    // Allocate temp buffer: m * 4 bytes
    lsl     x0, x20, #2
    bl      _malloc
    mov     x26, x0                 // temp buffer

    // ================================================================
    // Phase 1: Compute temp = W @ x + bias using dot products
    // ================================================================

    smstart sm
    ptrue   p0.s

    mov     x8, #0                  // row index
    lsl     x9, x21, #2            // n * 4 (row stride in W)

.Lfdlr_matvec_loop:
    cmp     x8, x20
    b.ge    .Lfdlr_matvec_done

    // w_ptr = W + row * n * 4
    mul     x10, x8, x9
    add     x10, x19, x10

    // Accumulate dot product
    mov     z28.s, #0

    mov     x11, #0
    whilelt p1.s, x11, x21

.Lfdlr_dot_loop:
    ld1w    {z0.s}, p1/z, [x10, x11, lsl #2]
    ld1w    {z1.s}, p1/z, [x22, x11, lsl #2]
    fmla    z28.s, p1/m, z0.s, z1.s

    incw    x11
    whilelt p1.s, x11, x21
    b.first .Lfdlr_dot_loop

    // Horizontal sum
    faddv   s2, p0, z28.s

    // Add bias
    ldr     s3, [x23, x8, lsl #2]
    fadd    s2, s2, s3

    // Store temp[row]
    str     s2, [x26, x8, lsl #2]

    add     x8, x8, #1
    b       .Lfdlr_matvec_loop

.Lfdlr_matvec_done:
    // ================================================================
    // Phase 2: Apply leaky ReLU
    // out[i] = temp[i] >= 0 ? temp[i] : alpha * temp[i]
    // ================================================================

    fmov    s4, w25                 // restore alpha
    mov     z30.s, s4               // broadcast alpha
    mov     z31.s, #0               // broadcast zero for comparison

    mov     x8, #0
    whilelt pn9.s, x8, x20, vlx4

.Lfdlr_leaky_loop:
    // Load temp
    ld1w    {z0.s-z3.s}, pn9/z, [x26, x8, lsl #2]

    // Compute alpha * temp
    fmul    z4.s, z0.s, z30.s
    fmul    z5.s, z1.s, z30.s
    fmul    z6.s, z2.s, z30.s
    fmul    z7.s, z3.s, z30.s

    // Compare: temp >= 0 ? Use fcmge to get predicate
    fcmge   p1.s, p0/z, z0.s, z31.s
    fcmge   p2.s, p0/z, z1.s, z31.s
    fcmge   p3.s, p0/z, z2.s, z31.s
    fcmge   p4.s, p0/z, z3.s, z31.s

    // Select: where temp >= 0 keep temp, else use alpha*temp
    // For lanes where p is true (>=0), keep z0; else use z4
    sel     z0.s, p1, z0.s, z4.s
    sel     z1.s, p2, z1.s, z5.s
    sel     z2.s, p3, z2.s, z6.s
    sel     z3.s, p4, z3.s, z7.s

    // Store output
    st1w    {z0.s-z3.s}, pn9, [x24, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x20, vlx4
    b.first .Lfdlr_leaky_loop

    smstop

    // Free temp buffer
    mov     x0, x26
    bl      _free

.Lfdlr_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     d8,  d9,  [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    ldp     x29, x30, [sp], #160
    ret
