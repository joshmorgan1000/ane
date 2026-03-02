// fused_dense_elu_fp32.s -- Fused dense (matvec) + bias + ELU activation
//
// Computes: temp = W @ x + bias, then ELU(temp, alpha)
//   ELU(x) = x >= 0 ? x : alpha * (exp(x) - 1)
//
// void fused_dense_elu_fp32(const float* W, int m, int n,
//                            const float* x, const float* bias,
//                            float alpha, float* out)
//
// AAPCS64: x0=W, x1=m, x2=n, x3=x, x4=bias, s0=alpha, x5=out

.section __TEXT,__text,regular,pure_instructions
.global _fused_dense_elu_fp32
.p2align 4

_fused_dense_elu_fp32:
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

    cbz     x1, .Lfde_done
    cbz     x2, .Lfde_done

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
    // Phase 1: Compute temp = W @ x + bias
    // ================================================================

    smstart                         // streaming SVE + ZA (needed for exp polynomial)
    ptrue   p0.s

    mov     x8, #0
    lsl     x9, x21, #2

.Lfde_matvec_loop:
    cmp     x8, x20
    b.ge    .Lfde_matvec_done

    mul     x10, x8, x9
    add     x10, x19, x10

    mov     z28.s, #0

    mov     x11, #0
    whilelt p1.s, x11, x21

.Lfde_dot_loop:
    ld1w    {z0.s}, p1/z, [x10, x11, lsl #2]
    ld1w    {z1.s}, p1/z, [x22, x11, lsl #2]
    fmla    z28.s, p1/m, z0.s, z1.s

    incw    x11
    whilelt p1.s, x11, x21
    b.first .Lfde_dot_loop

    faddv   s2, p0, z28.s
    ldr     s3, [x23, x8, lsl #2]
    fadd    s2, s2, s3
    str     s2, [x26, x8, lsl #2]

    add     x8, x8, #1
    b       .Lfde_matvec_loop

.Lfde_matvec_done:
    // ================================================================
    // Phase 2: Apply ELU
    // out[i] = temp[i] >= 0 ? temp[i] : alpha * (exp(temp[i]) - 1)
    // ================================================================

    // Load exp constants
    adr     x9, .Lfde_const
    ld1rw   {z8.s}, p0/z, [x9]         // log2(e)
    ld1rw   {z9.s}, p0/z, [x9, #4]     // ln(2)
    ld1rw   {z10.s}, p0/z, [x9, #8]    // 1.0
    ld1rw   {z11.s}, p0/z, [x9, #12]   // 0.5 = 1/2!
    ld1rw   {z12.s}, p0/z, [x9, #16]   // 1/6 = 1/3!
    ld1rw   {z13.s}, p0/z, [x9, #20]   // 1/24 = 1/4!
    ld1rw   {z14.s}, p0/z, [x9, #24]   // 1/120 = 1/5!
    ld1rw   {z15.s}, p0/z, [x9, #28]   // 1/720 = 1/6!
    ld1rw   {z27.s}, p0/z, [x9, #32]   // 88.0
    ld1rw   {z25.s}, p0/z, [x9, #36]   // -88.0

    // Restore and broadcast alpha
    fmov    s4, w25
    mov     z29.s, s4                   // broadcast alpha
    mov     z30.s, #0                   // broadcast zero

    mov     w11, #0                     // ZA row select

    mov     x8, #0
    whilelt pn9.s, x8, x20, vlx4

.Lfde_elu_loop:
    // Load temp (x values)
    ld1w    {z0.s-z3.s}, pn9/z, [x26, x8, lsl #2]

    // Save original x values
    mov     z16.d, z0.d
    mov     z17.d, z1.d
    mov     z18.d, z2.d
    mov     z19.d, z3.d

    // Clamp for exp
    fmin    z0.s, p0/m, z0.s, z27.s
    fmax    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z27.s
    fmax    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z27.s
    fmax    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z27.s
    fmax    z3.s, p0/m, z3.s, z25.s

    // Range reduce
    fmul    z20.s, z0.s, z8.s
    fmul    z21.s, z1.s, z8.s
    fmul    z22.s, z2.s, z8.s
    fmul    z23.s, z3.s, z8.s
    frintn  z20.s, p0/m, z20.s
    frintn  z21.s, p0/m, z21.s
    frintn  z22.s, p0/m, z22.s
    frintn  z23.s, p0/m, z23.s

    // r = x - n_val * ln(2)
    movprfx z4, z0
    fmls    z4.s, p0/m, z20.s, z9.s
    movprfx z5, z1
    fmls    z5.s, p0/m, z21.s, z9.s
    movprfx z6, z2
    fmls    z6.s, p0/m, z22.s, z9.s
    movprfx z7, z3
    fmls    z7.s, p0/m, z23.s, z9.s

    // Polynomial exp(r)
    zero    {za}

    fmla    za.s[w11, 0, vgx4], {z4.s-z7.s}, z10.s

    fmul    z0.s, z4.s, z4.s
    fmul    z1.s, z5.s, z5.s
    fmul    z2.s, z6.s, z6.s
    fmul    z3.s, z7.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z11.s

    fmul    z0.s, z0.s, z4.s
    fmul    z1.s, z1.s, z5.s
    fmul    z2.s, z2.s, z6.s
    fmul    z3.s, z3.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z12.s

    fmul    z0.s, z0.s, z4.s
    fmul    z1.s, z1.s, z5.s
    fmul    z2.s, z2.s, z6.s
    fmul    z3.s, z3.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z13.s

    fmul    z0.s, z0.s, z4.s
    fmul    z1.s, z1.s, z5.s
    fmul    z2.s, z2.s, z6.s
    fmul    z3.s, z3.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z14.s

    fmul    z0.s, z0.s, z4.s
    fmul    z1.s, z1.s, z5.s
    fmul    z2.s, z2.s, z6.s
    fmul    z3.s, z3.s, z7.s
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z15.s

    mova    {z0.s-z3.s}, za.s[w11, 0, vgx4]
    fadd    z0.s, z0.s, z10.s
    fadd    z1.s, z1.s, z10.s
    fadd    z2.s, z2.s, z10.s
    fadd    z3.s, z3.s, z10.s

    fcvtzs  z20.s, p0/m, z20.s
    fcvtzs  z21.s, p0/m, z21.s
    fcvtzs  z22.s, p0/m, z22.s
    fcvtzs  z23.s, p0/m, z23.s
    fscale  z0.s, p0/m, z0.s, z20.s
    fscale  z1.s, p0/m, z1.s, z21.s
    fscale  z2.s, p0/m, z2.s, z22.s
    fscale  z3.s, p0/m, z3.s, z23.s

    // z0-z3 = exp(x)
    // ELU negative part: alpha * (exp(x) - 1)
    fsub    z0.s, z0.s, z10.s      // exp(x) - 1
    fsub    z1.s, z1.s, z10.s
    fsub    z2.s, z2.s, z10.s
    fsub    z3.s, z3.s, z10.s

    fmul    z0.s, p0/m, z0.s, z29.s    // alpha * (exp(x) - 1)
    fmul    z1.s, p0/m, z1.s, z29.s
    fmul    z2.s, p0/m, z2.s, z29.s
    fmul    z3.s, p0/m, z3.s, z29.s

    // Select: where x >= 0, use original x; else use alpha*(exp(x)-1)
    fcmge   p1.s, p0/z, z16.s, z30.s
    fcmge   p2.s, p0/z, z17.s, z30.s
    fcmge   p3.s, p0/z, z18.s, z30.s
    fcmge   p4.s, p0/z, z19.s, z30.s

    sel     z0.s, p1, z16.s, z0.s
    sel     z1.s, p2, z17.s, z1.s
    sel     z2.s, p3, z18.s, z2.s
    sel     z3.s, p4, z19.s, z3.s

    // Store
    st1w    {z0.s-z3.s}, pn9, [x24, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x20, vlx4
    b.first .Lfde_elu_loop

    smstop

    mov     x0, x26
    bl      _free

.Lfde_done:
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

.p2align 2
.Lfde_const:
    .long   0x3FB8AA3B  // log2(e)
    .long   0x3F317218  // ln(2)
    .long   0x3F800000  // 1.0
    .long   0x3F000000  // 0.5
    .long   0x3E2AAAAB  // 1/6
    .long   0x3D2AAAAB  // 1/24
    .long   0x3C088889  // 1/120
    .long   0x3AB60B61  // 1/720
    .long   0x42B00000  // 88.0
    .long   0xC2B00000  // -88.0
