// fused_glu_fp32.s -- Fused Gated Linear Unit with sigmoid gate
//
// Computes: out[i] = x[i] * sigmoid(x[i])
//
// This is the generic GLU with sigmoid gating. SiLU (Swish) is a special case
// of GLU where gate = input. This kernel provides the building block for
// other GLU variants (e.g., feed the gate through a linear projection first).
//
// For two-input GLU (out = up * sigmoid(gate)), use fused_silu_gate_mul_fp32
// with gate/up split. This kernel handles the single-input case.
//
// void fused_glu_fp32(const float* x, long n, float* out)
//
// AAPCS64: x0=x, x1=n, x2=out

.section __TEXT,__text,regular,pure_instructions
.global _fused_glu_fp32
.p2align 4

_fused_glu_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8,  d9,  [sp, #32]
    stp     d10, d11, [sp, #48]
    stp     d12, d13, [sp, #64]
    stp     d14, d15, [sp, #80]

    cbz     x1, .Lfglu_done

    mov     x19, x0                 // input
    mov     x20, x2                 // output

    smstart                         // streaming SVE + ZA (needed for exp polynomial)
    ptrue   p0.s

    // Load constants
    adr     x9, .Lfglu_const
    ld1rw   {z8.s}, p0/z, [x9]         // log2(e)
    ld1rw   {z9.s}, p0/z, [x9, #4]     // ln(2)
    ld1rw   {z10.s}, p0/z, [x9, #8]    // 1.0
    ld1rw   {z11.s}, p0/z, [x9, #12]   // 0.5 = 1/2!
    ld1rw   {z12.s}, p0/z, [x9, #16]   // 1/6 = 1/3!
    ld1rw   {z13.s}, p0/z, [x9, #20]   // 1/24 = 1/4!
    ld1rw   {z14.s}, p0/z, [x9, #24]   // 1/120 = 1/5!
    ld1rw   {z15.s}, p0/z, [x9, #28]   // 1/720 = 1/6!
    ld1rw   {z25.s}, p0/z, [x9, #32]   // 88.0
    ld1rw   {z26.s}, p0/z, [x9, #36]   // -88.0

    mov     w11, #0                     // ZA vector select

    mov     x8, #0
    whilelt pn9.s, x8, x1, vlx4

.Lfglu_loop:
    // Load input x
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Save x for final multiply
    mov     z16.d, z0.d
    mov     z17.d, z1.d
    mov     z18.d, z2.d
    mov     z19.d, z3.d

    // Negate for exp(-x) to compute sigmoid
    fneg    z0.s, p0/m, z0.s
    fneg    z1.s, p0/m, z1.s
    fneg    z2.s, p0/m, z2.s
    fneg    z3.s, p0/m, z3.s

    // Clamp to [-88, 88]
    fmin    z0.s, p0/m, z0.s, z25.s
    fmax    z0.s, p0/m, z0.s, z26.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmax    z1.s, p0/m, z1.s, z26.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmax    z2.s, p0/m, z2.s, z26.s
    fmin    z3.s, p0/m, z3.s, z25.s
    fmax    z3.s, p0/m, z3.s, z26.s

    // Range reduce: n_val = round(-x * log2(e))
    fmul    z20.s, z0.s, z8.s
    fmul    z21.s, z1.s, z8.s
    fmul    z22.s, z2.s, z8.s
    fmul    z23.s, z3.s, z8.s
    frintn  z20.s, p0/m, z20.s
    frintn  z21.s, p0/m, z21.s
    frintn  z22.s, p0/m, z22.s
    frintn  z23.s, p0/m, z23.s

    // r = -x - n_val * ln(2)
    movprfx z4, z0
    fmls    z4.s, p0/m, z20.s, z9.s
    movprfx z5, z1
    fmls    z5.s, p0/m, z21.s, z9.s
    movprfx z6, z2
    fmls    z6.s, p0/m, z22.s, z9.s
    movprfx z7, z3
    fmls    z7.s, p0/m, z23.s, z9.s

    // Polynomial exp(r) via FMLA VGx4 with ZA
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

    // z0-z3 = exp(-x)
    // sigmoid = 1 / (1 + exp(-x))
    fadd    z0.s, z0.s, z10.s
    fadd    z1.s, z1.s, z10.s
    fadd    z2.s, z2.s, z10.s
    fadd    z3.s, z3.s, z10.s

    movprfx z4, z10
    fdiv    z4.s, p0/m, z4.s, z0.s
    movprfx z5, z10
    fdiv    z5.s, p0/m, z5.s, z1.s
    movprfx z6, z10
    fdiv    z6.s, p0/m, z6.s, z2.s
    movprfx z7, z10
    fdiv    z7.s, p0/m, z7.s, z3.s

    // GLU output = x * sigmoid(x)
    zero    {za}
    fmla    za.s[w11, 0, vgx4], {z16.s-z19.s}, {z4.s-z7.s}
    mova    {z0.s-z3.s}, za.s[w11, 0, vgx4]

    // Store
    st1w    {z0.s-z3.s}, pn9, [x20, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x1, vlx4
    b.first .Lfglu_loop

    smstop

.Lfglu_done:
    ldp     x19, x20, [sp, #16]
    ldp     d8,  d9,  [sp, #32]
    ldp     d10, d11, [sp, #48]
    ldp     d12, d13, [sp, #64]
    ldp     d14, d15, [sp, #80]
    ldp     x29, x30, [sp], #96
    ret

.p2align 2
.Lfglu_const:
    .long   0x3FB8AA3B  // log2(e)
    .long   0x3F317218  // ln(2)
    .long   0x3F800000  // 1.0
    .long   0x3F000000  // 0.5 = 1/2!
    .long   0x3E2AAAAB  // 1/6 = 1/3!
    .long   0x3D2AAAAB  // 1/24 = 1/4!
    .long   0x3C088889  // 1/120 = 1/5!
    .long   0x3AB60B61  // 1/720 = 1/6!
    .long   0x42B00000  // 88.0
    .long   0xC2B00000  // -88.0
