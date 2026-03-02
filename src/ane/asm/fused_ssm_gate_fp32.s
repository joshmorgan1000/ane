// fused_ssm_gate_fp32.s — Fused SSM gate: out[i] = sigmoid(gate[i]) * y[i]
//
// Computes: out[i] = sigmoid(gate[i]) * y[i]
// Uses FMLA VGx4 with ZA accumulation for sigmoid polynomial
//
// void fused_ssm_gate_fp32(const float* gate, const float* y,
//                           float* out, long n)
// AAPCS: x0=gate, x1=y, x2=out, x3=n

.section __TEXT,__text,regular,pure_instructions
.global _fused_ssm_gate_fp32
.p2align 4

_fused_ssm_gate_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8,  d9,  [sp, #32]
    stp     d10, d11, [sp, #48]
    stp     d12, d13, [sp, #64]
    stp     d14,  d15,  [sp, #80]

    cbz     x3, .Lsg_done
    mov     x19, x0             // save gate ptr
    mov     x20, x1             // save y ptr

    smstart                     // streaming SVE + ZA access
    ptrue   p0.s

    // Load constants
    adr     x9, .Lsg_const
    ld1rw   {z16.s}, p0/z, [x9]        // log2(e)
    ld1rw   {z17.s}, p0/z, [x9, #4]    // ln(2)
    ld1rw   {z9.s}, p0/z, [x9, #8]     // 1.0
    ld1rw   {z10.s}, p0/z, [x9, #12]   // 0.5 = 1/2!
    ld1rw   {z11.s}, p0/z, [x9, #16]   // 1/6 = 1/3!
    ld1rw   {z12.s}, p0/z, [x9, #20]   // 1/24 = 1/4!
    ld1rw   {z13.s}, p0/z, [x9, #24]   // 1/120 = 1/5!
    ld1rw   {z14.s}, p0/z, [x9, #28]   // 1/720 = 1/6!
    ld1rw   {z15.s}, p0/z, [x9, #32]   // 1/5040 = 1/7!
    ld1rw   {z25.s}, p0/z, [x9, #36]   // 88.0
    ld1rw   {z26.s}, p0/z, [x9, #40]   // -88.0

    mov     w11, #0                     // ZA vector select register
    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lsg_loop:
    // Load gate values
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Negate: -gate
    fneg    z0.s, p0/m, z0.s
    fneg    z1.s, p0/m, z1.s
    fneg    z2.s, p0/m, z2.s
    fneg    z3.s, p0/m, z3.s

    // Clamp to [-88, 88]
    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s
    fmax    z0.s, p0/m, z0.s, z26.s
    fmax    z1.s, p0/m, z1.s, z26.s
    fmax    z2.s, p0/m, z2.s, z26.s
    fmax    z3.s, p0/m, z3.s, z26.s

    // Range reduce: n_val = round(-gate * log2e)
    fmul    z28.s, z0.s, z16.s
    fmul    z29.s, z1.s, z16.s
    fmul    z30.s, z2.s, z16.s
    fmul    z31.s, z3.s, z16.s
    frintn  z28.s, p0/m, z28.s
    frintn  z29.s, p0/m, z29.s
    frintn  z30.s, p0/m, z30.s
    frintn  z31.s, p0/m, z31.s

    // r = -gate - n_val * ln2 → z4-z7
    movprfx z4, z0
    fmls    z4.s, p0/m, z28.s, z17.s
    movprfx z5, z1
    fmls    z5.s, p0/m, z29.s, z17.s
    movprfx z6, z2
    fmls    z6.s, p0/m, z30.s, z17.s
    movprfx z7, z3
    fmls    z7.s, p0/m, z31.s, z17.s

    // Polynomial exp(r) via FMLA VGx4 with ZA accumulation
    // p(r) = 1 + r/2! + r²/3! + r³/4! + r⁴/5! + r⁵/6! + r⁶/7!
    zero    {za}

    // Accumulate c1*r (c1=1.0)
    fmla    za.s[w11, 0, vgx4], {z4.s-z7.s}, z9.s

    // r²
    fmul    z0.s, z4.s, z4.s
    fmul    z1.s, z5.s, z5.s
    fmul    z2.s, z6.s, z6.s
    fmul    z3.s, z7.s, z7.s
    // Accumulate c2*r² (c2=0.5)
    fmla    za.s[w11, 0, vgx4], {z0.s-z3.s}, z10.s

    // r³ = r² * r
    fmul    z18.s, z0.s, z4.s
    fmul    z19.s, z1.s, z5.s
    fmul    z20.s, z2.s, z6.s
    fmul    z21.s, z3.s, z7.s
    // Accumulate c3*r³ (c3=1/6)
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z11.s

    // r⁴ = r² * r²
    fmul    z18.s, z0.s, z0.s
    fmul    z19.s, z1.s, z1.s
    fmul    z20.s, z2.s, z2.s
    fmul    z21.s, z3.s, z3.s
    // Accumulate c4*r⁴ (c4=1/24)
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z12.s

    // r⁵ = r⁴ * r (use z20-z23 to avoid clobbering z9-z11 constants)
    // Must compute [2] and [3] FIRST before z20/z21 are overwritten
    fmul    z22.s, z20.s, z6.s       // r⁵[2] = r⁴[2] * r[2]
    fmul    z23.s, z21.s, z7.s       // r⁵[3] = r⁴[3] * r[3]
    fmul    z20.s, z18.s, z4.s       // r⁵[0] = r⁴[0] * r[0] (now safe to overwrite z20)
    fmul    z21.s, z19.s, z5.s       // r⁵[1] = r⁴[1] * r[1] (now safe to overwrite z21)
    // Accumulate c5*r⁵ (c5=1/120)
    fmla    za.s[w11, 0, vgx4], {z20.s-z23.s}, z13.s

    // r⁶ = r⁴ * r² (use z20-z23 — safe: r⁵ results already accumulated into ZA)
    // z18-z19 still have r⁴[0-1]; recompute r⁴[2-3] from r²[2-3]
    // z25/z26 are NOT touched here — they hold the 88.0/-88.0 clamp constants
    fmul    z20.s, z18.s, z0.s       // r⁶[0] = r⁴[0] * r²[0]
    fmul    z21.s, z19.s, z1.s       // r⁶[1] = r⁴[1] * r²[1]
    fmul    z22.s, z2.s, z2.s        // recompute r⁴[2] = r²[2]²
    fmul    z22.s, z22.s, z2.s       // r⁶[2] = r⁴[2] * r²[2]
    fmul    z23.s, z3.s, z3.s        // recompute r⁴[3] = r²[3]²
    fmul    z23.s, z23.s, z3.s       // r⁶[3] = r⁴[3] * r²[3]
    // Accumulate c6*r⁶ (c6=1/720)
    fmla    za.s[w11, 0, vgx4], {z20.s-z23.s}, z14.s

    // r⁷ = r⁶ * r
    // r⁶ is in z20-z23; r is in z4-z7.  Write r⁷ into z18-z21:
    //   z18/z19 held r⁴[0-1] — both consumed above, now free.
    //   z20/z21 held r⁶[0-1] — fmla already consumed them, now free.
    fmul    z18.s, z20.s, z4.s       // r⁷[0] = r⁶[0] * r[0]
    fmul    z19.s, z21.s, z5.s       // r⁷[1] = r⁶[1] * r[1]
    fmul    z20.s, z22.s, z6.s       // r⁷[2] = r⁶[2] * r[2]
    fmul    z21.s, z23.s, z7.s       // r⁷[3] = r⁶[3] * r[3]
    // Accumulate c7*r⁷ (c7=1/5040)
    fmla    za.s[w11, 0, vgx4], {z18.s-z21.s}, z15.s

    // Extract accumulated polynomial result and add c0=1.0
    mova    {z0.s-z3.s}, za.s[w11, 0, vgx4]
    fadd    z0.s, z0.s, z9.s
    fadd    z1.s, z1.s, z9.s
    fadd    z2.s, z2.s, z9.s
    fadd    z3.s, z3.s, z9.s

    // Scale by 2^n_val to get exp(-gate)
    fcvtzs  z28.s, p0/m, z28.s
    fcvtzs  z29.s, p0/m, z29.s
    fcvtzs  z30.s, p0/m, z30.s
    fcvtzs  z31.s, p0/m, z31.s
    fscale  z0.s, p0/m, z0.s, z28.s
    fscale  z1.s, p0/m, z1.s, z29.s
    fscale  z2.s, p0/m, z2.s, z30.s
    fscale  z3.s, p0/m, z3.s, z31.s

    // Compute sigmoid = 1 / (1 + exp(-gate))
    fadd    z0.s, z0.s, z9.s            // 1 + exp(-gate)
    fadd    z1.s, z1.s, z9.s
    fadd    z2.s, z2.s, z9.s
    fadd    z3.s, z3.s, z9.s
    movprfx z4, z9
    fdiv    z4.s, p0/m, z4.s, z0.s      // sigmoid in z4-z7
    movprfx z5, z9
    fdiv    z5.s, p0/m, z5.s, z1.s
    movprfx z6, z9
    fdiv    z6.s, p0/m, z6.s, z2.s
    movprfx z7, z9
    fdiv    z7.s, p0/m, z7.s, z3.s

    // out = sigmoid(gate) * y
    ld1w    {z0.s-z3.s}, pn9/z, [x20, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z4.s
    fmul    z1.s, p0/m, z1.s, z5.s
    fmul    z2.s, p0/m, z2.s, z6.s
    fmul    z3.s, p0/m, z3.s, z7.s

    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lsg_loop

    smstop

.Lsg_done:
    ldp     x19, x20, [sp, #16]
    ldp     d8,  d9,  [sp, #32]
    ldp     d10, d11, [sp, #48]
    ldp     d12, d13, [sp, #64]
    ldp     d14,  d15,  [sp, #80]
    ldp     x29, x30, [sp], #96
    ret

.p2align 2
.Lsg_const:
    .long   0x3FB8AA3B      // log2(e) = 1.4426950408889634
    .long   0x3F317218      // ln(2) = 0.6931471805599453
    .long   0x3F800000      // 1.0
    .long   0x3F000000      // 0.5 = 1/2!
    .long   0x3E2AAAAB      // 1/6 = 1/3!
    .long   0x3D2AAAAB      // 1/24 = 1/4!
    .long   0x3C088889      // 1/120 = 1/5!
    .long   0x3AB60B61      // 1/720 = 1/6!
    .long   0x39500D01      // 1/5040 = 1/7!
    .long   0x42B00000      // 88.0
    .long   0xC2B00000      // -88.0
