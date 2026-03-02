// hard_sigmoid_fp32.s — HardSigmoid activation via SME2 streaming SVE
//
// void hard_sigmoid_fp32(const float *input, float *output, long n)
// AAPCS: x0=input, x1=output, x2=n
//
// Computes: output[i] = clamp(input[i] / 6.0 + 0.5, 0.0, 1.0)
//
// Steps:
//   1. fmul by z16 (1/6) — scale
//   2. fadd by z17 (0.5) — shift
//   3. fclamp {z0.s-z3.s}, z28.s, z29.s — clamp [0, 1]  (1 instruction)
//
// Constants: z16=1/6, z17=0.5, z28=0.0 (lower bound), z29=1.0 (upper bound).
//
// Processes 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _hard_sigmoid_fp32
.p2align 4

_hard_sigmoid_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x2, .Ldone

    smstart sm

    ptrue   p0.s

    adr     x9, .Lconst
    ld1rw   {z16.s}, p0/z, [x9]        // 1/6 = 0x3E2AAAAB
    ld1rw   {z17.s}, p0/z, [x9, #4]    // 0.5 = 0x3F000000
    ld1rw   {z28.s}, p0/z, [x9, #8]    // 0.0 (lower clamp)
    ld1rw   {z29.s}, p0/z, [x9, #12]   // 1.0 (upper clamp)

    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]

    // Scale: output = input * (1/6)
    fmul    z0.s, p0/m, z0.s, z16.s
    fmul    z1.s, p0/m, z1.s, z16.s
    fmul    z2.s, p0/m, z2.s, z16.s
    fmul    z3.s, p0/m, z3.s, z16.s

    // Shift: output = output + 0.5
    fadd    z0.s, p0/m, z0.s, z17.s
    fadd    z1.s, p0/m, z1.s, z17.s
    fadd    z2.s, p0/m, z2.s, z17.s
    fadd    z3.s, p0/m, z3.s, z17.s

    // Clamp [0.0, 1.0] — multi-vector form (single instruction)
    fclamp  {z0.s-z3.s}, z28.s, z29.s

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

.p2align 2
.Lconst:
    .long   0x3E2AAAAB  // 1/6 = 0.16666667f
    .long   0x3F000000  // 0.5f
    .long   0x00000000  // 0.0f
    .long   0x3F800000  // 1.0f
