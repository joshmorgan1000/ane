// dct4_fp32.s — 4-point DCT-II transform via SME2 streaming SVE
//
// void dct4_fp32(const float *input, float *output, long n)
//
// Applies a 4-point Type-II Discrete Cosine Transform to each group of 4
// consecutive floats in the input array. n must be a multiple of 4.
// Supports in-place operation (output == input).
//
// AAPCS: x0=input, x1=output, x2=n
//
// DCT-II formula for each group [a, b, c, d]:
//   X[0] = a + b + c + d
//   X[1] = cos(π/8)·(a-d) + cos(3π/8)·(b-c)
//   X[2] = cos(π/4)·(a - b - c + d)
//   X[3] = cos(3π/8)·(a-d) - cos(π/8)·(b-c)
//
// Fast butterfly decomposition:
//   Stage 1: s0=a+d, s1=b+c, s2=b-c, s3=a-d
//   Stage 2: X0=s0+s1, X1=c0·s3+c1·s2, X2=c2·(s0-s1), X3=c1·s3-c0·s2
//
// Uses SVE ld4w/st4w for hardware deinterleaving of 4-element groups.
// On M4 (SVLb=64): 16 groups = 64 floats = 256 bytes per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _dct4_fp32
.p2align 4

_dct4_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x2, .Ldone

    smstart sm

    // total_groups = n / 4
    lsr     x4, x2, #2

    // All-true predicate for fmla/fmls (predicated operations)
    ptrue   p0.s

    // Load DCT-II cosine coefficients via constant pool
    adr     x9, .Lconst
    ld1rw   {z28.s}, p0/z, [x9]         // z28 = cos(π/8)
    ld1rw   {z29.s}, p0/z, [x9, #4]     // z29 = cos(3π/8)
    ld1rw   {z30.s}, p0/z, [x9, #8]     // z30 = cos(π/4)

    // Group index
    mov     x8, #0

    // Initial predicate
    whilelt p1.s, x8, x4

.Lloop:
    b.none  .Lexit

    // Element offset = group_index × 4
    lsl     x9, x8, #2

    // ---- Deinterleave load: [a0,b0,c0,d0, ...] → z0=a, z1=b, z2=c, z3=d
    ld4w    {z0.s, z1.s, z2.s, z3.s}, p1/z, [x0, x9, lsl #2]

    // ---- Stage 1: Butterfly (all unpredicated, constructive)
    fadd    z4.s, z0.s, z3.s            // s0 = a + d
    fadd    z5.s, z1.s, z2.s            // s1 = b + c
    fsub    z6.s, z1.s, z2.s            // s2 = b - c
    fsub    z7.s, z0.s, z3.s            // s3 = a - d

    // ---- Stage 2: DCT-II twiddle factors
    //
    // X[0] = s0 + s1
    fadd    z0.s, z4.s, z5.s

    // X[1] = c0·s3 + c1·s2  (fmul + fused multiply-accumulate)
    fmul    z1.s, z7.s, z28.s           // z1 = s3 × cos(π/8)
    fmla    z1.s, p0/m, z6.s, z29.s     // z1 += s2 × cos(3π/8)

    // X[2] = c2·(s0 - s1)
    fsub    z2.s, z4.s, z5.s            // z2 = s0 - s1
    fmul    z2.s, z2.s, z30.s           // z2 *= cos(π/4)

    // X[3] = c1·s3 - c0·s2  (fmul + fused multiply-subtract)
    fmul    z3.s, z7.s, z29.s           // z3 = s3 × cos(3π/8)
    fmls    z3.s, p0/m, z6.s, z28.s     // z3 -= s2 × cos(π/8)

    // ---- Interleave store: z0=X0, z1=X1, z2=X2, z3=X3
    st4w    {z0.s, z1.s, z2.s, z3.s}, p1, [x1, x9, lsl #2]

    // Advance group index by VL words (16 on M4)
    incw    x8
    whilelt p1.s, x8, x4
    b       .Lloop

.Lexit:
    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret

// Constant pool
.p2align 2
.Lconst:
    .long   0x3F6C835E  // cos(π/8) = 0.923879533
    .long   0x3EC3EF15  // cos(3π/8) = 0.382683432
    .long   0x3F3504F3  // cos(π/4) = 0.707106781
