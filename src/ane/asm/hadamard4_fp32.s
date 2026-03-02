// hadamard4_fp32.s — Fused 4-point Hadamard butterfly transform via SME2 streaming SVE
//
// void hadamard4_fp32(const float *input, float *output, long n)
//
// For each group of 4 interleaved floats [a, b, c, d]:
//   1. Convert to fixed-point: multiply by 16384.0, truncate to int32
//   2. Butterfly stage 1: e=a+b, f=a-b, g=c+d, h=c-d
//   3. Butterfly stage 2 with ASR: w=(e+g)>>1, x=(f+h)>>1, y=(e-g)>>1, z=(f-h)>>1
//   4. Convert back to float: int32→float, multiply by 1/16384.0
//
// The integer >>1 (arithmetic shift right) preserves truncation-toward-negative-infinity
// rounding, which is NOT equivalent to floating-point division by 2.
//
// Uses SVE ld4w/st4w for hardware deinterleaving of the 4-way structure.
// On M4 with SVLb=64, each iteration processes 16 groups = 64 floats = 256 bytes.
//
// n must be a multiple of 4. Supports in-place (output == input).

.section __TEXT,__text,regular,pure_instructions
.global _hadamard4_fp32
.p2align 4

_hadamard4_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]

    cbz     x2, .Ldone

    smstart sm

    // total_groups = n / 4
    lsr     x4, x2, #2

    // VL in words (16 on M4)
    cntw    x5

    // Load FIXED_SCALE = 16384.0f = 0x46800000 into z24
    movz    w9, #0x4680, lsl #16
    fmov    s0, w9
    mov     z24.s, s0

    // Load INV_SCALE = 1/16384.0f = 0x38800000 into z25
    movz    w9, #0x3880, lsl #16
    fmov    s0, w9
    mov     z25.s, s0

    // All-true predicate for fcvtzs/scvtf (predicated operations)
    ptrue   p0.s

    // Group index
    mov     x8, #0

    // Initial whilelt for loop entry
    whilelt p1.s, x8, x4

.Lloop:
    b.none  .Lexit

    // Element offset = group_index * 4 (for addressing into float array)
    lsl     x9, x8, #2

    // ---- Deinterleave load: [a0,b0,c0,d0, a1,b1,c1,d1, ...] → z0=a, z1=b, z2=c, z3=d
    ld4w    {z0.s, z1.s, z2.s, z3.s}, p1/z, [x0, x9, lsl #2]

    // ---- Float → Fixed-point: multiply by FIXED_SCALE
    fmul    z0.s, z0.s, z24.s
    fmul    z1.s, z1.s, z24.s
    fmul    z2.s, z2.s, z24.s
    fmul    z3.s, z3.s, z24.s

    // ---- Float → Int32 (truncation toward zero, matching C cast)
    fcvtzs  z0.s, p0/m, z0.s
    fcvtzs  z1.s, p0/m, z1.s
    fcvtzs  z2.s, p0/m, z2.s
    fcvtzs  z3.s, p0/m, z3.s

    // ---- Butterfly stage 1
    add     z4.s, z0.s, z1.s        // e = a + b
    sub     z5.s, z0.s, z1.s        // f = a - b
    add     z6.s, z2.s, z3.s        // g = c + d
    sub     z7.s, z2.s, z3.s        // h = c - d

    // ---- Butterfly stage 2 (pre-shift sums)
    add     z0.s, z4.s, z6.s        // e + g
    add     z1.s, z5.s, z7.s        // f + h
    sub     z2.s, z4.s, z6.s        // e - g
    sub     z3.s, z5.s, z7.s        // f - h

    // ---- Arithmetic shift right by 1 (truncation toward -inf)
    asr     z0.s, z0.s, #1
    asr     z1.s, z1.s, #1
    asr     z2.s, z2.s, #1
    asr     z3.s, z3.s, #1

    // ---- Int32 → Float
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s

    // ---- Multiply by INV_SCALE to convert back
    fmul    z0.s, z0.s, z25.s
    fmul    z1.s, z1.s, z25.s
    fmul    z2.s, z2.s, z25.s
    fmul    z3.s, z3.s, z25.s

    // ---- Interleave store: z0=w, z1=x, z2=y, z3=z → [w0,x0,y0,z0, w1,x1,y1,z1, ...]
    st4w    {z0.s, z1.s, z2.s, z3.s}, p1, [x1, x9, lsl #2]

    // Advance group index by VL words
    incw    x8
    whilelt p1.s, x8, x4
    b       .Lloop

.Lexit:
    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10,  d11,  [sp, #32]
    ldp     d12,  d13,  [sp, #48]
    ldp     d14,  d15,  [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
