// quantize_fp32_int4.s — FP32 to packed INT4 quantization via SME2 streaming SVE
//
// void quantize_fp32_int4(const float *input, uint8_t *output, float scale, float zero_point, long n)
//
// Computes output[i/2] = (clamp(round(input[i] * scale + zero_point), 0, 15) & 0x0F) | (clamp(round(input[i+1] * scale + zero_point), 0, 15) << 4)
// Each byte stores two 4-bit unsigned values: low nibble = even index, high nibble = odd index
// AAPCS: x0=input, x1=output, s0=scale, s1=zero_point, x2=n
//
// Optimized with:
// - fmad for fused multiply-add (input * scale + zero_point in one instruction)
// - sclamp for multi-vector clamping to [0, 15]
// - Streamlined narrowing pipeline
//
// Processes 64 floats (256 bytes) per iteration → 32 packed output bytes.

.section __TEXT,__text,regular,pure_instructions
.global _quantize_fp32_int4
.p2align 4

_quantize_fp32_int4:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    // Save float args before smstart (smstart zeroes all z-regs/predicates)
    fmov    w9, s0                  // scale
    fmov    w10, s1                 // zero_point

    cbz     x2, .Ldone

    smstart sm

    ptrue   p0.s
    ptrue   p1.b
    ptrue   pn8.s                   // predicate-as-counter for 4-vector ops

    // Restore float args and broadcast
    fmov    s0, w9
    fmov    s1, w10
    mov     z16.s, s0               // z16 = scale (broadcast to all lanes)
    mov     z17.s, s1               // z17 = zero_point (broadcast)

    // Saturation bounds for unsigned 4-bit: [0, 15]
    mov     w11, #0
    dup     z18.s, w11              // min = 0
    mov     w11, #15
    dup     z19.s, w11              // max = 15

    // Nibble mask: 0x0F
    mov     w12, #0x0F
    dup     z20.b, w12

    // Calculate number of full 4-vector iterations (64 floats each)
    cntw    x10                     // SVL in words (16 on M4)
    lsl     x11, x10, #2            // 4 * SVL = 64 floats per iteration
    mov     x8, #0                  // input index
    mov     x13, #0                 // output index

    // Main loop: process 64 floats (4 vectors) per iteration → 32 output bytes
.Lloop_main:
    sub     x12, x2, x8             // remaining elements
    cmp     x12, x11
    b.lt    .Lloop_tail             // < 64 remaining, go to tail

    // Load 4 vectors (64 floats = 256 bytes)
    ld1w    {z0.s - z3.s}, pn8/z, [x0, x8, lsl #2]

    // Fused multiply-add: result = input * scale + zero_point
    // fmad z.s, p/m, z_mul.s, z_add.s computes: z = z * z_mul + z_add
    fmad    z0.s, p0/m, z16.s, z17.s
    fmad    z1.s, p0/m, z16.s, z17.s
    fmad    z2.s, p0/m, z16.s, z17.s
    fmad    z3.s, p0/m, z16.s, z17.s

    // Round to nearest
    frintn  z0.s, p0/m, z0.s
    frintn  z1.s, p0/m, z1.s
    frintn  z2.s, p0/m, z2.s
    frintn  z3.s, p0/m, z3.s

    // Convert to int32
    fcvtzs  z0.s, p0/m, z0.s
    fcvtzs  z1.s, p0/m, z1.s
    fcvtzs  z2.s, p0/m, z2.s
    fcvtzs  z3.s, p0/m, z3.s

    // Clamp to [0, 15] using multi-vector sclamp
    sclamp  {z0.s - z3.s}, z18.s, z19.s

    // Narrow int32 → int16: pairs of vectors
    uzp1    z4.h, z0.h, z1.h        // z0,z1 → z4 (32 int16s)
    uzp1    z5.h, z2.h, z3.h        // z2,z3 → z5 (32 int16s)

    // Narrow int16 → int8: combine into single vector
    uzp1    z6.b, z4.b, z5.b        // z4,z5 → z6 (64 int8s)

    // Pack pairs of int8 values into nibbles
    // z6 contains: [v0, v1, v2, v3, v4, v5, ...] (64 bytes)
    // We need: [(v0 & 0x0F) | (v1 << 4), (v2 & 0x0F) | (v3 << 4), ...]

    // Extract even indices (low nibbles): z7 = [v0, v2, v4, ...]
    uzp1    z7.b, z6.b, z6.b        // take every other byte starting at 0
    // Extract odd indices (high nibbles): z8 = [v1, v3, v5, ...]
    uzp2    z8.b, z6.b, z6.b        // take every other byte starting at 1

    // Mask low nibbles (values are [0,15] so mask is technically redundant, but safe)
    and     z7.b, z7.b, z20.b       // z7 = low & 0x0F

    // Shift high nibbles left by 4
    lsl     z8.b, z8.b, #4          // z8 = high << 4

    // Combine: output = low | high
    orr     z9.b, z7.b, z8.b        // z9 = (low & 0x0F) | (high << 4)

    // Store 32 bytes (first half of z9)
    st1b    {z9.b}, p1, [x1, x13]

    add     x8, x8, x11             // advance input by 64 floats
    lsr     x14, x11, #1            // 64 / 2 = 32 output bytes
    add     x13, x13, x14           // advance output by 32 bytes
    b       .Lloop_main

    // Tail loop: process remaining elements with single-vector predicated ops
.Lloop_tail:
    whilelt p2.s, x8, x2
    b.none  .Lexit

    ld1w    {z0.s}, p2/z, [x0, x8, lsl #2]

    // Fused multiply-add
    fmad    z0.s, p0/m, z16.s, z17.s
    frintn  z0.s, p0/m, z0.s
    fcvtzs  z0.s, p0/m, z0.s

    // Clamp (single vector)
    smax    z0.s, p0/m, z0.s, z18.s
    smin    z0.s, p0/m, z0.s, z19.s

    // Narrow to int8
    uzp1    z1.h, z0.h, z0.h
    uzp1    z2.b, z1.b, z1.b        // z2 contains 16 int8 values

    // Pack pairs: extract even/odd, mask, shift, combine
    uzp1    z3.b, z2.b, z2.b        // even indices
    uzp2    z4.b, z2.b, z2.b        // odd indices
    and     z3.b, z3.b, z20.b       // mask low nibble
    lsl     z4.b, z4.b, #4          // shift high nibble
    orr     z5.b, z3.b, z4.b        // combine

    // Store 8 bytes (16 nibbles = 8 bytes for SVL=16)
    cntw    x10
    lsr     x14, x10, #1            // SVL/2 = 8 output bytes per vector
    lsr     x15, x2, #1             // n/2 = output size in bytes
    whilelt p3.b, x13, x15          // predicate for output bytes
    st1b    {z5.b}, p3, [x1, x13]

    incw    x8                      // advance by 16 floats
    add     x13, x13, x14           // advance by 8 bytes
    b       .Lloop_tail

.Lexit:
    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #96
    ret
