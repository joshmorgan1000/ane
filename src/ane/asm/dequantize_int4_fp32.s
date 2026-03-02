// dequantize_int4_fp32.s — INT4 (packed) to FP32 dequantization via SME2 streaming SVE
//
// void dequantize_int4_fp32(const uint8_t *input, float *output,
//                            float scale, float zero_point, long n)
//
// Computes output[i] = ((float)nibble[i] - zero_point) / scale
// where each input byte contains two 4-bit unsigned values:
//   low nibble (bits [3:0])  → output[2*byte_idx]
//   high nibble (bits [7:4]) → output[2*byte_idx + 1]
//
// AAPCS: x0=input (packed bytes, n/2 elements), x1=output (n floats),
//        s0=scale, s1=zero_point, x2=n (output element count, must be even)
//
// Strategy:
//   - ld1b {z.h} loads 32 packed bytes, zero-extending each to 16-bit
//   - AND/LSR extract low/high nibbles as halfwords
//   - uunpklo/uunpkhi widen halfwords to 32-bit words
//   - ucvtf converts unsigned int32 to float
//   - fsub zero_point, fmul by exact 1/scale (precomputed via scalar fdiv)
//   - zip1/zip2 interleave lo/hi nibble results for correct output order
//   - 4 predicated st1w stores (64 output floats per iteration on M4)
//
// Per iteration: 32 input bytes → 64 output floats (4 full z-vectors on M4)

.section __TEXT,__text,regular,pure_instructions
.global _dequantize_int4_fp32
.p2align 4

_dequantize_int4_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]

    // Save float args before smstart (smstart zeroes all z-regs and predicates)
    fmov    w3, s0                  // w3 = scale bits
    fmov    w4, s1                  // w4 = zero_point bits

    cbz     x2, .Ldone

    // Compute exact 1/scale BEFORE entering streaming mode
    // (scalar FP operations are safe outside streaming mode)
    fmov    s2, #1.0
    fdiv    s3, s2, s0              // s3 = 1.0 / scale (exactly rounded)
    fmov    w5, s3                  // w5 = 1/scale bits

    smstart sm

    ptrue   p0.s                    // all-true predicate for .s operations

    // Broadcast constants from saved GPRs
    fmov    s0, w5                  // restore 1/scale
    mov     z24.s, s0               // z24 = 1/scale broadcast
    fmov    s1, w4                  // restore zero_point
    mov     z25.s, s1               // z25 = zero_point broadcast

    // Counters
    lsr     x3, x2, #1             // x3 = n/2 = total input bytes
    mov     x8, #0                  // byte input index
    mov     x9, #0                  // float output index

.Lloop:
    // Exit when all output elements have been written
    cmp     x9, x2
    b.ge    .Lexit

    // Predicate for byte load: each active halfword lane loads one byte
    // whilelt p1.h sets lane i active if (x8 + i < x3), up to 32 lanes on M4
    whilelt p1.h, x8, x3

    // Load up to 32 packed bytes, zero-extending each to 16-bit halfword
    // On M4: 32 halfword lanes × 1 byte each = 32 bytes from input
    ld1b    {z0.h}, p1/z, [x0, x8]

    // Extract low nibbles: bits [3:0] of each packed byte
    mov     z1.d, z0.d              // copy (AND is destructive in SVE)
    and     z1.h, z1.h, #0x000f    // z1.h = low nibbles (0-15)

    // Extract high nibbles: bits [7:4] of each packed byte
    lsr     z2.h, z0.h, #4         // z2.h = high nibbles (0-15)
    // No mask needed: ld1b zero-extended, so bits [15:8] were already 0

    // Widen low nibbles: halfword → word (32 halfwords → 2×16 words)
    uunpklo z3.s, z1.h             // z3 = [lo0, lo1, ..., lo15]
    uunpkhi z4.s, z1.h             // z4 = [lo16, lo17, ..., lo31]

    // Widen high nibbles: halfword → word
    uunpklo z5.s, z2.h             // z5 = [hi0, hi1, ..., hi15]
    uunpkhi z6.s, z2.h             // z6 = [hi16, hi17, ..., hi31]

    // Convert unsigned int32 to float
    ucvtf   z3.s, p0/m, z3.s
    ucvtf   z4.s, p0/m, z4.s
    ucvtf   z5.s, p0/m, z5.s
    ucvtf   z6.s, p0/m, z6.s

    // Subtract zero_point: (float_nibble - zero_point)
    fsub    z3.s, p0/m, z3.s, z25.s
    fsub    z4.s, p0/m, z4.s, z25.s
    fsub    z5.s, p0/m, z5.s, z25.s
    fsub    z6.s, p0/m, z6.s, z25.s

    // Multiply by 1/scale: exact reciprocal precomputed via fdiv
    fmul    z3.s, p0/m, z3.s, z24.s
    fmul    z4.s, p0/m, z4.s, z24.s
    fmul    z5.s, p0/m, z5.s, z24.s
    fmul    z6.s, p0/m, z6.s, z24.s

    // Interleave [lo, hi] pairs for correct output order:
    //   byte[k] → output[2k] = dequant(lo_nibble), output[2k+1] = dequant(hi_nibble)
    //
    // z3 = [lo0..lo15], z5 = [hi0..hi15]
    //   zip1 → [lo0, hi0, lo1, hi1, ..., lo7, hi7]   (16 floats)
    //   zip2 → [lo8, hi8, lo9, hi9, ..., lo15, hi15]  (16 floats)
    // z4 = [lo16..lo31], z6 = [hi16..hi31]
    //   zip1 → [lo16, hi16, ..., lo23, hi23]           (16 floats)
    //   zip2 → [lo24, hi24, ..., lo31, hi31]           (16 floats)
    zip1    z0.s, z3.s, z5.s
    zip2    z1.s, z3.s, z5.s
    zip1    z2.s, z4.s, z6.s
    zip2    z3.s, z4.s, z6.s

    // Store 4 vectors (up to 64 output floats), each predicated against n
    whilelt p2.s, x9, x2
    st1w    {z0.s}, p2, [x1, x9, lsl #2]
    incw    x9

    whilelt p2.s, x9, x2
    st1w    {z1.s}, p2, [x1, x9, lsl #2]
    incw    x9

    whilelt p2.s, x9, x2
    st1w    {z2.s}, p2, [x1, x9, lsl #2]
    incw    x9

    whilelt p2.s, x9, x2
    st1w    {z3.s}, p2, [x1, x9, lsl #2]
    incw    x9

    // Advance byte input index by 32 (= cnth = halfwords per vector on M4)
    cnth    x6
    add     x8, x8, x6
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
