// gaussian_noise_fp32.s — Add counter-based pseudo-random noise via SME2 streaming SVE
//
// void gaussian_noise_fp32(const float* input, float* output, float stddev,
//                          long n, uint64_t seed)
//
// AAPCS: x0=input, x1=output, s0=stddev, x2=n, x3=seed
//
// Algorithm:
//   seed_xor = (uint32_t)(seed ^ (seed >> 32))
//   For each element i:
//     k = seed_xor ^ (uint32_t)i
//     k ^= k >> 16; k *= 0x045D9F3B
//     k ^= k >> 15; k *= 0xD168AAD5
//     k ^= k >> 16
//     noise = (float)(k >> 8) / 8388608.0 - 1.0   // -> [-1.0, ~1.0)
//     output[i] = input[i] + noise * stddev
//
// stddev is saved before smstart, then broadcast into z28.
// Hash constants: z16=0x045D9F3B, z17=0xD168AAD5.
// z18=1/8388608.0 (2^-23), z19=1.0, z20=seed_xor broadcast.
//
// Processes 64 floats (256 bytes) per iteration on M4 via vlx4.

.section __TEXT,__text,regular,pure_instructions
.global _gaussian_noise_fp32
.p2align 4

_gaussian_noise_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]
    stp     x19, x20, [sp, #80]
    // Save stddev (s0) before smstart zeroes float registers
    str     s0, [sp, #96]

    cbz     x2, .Ldone

    // seed_xor = (uint32_t)(seed ^ (seed >> 32))
    lsr     x9, x3, #32
    eor     x9, x3, x9
    mov     w19, w9                 // w19 = seed_xor (callee-saved 32-bit)

    smstart sm

    ptrue   p0.s

    ld1rw   {z28.s}, p0/z, [sp, #96]   // broadcast stddev

    // Hash constant 0x045D9F3B
    mov     w9, #0x9F3B
    movk    w9, #0x045D, lsl #16
    dup     z16.s, w9

    // Hash constant 0xD168AAD5
    mov     w9, #0xAAD5
    movk    w9, #0xD168, lsl #16
    dup     z17.s, w9

    // 1.0 / 8388608.0 = 2^-23 = 0x37000000
    mov     w9, #0
    movk    w9, #0x3700, lsl #16
    dup     z18.s, w9

    // 1.0f = 0x3F800000
    mov     w9, #0
    movk    w9, #0x3F80, lsl #16
    dup     z19.s, w9

    // seed_xor broadcast
    dup     z20.s, w19

    // vector length in 32-bit elements (16 on M4)
    cntw    x20

    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    // Build index vectors: z0=[x8..x8+VL-1], z1=[x8+VL..], z2, z3
    mov     w9, w8
    index   z0.s, w9, #1
    add     w9, w9, w20
    index   z1.s, w9, #1
    add     w9, w9, w20
    index   z2.s, w9, #1
    add     w9, w9, w20
    index   z3.s, w9, #1

    // XOR each index with seed_xor
    eor     z0.d, z0.d, z20.d
    eor     z1.d, z1.d, z20.d
    eor     z2.d, z2.d, z20.d
    eor     z3.d, z3.d, z20.d

    // Hash round 1: k ^= k >> 16
    lsr     z4.s, z0.s, #16
    lsr     z5.s, z1.s, #16
    lsr     z6.s, z2.s, #16
    lsr     z7.s, z3.s, #16
    eor     z0.d, z0.d, z4.d
    eor     z1.d, z1.d, z5.d
    eor     z2.d, z2.d, z6.d
    eor     z3.d, z3.d, z7.d

    // k *= 0x045D9F3B
    mul     z0.s, p0/m, z0.s, z16.s
    mul     z1.s, p0/m, z1.s, z16.s
    mul     z2.s, p0/m, z2.s, z16.s
    mul     z3.s, p0/m, z3.s, z16.s

    // Hash round 2: k ^= k >> 15
    lsr     z4.s, z0.s, #15
    lsr     z5.s, z1.s, #15
    lsr     z6.s, z2.s, #15
    lsr     z7.s, z3.s, #15
    eor     z0.d, z0.d, z4.d
    eor     z1.d, z1.d, z5.d
    eor     z2.d, z2.d, z6.d
    eor     z3.d, z3.d, z7.d

    // k *= 0xD168AAD5
    mul     z0.s, p0/m, z0.s, z17.s
    mul     z1.s, p0/m, z1.s, z17.s
    mul     z2.s, p0/m, z2.s, z17.s
    mul     z3.s, p0/m, z3.s, z17.s

    // Hash round 3: k ^= k >> 16
    lsr     z4.s, z0.s, #16
    lsr     z5.s, z1.s, #16
    lsr     z6.s, z2.s, #16
    lsr     z7.s, z3.s, #16
    eor     z0.d, z0.d, z4.d
    eor     z1.d, z1.d, z5.d
    eor     z2.d, z2.d, z6.d
    eor     z3.d, z3.d, z7.d

    // Convert to float in [-1.0, ~1.0)
    lsr     z0.s, z0.s, #8
    lsr     z1.s, z1.s, #8
    lsr     z2.s, z2.s, #8
    lsr     z3.s, z3.s, #8

    ucvtf   z0.s, p0/m, z0.s
    ucvtf   z1.s, p0/m, z1.s
    ucvtf   z2.s, p0/m, z2.s
    ucvtf   z3.s, p0/m, z3.s

    fmul    z0.s, p0/m, z0.s, z18.s
    fmul    z1.s, p0/m, z1.s, z18.s
    fmul    z2.s, p0/m, z2.s, z18.s
    fmul    z3.s, p0/m, z3.s, z18.s

    fsub    z0.s, p0/m, z0.s, z19.s
    fsub    z1.s, p0/m, z1.s, z19.s
    fsub    z2.s, p0/m, z2.s, z19.s
    fsub    z3.s, p0/m, z3.s, z19.s

    // noise * stddev
    fmul    z0.s, p0/m, z0.s, z28.s
    fmul    z1.s, p0/m, z1.s, z28.s
    fmul    z2.s, p0/m, z2.s, z28.s
    fmul    z3.s, p0/m, z3.s, z28.s

    // output = input + noise*stddev
    ld1w    {z4.s-z7.s}, pn9/z, [x0, x8, lsl #2]
    fadd    z0.s, p0/m, z0.s, z4.s
    fadd    z1.s, p0/m, z1.s, z5.s
    fadd    z2.s, p0/m, z2.s, z6.s
    fadd    z3.s, p0/m, z3.s, z7.s

    st1w    {z0.s-z3.s}, pn9, [x1, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x2, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     x19, x20, [sp, #80]
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #112
    ret
