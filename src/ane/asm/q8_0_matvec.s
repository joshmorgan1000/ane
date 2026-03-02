// q8_0_matvec.s — W8A8 block-quantized matrix-vector multiply using SVE2 SDOT
//
// void q8_0_matvec(
//     const int8_t* A_quants,      // x0: [K] int8 activation quants
//     const float*  A_scales,      // x1: [K/32] fp32 per-block activation scales
//     const int8_t* B_quants_bm,   // x2: [n_blocks][8][N_pad16][4] SDOT-layout quants
//     const float*  B_scales_bm,   // x3: [n_blocks][N] block-major fp32 scales
//     float*        C,             // x4: [N] fp32 output
//     long          N,             // x5: output elements
//     long          K              // x6: input elements (multiple of 32)
// )
//
// B_quants SDOT layout: for block b, sub-group s (0..7), row j:
//   offset = b * sub_stride + s * tile_stride + j * 4
//   where sub_stride = 8 * N_pad16 * 4
//         tile_stride = N_pad16 * 4
//         N_pad16 = ((N + 15) & ~15)
//
// For 16 consecutive rows at n_tile: 64 contiguous bytes = 1 z-register.
//
// SDOT z_acc.s, z_b.b, z_a.b:
//   For each of 16 .s lanes: acc[i] += dot4(b[i*4..i*4+3], a[i*4..i*4+3])
//   z_a = broadcast of 4 A_quant bytes to all lanes (DUP)
//   z_b = 4 bytes from each of 16 B rows (contiguous load)

.section __TEXT,__text,regular,pure_instructions
.global _q8_0_matvec
.p2align 4

_q8_0_matvec:
    // Prologue
    stp     x29, x30, [sp, #-160]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8,  d9,  [sp, #96]
    stp     d10,  d11,  [sp, #112]
    stp     d12,  d13,  [sp, #128]
    stp     d14,  d15,  [sp, #144]

    // Validate
    cbz     x5, .Ldone
    cbz     x6, .Ldone

    // Save args to callee-saved
    mov     x19, x0                 // A_quants
    mov     x20, x1                 // A_scales
    mov     x21, x2                 // B_quants_bm
    mov     x22, x3                 // B_scales_bm
    mov     x23, x4                 // C
    mov     x24, x5                 // N
    mov     x25, x6                 // K

    // n_blocks = K / 32
    lsr     x26, x25, #5            // x26 = n_blocks

    // N_pad16 = (N + 15) & ~15
    add     x27, x24, #15
    and     x27, x27, #-16          // x27 = N_pad16

    // Precompute strides for B_quants layout
    // tile_stride = N_pad16 * 4 (bytes between sub-groups for one tile of 16)
    lsl     x28, x27, #2            // x28 = tile_stride = N_pad16 * 4

    // sub_stride = 8 * tile_stride = 8 * N_pad16 * 4
    // We'll compute per-block offset in the loop: block * sub_stride

    smstart sm

    ptrue   p0.s                    // all-true predicate for 16 .s lanes
    ptrue   p2.b                    // all-true at BYTE granularity for ld1b
                                    // CRITICAL: Apple M4 whilelt p.s sets only 1 predicate
                                    // bit per .s element, not 4. Using p.s with ld1b {z.b}
                                    // loads only 16/64 bytes. Must use ptrue .b instead.
                                    // B_quants padding rows are zeroed, so extra loads are safe.

    // ── Outer loop: tiles of 16 output elements ──
    mov     x8, #0                  // n_tile = 0

.Ln_tile_loop:
    cmp     x8, x24
    b.ge    .Lend_tiles

    // Create predicate for this tile (handles tail)
    whilelt p1.s, x8, x24          // p1 = mask for valid output lanes

    // Zero FP32 accumulator
    mov     z6.d, #0                // z6 = running FP32 sum

    // ── Block loop: K/32 blocks ──
    mov     x9, #0                  // block = 0

.Lblock_loop:
    cmp     x9, x26
    b.ge    .Lblock_done

    // Load A_scales[block] → broadcast to all lanes
    // smstart zeroes z-regs, so we use ldr + dup via GP
    ldr     w10, [x20, x9, lsl #2]  // w10 = A_scales[block] (as raw bits)
    // Broadcast to z8 via GP→SIMD
    dup     z8.s, w10               // z8 = A_scale broadcast

    // Load B_scales_bm[block * N + n_tile .. +15]
    mul     x13, x9, x24           // block * N
    add     x13, x13, x8           // block * N + n_tile
    add     x14, x22, x13, lsl #2  // &B_scales_bm[block*N + n_tile]
    ld1w    {z9.s}, p1/z, [x14]    // z9 = B_scales (predicated)

    // combined_scales = A_scale * B_scales
    movprfx z7, z8
    fmul    z7.s, p0/m, z7.s, z9.s // z7 = combined_scales

    // Zero INT32 SDOT accumulator
    mov     z5.d, #0

    // Compute base address for this block's B_quants:
    // base = B_quants_bm + block * 8 * N_pad16 * 4
    lsl     x13, x28, #3           // 8 * tile_stride = 8 * N_pad16 * 4 = sub_stride
    mul     x13, x9, x13           // block * sub_stride
    add     x13, x21, x13          // B_quants_bm + block * sub_stride

    // A_quants base for this block: A_quants + block * 32
    add     x14, x19, x9, lsl #5   // &A_quants[block * 32]

    // ── Sub-group loop: 8 × SDOT per block ──
    // Unrolled for performance (8 iterations)

    // sub 0
    ldr     w10, [x14]             // 4 bytes of A_quants[block*32 + 0]
    dup     z4.s, w10              // broadcast to all 16 lanes
    add     x15, x13, x8, lsl #2   // base + n_tile * 4
    ld1b    {z0.b}, p2/z, [x15]    // 64 bytes of B_quants
    sdot    z5.s, z0.b, z4.b

    // sub 1
    ldr     w10, [x14, #4]
    dup     z4.s, w10
    add     x15, x13, x28          // base + tile_stride
    add     x15, x15, x8, lsl #2
    ld1b    {z0.b}, p2/z, [x15]
    sdot    z5.s, z0.b, z4.b

    // sub 2
    ldr     w10, [x14, #8]
    dup     z4.s, w10
    add     x15, x13, x28, lsl #1  // base + 2*tile_stride
    add     x15, x15, x8, lsl #2
    ld1b    {z0.b}, p2/z, [x15]
    sdot    z5.s, z0.b, z4.b

    // sub 3
    ldr     w10, [x14, #12]
    dup     z4.s, w10
    add     x16, x28, x28, lsl #1  // 3*tile_stride
    add     x15, x13, x16
    add     x15, x15, x8, lsl #2
    ld1b    {z0.b}, p2/z, [x15]
    sdot    z5.s, z0.b, z4.b

    // sub 4
    ldr     w10, [x14, #16]
    dup     z4.s, w10
    lsl     x16, x28, #2           // 4*tile_stride
    add     x15, x13, x16
    add     x15, x15, x8, lsl #2
    ld1b    {z0.b}, p2/z, [x15]
    sdot    z5.s, z0.b, z4.b

    // sub 5
    ldr     w10, [x14, #20]
    dup     z4.s, w10
    add     x16, x28, x28, lsl #2  // 5*tile_stride
    add     x15, x13, x16
    add     x15, x15, x8, lsl #2
    ld1b    {z0.b}, p2/z, [x15]
    sdot    z5.s, z0.b, z4.b

    // sub 6
    ldr     w10, [x14, #24]
    dup     z4.s, w10
    lsl     x16, x28, #1           // 2*tile_stride
    add     x16, x16, x28, lsl #2  // + 4*tile_stride = 6*tile_stride
    add     x15, x13, x16
    add     x15, x15, x8, lsl #2
    ld1b    {z0.b}, p2/z, [x15]
    sdot    z5.s, z0.b, z4.b

    // sub 7
    ldr     w10, [x14, #28]
    dup     z4.s, w10
    lsl     x16, x28, #3           // 8*tile_stride
    sub     x16, x16, x28          // 7*tile_stride
    add     x15, x13, x16
    add     x15, x15, x8, lsl #2
    ld1b    {z0.b}, p2/z, [x15]
    sdot    z5.s, z0.b, z4.b

    // ── End sub-group: z5 has 16 INT32 partial sums ──

    // INT32 → FP32
    scvtf   z10.s, p0/m, z5.s

    // Scale by combined_scales and accumulate
    fmla    z6.s, p0/m, z10.s, z7.s // z6 += z10 * z7

    // Next block
    add     x9, x9, #1
    b       .Lblock_loop

.Lblock_done:
    // Store result for this tile
    add     x14, x23, x8, lsl #2   // &C[n_tile]
    st1w    {z6.s}, p1, [x14]      // predicated store (handles tail)

    // Next tile
    add     x8, x8, #16
    b       .Ln_tile_loop

.Lend_tiles:
    smstop

.Ldone:
    // Epilogue
    ldp     d8,  d9,  [sp, #96]
    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     d10,  d11,  [sp, #112]
    ldp     d12,  d13,  [sp, #128]
    ldp     d14,  d15,  [sp, #144]
    ldp     x29, x30, [sp], #160
    ret
