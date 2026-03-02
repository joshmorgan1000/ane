// transpose_fp32.s — FP32 matrix transpose via NEON 4×4 tiled TRN1/TRN2
//
// void transpose_fp32(const float *src, float *dst, long rows, long cols)
// AAPCS: x0=src, x1=dst, x2=rows, x3=cols
//
// src is row-major M×N.  dst is row-major N×M (the transpose).
// dst[j * rows + i] = src[i * cols + j]
//
// Strategy: Process 4×4 tiles using NEON TRN1/TRN2 (two-stage transpose).
// Edge tiles handled with scalar fallback.  No streaming mode — transpose is
// pure data rearrangement and scatter stores are unavailable in streaming mode.

.section __TEXT,__text,regular,pure_instructions
.global _transpose_fp32
.p2align 4

_transpose_fp32:
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]

    // x0=src, x1=dst, x2=rows, x3=cols
    mov     x19, x0             // src
    mov     x20, x1             // dst
    mov     x21, x2             // rows (M)
    mov     x22, x3             // cols (N)

    cbz     x21, .L_tr_done
    cbz     x22, .L_tr_done

    // Strides in bytes
    lsl     x23, x22, #2        // src_stride = cols * 4
    lsl     x24, x21, #2        // dst_stride = rows * 4

    // Tile boundaries: round down to multiple of 4
    bic     x4, x21, #3         // rows_4
    bic     x5, x22, #3         // cols_4

    // ══════════════════════════════════════════════════
    // 1. Main 4×4 tile loop: i∈[0,rows_4), j∈[0,cols_4)
    // ══════════════════════════════════════════════════
    mov     x8, #0              // i (row)
.L_tr_tile_i:
    cmp     x8, x4
    b.ge    .L_tr_right_edge

    mov     x9, #0              // j (col)
.L_tr_tile_j:
    cmp     x9, x5
    b.ge    .L_tr_tile_i_next

    // src_base = src + (i * cols + j) * 4
    madd    x10, x8, x22, x9
    lsl     x10, x10, #2
    add     x10, x19, x10

    // Load 4 rows × 4 floats
    ldr     q0, [x10]
    ldr     q1, [x10, x23]
    add     x11, x23, x23       // 2 * src_stride
    ldr     q2, [x10, x11]
    add     x11, x11, x23       // 3 * src_stride
    ldr     q3, [x10, x11]

    // 4×4 in-register transpose (two-stage TRN)
    // Stage 1: 32-bit element interleave
    trn1    v4.4s, v0.4s, v1.4s  // [a00, a10, a02, a12]
    trn2    v5.4s, v0.4s, v1.4s  // [a01, a11, a03, a13]
    trn1    v6.4s, v2.4s, v3.4s  // [a20, a30, a22, a32]
    trn2    v7.4s, v2.4s, v3.4s  // [a21, a31, a23, a33]
    // Stage 2: 64-bit element interleave
    trn1    v0.2d, v4.2d, v6.2d  // [a00, a10, a20, a30] → dst row j+0
    trn2    v2.2d, v4.2d, v6.2d  // [a02, a12, a22, a32] → dst row j+2
    trn1    v1.2d, v5.2d, v7.2d  // [a01, a11, a21, a31] → dst row j+1
    trn2    v3.2d, v5.2d, v7.2d  // [a03, a13, a23, a33] → dst row j+3

    // dst_base = dst + (j * rows + i) * 4
    madd    x10, x9, x21, x8
    lsl     x10, x10, #2
    add     x10, x20, x10

    // Store 4 transposed rows × 4 floats
    str     q0, [x10]
    str     q1, [x10, x24]
    add     x11, x24, x24       // 2 * dst_stride
    str     q2, [x10, x11]
    add     x11, x11, x24       // 3 * dst_stride
    str     q3, [x10, x11]

    add     x9, x9, #4
    b       .L_tr_tile_j

.L_tr_tile_i_next:
    add     x8, x8, #4
    b       .L_tr_tile_i

    // ══════════════════════════════════════════════════
    // 2. Right edge: i∈[0,rows), j∈[cols_4,cols)
    // ══════════════════════════════════════════════════
.L_tr_right_edge:
    cmp     x5, x22
    b.eq    .L_tr_bottom_edge    // no right edge if cols is multiple of 4

    mov     x8, #0              // i
.L_tr_re_i:
    cmp     x8, x21
    b.ge    .L_tr_bottom_edge
    mov     x9, x5              // j = cols_4
.L_tr_re_j:
    cmp     x9, x22
    b.ge    .L_tr_re_i_next

    // dst[j][i] = src[i][j]
    madd    x10, x8, x22, x9
    ldr     s0, [x19, x10, lsl #2]
    madd    x10, x9, x21, x8
    str     s0, [x20, x10, lsl #2]

    add     x9, x9, #1
    b       .L_tr_re_j
.L_tr_re_i_next:
    add     x8, x8, #1
    b       .L_tr_re_i

    // ══════════════════════════════════════════════════
    // 3. Bottom edge: i∈[rows_4,rows), j∈[0,cols_4)
    // ══════════════════════════════════════════════════
.L_tr_bottom_edge:
    cmp     x4, x21
    b.eq    .L_tr_done           // no bottom edge if rows is multiple of 4

    mov     x8, x4              // i = rows_4
.L_tr_be_i:
    cmp     x8, x21
    b.ge    .L_tr_done
    mov     x9, #0              // j
.L_tr_be_j:
    cmp     x9, x5
    b.ge    .L_tr_be_i_next

    madd    x10, x8, x22, x9
    ldr     s0, [x19, x10, lsl #2]
    madd    x10, x9, x21, x8
    str     s0, [x20, x10, lsl #2]

    add     x9, x9, #1
    b       .L_tr_be_j
.L_tr_be_i_next:
    add     x8, x8, #1
    b       .L_tr_be_i

.L_tr_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x29, x30, [sp], #64
    ret
