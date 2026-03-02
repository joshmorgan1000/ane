// matmul_bfp16_nt.s — BF16 matmul C = A × B^T via SME2 BFMOPA
//
// void matmul_bfp16_nt(const __bf16 *A, const __bf16 *B, float *C,
//                      long M, long N, long K)
//
// A is M×K row-major (bf16)
// B is N×K row-major (bf16) - transposed to get K×N
// C is M×N row-major (FP32 output)
//
// C[m,n] = sum_k A[m,k] * B[n,k]
//
// Outer product formulation:
//   C = sum_k outer(A[:,k], B[:,k])
// Where A[:,k] is column k of A (M×K) - STRIDED (stride = K*2 bytes)
// And B[:,k] is column k of B (N×K) - STRIDED (stride = K*2 bytes)
//
// Unlike NN where B rows are contiguous, here BOTH A and B columns
// require ZA transposition preprocessing.
//
// Three phases:
//   Phase 1: preprocess_r — transpose B columns via ZA into N-tile format
//   Phase 2: preprocess_l — transpose A columns via ZA into M-tile format
//   Phase 3: matmul_opt  — BFMOPA outer products

.section __TEXT,__text,regular,pure_instructions
.global _matmul_bfp16_nt
.p2align 4

_matmul_bfp16_nt:
    stp     x29, x30, [sp, #-224]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]

    // Save arguments
    mov     x21, x0                 // A (M×K)
    mov     x22, x1                 // B (N×K)
    mov     x23, x2                 // C (M×N)
    mov     x24, x3                 // M
    mov     x25, x4                 // N
    mov     x26, x5                 // K

    // Early exit
    cbz     x24, .Lnt_done
    cbz     x25, .Lnt_done
    cbz     x26, .Lnt_done

    // Query streaming SVL
    smstart sm
    cntw    x15                     // SVLs
    cntb    x17                     // SVLb
    smstop  sm

    // K_mod = ceil(K / SVLb) * SVLb
    add     x27, x26, x17
    sub     x27, x27, #1
    udiv    x27, x27, x17
    mul     x27, x27, x17

    // A_mod size = ceil(M / SVLs) * SVLs * K_mod * 4 bytes
    add     x8, x24, x15
    sub     x8, x8, #1
    udiv    x8, x8, x15
    mul     x8, x8, x15             // M_pad
    mul     x28, x8, x27            // M_pad * K_mod
    lsl     x28, x28, #2            // bytes (stored as .s)

    // B_mod size = ceil(N / SVLs) * SVLs * K_mod * 4 bytes
    add     x8, x25, x15
    sub     x8, x8, #1
    udiv    x8, x8, x15
    mul     x8, x8, x15             // N_pad
    mul     x8, x8, x27
    lsl     x8, x8, #2              // bytes (stored as .s after transposition)
    str     x8, [sp, #128]          // B_mod size
    stp     d8,  d9,  [sp, #144]
    stp     d10,  d11,  [sp, #160]
    stp     d12,  d13,  [sp, #176]
    stp     d14,  d15,  [sp, #192]

    // Allocate A_mod
    mov     x0, x28
    bl      _malloc
    mov     x19, x0
    mov     x0, x19
    mov     x1, x28
    bl      _bzero

    // Allocate B_mod
    ldr     x0, [sp, #128]
    bl      _malloc
    mov     x20, x0
    mov     x0, x20
    ldr     x1, [sp, #128]
    bl      _bzero

    smstart

    // ================================================================
    // Phase 1: preprocess_r — transpose B columns via ZA
    // ================================================================
    // B is N×K, we need columns (strided by K*2 bytes)
    // Same ZA transposition technique as preprocess_l

    cntw    x5                      // SVLs
    lsl     x18, x26, #1            // K*2 (B row stride in bytes, since B is N×K)
    mul     x11, x5, x18            // SVLs * K * 2 (N-tile stride in original B)
    mul     x15, x5, x27            // SVLs * K_mod (.s element count per N-tile)
    lsl     x2, x15, #2             // SVLs * K_mod * 4 (B_mod byte stride per N-tile)

    mul     x4, x5, x5              // SVLs^2
    lsl     x16, x4, #1
    add     x16, x16, x4            // 3 * SVLs^2
    cntb    x17                     // SVLb

    mov     x28, #0                 // Loop_N counter
    mov     x8, x22                 // B base
    mov     x9, x20                 // B_mod base
    whilelt p0.s, x28, x25          // compare against N

.Lnt_pp_r_N:
    mov     x7, x9
    mov     x10, x8
    add     x3, x8, x18             // exit = B_base + K*2 (real data extent)
    whilelt pn12.b, x10, x3, vlx4
    mov     x13, #0
    mov     x14, x4
    lsl     x0, x4, #1
    mov     x1, x16

.Lnt_pp_r_K:
    mov     x6, x10

    mov     w12, #0
.Lnt_pp_r_load:
    psel    pn8, pn12, p0.b[w12, #0]
    psel    pn9, pn12, p0.b[w12, #4]
    ld1b    {z0.b-z3.b}, pn8/z, [x6]
    ld1b    {z4.b-z7.b}, pn9/z, [x6, x18]
    mova    za0h.b[w12, 0:3], {z0.b-z3.b}
    mova    za0h.b[w12, 4:7], {z4.b-z7.b}
    add     w12, w12, #8
    add     x6, x6, x18, lsl #1
    cmp     w12, w17
    b.mi    .Lnt_pp_r_load

    mov     w12, #0
.Lnt_pp_r_store:
    whilelt pn8.s, x13, x15, vlx4
    whilelt pn9.s, x14, x15, vlx4
    whilelt pn10.s, x0, x15, vlx4
    whilelt pn11.s, x1, x15, vlx4
    mova    {z0.s-z3.s}, za0v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1v.s[w12, 0:3]
    mova    {z8.s-z11.s}, za2v.s[w12, 0:3]
    mova    {z12.s-z15.s}, za3v.s[w12, 0:3]
    add     w12, w12, #4
    st1w    {z0.s-z3.s}, pn8, [x7, x13, lsl #2]
    st1w    {z4.s-z7.s}, pn9, [x7, x14, lsl #2]
    st1w    {z8.s-z11.s}, pn10, [x7, x0, lsl #2]
    st1w    {z12.s-z15.s}, pn11, [x7, x1, lsl #2]
    incw    x13, all, mul #4
    incw    x14, all, mul #4
    incw    x0, all, mul #4
    incw    x1, all, mul #4
    cmp     w12, w5
    b.mi    .Lnt_pp_r_store

    add     x13, x13, x16
    add     x14, x14, x16
    add     x0, x0, x16
    add     x1, x1, x16
    addvl   x10, x10, #4
    whilelt pn12.b, x10, x3, vlx4
    b.first .Lnt_pp_r_K

    add     x8, x8, x11             // B_base += SVLs * K * 2
    add     x9, x9, x2              // B_mod_base += SVLs * K_mod * 4

    incw    x28
    whilelt p0.s, x28, x25
    b.first .Lnt_pp_r_N

    // ================================================================
    // Phase 2: preprocess_l — transpose A columns via ZA
    // ================================================================
    // Same as standard matmul - A is M×K, columns are strided

    cntw    x5                      // SVLs
    lsl     x18, x26, #1            // K*2 (A row stride in bytes)
    mul     x11, x5, x18            // SVLs * K * 2 (M-tile stride in original A)
    mul     x15, x5, x27            // SVLs * K_mod
    lsl     x2, x15, #2             // SVLs * K_mod * 4

    mul     x4, x5, x5              // SVLs^2
    lsl     x16, x4, #1
    add     x16, x16, x4            // 3 * SVLs^2
    cntb    x17                     // SVLb

    mov     x28, #0                 // Loop_M counter
    mov     x8, x21                 // A base
    mov     x9, x19                 // A_mod base
    whilelt p0.s, x28, x24

.Lnt_pp_l_M:
    mov     x7, x9
    mov     x10, x8
    add     x3, x8, x18             // exit = A_base + K*2
    whilelt pn12.b, x10, x3, vlx4
    mov     x13, #0
    mov     x14, x4
    lsl     x0, x4, #1
    mov     x1, x16

.Lnt_pp_l_K:
    mov     x6, x10

    mov     w12, #0
.Lnt_pp_l_load:
    psel    pn8, pn12, p0.b[w12, #0]
    psel    pn9, pn12, p0.b[w12, #4]
    ld1b    {z0.b-z3.b}, pn8/z, [x6]
    ld1b    {z4.b-z7.b}, pn9/z, [x6, x18]
    mova    za0h.b[w12, 0:3], {z0.b-z3.b}
    mova    za0h.b[w12, 4:7], {z4.b-z7.b}
    add     w12, w12, #8
    add     x6, x6, x18, lsl #1
    cmp     w12, w17
    b.mi    .Lnt_pp_l_load

    mov     w12, #0
.Lnt_pp_l_store:
    whilelt pn8.s, x13, x15, vlx4
    whilelt pn9.s, x14, x15, vlx4
    whilelt pn10.s, x0, x15, vlx4
    whilelt pn11.s, x1, x15, vlx4
    mova    {z0.s-z3.s}, za0v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1v.s[w12, 0:3]
    mova    {z8.s-z11.s}, za2v.s[w12, 0:3]
    mova    {z12.s-z15.s}, za3v.s[w12, 0:3]
    add     w12, w12, #4
    st1w    {z0.s-z3.s}, pn8, [x7, x13, lsl #2]
    st1w    {z4.s-z7.s}, pn9, [x7, x14, lsl #2]
    st1w    {z8.s-z11.s}, pn10, [x7, x0, lsl #2]
    st1w    {z12.s-z15.s}, pn11, [x7, x1, lsl #2]
    incw    x13, all, mul #4
    incw    x14, all, mul #4
    incw    x0, all, mul #4
    incw    x1, all, mul #4
    cmp     w12, w5
    b.mi    .Lnt_pp_l_store

    add     x13, x13, x16
    add     x14, x14, x16
    add     x0, x0, x16
    add     x1, x1, x16
    addvl   x10, x10, #4
    whilelt pn12.b, x10, x3, vlx4
    b.first .Lnt_pp_l_K

    add     x8, x8, x11
    add     x9, x9, x2

    incw    x28
    whilelt p0.s, x28, x24
    b.first .Lnt_pp_l_M

    // ================================================================
    // Phase 3: matmul_opt — BFMOPA outer products
    // ================================================================
    cntb    x6                      // SVLb
    cntw    x15                     // SVLs
    lsl     x11, x25, #2            // 4*N
    mul     x0, x15, x25            // SVLs*N
    add     x2, x0, x25             // (SVLs+1)*N

    // A_mod stride per M-tile: K_mod * SVLb
    mul     x7, x27, x6

    // B_mod stride per N-tile: K_mod * SVLb (same structure as A_mod now)
    mul     x18, x27, x6

    lsl     x3, x24, #2             // 4*M
    mov     x4, x19                 // a_base = A_mod
    mov     x28, x23                // C base
    mov     x12, #0                 // M counter
    mov     x16, #0
    mov     w15, #0
    sub     w6, w6, #8
    ptrue   pn10.b
    whilelt p2.b, x12, x3

.Lnt_mm_M:
    addvl   x12, x12, #1
    whilelt p3.b, x12, x3

    mov     x5, x20                 // b_base = B_mod
    mov     x22, x28                // c_ptr
    mov     x13, #0

    whilelt pn9.b, x13, x11, vlx2

.Lnt_mm_N:
    mov     x8, x4                  // a_ptr
    mov     x9, x5                  // b_ptr
    mov     x23, x22

    pext    {p0.b, p1.b}, pn9[0]

    zero    {za}

    add     x10, x4, x7             // a_end = a_base + K_mod * SVLb
    add     x17, x4, x7
    addvl   x14, x17, #-1           // for unrolled loop bound

.Lnt_K_start:
    ld1b    {z1.b}, p2/z, [x8]
    whilelt pn10.b, x8, x17, vlx2
    ld1b    {z2.b-z3.b}, pn9/z, [x9]
    bfmopa  za0.s, p2/m, p0/m, z1.h, z2.h
    ld1b    {z5.b}, p3/z, [x8, x7]
    addvl   x8, x8, #1

.Lnt_Loop_K:
    bfmopa  za2.s, p3/m, p0/m, z5.h, z2.h
    bfmopa  za1.s, p2/m, p1/m, z1.h, z3.h
    psel    pn11, pn10, p3.s[w15, #0]
    ld1b    {z0.b-z1.b}, pn10/z, [x8]
    bfmopa  za3.s, p3/m, p1/m, z5.h, z3.h
    ld1b    {z6.b-z7.b}, pn9/z, [x9, #2, mul vl]

    bfmopa  za0.s, p2/m, p0/m, z0.h, z6.h
    ld1b    {z4.b-z5.b}, pn11/z, [x8, x7]

    bfmopa  za2.s, p3/m, p0/m, z4.h, z6.h
    addvl   x9, x9, #4

    bfmopa  za1.s, p2/m, p1/m, z0.h, z7.h

    bfmopa  za3.s, p3/m, p1/m, z4.h, z7.h
    ld1b    {z2.b-z3.b}, pn9/z, [x9]

    bfmopa  za0.s, p2/m, p0/m, z1.h, z2.h
    addvl   x8, x8, #2

    cmp     x8, x14
    b.mi    .Lnt_Loop_K

    bfmopa  za2.s, p3/m, p0/m, z5.h, z2.h
    bfmopa  za1.s, p2/m, p1/m, z1.h, z3.h
    bfmopa  za3.s, p3/m, p1/m, z5.h, z3.h
    addvl   x9, x9, #2

    cmp     x8, x10
    b.ge    .Lnt_mm_store

.Lnt_Ktail:
    ld1b    {z1.b}, p2/z, [x8]
    ld1b    {z2.b-z3.b}, pn9/z, [x9]
    bfmopa  za0.s, p2/m, p0/m, z1.h, z2.h
    ld1b    {z14.b}, p3/z, [x8, x7]
    bfmopa  za2.s, p3/m, p0/m, z14.h, z2.h
    bfmopa  za1.s, p2/m, p1/m, z1.h, z3.h
    addvl   x9, x9, #2
    bfmopa  za3.s, p3/m, p1/m, z14.h, z3.h

.Lnt_mm_store:
    mov     w14, #0
    psel    pn8, pn9, p2.b[w14, 0]
    psel    pn11, pn9, p3.b[w14, 0]
    mova    {z0.b-z3.b}, za0h.b[w14, 0:3]
    st1w    {z0.s-z1.s}, pn8, [x23]
    st1w    {z2.s-z3.s}, pn11, [x23, x0, lsl #2]

.Lnt_mm_store_loop:
    psel    pn8, pn9, p2.b[w14, 4]
    psel    pn11, pn9, p3.b[w14, 4]
    mova    {z0.b-z3.b}, za0h.b[w14, 4:7]
    st1w    {z0.s-z1.s}, pn8, [x23, x25, lsl #2]
    st1w    {z2.s-z3.s}, pn11, [x23, x2, lsl #2]

    add     x23, x23, x25, lsl #3
    add     w14, w14, #8

    psel    pn8, pn9, p2.b[w14, 0]
    psel    pn11, pn9, p3.b[w14, 0]
    mova    {z0.b-z3.b}, za0h.b[w14, 0:3]
    st1w    {z0.s-z1.s}, pn8, [x23]
    st1w    {z2.s-z3.s}, pn11, [x23, x0, lsl #2]
    cmp     w14, w6
    b.mi    .Lnt_mm_store_loop

    psel    pn8, pn9, p2.b[w14, 4]
    psel    pn11, pn9, p3.b[w14, 4]
    mova    {z0.b-z3.b}, za0h.b[w14, 4:7]
    st1w    {z0.s-z1.s}, pn8, [x23, x25, lsl #2]
    st1w    {z2.s-z3.s}, pn11, [x23, x2, lsl #2]

    // Next N tile
    addvl   x22, x22, #2
    addvl   x13, x13, #2
    add     x5, x5, x18             // B_mod += K_mod * SVLb
    whilelt pn9.b, x13, x11, vlx2
    b.first .Lnt_mm_N

    // Next M tile
    add     x4, x4, x7, lsl #1      // A_mod += K_mod * SVLb * 2
    add     x28, x28, x0, lsl #3    // C += SVLs * N * 4 * 2
    addvl   x12, x12, #1
    whilelt p2.b, x12, x3
    b.first .Lnt_mm_M

    smstop

    // Free buffers
    mov     x0, x19
    bl      _free
    mov     x0, x20
    bl      _free

.Lnt_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     d8,  d9,  [sp, #144]
    ldp     d10,  d11,  [sp, #160]
    ldp     d12,  d13,  [sp, #176]
    ldp     d14,  d15,  [sp, #192]
    ldp     x29, x30, [sp], #224
    ret
