// matmul_bfp16_tn.s — BF16 matmul C = A^T × B via SME2 BFMOPA
//
// void matmul_bfp16_tn(const __bf16 *A, const __bf16 *B, float *C,
//                      long M, long N, long K)
//
// A is K×M row-major (transposed to get M×K)
// B is K×N row-major
// C is M×N row-major (FP32 output)
//
// C[m,n] = sum_k A[k,m] * B[k,n]
//
// Outer product formulation:
//   C = sum_k outer(A[k,:], B[k,:])
// Where A[k,:] is row k of (K×M) matrix A - CONTIGUOUS
// And B[k,:] is row k of (K×N) matrix B - CONTIGUOUS
//
// This is simpler than NN because both operands are contiguous rows!
// No complex ZA transposition needed for A.
//
// Three phases:
//   Phase 1: preprocess_r — rearrange B rows into N-tile-contiguous format (2-way zip)
//   Phase 2: preprocess_l — rearrange A rows into M-tile-contiguous format
//   Phase 3: matmul_opt  — BFMOPA outer products

.section __TEXT,__text,regular,pure_instructions
.global _matmul_bfp16_tn
.p2align 4

_matmul_bfp16_tn:
    stp     x29, x30, [sp, #-224]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]

    // Save arguments
    mov     x21, x0                 // A (K×M)
    mov     x22, x1                 // B (K×N)
    mov     x23, x2                 // C (M×N)
    mov     x24, x3                 // M
    mov     x25, x4                 // N
    mov     x26, x5                 // K

    // Early exit
    cbz     x24, .Ltn_done
    cbz     x25, .Ltn_done
    cbz     x26, .Ltn_done

    // Query streaming SVL
    smstart sm
    cntw    x15                     // SVLs
    cntb    x17                     // SVLb
    cnth    x9                      // SVLh (must be inside streaming mode)
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

    // B_mod size = ceil(N / SVLh) * SVLh * K_mod * 2 bytes
    add     x8, x25, x9
    sub     x8, x8, #1
    udiv    x8, x8, x9
    mul     x8, x8, x9              // N_pad
    mul     x8, x8, x27
    lsl     x8, x8, #1              // bytes (bf16)
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
    // Phase 1: preprocess_r — rearrange B into N-tile-contiguous format
    // B is K×N row-major, rows are contiguous
    // ================================================================
    cntb    x5                      // VL bytes
    lsl     x16, x25, #1            // N * 2 (B row stride in bytes)
    mul     x11, x27, x5            // K_mod * VL bytes (B_mod stride per N-tile)
    mov     x15, #0

    ptrue   pn9.b

    mov     x8, x22                 // B base
    mov     x9, x20                 // B_mod base
    add     x10, x22, x25, lsl #1   // B + N*2 bytes
    whilelt p2.b, x8, x10

.Ltn_pp_r_N:
    mov     x7, x8
    mov     x12, x9
    whilelt p1.h, xzr, x26

    psel    pn11, pn9, p2.b[w15, 0]

    mov     x6, xzr
.Ltn_pp_r_K:
    psel    p0, p2, p1.h[w15, 0]
    psel    p3, p2, p1.h[w15, 1]
    ld1b    {z0.b}, p0/z, [x7]
    ld1b    {z1.b}, p3/z, [x7, x16]

    zip     {z4.h-z5.h}, z0.h, z1.h

    st1b    {z4.b-z5.b}, pn11, [x12]

    add     x7, x7, x25, lsl #2     // 2 rows × N × 2 bytes = 4*N bytes
    addvl   x12, x12, #2
    add     x6, x6, #2
    whilelt p1.h, x6, x26
    b.first .Ltn_pp_r_K

    add     x9, x9, x11, lsl #1
    addvl   x8, x8, #1
    whilelt p2.b, x8, x10
    b.first .Ltn_pp_r_N

    // ================================================================
    // Phase 2: preprocess_l — rearrange A (K×M) into M-tile format
    // A[k,:] is row k, contiguous M elements (bf16)
    // Use ZA transposition to get M-tile-contiguous layout
    // ================================================================
    cntw    x5                      // SVLs
    lsl     x18, x24, #1            // M * 2 (A row stride in bytes, since A is K×M)
    mul     x11, x5, x18            // SVLs * M * 2 (K steps for SVLs rows)
    mul     x15, x5, x27            // SVLs * K_mod
    lsl     x2, x15, #2             // SVLs * K_mod * 4 bytes

    mul     x4, x5, x5              // SVLs^2
    lsl     x16, x4, #1
    add     x16, x16, x4            // 3 * SVLs^2
    cntb    x17                     // SVLb

    mov     x28, #0                 // Loop_M counter
    mov     x8, x21                 // A base
    mov     x9, x19                 // A_mod base
    whilelt p0.s, x28, x24

.Ltn_pp_l_M:
    mov     x7, x9
    mov     x10, x8
    add     x3, x8, x18             // exit = A_base + M*2
    whilelt pn12.b, x10, x3, vlx4
    mov     x13, #0
    mov     x14, x4
    lsl     x0, x4, #1
    mov     x1, x16

.Ltn_pp_l_K:
    mov     x6, x10

    mov     w12, #0
.Ltn_pp_l_load:
    psel    pn8, pn12, p0.b[w12, #0]
    psel    pn9, pn12, p0.b[w12, #4]
    ld1b    {z0.b-z3.b}, pn8/z, [x6]
    ld1b    {z4.b-z7.b}, pn9/z, [x6, x18]
    mova    za0h.b[w12, 0:3], {z0.b-z3.b}
    mova    za0h.b[w12, 4:7], {z4.b-z7.b}
    add     w12, w12, #8
    add     x6, x6, x18, lsl #1
    cmp     w12, w17
    b.mi    .Ltn_pp_l_load

    mov     w12, #0
.Ltn_pp_l_store:
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
    b.mi    .Ltn_pp_l_store

    add     x13, x13, x16
    add     x14, x14, x16
    add     x0, x0, x16
    add     x1, x1, x16
    addvl   x10, x10, #4
    whilelt pn12.b, x10, x3, vlx4
    b.first .Ltn_pp_l_K

    add     x8, x8, x11
    add     x9, x9, x2

    incw    x28
    whilelt p0.s, x28, x24
    b.first .Ltn_pp_l_M

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

    // B_mod stride per N-tile: K_mod * VL bytes
    cntb    x17
    mul     x18, x27, x17

    lsl     x3, x24, #2             // 4*M
    mov     x4, x19                 // a_base = A_mod
    mov     x28, x23                // C base
    mov     x12, #0                 // M counter
    mov     x16, #0
    mov     w15, #0
    sub     w6, w6, #8
    ptrue   pn10.b
    whilelt p2.b, x12, x3

.Ltn_mm_M:
    addvl   x12, x12, #1
    whilelt p3.b, x12, x3

    mov     x5, x20                 // b_base = B_mod
    mov     x22, x28                // c_ptr
    mov     x13, #0

    whilelt pn9.b, x13, x11, vlx2

.Ltn_mm_N:
    mov     x8, x4                  // a_ptr
    mov     x9, x5                  // b_ptr
    mov     x23, x22

    pext    {p0.b, p1.b}, pn9[0]

    zero    {za}

    add     x10, x4, x7
    add     x17, x4, x7
    addvl   x14, x17, #-1

.Ltn_K_start:
    ld1b    {z1.b}, p2/z, [x8]
    whilelt pn10.b, x8, x17, vlx2
    ld1b    {z2.b-z3.b}, pn9/z, [x9]
    bfmopa  za0.s, p2/m, p0/m, z1.h, z2.h
    ld1b    {z5.b}, p3/z, [x8, x7]
    addvl   x8, x8, #1

.Ltn_Loop_K:
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
    b.mi    .Ltn_Loop_K

    bfmopa  za2.s, p3/m, p0/m, z5.h, z2.h
    bfmopa  za1.s, p2/m, p1/m, z1.h, z3.h
    bfmopa  za3.s, p3/m, p1/m, z5.h, z3.h
    addvl   x9, x9, #2

    cmp     x8, x10
    b.ge    .Ltn_mm_store

.Ltn_Ktail:
    ld1b    {z1.b}, p2/z, [x8]
    ld1b    {z2.b-z3.b}, pn9/z, [x9]
    bfmopa  za0.s, p2/m, p0/m, z1.h, z2.h
    ld1b    {z14.b}, p3/z, [x8, x7]
    bfmopa  za2.s, p3/m, p0/m, z14.h, z2.h
    bfmopa  za1.s, p2/m, p1/m, z1.h, z3.h
    addvl   x9, x9, #2
    bfmopa  za3.s, p3/m, p1/m, z14.h, z3.h

.Ltn_mm_store:
    mov     w14, #0
    psel    pn8, pn9, p2.b[w14, 0]
    psel    pn11, pn9, p3.b[w14, 0]
    mova    {z0.b-z3.b}, za0h.b[w14, 0:3]
    st1w    {z0.s-z1.s}, pn8, [x23]
    st1w    {z2.s-z3.s}, pn11, [x23, x0, lsl #2]

.Ltn_mm_store_loop:
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
    b.mi    .Ltn_mm_store_loop

    psel    pn8, pn9, p2.b[w14, 4]
    psel    pn11, pn9, p3.b[w14, 4]
    mova    {z0.b-z3.b}, za0h.b[w14, 4:7]
    st1w    {z0.s-z1.s}, pn8, [x23, x25, lsl #2]
    st1w    {z2.s-z3.s}, pn11, [x23, x2, lsl #2]

    // Next N tile
    addvl   x22, x22, #2
    addvl   x13, x13, #2
    add     x5, x5, x18
    whilelt pn9.b, x13, x11, vlx2
    b.first .Ltn_mm_N

    // Next M tile
    add     x4, x4, x7, lsl #1
    add     x28, x28, x0, lsl #3
    addvl   x12, x12, #1
    whilelt p2.b, x12, x3
    b.first .Ltn_mm_M

    smstop

    // Free buffers
    mov     x0, x19
    bl      _free
    mov     x0, x20
    bl      _free

.Ltn_done:
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
