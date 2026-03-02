// matmul_fp32_tn.s — FP32 matmul C = A^T × B via SME2 FMOPA
//
// void matmul_fp32_tn(const float *A, const float *B, float *C,
//                     long M, long N, long K)
//
// A is K×M row-major (transposed to get M×K)
// B is K×N row-major
// C is M×N row-major
//
// C[m,n] = sum_k A[k,m] * B[k,n]
//
// Outer product formulation:
//   C = sum_k outer(A[k,:], B[k,:])
// Where A[k,:] is row k of (K×M) matrix A - CONTIGUOUS
// And B[k,:] is row k of (K×N) matrix B - CONTIGUOUS
//
// This is actually simpler than NN because both operands are contiguous rows!
// No complex ZA transposition needed for A.
//
// Three phases:
//   Phase 1: preprocess_r — rearrange B rows into N-tile-contiguous format
//   Phase 2: preprocess_l — rearrange A rows into M-tile-contiguous format
//   Phase 3: matmul_opt  — FMOPA outer products
//
// Registers:
//   x19 = A_mod ptr    x20 = B_mod ptr
//   x21 = A            x22 = B
//   x23 = C            x24 = M    x25 = N    x26 = K

.section __TEXT,__text,regular,pure_instructions
.global _matmul_fp32_tn
.p2align 4

_matmul_fp32_tn:
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
    lsl     x28, x28, #2            // bytes

    // B_mod size = ceil(N / (2*SVLs)) * (2*SVLs) * K_mod * 4 bytes
    lsl     x9, x15, #1             // 2*SVLs
    add     x8, x25, x9
    sub     x8, x8, #1
    udiv    x8, x8, x9
    mul     x8, x8, x9              // N_pad
    mul     x8, x8, x27
    lsl     x8, x8, #2
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
    // Same as NN matmul since B is K×N row-major
    // ================================================================
    cntb    x5
    lsl     x11, x25, #2            // N * 4 (B row stride)
    lsl     x17, x5, #1             // 2*SVLb
    mul     x16, x27, x17           // K_mod * 2*SVLb

    ptrue   pn9.b

    mov     x8, x22                 // B base
    mov     x9, x20                 // B_mod base
    lsl     x10, x25, #2
    add     x10, x10, x22           // end condition
    whilelt pn11.b, x8, x10, vlx2

.Ltn_pp_r_N:
    mov     x7, x8
    mov     x12, x9
    mov     x6, xzr

.Ltn_pp_r_K:
    cmp     x6, x26
    b.ge    .Ltn_pp_r_N_next

    ld1b    {z0.b-z1.b}, pn11/z, [x7]
    st1b    {z0.b-z1.b}, pn11, [x12]

    add     x7, x7, x11
    addvl   x12, x12, #2
    add     x6, x6, #1
    b       .Ltn_pp_r_K

.Ltn_pp_r_N_next:
    add     x9, x9, x16
    addvl   x8, x8, #2
    whilelt pn11.b, x8, x10, vlx2
    b.first .Ltn_pp_r_N

    // ================================================================
    // Phase 2: preprocess_l — rearrange A (K×M) into M-tile format
    // For TN: A is K×M, we want to pack rows of A into M-tile chunks
    // A[k,:] is row k, contiguous M elements
    // Store as: for each M-tile, K_mod rows of that tile
    // ================================================================
    cntw    x5                      // SVLs
    lsl     x18, x24, #2            // M * 4 (A row stride, since A is K×M)
    lsl     x17, x5, #1             // 2*SVLs
    mul     x16, x27, x17           // K_mod * 2*SVLs * 4 bytes per M-tile
    lsl     x16, x16, #1            // adjust for float

    ptrue   pn9.b

    mov     x8, x21                 // A base
    mov     x9, x19                 // A_mod base
    lsl     x10, x24, #2
    add     x10, x10, x21           // end = A + M*4
    whilelt pn11.b, x8, x10, vlx2

.Ltn_pp_l_M:
    mov     x7, x8                  // A_ptr for this M-tile
    mov     x12, x9                 // A_mod_ptr
    mov     x6, xzr                 // k counter

.Ltn_pp_l_K:
    cmp     x6, x26
    b.ge    .Ltn_pp_l_M_next

    // Load row k of A at M-tile offset
    ld1b    {z0.b-z1.b}, pn11/z, [x7]
    st1b    {z0.b-z1.b}, pn11, [x12]

    add     x7, x7, x18             // next row of A (stride = M*4)
    addvl   x12, x12, #2
    add     x6, x6, #1
    b       .Ltn_pp_l_K

.Ltn_pp_l_M_next:
    mul     x13, x27, x5
    lsl     x13, x13, #3            // K_mod * SVLs * 8 = K_mod * 2*SVLb bytes (2 vectors per K-step)
    add     x9, x9, x13
    addvl   x8, x8, #2
    whilelt pn11.b, x8, x10, vlx2
    b.first .Ltn_pp_l_M

    // ================================================================
    // Phase 3: matmul_opt — FMOPA outer products
    // For each K: za += outer(A_mod[k, m_tile], B_mod[k, n_tile])
    // ================================================================
    cntb    x6                      // SVLb
    cntw    x15                     // SVLs
    lsl     x11, x25, #2            // 4*N
    mul     x0, x15, x25            // SVLs*N
    add     x2, x0, x25             // (SVLs+1)*N

    // A_mod stride per M-tile: K_mod * 2*SVLs * 4 = K_mod * 2*SVLb bytes (2 vectors per K-step)
    mul     x7, x27, x15
    lsl     x7, x7, #3

    // B_mod stride per N-tile: K_mod * 2*SVLb
    lsl     x17, x6, #1
    mul     x18, x27, x17

    lsl     x3, x24, #2             // 4*M
    mov     x4, x19                 // a_base = A_mod
    mov     x28, x23                // C base
    mov     x12, #0                 // M counter
    mov     x16, #0
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

    mov     x10, #0                 // k counter

.Ltn_mm_K:
    cmp     x10, x27                // compare to K_mod
    b.ge    .Ltn_mm_store

    // Load A_mod[k] for this M-tile
    ld1w    {z0.s}, p2/z, [x8]
    ld1w    {z1.s}, p3/z, [x8, #1, mul vl]

    // Load B_mod[k] for this N-tile
    ld1w    {z2.s-z3.s}, pn9/z, [x9]

    // Outer products
    fmopa   za0.s, p2/m, p0/m, z0.s, z2.s
    fmopa   za1.s, p2/m, p1/m, z0.s, z3.s
    fmopa   za2.s, p3/m, p0/m, z1.s, z2.s
    fmopa   za3.s, p3/m, p1/m, z1.s, z3.s

    // Advance k
    addvl   x8, x8, #2              // A_mod: 2 vectors per k
    addvl   x9, x9, #2              // B_mod: 2 vectors per k
    add     x10, x10, #1
    b       .Ltn_mm_K

.Ltn_mm_store:
    // Store ZA tiles to C
    mov     w14, #0
.Ltn_store_loop:
    psel    pn8, pn9, p2.b[w14, 0]
    psel    pn11, pn9, p3.b[w14, 0]
    mova    {z4.b-z7.b}, za0h.b[w14, 0:3]
    st1w    {z4.s-z5.s}, pn8, [x23]
    st1w    {z6.s-z7.s}, pn11, [x23, x0, lsl #2]

    psel    pn8, pn9, p2.b[w14, 4]
    psel    pn11, pn9, p3.b[w14, 4]
    mova    {z4.b-z7.b}, za0h.b[w14, 4:7]
    st1w    {z4.s-z5.s}, pn8, [x23, x25, lsl #2]
    st1w    {z6.s-z7.s}, pn11, [x23, x2, lsl #2]

    add     x23, x23, x25, lsl #3
    add     w14, w14, #8
    cmp     w14, #56
    b.le    .Ltn_store_loop

    // Next N tile
    addvl   x22, x22, #2
    addvl   x13, x13, #2
    add     x5, x5, x18             // B_mod += K_mod * 2*SVLb
    whilelt pn9.b, x13, x11, vlx2
    b.first .Ltn_mm_N

    // Next M tile
    add     x4, x4, x7              // A_mod += K_mod * SVLs * 4
    add     x28, x28, x0, lsl #3    // C += SVLs * N * 4 * 2
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
