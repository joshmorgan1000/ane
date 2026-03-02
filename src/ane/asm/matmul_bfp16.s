// matmul_bfp16.s — BF16 matmul via SME2, following matmul_int8.s architecture
//
// Two entry points:
//   void matmul_bfp16(const __bf16 *A, const __bf16 *B, float *C,
//                     long M, long N, long K, void *a_work, void *b_work)
//   void matmul_bfp16_f16acc(const __bf16 *A, const __bf16 *B, __bf16 *C,
//                            long M, long N, long K, void *a_work, void *b_work)
//
// a_work / b_work: optional scratch buffers for preprocessed A and B.
//   If NULL, malloc'd internally (convenient but slower for repeated calls).
//   If non-NULL, caller provides pre-allocated workspace (zero allocation overhead).
//
// Single function, three phases (identical structure to matmul_int8):
//   Phase 1: preprocess_r — rearrange B → B_mod (2-way zip, rank-2)
//   Phase 2: preprocess_l — rearrange A → A_mod (byte ZA transposition)
//   Phase 3: matmul_opt  — bfmopa za.s, software-pipelined K loop
//
// Key differences from int8:
//   - Element size: 1 byte → 2 bytes (bf16)
//   - Outer product rank: 4 (smopa .b) → 2 (bfmopa .h)
//   - K interleave: 4-way zip → 2-way zip, ceil(K/4) → ceil(K/2)
//   - Phase 1: 2 row loads + 2-way zip (no upper/lower split)
//   - Phase 2: same byte ZA ops, row stride K*2 instead of K
//   - Phase 3: bfmopa za.s instead of smopa za.s
//
// Persistent registers (callee-saved):
//   x19 = A_mod ptr    x20 = B_mod ptr
//   x21 = A            x22 = B (free after preprocessing)
//   x23 = C            x24 = M    x25 = N    x26 = K
//   x27 = K_mod        x28 = scratch (callee-saved, survives malloc)
//
// Stack frame (160 bytes):
//   [sp, #0]:   x29, x30
//   [sp, #16]:  x19, x20
//   [sp, #32]:  x21, x22
//   [sp, #48]:  x23, x24
//   [sp, #64]:  x25, x26
//   [sp, #80]:  x27, x28
//   [sp, #96]:  a_work, b_work (saved from x6, x7)
//   [sp, #112]: need_free_a, need_free_b (flags)
//   [sp, #128]: b_mod_size, (spare)

.section __TEXT,__text,regular,pure_instructions

// ============================================================
// FP32 accumulator entry point
// ============================================================
.global _matmul_bfp16
.p2align 4
_matmul_bfp16:
    mov     x8, #0
    b       .Lbf_common

// ============================================================
// FP16 accumulator entry point (wrapper: fp32 path + convert)
// ============================================================
.global _matmul_bfp16_f16acc
.p2align 4
_matmul_bfp16_f16acc:
    stp     x29, x30, [sp, #-208]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]

    mov     x19, x0                 // A
    mov     x20, x1                 // B
    mov     x21, x2                 // C (bf16*)
    mov     x22, x3                 // M
    mov     x23, x4                 // N
    mov     x24, x5                 // K
    mov     x25, x6                 // a_work
    mov     x26, x7                 // b_work

    // Allocate float temp: M*N*4
    mul     x0, x22, x23
    lsl     x0, x0, #2
    bl      _malloc
    mov     x27, x0                 // fp32_tmp

    // Call matmul_bfp16(A, B, fp32_tmp, M, N, K, a_work, b_work)
    mov     x0, x19
    mov     x1, x20
    mov     x2, x27
    mov     x3, x22
    mov     x4, x23
    mov     x5, x24
    mov     x6, x25
    mov     x7, x26
    bl      _matmul_bfp16

    // Convert fp32_tmp → C (bf16)
    mul     x28, x22, x23          // M*N
    mov     x8, x27                // src (float*)
    mov     x9, x21                // dst (bf16*)
    mov     x10, #0
.Lf16acc_cvt:
    cmp     x10, x28
    b.ge    .Lf16acc_cvt_done
    ldr     s0, [x8, x10, lsl #2]
    bfcvt   h0, s0
    str     h0, [x9, x10, lsl #1]
    add     x10, x10, #1
    b       .Lf16acc_cvt
.Lf16acc_cvt_done:

    mov     x0, x27
    bl      _free

    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     d8,  d9,  [sp, #144]
    ldp     d10,  d11,  [sp, #160]
    ldp     d12,  d13,  [sp, #176]
    ldp     d14,  d15,  [sp, #192]
    ldp     x29, x30, [sp], #208
    ret

// ============================================================
// Common implementation
// ============================================================
.Lbf_common:
    stp     x29, x30, [sp, #-208]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]

    // Save arguments to callee-saved registers
    mov     x21, x0                 // A
    mov     x22, x1                 // B
    mov     x23, x2                 // C
    mov     x24, x3                 // M
    mov     x25, x4                 // N
    mov     x26, x5                 // K
    mov     x27, x8                 // (unused flag)
    stp     x6, x7, [sp, #96]      // save workspace ptrs

    // ── Query streaming SVL ─────────────────────────────────
    smstart sm
    cntw    x15                     // SVLs
    cnth    x17                     // SVLh  (int8 uses cntb for SVLb)
    smstop  sm

    // ── Compute K_mod ─────────────────────────────────────────
    // K_mod = ceil(K/SVLb)*SVLb (pad K to SVLb boundary - same as FP32)
    // x17 = SVLh, but we need SVLb. SVLb = 2*SVLh for BF16.
    lsl     x8, x17, #1            // SVLb = 2*SVLh (64 on M4)
    add     x27, x26, x8
    sub     x27, x27, #1
    udiv    x27, x27, x8
    mul     x27, x27, x8           // K_mod (callee-saved in x27)

    // ── Compute buffer sizes ────────────────────────────────
    // A_mod = M_pad * K_mod * 4 bytes (Phase 2 stores as .s elements!)
    // M_pad = ceil(M / SVLs) * SVLs (match Phase 2 which processes SVLs rows at a time)
    add     x8, x24, x15
    sub     x8, x8, #1
    udiv    x8, x8, x15
    mul     x8, x8, x15            // M_pad (multiple of SVLs)
    mul     x28, x8, x27           // M_pad * K_mod (elements)
    lsl     x28, x28, #2           // * 4 bytes (Phase 2 stores as .s!)

    // B_mod = N_pad * K_mod * 2 bytes
    add     x8, x25, x17
    sub     x8, x8, #1
    udiv    x8, x8, x17
    mul     x8, x8, x17            // N_pad = ceil(N/SVLh)*SVLh
    mul     x8, x8, x27            // N_pad * K_mod (bf16 elements)
    lsl     x8, x8, #1             // * 2 bytes per bf16
    str     x8, [sp, #128]
    stp     d8,  d9,  [sp, #144]
    stp     d10,  d11,  [sp, #160]
    stp     d12,  d13,  [sp, #176]
    stp     d14,  d15,  [sp, #192]

    // ── Allocate A_mod ──────────────────────────────────────
    ldr     x6, [sp, #96]
    cbnz    x6, .Lbf_use_a_work
    mov     x0, x28
    bl      _malloc
    mov     x19, x0
    mov     x8, #1
    str     x8, [sp, #112]
    b       .Lbf_a_done
.Lbf_use_a_work:
    mov     x19, x6
    str     xzr, [sp, #112]
.Lbf_a_done:

    // Zero A_mod (padding bytes must be zero for correct ZA transposition)
    mov     x0, x19
    mov     x1, x28
    bl      _bzero

    // ── Allocate B_mod ──────────────────────────────────────
    ldr     x7, [sp, #104]
    cbnz    x7, .Lbf_use_b_work
    ldr     x0, [sp, #128]
    bl      _malloc
    mov     x20, x0
    mov     x8, #1
    str     x8, [sp, #120]
    b       .Lbf_b_done
.Lbf_use_b_work:
    mov     x20, x7
    str     xzr, [sp, #120]
.Lbf_b_done:

    // Zero B_mod (padding rows beyond K must be zero)
    mov     x0, x20
    ldr     x1, [sp, #128]
    bl      _bzero

    // ── Enter streaming mode + ZA ───────────────────────────
    smstart

    // ================================================================
    // Phase 1: preprocess_r  — rearrange B → B_mod
    // ================================================================
    // bf16 adaptation: 2-way halfword zip (rank-2), no upper/lower split.
    // Loads B rows as raw bytes, zips as halfwords, stores as bytes.
    //
    // int8: 4 rows loaded, 4-way byte zip, 4 vectors → lower/upper pairs
    // bf16: 2 rows loaded, 2-way halfword zip, 2 vectors → single pair

    cntb    x5                      // VL bytes (same as int8)
    lsl     x16, x25, #1           // N*2 bytes (bf16 row stride)
    // K_mod already in x27
    mul     x11, x27, x5           // K_mod*VL_bytes (B_mod stride per N-tile)
    // (int8: lsl x17, x11, #1 for half-tile offset — not needed)
    mov     x15, #0                // psel variable
    // (int8: cnth x13 for SVLb/2 — not needed, no upper half)

    ptrue   pn9.b

    mov     x8, x22                // b_base = B
    mov     x9, x20                // b_mod_base = B_mod
    add     x10, x22, x25, lsl #1  // B + N*2 bytes  (int8: B + N)
    whilelt p2.b, x8, x10          // N dimension predicate (byte addresses)

.Lbf_pp_r_N:
    mov     x7, x8                 // b_ptr
    mov     x12, x9                // b_mod_ptr
    whilelt p1.h, xzr, x26         // K dim predicate (int8: p1.b)

    psel    pn11, pn9, p2.b[w15, 0]
    // (int8: psel pn12 for upper half — not needed)

    mov     x6, xzr                // Loop_K counter
.Lbf_pp_r_K:
    // 2 row loads (int8: 4 row loads)
    psel    p0, p2, p1.h[w15, 0]   // (int8: p1.b[w15, 0])
    psel    p3, p2, p1.h[w15, 1]   // (int8: p1.b[w15, 1])
    ld1b    {z0.b}, p0/z, [x7]
    ld1b    {z1.b}, p3/z, [x7, x16]  // offset N*2  (int8: [x7, x25] = N)
    // (int8: 2 more loads for rows 2,3 — not needed)

    // 2-way halfword zip (int8: 4-way byte zip)
    zip     {z4.h-z5.h}, z0.h, z1.h

    // Single store (int8: lower + upper pair stores)
    st1b    {z4.b-z5.b}, pn11, [x12]
    // (int8: st1b z10-z11, pn12, [x12, x17] — not needed)

    add     x7, x7, x25, lsl #2    // b_ptr += 4*N bytes (same as int8)
                                    // (2 bf16 rows × N × 2 = 4*N bytes)
    addvl   x12, x12, #2           // b_mod_ptr += 2*VL (same)
    add     x6, x6, #2             // (int8: +4)
    whilelt p1.h, x6, x26          // (int8: p1.b)
    b.first .Lbf_pp_r_K

    add     x9, x9, x11, lsl #1    // b_mod_base += 2*ceil(K/2)*VL = full N-tile
                                    // (int8: x17, lsl #1 = 4*ceil(K/4)*SVLb)
    addvl   x8, x8, #1             // b_base += VL bytes (same)
    whilelt p2.b, x8, x10
    b.first .Lbf_pp_r_N

    // ================================================================
    // Phase 2: preprocess_l  — rearrange A → A_mod
    // ================================================================
    // Identical byte-ZA transposition as int8/fp32.
    // Row stride is K*2 bytes (bf16). Uses K_mod from x27.

    cntw    x5                      // SVLs
    lsl     x18, x26, #1           // K*2 = bf16 row stride in bytes
    mul     x11, x5, x18           // SVLs * K*2 (M-tile stride in original A)
    mul     x15, x5, x27           // SVLs * K_mod (.s element count per M-tile)
    lsl     x2, x15, #2            // SVLs * K_mod * 4 bytes (A_mod stride per M-tile)

    mul     x4, x5, x5             // SVLs*SVLs (same)
    lsl     x16, x4, #1
    add     x16, x16, x4           // 3*SVLs*SVLs (same)
    cntb    x17                     // SVLb (same)

    mov     x28, #0                // Loop_M counter
    mov     x8, x21                // matLeft base
    mov     x9, x19                // matLeft_mod base
    whilelt p0.s, x28, x24

.Lbf_pp_l_M:
    mov     x7, x9
    mov     x10, x8
    add     x3, x8, x18            // x8 + K*2 bytes (int8: x8 + K)
    whilelt pn12.b, x10, x3, vlx4
    mov     x13, #0
    mov     x14, x4
    lsl     x0, x4, #1
    mov     x1, x16

.Lbf_pp_l_K:
    mov     x6, x10

    mov     w12, #0
.Lbf_pp_l_load:
    psel    pn8, pn12, p0.b[w12, #0]
    psel    pn9, pn12, p0.b[w12, #4]
    ld1b    {z0.b-z3.b}, pn8/z, [x6]
    ld1b    {z4.b-z7.b}, pn9/z, [x6, x18]    // offset K*2 (int8: x26 = K)
    mova    za0h.b[w12, 0:3], {z0.b-z3.b}
    mova    za0h.b[w12, 4:7], {z4.b-z7.b}
    add     w12, w12, #8
    add     x6, x6, x18, lsl #1    // 2 rows × K*2 bytes (int8: x26, lsl #1)
    cmp     w12, w17                // SVLb (same)
    b.mi    .Lbf_pp_l_load

    mov     w12, #0
.Lbf_pp_l_store:
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
    b.mi    .Lbf_pp_l_store

    add     x13, x13, x16
    add     x14, x14, x16
    add     x0, x0, x16
    add     x1, x1, x16
    addvl   x10, x10, #4
    whilelt pn12.b, x10, x3, vlx4
    b.first .Lbf_pp_l_K

    add     x8, x8, x11            // A base += SVLs*K*2 (int8: SVLs*K)
    add     x9, x9, x2

    incw    x28
    whilelt p0.s, x28, x24
    b.first .Lbf_pp_l_M

    // ================================================================
    // Phase 3: matmul_opt  — bfmopa za.s
    // ================================================================
    // Identical structure to fp32 Phase 3.
    // x7 = K_mod * SVLb (A_mod M-tile stride, accounts for ZA padding).

    cntb    x6                      // SVLb
    cntw    x15                     // SVLs
    lsl     x11, x25, #2           // 4*N
    mul     x21, x15, x25          // SVLs*N
    add     x2, x21, x25           // (SVLs+1)*N
    mul     x7, x27, x6            // K_mod * SVLb
    lsl     x0, x24, #2            // 4*M (same)
    mov     x3, x19                // a_base = A_mod (same)
    mov     x28, x23               // matResult (same)
    mov     x12, #0                // Loop_M counter (same)
    mov     x15, #0                // psel variable (same)
    sub     w6, w6, #8             // SVLb-8 (same)
    ptrue   pn10.b
    whilelt p2.b, x12, x0

.Lbf_mm_M:
    addvl   x12, x12, #1
    whilelt p3.b, x12, x0

    mov     x4, x20                // b_base = B_mod
    mov     x22, x28               // c_base
    mov     x13, #0
    add     x10, x3, x7
    add     x17, x3, x7
    addvl   x9, x17, #-1

    whilelt pn9.b, x13, x11, vlx2

.Lbf_mm_N:
    mov     x8, x3
    mov     x16, x4
    mov     x23, x22

    pext    {p0.b, p1.b}, pn9[0]

    zero    {za}

    // ── bfmopa K loop (identical pipeline to int8 signed) ──
.Lbf_K_start:
    ld1b    {z1.b}, p2/z, [x8]
    whilelt pn10.b, x8, x17, vlx2
    ld1b    {z2.b-z3.b}, pn9/z, [x16]
    bfmopa  za0.s, p2/m, p0/m, z1.h, z2.h    // (int8: smopa ... z1.b, z2.b)
    ld1b    {z5.b}, p3/z, [x8, x7]
    addvl   x8, x8, #1

.Lbf_Loop_K:
    bfmopa  za2.s, p3/m, p0/m, z5.h, z2.h
    bfmopa  za1.s, p2/m, p1/m, z1.h, z3.h
    psel    pn11, pn10, p3.s[w15, #0]
    ld1b    {z0.b-z1.b}, pn10/z, [x8]
    bfmopa  za3.s, p3/m, p1/m, z5.h, z3.h
    ld1b    {z6.b-z7.b}, pn9/z, [x16, #2, mul vl]

    bfmopa  za0.s, p2/m, p0/m, z0.h, z6.h
    ld1b    {z4.b-z5.b}, pn11/z, [x8, x7]

    bfmopa  za2.s, p3/m, p0/m, z4.h, z6.h
    addvl   x16, x16, #4

    bfmopa  za1.s, p2/m, p1/m, z0.h, z7.h

    bfmopa  za3.s, p3/m, p1/m, z4.h, z7.h
    ld1b    {z2.b-z3.b}, pn9/z, [x16]

    bfmopa  za0.s, p2/m, p0/m, z1.h, z2.h
    addvl   x8, x8, #2

    cmp     x8, x9
    b.mi    .Lbf_Loop_K

    bfmopa  za2.s, p3/m, p0/m, z5.h, z2.h
    bfmopa  za1.s, p2/m, p1/m, z1.h, z3.h
    bfmopa  za3.s, p3/m, p1/m, z5.h, z3.h
    addvl   x16, x16, #2

    cmp     x8, x10
    b.ge    .Lbf_mm_store

.Lbf_Ktail:
    ld1b    {z1.b}, p2/z, [x8]
    ld1b    {z2.b-z3.b}, pn9/z, [x16]
    bfmopa  za0.s, p2/m, p0/m, z1.h, z2.h
    ld1b    {z14.b}, p3/z, [x8, x7]
    bfmopa  za2.s, p3/m, p0/m, z14.h, z2.h
    bfmopa  za1.s, p2/m, p1/m, z1.h, z3.h
    addvl   x16, x16, #2
    bfmopa  za3.s, p3/m, p1/m, z14.h, z3.h

    // ── Store results ──────────────────────────────────────
    // For partial M (when p3 has no active lanes), we need special handling
    // because the za0 rows 2-3 store (gated by p3) would be skipped.
.Lbf_mm_store:
    mov     w14, #0
    psel    pn8, pn9, p2.b[w14, 0]
    psel    pn11, pn9, p3.b[w14, 0]
    mova    {z0.b-z3.b}, za0h.b[w14, 0:3]
    st1w    {z0.s-z1.s}, pn8, [x23]
    st1w    {z2.s-z3.s}, pn11, [x23, x21, lsl #2]

.Lbf_mm_store_loop:
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
    st1w    {z2.s-z3.s}, pn11, [x23, x21, lsl #2]
    cmp     w14, w6
    b.mi    .Lbf_mm_store_loop

    psel    pn8, pn9, p2.b[w14, 4]
    psel    pn11, pn9, p3.b[w14, 4]
    mova    {z0.b-z3.b}, za0h.b[w14, 4:7]
    st1w    {z0.s-z1.s}, pn8, [x23, x25, lsl #2]
    st1w    {z2.s-z3.s}, pn11, [x23, x2, lsl #2]

    addvl   x22, x22, #2
    addvl   x13, x13, #2
    whilelt pn9.b, x13, x11, vlx2
    add     x4, x4, x7, lsl #1
    b.first .Lbf_mm_N

    add     x3, x3, x7, lsl #1
    add     x28, x28, x21, lsl #3
    addvl   x12, x12, #1
    whilelt p2.b, x12, x0
    b.first .Lbf_mm_M

    // ── Exit streaming mode ─────────────────────────────────
    smstop

    // ── Conditionally free buffers ──────────────────────────
    ldr     x8, [sp, #112]
    cbz     x8, .Lbf_skip_free_a
    mov     x0, x19
    bl      _free
.Lbf_skip_free_a:
    ldr     x8, [sp, #120]
    cbz     x8, .Lbf_skip_free_b
    mov     x0, x20
    bl      _free
.Lbf_skip_free_b:

    // ── Restore and return ──────────────────────────────────
    // Restore callee-saved SIMD registers
    ldp     d8,  d9,  [sp, #144]
    ldp     d10, d11, [sp, #160]
    ldp     d12, d13, [sp, #176]
    ldp     d14, d15, [sp, #192]
    // Restore callee-saved GPRs
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     x29, x30, [sp], #208
    ret

