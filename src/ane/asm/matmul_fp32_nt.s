// matmul_fp32_nt.s — FP32 matmul C = A × B^T via SME2 FMOPA
//
// void matmul_fp32_nt(const float *A, const float *B, float *C,
//                     long M, long N, long K)
//
// A is M×K row-major
// B is N×K row-major (transposed to get K×N)
// C is M×N row-major
//
// C[m,n] = sum_k A[m,k] * B[n,k]
//
// Preprocessing:
//   Phase 1: preprocess_r for NT — transpose B (N×K) into the same B_mod format
//            that the NN kernel's preprocess_r produces.
//            B_mod layout per N-tile (2*SVLs rows starting at n0):
//              K_mod sequential 2*SVLb-byte slots.
//              Slot k: [B[n0..n0+SVLs-1, k] | B[n0+SVLs..n0+2*SVLs-1, k]]
//            Method: two ZA-block-transpose passes per N-tile, one for each
//            SVLs-row half. Pass 1 uses standard preprocess_l ZA technique on
//            B[n0..n0+SVLs-1,:] and stores columns at doubled-stride positions
//            (element offsets 0, 2*SVLs, 4*SVLs, ...) using separate st1w per
//            vector with bounds-checking predicates. Pass 2 does the same for
//            B[n0+SVLs..n0+2*SVLs-1,:] at offset positions (SVLs, 3*SVLs, ...).
//   Phase 2: preprocess_l — ZA transposition of A (identical to NN kernel).
//            x3/x18 aliasing fixed: exit pointer saved to [sp,#120].
//   Phase 3: matmul_opt  — FMOPA outer products (identical to NN kernel).
//            B_mod N-tile advance fixed: add x5, x5, x3, lsl #1.
//
// Persistent registers (callee-saved):
//   x19 = A_mod ptr    x20 = B_mod ptr
//   x21 = A            x22 = B
//   x23 = C            x24 = M    x25 = N    x26 = K
//   x27 = K_mod
//
// Stack frame (224 bytes):
//   [sp, #0]:   x29, x30
//   [sp, #16]:  x19, x20
//   [sp, #32]:  x21, x22
//   [sp, #48]:  x23, x24
//   [sp, #64]:  x25, x26
//   [sp, #80]:  x27, x28
//   [sp, #96]:  K*4 (row stride, shared Phases 1+2)
//   [sp, #104]: K_mod * SVLb           (Phase 3: B-tile stride)
//   [sp, #112]: 4*M                    (Phase 3: M-loop bound)
//   [sp, #120]: K-loop exit pointer    (Phases 1+2: row_base + K*4)
//   [sp, #128]: B_mod size
//   [sp, #136]: (spare)
//   [sp, #144]: d8,  d9
//   [sp, #160]: d10, d11
//   [sp, #176]: d12, d13
//   [sp, #192]: d14, d15

.section __TEXT,__text,regular,pure_instructions
.global _matmul_fp32_nt
.p2align 4

_matmul_fp32_nt:
    stp     x29, x30, [sp, #-224]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]

    mov     x21, x0                 // A (M×K)
    mov     x22, x1                 // B (N×K)
    mov     x23, x2                 // C (M×N)
    mov     x24, x3                 // M
    mov     x25, x4                 // N
    mov     x26, x5                 // K

    cbz     x24, .Lnt_done
    cbz     x25, .Lnt_done
    cbz     x26, .Lnt_done

    smstart sm
    cntw    x15                     // SVLs
    cntb    x17                     // SVLb
    smstop  sm

    // K_mod = ceil(K / SVLb) * SVLb  (in elements; SVLb=cntb=64 on M4)
    add     x27, x26, x17
    sub     x27, x27, #1
    udiv    x27, x27, x17
    mul     x27, x27, x17

    // A_mod size = ceil(M / SVLs) * SVLs * K_mod * 4 bytes
    add     x8, x24, x15
    sub     x8, x8, #1
    udiv    x8, x8, x15
    mul     x8, x8, x15
    mul     x28, x8, x27
    lsl     x28, x28, #2

    // B_mod size = ceil(N / (2*SVLs)) * (2*SVLs) * K_mod * 4 bytes
    lsl     x9, x15, #1             // 2*SVLs
    add     x8, x25, x9
    sub     x8, x8, #1
    udiv    x8, x8, x9
    mul     x8, x8, x9
    mul     x8, x8, x27
    lsl     x8, x8, #2
    str     x8, [sp, #128]

    stp     d8,  d9,  [sp, #144]
    stp     d10, d11, [sp, #160]
    stp     d12, d13, [sp, #176]
    stp     d14, d15, [sp, #192]

    mov     x0, x28
    bl      _malloc
    mov     x19, x0
    mov     x0, x19
    mov     x1, x28
    bl      _bzero

    ldr     x0, [sp, #128]
    bl      _malloc
    mov     x20, x0
    mov     x0, x20
    ldr     x1, [sp, #128]
    bl      _bzero

    smstart

    // ================================================================
    // Phase 1: preprocess_r for B (N×K) → B_mod
    // ================================================================
    // B is N×K row-major. Each B row = K contiguous floats.
    // B_mod format: per N-tile (2*SVLs rows of B), K_mod sequential 2*SVLb-byte
    // slots, slot k = [B[n0..n0+SVLs-1,k] | B[n0+SVLs..n0+2*SVLs-1,k]].
    //
    // Two ZA-transpose passes per N-tile:
    //   Pass 1: B[n0..n0+SVLs-1,:] → B_mod at element offsets 0,2*SVLs,4*SVLs,...
    //   Pass 2: B[n0+SVLs..n0+2*SVLs-1,:] → offsets SVLs,3*SVLs,5*SVLs,...
    //
    // Each pass uses the same ZA horizontal-load / vertical-extract technique as
    // preprocess_l. The store loop uses separate single-vector st1w per K-step
    // with predicated bounds checking (B_mod pre-zeroed, K-steps beyond K_mod
    // are zeros that may be written but stay within allocated buffer).
    //
    // Register allocation:
    //   x5  = SVLs           x17 = SVLb
    //   x8  = current N-tile first-half base in B (B + n0*K*4)
    //   x9  = current B_mod N-tile base
    //   x11 = 2*SVLs*K*4   (N-tile stride in B = step for outer N loop)
    //   x4  = SVLs*K*4     (half-tile offset: B[n0+SVLs,0] = B[n0,0] + x4)
    //   x28 = N-tile counter (in SVLs steps)
    //   p0  = first-half N predicate  (whilelt p0.s, x28, x25)
    //   p1  = second-half N predicate (whilelt p1.s, x28+SVLs, x25)
    //   [sp,#96]  = K*4 stride
    //   [sp,#120] = K-loop exit pointer for current pass

    cntw    x5                      // SVLs
    lsl     x3, x26, #2             // K*4
    str     x3, [sp, #96]
    lsl     x11, x5, #1
    mul     x11, x11, x3            // 2*SVLs*K*4  (N-tile stride)
    lsr     x4, x11, #1             // SVLs*K*4    (half-tile offset)
    cntb    x17                     // SVLb

    mov     x28, #0                 // N-tile counter (in SVLs steps)
    mov     x8, x22                 // B first-half base
    mov     x9, x20                 // B_mod base
    whilelt p0.s, x28, x25

.Lnt_pp_r_N:
    // Compute second-half predicate p1
    add     x0, x28, x5             // x28 + SVLs
    whilelt p1.s, x0, x25

    // B_mod tile base (for both passes)
    mov     x7, x9

    // ── Pass 1: B[n0..n0+SVLs-1, :] ──────────────────────────────────
    // Uses p0 (first-half rows). Stores at B_mod element offsets:
    //   K-step k → offset 2*k*SVLs.
    // x_lim = 2 * SVLs * K_mod (element limit for pass 1 in B_mod).

    mul     x2, x5, x27             // SVLs * K_mod
    lsl     x2, x2, #1              // 2 * SVLs * K_mod  (= x_lim for pass 1)

    mov     x10, x8                 // B load ptr: starts at B[n0, 0]
    ldr     x3, [sp, #96]
    add     x3, x8, x3              // exit = B[n0,0] + K*4
    str     x3, [sp, #120]

    mov     x13, #0                 // running B_mod element offset (pass 1 starts at 0)
    whilelt pn12.b, x10, x3, vlx4

.Lnt_pp_r_p1_K:
    mov     x6, x10
    mov     w12, #0

.Lnt_pp_r_p1_load:
    psel    pn8, pn12, p0.b[w12, #0]
    psel    pn9, pn12, p0.b[w12, #4]
    ldr     x3, [sp, #96]
    ld1b    {z0.b-z3.b}, pn8/z, [x6]
    ld1b    {z4.b-z7.b}, pn9/z, [x6, x3]
    mova    za0h.b[w12, 0:3], {z0.b-z3.b}
    mova    za0h.b[w12, 4:7], {z4.b-z7.b}
    add     w12, w12, #8
    add     x6, x6, x3, lsl #1
    cmp     w12, w17
    b.mi    .Lnt_pp_r_p1_load

    // Guard: skip store if x13 >= x_lim (this K-block is beyond K_mod)
    cmp     x13, x2
    b.ge    .Lnt_pp_r_p1_K_done

    ptrue   p2.s                    // store predicate (all lanes active)

    // Store each ZA tile sequentially: za0 covers K-steps k_off+0..SVLs-1,
    // za1 covers k_off+SVLs..2*SVLs-1, etc. Each column stored at stride
    // 2*SVLs elements (one B_mod slot width). Processing tiles in order
    // ensures columns map to the correct sequential B_mod slots.
    mov     w12, #0
.Lnt_pp_r_p1_za0:
    mova    {z0.s-z3.s}, za0v.s[w12, 0:3]
    add     w12, w12, #4
    st1w    {z0.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z1.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z2.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z3.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    cmp     w12, w5
    b.mi    .Lnt_pp_r_p1_za0

    mov     w12, #0
.Lnt_pp_r_p1_za1:
    mova    {z0.s-z3.s}, za1v.s[w12, 0:3]
    add     w12, w12, #4
    st1w    {z0.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z1.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z2.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z3.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    cmp     w12, w5
    b.mi    .Lnt_pp_r_p1_za1

    mov     w12, #0
.Lnt_pp_r_p1_za2:
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    add     w12, w12, #4
    st1w    {z0.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z1.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z2.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z3.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    cmp     w12, w5
    b.mi    .Lnt_pp_r_p1_za2

    mov     w12, #0
.Lnt_pp_r_p1_za3:
    mova    {z0.s-z3.s}, za3v.s[w12, 0:3]
    add     w12, w12, #4
    st1w    {z0.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z1.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z2.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z3.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    cmp     w12, w5
    b.mi    .Lnt_pp_r_p1_za3

.Lnt_pp_r_p1_K_done:
    addvl   x10, x10, #4            // advance K-block by 4*SVLb bytes
    ldr     x3, [sp, #120]          // reload exit pointer
    whilelt pn12.b, x10, x3, vlx4
    b.first .Lnt_pp_r_p1_K

    // ── Pass 2: B[n0+SVLs..n0+2*SVLs-1, :] ──────────────────────────
    // Uses p1 (second-half rows). Stores at B_mod element offsets:
    //   K-step k → offset 2*k*SVLs + SVLs.
    // Same x_lim = 2 * SVLs * K_mod (already in x2).

    add     x10, x8, x4             // B[n0+SVLs, 0]
    ldr     x3, [sp, #96]
    add     x3, x10, x3             // exit = B[n0+SVLs, 0] + K*4
    str     x3, [sp, #120]

    mov     x13, x5                 // pass 2 starts at element offset SVLs
    whilelt pn12.b, x10, x3, vlx4

.Lnt_pp_r_p2_K:
    mov     x6, x10
    mov     w12, #0

.Lnt_pp_r_p2_load:
    psel    pn8, pn12, p1.b[w12, #0]
    psel    pn9, pn12, p1.b[w12, #4]
    ldr     x3, [sp, #96]
    ld1b    {z0.b-z3.b}, pn8/z, [x6]
    ld1b    {z4.b-z7.b}, pn9/z, [x6, x3]
    mova    za0h.b[w12, 0:3], {z0.b-z3.b}
    mova    za0h.b[w12, 4:7], {z4.b-z7.b}
    add     w12, w12, #8
    add     x6, x6, x3, lsl #1
    cmp     w12, w17
    b.mi    .Lnt_pp_r_p2_load

    cmp     x13, x2
    b.ge    .Lnt_pp_r_p2_K_done

    ptrue   p2.s                    // store predicate (all lanes active)

    mov     w12, #0
.Lnt_pp_r_p2_za0:
    mova    {z0.s-z3.s}, za0v.s[w12, 0:3]
    add     w12, w12, #4
    st1w    {z0.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z1.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z2.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z3.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    cmp     w12, w5
    b.mi    .Lnt_pp_r_p2_za0

    mov     w12, #0
.Lnt_pp_r_p2_za1:
    mova    {z0.s-z3.s}, za1v.s[w12, 0:3]
    add     w12, w12, #4
    st1w    {z0.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z1.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z2.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z3.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    cmp     w12, w5
    b.mi    .Lnt_pp_r_p2_za1

    mov     w12, #0
.Lnt_pp_r_p2_za2:
    mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
    add     w12, w12, #4
    st1w    {z0.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z1.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z2.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z3.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    cmp     w12, w5
    b.mi    .Lnt_pp_r_p2_za2

    mov     w12, #0
.Lnt_pp_r_p2_za3:
    mova    {z0.s-z3.s}, za3v.s[w12, 0:3]
    add     w12, w12, #4
    st1w    {z0.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z1.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z2.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    st1w    {z3.s}, p2, [x7, x13, lsl #2]
    add     x13, x13, x5, lsl #1
    cmp     w12, w5
    b.mi    .Lnt_pp_r_p2_za3

.Lnt_pp_r_p2_K_done:
    addvl   x10, x10, #4
    ldr     x3, [sp, #120]
    whilelt pn12.b, x10, x3, vlx4
    b.first .Lnt_pp_r_p2_K

    // ── Advance to next N-tile ─────────────────────────────────────
    add     x8, x8, x11             // B base += 2*SVLs*K*4
    // B_mod base += 2*SVLs*K_mod*4 = SVLs*K_mod*8 bytes
    mul     x14, x5, x27            // SVLs * K_mod (elements)
    lsl     x14, x14, #3            // * 8 bytes = 2 * SVLs * K_mod * 4 bytes
    add     x9, x9, x14

    add     x28, x28, x5, lsl #1   // x28 += 2*SVLs (both halves processed)
    whilelt p0.s, x28, x25
    b.first .Lnt_pp_r_N

    // ================================================================
    // Phase 2: preprocess_l — ZA transposition of A (M×K)
    // ================================================================
    // Identical to NN kernel Phase 2. Registers:
    //   x5=SVLs, x3=K*4 (saved [sp,#96]), x11=SVLs*K*4, x15=SVLs*K_mod,
    //   x2=SVLs*K_mod*4, x4=SVLs^2, x16=3*SVLs^2, x17=SVLb.
    // Exit pointer fix: saved to [sp,#120] (not aliased with stride in x3).

    cntw    x5                      // SVLs
    lsl     x3, x26, #2             // K*4 (A row stride)
    str     x3, [sp, #96]
    mul     x11, x5, x3             // SVLs * K * 4
    mul     x15, x5, x27            // SVLs * K_mod
    lsl     x2, x15, #2             // SVLs * K_mod * 4

    mul     x4, x5, x5              // SVLs^2
    lsl     x16, x4, #1
    add     x16, x16, x4            // 3 * SVLs^2
    cntb    x17                     // SVLb

    mov     x28, #0
    mov     x8, x21                 // A base
    mov     x9, x19                 // A_mod base
    whilelt p0.s, x28, x24

.Lnt_pp_l_M:
    mov     x7, x9
    mov     x10, x8
    ldr     x3, [sp, #96]           // K*4
    add     x3, x8, x3              // exit = A_base + K*4
    str     x3, [sp, #120]          // save exit pointer
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
    ldr     x3, [sp, #96]           // reload stride
    ld1b    {z0.b-z3.b}, pn8/z, [x6]
    ld1b    {z4.b-z7.b}, pn9/z, [x6, x3]
    mova    za0h.b[w12, 0:3], {z0.b-z3.b}
    mova    za0h.b[w12, 4:7], {z4.b-z7.b}
    add     w12, w12, #8
    add     x6, x6, x3, lsl #1
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
    ldr     x3, [sp, #120]          // reload saved exit pointer
    whilelt pn12.b, x10, x3, vlx4
    b.first .Lnt_pp_l_K

    add     x8, x8, x11
    add     x9, x9, x2

    incw    x28
    whilelt p0.s, x28, x24
    b.first .Lnt_pp_l_M

    // ================================================================
    // Phase 3: matmul_opt — FMOPA outer products (identical to NN kernel)
    // ================================================================

    cntb    x6                      // SVLb
    cntw    x15                     // SVLs
    lsl     x11, x25, #2            // 4*N
    mul     x21, x15, x25           // SVLs*N  (reuse x21 — A ptr not needed here)
    add     x2, x21, x25            // (SVLs+1)*N
    mul     x7, x27, x6             // K_mod * SVLb  (A_mod M-tile stride)
    lsl     x0, x24, #2             // 4*M
    mul     x3, x27, x6             // K_mod * SVLb  (B_mod N-tile stride base; *2 applied at N-advance)
    str     x3, [sp, #104]
    str     x0, [sp, #112]
    mov     x4, x19                 // a_base = A_mod
    mov     x28, x23                // C base
    mov     x12, #0
    mov     x16, #0
    mov     w15, #0                 // psel index (must be w12-w15)
    sub     w6, w6, #8              // SVLb-8
    ptrue   pn10.b
    whilelt p2.b, x12, x0

.Lnt_mm_M:
    addvl   x12, x12, #1
    whilelt p3.b, x12, x0

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
    addvl   x14, x17, #-1

.Lnt_f_K_start:
    ld1b    {z1.b}, p2/z, [x8]
    whilelt pn10.b, x8, x17, vlx2
    ld1b    {z2.b-z3.b}, pn9/z, [x9]
    fmopa   za0.s, p2/m, p0/m, z1.s, z2.s
    ld1b    {z5.b}, p3/z, [x8, x7]
    addvl   x8, x8, #1

.Lnt_f_Loop_K:
    fmopa   za2.s, p3/m, p0/m, z5.s, z2.s
    fmopa   za1.s, p2/m, p1/m, z1.s, z3.s
    psel    pn11, pn10, p3.s[w15, #0]
    ld1b    {z0.b-z1.b}, pn10/z, [x8]
    fmopa   za3.s, p3/m, p1/m, z5.s, z3.s
    ld1b    {z6.b-z7.b}, pn9/z, [x9, #2, mul vl]

    fmopa   za0.s, p2/m, p0/m, z0.s, z6.s
    ld1b    {z4.b-z5.b}, pn11/z, [x8, x7]

    fmopa   za2.s, p3/m, p0/m, z4.s, z6.s
    addvl   x9, x9, #4

    fmopa   za1.s, p2/m, p1/m, z0.s, z7.s

    fmopa   za3.s, p3/m, p1/m, z4.s, z7.s
    ld1b    {z2.b-z3.b}, pn9/z, [x9]

    fmopa   za0.s, p2/m, p0/m, z1.s, z2.s
    addvl   x8, x8, #2

    cmp     x8, x14
    b.mi    .Lnt_f_Loop_K

    fmopa   za2.s, p3/m, p0/m, z5.s, z2.s
    fmopa   za1.s, p2/m, p1/m, z1.s, z3.s
    fmopa   za3.s, p3/m, p1/m, z5.s, z3.s
    addvl   x9, x9, #2

    cmp     x8, x10
    b.ge    .Lnt_mm_store

.Lnt_f_Ktail:
    ld1b    {z1.b}, p2/z, [x8]
    ld1b    {z2.b-z3.b}, pn9/z, [x9]
    fmopa   za0.s, p2/m, p0/m, z1.s, z2.s
    ld1b    {z14.b}, p3/z, [x8, x7]
    fmopa   za2.s, p3/m, p0/m, z14.s, z2.s
    fmopa   za1.s, p2/m, p1/m, z1.s, z3.s
    addvl   x9, x9, #2
    fmopa   za3.s, p3/m, p1/m, z14.s, z3.s

.Lnt_mm_store:
    mov     w14, #0
    psel    pn8, pn9, p2.b[w14, 0]
    psel    pn11, pn9, p3.b[w14, 0]
    mova    {z0.b-z3.b}, za0h.b[w14, 0:3]
    st1w    {z0.s-z1.s}, pn8, [x23]
    st1w    {z2.s-z3.s}, pn11, [x23, x21, lsl #2]

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
    st1w    {z2.s-z3.s}, pn11, [x23, x21, lsl #2]
    cmp     w14, w6
    b.mi    .Lnt_mm_store_loop

    psel    pn8, pn9, p2.b[w14, 4]
    psel    pn11, pn9, p3.b[w14, 4]
    mova    {z0.b-z3.b}, za0h.b[w14, 4:7]
    st1w    {z0.s-z1.s}, pn8, [x23, x25, lsl #2]
    st1w    {z2.s-z3.s}, pn11, [x23, x2, lsl #2]

    // Next N tile: B_mod += K_mod * 2*SVLb bytes (FIXED: lsl #1)
    addvl   x22, x22, #2
    addvl   x13, x13, #2
    ldr     x3, [sp, #104]          // K_mod * SVLb
    add     x5, x5, x3, lsl #1     // B_mod += K_mod * 2*SVLb
    whilelt pn9.b, x13, x11, vlx2
    b.first .Lnt_mm_N

    // Next M tile
    add     x4, x4, x7, lsl #1     // A_mod += K_mod * SVLb * 2
    add     x28, x28, x21, lsl #3  // C += SVLs * N * 4 * 2
    addvl   x12, x12, #1
    ldr     x3, [sp, #112]          // reload 4*M
    whilelt p2.b, x12, x3
    b.first .Lnt_mm_M

    smstop

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
    ldp     d10, d11, [sp, #160]
    ldp     d12, d13, [sp, #176]
    ldp     d14, d15, [sp, #192]
    ldp     x29, x30, [sp], #224
    ret
