// matmul_fp32.s — FP32 matmul via SME2, following ARM SME Programmer's Guide §6
//
// void matmul_fp32(const float *A, const float *B, float *C,
//                  long M, long N, long K)
//
// A is M×K row-major, B is K×N row-major, C is M×N row-major.
// Scratch buffers for preprocessed A and B are allocated internally via
// malloc and freed before return. No caller-provided workspace required.
//
// Single function, three phases:
//   Phase 1: preprocess_r (§6.2) — rearrange B → B_mod
//   Phase 2: preprocess_l (§6.5) — rearrange A → A_mod
//   Phase 3: matmul_opt  (§6.8) — multiply A_mod × B_mod → C
//
// Key FP32 adaptation: FMOPA has rank 1 (vs rank 4 for SMOPA), but the ZA
// byte-level transposition in Phase 2 processes 4*SVLb bytes per K step.
// For fp32 (4 bytes/element), each step covers SVLb float elements. To avoid
// dropping real data from the ZA transposition, K is rounded up:
//   K_mod = ceil(K * 4 / (4*SVLb)) * SVLb = ceil(K / SVLb) * SVLb
// where SVLb = cntb (64 on M4). Buffers are zeroed so padding is harmless.
//
// Persistent registers (callee-saved):
//   x19 = A_mod ptr    x20 = B_mod ptr
//   x21 = A            x22 = B (free after preprocessing)
//   x23 = C            x24 = M    x25 = N    x26 = K
//   x27 = K_mod        x28 = A_mod_size (reused as scratch in Phase 3)
//
// Stack frame (176 bytes):
//   [sp, #0]:   x29, x30
//   [sp, #16]:  x19, x20
//   [sp, #32]:  x21, x22
//   [sp, #48]:  x23, x24
//   [sp, #64]:  x25, x26
//   [sp, #80]:  x27, x28
//   [sp, #96]:  b_mod_size, (spare)
//   [sp, #112]: d8, d9
//   [sp, #128]: d10, d11
//   [sp, #144]: d12, d13
//   [sp, #160]: d14, d15

.section __TEXT,__text,regular,pure_instructions
.global _matmul_fp32
.p2align 4

_matmul_fp32:
    stp     x29, x30, [sp, #-176]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8,  d9,  [sp, #112]
    stp     d10, d11, [sp, #128]
    stp     d12, d13, [sp, #144]
    stp     d14, d15, [sp, #160]

    // Save arguments to callee-saved registers
    mov     x21, x0                 // A
    mov     x22, x1                 // B
    mov     x23, x2                 // C
    mov     x24, x3                 // M
    mov     x25, x4                 // N
    mov     x26, x5                 // K

    // ── Query streaming SVL ─────────────────────────────────
    smstart sm
    cntw    x15                     // SVLs (streaming)
    cntb    x17                     // SVLb (streaming)
    smstop  sm

    // ── Compute K_mod ─────────────────────────────────────────
    // K_mod = ceil(K / SVLb) * SVLb  (round K up to SVLb-element boundary)
    // This ensures Phase 2's ZA transposition can store all output without
    // truncation. SVLb = cntb (64 on M4). For K=100: K_mod=128.
    add     x27, x26, x17
    sub     x27, x27, #1
    udiv    x27, x27, x17
    mul     x27, x27, x17          // K_mod (callee-saved in x27)

    // ── Compute buffer sizes ────────────────────────────────
    // A_mod = ceil(M/SVLs)*SVLs * K_mod * 4  (bytes)
    add     x8, x24, x15
    sub     x8, x8, #1
    udiv    x8, x8, x15
    mul     x8, x8, x15            // M_pad
    mul     x28, x8, x27           // M_pad * K_mod (elements)
    lsl     x28, x28, #2           // * 4 bytes

    // B_mod = ceil(N/(2*SVLs))*(2*SVLs) * K_mod * 4  (bytes)
    lsl     x9, x15, #1            // 2*SVLs
    add     x8, x25, x9
    sub     x8, x8, #1
    udiv    x8, x8, x9
    mul     x8, x8, x9             // N_pad (rounded to 2*SVLs)
    mul     x8, x8, x27            // N_pad * K_mod (elements)
    lsl     x8, x8, #2             // * 4 bytes
    str     x8, [sp, #96]          // save B_mod size to stack

    // ── Allocate A_mod ──────────────────────────────────────
    mov     x0, x28
    bl      _malloc
    mov     x19, x0

    // Zero A_mod (padding bytes must be zero for correct ZA transposition)
    mov     x0, x19
    mov     x1, x28
    bl      _bzero

    // ── Allocate B_mod ──────────────────────────────────────
    ldr     x0, [sp, #96]          // B_mod size
    bl      _malloc
    mov     x20, x0

    // Zero B_mod (padding rows beyond K must be zero)
    mov     x0, x20
    ldr     x1, [sp, #96]
    bl      _bzero

    // ── Enter streaming mode + ZA ───────────────────────────
    smstart

    // ================================================================
    // Phase 1: preprocess_r  (ARM SME Guide §6.2)
    // ================================================================
    // FP32: no zip interleaving (rank-1). Simple blocked copy of B into
    // N-tile-contiguous layout. K_mod rows allocated, only K written;
    // the rest stay zero from bzero above.

    cntb    x5                      // SVLb
    lsl     x11, x25, #2           // N * 4 (byte stride between B rows)
    // B_mod stride per N-tile: K_mod * 2*SVLb bytes
    lsl     x17, x5, #1            // 2*SVLb
    mul     x16, x27, x17          // K_mod * 2*SVLb

    mov     x15, #0                // psel variable

    ptrue   pn9.b

    mov     x8, x22                // b_base = B (byte ptr)
    mov     x9, x20                // b_mod_base = B_mod
    lsl     x10, x25, #2           // N*4 (byte extent of one B row)
    add     x10, x10, x22          // exit condition: b_base + N*4
    whilelt pn11.b, x8, x10, vlx2

.Lpp_r_N:
    mov     x7, x8                 // b_ptr = current N-tile in B
    mov     x12, x9                // b_mod_ptr
    mov     x6, xzr                // K counter

.Lpp_r_K:
    cmp     x6, x26                // compare against real K (not K_mod)
    b.ge    .Lpp_r_N_next

    ld1b    {z0.b-z1.b}, pn11/z, [x7]
    st1b    {z0.b-z1.b}, pn11, [x12]

    add     x7, x7, x11            // b_ptr += N*4 (next row of B)
    addvl   x12, x12, #2           // b_mod_ptr += 2*SVLb
    add     x6, x6, #1
    b       .Lpp_r_K

.Lpp_r_N_next:
    add     x9, x9, x16            // b_mod_base += K_mod * 2*SVLb
    addvl   x8, x8, #2             // b_base += 2*SVLb (next N-tile)
    whilelt pn11.b, x8, x10, vlx2
    b.first .Lpp_r_N

    // ================================================================
    // Phase 2: preprocess_l  (ARM SME Guide §6.5)
    // ================================================================
    // ZA byte transposition — identical to int8. Key fp32 adaptations:
    //   - Row stride x18 = K*4 (not K)
    //   - x15 = SVLs * K_mod (not SVLs * ceil(K/4))
    //   - Exit condition: A_base + K*4 (real data only; padding stays zero)

    cntw    x5                      // SVLs
    lsl     x18, x26, #2           // K*4 (A row stride in bytes)
    mul     x11, x5, x18           // SVLs * K * 4 (M-tile stride in original A)
    mul     x15, x5, x27           // SVLs * K_mod (.s element count per M-tile)
    lsl     x2, x15, #2            // SVLs * K_mod * 4 (A_mod byte stride per M-tile)

    mul     x4, x5, x5             // SVLs^2
    lsl     x16, x4, #1
    add     x16, x16, x4           // 3 * SVLs^2
    cntb    x17                     // SVLb

    mov     x28, #0                // Loop_M counter
    mov     x8, x21                // matLeft base (working)
    mov     x9, x19                // matLeft_mod base (working)
    whilelt p0.s, x28, x24

.Lpp_l_M:
    mov     x7, x9
    mov     x10, x8
    add     x3, x8, x18            // exit = A_base + K*4 (real data extent)
    whilelt pn12.b, x10, x3, vlx4
    mov     x13, #0
    mov     x14, x4
    lsl     x0, x4, #1
    mov     x1, x16

.Lpp_l_K:
    mov     x6, x10

    mov     w12, #0
.Lpp_l_load:
    psel    pn8, pn12, p0.b[w12, #0]
    psel    pn9, pn12, p0.b[w12, #4]
    ld1b    {z0.b-z3.b}, pn8/z, [x6]
    ld1b    {z4.b-z7.b}, pn9/z, [x6, x18]
    mova    za0h.b[w12, 0:3], {z0.b-z3.b}
    mova    za0h.b[w12, 4:7], {z4.b-z7.b}
    add     w12, w12, #8
    add     x6, x6, x18, lsl #1
    cmp     w12, w17
    b.mi    .Lpp_l_load

    mov     w12, #0
.Lpp_l_store:
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
    b.mi    .Lpp_l_store

    add     x13, x13, x16
    add     x14, x14, x16
    add     x0, x0, x16
    add     x1, x1, x16
    addvl   x10, x10, #4
    whilelt pn12.b, x10, x3, vlx4
    b.first .Lpp_l_K

    add     x8, x8, x11
    add     x9, x9, x2

    incw    x28
    whilelt p0.s, x28, x24
    b.first .Lpp_l_M

    // ================================================================
    // Phase 3: matmul_opt  (ARM SME Guide §6.8)
    // ================================================================
    // x7 = K_mod * SVLb (A_mod M-tile stride, accounts for ZA padding).
    // Extra K iterations beyond real K process zeros → no contribution.

    cntb    x6                      // SVLb
    cntw    x15                     // SVLs
    lsl     x11, x25, #2           // 4*N
    mul     x21, x15, x25          // SVLs*N
    add     x2, x21, x25           // (SVLs+1)*N
    mul     x7, x27, x6            // K_mod * SVLb
    lsl     x0, x24, #2            // 4*M
    mov     x3, x19                // a_base = A_mod
    mov     x28, x23               // matResult
    mov     x12, #0                // Loop_M counter
    mov     x15, #0                // psel variable
    sub     w6, w6, #8             // SVLb-8
    ptrue   pn10.b
    whilelt p2.b, x12, x0

.Lmm_M:
    addvl   x12, x12, #1
    whilelt p3.b, x12, x0

    mov     x4, x20                // b_base = B_mod
    mov     x22, x28               // c_base
    mov     x13, #0
    add     x10, x3, x7
    add     x17, x3, x7
    addvl   x9, x17, #-1

    whilelt pn9.b, x13, x11, vlx2

.Lmm_N:
    mov     x8, x3
    mov     x16, x4
    mov     x23, x22

    pext    {p0.b, p1.b}, pn9[0]

    zero    {za}

.Lf_K_start:
    ld1b    {z1.b}, p2/z, [x8]
    whilelt pn10.b, x8, x17, vlx2
    ld1b    {z2.b-z3.b}, pn9/z, [x16]
    fmopa   za0.s, p2/m, p0/m, z1.s, z2.s
    ld1b    {z5.b}, p3/z, [x8, x7]
    addvl   x8, x8, #1

.Lf_Loop_K:
    fmopa   za2.s, p3/m, p0/m, z5.s, z2.s
    fmopa   za1.s, p2/m, p1/m, z1.s, z3.s
    psel    pn11, pn10, p3.s[w15, #0]
    ld1b    {z0.b-z1.b}, pn10/z, [x8]
    fmopa   za3.s, p3/m, p1/m, z5.s, z3.s
    ld1b    {z6.b-z7.b}, pn9/z, [x16, #2, mul vl]

    fmopa   za0.s, p2/m, p0/m, z0.s, z6.s
    ld1b    {z4.b-z5.b}, pn11/z, [x8, x7]

    fmopa   za2.s, p3/m, p0/m, z4.s, z6.s
    addvl   x16, x16, #4

    fmopa   za1.s, p2/m, p1/m, z0.s, z7.s

    fmopa   za3.s, p3/m, p1/m, z4.s, z7.s
    ld1b    {z2.b-z3.b}, pn9/z, [x16]

    fmopa   za0.s, p2/m, p0/m, z1.s, z2.s
    addvl   x8, x8, #2

    cmp     x8, x9
    b.mi    .Lf_Loop_K

    fmopa   za2.s, p3/m, p0/m, z5.s, z2.s
    fmopa   za1.s, p2/m, p1/m, z1.s, z3.s
    fmopa   za3.s, p3/m, p1/m, z5.s, z3.s
    addvl   x16, x16, #2

    cmp     x8, x10
    b.ge    .Lmm_store

.Lf_Ktail:
    ld1b    {z1.b}, p2/z, [x8]
    ld1b    {z2.b-z3.b}, pn9/z, [x16]
    fmopa   za0.s, p2/m, p0/m, z1.s, z2.s
    ld1b    {z14.b}, p3/z, [x8, x7]
    fmopa   za2.s, p3/m, p0/m, z14.s, z2.s
    fmopa   za1.s, p2/m, p1/m, z1.s, z3.s
    addvl   x16, x16, #2
    fmopa   za3.s, p3/m, p1/m, z14.s, z3.s

    // ── Store results ──────────────────────────────────────
.Lmm_store:
    mov     w14, #0
    psel    pn8, pn9, p2.b[w14, 0]
    psel    pn11, pn9, p3.b[w14, 0]
    mova    {z0.b-z3.b}, za0h.b[w14, 0:3]
    st1w    {z0.s-z1.s}, pn8, [x23]
    st1w    {z2.s-z3.s}, pn11, [x23, x21, lsl #2]

.Lmm_store_loop:
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
    b.mi    .Lmm_store_loop

    psel    pn8, pn9, p2.b[w14, 4]
    psel    pn11, pn9, p3.b[w14, 4]
    mova    {z0.b-z3.b}, za0h.b[w14, 4:7]
    st1w    {z0.s-z1.s}, pn8, [x23, x25, lsl #2]
    st1w    {z2.s-z3.s}, pn11, [x23, x2, lsl #2]

    addvl   x22, x22, #2
    addvl   x13, x13, #2
    whilelt pn9.b, x13, x11, vlx2
    add     x4, x4, x7, lsl #1
    b.first .Lmm_N

    add     x3, x3, x7, lsl #1
    add     x28, x28, x21, lsl #3
    addvl   x12, x12, #1
    whilelt p2.b, x12, x0
    b.first .Lmm_M

    // ── Exit streaming mode ─────────────────────────────────
    smstop

    // ── Free scratch buffers ────────────────────────────────
    mov     x0, x19
    bl      _free
    mov     x0, x20
    bl      _free

    // ── Restore and return ──────────────────────────────────
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     d8,  d9,  [sp, #112]
    ldp     d10, d11, [sp, #128]
    ldp     d12, d13, [sp, #144]
    ldp     d14, d15, [sp, #160]
    ldp     x29, x30, [sp], #176
    ret
