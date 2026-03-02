// matmul_int8.s — INT8 matmul via SME2, following ARM SME Programmer's Guide §6
//
// Two entry points:
//   void matmul_int8(const int8_t *A, const int8_t *B, int32_t *C,
//                    long M, long N, long K, void *a_work, void *b_work)
//   void matmul_uint8(const uint8_t *A, const uint8_t *B, uint32_t *C,
//                     long M, long N, long K, void *a_work, void *b_work)
//
// a_work / b_work: optional scratch buffers for preprocessed A and B.
//   If NULL, malloc'd internally (convenient but slower for repeated calls).
//   If non-NULL, caller provides pre-allocated workspace (zero allocation overhead).
//
// Single function, three phases:
//   Phase 1: preprocess_r (§6.2) — rearrange B → B_mod
//   Phase 2: preprocess_l (§6.5) — rearrange A → A_mod
//   Phase 3: matmul_opt  (§6.8) — multiply A_mod × B_mod → C
//
// Persistent registers (callee-saved):
//   x19 = A_mod ptr    x20 = B_mod ptr
//   x21 = A            x22 = B (free after preprocessing)
//   x23 = C            x24 = M    x25 = N    x26 = K
//   x27 = signed/unsigned flag (0=signed, 1=unsigned)
//   x28 = scratch (callee-saved, survives malloc)
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
// Signed entry point
// ============================================================
.global _matmul_int8
.p2align 4
_matmul_int8:
    mov     x8, #0                 // flag = signed
    b       .Lcommon

// ============================================================
// Unsigned entry point
// ============================================================
.global _matmul_uint8
.p2align 4
_matmul_uint8:
    mov     x8, #1                 // flag = unsigned
    b       .Lcommon

// ============================================================
// Common implementation
// ============================================================
.Lcommon:
    stp     x29, x30, [sp, #-224]!
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
    mov     x27, x8                 // flag
    stp     x6, x7, [sp, #96]      // save workspace ptrs (clobbered by malloc)

    // ── Query streaming SVL ─────────────────────────────────
    smstart sm
    cntw    x15                     // SVLs (streaming)
    cntb    x17                     // SVLb (streaming)
    smstop  sm

    // ── Compute buffer sizes ────────────────────────────────
    // K_mod = ceil(K/4)*4
    add     x9, x26, #3
    and     x9, x9, #~3

    // A_mod = ceil(M/SVLs)*SVLs * K_mod
    add     x8, x24, x15
    sub     x8, x8, #1
    udiv    x8, x8, x15
    mul     x8, x8, x15            // M_pad
    mul     x28, x8, x9            // A_mod size (x28 callee-saved, survives malloc)

    // B_mod = ceil(N/SVLb)*SVLb * K_mod
    add     x8, x25, x17
    sub     x8, x8, #1
    udiv    x8, x8, x17
    mul     x8, x8, x17            // N_pad
    mul     x8, x8, x9             // B_mod size
    str     x8, [sp, #128]         // save B_mod size to stack
    stp     d8,  d9,  [sp, #144]
    stp     d10,  d11,  [sp, #160]
    stp     d12,  d13,  [sp, #176]
    stp     d14,  d15,  [sp, #192]

    // ── Allocate A_mod (from workspace or malloc) ───────────
    ldr     x6, [sp, #96]          // a_work
    cbnz    x6, .Luse_a_work
    mov     x0, x28
    bl      _malloc
    mov     x19, x0
    mov     x8, #1
    str     x8, [sp, #112]         // need_free_a = 1
    b       .La_done
.Luse_a_work:
    mov     x19, x6
    str     xzr, [sp, #112]        // need_free_a = 0
.La_done:

    // ── Allocate B_mod (from workspace or malloc) ───────────
    ldr     x7, [sp, #104]         // b_work
    cbnz    x7, .Luse_b_work
    ldr     x0, [sp, #128]         // B_mod size
    bl      _malloc
    mov     x20, x0
    mov     x8, #1
    str     x8, [sp, #120]         // need_free_b = 1
    b       .Lb_done
.Luse_b_work:
    mov     x20, x7
    str     xzr, [sp, #120]        // need_free_b = 0
.Lb_done:

    // ── Enter streaming mode + ZA ───────────────────────────
    smstart

    // ================================================================
    // Phase 1: preprocess_r  (ARM SME Guide §6.2)
    // ================================================================

    cntb    x5                      // SVLb
    lsl     x16, x25, #1           // 2*N
    add     x3, x16, x25           // 3*N
    add     x4, x26, #3
    lsr     x4, x4, #2             // ceil(K/4)
    mul     x11, x4, x5            // ceil(K/4)*SVLb
    lsl     x17, x11, #1           // 2*ceil(K/4)*SVLb
    mov     x15, #0                // psel variable
    cnth    x13                     // SVLb/2

    ptrue   pn9.b

    mov     x8, x22                // b_base = B
    mov     x9, x20                // b_mod_base = B_mod
    add     x10, x22, x25          // N dimension exit condition
    whilelt p2.b, x8, x10          // N dimension predicate

.Lpp_r_N:
    mov     x7, x8                 // b_ptr
    mov     x12, x9                // b_mod_ptr
    whilelt p1.b, xzr, x26         // K dimension predicate

    psel    pn11, pn9, p2.b[w15, 0]
    psel    pn12, pn9, p2.b[w13, 0]

    mov     x6, xzr                // Loop_K counter
.Lpp_r_K:
    psel    p0, p2, p1.b[w15, 0]
    psel    p3, p2, p1.b[w15, 1]
    ld1b    {z0.b}, p0/z, [x7]
    ld1b    {z1.b}, p3/z, [x7, x25]

    psel    p0, p2, p1.b[w15, 2]
    psel    p3, p2, p1.b[w15, 3]
    ld1b    {z2.b}, p0/z, [x7, x16]
    ld1b    {z3.b}, p3/z, [x7, x3]

    zip     {z8.b-z11.b}, {z0.b-z3.b}

    st1b    {z8.b-z9.b}, pn11, [x12]
    st1b    {z10.b-z11.b}, pn12, [x12, x17]

    add     x7, x7, x25, lsl #2    // b_ptr += 4*N
    addvl   x12, x12, #2           // b_mod_ptr += 2*SVLb
    add     x6, x6, #4
    whilelt p1.b, x6, x26
    b.first .Lpp_r_K

    add     x9, x9, x17, lsl #1    // b_mod_base += 4*ceil(K/4)*SVLb
    addvl   x8, x8, #1             // b_base += SVLb
    whilelt p2.b, x8, x10
    b.first .Lpp_r_N

    // ================================================================
    // Phase 2: preprocess_l  (ARM SME Guide §6.5)
    // ================================================================

    cntw    x5                      // SVLs
    mul     x11, x5, x26           // SVLs*K
    add     x2, x26, #3
    lsr     x2, x2, #2             // ceil(K/4)
    mul     x15, x2, x5            // SVLs*ceil(K/4)
    lsl     x2, x15, #2            // SVLs*ceil(K/4)*4

    mul     x4, x5, x5             // SVLs*SVLs
    lsl     x16, x4, #1
    add     x16, x16, x4           // 3*SVLs*SVLs
    cntb    x17                     // SVLb

    mov     x28, #0                // Loop_M counter
    mov     x8, x21                // matLeft base (working)
    mov     x9, x19                // matLeft_mod base (working)
    whilelt p0.s, x28, x24

.Lpp_l_M:
    mov     x7, x9
    mov     x10, x8
    add     x3, x8, x26
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
    ld1b    {z4.b-z7.b}, pn9/z, [x6, x26]
    mova    za0h.b[w12, 0:3], {z0.b-z3.b}
    mova    za0h.b[w12, 4:7], {z4.b-z7.b}
    add     w12, w12, #8
    add     x6, x6, x26, lsl #1
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

    cntb    x6                      // SVLb
    cntw    x15                     // SVLs
    lsl     x11, x25, #2           // 4*N
    mul     x21, x15, x25          // SVLs*ldc
    add     x2, x21, x25           // (SVLs+1)*ldc
    add     x7, x26, #3
    lsr     x7, x7, #2
    mul     x7, x7, x6             // ceil(K/4)*SVLb
    lsl     x0, x24, #2            // 4*M
    mov     x3, x19                // a_base = A_mod
    mov     x28, x23               // matResult (advances per M)
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

    cbnz    x27, .Lu_K_start

    // ── Signed (smopa) ─────────────────────────────────────
.Ls_K_start:
    ld1b    {z1.b}, p2/z, [x8]
    whilelt pn10.b, x8, x17, vlx2
    ld1b    {z2.b-z3.b}, pn9/z, [x16]
    smopa   za0.s, p2/m, p0/m, z1.b, z2.b
    ld1b    {z5.b}, p3/z, [x8, x7]
    addvl   x8, x8, #1

.Ls_Loop_K:
    smopa   za2.s, p3/m, p0/m, z5.b, z2.b
    smopa   za1.s, p2/m, p1/m, z1.b, z3.b
    psel    pn11, pn10, p3.s[w15, #0]
    ld1b    {z0.b-z1.b}, pn10/z, [x8]
    smopa   za3.s, p3/m, p1/m, z5.b, z3.b
    ld1b    {z6.b-z7.b}, pn9/z, [x16, #2, mul vl]

    smopa   za0.s, p2/m, p0/m, z0.b, z6.b
    ld1b    {z4.b-z5.b}, pn11/z, [x8, x7]

    smopa   za2.s, p3/m, p0/m, z4.b, z6.b
    addvl   x16, x16, #4

    smopa   za1.s, p2/m, p1/m, z0.b, z7.b

    smopa   za3.s, p3/m, p1/m, z4.b, z7.b
    ld1b    {z2.b-z3.b}, pn9/z, [x16]

    smopa   za0.s, p2/m, p0/m, z1.b, z2.b
    addvl   x8, x8, #2

    cmp     x8, x9
    b.mi    .Ls_Loop_K

    smopa   za2.s, p3/m, p0/m, z5.b, z2.b
    smopa   za1.s, p2/m, p1/m, z1.b, z3.b
    smopa   za3.s, p3/m, p1/m, z5.b, z3.b
    addvl   x16, x16, #2

    cmp     x8, x10
    b.ge    .Lmm_store

.Ls_Ktail:
    ld1b    {z1.b}, p2/z, [x8]
    ld1b    {z2.b-z3.b}, pn9/z, [x16]
    smopa   za0.s, p2/m, p0/m, z1.b, z2.b
    ld1b    {z14.b}, p3/z, [x8, x7]
    smopa   za2.s, p3/m, p0/m, z14.b, z2.b
    smopa   za1.s, p2/m, p1/m, z1.b, z3.b
    addvl   x16, x16, #2
    smopa   za3.s, p3/m, p1/m, z14.b, z3.b
    b       .Lmm_store

    // ── Unsigned (umopa) ───────────────────────────────────
.Lu_K_start:
    ld1b    {z1.b}, p2/z, [x8]
    whilelt pn10.b, x8, x17, vlx2
    ld1b    {z2.b-z3.b}, pn9/z, [x16]
    umopa   za0.s, p2/m, p0/m, z1.b, z2.b
    ld1b    {z5.b}, p3/z, [x8, x7]
    addvl   x8, x8, #1

.Lu_Loop_K:
    umopa   za2.s, p3/m, p0/m, z5.b, z2.b
    umopa   za1.s, p2/m, p1/m, z1.b, z3.b
    psel    pn11, pn10, p3.s[w15, #0]
    ld1b    {z0.b-z1.b}, pn10/z, [x8]
    umopa   za3.s, p3/m, p1/m, z5.b, z3.b
    ld1b    {z6.b-z7.b}, pn9/z, [x16, #2, mul vl]

    umopa   za0.s, p2/m, p0/m, z0.b, z6.b
    ld1b    {z4.b-z5.b}, pn11/z, [x8, x7]

    umopa   za2.s, p3/m, p0/m, z4.b, z6.b
    addvl   x16, x16, #4

    umopa   za1.s, p2/m, p1/m, z0.b, z7.b

    umopa   za3.s, p3/m, p1/m, z4.b, z7.b
    ld1b    {z2.b-z3.b}, pn9/z, [x16]

    umopa   za0.s, p2/m, p0/m, z1.b, z2.b
    addvl   x8, x8, #2

    cmp     x8, x9
    b.mi    .Lu_Loop_K

    umopa   za2.s, p3/m, p0/m, z5.b, z2.b
    umopa   za1.s, p2/m, p1/m, z1.b, z3.b
    umopa   za3.s, p3/m, p1/m, z5.b, z3.b
    addvl   x16, x16, #2

    cmp     x8, x10
    b.ge    .Lmm_store

.Lu_Ktail:
    ld1b    {z1.b}, p2/z, [x8]
    ld1b    {z2.b-z3.b}, pn9/z, [x16]
    umopa   za0.s, p2/m, p0/m, z1.b, z2.b
    ld1b    {z14.b}, p3/z, [x8, x7]
    umopa   za2.s, p3/m, p0/m, z14.b, z2.b
    umopa   za1.s, p2/m, p1/m, z1.b, z3.b
    addvl   x16, x16, #2
    umopa   za3.s, p3/m, p1/m, z14.b, z3.b

    // ── Store results (shared) ──────────────────────────────
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

    // ── Conditionally free buffers ──────────────────────────
    ldr     x8, [sp, #112]         // need_free_a?
    cbz     x8, .Lskip_free_a
    mov     x0, x19
    bl      _free
.Lskip_free_a:
    ldr     x8, [sp, #120]         // need_free_b?
    cbz     x8, .Lskip_free_b
    mov     x0, x20
    bl      _free
.Lskip_free_b:

    // ── Restore and return ──────────────────────────────────
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

