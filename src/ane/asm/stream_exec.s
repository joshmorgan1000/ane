// stream_exec.s — Jump-table bytecode interpreter for CompiledProgram
//
// void stream_exec(const ane::CompiledProgram* program)
// AAPCS: x0 = pointer to CompiledProgram
//
// CompiledProgram layout (C++ std::vector has {ptr, size, cap}):
//   [0]   bytecodes.data()    (ptr)   → x23 (bytecode base), x19 (bytecode IP)
//   [8]   bytecodes.size()
//   [16]  bytecodes.capacity()
//   [24]  operands.data()     (ptr)   → x20 (operand pointer)
//   [32]  operands.size()
//   [40]  operands.capacity()
//   [48]  loop_count           (u32)  → x21 (loop counter)
//   [52]  (padding)
//   [56]  output               (ptr)  → stored as last operand by compiler
//
// Register allocation:
//   x19 = bytecode IP (advances per opcode)
//   x20 = operand pointer (advances per consumed operand)
//   x21 = loop counter (decremented by halt)
//   x22 = scratch / k_steps
//   x23 = bytecode base (reset IP on loop)
//   x24 = operand base (reset operand ptr on loop)
//   x25 = jump table base
//
// Opcodes:
//   0x00 halt         — decrement loop counter, loop or exit
//   0x01 zero_za      — zero {za}
//   0x02 acc_smopa    — fused ld1b+smopa loop, +u32 k_steps, 2 operands
//   0x03 acc_umopa    — fused ld1b+umopa loop
//   0x04 acc_usmopa   — fused ld1b+usmopa loop
//   0x05 acc_sumopa   — fused ld1b+sumopa loop
//   0x06 store_tiles  — store za0-za3 → output, 1 operand
//   0x07 load_rows_i8 — ld1b 128 bytes → z0,z1, 1 operand
//   0x08 load_cols_i8 — ld1b 128 bytes → z2,z3, 1 operand
//   0x09 smopa_2x2    — 4× smopa z0-z3 → za0-za3 (no load)
//   0x0A umopa_2x2    — 4× umopa
//   0x0B usmopa_2x2   — 4× usmopa
//   0x0C load_bias    — load int32 bias into za0-za3 from memory, 1 operand
//   0x0D scale_store  — int32→float, multiply by scale, store. 1 float imm + 1 operand

.section __TEXT,__text,regular,pure_instructions
.global _stream_exec
.p2align 4

_stream_exec:
    // ── Prologue ──
    stp     x29, x30, [sp, #-128]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     d8,  d9,  [sp, #80]
    stp     d10, d11, [sp, #96]
    stp     d12, d13, [sp, #112]

    // ── Load CompiledProgram fields ──
    ldr     x23, [x0, #0]          // bytecodes.data() → bytecode base
    ldr     x24, [x0, #24]         // operands.data()  → operand base
    ldr     w21, [x0, #48]         // loop_count

    mov     x19, x23               // bytecode IP = base
    mov     x20, x24               // operand ptr = base

    // ── Enter streaming mode ──
    smstart

    // ── Load jump table base (must be after smstart, adrp is fine in streaming) ──
    adrp    x25, .Ljump_table@PAGE
    add     x25, x25, .Ljump_table@PAGEOFF

    // ── Dispatch ──
.Ldispatch:
    ldrb    w9, [x19], #1          // fetch opcode, advance IP
    cmp     w9, #14                // NUM_OPCODES
    b.hs    .Lop_halt              // out-of-range → halt
    ldrsw   x10, [x25, x9, lsl #2] // load relative offset (32-bit signed)
    add     x10, x25, x10          // absolute target = table_base + offset
    br      x10

// ================================================================
// Jump Table (PC-relative offsets, avoids text relocations on macOS)
// ================================================================
.p2align 2
.Ljump_table:
    .long   .Lop_halt       - .Ljump_table  // 0x00
    .long   .Lop_zero_za    - .Ljump_table  // 0x01
    .long   .Lop_acc_smopa  - .Ljump_table  // 0x02
    .long   .Lop_acc_umopa  - .Ljump_table  // 0x03
    .long   .Lop_acc_usmopa - .Ljump_table  // 0x04
    .long   .Lop_acc_sumopa - .Ljump_table  // 0x05
    .long   .Lop_store_tiles - .Ljump_table // 0x06
    .long   .Lop_load_rows_i8 - .Ljump_table // 0x07
    .long   .Lop_load_cols_i8 - .Ljump_table // 0x08
    .long   .Lop_smopa_2x2  - .Ljump_table // 0x09
    .long   .Lop_umopa_2x2  - .Ljump_table // 0x0A
    .long   .Lop_usmopa_2x2 - .Ljump_table // 0x0B
    .long   .Lop_load_bias  - .Ljump_table // 0x0C
    .long   .Lop_scale_store - .Ljump_table // 0x0D

// ================================================================
// HALT (0x00) — decrement loop counter, loop or exit
// ================================================================
.Lop_halt:
    sub     w21, w21, #1
    cbz     w21, .Lexit
    // Reset bytecode IP, operand pointer keeps advancing
    mov     x19, x23
    b       .Ldispatch

.Lexit:
    smstop
    ldp     d12, d13, [sp, #112]
    ldp     d10, d11, [sp, #96]
    ldp     d8,  d9,  [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #128
    ret

// ================================================================
// ZERO_ZA (0x01)
// ================================================================
.Lop_zero_za:
    zero    {za}
    b       .Ldispatch

// ================================================================
// ACC_SMOPA (0x02) — fused ld1b+smopa loop
// Encoding: [0x02] [k_steps: 4 bytes LE]
// Operands: row_ptr, col_ptr
// ================================================================
.Lop_acc_smopa:
    ldr     w22, [x19]             // k_steps (4 bytes LE)
    add     x19, x19, #4           // advance IP past immediate
    ldr     x8, [x20], #8          // consume row_ptr
    ldr     x11, [x20], #8         // consume col_ptr
    ptrue   p0.b
    cbz     w22, .Ldispatch
.Lacc_smopa_loop:
    ld1b    {z0.b}, p0/z, [x8]
    ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x11]
    ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
    smopa   za0.s, p0/m, p0/m, z0.b, z2.b
    smopa   za1.s, p0/m, p0/m, z0.b, z3.b
    smopa   za2.s, p0/m, p0/m, z1.b, z2.b
    smopa   za3.s, p0/m, p0/m, z1.b, z3.b
    addvl   x8, x8, #2
    addvl   x11, x11, #2
    sub     w22, w22, #1
    cbnz    w22, .Lacc_smopa_loop
    b       .Ldispatch

// ================================================================
// ACC_UMOPA (0x03) — fused ld1b+umopa loop
// ================================================================
.Lop_acc_umopa:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x20], #8
    ldr     x11, [x20], #8
    ptrue   p0.b
    cbz     w22, .Ldispatch
.Lacc_umopa_loop:
    ld1b    {z0.b}, p0/z, [x8]
    ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x11]
    ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
    umopa   za0.s, p0/m, p0/m, z0.b, z2.b
    umopa   za1.s, p0/m, p0/m, z0.b, z3.b
    umopa   za2.s, p0/m, p0/m, z1.b, z2.b
    umopa   za3.s, p0/m, p0/m, z1.b, z3.b
    addvl   x8, x8, #2
    addvl   x11, x11, #2
    sub     w22, w22, #1
    cbnz    w22, .Lacc_umopa_loop
    b       .Ldispatch

// ================================================================
// ACC_USMOPA (0x04) — fused ld1b+usmopa loop
// ================================================================
.Lop_acc_usmopa:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x20], #8
    ldr     x11, [x20], #8
    ptrue   p0.b
    cbz     w22, .Ldispatch
.Lacc_usmopa_loop:
    ld1b    {z0.b}, p0/z, [x8]
    ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x11]
    ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
    usmopa  za0.s, p0/m, p0/m, z0.b, z2.b
    usmopa  za1.s, p0/m, p0/m, z0.b, z3.b
    usmopa  za2.s, p0/m, p0/m, z1.b, z2.b
    usmopa  za3.s, p0/m, p0/m, z1.b, z3.b
    addvl   x8, x8, #2
    addvl   x11, x11, #2
    sub     w22, w22, #1
    cbnz    w22, .Lacc_usmopa_loop
    b       .Ldispatch

// ================================================================
// ACC_SUMOPA (0x05) — fused ld1b+sumopa loop
// Note: sumopa = signed rows × unsigned cols (opposite of usmopa)
// ================================================================
.Lop_acc_sumopa:
    ldr     w22, [x19]
    add     x19, x19, #4
    ldr     x8, [x20], #8
    ldr     x11, [x20], #8
    ptrue   p0.b
    cbz     w22, .Ldispatch
.Lacc_sumopa_loop:
    ld1b    {z0.b}, p0/z, [x8]
    ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x11]
    ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
    smopa   za0.s, p0/m, p0/m, z0.b, z2.b
    smopa   za1.s, p0/m, p0/m, z0.b, z3.b
    smopa   za2.s, p0/m, p0/m, z1.b, z2.b
    smopa   za3.s, p0/m, p0/m, z1.b, z3.b
    addvl   x8, x8, #2
    addvl   x11, x11, #2
    sub     w22, w22, #1
    cbnz    w22, .Lacc_sumopa_loop
    b       .Ldispatch

// ================================================================
// STORE_TILES (0x06)
// Stores the 2x2 tile group as (2*SVLs) × (2*SVLs) row-major int32.
// Output is z_vector* — each z_vector is 64 bytes = 16 int32s = one SVLs row.
// A full row of output is 2 z_vectors wide (left half + right half).
// Consumes one operand: destination z_vector pointer.
//
// Layout (M4, SVLs=16, GROUP_DIM=32):
//   Row  0: z_vec[0]=za0_row0_left, z_vec[1]=za1_row0_right
//   Row  1: z_vec[2]=za0_row1_left, z_vec[3]=za1_row1_right
//   ...
//   Row 15: z_vec[30]=za0_row15_left, z_vec[31]=za1_row15_right
//   Row 16: z_vec[32]=za2_row0_left, z_vec[33]=za3_row0_right
//   ...
//   Row 31: z_vec[62]=za2_row15_left, z_vec[63]=za3_row15_right
// ================================================================
.Lop_store_tiles:
    ldr     x8, [x20], #8          // consume destination pointer
    ptrue   p0.s
    cntw    x9                     // SVLs (16 on M4)
    lsl     x10, x9, #3           // row stride in bytes: 2*SVLs*4 = SVLs*8
    // ---- Upper half: za0 (left) + za1 (right), SVLs rows ----
    mov     w12, #0
.Lse_store_upper:
    mova    {z4.s-z7.s}, za0h.s[w12, 0:3]
    mova    {z8.s-z11.s}, za1h.s[w12, 0:3]
    st1w    {z4.s}, p0, [x8]
    st1w    {z8.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z5.s}, p0, [x8]
    st1w    {z9.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z6.s}, p0, [x8]
    st1w    {z10.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z7.s}, p0, [x8]
    st1w    {z11.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .Lse_store_upper
    // ---- Lower half: za2 (left) + za3 (right), SVLs rows ----
    mov     w12, #0
.Lse_store_lower:
    mova    {z4.s-z7.s}, za2h.s[w12, 0:3]
    mova    {z8.s-z11.s}, za3h.s[w12, 0:3]
    st1w    {z4.s}, p0, [x8]
    st1w    {z8.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z5.s}, p0, [x8]
    st1w    {z9.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z6.s}, p0, [x8]
    st1w    {z10.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z7.s}, p0, [x8]
    st1w    {z11.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .Lse_store_lower
    b       .Ldispatch

// ================================================================
// LOAD_ROWS_I8 (0x07) — load 128 bytes into z0, z1
// ================================================================
.Lop_load_rows_i8:
    ldr     x8, [x20], #8
    ptrue   p0.b
    ld1b    {z0.b}, p0/z, [x8]
    cntb    x9
    ld1b    {z1.b}, p0/z, [x8, x9]
    b       .Ldispatch

// ================================================================
// LOAD_COLS_I8 (0x08) — load 128 bytes into z2, z3
// ================================================================
.Lop_load_cols_i8:
    ldr     x8, [x20], #8
    ptrue   p0.b
    ld1b    {z2.b}, p0/z, [x8]
    cntb    x9
    ld1b    {z3.b}, p0/z, [x8, x9]
    b       .Ldispatch

// ================================================================
// SMOPA_2x2 (0x09) — 4× smopa, no load
// ================================================================
.Lop_smopa_2x2:
    ptrue   p0.b
    smopa   za0.s, p0/m, p0/m, z0.b, z2.b
    smopa   za1.s, p0/m, p0/m, z0.b, z3.b
    smopa   za2.s, p0/m, p0/m, z1.b, z2.b
    smopa   za3.s, p0/m, p0/m, z1.b, z3.b
    b       .Ldispatch

// ================================================================
// UMOPA_2x2 (0x0A) — 4× umopa, no load
// ================================================================
.Lop_umopa_2x2:
    ptrue   p0.b
    umopa   za0.s, p0/m, p0/m, z0.b, z2.b
    umopa   za1.s, p0/m, p0/m, z0.b, z3.b
    umopa   za2.s, p0/m, p0/m, z1.b, z2.b
    umopa   za3.s, p0/m, p0/m, z1.b, z3.b
    b       .Ldispatch

// ================================================================
// USMOPA_2x2 (0x0B) — 4× usmopa, no load
// ================================================================
.Lop_usmopa_2x2:
    ptrue   p0.b
    usmopa  za0.s, p0/m, p0/m, z0.b, z2.b
    usmopa  za1.s, p0/m, p0/m, z0.b, z3.b
    usmopa  za2.s, p0/m, p0/m, z1.b, z2.b
    usmopa  za3.s, p0/m, p0/m, z1.b, z3.b
    b       .Ldispatch

// ================================================================
// LOAD_BIAS (0x0C)
// Loads int32 bias data from memory into ZA tiles (reverse of store_tiles).
// Same layout as store: (2*SVLs) × (2*SVLs) row-major int32.
// Consumes one operand: source pointer.
// This replaces zero_za when you want to accumulate on top of bias.
// ================================================================
.Lop_load_bias:
    ldr     x8, [x20], #8          // consume source pointer
    ptrue   p0.s
    cntw    x9                     // SVLs (16 on M4)
    lsl     x10, x9, #3           // row stride in bytes: 2*SVLs*4
    // ---- Upper half: → za0 (left) + za1 (right), SVLs rows ----
    mov     w12, #0
.Lse_loadbias_upper:
    ld1w    {z4.s}, p0/z, [x8]
    ld1w    {z8.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    ld1w    {z5.s}, p0/z, [x8]
    ld1w    {z9.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    ld1w    {z6.s}, p0/z, [x8]
    ld1w    {z10.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    ld1w    {z7.s}, p0/z, [x8]
    ld1w    {z11.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    mova    za0h.s[w12, 0:3], {z4.s-z7.s}
    mova    za1h.s[w12, 0:3], {z8.s-z11.s}
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .Lse_loadbias_upper
    // ---- Lower half: → za2 (left) + za3 (right), SVLs rows ----
    mov     w12, #0
.Lse_loadbias_lower:
    ld1w    {z4.s}, p0/z, [x8]
    ld1w    {z8.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    ld1w    {z5.s}, p0/z, [x8]
    ld1w    {z9.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    ld1w    {z6.s}, p0/z, [x8]
    ld1w    {z10.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    ld1w    {z7.s}, p0/z, [x8]
    ld1w    {z11.s}, p0/z, [x8, x9, lsl #2]
    add     x8, x8, x10
    mova    za2h.s[w12, 0:3], {z4.s-z7.s}
    mova    za3h.s[w12, 0:3], {z8.s-z11.s}
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .Lse_loadbias_lower
    b       .Ldispatch

// ================================================================
// SCALE_STORE (0x0D)
// Reads ZA tiles (int32), converts to float32, multiplies by a scalar
// scale factor, and stores as float32.
// Encoding: [0x0D] [scale: 4 bytes LE float]
// Consumes one operand: destination pointer.
//
// This is the dequantization step: int32_accum × scale → float32 output.
// The scale is a float immediate in the bytecode stream.
// ================================================================
.Lop_scale_store:
    ldr     s16, [x19]             // load float scale into s16 (bottom of z16)
    add     x19, x19, #4           // advance IP past float immediate
    ldr     x8, [x20], #8          // consume destination pointer
    ptrue   p0.s
    cntw    x9                     // SVLs
    lsl     x10, x9, #3           // row stride
    // Broadcast scale into z16 for fmul
    mov     z16.s, s16
    // ---- Upper half: za0 (left) + za1 (right) ----
    mov     w12, #0
.Lse_scalestore_upper:
    // Extract 4 rows from za0, za1
    mova    {z4.s-z7.s}, za0h.s[w12, 0:3]
    mova    {z8.s-z11.s}, za1h.s[w12, 0:3]
    // Convert int32 → float32 (scvtf) and multiply by scale
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z16.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z16.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z16.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z16.s
    scvtf   z8.s, p0/m, z8.s
    fmul    z8.s, z8.s, z16.s
    scvtf   z9.s, p0/m, z9.s
    fmul    z9.s, z9.s, z16.s
    scvtf   z10.s, p0/m, z10.s
    fmul    z10.s, z10.s, z16.s
    scvtf   z11.s, p0/m, z11.s
    fmul    z11.s, z11.s, z16.s
    // Store float32 results
    st1w    {z4.s}, p0, [x8]
    st1w    {z8.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z5.s}, p0, [x8]
    st1w    {z9.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z6.s}, p0, [x8]
    st1w    {z10.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z7.s}, p0, [x8]
    st1w    {z11.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .Lse_scalestore_upper
    // ---- Lower half: za2 (left) + za3 (right) ----
    mov     w12, #0
.Lse_scalestore_lower:
    mova    {z4.s-z7.s}, za2h.s[w12, 0:3]
    mova    {z8.s-z11.s}, za3h.s[w12, 0:3]
    scvtf   z4.s, p0/m, z4.s
    fmul    z4.s, z4.s, z16.s
    scvtf   z5.s, p0/m, z5.s
    fmul    z5.s, z5.s, z16.s
    scvtf   z6.s, p0/m, z6.s
    fmul    z6.s, z6.s, z16.s
    scvtf   z7.s, p0/m, z7.s
    fmul    z7.s, z7.s, z16.s
    scvtf   z8.s, p0/m, z8.s
    fmul    z8.s, z8.s, z16.s
    scvtf   z9.s, p0/m, z9.s
    fmul    z9.s, z9.s, z16.s
    scvtf   z10.s, p0/m, z10.s
    fmul    z10.s, z10.s, z16.s
    scvtf   z11.s, p0/m, z11.s
    fmul    z11.s, z11.s, z16.s
    st1w    {z4.s}, p0, [x8]
    st1w    {z8.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z5.s}, p0, [x8]
    st1w    {z9.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z6.s}, p0, [x8]
    st1w    {z10.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    st1w    {z7.s}, p0, [x8]
    st1w    {z11.s}, p0, [x8, x9, lsl #2]
    add     x8, x8, x10
    add     w12, w12, #4
    cmp     w12, w9
    b.lt    .Lse_scalestore_lower
    b       .Ldispatch
