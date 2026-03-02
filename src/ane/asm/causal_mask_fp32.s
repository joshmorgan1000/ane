// causal_mask_fp32.s — Apply causal (lower-triangular) mask via SME2 streaming SVE
//
// void causal_mask_fp32(float* data, long seq_len, long num_heads)
//
// For each of num_heads * seq_len rows, fills positions j > i with -inf
// (0xFF800000), implementing the standard autoregressive causal mask.
//
// Memory layout: data[h * seq_len * seq_len + i * seq_len + j]
// For row i: positions 0..i unchanged, positions i+1..seq_len-1 = -inf
//
// AAPCS: x0=data (in-place), x1=seq_len, x2=num_heads

.section __TEXT,__text,regular,pure_instructions
.global _causal_mask_fp32
.p2align 4

_causal_mask_fp32:
    stp     x29, x30, [sp, #-128]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     d8,  d9,  [sp, #64]
    stp     d10,  d11,  [sp, #80]
    stp     d12,  d13,  [sp, #96]
    stp     d14,  d15,  [sp, #112]

    cbz     x1, .Ldone
    cbz     x2, .Ldone

    mov     x19, x0             // data
    mov     x20, x1             // seq_len
    mov     x21, x2             // num_heads

    // total_rows = num_heads * seq_len
    mul     x22, x20, x21

    smstart sm

    ptrue   p0.s

    // Load -inf constant and replicate to z0-z3 for bulk stores
    adr     x9, .Lneginf
    ld1rw   {z16.s}, p0/z, [x9]
    mov     z0.d, z16.d
    mov     z1.d, z16.d
    mov     z2.d, z16.d
    mov     z3.d, z16.d

    // row_idx = 0
    mov     x23, #0

.Lrow_loop:
    cmp     x23, x22
    b.ge    .Lexit

    // i = row_idx % seq_len (position within head)
    udiv    x10, x23, x20
    msub    x24, x10, x20, x23  // x24 = row_idx mod seq_len = i

    // valid = i + 1 (number of elements to keep)
    add     x10, x24, #1

    // If valid >= seq_len, entire row is valid — skip
    cmp     x10, x20
    b.ge    .Lnext_row

    // fill_count = seq_len - valid
    sub     x11, x20, x10       // x11 = number of elements to fill with -inf

    // fill_ptr = data + (row_idx * seq_len + valid) * sizeof(float)
    mul     x12, x23, x20
    add     x12, x12, x10
    add     x13, x19, x12, lsl #2

    // Fill x11 floats at x13 with -inf using vlx4 streaming stores
    mov     x8, #0
    whilelt pn9.s, x8, x11, vlx4

.Lfill_loop:
    st1w    {z0.s-z3.s}, pn9, [x13, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x11, vlx4
    b.first .Lfill_loop

.Lnext_row:
    add     x23, x23, #1
    b       .Lrow_loop

.Lexit:
    smstop

.Ldone:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     d8,  d9,  [sp, #64]
    ldp     d10,  d11,  [sp, #80]
    ldp     d12,  d13,  [sp, #96]
    ldp     d14,  d15,  [sp, #112]
    ldp     x29, x30, [sp], #128
    ret

.p2align 2
.Lneginf:
    .long   0xFF800000              // -inf as IEEE-754
