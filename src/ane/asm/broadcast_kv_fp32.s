// broadcast_kv_fp32.s — Broadcast KV heads for GQA attention via SME2 streaming SVE
//
// void broadcast_kv_fp32(const float* in, float* out,
//                        long seq_len, long head_dim,
//                        long n_kv_heads, long n_heads)
//
// For each KV head h (0..n_kv_heads-1), copies its seq_len*head_dim floats
// to n_heads/n_kv_heads positions in the output, producing the expanded
// tensor needed for GQA attention.
//
// AAPCS: x0=in, x1=out, x2=seq_len, x3=head_dim, x4=n_kv_heads, x5=n_heads

.section __TEXT,__text,regular,pure_instructions
.global _broadcast_kv_fp32
.p2align 4

_broadcast_kv_fp32:
    stp     x29, x30, [sp, #-128]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     d8,  d9,  [sp, #64]
    stp     d10,  d11,  [sp, #80]
    stp     d12,  d13,  [sp, #96]
    stp     d14,  d15,  [sp, #112]

    // Early exit if nothing to do
    cbz     x4, .Ldone
    cbz     x5, .Ldone

    // Save args to callee-saved registers
    mov     x19, x0             // in
    mov     x20, x1             // out
    mov     x21, x2             // seq_len
    mov     x22, x3             // head_dim
    mov     x23, x4             // n_kv_heads
    mov     x24, x5             // n_heads

    // repeat = n_heads / n_kv_heads
    udiv    x9, x24, x23       // x9 = repeat factor
    // block_size = seq_len * head_dim (floats per KV head)
    mul     x10, x21, x22      // x10 = block_size

    cbz     x10, .Ldone

    smstart sm

    // Outer loop: iterate over KV heads
    mov     x11, #0             // x11 = kv_head index

.Lkv_head_loop:
    // src = in + kv_head * block_size * 4
    mul     x12, x11, x10
    add     x13, x19, x12, lsl #2   // x13 = src pointer

    // base_dst_head = kv_head * repeat
    mul     x14, x11, x9       // x14 = base destination head index

    // Inner loop: iterate over repeat copies
    mov     x15, #0             // x15 = repeat index

.Lrepeat_loop:
    // dst_head = base_dst_head + repeat_idx
    add     x16, x14, x15
    // dst = out + dst_head * block_size * 4
    mul     x16, x16, x10
    add     x17, x20, x16, lsl #2   // x17 = dst pointer

    // Copy block_size floats from x13 to x17 using vlx4
    mov     x8, #0
    whilelt pn9.s, x8, x10, vlx4

.Lcopy_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x13, x8, lsl #2]
    st1w    {z0.s-z3.s}, pn9, [x17, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x10, vlx4
    b.first .Lcopy_loop

    add     x15, x15, #1
    cmp     x15, x9
    b.lt    .Lrepeat_loop

    add     x11, x11, #1
    cmp     x11, x23
    b.lt    .Lkv_head_loop

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
