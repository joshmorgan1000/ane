// kv_cache_append_fp32.s -- Append new K/V entries to cache via SME2 streaming SVE
//
// void kv_cache_append_fp32(
//     float* k_cache,              // x0: key cache [max_len, num_heads, head_dim]
//     float* v_cache,              // x1: value cache [max_len, num_heads, head_dim]
//     const float* new_k,          // x2: new key [num_heads, head_dim]
//     const float* new_v,          // x3: new value [num_heads, head_dim]
//     long pos,                    // x4: position to write (current cache length)
//     long num_heads,              // x5
//     long head_dim                // x6: 64, 128, or 256
// )
//
// For each head h, copies head_dim floats from new_k/new_v into
// k_cache[pos,h,:] and v_cache[pos,h,:].
// Uses streaming SVE ld1w/st1w for bulk vectorized copy.

.section __TEXT,__text,regular,pure_instructions
.global _kv_cache_append_fp32
.p2align 4

_kv_cache_append_fp32:
    stp     x29, x30, [sp, #-128]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     d8,  d9,  [sp, #64]
    stp     d10, d11, [sp, #80]
    stp     d12, d13, [sp, #96]
    stp     d14, d15, [sp, #112]

    // Early exit
    cbz     x5, .Lkva_done
    cbz     x6, .Lkva_done

    // Save args
    mov     x19, x0                     // k_cache
    mov     x20, x1                     // v_cache
    mov     x21, x2                     // new_k
    mov     x22, x3                     // new_v
    mov     x23, x5                     // num_heads
    mov     x24, x6                     // head_dim

    // pos_stride = num_heads * head_dim (floats between positions)
    mul     x9, x23, x24               // pos_stride

    // cache_offset = pos * pos_stride (floats)
    mul     x10, x4, x9                // pos * num_heads * head_dim

    smstart sm

    // Loop over heads
    mov     x11, #0                     // h = 0

.Lkva_head_loop:
    cmp     x11, x23
    b.ge    .Lkva_exit

    // head_offset = h * head_dim
    mul     x12, x11, x24

    // dst_k = k_cache + (cache_offset + head_offset) * 4
    add     x13, x10, x12
    add     x14, x19, x13, lsl #2

    // dst_v = v_cache + (cache_offset + head_offset) * 4
    add     x15, x20, x13, lsl #2

    // src_k = new_k + head_offset * 4
    add     x16, x21, x12, lsl #2

    // src_v = new_v + head_offset * 4
    add     x17, x22, x12, lsl #2

    // Copy head_dim floats: new_k -> k_cache[pos,h,:]
    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lkva_copy_k:
    ld1w    {z0.s-z3.s}, pn9/z, [x16, x8, lsl #2]
    st1w    {z0.s-z3.s}, pn9, [x14, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lkva_copy_k

    // Copy head_dim floats: new_v -> v_cache[pos,h,:]
    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lkva_copy_v:
    ld1w    {z0.s-z3.s}, pn9/z, [x17, x8, lsl #2]
    st1w    {z0.s-z3.s}, pn9, [x15, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lkva_copy_v

    add     x11, x11, #1
    b       .Lkva_head_loop

.Lkva_exit:
    smstop

.Lkva_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     d8,  d9,  [sp, #64]
    ldp     d10, d11, [sp, #80]
    ldp     d12, d13, [sp, #96]
    ldp     d14, d15, [sp, #112]
    ldp     x29, x30, [sp], #128
    ret
