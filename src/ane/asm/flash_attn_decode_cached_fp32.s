// flash_attn_decode_cached_fp32.s -- Flash Attention cached decode via SME2
//
// void flash_attn_decode_cached_fp32(
//     const float* q,          // x0: [num_heads, head_dim]
//     const float* k_cache,    // x1: [cache_len, num_heads, head_dim]
//     const float* v_cache,    // x2: [cache_len, num_heads, head_dim]
//     float* output,           // x3: [num_heads, head_dim]
//     long cache_len,          // x4
//     long num_heads,          // x5
//     long head_dim,           // x6
//     float scale              // s0
// )
//
// Flash Attention decode: online-softmax for single-query attention against cache.
// Same algorithm as prefill but for decode (single query, separate K/V cache).

.section __TEXT,__text,regular,pure_instructions
.global _flash_attn_decode_cached_fp32
.p2align 4

_flash_attn_decode_cached_fp32:
    stp     x29, x30, [sp, #-176]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8,  d9,  [sp, #96]
    stp     d10, d11, [sp, #112]
    stp     d12, d13, [sp, #128]
    stp     d14, d15, [sp, #144]

    // Early exit
    cbz     x4, .Lfdc_done
    cbz     x5, .Lfdc_done
    cbz     x6, .Lfdc_done

    // Preserve scale
    fmov    w9, s0
    str     w9, [sp, #160]

    // Save args
    mov     x19, x0                     // q
    mov     x20, x1                     // k_cache
    mov     x21, x2                     // v_cache
    mov     x22, x3                     // output
    mov     x23, x4                     // cache_len
    mov     x24, x5                     // num_heads
    mov     x25, x6                     // head_dim

    // cache_pos_stride = num_heads * head_dim
    mul     x26, x24, x25

    smstart sm

    ldr     w9, [sp, #160]
    fmov    s0, w9
    mov     z31.s, s0

    ptrue   p0.s

    adr     x9, .Lfdc_consts
    ld1rw   {z21.s}, p0/z, [x9]
    ld1rw   {z22.s}, p0/z, [x9, #4]
    ld1rw   {z23.s}, p0/z, [x9, #8]
    ld1rw   {z24.s}, p0/z, [x9, #12]
    ld1rw   {z25.s}, p0/z, [x9, #16]
    ld1rw   {z26.s}, p0/z, [x9, #20]
    ld1rw   {z27.s}, p0/z, [x9, #24]
    ld1rw   {z29.s}, p0/z, [x9, #28]
    fmov    z28.s, #1.0

    // Head loop
    mov     x10, #0

.Lfdc_head_loop:
    cmp     x10, x24
    b.ge    .Lfdc_exit

    mul     x11, x10, x25              // head_offset
    add     x14, x19, x11, lsl #2      // q_ptr
    add     x27, x22, x11, lsl #2      // out_ptr

    // Initialize: m = -inf, l = 0
    fmov    s13, s29
    fmov    s14, #0.0

    // Zero output
    mov     x8, #0
    whilelt pn9.s, x8, x25, vlx4

.Lfdc_zero:
    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0
    st1w    {z0.s-z3.s}, pn9, [x27, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x25, vlx4
    b.first .Lfdc_zero

    // KV loop
    mov     x15, #0

.Lfdc_kv_loop:
    cmp     x15, x23
    b.ge    .Lfdc_normalize

    // k_ptr = k_cache + (j * cache_pos_stride + head_offset)
    mul     x16, x15, x26
    add     x16, x16, x11
    add     x17, x20, x16, lsl #2

    // Dot product Q . K
    mov     z8.d, #0
    mov     z9.d, #0
    mov     z10.d, #0
    mov     z11.d, #0
    mov     x8, #0
    whilelt pn9.s, x8, x25, vlx4

.Lfdc_dot:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]
    fmla    z8.s,  p0/m, z0.s, z4.s
    fmla    z9.s,  p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x25, vlx4
    b.first .Lfdc_dot

    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s
    faddv   s8, p0, z8.s
    fmul    s8, s8, s31                 // score

    // Online softmax
    fmax    s15, s13, s8                // m_new

    // alpha = exp(m - m_new)
    fsub    s9, s13, s15
    fmul    s9, s9, s21
    frintn  s11, s9
    fsub    s9, s9, s11
    fcvtzs  w8, s11
    fmov    s12, s22
    fmadd   s12, s12, s9, s23
    fmadd   s12, s12, s9, s24
    fmadd   s12, s12, s9, s25
    fmadd   s12, s12, s9, s26
    fmadd   s12, s12, s9, s27
    fmadd   s12, s12, s9, s28
    fmov    s9, s12
    mov     z12.s, w8
    fscale  z9.s, p0/m, z9.s, z12.s
    fmov    s9, s9                      // alpha

    // p = exp(s - m_new)
    fsub    s10, s8, s15
    fmul    s10, s10, s21
    frintn  s11, s10
    fsub    s10, s10, s11
    fcvtzs  w8, s11
    fmov    s12, s22
    fmadd   s12, s12, s10, s23
    fmadd   s12, s12, s10, s24
    fmadd   s12, s12, s10, s25
    fmadd   s12, s12, s10, s26
    fmadd   s12, s12, s10, s27
    fmadd   s12, s12, s10, s28
    fmov    s10, s12
    mov     z12.s, w8
    fscale  z10.s, p0/m, z10.s, z12.s
    fmov    s10, s10                    // p

    // NaN clamp alpha (first iteration: m=-inf, m-m_new=-inf)
    fcmp    s9, s9
    b.vc    .Lfdc_alpha_ok
    fmov    s9, #0.0
.Lfdc_alpha_ok:

    // l = alpha * l + p
    fmadd   s14, s9, s14, s10

    // O[:] = alpha * O[:] + p * V
    mov     z16.s, s9                   // broadcast alpha
    mov     z17.s, s10                  // broadcast p

    // v_ptr = v_cache + (j * cache_pos_stride + head_offset)
    mul     x16, x15, x26
    add     x16, x16, x11
    add     x17, x21, x16, lsl #2

    mov     x8, #0
    whilelt pn9.s, x8, x25, vlx4

.Lfdc_update:
    ld1w    {z0.s-z3.s}, pn9/z, [x27, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z16.s
    fmul    z1.s, p0/m, z1.s, z16.s
    fmul    z2.s, p0/m, z2.s, z16.s
    fmul    z3.s, p0/m, z3.s, z16.s
    fmla    z0.s, p0/m, z17.s, z4.s
    fmla    z1.s, p0/m, z17.s, z5.s
    fmla    z2.s, p0/m, z17.s, z6.s
    fmla    z3.s, p0/m, z17.s, z7.s
    st1w    {z0.s-z3.s}, pn9, [x27, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x25, vlx4
    b.first .Lfdc_update

    fmov    s13, s15                    // m = m_new

    add     x15, x15, #1
    b       .Lfdc_kv_loop

.Lfdc_normalize:
    fmov    s15, #1.0
    fdiv    s15, s15, s14
    mov     z30.s, s15

    mov     x8, #0
    whilelt pn9.s, x8, x25, vlx4

.Lfdc_norm:
    ld1w    {z0.s-z3.s}, pn9/z, [x27, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z30.s
    fmul    z1.s, p0/m, z1.s, z30.s
    fmul    z2.s, p0/m, z2.s, z30.s
    fmul    z3.s, p0/m, z3.s, z30.s
    st1w    {z0.s-z3.s}, pn9, [x27, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x25, vlx4
    b.first .Lfdc_norm

    add     x10, x10, #1
    b       .Lfdc_head_loop

.Lfdc_exit:
    smstop

.Lfdc_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     d8,  d9,  [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    ldp     x29, x30, [sp], #176
    ret

.p2align 2
.Lfdc_consts:
    .float 1.4426950408889634
    .float 0.00015403530393381609
    .float 0.0013333558146428443
    .float 0.009618129107628477
    .float 0.05550410866482158
    .float 0.24022650695910072
    .float 0.6931471805599453
    .long  0xFF800000
