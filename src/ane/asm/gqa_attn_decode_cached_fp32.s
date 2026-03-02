// gqa_attn_decode_cached_fp32.s -- GQA cached decode attention via SME2
//
// void gqa_attn_decode_cached_fp32(
//     const float* q,           // x0: [n_heads, head_dim]
//     const float* k_cache,     // x1: [cache_len, n_kv_heads, head_dim]
//     const float* v_cache,     // x2: [cache_len, n_kv_heads, head_dim]
//     float* output,            // x3: [n_heads, head_dim]
//     long cache_len,           // x4
//     long n_heads,             // x5
//     long n_kv_heads,          // x6
//     long head_dim,            // x7
//     float scale               // s0
// )
//
// GQA decode: single-token decode with KV cache. Each Q head attends to its
// shared KV head group. Group mapping: g = h * n_kv_heads / n_heads.
//
// K/V cache layout: [cache_len, n_kv_heads, head_dim]
// KV stride between positions = n_kv_heads * head_dim

.section __TEXT,__text,regular,pure_instructions
.global _gqa_attn_decode_cached_fp32
.p2align 4

_gqa_attn_decode_cached_fp32:
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
    cbz     x4, .Lgdc_done          // cache_len == 0
    cbz     x5, .Lgdc_done          // n_heads == 0
    cbz     x6, .Lgdc_done          // n_kv_heads == 0
    cbz     x7, .Lgdc_done          // head_dim == 0

    // Preserve scale
    fmov    w9, s0
    str     w9, [sp, #160]

    // Save args
    mov     x19, x0                 // q
    mov     x20, x1                 // k_cache
    mov     x21, x2                 // v_cache
    mov     x22, x3                 // output
    mov     x23, x4                 // cache_len
    mov     x24, x5                 // n_heads
    mov     x25, x6                 // n_kv_heads
    mov     x26, x7                 // head_dim

    // kv_pos_stride = n_kv_heads * head_dim (stride between cache positions)
    mul     x27, x25, x26

    // Allocate scores buffer: cache_len floats
    lsl     x0, x23, #2
    bl      _malloc
    mov     x28, x0

    smstart sm

    ldr     w9, [sp, #160]
    fmov    s0, w9
    mov     z31.s, s0

    ptrue   p0.s

    adr     x9, .Lgdc_consts
    ld1rw   {z21.s}, p0/z, [x9]
    ld1rw   {z22.s}, p0/z, [x9, #4]
    ld1rw   {z23.s}, p0/z, [x9, #8]
    ld1rw   {z24.s}, p0/z, [x9, #12]
    ld1rw   {z25.s}, p0/z, [x9, #16]
    ld1rw   {z26.s}, p0/z, [x9, #20]
    ld1rw   {z27.s}, p0/z, [x9, #24]
    ld1rw   {z29.s}, p0/z, [x9, #28]
    fmov    z28.s, #1.0

    // ---- Head loop ----
    mov     x10, #0                 // h = 0

.Lgdc_head_loop:
    cmp     x10, x24
    b.ge    .Lgdc_exit

    // Group: g = h * n_kv_heads / n_heads
    mul     x11, x10, x25
    udiv    x11, x11, x24          // g

    // q_ptr = q + h * head_dim * 4
    mul     x12, x10, x26
    add     x14, x19, x12, lsl #2

    // k_head_off = g * head_dim (within a cache position)
    mul     x13, x11, x26

    // v_head_off = g * head_dim
    // (same as k_head_off since K and V caches have same layout)

    // ================================================================
    // Phase 1: scores[j] = Q[h,:] @ K_cache[j,g,:] * scale
    // ================================================================
    mov     x15, #0

.Lgdc_score_loop:
    cmp     x15, x23
    b.ge    .Lgdc_softmax

    // k_ptr = k_cache + (j * kv_pos_stride + k_head_off) * 4
    mul     x16, x15, x27
    add     x16, x16, x13
    add     x17, x20, x16, lsl #2

    mov     z8.d, #0
    mov     z9.d, #0
    mov     z10.d, #0
    mov     z11.d, #0

    mov     x8, #0
    whilelt pn9.s, x8, x26, vlx4

.Lgdc_dot:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]
    fmla    z8.s,  p0/m, z0.s, z4.s
    fmla    z9.s,  p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x26, vlx4
    b.first .Lgdc_dot

    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s
    faddv   s8, p0, z8.s
    fmul    s8, s8, s31
    str     s8, [x28, x15, lsl #2]

    add     x15, x15, #1
    b       .Lgdc_score_loop

    // ================================================================
    // Phase 2: Softmax
    // ================================================================
.Lgdc_softmax:
    mov     z4.d, z29.d
    mov     z5.d, z29.d
    mov     z6.d, z29.d
    mov     z7.d, z29.d
    mov     x8, #0
    whilelt pn9.s, x8, x23, vlx4

.Lgdc_sm_max:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]
    fmax    z4.s, p0/m, z4.s, z0.s
    fmax    z5.s, p0/m, z5.s, z1.s
    fmax    z6.s, p0/m, z6.s, z2.s
    fmax    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x23, vlx4
    b.first .Lgdc_sm_max

    fmax    z4.s, p0/m, z4.s, z5.s
    fmax    z6.s, p0/m, z6.s, z7.s
    fmax    z4.s, p0/m, z4.s, z6.s
    fmaxv   s20, p0, z4.s
    mov     z20.s, s20

    mov     x8, #0
    whilelt pn9.s, x8, x23, vlx4

.Lgdc_sm_exp:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]
    fsub    z0.s, p0/m, z0.s, z20.s
    fsub    z1.s, p0/m, z1.s, z20.s
    fsub    z2.s, p0/m, z2.s, z20.s
    fsub    z3.s, p0/m, z3.s, z20.s
    fmul    z0.s, p0/m, z0.s, z21.s
    fmul    z1.s, p0/m, z1.s, z21.s
    fmul    z2.s, p0/m, z2.s, z21.s
    fmul    z3.s, p0/m, z3.s, z21.s
    movprfx z8, z0
    frintn  z8.s, p0/m, z0.s
    movprfx z9, z1
    frintn  z9.s, p0/m, z1.s
    movprfx z10, z2
    frintn  z10.s, p0/m, z2.s
    movprfx z11, z3
    frintn  z11.s, p0/m, z3.s
    fsub    z0.s, p0/m, z0.s, z8.s
    fsub    z1.s, p0/m, z1.s, z9.s
    fsub    z2.s, p0/m, z2.s, z10.s
    fsub    z3.s, p0/m, z3.s, z11.s
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s
    mov     z16.d, z22.d
    mov     z17.d, z22.d
    mov     z18.d, z22.d
    mov     z19.d, z22.d
    fmad    z16.s, p0/m, z0.s, z23.s
    fmad    z17.s, p0/m, z1.s, z23.s
    fmad    z18.s, p0/m, z2.s, z23.s
    fmad    z19.s, p0/m, z3.s, z23.s
    fmad    z16.s, p0/m, z0.s, z24.s
    fmad    z17.s, p0/m, z1.s, z24.s
    fmad    z18.s, p0/m, z2.s, z24.s
    fmad    z19.s, p0/m, z3.s, z24.s
    fmad    z16.s, p0/m, z0.s, z25.s
    fmad    z17.s, p0/m, z1.s, z25.s
    fmad    z18.s, p0/m, z2.s, z25.s
    fmad    z19.s, p0/m, z3.s, z25.s
    fmad    z16.s, p0/m, z0.s, z26.s
    fmad    z17.s, p0/m, z1.s, z26.s
    fmad    z18.s, p0/m, z2.s, z26.s
    fmad    z19.s, p0/m, z3.s, z26.s
    fmad    z16.s, p0/m, z0.s, z27.s
    fmad    z17.s, p0/m, z1.s, z27.s
    fmad    z18.s, p0/m, z2.s, z27.s
    fmad    z19.s, p0/m, z3.s, z27.s
    fmad    z16.s, p0/m, z0.s, z28.s
    fmad    z17.s, p0/m, z1.s, z28.s
    fmad    z18.s, p0/m, z2.s, z28.s
    fmad    z19.s, p0/m, z3.s, z28.s
    fscale  z16.s, p0/m, z16.s, z8.s
    fscale  z17.s, p0/m, z17.s, z9.s
    fscale  z18.s, p0/m, z18.s, z10.s
    fscale  z19.s, p0/m, z19.s, z11.s
    st1w    {z16.s-z19.s}, pn9, [x28, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x23, vlx4
    b.first .Lgdc_sm_exp

    // Sum
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0
    mov     x8, #0
    whilelt pn8.s, x8, x23, vlx4

.Lgdc_sm_sum:
    ld1w    {z0.s-z3.s}, pn8/z, [x28, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn8.s, x8, x23, vlx4
    b.first .Lgdc_sm_sum

    fadd    z4.s, p0/m, z4.s, z5.s
    fadd    z6.s, p0/m, z6.s, z7.s
    fadd    z4.s, p0/m, z4.s, z6.s
    faddv   s4, p0, z4.s
    mov     z4.s, s4
    fmov    z5.s, #1.0
    fdiv    z5.s, p0/m, z5.s, z4.s
    mov     z30.d, z5.d

    // Normalize
    mov     x8, #0
    whilelt pn9.s, x8, x23, vlx4

.Lgdc_sm_norm:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z30.s
    fmul    z1.s, p0/m, z1.s, z30.s
    fmul    z2.s, p0/m, z2.s, z30.s
    fmul    z3.s, p0/m, z3.s, z30.s
    st1w    {z0.s-z3.s}, pn9, [x28, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x23, vlx4
    b.first .Lgdc_sm_norm

    // ================================================================
    // Phase 3: output[h,:] = sum_j weights[j] * V_cache[j,g,:]
    // ================================================================
    // out_ptr = output + h * head_dim * 4
    mul     x12, x10, x26
    add     x14, x22, x12, lsl #2

    // Zero output
    mov     x8, #0
    whilelt pn9.s, x8, x26, vlx4

.Lgdc_zero_out:
    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0
    st1w    {z0.s-z3.s}, pn9, [x14, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x26, vlx4
    b.first .Lgdc_zero_out

    // Accumulate
    mov     x15, #0

.Lgdc_accum:
    cmp     x15, x23
    b.ge    .Lgdc_next_head

    ldr     s8, [x28, x15, lsl #2]
    mov     z8.s, s8

    // v_ptr = v_cache + (j * kv_pos_stride + v_head_off) * 4
    mul     x16, x15, x27
    add     x16, x16, x13          // + g * head_dim
    add     x17, x21, x16, lsl #2

    mov     x8, #0
    whilelt pn9.s, x8, x26, vlx4

.Lgdc_accum_dim:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]
    fmla    z0.s, p0/m, z8.s, z4.s
    fmla    z1.s, p0/m, z8.s, z5.s
    fmla    z2.s, p0/m, z8.s, z6.s
    fmla    z3.s, p0/m, z8.s, z7.s
    st1w    {z0.s-z3.s}, pn9, [x14, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x26, vlx4
    b.first .Lgdc_accum_dim

    add     x15, x15, #1
    b       .Lgdc_accum

.Lgdc_next_head:
    add     x10, x10, #1
    b       .Lgdc_head_loop

.Lgdc_exit:
    smstop

    mov     x0, x28
    bl      _free

.Lgdc_done:
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
.Lgdc_consts:
    .float 1.4426950408889634
    .float 0.00015403530393381609
    .float 0.0013333558146428443
    .float 0.009618129107628477
    .float 0.05550410866482158
    .float 0.24022650695910072
    .float 0.6931471805599453
    .long  0xFF800000
