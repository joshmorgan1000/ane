// rope_fp32.s — Rotary Position Embedding (RoPE) for FP32 vectors
//
// void rope_fp32(
//     float*       q,             // x0: [num_q_heads * head_dim] Q projections
//     float*       k,             // x1: [num_kv_heads * head_dim] K projections
//     const float* cos_cache,     // x2: [head_dim/2] cos values for this position
//     const float* sin_cache,     // x3: [head_dim/2] sin values for this position
//     long         num_q_heads,   // x4: number of query heads
//     long         num_kv_heads,  // x5: number of KV heads
//     long         head_dim       // x6: dimension per head
// )
//
// For each head:
//   half_dim = head_dim / 2
//   for i in 0..half_dim-1:
//     x0 = data[i], x1 = data[i + half_dim]
//     data[i]            = x0 * cos[i] - x1 * sin[i]
//     data[i + half_dim] = x0 * sin[i] + x1 * cos[i]

.section __TEXT,__text,regular,pure_instructions
.global _rope_fp32
.p2align 4

_rope_fp32:
    stp     x29, x30, [sp, #-144]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     d8,  d9,  [sp, #80]
    stp     d10,  d11,  [sp, #96]
    stp     d12,  d13,  [sp, #112]
    stp     d14,  d15,  [sp, #128]

    // Validate
    cbz     x6, .Lrope_done

    mov     x19, x0                 // q
    mov     x20, x1                 // k
    mov     x21, x2                 // cos_cache
    mov     x22, x3                 // sin_cache
    mov     x23, x4                 // num_q_heads
    mov     x24, x5                 // num_kv_heads
    mov     x25, x6                 // head_dim

    // half_dim = head_dim / 2
    lsr     x26, x25, #1            // x26 = half_dim

    smstart sm
    ptrue   p0.s

    // ── Process Q heads ──
    mov     x8, #0                  // head counter
    mov     x9, x19                 // current Q head pointer

.Lrope_q_loop:
    cmp     x8, x23
    b.ge    .Lrope_k_start

    // Apply RoPE to this head
    // x9 points to head data [head_dim]
    // First half: x9[0..half_dim-1]
    // Second half: x9[half_dim..head_dim-1]
    add     x10, x9, x26, lsl #2   // &data[half_dim]

    mov     x11, #0                 // i = 0
    whilelt p1.s, x11, x26         // predicate for half_dim elements

.Lrope_q_inner:
    // Load x0 = data[i], x1 = data[i + half_dim]
    ld1w    {z0.s}, p1/z, [x9, x11, lsl #2]    // z0 = data[i]
    ld1w    {z1.s}, p1/z, [x10, x11, lsl #2]   // z1 = data[i + half_dim]

    // Load cos/sin for these positions
    ld1w    {z2.s}, p1/z, [x21, x11, lsl #2]   // z2 = cos[i]
    ld1w    {z3.s}, p1/z, [x22, x11, lsl #2]   // z3 = sin[i]

    // result_lo = x0 * cos - x1 * sin
    fmul    z4.s, z0.s, z2.s       // x0 * cos
    fmls    z4.s, p0/m, z1.s, z3.s // - x1 * sin

    // result_hi = x0 * sin + x1 * cos
    fmul    z5.s, z1.s, z2.s       // x1 * cos
    fmla    z5.s, p0/m, z0.s, z3.s // + x0 * sin

    // Store results
    st1w    {z4.s}, p1, [x9, x11, lsl #2]
    st1w    {z5.s}, p1, [x10, x11, lsl #2]

    incw    x11                     // i += SVLs (16)
    whilelt p1.s, x11, x26
    b.first .Lrope_q_inner

    // Next Q head
    add     x9, x9, x25, lsl #2    // advance by head_dim floats
    add     x8, x8, #1
    b       .Lrope_q_loop

.Lrope_k_start:
    // ── Process K heads ──
    mov     x8, #0
    mov     x9, x20                 // current K head pointer

.Lrope_k_loop:
    cmp     x8, x24
    b.ge    .Lrope_end

    add     x10, x9, x26, lsl #2   // &data[half_dim]

    mov     x11, #0
    whilelt p1.s, x11, x26

.Lrope_k_inner:
    ld1w    {z0.s}, p1/z, [x9, x11, lsl #2]
    ld1w    {z1.s}, p1/z, [x10, x11, lsl #2]
    ld1w    {z2.s}, p1/z, [x21, x11, lsl #2]
    ld1w    {z3.s}, p1/z, [x22, x11, lsl #2]

    fmul    z4.s, z0.s, z2.s
    fmls    z4.s, p0/m, z1.s, z3.s

    fmul    z5.s, z1.s, z2.s
    fmla    z5.s, p0/m, z0.s, z3.s

    st1w    {z4.s}, p1, [x9, x11, lsl #2]
    st1w    {z5.s}, p1, [x10, x11, lsl #2]

    incw    x11
    whilelt p1.s, x11, x26
    b.first .Lrope_k_inner

    add     x9, x9, x25, lsl #2
    add     x8, x8, #1
    b       .Lrope_k_loop

.Lrope_end:
    smstop

.Lrope_done:
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     d8,  d9,  [sp, #80]
    ldp     d10,  d11,  [sp, #96]
    ldp     d12,  d13,  [sp, #112]
    ldp     d14,  d15,  [sp, #128]
    ldp     x29, x30, [sp], #144
    ret
