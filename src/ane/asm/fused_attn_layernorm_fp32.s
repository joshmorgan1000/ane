// fused_attn_layernorm_fp32.s -- Fused attention output + layernorm via SME2
//
// void fused_attn_layernorm_fp32(
//     const float* qkv,        // x0: interleaved [seq_len, 3, num_heads, head_dim]
//     float* output,            // x1: [seq_len, num_heads * head_dim]
//     const float* gamma,       // x2: layernorm scale [num_heads * head_dim]
//     const float* beta,        // x3: layernorm shift [num_heads * head_dim]
//     long seq_len,             // x4
//     long num_heads,           // x5
//     long head_dim,            // x6: 64, 128, or 256
//     float scale,              // s0: attention scale 1/sqrt(head_dim)
//     float eps                 // s1: layernorm epsilon (typically 1e-5)
// )
//
// Fused kernel: for each sequence position:
//   1. Run non-causal SDP attention for all heads -> attn_out[num_heads * head_dim]
//   2. Apply layernorm to attn_out: normalize, scale by gamma, shift by beta
//   3. Write final result to output
// Eliminates intermediate attention output buffer.

.section __TEXT,__text,regular,pure_instructions
.global _fused_attn_layernorm_fp32
.p2align 4

_fused_attn_layernorm_fp32:
    stp     x29, x30, [sp, #-192]!
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
    // [sp,#160]: scale bits
    // [sp,#164]: eps bits
    // [sp,#168]: scores buffer ptr
    // [sp,#176]: attn_out buffer ptr

    // Early exit
    cbz     x4, .Lfal_done              // seq_len == 0
    cbz     x5, .Lfal_done              // num_heads == 0
    cbz     x6, .Lfal_done              // head_dim == 0

    // Preserve scale and eps before malloc (s0, s1 clobbered)
    fmov    w9, s0
    str     w9, [sp, #160]
    fmov    w10, s1
    str     w10, [sp, #164]

    // Save args
    mov     x19, x0                     // qkv
    mov     x20, x1                     // output
    mov     x21, x2                     // gamma
    mov     x22, x3                     // beta
    mov     x23, x4                     // seq_len
    mov     x24, x5                     // num_heads
    mov     x25, x6                     // head_dim

    // hidden_dim = num_heads * head_dim
    mul     x26, x24, x25              // hidden_dim

    // qkv_pos_stride = 3 * num_heads * head_dim
    mov     x9, #3
    mul     x27, x26, x9               // qkv_pos_stride

    // Allocate scores buffer: seq_len floats
    lsl     x0, x23, #2
    bl      _malloc
    str     x0, [sp, #168]

    // Allocate attn_out buffer: hidden_dim floats (temporary per position)
    lsl     x0, x26, #2
    bl      _malloc
    str     x0, [sp, #176]

    smstart sm

    // Restore scale and eps
    ldr     w9, [sp, #160]
    fmov    s0, w9
    mov     z31.s, s0                   // z31 = broadcast scale
    ldr     w10, [sp, #164]
    fmov    s1, w10
    // eps stays in s1 for layernorm

    ptrue   p0.s

    // Load exp constants for softmax
    adr     x9, .Lfal_consts
    ld1rw   {z21.s}, p0/z, [x9]        // log2e
    ld1rw   {z22.s}, p0/z, [x9, #4]    // c6
    ld1rw   {z23.s}, p0/z, [x9, #8]    // c5
    ld1rw   {z24.s}, p0/z, [x9, #12]   // c4
    ld1rw   {z25.s}, p0/z, [x9, #16]   // c3
    ld1rw   {z26.s}, p0/z, [x9, #20]   // c2
    ld1rw   {z27.s}, p0/z, [x9, #24]   // c1 = ln2
    ld1rw   {z29.s}, p0/z, [x9, #28]   // -inf
    fmov    z28.s, #1.0                 // c0

    // ---- Outer loop: for each position i ----
    mov     x12, #0                     // i = 0

.Lfal_pos_loop:
    cmp     x12, x23
    b.ge    .Lfal_exit

    // Zero attn_out buffer
    ldr     x14, [sp, #176]            // attn_out ptr
    mov     x8, #0
    whilelt pn9.s, x8, x26, vlx4

.Lfal_zero_attn:
    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0
    st1w    {z0.s-z3.s}, pn9, [x14, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x26, vlx4
    b.first .Lfal_zero_attn

    // For each head: compute attention and accumulate into attn_out
    mov     x10, #0                     // h = 0

.Lfal_head_loop:
    cmp     x10, x24
    b.ge    .Lfal_layernorm

    mul     x11, x10, x25              // head_offset = h * head_dim

    // q_ptr = qkv + (i * qkv_pos_stride + head_offset) * 4
    mul     x13, x12, x27              // i * qkv_pos_stride
    add     x13, x13, x11              // + head_offset
    add     x14, x19, x13, lsl #2      // q_ptr

    ldr     x28, [sp, #168]            // scores buffer

    // Phase 1: scores[j] = Q[i,h,:] @ K[j,h,:] * scale
    mov     x15, #0                     // j = 0

.Lfal_score_loop:
    cmp     x15, x23
    b.ge    .Lfal_softmax

    // k_ptr = qkv + (j * qkv_pos_stride + num_heads*head_dim + head_offset) * 4
    mul     x16, x15, x27
    add     x16, x16, x26              // + K offset (num_heads * head_dim)
    add     x16, x16, x11              // + head_offset
    add     x17, x19, x16, lsl #2

    // Dot product
    mov     z8.d, #0
    mov     z9.d, #0
    mov     z10.d, #0
    mov     z11.d, #0

    mov     x8, #0
    whilelt pn9.s, x8, x25, vlx4

.Lfal_dot_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]
    fmla    z8.s,  p0/m, z0.s, z4.s
    fmla    z9.s,  p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x25, vlx4
    b.first .Lfal_dot_loop

    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s
    faddv   s8, p0, z8.s
    fmul    s8, s8, s31
    str     s8, [x28, x15, lsl #2]

    add     x15, x15, #1
    b       .Lfal_score_loop

    // Phase 2: Softmax
.Lfal_softmax:
    // 2a: max
    mov     z4.d, z29.d
    mov     z5.d, z29.d
    mov     z6.d, z29.d
    mov     z7.d, z29.d
    mov     x8, #0
    whilelt pn9.s, x8, x23, vlx4

.Lfal_sm_max:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]
    fmax    z4.s, p0/m, z4.s, z0.s
    fmax    z5.s, p0/m, z5.s, z1.s
    fmax    z6.s, p0/m, z6.s, z2.s
    fmax    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x23, vlx4
    b.first .Lfal_sm_max

    fmax    z4.s, p0/m, z4.s, z5.s
    fmax    z6.s, p0/m, z6.s, z7.s
    fmax    z4.s, p0/m, z4.s, z6.s
    fmaxv   s20, p0, z4.s
    mov     z20.s, s20

    // 2b: exp(score - max)
    mov     x8, #0
    whilelt pn9.s, x8, x23, vlx4

.Lfal_sm_exp:
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
    b.first .Lfal_sm_exp

    // 2c: sum
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0
    mov     x8, #0
    whilelt pn8.s, x8, x23, vlx4

.Lfal_sm_sum:
    ld1w    {z0.s-z3.s}, pn8/z, [x28, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn8.s, x8, x23, vlx4
    b.first .Lfal_sm_sum

    fadd    z4.s, p0/m, z4.s, z5.s
    fadd    z6.s, p0/m, z6.s, z7.s
    fadd    z4.s, p0/m, z4.s, z6.s
    faddv   s4, p0, z4.s
    mov     z4.s, s4
    fmov    z5.s, #1.0
    fdiv    z5.s, p0/m, z5.s, z4.s
    mov     z30.d, z5.d                 // inv_sum

    // 2d: normalize scores
    mov     x8, #0
    whilelt pn9.s, x8, x23, vlx4

.Lfal_sm_norm:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z30.s
    fmul    z1.s, p0/m, z1.s, z30.s
    fmul    z2.s, p0/m, z2.s, z30.s
    fmul    z3.s, p0/m, z3.s, z30.s
    st1w    {z0.s-z3.s}, pn9, [x28, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x23, vlx4
    b.first .Lfal_sm_norm

    // Phase 3: attn_out[h*head_dim..] += weights[j] * V[j,h,:]
    ldr     x14, [sp, #176]            // attn_out ptr
    // offset into attn_out for this head
    add     x14, x14, x11, lsl #2      // attn_out + head_offset * 4

    mov     x15, #0

.Lfal_accum_loop:
    cmp     x15, x23
    b.ge    .Lfal_next_head

    ldr     s8, [x28, x15, lsl #2]
    mov     z8.s, s8

    // v_ptr = qkv + (j * qkv_pos_stride + 2*num_heads*head_dim + head_offset) * 4
    mul     x16, x15, x27
    add     x16, x16, x26, lsl #1      // + 2*hidden_dim = V offset
    add     x16, x16, x11              // + head_offset
    add     x17, x19, x16, lsl #2

    mov     x8, #0
    whilelt pn9.s, x8, x25, vlx4

.Lfal_accum_dim:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]
    fmla    z0.s, p0/m, z8.s, z4.s
    fmla    z1.s, p0/m, z8.s, z5.s
    fmla    z2.s, p0/m, z8.s, z6.s
    fmla    z3.s, p0/m, z8.s, z7.s
    st1w    {z0.s-z3.s}, pn9, [x14, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x25, vlx4
    b.first .Lfal_accum_dim

    add     x15, x15, #1
    b       .Lfal_accum_loop

.Lfal_next_head:
    add     x10, x10, #1
    b       .Lfal_head_loop

    // ============================================================
    // LayerNorm on attn_out[hidden_dim], write to output[i,:]
    // ============================================================
.Lfal_layernorm:
    ldr     x14, [sp, #176]            // attn_out

    // Step 1: compute mean
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0
    mov     x8, #0
    whilelt pn9.s, x8, x26, vlx4

.Lfal_ln_mean:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x26, vlx4
    b.first .Lfal_ln_mean

    fadd    z4.s, p0/m, z4.s, z5.s
    fadd    z6.s, p0/m, z6.s, z7.s
    fadd    z4.s, p0/m, z4.s, z6.s
    faddv   s4, p0, z4.s

    // mean = sum / hidden_dim
    scvtf   s5, x26
    fdiv    s4, s4, s5
    mov     z20.s, s4                   // z20 = broadcast mean

    // Step 2: compute variance
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0
    mov     x8, #0
    whilelt pn9.s, x8, x26, vlx4

.Lfal_ln_var:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    fsub    z0.s, p0/m, z0.s, z20.s
    fsub    z1.s, p0/m, z1.s, z20.s
    fsub    z2.s, p0/m, z2.s, z20.s
    fsub    z3.s, p0/m, z3.s, z20.s
    fmla    z4.s, p0/m, z0.s, z0.s
    fmla    z5.s, p0/m, z1.s, z1.s
    fmla    z6.s, p0/m, z2.s, z2.s
    fmla    z7.s, p0/m, z3.s, z3.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x26, vlx4
    b.first .Lfal_ln_var

    fadd    z4.s, p0/m, z4.s, z5.s
    fadd    z6.s, p0/m, z6.s, z7.s
    fadd    z4.s, p0/m, z4.s, z6.s
    faddv   s4, p0, z4.s

    // var = sumsq / hidden_dim (re-compute s5 since z5 was zeroed for variance accumulation)
    scvtf   s5, x26
    fdiv    s4, s4, s5
    // inv_std = 1 / sqrt(var + eps)
    // Reload eps (s1 clobbered by variance loop loading z0-z3 which includes s1)
    ldr     w9, [sp, #164]
    fmov    s1, w9
    fadd    s4, s4, s1                  // var + eps
    fsqrt   s4, s4
    fmov    s6, #1.0
    fdiv    s4, s6, s4
    mov     z19.s, s4                   // z19 = broadcast inv_std

    // Step 3: normalize, scale, shift, write
    // out_ptr = output + i * hidden_dim * 4
    mul     x16, x12, x26
    add     x17, x20, x16, lsl #2      // out_ptr

    mov     x8, #0
    whilelt pn9.s, x8, x26, vlx4

.Lfal_ln_apply:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]     // attn_out
    ld1w    {z4.s-z7.s}, pn9/z, [x21, x8, lsl #2]     // gamma
    ld1w    {z8.s-z11.s}, pn9/z, [x22, x8, lsl #2]    // beta

    // (x - mean) * inv_std
    fsub    z0.s, p0/m, z0.s, z20.s
    fsub    z1.s, p0/m, z1.s, z20.s
    fsub    z2.s, p0/m, z2.s, z20.s
    fsub    z3.s, p0/m, z3.s, z20.s
    fmul    z0.s, p0/m, z0.s, z19.s
    fmul    z1.s, p0/m, z1.s, z19.s
    fmul    z2.s, p0/m, z2.s, z19.s
    fmul    z3.s, p0/m, z3.s, z19.s

    // * gamma + beta
    fmla    z8.s, p0/m, z0.s, z4.s     // beta + (normalized * gamma) -- but this is wrong order
    fmla    z9.s, p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s

    st1w    {z8.s-z11.s}, pn9, [x17, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x26, vlx4
    b.first .Lfal_ln_apply

    add     x12, x12, #1
    b       .Lfal_pos_loop

.Lfal_exit:
    smstop

    // Free scores buffer
    ldr     x0, [sp, #168]
    bl      _free

    // Free attn_out buffer
    ldr     x0, [sp, #176]
    bl      _free

.Lfal_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     d8,  d9,  [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    ldp     x29, x30, [sp], #192
    ret

.p2align 2
.Lfal_consts:
    .float 1.4426950408889634       // log2e
    .float 0.00015403530393381609   // c6
    .float 0.0013333558146428443    // c5
    .float 0.009618129107628477     // c4
    .float 0.05550410866482158      // c3
    .float 0.24022650695910072      // c2
    .float 0.6931471805599453       // c1 = ln2
    .long  0xFF800000               // -inf
