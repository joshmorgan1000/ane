// sdp_attn_prefill_causal_fp32.s -- Causal self-attention prefill via SME2
//
// void sdp_attn_prefill_causal_fp32(
//     const float* qkv,      // x0: interleaved Q/K/V [seq_len, 3, num_heads, head_dim]
//     float* output,          // x1: output [seq_len, num_heads, head_dim]
//     long seq_len,           // x2
//     long num_heads,         // x3
//     long head_dim,          // x4: 64, 128, or 256
//     float scale             // s0: 1/sqrt(head_dim)
// )
//
// Same as non-causal but applies causal mask: for query position i,
// scores[j] = -inf for j > i (future positions masked out).

.section __TEXT,__text,regular,pure_instructions
.global _sdp_attn_prefill_causal_fp32
.p2align 4

_sdp_attn_prefill_causal_fp32:
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
    cbz     x2, .Lc_done
    cbz     x3, .Lc_done
    cbz     x4, .Lc_done

    // Preserve scale before malloc
    fmov    w9, s0
    str     w9, [sp, #160]

    // Save args
    mov     x19, x0                     // qkv
    mov     x20, x1                     // output
    mov     x21, x2                     // seq_len
    mov     x22, x3                     // num_heads
    mov     x23, x4                     // head_dim

    // Compute strides
    mov     x9, #3
    mul     x24, x22, x23              // num_heads * head_dim
    mul     x24, x24, x9               // qkv_pos_stride = 3 * num_heads * head_dim
    mul     x25, x22, x23              // out_pos_stride = num_heads * head_dim

    // Allocate scores buffer: seq_len floats
    lsl     x0, x21, #2
    bl      _malloc
    mov     x28, x0

    smstart sm

    // Restore scale
    ldr     w9, [sp, #160]
    fmov    s0, w9
    mov     z31.s, s0

    ptrue   p0.s

    // Load exp constants
    adr     x9, .Lc_consts
    ld1rw   {z21.s}, p0/z, [x9]
    ld1rw   {z22.s}, p0/z, [x9, #4]
    ld1rw   {z23.s}, p0/z, [x9, #8]
    ld1rw   {z24.s}, p0/z, [x9, #12]
    ld1rw   {z25.s}, p0/z, [x9, #16]
    ld1rw   {z26.s}, p0/z, [x9, #20]
    ld1rw   {z27.s}, p0/z, [x9, #24]
    ld1rw   {z29.s}, p0/z, [x9, #28]   // -inf
    fmov    z28.s, #1.0

    // Head loop
    mov     x10, #0

.Lc_head_loop:
    cmp     x10, x22
    b.ge    .Lc_exit

    mul     x11, x10, x23              // head_offset

    // Query loop
    mov     x12, #0

.Lc_query_loop:
    cmp     x12, x21
    b.ge    .Lc_next_head

    // q_ptr = qkv + (i * qkv_pos_stride + head_offset) * 4
    mul     x13, x12, x24
    add     x13, x13, x11
    add     x14, x19, x13, lsl #2

    // ================================================================
    // Phase 1: Compute scores[j] = Q[i,h,:] @ K[j,h,:] * scale
    //          For j <= i only; set scores[j] = -inf for j > i
    // ================================================================
    mov     x15, #0

.Lc_score_loop:
    cmp     x15, x21
    b.ge    .Lc_softmax

    // Causal check: if j > i, store -inf and skip dot product
    cmp     x15, x12
    b.gt    .Lc_mask_inf

    // k_ptr
    mul     x16, x15, x24
    add     x16, x16, x25              // K offset
    add     x16, x16, x11
    add     x17, x19, x16, lsl #2

    // Dot product
    mov     z8.d, #0
    mov     z9.d, #0
    mov     z10.d, #0
    mov     z11.d, #0
    mov     x8, #0
    whilelt pn9.s, x8, x23, vlx4

.Lc_dot_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]
    fmla    z8.s,  p0/m, z0.s, z4.s
    fmla    z9.s,  p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x23, vlx4
    b.first .Lc_dot_loop

    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s
    faddv   s8, p0, z8.s
    fmul    s8, s8, s31
    str     s8, [x28, x15, lsl #2]
    add     x15, x15, #1
    b       .Lc_score_loop

.Lc_mask_inf:
    // Store -inf for masked positions
    str     s29, [x28, x15, lsl #2]
    add     x15, x15, #1
    b       .Lc_score_loop

    // ================================================================
    // Phase 2: Softmax over scores (seq_len elements)
    // ================================================================
.Lc_softmax:
    // Find max
    mov     z4.d, z29.d
    mov     z5.d, z29.d
    mov     z6.d, z29.d
    mov     z7.d, z29.d
    mov     x8, #0
    whilelt pn9.s, x8, x21, vlx4

.Lc_sm_max:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]
    fmax    z4.s, p0/m, z4.s, z0.s
    fmax    z5.s, p0/m, z5.s, z1.s
    fmax    z6.s, p0/m, z6.s, z2.s
    fmax    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x21, vlx4
    b.first .Lc_sm_max

    fmax    z4.s, p0/m, z4.s, z5.s
    fmax    z6.s, p0/m, z6.s, z7.s
    fmax    z4.s, p0/m, z4.s, z6.s
    fmaxv   s20, p0, z4.s
    mov     z20.s, s20

    // Exp
    mov     x8, #0
    whilelt pn9.s, x8, x21, vlx4

.Lc_sm_exp:
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
    // Clamp NaN to 0 (from -inf causal mask entries: -inf - max = -inf, exp polynomial produces NaN)
    mov     z12.d, #0
    fcmeq   p1.s, p0/z, z16.s, z16.s
    sel     z16.s, p1, z16.s, z12.s
    fcmeq   p1.s, p0/z, z17.s, z17.s
    sel     z17.s, p1, z17.s, z12.s
    fcmeq   p1.s, p0/z, z18.s, z18.s
    sel     z18.s, p1, z18.s, z12.s
    fcmeq   p1.s, p0/z, z19.s, z19.s
    sel     z19.s, p1, z19.s, z12.s
    st1w    {z16.s-z19.s}, pn9, [x28, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x21, vlx4
    b.first .Lc_sm_exp

    // Sum
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0
    mov     x8, #0
    whilelt pn8.s, x8, x21, vlx4

.Lc_sm_sum:
    ld1w    {z0.s-z3.s}, pn8/z, [x28, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn8.s, x8, x21, vlx4
    b.first .Lc_sm_sum

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
    whilelt pn9.s, x8, x21, vlx4

.Lc_sm_norm:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z30.s
    fmul    z1.s, p0/m, z1.s, z30.s
    fmul    z2.s, p0/m, z2.s, z30.s
    fmul    z3.s, p0/m, z3.s, z30.s
    st1w    {z0.s-z3.s}, pn9, [x28, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x21, vlx4
    b.first .Lc_sm_norm

    // ================================================================
    // Phase 3: output[i,h,:] = sum_j weights[j] * V[j,h,:]
    // ================================================================
    mul     x13, x12, x25
    add     x13, x13, x11
    add     x14, x20, x13, lsl #2

    // Zero output
    mov     x8, #0
    whilelt pn9.s, x8, x23, vlx4

.Lc_zero_out:
    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0
    st1w    {z0.s-z3.s}, pn9, [x14, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x23, vlx4
    b.first .Lc_zero_out

    // Accumulate (only up to j <= i for efficiency, since weights[j>i] == 0)
    mov     x15, #0

.Lc_accum_loop:
    // Only accumulate j <= i (weights beyond are 0 from causal mask)
    add     x16, x12, #1               // i+1
    cmp     x15, x16
    b.ge    .Lc_next_query

    ldr     s8, [x28, x15, lsl #2]
    mov     z8.s, s8

    // v_ptr
    mul     x16, x15, x24
    add     x16, x16, x25, lsl #1      // V offset = 2 * (num_heads * head_dim)
    add     x16, x16, x11
    add     x17, x19, x16, lsl #2

    mov     x8, #0
    whilelt pn9.s, x8, x23, vlx4

.Lc_accum_dim:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]
    fmla    z0.s, p0/m, z8.s, z4.s
    fmla    z1.s, p0/m, z8.s, z5.s
    fmla    z2.s, p0/m, z8.s, z6.s
    fmla    z3.s, p0/m, z8.s, z7.s
    st1w    {z0.s-z3.s}, pn9, [x14, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x23, vlx4
    b.first .Lc_accum_dim

    add     x15, x15, #1
    b       .Lc_accum_loop

.Lc_next_query:
    add     x12, x12, #1
    b       .Lc_query_loop

.Lc_next_head:
    add     x10, x10, #1
    b       .Lc_head_loop

.Lc_exit:
    smstop

    mov     x0, x28
    bl      _free

.Lc_done:
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
.Lc_consts:
    .float 1.4426950408889634       // log2e
    .float 0.00015403530393381609   // c6
    .float 0.0013333558146428443    // c5
    .float 0.009618129107628477     // c4
    .float 0.05550410866482158      // c3
    .float 0.24022650695910072      // c2
    .float 0.6931471805599453       // c1 = ln2
    .long  0xFF800000               // -inf
