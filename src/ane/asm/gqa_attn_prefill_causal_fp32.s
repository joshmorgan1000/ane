// gqa_attn_prefill_causal_fp32.s -- GQA causal prefill attention via SME2
//
// void gqa_attn_prefill_causal_fp32(
//     const float* qkv,       // x0: interleaved [seq_len, (n_heads + 2*n_kv_heads), head_dim]
//     float* output,           // x1: [seq_len, n_heads, head_dim]
//     long seq_len,            // x2
//     long n_heads,            // x3: number of query heads
//     long n_kv_heads,         // x4: number of KV heads
//     long head_dim,           // x5: 64, 128, or 256
//     float scale              // s0: 1/sqrt(head_dim)
// )
//
// GQA causal prefill: multiple Q heads share the same KV head.
// Group mapping: g = h * n_kv_heads / n_heads (integer division).
//
// QKV layout (interleaved per position):
//   Q[p,h,d] = qkv[(p * total_heads + h) * head_dim + d]           h in [0, n_heads)
//   K[p,g,d] = qkv[(p * total_heads + n_heads + g) * head_dim + d] g in [0, n_kv_heads)
//   V[p,g,d] = qkv[(p * total_heads + n_heads + n_kv_heads + g) * head_dim + d]
// where total_heads = n_heads + 2*n_kv_heads
//
// Causal mask: score[i,j] = -inf for j > i (upper triangular mask).

.section __TEXT,__text,regular,pure_instructions
.global _gqa_attn_prefill_causal_fp32
.p2align 4

_gqa_attn_prefill_causal_fp32:
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
    cbz     x2, .Lgcp_done          // seq_len == 0
    cbz     x3, .Lgcp_done          // n_heads == 0
    cbz     x4, .Lgcp_done          // n_kv_heads == 0
    cbz     x5, .Lgcp_done          // head_dim == 0

    // Preserve scale: s0 -> stack (safe from malloc clobber)
    fmov    w9, s0
    str     w9, [sp, #160]

    // Save args to callee-saved regs
    mov     x19, x0                 // qkv
    mov     x20, x1                 // output
    mov     x21, x2                 // seq_len
    mov     x22, x3                 // n_heads
    mov     x23, x4                 // n_kv_heads
    mov     x24, x5                 // head_dim

    // total_heads = n_heads + 2*n_kv_heads
    add     x25, x22, x23, lsl #1  // total_heads

    // qkv_pos_stride = total_heads * head_dim (floats per position in qkv)
    mul     x26, x25, x24          // qkv_pos_stride

    // out_pos_stride = n_heads * head_dim (floats per position in output)
    mul     x27, x22, x24          // out_pos_stride

    // Allocate scores buffer: seq_len floats
    lsl     x0, x21, #2            // seq_len * 4 bytes
    bl      _malloc
    mov     x28, x0                 // scores buffer

    smstart sm

    // Restore scale from stack
    ldr     w9, [sp, #160]
    fmov    s0, w9
    mov     z31.s, s0               // z31 = broadcast scale

    ptrue   p0.s

    // Load exp constants for softmax
    adr     x9, .Lgcp_consts
    ld1rw   {z21.s}, p0/z, [x9]        // log2e
    ld1rw   {z22.s}, p0/z, [x9, #4]    // c6
    ld1rw   {z23.s}, p0/z, [x9, #8]    // c5
    ld1rw   {z24.s}, p0/z, [x9, #12]   // c4
    ld1rw   {z25.s}, p0/z, [x9, #16]   // c3
    ld1rw   {z26.s}, p0/z, [x9, #20]   // c2
    ld1rw   {z27.s}, p0/z, [x9, #24]   // c1 = ln2
    ld1rw   {z29.s}, p0/z, [x9, #28]   // -inf
    fmov    z28.s, #1.0                 // c0

    // ---- Outer loop: for each Q head h ----
    mov     x10, #0                 // h = 0

.Lgcp_head_loop:
    cmp     x10, x22
    b.ge    .Lgcp_exit

    // Compute group index: g = h * n_kv_heads / n_heads
    mul     x11, x10, x23          // h * n_kv_heads
    udiv    x11, x11, x22          // g = h * n_kv_heads / n_heads

    // q_head_off = h * head_dim (offset within a position for Q head h)
    mul     x12, x10, x24

    // k_head_off = (n_heads + g) * head_dim (offset within position for K group g)
    add     x13, x22, x11
    mul     x13, x13, x24

    // v_head_off = (n_heads + n_kv_heads + g) * head_dim
    add     x14, x22, x23
    add     x14, x14, x11
    mul     x14, x14, x24          // v_head_off

    // out_head_off = h * head_dim (within output position)
    // Same as q_head_off for output since output is [seq_len, n_heads, head_dim]
    // but output stride is out_pos_stride, not qkv_pos_stride
    mul     x15, x10, x24          // out_head_off (same as x12)

    // ---- Inner loop: for each query position i ----
    mov     x16, #0                 // i = 0

.Lgcp_query_loop:
    cmp     x16, x21
    b.ge    .Lgcp_next_head

    // q_ptr = qkv + (i * qkv_pos_stride + q_head_off) * 4
    mul     x17, x16, x26          // i * qkv_pos_stride
    add     x17, x17, x12          // + q_head_off
    add     x0, x19, x17, lsl #2   // q_ptr

    // ================================================================
    // Phase 1: scores[j] = dot(Q[i,h,:], K[j,g,:]) * scale, with causal mask
    // Only compute for j in [0, i] (causal: can only attend to past+current)
    // ================================================================
    mov     x1, #0                  // j = 0

    // Causal: effective_len = i + 1
    add     x2, x16, #1            // effective_len for causal

.Lgcp_score_loop:
    cmp     x1, x2                  // j < effective_len (i+1)
    b.ge    .Lgcp_fill_neg_inf

    // k_ptr = qkv + (j * qkv_pos_stride + k_head_off) * 4
    mul     x3, x1, x26
    add     x3, x3, x13
    add     x4, x19, x3, lsl #2    // k_ptr

    // Dot product: Q[i,h,:] @ K[j,g,:]
    mov     z8.d, #0
    mov     z9.d, #0
    mov     z10.d, #0
    mov     z11.d, #0

    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lgcp_dot_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]    // Q chunk
    ld1w    {z4.s-z7.s}, pn9/z, [x4, x8, lsl #2]    // K chunk
    fmla    z8.s,  p0/m, z0.s, z4.s
    fmla    z9.s,  p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lgcp_dot_loop

    // Reduce
    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s
    faddv   s8, p0, z8.s
    fmul    s8, s8, s31
    str     s8, [x28, x1, lsl #2]

    add     x1, x1, #1
    b       .Lgcp_score_loop

    // Fill remaining positions (j > i) with -inf
.Lgcp_fill_neg_inf:
    cmp     x2, x21
    b.ge    .Lgcp_softmax

    // Fill masked positions with large negative value (-1e9).
    // We avoid IEEE -inf because the polynomial exp approximation computes
    // 2^f where f = x*log2e - round(x*log2e); for x=-inf this produces NaN
    // (since -inf - (-inf) = NaN). A large finite negative gives exp() ~ 0.
    // Load -1e9 as float bits: 0xCE6E6B28
    mov     w9, #0x6B28
    movk    w9, #0xCE6E, lsl #16   // w9 = float bits for approximately -1e9
    mov     x1, x2                  // j = i+1
.Lgcp_fill_loop:
    cmp     x1, x21
    b.ge    .Lgcp_softmax
    str     w9, [x28, x1, lsl #2]
    add     x1, x1, #1
    b       .Lgcp_fill_loop

    // ================================================================
    // Phase 2: Softmax over scores (seq_len elements, masked positions are -inf)
    // ================================================================
.Lgcp_softmax:
    // 2a: Find max
    mov     z4.d, z29.d
    mov     z5.d, z29.d
    mov     z6.d, z29.d
    mov     z7.d, z29.d

    mov     x8, #0
    whilelt pn9.s, x8, x21, vlx4

.Lgcp_sm_max:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]
    fmax    z4.s, p0/m, z4.s, z0.s
    fmax    z5.s, p0/m, z5.s, z1.s
    fmax    z6.s, p0/m, z6.s, z2.s
    fmax    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x21, vlx4
    b.first .Lgcp_sm_max

    fmax    z4.s, p0/m, z4.s, z5.s
    fmax    z6.s, p0/m, z6.s, z7.s
    fmax    z4.s, p0/m, z4.s, z6.s
    fmaxv   s20, p0, z4.s
    mov     z20.s, s20              // broadcast max

    // 2b: Compute exp(score - max) in-place
    mov     x8, #0
    whilelt pn9.s, x8, x21, vlx4

.Lgcp_sm_exp:
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
    whilelt pn9.s, x8, x21, vlx4
    b.first .Lgcp_sm_exp

    // 2c: Sum exp values
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0

    mov     x8, #0
    whilelt pn8.s, x8, x21, vlx4

.Lgcp_sm_sum:
    ld1w    {z0.s-z3.s}, pn8/z, [x28, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn8.s, x8, x21, vlx4
    b.first .Lgcp_sm_sum

    fadd    z4.s, p0/m, z4.s, z5.s
    fadd    z6.s, p0/m, z6.s, z7.s
    fadd    z4.s, p0/m, z4.s, z6.s
    faddv   s4, p0, z4.s
    mov     z4.s, s4

    fmov    z5.s, #1.0
    fdiv    z5.s, p0/m, z5.s, z4.s
    mov     z30.d, z5.d             // z30 = 1/sum

    // 2d: Normalize scores
    mov     x8, #0
    whilelt pn9.s, x8, x21, vlx4

.Lgcp_sm_norm:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z30.s
    fmul    z1.s, p0/m, z1.s, z30.s
    fmul    z2.s, p0/m, z2.s, z30.s
    fmul    z3.s, p0/m, z3.s, z30.s
    st1w    {z0.s-z3.s}, pn9, [x28, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x21, vlx4
    b.first .Lgcp_sm_norm

    // ================================================================
    // Phase 3: output[i,h,:] = sum_j scores[j] * V[j,g,:]
    // ================================================================
    // out_ptr = output + (i * out_pos_stride + out_head_off) * 4
    mul     x0, x16, x27           // i * out_pos_stride
    add     x0, x0, x15            // + out_head_off
    add     x0, x20, x0, lsl #2   // out_ptr

    // Zero output accumulator
    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lgcp_zero_out:
    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0
    st1w    {z0.s-z3.s}, pn9, [x0, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lgcp_zero_out

    // Accumulate: out += scores[j] * V[j,g,:]
    mov     x1, #0                  // j = 0

.Lgcp_accum_loop:
    cmp     x1, x21
    b.ge    .Lgcp_next_query

    // Load scores[j] and broadcast
    ldr     s8, [x28, x1, lsl #2]
    mov     z8.s, s8

    // v_ptr = qkv + (j * qkv_pos_stride + v_head_off) * 4
    mul     x3, x1, x26
    add     x3, x3, x14
    add     x4, x19, x3, lsl #2    // v_ptr

    // out[d] += scores[j] * V[j,g,d]
    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lgcp_accum_dim:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]    // current output
    ld1w    {z4.s-z7.s}, pn9/z, [x4, x8, lsl #2]    // V[j,g,:]
    fmla    z0.s, p0/m, z8.s, z4.s
    fmla    z1.s, p0/m, z8.s, z5.s
    fmla    z2.s, p0/m, z8.s, z6.s
    fmla    z3.s, p0/m, z8.s, z7.s
    st1w    {z0.s-z3.s}, pn9, [x0, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lgcp_accum_dim

    add     x1, x1, #1
    b       .Lgcp_accum_loop

.Lgcp_next_query:
    add     x16, x16, #1
    b       .Lgcp_query_loop

.Lgcp_next_head:
    add     x10, x10, #1
    b       .Lgcp_head_loop

.Lgcp_exit:
    smstop

    // Free scores buffer
    mov     x0, x28
    bl      _free

.Lgcp_done:
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

// Constant pool for softmax exp
.p2align 2
.Lgcp_consts:
    .float 1.4426950408889634       // log2e
    .float 0.00015403530393381609   // c6
    .float 0.0013333558146428443    // c5
    .float 0.009618129107628477     // c4
    .float 0.05550410866482158      // c3
    .float 0.24022650695910072      // c2
    .float 0.6931471805599453       // c1 = ln2
    .long  0xFF800000               // -inf
