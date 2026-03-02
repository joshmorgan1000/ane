// sdp_attn_backward_fp32.s -- SDP attention backward pass via SME2
//
// void sdp_attn_backward_fp32(
//     const float* dO,           // x0: gradient of output [seq_len, num_heads, head_dim]
//     const float* q,            // x1: cached Q from forward [seq_len, num_heads, head_dim]
//     const float* k,            // x2: cached K from forward [seq_len, num_heads, head_dim]
//     const float* v,            // x3: cached V from forward [seq_len, num_heads, head_dim]
//     const float* attn_weights, // x4: cached softmax output [num_heads, seq_len, seq_len]
//     float* dQ,                 // x5: output gradient for Q
//     float* dK,                 // x6: output gradient for K
//     float* dV,                 // x7: output gradient for V
//     long seq_len,              // [sp+0] after frame (9th integer arg)
//     long num_heads,            // [sp+8] after frame (10th integer arg)
//     long head_dim,             // [sp+16] after frame (11th integer arg)
//     float scale                // s0 (1st float arg, independent register track)
// )
//
// Backward algorithm (per head h):
//   Given forward: scores = scale * Q @ K^T, W = softmax(scores), O = W @ V
//
//   1. dV[i,h,:] += sum_j W[j,i] * dO[j,h,:]    (= W^T @ dO)
//   2. d_weights[i,j] = dO[i,h,:] @ V[j,h,:]^T   (= dO @ V^T)
//   3. d_scores = softmax_backward(d_weights, W)
//      d_scores[i,j] = W[i,j] * (d_weights[i,j] - sum_k(d_weights[i,k] * W[i,k]))
//   4. d_scores *= scale
//   5. dQ[i,h,:] += sum_j d_scores[i,j] * K[j,h,:]  (= d_scores @ K)
//   6. dK[j,h,:] += sum_i d_scores[i,j] * Q[i,h,:]  (= d_scores^T @ Q)

.section __TEXT,__text,regular,pure_instructions
.global _sdp_attn_backward_fp32
.p2align 4

// Stack frame: 208 bytes
//   [sp+0]:   x29, x30
//   [sp+16]:  x19, x20
//   [sp+32]:  x21, x22
//   [sp+48]:  x23, x24
//   [sp+64]:  x25, x26
//   [sp+80]:  x27, x28
//   [sp+96]:  d8, d9
//   [sp+112]: d10, d11
//   [sp+128]: d12, d13
//   [sp+144]: d14, d15
//   [sp+160]: local: head_dim (8)
//   [sp+168]: local: scale_bits (4, stored as w)
//   [sp+176]: local: pos_stride (8)
//   [sp+184]: local: dw_buf_ptr (8)
//   [sp+192]: local: (spare 16)
//
// Calling convention note:
//   Integer/pointer args x0-x7, overflow to stack.
//   Float args use independent s0-s7 track.
//   scale (float) arrives in s0, NOT on the stack.

_sdp_attn_backward_fp32:
    stp     x29, x30, [sp, #-208]!
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

    // Save register args
    mov     x19, x0                 // dO
    mov     x20, x1                 // q
    mov     x21, x2                 // k
    mov     x22, x3                 // v
    mov     x23, x4                 // attn_weights
    mov     x24, x5                 // dQ
    mov     x25, x6                 // dK
    mov     x26, x7                 // dV

    // Load stack args (past 208-byte frame)
    // Integer overflow: seq_len, num_heads, head_dim on stack
    // Float: scale arrives in s0 (independent float register track)
    ldr     x27, [sp, #208]         // seq_len
    ldr     x28, [sp, #216]         // num_heads

    ldr     x8, [sp, #224]          // head_dim
    str     x8, [sp, #160]          // store head_dim locally

    // Save scale (s0) to stack -- survives malloc which clobbers s0
    fmov    w9, s0
    str     w9, [sp, #168]          // store scale bits locally

    // Early exit
    cbz     x27, .Labk_done
    cbz     x28, .Labk_done
    ldr     x8, [sp, #160]
    cbz     x8, .Labk_done

    // Compute strides
    ldr     x8, [sp, #160]          // head_dim
    mul     x9, x28, x8             // pos_stride = num_heads * head_dim
    str     x9, [sp, #176]          // save pos_stride

    // Allocate d_weights buffer: seq_len * seq_len floats (per head, reused)
    mul     x0, x27, x27
    lsl     x0, x0, #2
    bl      _malloc
    str     x0, [sp, #184]          // d_weights buffer ptr

    // Restore scale from stack (saved before malloc)
    ldr     w9, [sp, #168]

    smstart sm

    // Restore scale in streaming mode (smstart zeroed z-regs)
    fmov    s15, w9
    mov     z31.s, s15              // z31 = broadcast scale for vectorized use

    ptrue   p0.s

    // ---- Per-head loop ----
    mov     x10, #0                 // h = 0

.Labk_head_loop:
    cmp     x10, x28
    b.ge    .Labk_exit

    ldr     x8, [sp, #160]          // head_dim
    mul     x11, x10, x8            // head_offset = h * head_dim

    // W_ptr = attn_weights + h * seq_len * seq_len * 4
    mul     x12, x27, x27           // seq_len^2
    mul     x12, x12, x10           // h * seq_len^2
    add     x12, x23, x12, lsl #2   // W_ptr for this head

    // ================================================================
    // Step 1: dV[j,h,:] += sum_i W[i,j] * dO[i,h,:]
    // ================================================================
    // For each j (KV position):
    mov     x13, #0                 // j = 0

.Labk_dv_j:
    cmp     x13, x27
    b.ge    .Labk_step2

    // dV_ptr = dV + (j * pos_stride + head_offset) * 4
    ldr     x9, [sp, #176]
    mul     x14, x13, x9
    add     x14, x14, x11
    add     x14, x26, x14, lsl #2   // dV[j,h,:]

    // For each i (query position):
    mov     x15, #0

.Labk_dv_i:
    cmp     x15, x27
    b.ge    .Labk_dv_j_next

    // W[i,j] = attn_weights[h, i, j]
    mul     x16, x15, x27
    add     x16, x16, x13
    ldr     s8, [x12, x16, lsl #2]
    mov     z8.s, s8

    // dO_ptr = dO + (i * pos_stride + head_offset) * 4
    ldr     x9, [sp, #176]
    mul     x16, x15, x9
    add     x16, x16, x11
    add     x17, x19, x16, lsl #2

    // dV[j,h,d] += W[i,j] * dO[i,h,d]
    mov     x8, #0
    ldr     x9, [sp, #160]
    whilelt pn9.s, x8, x9, vlx4

.Labk_dv_accum:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]
    fmla    z0.s, p0/m, z8.s, z4.s
    fmla    z1.s, p0/m, z8.s, z5.s
    fmla    z2.s, p0/m, z8.s, z6.s
    fmla    z3.s, p0/m, z8.s, z7.s
    st1w    {z0.s-z3.s}, pn9, [x14, x8, lsl #2]
    incw    x8, all, mul #4
    ldr     x9, [sp, #160]
    whilelt pn9.s, x8, x9, vlx4
    b.first .Labk_dv_accum

    add     x15, x15, #1
    b       .Labk_dv_i

.Labk_dv_j_next:
    add     x13, x13, #1
    b       .Labk_dv_j

    // ================================================================
    // Step 2: d_weights[i,j] = dO[i,h,:] @ V[j,h,:]
    // ================================================================
.Labk_step2:
    ldr     x14, [sp, #184]         // d_weights buffer

    mov     x13, #0                 // i

.Labk_dw_i:
    cmp     x13, x27
    b.ge    .Labk_step3

    // dO_ptr for query i
    ldr     x9, [sp, #176]
    mul     x16, x13, x9
    add     x16, x16, x11
    add     x17, x19, x16, lsl #2

    mov     x15, #0                 // j

.Labk_dw_j:
    cmp     x15, x27
    b.ge    .Labk_dw_i_next

    // V_ptr for position j
    ldr     x9, [sp, #176]
    mul     x16, x15, x9
    add     x16, x16, x11
    add     x8, x22, x16, lsl #2    // V[j,h,:]

    // Dot product dO[i,h,:] @ V[j,h,:]
    mov     z8.d, #0
    mov     z9.d, #0
    mov     z10.d, #0
    mov     z11.d, #0

    mov     x16, #0
    ldr     x9, [sp, #160]
    whilelt pn9.s, x16, x9, vlx4

.Labk_dw_dot:
    ld1w    {z0.s-z3.s}, pn9/z, [x17, x16, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x8, x16, lsl #2]
    fmla    z8.s,  p0/m, z0.s, z4.s
    fmla    z9.s,  p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s
    incw    x16, all, mul #4
    ldr     x9, [sp, #160]
    whilelt pn9.s, x16, x9, vlx4
    b.first .Labk_dw_dot

    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s
    faddv   s8, p0, z8.s

    // Store d_weights[i,j]
    mul     x16, x13, x27
    add     x16, x16, x15
    str     s8, [x14, x16, lsl #2]

    add     x15, x15, #1
    b       .Labk_dw_j

.Labk_dw_i_next:
    add     x13, x13, #1
    b       .Labk_dw_i

    // ================================================================
    // Step 3: d_scores = softmax_backward(d_weights, W) * scale
    // d_scores[i,j] = W[i,j] * (d_weights[i,j] - row_dot[i]) * scale
    // where row_dot[i] = sum_k d_weights[i,k] * W[i,k]
    // ================================================================
.Labk_step3:
    ldr     x14, [sp, #184]         // d_weights

    mov     x13, #0                 // i

.Labk_sb_i:
    cmp     x13, x27
    b.ge    .Labk_step5

    // Compute row_dot[i] = sum_j d_weights[i,j] * W[i,j]
    mul     x16, x13, x27           // row offset
    add     x17, x14, x16, lsl #2   // d_weights row ptr
    add     x8, x12, x16, lsl #2    // W row ptr

    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0

    mov     x15, #0
    whilelt pn9.s, x15, x27, vlx4

.Labk_rowdot:
    ld1w    {z0.s-z3.s}, pn9/z, [x17, x15, lsl #2]
    ld1w    {z8.s-z11.s}, pn9/z, [x8, x15, lsl #2]
    fmla    z4.s, p0/m, z0.s, z8.s
    fmla    z5.s, p0/m, z1.s, z9.s
    fmla    z6.s, p0/m, z2.s, z10.s
    fmla    z7.s, p0/m, z3.s, z11.s
    incw    x15, all, mul #4
    whilelt pn9.s, x15, x27, vlx4
    b.first .Labk_rowdot

    fadd    z4.s, p0/m, z4.s, z5.s
    fadd    z6.s, p0/m, z6.s, z7.s
    fadd    z4.s, p0/m, z4.s, z6.s
    faddv   s4, p0, z4.s
    mov     z30.s, s4               // z30 = row_dot broadcast

    // Compute d_scores[i,j] = W[i,j] * (d_weights[i,j] - row_dot) * scale
    mov     x15, #0
    whilelt pn9.s, x15, x27, vlx4

.Labk_dscores:
    ld1w    {z0.s-z3.s}, pn9/z, [x17, x15, lsl #2]    // d_weights[i,j]
    ld1w    {z8.s-z11.s}, pn9/z, [x8, x15, lsl #2]    // W[i,j]

    // d_weights - row_dot
    fsub    z0.s, p0/m, z0.s, z30.s
    fsub    z1.s, p0/m, z1.s, z30.s
    fsub    z2.s, p0/m, z2.s, z30.s
    fsub    z3.s, p0/m, z3.s, z30.s

    // * W[i,j]
    fmul    z0.s, p0/m, z0.s, z8.s
    fmul    z1.s, p0/m, z1.s, z9.s
    fmul    z2.s, p0/m, z2.s, z10.s
    fmul    z3.s, p0/m, z3.s, z11.s

    // * scale
    fmul    z0.s, p0/m, z0.s, z31.s
    fmul    z1.s, p0/m, z1.s, z31.s
    fmul    z2.s, p0/m, z2.s, z31.s
    fmul    z3.s, p0/m, z3.s, z31.s

    // Store back to d_weights buffer (now d_scores)
    st1w    {z0.s-z3.s}, pn9, [x17, x15, lsl #2]

    incw    x15, all, mul #4
    whilelt pn9.s, x15, x27, vlx4
    b.first .Labk_dscores

    add     x13, x13, #1
    b       .Labk_sb_i

    // ================================================================
    // Step 5: dQ[i,h,:] += sum_j d_scores[i,j] * K[j,h,:]
    // ================================================================
.Labk_step5:
    ldr     x14, [sp, #184]         // d_scores buffer

    mov     x13, #0                 // i

.Labk_dq_i:
    cmp     x13, x27
    b.ge    .Labk_step6

    // dQ_ptr for position i
    ldr     x9, [sp, #176]
    mul     x16, x13, x9
    add     x16, x16, x11
    add     x17, x24, x16, lsl #2   // dQ[i,h,:]

    // d_scores row for i
    mul     x16, x13, x27
    add     x8, x14, x16, lsl #2

    mov     x15, #0                 // j

.Labk_dq_j:
    cmp     x15, x27
    b.ge    .Labk_dq_i_next

    // d_scores[i,j]
    ldr     s8, [x8, x15, lsl #2]
    mov     z8.s, s8

    // K_ptr for position j
    ldr     x9, [sp, #176]
    mul     x16, x15, x9
    add     x16, x16, x11
    add     x9, x21, x16, lsl #2    // K[j,h,:]

    // dQ[i,h,d] += d_scores[i,j] * K[j,h,d]
    mov     x16, #0
    ldr     x3, [sp, #160]          // head_dim (use x3 as temp -- it was overwritten)
    whilelt pn9.s, x16, x3, vlx4

.Labk_dq_accum:
    ld1w    {z0.s-z3.s}, pn9/z, [x17, x16, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x9, x16, lsl #2]
    fmla    z0.s, p0/m, z8.s, z4.s
    fmla    z1.s, p0/m, z8.s, z5.s
    fmla    z2.s, p0/m, z8.s, z6.s
    fmla    z3.s, p0/m, z8.s, z7.s
    st1w    {z0.s-z3.s}, pn9, [x17, x16, lsl #2]
    incw    x16, all, mul #4
    ldr     x3, [sp, #160]
    whilelt pn9.s, x16, x3, vlx4
    b.first .Labk_dq_accum

    add     x15, x15, #1
    b       .Labk_dq_j

.Labk_dq_i_next:
    add     x13, x13, #1
    b       .Labk_dq_i

    // ================================================================
    // Step 6: dK[j,h,:] += sum_i d_scores[i,j] * Q[i,h,:]
    // ================================================================
.Labk_step6:
    ldr     x14, [sp, #184]         // d_scores

    mov     x13, #0                 // j

.Labk_dk_j:
    cmp     x13, x27
    b.ge    .Labk_next_head

    // dK_ptr for position j
    ldr     x9, [sp, #176]
    mul     x16, x13, x9
    add     x16, x16, x11
    add     x17, x25, x16, lsl #2   // dK[j,h,:]

    mov     x15, #0                 // i

.Labk_dk_i:
    cmp     x15, x27
    b.ge    .Labk_dk_j_next

    // d_scores[i,j]
    mul     x16, x15, x27
    add     x16, x16, x13
    ldr     s8, [x14, x16, lsl #2]
    mov     z8.s, s8

    // Q_ptr for position i
    ldr     x9, [sp, #176]
    mul     x16, x15, x9
    add     x16, x16, x11
    add     x9, x20, x16, lsl #2    // Q[i,h,:]

    // dK[j,h,d] += d_scores[i,j] * Q[i,h,d]
    mov     x16, #0
    ldr     x3, [sp, #160]
    whilelt pn9.s, x16, x3, vlx4

.Labk_dk_accum:
    ld1w    {z0.s-z3.s}, pn9/z, [x17, x16, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x9, x16, lsl #2]
    fmla    z0.s, p0/m, z8.s, z4.s
    fmla    z1.s, p0/m, z8.s, z5.s
    fmla    z2.s, p0/m, z8.s, z6.s
    fmla    z3.s, p0/m, z8.s, z7.s
    st1w    {z0.s-z3.s}, pn9, [x17, x16, lsl #2]
    incw    x16, all, mul #4
    ldr     x3, [sp, #160]
    whilelt pn9.s, x16, x3, vlx4
    b.first .Labk_dk_accum

    add     x15, x15, #1
    b       .Labk_dk_i

.Labk_dk_j_next:
    add     x13, x13, #1
    b       .Labk_dk_j

.Labk_next_head:
    add     x10, x10, #1
    b       .Labk_head_loop

.Labk_exit:
    smstop

    // Free d_weights buffer
    ldr     x0, [sp, #184]
    bl      _free

.Labk_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     d8,  d9,  [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    ldp     x29, x30, [sp], #208
    ret
