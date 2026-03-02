// cross_attn_prefill_fp32.s -- Cross-attention prefill via SME2 streaming SVE
//
// void cross_attn_prefill_fp32(
//     const float* q,        // x0: decoder queries  [q_len, num_heads, head_dim]
//     const float* k,        // x1: encoder keys     [kv_len, num_heads, head_dim]
//     const float* v,        // x2: encoder values   [kv_len, num_heads, head_dim]
//     float* output,         // x3: output           [q_len, num_heads, head_dim]
//     long q_len,            // x4: decoder sequence length
//     long kv_len,           // x5: encoder sequence length (may differ from q_len)
//     long num_heads,        // x6
//     long head_dim,         // x7: 64, 128, or 256
//     float scale            // s0: 1/sqrt(head_dim)
// )
//
// Cross-attention: Q from decoder, K/V from encoder (separate buffers).
// No causal masking -- decoder can attend to all encoder positions.
// Score matrix is q_len x kv_len (rectangular, not square).
//
// Algorithm (per head h):
//   For each query position i (0..q_len-1):
//     1. Compute scores[j] = Q[i,h,:] @ K[j,h,:] * scale  for j in 0..kv_len-1
//     2. Apply softmax over scores (kv_len elements)
//     3. Compute output[i,h,:] = sum_j scores[j] * V[j,h,:]
//
// Memory layout: [pos, head, dim] -- head_dim is the innermost stride.
// Stride between positions = num_heads * head_dim floats.
// Stride between heads within a position = head_dim floats.

.section __TEXT,__text,regular,pure_instructions
.global _cross_attn_prefill_fp32
.p2align 4

_cross_attn_prefill_fp32:
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
    cbz     x4, .Lcap_done          // q_len == 0
    cbz     x5, .Lcap_done          // kv_len == 0
    cbz     x6, .Lcap_done          // num_heads == 0
    cbz     x7, .Lcap_done          // head_dim == 0

    // Preserve scale: s0 -> stack slot (w9 is caller-saved, clobbered by malloc)
    fmov    w9, s0
    str     w9, [sp, #160]          // save scale bits to stack (bytes 160-163 are free)

    // Save args to callee-saved regs
    mov     x19, x0                 // q
    mov     x20, x1                 // k
    mov     x21, x2                 // v
    mov     x22, x3                 // output
    mov     x23, x4                 // q_len
    mov     x24, x5                 // kv_len
    mov     x25, x6                 // num_heads
    mov     x26, x7                 // head_dim

    // Compute strides (in floats)
    // pos_stride = num_heads * head_dim
    mul     x27, x25, x26          // pos_stride (floats)

    // Allocate temp buffer for scores: kv_len floats (on heap)
    lsl     x0, x24, #2            // kv_len * 4 bytes
    bl      _malloc
    mov     x28, x0                 // scores buffer

    smstart sm

    // Restore scale from stack (safe from malloc clobber)
    ldr     w9, [sp, #160]
    fmov    s0, w9
    mov     z31.s, s0               // z31 = broadcast scale

    ptrue   p0.s

    // Load exp constants for softmax
    adr     x9, .Lcap_consts
    ld1rw   {z21.s}, p0/z, [x9]        // log2e
    ld1rw   {z22.s}, p0/z, [x9, #4]    // c6
    ld1rw   {z23.s}, p0/z, [x9, #8]    // c5
    ld1rw   {z24.s}, p0/z, [x9, #12]   // c4
    ld1rw   {z25.s}, p0/z, [x9, #16]   // c3
    ld1rw   {z26.s}, p0/z, [x9, #20]   // c2
    ld1rw   {z27.s}, p0/z, [x9, #24]   // c1 = ln2
    ld1rw   {z29.s}, p0/z, [x9, #28]   // -inf
    fmov    z28.s, #1.0                 // c0

    // ---- Outer loop: for each head h ----
    mov     x10, #0                 // h = 0

.Lcap_head_loop:
    cmp     x10, x25
    b.ge    .Lcap_exit

    // head_offset = h * head_dim (floats)
    mul     x11, x10, x26          // head_offset

    // ---- Inner loop: for each query position i ----
    mov     x12, #0                 // i = 0

.Lcap_query_loop:
    cmp     x12, x23
    b.ge    .Lcap_next_head

    // q_ptr = q + (i * pos_stride + head_offset) * 4
    mul     x13, x12, x27          // i * pos_stride
    add     x13, x13, x11          // + head_offset
    add     x14, x19, x13, lsl #2  // q_ptr (byte address)

    // ================================================================
    // Phase 1: Compute scores[j] = dot(Q[i,h,:], K[j,h,:]) * scale
    // ================================================================
    mov     x15, #0                 // j = 0

.Lcap_score_loop:
    cmp     x15, x24
    b.ge    .Lcap_softmax

    // k_ptr = k + (j * pos_stride + head_offset) * 4
    mul     x16, x15, x27
    add     x16, x16, x11
    add     x17, x20, x16, lsl #2  // k_ptr

    // Dot product: Q[i,h,:] @ K[j,h,:]
    mov     z8.d, #0                // acc0
    mov     z9.d, #0                // acc1
    mov     z10.d, #0               // acc2
    mov     z11.d, #0               // acc3

    mov     x8, #0
    whilelt pn9.s, x8, x26, vlx4

.Lcap_dot_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]   // Q chunk
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]   // K chunk

    fmla    z8.s,  p0/m, z0.s, z4.s
    fmla    z9.s,  p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x26, vlx4
    b.first .Lcap_dot_loop

    // Reduce 4 accumulators -> scalar
    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s
    faddv   s8, p0, z8.s

    // Multiply by scale
    fmul    s8, s8, s31

    // Store score[j]
    str     s8, [x28, x15, lsl #2]

    add     x15, x15, #1
    b       .Lcap_score_loop

    // ================================================================
    // Phase 2: Softmax over scores (kv_len elements)
    // ================================================================
.Lcap_softmax:
    // 2a: Find max
    mov     z4.d, z29.d             // -inf
    mov     z5.d, z29.d
    mov     z6.d, z29.d
    mov     z7.d, z29.d

    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lcap_sm_max:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]
    fmax    z4.s, p0/m, z4.s, z0.s
    fmax    z5.s, p0/m, z5.s, z1.s
    fmax    z6.s, p0/m, z6.s, z2.s
    fmax    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lcap_sm_max

    fmax    z4.s, p0/m, z4.s, z5.s
    fmax    z6.s, p0/m, z6.s, z7.s
    fmax    z4.s, p0/m, z4.s, z6.s
    fmaxv   s20, p0, z4.s
    mov     z20.s, s20              // broadcast max

    // 2b: Compute exp(score - max) in-place
    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lcap_sm_exp:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]

    fsub    z0.s, p0/m, z0.s, z20.s
    fsub    z1.s, p0/m, z1.s, z20.s
    fsub    z2.s, p0/m, z2.s, z20.s
    fsub    z3.s, p0/m, z3.s, z20.s

    // z_scaled = x * log2e
    fmul    z0.s, p0/m, z0.s, z21.s
    fmul    z1.s, p0/m, z1.s, z21.s
    fmul    z2.s, p0/m, z2.s, z21.s
    fmul    z3.s, p0/m, z3.s, z21.s

    // n = round(z_scaled)
    movprfx z8, z0
    frintn  z8.s, p0/m, z0.s
    movprfx z9, z1
    frintn  z9.s, p0/m, z1.s
    movprfx z10, z2
    frintn  z10.s, p0/m, z2.s
    movprfx z11, z3
    frintn  z11.s, p0/m, z3.s

    // f = z_scaled - n
    fsub    z0.s, p0/m, z0.s, z8.s
    fsub    z1.s, p0/m, z1.s, z9.s
    fsub    z2.s, p0/m, z2.s, z10.s
    fsub    z3.s, p0/m, z3.s, z11.s

    // Convert n to integer
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s

    // Horner: 2^f
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
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lcap_sm_exp

    // 2c: Sum exp values
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0

    mov     x8, #0
    whilelt pn8.s, x8, x24, vlx4

.Lcap_sm_sum:
    ld1w    {z0.s-z3.s}, pn8/z, [x28, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn8.s, x8, x24, vlx4
    b.first .Lcap_sm_sum

    fadd    z4.s, p0/m, z4.s, z5.s
    fadd    z6.s, p0/m, z6.s, z7.s
    fadd    z4.s, p0/m, z4.s, z6.s
    faddv   s4, p0, z4.s
    mov     z4.s, s4

    // 1/sum via fdiv for accuracy
    fmov    z5.s, #1.0
    fdiv    z5.s, p0/m, z5.s, z4.s
    mov     z30.d, z5.d             // z30 = 1/sum

    // 2d: Normalize scores
    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lcap_sm_norm:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z30.s
    fmul    z1.s, p0/m, z1.s, z30.s
    fmul    z2.s, p0/m, z2.s, z30.s
    fmul    z3.s, p0/m, z3.s, z30.s
    st1w    {z0.s-z3.s}, pn9, [x28, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lcap_sm_norm

    // ================================================================
    // Phase 3: Output[i,h,:] = sum_j scores[j] * V[j,h,:]
    // ================================================================
    // out_ptr = output + (i * pos_stride + head_offset) * 4
    mul     x13, x12, x27
    add     x13, x13, x11
    add     x14, x22, x13, lsl #2  // out_ptr

    // Zero output accumulator (head_dim floats at out_ptr)
    mov     x8, #0
    whilelt pn9.s, x8, x26, vlx4

.Lcap_zero_out:
    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0
    st1w    {z0.s-z3.s}, pn9, [x14, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x26, vlx4
    b.first .Lcap_zero_out

    // Accumulate: out += scores[j] * V[j,h,:]
    mov     x15, #0                 // j = 0

.Lcap_accum_loop:
    cmp     x15, x24
    b.ge    .Lcap_next_query

    // Load scores[j] and broadcast
    ldr     s8, [x28, x15, lsl #2]
    mov     z8.s, s8

    // v_ptr = v + (j * pos_stride + head_offset) * 4
    mul     x16, x15, x27
    add     x16, x16, x11
    add     x17, x21, x16, lsl #2  // v_ptr

    // out[d] += scores[j] * V[j,h,d]
    mov     x8, #0
    whilelt pn9.s, x8, x26, vlx4

.Lcap_accum_dim:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]   // current output
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]   // V[j,h,:]

    fmla    z0.s, p0/m, z8.s, z4.s
    fmla    z1.s, p0/m, z8.s, z5.s
    fmla    z2.s, p0/m, z8.s, z6.s
    fmla    z3.s, p0/m, z8.s, z7.s

    st1w    {z0.s-z3.s}, pn9, [x14, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x26, vlx4
    b.first .Lcap_accum_dim

    add     x15, x15, #1
    b       .Lcap_accum_loop

.Lcap_next_query:
    add     x12, x12, #1
    b       .Lcap_query_loop

.Lcap_next_head:
    add     x10, x10, #1
    b       .Lcap_head_loop

.Lcap_exit:
    smstop

    // Free scores buffer
    mov     x0, x28
    bl      _free

.Lcap_done:
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
.Lcap_consts:
    .float 1.4426950408889634       // log2e
    .float 0.00015403530393381609   // c6
    .float 0.0013333558146428443    // c5
    .float 0.009618129107628477     // c4
    .float 0.05550410866482158      // c3
    .float 0.24022650695910072      // c2
    .float 0.6931471805599453       // c1 = ln2
    .long  0xFF800000               // -inf
