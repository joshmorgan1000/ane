// fused_attn_residual_fp32.s -- Fused attention output + residual add via SME2
//
// void fused_attn_residual_fp32(
//     const float* qkv,        // x0: interleaved [seq_len, 3, num_heads, head_dim]
//     const float* residual,   // x1: residual input [seq_len, num_heads, head_dim]
//     float* output,            // x2: [seq_len, num_heads, head_dim]
//     long seq_len,             // x3
//     long num_heads,           // x4
//     long head_dim,            // x5
//     float scale               // s0
// )
//
// Fused kernel: output = attention(qkv) + residual
// For each position: compute attention output for all heads, add residual, write.

.section __TEXT,__text,regular,pure_instructions
.global _fused_attn_residual_fp32
.p2align 4

_fused_attn_residual_fp32:
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
    // [sp,#160]: scale bits
    // [sp,#168]: scores buffer ptr

    // Early exit
    cbz     x3, .Lfar_done
    cbz     x4, .Lfar_done
    cbz     x5, .Lfar_done

    // Preserve scale
    fmov    w9, s0
    str     w9, [sp, #160]

    // Save args
    mov     x19, x0                     // qkv
    mov     x20, x1                     // residual
    mov     x21, x2                     // output
    mov     x22, x3                     // seq_len
    mov     x23, x4                     // num_heads
    mov     x24, x5                     // head_dim

    // hidden_dim = num_heads * head_dim
    mul     x25, x23, x24

    // qkv_pos_stride = 3 * num_heads * head_dim
    mov     x9, #3
    mul     x26, x25, x9

    // Allocate scores: seq_len floats
    lsl     x0, x22, #2
    bl      _malloc
    str     x0, [sp, #168]

    smstart sm

    // Restore scale
    ldr     w9, [sp, #160]
    fmov    s0, w9
    mov     z31.s, s0

    ptrue   p0.s

    // Load exp constants
    adr     x9, .Lfar_consts
    ld1rw   {z21.s}, p0/z, [x9]
    ld1rw   {z22.s}, p0/z, [x9, #4]
    ld1rw   {z23.s}, p0/z, [x9, #8]
    ld1rw   {z24.s}, p0/z, [x9, #12]
    ld1rw   {z25.s}, p0/z, [x9, #16]
    ld1rw   {z26.s}, p0/z, [x9, #20]
    ld1rw   {z27.s}, p0/z, [x9, #24]
    ld1rw   {z29.s}, p0/z, [x9, #28]
    fmov    z28.s, #1.0

    // ---- For each head h ----
    mov     x10, #0

.Lfar_head_loop:
    cmp     x10, x23
    b.ge    .Lfar_exit

    mul     x11, x10, x24              // head_offset = h * head_dim

    // For each query position i
    mov     x12, #0

.Lfar_query_loop:
    cmp     x12, x22
    b.ge    .Lfar_next_head

    // q_ptr
    mul     x13, x12, x26
    add     x13, x13, x11
    add     x14, x19, x13, lsl #2

    ldr     x28, [sp, #168]

    // Phase 1: scores
    mov     x15, #0

.Lfar_score_loop:
    cmp     x15, x22
    b.ge    .Lfar_softmax

    mul     x16, x15, x26
    add     x16, x16, x25              // K offset
    add     x16, x16, x11
    add     x17, x19, x16, lsl #2

    mov     z8.d, #0
    mov     z9.d, #0
    mov     z10.d, #0
    mov     z11.d, #0

    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lfar_dot:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]
    fmla    z8.s,  p0/m, z0.s, z4.s
    fmla    z9.s,  p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lfar_dot

    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s
    faddv   s8, p0, z8.s
    fmul    s8, s8, s31
    str     s8, [x28, x15, lsl #2]

    add     x15, x15, #1
    b       .Lfar_score_loop

    // Phase 2: Softmax
.Lfar_softmax:
    mov     z4.d, z29.d
    mov     z5.d, z29.d
    mov     z6.d, z29.d
    mov     z7.d, z29.d
    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

.Lfar_sm_max:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]
    fmax    z4.s, p0/m, z4.s, z0.s
    fmax    z5.s, p0/m, z5.s, z1.s
    fmax    z6.s, p0/m, z6.s, z2.s
    fmax    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lfar_sm_max

    fmax    z4.s, p0/m, z4.s, z5.s
    fmax    z6.s, p0/m, z6.s, z7.s
    fmax    z4.s, p0/m, z4.s, z6.s
    fmaxv   s20, p0, z4.s
    mov     z20.s, s20

    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

.Lfar_sm_exp:
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
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lfar_sm_exp

    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0
    mov     x8, #0
    whilelt pn8.s, x8, x22, vlx4

.Lfar_sm_sum:
    ld1w    {z0.s-z3.s}, pn8/z, [x28, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn8.s, x8, x22, vlx4
    b.first .Lfar_sm_sum

    fadd    z4.s, p0/m, z4.s, z5.s
    fadd    z6.s, p0/m, z6.s, z7.s
    fadd    z4.s, p0/m, z4.s, z6.s
    faddv   s4, p0, z4.s
    mov     z4.s, s4
    fmov    z5.s, #1.0
    fdiv    z5.s, p0/m, z5.s, z4.s
    mov     z30.d, z5.d

    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

.Lfar_sm_norm:
    ld1w    {z0.s-z3.s}, pn9/z, [x28, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z30.s
    fmul    z1.s, p0/m, z1.s, z30.s
    fmul    z2.s, p0/m, z2.s, z30.s
    fmul    z3.s, p0/m, z3.s, z30.s
    st1w    {z0.s-z3.s}, pn9, [x28, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lfar_sm_norm

    // Phase 3: output[i,h,:] = sum_j weights[j] * V[j,h,:] + residual[i,h,:]
    // out_ptr = output + (i * out_stride + head_offset) * 4
    mul     x13, x12, x25
    add     x13, x13, x11
    add     x14, x21, x13, lsl #2

    // res_ptr = residual + (i * out_stride + head_offset) * 4
    add     x17, x20, x13, lsl #2

    // Load residual first, then accumulate V weighted
    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lfar_copy_res:
    ld1w    {z0.s-z3.s}, pn9/z, [x17, x8, lsl #2]
    st1w    {z0.s-z3.s}, pn9, [x14, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lfar_copy_res

    // Accumulate weighted V
    mov     x15, #0

.Lfar_accum:
    cmp     x15, x22
    b.ge    .Lfar_next_query

    ldr     s8, [x28, x15, lsl #2]
    mov     z8.s, s8

    // v_ptr
    mul     x16, x15, x26
    add     x16, x16, x25, lsl #1
    add     x16, x16, x11
    add     x17, x19, x16, lsl #2

    mov     x8, #0
    whilelt pn9.s, x8, x24, vlx4

.Lfar_accum_dim:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]
    fmla    z0.s, p0/m, z8.s, z4.s
    fmla    z1.s, p0/m, z8.s, z5.s
    fmla    z2.s, p0/m, z8.s, z6.s
    fmla    z3.s, p0/m, z8.s, z7.s
    st1w    {z0.s-z3.s}, pn9, [x14, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x24, vlx4
    b.first .Lfar_accum_dim

    add     x15, x15, #1
    b       .Lfar_accum

.Lfar_next_query:
    add     x12, x12, #1
    b       .Lfar_query_loop

.Lfar_next_head:
    add     x10, x10, #1
    b       .Lfar_head_loop

.Lfar_exit:
    smstop

    ldr     x0, [sp, #168]
    bl      _free

.Lfar_done:
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
.Lfar_consts:
    .float 1.4426950408889634
    .float 0.00015403530393381609
    .float 0.0013333558146428443
    .float 0.009618129107628477
    .float 0.05550410866482158
    .float 0.24022650695910072
    .float 0.6931471805599453
    .long  0xFF800000
