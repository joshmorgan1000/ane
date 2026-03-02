// flash_attn_prefill_causal_fp32.s -- Flash Attention causal prefill via SME2
//
// void flash_attn_prefill_causal_fp32(
//     const float* qkv,      // x0: interleaved Q/K/V [seq_len, 3, num_heads, head_dim]
//     float* output,          // x1: output [seq_len, num_heads, head_dim]
//     long seq_len,           // x2
//     long num_heads,         // x3
//     long head_dim,          // x4: 64, 128, or 256
//     float scale             // s0: 1/sqrt(head_dim)
// )
//
// Flash Attention with causal mask: online-softmax tiled algorithm.
// For query position i, only key positions j <= i contribute.
// Positions j > i are skipped entirely (no -inf needed, just skip).

.section __TEXT,__text,regular,pure_instructions
.global _flash_attn_prefill_causal_fp32
.p2align 4

_flash_attn_prefill_causal_fp32:
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
    cbz     x2, .Lfcc_done
    cbz     x3, .Lfcc_done
    cbz     x4, .Lfcc_done

    // Preserve scale
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
    mul     x24, x22, x23
    mul     x24, x24, x9               // qkv_pos_stride
    mul     x25, x22, x23              // out_pos_stride = K offset

    smstart sm

    ldr     w9, [sp, #160]
    fmov    s0, w9
    mov     z31.s, s0

    ptrue   p0.s

    adr     x9, .Lfcc_consts
    ld1rw   {z21.s}, p0/z, [x9]        // log2e
    ld1rw   {z22.s}, p0/z, [x9, #4]    // c6
    ld1rw   {z23.s}, p0/z, [x9, #8]    // c5
    ld1rw   {z24.s}, p0/z, [x9, #12]   // c4
    ld1rw   {z25.s}, p0/z, [x9, #16]   // c3
    ld1rw   {z26.s}, p0/z, [x9, #20]   // c2
    ld1rw   {z27.s}, p0/z, [x9, #24]   // c1 = ln2
    ld1rw   {z29.s}, p0/z, [x9, #28]   // -inf
    fmov    z28.s, #1.0

    // Head loop
    mov     x10, #0

.Lfcc_head_loop:
    cmp     x10, x22
    b.ge    .Lfcc_exit

    mul     x11, x10, x23              // head_offset

    // Query loop
    mov     x12, #0

.Lfcc_query_loop:
    cmp     x12, x21
    b.ge    .Lfcc_next_head

    // q_ptr
    mul     x13, x12, x24
    add     x13, x13, x11
    add     x14, x19, x13, lsl #2

    // out_ptr
    mul     x13, x12, x25
    add     x13, x13, x11
    add     x27, x20, x13, lsl #2

    // Initialize: m = -inf, l = 0
    fmov    s13, s29                    // m = -inf
    fmov    s14, #0.0                   // l = 0

    // Zero output
    mov     x8, #0
    whilelt pn9.s, x8, x23, vlx4

.Lfcc_zero_out:
    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0
    st1w    {z0.s-z3.s}, pn9, [x27, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x23, vlx4
    b.first .Lfcc_zero_out

    // KV loop: only j = 0..i (causal: skip j > i)
    mov     x15, #0

.Lfcc_kv_loop:
    // Causal bound: j <= i
    cmp     x15, x12
    b.gt    .Lfcc_normalize

    // Score: s = Q . K * scale
    mul     x16, x15, x24
    add     x16, x16, x25
    add     x16, x16, x11
    add     x17, x19, x16, lsl #2      // k_ptr

    mov     z8.d, #0
    mov     z9.d, #0
    mov     z10.d, #0
    mov     z11.d, #0
    mov     x8, #0
    whilelt pn9.s, x8, x23, vlx4

.Lfcc_dot:
    ld1w    {z0.s-z3.s}, pn9/z, [x14, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x17, x8, lsl #2]
    fmla    z8.s,  p0/m, z0.s, z4.s
    fmla    z9.s,  p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x23, vlx4
    b.first .Lfcc_dot

    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s
    faddv   s8, p0, z8.s
    fmul    s8, s8, s31

    // Online softmax update
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

    // NaN clamp for alpha: when m was -inf and m_new = s,
    // m - m_new = -inf - s = -inf, exp(-inf) should be 0
    // The polynomial may produce NaN, so clamp:
    fcmp    s9, s9                      // sets NE if NaN
    b.vc    .Lfcc_alpha_ok              // if not NaN, skip
    fmov    s9, #0.0                    // clamp NaN to 0
.Lfcc_alpha_ok:

    // l = alpha * l + p
    fmadd   s14, s9, s14, s10

    // O[:] = alpha * O[:] + p * V[j,h,:]
    mov     z16.s, s9                   // broadcast alpha
    mov     z17.s, s10                  // broadcast p

    // v_ptr
    mul     x16, x15, x24
    add     x16, x16, x25, lsl #1
    add     x16, x16, x11
    add     x17, x19, x16, lsl #2

    mov     x8, #0
    whilelt pn9.s, x8, x23, vlx4

.Lfcc_update_out:
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
    whilelt pn9.s, x8, x23, vlx4
    b.first .Lfcc_update_out

    fmov    s13, s15                    // m = m_new

    add     x15, x15, #1
    b       .Lfcc_kv_loop

.Lfcc_normalize:
    // O[:] /= l
    fmov    s15, #1.0
    fdiv    s15, s15, s14
    mov     z30.s, s15

    mov     x8, #0
    whilelt pn9.s, x8, x23, vlx4

.Lfcc_norm:
    ld1w    {z0.s-z3.s}, pn9/z, [x27, x8, lsl #2]
    fmul    z0.s, p0/m, z0.s, z30.s
    fmul    z1.s, p0/m, z1.s, z30.s
    fmul    z2.s, p0/m, z2.s, z30.s
    fmul    z3.s, p0/m, z3.s, z30.s
    st1w    {z0.s-z3.s}, pn9, [x27, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x23, vlx4
    b.first .Lfcc_norm

    add     x12, x12, #1
    b       .Lfcc_query_loop

.Lfcc_next_head:
    add     x10, x10, #1
    b       .Lfcc_head_loop

.Lfcc_exit:
    smstop

.Lfcc_done:
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
.Lfcc_consts:
    .float 1.4426950408889634       // log2e
    .float 0.00015403530393381609   // c6
    .float 0.0013333558146428443    // c5
    .float 0.009618129107628477     // c4
    .float 0.05550410866482158      // c3
    .float 0.24022650695910072      // c2
    .float 0.6931471805599453       // c1 = ln2
    .long  0xFF800000               // -inf
