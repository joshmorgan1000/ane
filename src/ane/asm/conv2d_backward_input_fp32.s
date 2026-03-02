// conv2d_backward_input_fp32.s — Conv2D backward pass: compute dL/dX
//
// This computes the gradient with respect to the input tensor for a 2D convolution.
// Mathematically, this is a "transposed convolution" where we scatter gradients
// from output positions back to input positions.
//
// void conv2d_backward_input_fp32(
//     const float* grad_out,  // x0: [N, H_out, W_out, C_out] gradient from upstream
//     const float* weight,    // x1: [KH, KW, C_in, C_out] convolution weights  
//     float* grad_in,         // x2: [N, H, W, C_in] output: gradient w.r.t. input
//     long N,                 // x3: batch size
//     long H,                 // x4: input height
//     long W,                 // x5: input width
//     long C_in,              // x6: input channels
//     long KH,                // x7: kernel height
//     long KW,                // [sp, #0]: kernel width
//     long C_out,             // [sp, #8]: output channels
//     long stride,            // [sp, #16]: stride
//     long pad                // [sp, #24]: padding
// )
//
// Algorithm:
//   1. Zero-initialize grad_in
//   2. For each output position (oh, ow), scatter-add its contribution:
//      grad_in[n][ih][iw][ic] += sum_oc{ grad_out[n][oh][ow][oc] * weight[kh][kw][ic][oc] }
//      where ih = oh * stride + kh - pad, iw = ow * stride + kw - pad
//
// Optimization strategy:
//   The weight tensor [KH, KW, C_in, C_out] has C_out as the innermost dimension.
//   We vectorize over C_out (innermost) and iterate over C_in. For each ic:
//     - Load grad_out[n][oh][ow][0..C_out] as vector(s)
//     - Load weight[kh][kw][ic][0..C_out] as vector(s)  
//     - Compute dot product: sum_oc(grad_out[oc] * weight[ic][oc])
//     - Add result to grad_in[n][ih][iw][ic]
//
// Persistent registers (callee-saved):
//   x19 = grad_out ptr      x20 = weight ptr
//   x21 = grad_in ptr       x22 = N
//   x23 = H                 x24 = W
//   x25 = C_in              x26 = KH
//   x27 = KW                x28 = C_out
//
// Frame layout (288 bytes):
//   [sp, #0]:   x29, x30
//   [sp, #16]:  x19, x20
//   [sp, #32]:  x21, x22
//   [sp, #48]:  x23, x24
//   [sp, #64]:  x25, x26
//   [sp, #80]:  x27, x28
//   [sp, #96]:  d8, d9
//   [sp, #112]: d10, d11
//   [sp, #128]: d12, d13
//   [sp, #144]: d14, d15
//   [sp, #160]: stride
//   [sp, #168]: pad
//   [sp, #176]: H_out
//   [sp, #184]: W_out
//   [sp, #192]: grad_in_size (bytes)
//   [sp, #200]: grad_out_batch_stride
//   [sp, #208]: grad_in_batch_stride
//   [sp, #216]: weight_kh_stride
//   [sp, #224]: weight_kw_stride
//   [sp, #232]: grad_out_n (scratch)
//   [sp, #240]: grad_in_n (scratch)
//   [sp, #248]: partial_sum (scalar, 4 bytes)
//   [sp, #256]: spill space

.section __TEXT,__text,regular,pure_instructions
.global _conv2d_backward_input_fp32
.p2align 4

_conv2d_backward_input_fp32:
    // Prologue: save callee-saved registers
    stp     x29, x30, [sp, #-288]!
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

    // Save arguments to callee-saved registers
    mov     x19, x0                 // grad_out
    mov     x20, x1                 // weight
    mov     x21, x2                 // grad_in
    mov     x22, x3                 // N
    mov     x23, x4                 // H
    mov     x24, x5                 // W
    mov     x25, x6                 // C_in
    mov     x26, x7                 // KH

    // Load stack arguments (offset by 288 for our frame)
    ldr     x27, [sp, #288]         // KW
    ldr     x28, [sp, #296]         // C_out
    ldr     x8,  [sp, #304]         // stride
    ldr     x9,  [sp, #312]         // pad
    str     x8,  [sp, #160]         // save stride to local stack
    str     x9,  [sp, #168]         // save pad to local stack

    // Early exit checks
    cbz     x22, .Lbwi_done         // N == 0
    cbz     x23, .Lbwi_done         // H == 0
    cbz     x24, .Lbwi_done         // W == 0
    cbz     x25, .Lbwi_done         // C_in == 0

    // ── Compute derived dimensions ─────────────────────────────
    // H_out = (H + 2*pad - KH) / stride + 1
    // W_out = (W + 2*pad - KW) / stride + 1
    lsl     x10, x9, #1             // 2 * pad
    add     x11, x23, x10           // H + 2*pad
    sub     x11, x11, x26           // H + 2*pad - KH
    udiv    x11, x11, x8            // (H + 2*pad - KH) / stride
    add     x11, x11, #1            // H_out
    str     x11, [sp, #176]         // save H_out

    add     x12, x24, x10           // W + 2*pad
    sub     x12, x12, x27           // W + 2*pad - KW
    udiv    x12, x12, x8            // (W + 2*pad - KW) / stride
    add     x12, x12, #1            // W_out
    str     x12, [sp, #184]         // save W_out

    // ── Compute buffer sizes and strides ───────────────────────
    // grad_in_size = N * H * W * C_in * 4 bytes
    mul     x13, x23, x24           // H * W
    mul     x13, x13, x25           // H * W * C_in
    mul     x14, x22, x13           // N * H * W * C_in
    lsl     x14, x14, #2            // * 4 bytes
    str     x14, [sp, #192]         // save grad_in_size

    // grad_out_batch_stride = H_out * W_out * C_out (elements)
    mul     x15, x11, x12           // H_out * W_out
    mul     x15, x15, x28           // H_out * W_out * C_out
    str     x15, [sp, #200]         // save grad_out_batch_stride (elements)

    // grad_in_batch_stride = H * W * C_in (elements)
    str     x13, [sp, #208]         // save grad_in_batch_stride (elements)

    // weight_kh_stride = KW * C_in * C_out (elements per KH step)
    mul     x16, x27, x25           // KW * C_in
    mul     x16, x16, x28           // KW * C_in * C_out
    str     x16, [sp, #216]         // save weight_kh_stride

    // weight_kw_stride = C_in * C_out (elements per KW step)
    mul     x17, x25, x28           // C_in * C_out
    str     x17, [sp, #224]         // save weight_kw_stride

    // ── Zero-initialize grad_in ────────────────────────────────
    mov     x0, x21                 // grad_in ptr
    ldr     x1, [sp, #192]          // grad_in_size
    bl      _bzero

    // ── Enter streaming mode ───────────────────────────────────
    smstart sm

    ptrue   p0.s

    // ================================================================
    // Main computation loops
    // ================================================================
    // For n in 0..N:
    //   For oh in 0..H_out:
    //     For ow in 0..W_out:
    //       For kh in 0..KH:
    //         For kw in 0..KW:
    //           ih = oh * stride + kh - pad
    //           iw = ow * stride + kw - pad
    //           if (ih >= 0 && ih < H && iw >= 0 && iw < W):
    //             For ic in 0..C_in:
    //               sum = 0
    //               For oc in 0..C_out (vectorized):
    //                 sum += grad_out[n,oh,ow,oc] * weight[kh,kw,ic,oc]
    //               grad_in[n,ih,iw,ic] += sum

    mov     x10, #0                 // n = 0 (batch counter)

.Lbwi_batch_loop:
    cmp     x10, x22
    b.ge    .Lbwi_exit

    // Compute base pointers for this batch
    // grad_out_n = grad_out + n * H_out * W_out * C_out * 4
    ldr     x11, [sp, #200]         // grad_out_batch_stride (elements)
    mul     x11, x11, x10           // n * stride
    lsl     x11, x11, #2            // * 4 bytes
    add     x0, x19, x11            // grad_out_n
    str     x0, [sp, #232]          // save to scratch

    // grad_in_n = grad_in + n * H * W * C_in * 4
    ldr     x11, [sp, #208]         // grad_in_batch_stride (elements)
    mul     x11, x11, x10           // n * stride
    lsl     x11, x11, #2            // * 4 bytes
    add     x1, x21, x11            // grad_in_n
    str     x1, [sp, #240]          // save to scratch

    mov     x11, #0                 // oh = 0

.Lbwi_oh_loop:
    ldr     x8, [sp, #176]          // H_out
    cmp     x11, x8
    b.ge    .Lbwi_batch_next

    mov     x12, #0                 // ow = 0

.Lbwi_ow_loop:
    ldr     x8, [sp, #184]          // W_out
    cmp     x12, x8
    b.ge    .Lbwi_oh_next

    mov     x13, #0                 // kh = 0

.Lbwi_kh_loop:
    cmp     x13, x26                // KH
    b.ge    .Lbwi_ow_next

    // Compute ih = oh * stride + kh - pad
    ldr     x8, [sp, #160]          // stride
    mul     x14, x11, x8            // oh * stride
    add     x14, x14, x13           // oh * stride + kh
    ldr     x8, [sp, #168]          // pad
    subs    x14, x14, x8            // ih = oh * stride + kh - pad

    // Check ih bounds: ih >= 0 && ih < H
    b.mi    .Lbwi_kh_next           // ih < 0, skip
    cmp     x14, x23                // compare with H
    b.ge    .Lbwi_kh_next           // ih >= H, skip

    mov     x15, #0                 // kw = 0

.Lbwi_kw_loop:
    cmp     x15, x27                // KW
    b.ge    .Lbwi_kh_next

    // Compute iw = ow * stride + kw - pad
    ldr     x8, [sp, #160]          // stride
    mul     x16, x12, x8            // ow * stride
    add     x16, x16, x15           // ow * stride + kw
    ldr     x8, [sp, #168]          // pad
    subs    x16, x16, x8            // iw = ow * stride + kw - pad

    // Check iw bounds: iw >= 0 && iw < W
    b.mi    .Lbwi_kw_next           // iw < 0, skip
    cmp     x16, x24                // compare with W
    b.ge    .Lbwi_kw_next           // iw >= W, skip

    // ── Compute pointer to grad_out[n][oh][ow][0] ──────────────
    // offset = (oh * W_out + ow) * C_out
    ldr     x8, [sp, #184]          // W_out
    mul     x17, x11, x8            // oh * W_out
    add     x17, x17, x12           // oh * W_out + ow
    mul     x17, x17, x28           // (oh * W_out + ow) * C_out
    lsl     x17, x17, #2            // * 4 bytes
    ldr     x0, [sp, #232]          // grad_out_n
    add     x0, x0, x17             // grad_out[n][oh][ow][0] ptr

    // ── Compute pointer to weight[kh][kw][0][0] ────────────────
    // offset = kh * weight_kh_stride + kw * weight_kw_stride
    ldr     x8, [sp, #216]          // weight_kh_stride
    mul     x17, x13, x8            // kh * weight_kh_stride
    ldr     x8, [sp, #224]          // weight_kw_stride
    mul     x18, x15, x8            // kw * weight_kw_stride
    add     x17, x17, x18           // total offset (elements)
    lsl     x17, x17, #2            // * 4 bytes
    add     x2, x20, x17            // weight[kh][kw][0][0] ptr

    // ── Compute pointer to grad_in[n][ih][iw][0] ───────────────
    // offset = (ih * W + iw) * C_in
    mul     x17, x14, x24           // ih * W
    add     x17, x17, x16           // ih * W + iw
    mul     x17, x17, x25           // (ih * W + iw) * C_in
    lsl     x17, x17, #2            // * 4 bytes
    ldr     x3, [sp, #240]          // grad_in_n
    add     x3, x3, x17             // grad_in[n][ih][iw][0] ptr

    // ── Inner loop: vectorized dot product over C_out ──────────
    // For ic in 0..C_in:
    //   sum = dot(grad_out[oh,ow,0:C_out], weight[kh,kw,ic,0:C_out])
    //   grad_in[ih,iw,ic] += sum
    //
    // Weight layout: weight[kh][kw][ic][oc] - ic row, oc column
    // grad_out layout: grad_out[n][oh][ow][oc] - oc is contiguous
    // Both have C_out as the innermost dimension, perfect for vectorization!

    mov     x4, #0                  // ic = 0

.Lbwi_ic_loop:
    cmp     x4, x25                 // C_in
    b.ge    .Lbwi_kw_next

    // Initialize accumulator vector to zero
    mov     z8.d, #0
    mov     z9.d, #0
    mov     z10.d, #0
    mov     z11.d, #0

    // weight_ic_ptr = weight[kh][kw][ic][0] = weight_base + ic * C_out * 4
    mul     x5, x4, x28             // ic * C_out
    lsl     x5, x5, #2              // * 4 bytes
    add     x5, x2, x5              // weight[kh][kw][ic][0] ptr

    mov     x6, #0                  // oc = 0 (loop counter)
    whilelt pn9.s, x6, x28, vlx4    // predicate for C_out

.Lbwi_oc_vec_loop:
    // Load grad_out[n][oh][ow][oc:oc+64] (up to 4 vectors)
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x6, lsl #2]

    // Load weight[kh][kw][ic][oc:oc+64] (contiguous, perfect!)
    ld1w    {z4.s-z7.s}, pn9/z, [x5, x6, lsl #2]

    // Multiply and accumulate: z8-z11 += z0-z3 * z4-z7
    fmla    z8.s,  p0/m, z0.s, z4.s
    fmla    z9.s,  p0/m, z1.s, z5.s
    fmla    z10.s, p0/m, z2.s, z6.s
    fmla    z11.s, p0/m, z3.s, z7.s

    // Advance by 4 vectors (64 elements)
    incw    x6, all, mul #4
    whilelt pn9.s, x6, x28, vlx4
    b.first .Lbwi_oc_vec_loop

    // ── Reduce 4 accumulators to single vector, then to scalar ──
    // Tree reduce: z8 += z9, z10 += z11, then z8 += z10
    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s

    // Horizontal sum of z8 to scalar
    faddv   s16, p0, z8.s

    // Load current grad_in[n][ih][iw][ic], add result, store back
    ldr     s17, [x3, x4, lsl #2]   // load grad_in scalar
    fadd    s17, s17, s16           // accumulate
    str     s17, [x3, x4, lsl #2]   // store back

    add     x4, x4, #1              // ic++
    b       .Lbwi_ic_loop

.Lbwi_kw_next:
    add     x15, x15, #1            // kw++
    b       .Lbwi_kw_loop

.Lbwi_kh_next:
    add     x13, x13, #1            // kh++
    b       .Lbwi_kh_loop

.Lbwi_ow_next:
    add     x12, x12, #1            // ow++
    b       .Lbwi_ow_loop

.Lbwi_oh_next:
    add     x11, x11, #1            // oh++
    b       .Lbwi_oh_loop

.Lbwi_batch_next:
    add     x10, x10, #1            // n++
    b       .Lbwi_batch_loop

.Lbwi_exit:
    smstop

.Lbwi_done:
    // Epilogue: restore callee-saved registers
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     d8,  d9,  [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    ldp     x29, x30, [sp], #288
    ret
