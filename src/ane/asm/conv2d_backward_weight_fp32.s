// conv2d_backward_weight_fp32.s — Conv2D backward pass for weight gradients
//
// Computes gradients w.r.t. weights and optionally bias for 2D convolution.
//
// void conv2d_backward_weight_fp32(
//     const float* input,     // x0: [N, H, W, C_in] original input
//     const float* grad_out,  // x1: [N, H_out, W_out, C_out] gradient from upstream
//     float* grad_weight,     // x2: [KH, KW, C_in, C_out] output: gradient w.r.t. weights
//     float* grad_bias,       // x3: [C_out] output: gradient w.r.t. bias (or nullptr)
//     long N,                 // x4: batch size
//     long H,                 // x5: input height
//     long W,                 // x6: input width
//     long C_in,              // x7: input channels
//     long KH,                // [sp, #0]: kernel height
//     long KW,                // [sp, #8]: kernel width
//     long C_out,             // [sp, #16]: output channels
//     long stride,            // [sp, #24]: stride
//     long pad                // [sp, #32]: padding
// );
//
// Mathematical derivation:
//   grad_weight[kh][kw][ic][oc] = sum over (n, oh, ow):
//       input[n][oh*stride + kh - pad][ow*stride + kw - pad][ic] * grad_out[n][oh][ow][oc]
//   (with bounds checking for padding - treat out-of-bounds input as 0)
//
//   grad_bias[oc] = sum over (n, oh, ow): grad_out[n][oh][ow][oc]
//
// Memory layout (NHWC):
//   input:       linear at [n*H*W*C_in + ih*W*C_in + iw*C_in + ic]
//   grad_out:    linear at [n*H_out*W_out*C_out + oh*W_out*C_out + ow*C_out + oc]
//   grad_weight: linear at [kh*KW*C_in*C_out + kw*C_in*C_out + ic*C_out + oc]
//   grad_bias:   linear at [oc]
//
// Register allocation strategy:
//   Callee-saved (preserved across function):
//     x19 = input base ptr
//     x20 = grad_out base ptr
//     x21 = grad_weight base ptr
//     x22 = grad_bias base ptr (or nullptr)
//     x23 = N (batch size)
//     x24 = H (input height)
//     x25 = W (input width)
//     x26 = C_in
//     x27 = KH
//     x28 = KW
//
//   Stack slots (256 byte frame):
//     [sp, #160] = C_out
//     [sp, #168] = stride
//     [sp, #176] = pad
//     [sp, #184] = H_out
//     [sp, #192] = W_out
//     [sp, #200] = input_stride_h (W * C_in)
//     [sp, #208] = input_stride_n (H * W * C_in)
//     [sp, #216] = go_stride_h (W_out * C_out)
//     [sp, #224] = go_stride_n (H_out * W_out * C_out)
//
//   Loop variables (scratch - reloaded as needed):
//     x0 = n (batch index)
//     x1 = input_n ptr (input + n * input_stride_n)
//     x2 = go_n ptr (grad_out + n * go_stride_n)
//     x3 = oh (output height index)
//     x4 = go_oh ptr (go_n + oh * go_stride_h)
//     x5 = ow (output width index)
//     x6 = go_ptr (go_oh + ow * C_out) - grad_out for this position
//     x7 = kh (kernel height index)
//     x8 = kw (kernel width index)
//     x9-x15 = scratch for calculations

.section __TEXT,__text,regular,pure_instructions
.global _conv2d_backward_weight_fp32
.p2align 4

_conv2d_backward_weight_fp32:
    // ═══════════════════════════════════════════════════════════════════════
    // Prologue: save callee-saved registers
    // Frame size: 256 bytes (16-byte aligned)
    // ═══════════════════════════════════════════════════════════════════════
    stp     x29, x30, [sp, #-256]!
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

    // ═══════════════════════════════════════════════════════════════════════
    // Load all arguments into callee-saved registers
    // ═══════════════════════════════════════════════════════════════════════
    mov     x19, x0                 // input ptr
    mov     x20, x1                 // grad_out ptr
    mov     x21, x2                 // grad_weight ptr
    mov     x22, x3                 // grad_bias ptr (may be nullptr)
    mov     x23, x4                 // N (batch size)
    mov     x24, x5                 // H (input height)
    mov     x25, x6                 // W (input width)
    mov     x26, x7                 // C_in

    // Load stack arguments (caller's frame starts at sp+256)
    ldr     x27, [sp, #256]         // KH
    ldr     x28, [sp, #264]         // KW
    ldr     x9,  [sp, #272]         // C_out
    ldr     x10, [sp, #280]         // stride
    ldr     x11, [sp, #288]         // pad

    // Store to scratch area
    str     x9,  [sp, #160]         // C_out
    str     x10, [sp, #168]         // stride
    str     x11, [sp, #176]         // pad

    // ═══════════════════════════════════════════════════════════════════════
    // Compute derived dimensions
    // H_out = (H + 2*pad - KH) / stride + 1
    // W_out = (W + 2*pad - KW) / stride + 1
    // ═══════════════════════════════════════════════════════════════════════
    lsl     x12, x11, #1            // 2 * pad
    add     x13, x24, x12           // H + 2*pad
    sub     x13, x13, x27           // H + 2*pad - KH
    udiv    x13, x13, x10           // (H + 2*pad - KH) / stride
    add     x13, x13, #1            // H_out

    add     x14, x25, x12           // W + 2*pad
    sub     x14, x14, x28           // W + 2*pad - KW
    udiv    x14, x14, x10           // (W + 2*pad - KW) / stride
    add     x14, x14, #1            // W_out

    str     x13, [sp, #184]         // H_out
    str     x14, [sp, #192]         // W_out

    // Early exit if dimensions are invalid
    cbz     x23, .Lcbw_done         // N == 0
    cbz     x13, .Lcbw_done         // H_out == 0
    cbz     x14, .Lcbw_done         // W_out == 0
    cbz     x9,  .Lcbw_done         // C_out == 0

    // ═══════════════════════════════════════════════════════════════════════
    // Compute strides for array indexing (in elements, not bytes)
    // ═══════════════════════════════════════════════════════════════════════
    // Input strides (NHWC): [N, H, W, C_in]
    mul     x15, x25, x26           // input_stride_h = W * C_in
    mul     x16, x24, x15           // input_stride_n = H * input_stride_h
    str     x15, [sp, #200]         // input_stride_h
    str     x16, [sp, #208]         // input_stride_n

    // grad_out strides (NHWC): [N, H_out, W_out, C_out]
    mul     x17, x14, x9            // go_stride_h = W_out * C_out
    mul     x18, x13, x17           // go_stride_n = H_out * go_stride_h
    str     x17, [sp, #216]         // go_stride_h
    str     x18, [sp, #224]         // go_stride_n

    // ═══════════════════════════════════════════════════════════════════════
    // ENTER STREAMING MODE
    // ═══════════════════════════════════════════════════════════════════════
    smstart sm
    ptrue   p0.s

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 0: Zero-initialize grad_weight
    // Total size = KH * KW * C_in * C_out
    // ═══════════════════════════════════════════════════════════════════════
    mul     x0, x27, x28            // KH * KW
    mul     x0, x0, x26             // KH * KW * C_in
    ldr     x9, [sp, #160]          // reload C_out
    mul     x0, x0, x9              // KH * KW * C_in * C_out

    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0

    mov     x8, #0
    whilelt pn9.s, x8, x0, vlx4

.Lcbw_zero_weight_loop:
    st1w    {z0.s-z3.s}, pn9, [x21, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x0, vlx4
    b.first .Lcbw_zero_weight_loop

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 0b: Zero-initialize grad_bias if not nullptr
    // ═══════════════════════════════════════════════════════════════════════
    cbz     x22, .Lcbw_skip_zero_bias

    mov     x8, #0
    whilelt pn9.s, x8, x9, vlx4

.Lcbw_zero_bias_loop:
    st1w    {z0.s-z3.s}, pn9, [x22, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x9, vlx4
    b.first .Lcbw_zero_bias_loop

.Lcbw_skip_zero_bias:

    // ═══════════════════════════════════════════════════════════════════════
    // Main accumulation loops
    //
    // Loop structure (7 levels):
    //   for n in 0..N:
    //     for oh in 0..H_out:
    //       for ow in 0..W_out:
    //         // Accumulate bias grad
    //         for kh in 0..KH:
    //           for kw in 0..KW:
    //             if (ih, iw) valid:
    //               for ic in 0..C_in:
    //                 // vectorized over C_out
    // ═══════════════════════════════════════════════════════════════════════

    // Outer loop: n (batch)
    mov     x0, #0                  // n = 0

.Lcbw_loop_n:
    cmp     x0, x23
    b.ge    .Lcbw_exit_streaming

    // Compute base pointers for this batch
    ldr     x16, [sp, #208]         // input_stride_n
    mul     x1, x0, x16             // n * input_stride_n
    add     x1, x19, x1, lsl #2     // input_n (bytes)

    ldr     x18, [sp, #224]         // go_stride_n
    mul     x2, x0, x18             // n * go_stride_n
    add     x2, x20, x2, lsl #2     // go_n (bytes)

    // Loop: oh (output height)
    mov     x3, #0                  // oh = 0

.Lcbw_loop_oh:
    ldr     x13, [sp, #184]         // H_out
    cmp     x3, x13
    b.ge    .Lcbw_next_n

    // Compute grad_out row base: go_oh = go_n + oh * go_stride_h
    ldr     x17, [sp, #216]         // go_stride_h
    mul     x4, x3, x17             // oh * go_stride_h
    add     x4, x2, x4, lsl #2      // go_oh (bytes)

    // Loop: ow (output width)
    mov     x5, #0                  // ow = 0

.Lcbw_loop_ow:
    ldr     x14, [sp, #192]         // W_out
    cmp     x5, x14
    b.ge    .Lcbw_next_oh

    // Compute pointer to grad_out[n, oh, ow, :]: go_ptr = go_oh + ow * C_out
    ldr     x9, [sp, #160]          // C_out
    mul     x6, x5, x9              // ow * C_out
    add     x6, x4, x6, lsl #2      // go_ptr (bytes)

    // ═══════════════════════════════════════════════════════════════════════
    // Accumulate grad_bias if not nullptr
    // ═══════════════════════════════════════════════════════════════════════
    cbz     x22, .Lcbw_skip_bias_accum

    mov     x8, #0
    ldr     x9, [sp, #160]          // C_out
    whilelt pn9.s, x8, x9, vlx4

.Lcbw_bias_accum_loop:
    ld1w    {z4.s-z7.s}, pn9/z, [x22, x8, lsl #2]   // grad_bias
    ld1w    {z0.s-z3.s}, pn9/z, [x6, x8, lsl #2]    // grad_out
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s
    st1w    {z4.s-z7.s}, pn9, [x22, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x9, vlx4
    b.first .Lcbw_bias_accum_loop

.Lcbw_skip_bias_accum:

    // ═══════════════════════════════════════════════════════════════════════
    // Loop over kernel positions (kh, kw)
    // ═══════════════════════════════════════════════════════════════════════
    mov     x7, #0                  // kh = 0

.Lcbw_loop_kh:
    cmp     x7, x27                 // KH is in callee-saved x27
    b.ge    .Lcbw_next_ow

    // Compute ih = oh * stride + kh - pad
    ldr     x10, [sp, #168]         // stride
    ldr     x11, [sp, #176]         // pad
    mul     x12, x3, x10            // oh * stride
    add     x12, x12, x7            // oh * stride + kh
    subs    x12, x12, x11           // ih = oh * stride + kh - pad

    // Bounds check: ih >= 0 && ih < H
    b.mi    .Lcbw_next_kh           // ih < 0
    cmp     x12, x24                // H is in callee-saved x24
    b.ge    .Lcbw_next_kh           // ih >= H

    // ih is valid, compute input row base
    ldr     x15, [sp, #200]         // input_stride_h
    mul     x13, x12, x15           // ih * input_stride_h
    add     x13, x1, x13, lsl #2    // input_ih (bytes) - save in x13

    // Loop: kw
    mov     x8, #0                  // kw = 0

.Lcbw_loop_kw:
    cmp     x8, x28                 // KW is in callee-saved x28
    b.ge    .Lcbw_next_kh

    // Compute iw = ow * stride + kw - pad
    ldr     x10, [sp, #168]         // stride
    ldr     x11, [sp, #176]         // pad
    mul     x14, x5, x10            // ow * stride
    add     x14, x14, x8            // ow * stride + kw
    subs    x14, x14, x11           // iw = ow * stride + kw - pad

    // Bounds check: iw >= 0 && iw < W
    b.mi    .Lcbw_next_kw           // iw < 0
    cmp     x14, x25                // W is in callee-saved x25
    b.ge    .Lcbw_next_kw           // iw >= W

    // (ih, iw) is valid
    // Compute input position pointer: input_ptr = input_ih + iw * C_in
    mul     x14, x14, x26           // iw * C_in (C_in in callee-saved x26)
    add     x14, x13, x14, lsl #2   // input_ptr (bytes) - keep in x14

    // Compute grad_weight base for [kh, kw, :, :]
    // offset = (kh*KW + kw) * C_in * C_out
    ldr     x9, [sp, #160]          // C_out
    mul     x15, x7, x28            // kh * KW
    add     x15, x15, x8            // kh * KW + kw
    mul     x15, x15, x26           // (kh*KW + kw) * C_in
    mul     x15, x15, x9            // (kh*KW + kw) * C_in * C_out
    add     x15, x21, x15, lsl #2   // gw_base (bytes) - keep in x15

    // ═══════════════════════════════════════════════════════════════════════
    // Inner loop: accumulate over input channels (ic)
    // Use x9 = C_out (already loaded), x10 = ic counter, x11 = gw_ptr, x12 = oc offset
    // ═══════════════════════════════════════════════════════════════════════
    mov     x10, #0                 // ic = 0

.Lcbw_loop_ic:
    cmp     x10, x26                // C_in is in callee-saved x26
    b.ge    .Lcbw_next_kw

    // Load scalar input[n, ih, iw, ic] and broadcast
    ldr     w16, [x14, x10, lsl #2] // input_val (FP32 bits)
    dup     z31.s, w16              // broadcast to all lanes

    // Compute pointer to grad_weight[kh, kw, ic, :]
    // gw_ptr = gw_base + ic * C_out
    ldr     x9, [sp, #160]          // C_out
    mul     x11, x10, x9            // ic * C_out
    add     x11, x15, x11, lsl #2   // gw_ptr (bytes)

    // Vectorized accumulation over C_out
    mov     x12, #0                 // oc offset
    whilelt pn9.s, x12, x9, vlx4

.Lcbw_oc_accum_loop:
    // Load current grad_weight
    ld1w    {z4.s-z7.s}, pn9/z, [x11, x12, lsl #2]

    // Load grad_out (x6 is go_ptr, preserved through inner loops)
    ld1w    {z0.s-z3.s}, pn9/z, [x6, x12, lsl #2]

    // Accumulate: grad_weight += input_val * grad_out
    fmla    z4.s, p0/m, z0.s, z31.s
    fmla    z5.s, p0/m, z1.s, z31.s
    fmla    z6.s, p0/m, z2.s, z31.s
    fmla    z7.s, p0/m, z3.s, z31.s

    // Store updated grad_weight
    st1w    {z4.s-z7.s}, pn9, [x11, x12, lsl #2]

    incw    x12, all, mul #4
    whilelt pn9.s, x12, x9, vlx4
    b.first .Lcbw_oc_accum_loop

    add     x10, x10, #1            // ic++
    b       .Lcbw_loop_ic

.Lcbw_next_kw:
    add     x8, x8, #1              // kw++
    b       .Lcbw_loop_kw

.Lcbw_next_kh:
    add     x7, x7, #1              // kh++
    b       .Lcbw_loop_kh

.Lcbw_next_ow:
    add     x5, x5, #1              // ow++
    b       .Lcbw_loop_ow

.Lcbw_next_oh:
    add     x3, x3, #1              // oh++
    b       .Lcbw_loop_oh

.Lcbw_next_n:
    add     x0, x0, #1              // n++
    b       .Lcbw_loop_n

    // ═══════════════════════════════════════════════════════════════════════
    // EXIT STREAMING MODE
    // ═══════════════════════════════════════════════════════════════════════
.Lcbw_exit_streaming:
    smstop sm

    // ═══════════════════════════════════════════════════════════════════════
    // Epilogue: restore callee-saved registers
    // ═══════════════════════════════════════════════════════════════════════
.Lcbw_done:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     d8,  d9,  [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    ldp     x29, x30, [sp], #256
    ret
