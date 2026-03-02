// conv2d_fp32.s — FP32 2D convolution via im2col + SME2 FMOPA
//
// void conv2d_fp32(const float* input, const float* weight, float* output,
//                  long N, long H, long W, long C_in, long KH,
//                  long KW, long C_out, long stride, long pad)
//
// AAPCS: x0=input, x1=weight, x2=output, x3=N, x4=H, x5=W, x6=C_in, x7=KH
// Stack: [sp+FRAME_SIZE+0]=KW, [sp+FRAME_SIZE+8]=C_out,
//        [sp+FRAME_SIZE+16]=stride, [sp+FRAME_SIZE+24]=pad
//
// Layout: input[N,H,W,C_in] NHWC, weight[KH,KW,C_in,C_out] HWIO, output[N,H_out,W_out,C_out] NHWC
//
// Architecture: im2col + FMOPA matmul
// - Phase 0: im2col transform (before smstart)
// - Phase 1-3: FMOPA matmul (identical to matmul_fp32.s)
// - Phase 4: cleanup

.section __TEXT,__text,regular,pure_instructions
.global _conv2d_fp32
.p2align 4

_conv2d_fp32:
    stp     x29, x30, [sp, #-256]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8,  d9,  [sp, #112]
    stp     d10, d11, [sp, #128]
    stp     d12, d13, [sp, #144]
    stp     d14, d15, [sp, #160]

    // Save input parameters to callee-saved registers
    mov     x19, x0                 // input
    mov     x20, x1                 // weight
    mov     x21, x2                 // output
    mov     x22, x3                 // N
    mov     x23, x4                 // H
    mov     x24, x5                 // W
    mov     x25, x6                 // C_in
    mov     x26, x7                 // KH

    // Load stack parameters
    ldr     x27, [sp, #256]         // KW
    ldr     x28, [sp, #264]         // C_out
    ldr     x8, [sp, #272]          // stride
    ldr     x9, [sp, #280]          // pad
    str     x8, [sp, #192]          // save stride
    str     x9, [sp, #200]          // save pad
    str     x21, [sp, #232]         // save output ptr (will be overwritten later)

    // Early exit on N==0
    cbz     x22, .Ldone

    // Compute H_out = (H + 2*pad - KH) / stride + 1
    lsl     x8, x9, #1              // 2*pad
    add     x8, x23, x8             // H + 2*pad
    sub     x8, x8, x26             // H + 2*pad - KH
    ldr     x10, [sp, #192]         // stride
    udiv    x8, x8, x10
    add     x8, x8, #1              // H_out
    str     x8, [sp, #176]          // save H_out

    // Compute W_out = (W + 2*pad - KW) / stride + 1
    ldr     x9, [sp, #200]          // pad
    lsl     x9, x9, #1              // 2*pad
    add     x9, x24, x9             // W + 2*pad
    sub     x9, x9, x27             // W + 2*pad - KW
    udiv    x9, x9, x10
    add     x9, x9, #1              // W_out
    str     x9, [sp, #184]          // save W_out

    // M = N * H_out * W_out
    mul     x10, x22, x8            // N * H_out
    mul     x10, x10, x9            // N * H_out * W_out
    str     x10, [sp, #208]         // save M

    // K = KH * KW * C_in
    mul     x11, x26, x27           // KH * KW
    mul     x11, x11, x25           // KH * KW * C_in
    str     x11, [sp, #216]         // save K

    // Allocate im2col buffer: M * K * 4 bytes
    mul     x0, x10, x11
    lsl     x0, x0, #2
    str     x0, [sp, #224]          // save im2col_size
    bl      _malloc
    str     x0, [sp, #96]           // save im2col_ptr

    // Zero im2col buffer
    ldr     x1, [sp, #224]          // im2col_size
    bl      _bzero

    // ====================================================================
    // Phase 0: im2col filling
    // ====================================================================
    // Nested loops: for n in [0,N), oh in [0,H_out), ow in [0,W_out),
    //                   kh in [0,KH), kw in [0,KW), ic in [0,C_in)
    //
    // m = n*H_out*W_out + oh*W_out + ow
    // k = (kh*KW + kw)*C_in + ic
    // ih = oh*stride + kh - pad
    // iw = ow*stride + kw - pad
    // if (0 <= ih < H && 0 <= iw < W):
    //     im2col[m*K + k] = input[n*H*W*C_in + ih*W*C_in + iw*C_in + ic]

    ldr     x15, [sp, #96]          // im2col_ptr
    ldr     x16, [sp, #176]         // H_out
    ldr     x17, [sp, #184]         // W_out
    ldr     x14, [sp, #216]         // K

    mov     x0, #0                  // n = 0

.Lim2col_n:
    cmp     x0, x22                 // n < N?
    b.ge    .Lim2col_done

    mov     x1, #0                  // oh = 0

.Lim2col_oh:
    cmp     x1, x16                 // oh < H_out?
    b.ge    .Lim2col_n_next

    mov     x2, #0                  // ow = 0

.Lim2col_ow:
    cmp     x2, x17                 // ow < W_out?
    b.ge    .Lim2col_oh_next

    // Compute m = n*H_out*W_out + oh*W_out + ow
    mul     x3, x0, x16             // n * H_out
    mul     x3, x3, x17             // n * H_out * W_out
    mul     x4, x1, x17             // oh * W_out
    add     x3, x3, x4
    add     x3, x3, x2              // m

    mov     x4, #0                  // kh = 0

.Lim2col_kh:
    cmp     x4, x26                 // kh < KH?
    b.ge    .Lim2col_ow_next

    mov     x5, #0                  // kw = 0

.Lim2col_kw:
    cmp     x5, x27                 // kw < KW?
    b.ge    .Lim2col_kh_next

    mov     x6, #0                  // ic = 0

.Lim2col_ic:
    cmp     x6, x25                 // ic < C_in?
    b.ge    .Lim2col_kw_next

    // Compute ih = oh*stride + kh - pad
    ldr     x7, [sp, #192]          // stride
    mul     x7, x1, x7              // oh * stride
    add     x7, x7, x4              // oh*stride + kh
    ldr     x8, [sp, #200]          // pad
    sub     x7, x7, x8              // ih

    // Compute iw = ow*stride + kw - pad
    ldr     x8, [sp, #192]          // stride
    mul     x8, x2, x8              // ow * stride
    add     x8, x8, x5              // ow*stride + kw
    ldr     x9, [sp, #200]          // pad
    sub     x8, x8, x9              // iw

    // Check bounds: 0 <= ih < H && 0 <= iw < W
    cmp     x7, #0
    b.lt    .Lim2col_ic_next
    cmp     x7, x23                 // ih < H?
    b.ge    .Lim2col_ic_next
    cmp     x8, #0
    b.lt    .Lim2col_ic_next
    cmp     x8, x24                 // iw < W?
    b.ge    .Lim2col_ic_next

    // Compute k = (kh*KW + kw)*C_in + ic
    mul     x9, x4, x27             // kh * KW
    add     x9, x9, x5              // kh*KW + kw
    mul     x9, x9, x25             // (kh*KW + kw) * C_in
    add     x9, x9, x6              // k

    // Compute im2col offset: m*K + k
    mul     x10, x3, x14            // m * K
    add     x10, x10, x9            // m*K + k
    lsl     x10, x10, #2            // byte offset

    // Compute input offset: n*H*W*C_in + ih*W*C_in + iw*C_in + ic
    mul     x11, x0, x23            // n * H
    mul     x11, x11, x24           // n * H * W
    mul     x11, x11, x25           // n * H * W * C_in
    mul     x12, x7, x24            // ih * W
    mul     x12, x12, x25           // ih * W * C_in
    add     x11, x11, x12
    mul     x12, x8, x25            // iw * C_in
    add     x11, x11, x12
    add     x11, x11, x6            // input_offset
    lsl     x11, x11, #2            // byte offset

    // Copy: im2col[m*K + k] = input[offset]
    ldr     w12, [x19, x11]
    ldr     x13, [sp, #96]          // im2col_ptr
    str     w12, [x13, x10]

.Lim2col_ic_next:
    add     x6, x6, #1              // ic++
    b       .Lim2col_ic

.Lim2col_kw_next:
    add     x5, x5, #1              // kw++
    b       .Lim2col_kw

.Lim2col_kh_next:
    add     x4, x4, #1              // kh++
    b       .Lim2col_kh

.Lim2col_ow_next:
    add     x2, x2, #1              // ow++
    b       .Lim2col_ow

.Lim2col_oh_next:
    add     x1, x1, #1              // oh++
    b       .Lim2col_oh

.Lim2col_n_next:
    add     x0, x0, #1              // n++
    b       .Lim2col_n

.Lim2col_done:

    // ====================================================================
    // Setup for matmul phases
    // ====================================================================
    // A = im2col (M x K)
    // B = weight (K x N_mat), N_mat = C_out
    // C = output (M x N_mat)

    ldr     x21, [sp, #96]          // x21 = im2col_ptr (A for matmul)
    mov     x22, x20                // x22 = weight (B for matmul)
    ldr     x23, [sp, #232]         // x23 = output (C for matmul) - loaded from earlier save
    ldr     x24, [sp, #208]         // M
    mov     x25, x28                // N_mat = C_out
    ldr     x26, [sp, #216]         // K

    // Enter streaming mode to get SVL
    smstart sm
    cntw    x15                     // SVLs (streaming)
    cntb    x17                     // SVLb (streaming)
    smstop  sm

    // K_mod = ceil(K / SVLb) * SVLb
    add     x27, x26, x17
    sub     x27, x27, #1
    udiv    x27, x27, x17
    mul     x27, x27, x17

    // A_mod size = ceil(M/SVLs)*SVLs * K_mod * 4
    add     x8, x24, x15
    sub     x8, x8, #1
    udiv    x8, x8, x15
    mul     x8, x8, x15
    mul     x28, x8, x27
    lsl     x28, x28, #2

    // B_mod size = ceil(N/(2*SVLs))*(2*SVLs) * K_mod * 4
    lsl     x9, x15, #1
    add     x8, x25, x9
    sub     x8, x8, #1
    udiv    x8, x8, x9
    mul     x8, x8, x9
    mul     x8, x8, x27
    lsl     x8, x8, #2
    str     x8, [sp, #240]          // save B_mod_size

    // Allocate A_mod
    mov     x0, x28
    bl      _malloc
    mov     x19, x0                 // x19 = A_mod

    mov     x0, x19
    mov     x1, x28
    bl      _bzero

    // Allocate B_mod
    ldr     x0, [sp, #240]
    bl      _malloc
    mov     x20, x0                 // x20 = B_mod

    mov     x0, x20
    ldr     x1, [sp, #240]
    bl      _bzero

    // ====================================================================
    // Phase 1: preprocess_r (weight → B_mod)
    // ====================================================================
    smstart

    cntb    x5
    lsl     x11, x25, #2            // N_mat * 4
    lsl     x17, x5, #1
    mul     x16, x27, x17

    mov     x15, #0
    ptrue   pn9.b

    mov     x8, x22                 // B (weight)
    mov     x9, x20                 // B_mod
    lsl     x10, x25, #2
    add     x10, x10, x22
    whilelt pn11.b, x8, x10, vlx2

.Lpp_r_N:
    mov     x7, x8
    mov     x12, x9
    mov     x6, xzr

.Lpp_r_K:
    cmp     x6, x26                 // k < K?
    b.ge    .Lpp_r_N_next

    ld1b    {z0.b-z1.b}, pn11/z, [x7]
    st1b    {z0.b-z1.b}, pn11, [x12]

    add     x7, x7, x11
    addvl   x12, x12, #2
    add     x6, x6, #1
    b       .Lpp_r_K

.Lpp_r_N_next:
    add     x9, x9, x16
    addvl   x8, x8, #2
    whilelt pn11.b, x8, x10, vlx2
    b.first .Lpp_r_N

    // ====================================================================
    // Phase 2: preprocess_l (A / im2col → A_mod via ZA transpose)
    // ====================================================================
    cntw    x5
    lsl     x18, x26, #2            // K * 4
    mul     x11, x5, x18
    mul     x15, x5, x27
    lsl     x2, x15, #2

    mul     x4, x5, x5
    lsl     x16, x4, #1
    add     x16, x16, x4
    cntb    x17

    mov     x28, #0
    mov     x8, x21                 // A (im2col)
    mov     x9, x19                 // A_mod
    whilelt p0.s, x28, x24          // M

.Lpp_l_M:
    mov     x7, x9
    mov     x10, x8
    add     x3, x8, x18
    whilelt pn12.b, x10, x3, vlx4
    mov     x13, #0
    mov     x14, x4
    lsl     x0, x4, #1
    mov     x1, x16

.Lpp_l_K:
    mov     x6, x10

    mov     w12, #0
.Lpp_l_load:
    psel    pn8, pn12, p0.b[w12, #0]
    psel    pn9, pn12, p0.b[w12, #4]
    ld1b    {z0.b-z3.b}, pn8/z, [x6]
    ld1b    {z4.b-z7.b}, pn9/z, [x6, x18]
    mova    za0h.b[w12, 0:3], {z0.b-z3.b}
    mova    za0h.b[w12, 4:7], {z4.b-z7.b}
    add     w12, w12, #8
    add     x6, x6, x18, lsl #1
    cmp     w12, w17
    b.mi    .Lpp_l_load

    mov     w12, #0
.Lpp_l_store:
    whilelt pn8.s, x13, x15, vlx4
    whilelt pn9.s, x14, x15, vlx4
    whilelt pn10.s, x0, x15, vlx4
    whilelt pn11.s, x1, x15, vlx4
    mova    {z0.s-z3.s}, za0v.s[w12, 0:3]
    mova    {z4.s-z7.s}, za1v.s[w12, 0:3]
    mova    {z8.s-z11.s}, za2v.s[w12, 0:3]
    mova    {z12.s-z15.s}, za3v.s[w12, 0:3]
    add     w12, w12, #4
    st1w    {z0.s-z3.s}, pn8, [x7, x13, lsl #2]
    st1w    {z4.s-z7.s}, pn9, [x7, x14, lsl #2]
    st1w    {z8.s-z11.s}, pn10, [x7, x0, lsl #2]
    st1w    {z12.s-z15.s}, pn11, [x7, x1, lsl #2]
    incw    x13, all, mul #4
    incw    x14, all, mul #4
    incw    x0, all, mul #4
    incw    x1, all, mul #4
    cmp     w12, w5
    b.mi    .Lpp_l_store

    add     x13, x13, x16
    add     x14, x14, x16
    add     x0, x0, x16
    add     x1, x1, x16
    addvl   x10, x10, #4
    whilelt pn12.b, x10, x3, vlx4
    b.first .Lpp_l_K

    add     x8, x8, x11
    add     x9, x9, x2

    incw    x28
    whilelt p0.s, x28, x24
    b.first .Lpp_l_M

    // ====================================================================
    // Phase 3: matmul_opt (A_mod × B_mod → C / output)
    // ====================================================================
    cntb    x6
    cntw    x15
    lsl     x11, x25, #2            // N_mat * 4
    mul     x21, x15, x25
    add     x2, x21, x25
    mul     x7, x27, x6
    lsl     x0, x24, #2             // M * 4
    mov     x3, x19                 // A_mod
    mov     x28, x23                // C (output)
    mov     x12, #0
    mov     x15, #0
    sub     w6, w6, #8
    ptrue   pn10.b
    whilelt p2.b, x12, x0

.Lmm_M:
    addvl   x12, x12, #1
    whilelt p3.b, x12, x0

    mov     x4, x20                 // B_mod
    mov     x22, x28
    mov     x13, #0
    add     x10, x3, x7
    add     x17, x3, x7
    addvl   x9, x17, #-1

    whilelt pn9.b, x13, x11, vlx2

.Lmm_N:
    mov     x8, x3
    mov     x16, x4
    mov     x23, x22

    pext    {p0.b, p1.b}, pn9[0]

    zero    {za}

.Lf_K_start:
    ld1b    {z1.b}, p2/z, [x8]
    whilelt pn10.b, x8, x17, vlx2
    ld1b    {z2.b-z3.b}, pn9/z, [x16]
    fmopa   za0.s, p2/m, p0/m, z1.s, z2.s
    ld1b    {z5.b}, p3/z, [x8, x7]
    addvl   x8, x8, #1

.Lf_Loop_K:
    fmopa   za2.s, p3/m, p0/m, z5.s, z2.s
    fmopa   za1.s, p2/m, p1/m, z1.s, z3.s
    psel    pn11, pn10, p3.s[w15, #0]
    ld1b    {z0.b-z1.b}, pn10/z, [x8]
    fmopa   za3.s, p3/m, p1/m, z5.s, z3.s
    ld1b    {z6.b-z7.b}, pn9/z, [x16, #2, mul vl]

    fmopa   za0.s, p2/m, p0/m, z0.s, z6.s
    ld1b    {z4.b-z5.b}, pn11/z, [x8, x7]

    fmopa   za2.s, p3/m, p0/m, z4.s, z6.s
    addvl   x16, x16, #4

    fmopa   za1.s, p2/m, p1/m, z0.s, z7.s

    fmopa   za3.s, p3/m, p1/m, z4.s, z7.s
    ld1b    {z2.b-z3.b}, pn9/z, [x16]

    fmopa   za0.s, p2/m, p0/m, z1.s, z2.s
    addvl   x8, x8, #2

    cmp     x8, x9
    b.mi    .Lf_Loop_K

    fmopa   za2.s, p3/m, p0/m, z5.s, z2.s
    fmopa   za1.s, p2/m, p1/m, z1.s, z3.s
    fmopa   za3.s, p3/m, p1/m, z5.s, z3.s
    addvl   x16, x16, #2

    cmp     x8, x10
    b.ge    .Lmm_store

.Lf_Ktail:
    ld1b    {z1.b}, p2/z, [x8]
    ld1b    {z2.b-z3.b}, pn9/z, [x16]
    fmopa   za0.s, p2/m, p0/m, z1.s, z2.s
    ld1b    {z14.b}, p3/z, [x8, x7]
    fmopa   za2.s, p3/m, p0/m, z14.s, z2.s
    fmopa   za1.s, p2/m, p1/m, z1.s, z3.s
    addvl   x16, x16, #2
    fmopa   za3.s, p3/m, p1/m, z14.s, z3.s

.Lmm_store:
    mov     w14, #0
    psel    pn8, pn9, p2.b[w14, 0]
    psel    pn11, pn9, p3.b[w14, 0]
    mova    {z0.b-z3.b}, za0h.b[w14, 0:3]
    st1w    {z0.s-z1.s}, pn8, [x23]
    st1w    {z2.s-z3.s}, pn11, [x23, x21, lsl #2]

.Lmm_store_loop:
    psel    pn8, pn9, p2.b[w14, 4]
    psel    pn11, pn9, p3.b[w14, 4]
    mova    {z0.b-z3.b}, za0h.b[w14, 4:7]
    st1w    {z0.s-z1.s}, pn8, [x23, x25, lsl #2]
    st1w    {z2.s-z3.s}, pn11, [x23, x2, lsl #2]

    add     x23, x23, x25, lsl #3
    add     w14, w14, #8

    psel    pn8, pn9, p2.b[w14, 0]
    psel    pn11, pn9, p3.b[w14, 0]
    mova    {z0.b-z3.b}, za0h.b[w14, 0:3]
    st1w    {z0.s-z1.s}, pn8, [x23]
    st1w    {z2.s-z3.s}, pn11, [x23, x21, lsl #2]
    cmp     w14, w6
    b.mi    .Lmm_store_loop

    psel    pn8, pn9, p2.b[w14, 4]
    psel    pn11, pn9, p3.b[w14, 4]
    mova    {z0.b-z3.b}, za0h.b[w14, 4:7]
    st1w    {z0.s-z1.s}, pn8, [x23, x25, lsl #2]
    st1w    {z2.s-z3.s}, pn11, [x23, x2, lsl #2]

    addvl   x22, x22, #2
    addvl   x13, x13, #2
    whilelt pn9.b, x13, x11, vlx2
    add     x4, x4, x7, lsl #1
    b.first .Lmm_N

    add     x3, x3, x7, lsl #1
    add     x28, x28, x21, lsl #3
    addvl   x12, x12, #1
    whilelt p2.b, x12, x0
    b.first .Lmm_M

    smstop

    // ====================================================================
    // Phase 4: Cleanup
    // ====================================================================
    // Free A_mod (x19)
    mov     x0, x19
    bl      _free

    // Free B_mod (x20)
    mov     x0, x20
    bl      _free

    // Free im2col buffer
    ldr     x0, [sp, #96]
    bl      _free

.Ldone:
    ldp     d8,  d9,  [sp, #112]
    ldp     d10, d11, [sp, #128]
    ldp     d12, d13, [sp, #144]
    ldp     d14, d15, [sp, #160]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     x29, x30, [sp], #256
    ret
