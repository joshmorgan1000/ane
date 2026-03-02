// global_max_pool_fp32.s — Global max pooling over spatial dimensions (H*W)
//
// void global_max_pool_fp32(const float* input, float* output,
//                           long batch, long spatial, long channels)
//
// Input:  [batch, spatial, channels] flattened
// Output: [batch, channels]
//
// For each batch b and each spatial position s in [0, spatial),
// compute max over s for each channel c.
//
// AAPCS64: x0=input, x1=output, x2=batch, x3=spatial, x4=channels

.section __TEXT,__text,regular,pure_instructions
.global _global_max_pool_fp32
.p2align 4

_global_max_pool_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     d8,  d9,  [sp, #64]
    stp     d10, d11, [sp, #80]
    stp     d12, d13, [sp, #96]
    // [sp, #112] is scratch for constants

    // Early exit checks
    cbz     x2, .Ldone                  // batch == 0
    cbz     x3, .Ldone                  // spatial == 0
    cbz     x4, .Ldone                  // channels == 0

    // Save parameters
    mov     x19, x2                     // batch
    mov     x20, x3                     // spatial
    mov     x21, x4                     // channels
    mov     x22, x0                     // input
    mov     x23, x1                     // output

    // Stride for one spatial position: channels * sizeof(float)
    lsl     x24, x21, #2                // spatial_stride = channels * 4

    smstart sm

    ptrue   p0.s

    // Load -inf constant and broadcast to z12-z15
    adr     x9, .Lneginf
    ld1rw   {z12.s}, p0/z, [x9]
    mov     z13.d, z12.d
    mov     z14.d, z12.d
    mov     z15.d, z12.d

    // Outer loop: for each batch
    mov     x10, #0                     // batch index

.Lbatch_loop:
    cmp     x10, x19
    b.ge    .Ldone_sme

    // Compute input/output base pointers for this batch
    mul     x11, x10, x21
    mul     x11, x11, x20               // input_base_offset = b * spatial * channels
    lsl     x11, x11, #2                // in bytes
    add     x25, x22, x11               // input_base = input + offset
    mul     x12, x10, x21
    lsl     x12, x12, #2                // output_base_offset = b * channels (in bytes)
    add     x26, x23, x12               // output_base = output + offset

    // Initialize output to -inf for this batch
    mov     x13, #0                     // channel offset
    whilelt pn9.s, x13, x21, vlx4

.Linit_loop:
    st1w    {z12.s, z13.s, z14.s, z15.s}, pn9, [x26, x13, lsl #2]
    incw    x13, all, mul #4
    whilelt pn9.s, x13, x21, vlx4
    b.first .Linit_loop

    // Inner loop: for each spatial position
    mov     x14, #0                     // spatial index

.Lspatial_loop:
    cmp     x14, x20
    b.ge    .Lbatch_next

    // Compute input address for this spatial position
    mul     x15, x14, x21
    lsl     x15, x15, #2                // offset = s * channels * 4
    add     x27, x25, x15               // addr = input_base + offset

    // Channel loop: process all channels for this spatial position
    mov     x13, #0                     // channel offset
    whilelt pn9.s, x13, x21, vlx4

.Lchannel_loop:
    // Load input[b,s,c:c+64]
    ld1w    {z0.s, z1.s, z2.s, z3.s}, pn9/z, [x27, x13, lsl #2]
    // Load current max output[b,c:c+64]
    ld1w    {z4.s, z5.s, z6.s, z7.s}, pn9/z, [x26, x13, lsl #2]
    // Compute fmax
    fmax    z4.s, p0/m, z4.s, z0.s
    fmax    z5.s, p0/m, z5.s, z1.s
    fmax    z6.s, p0/m, z6.s, z2.s
    fmax    z7.s, p0/m, z7.s, z3.s
    // Store back to output
    st1w    {z4.s, z5.s, z6.s, z7.s}, pn9, [x26, x13, lsl #2]

    incw    x13, all, mul #4
    whilelt pn9.s, x13, x21, vlx4
    b.first .Lchannel_loop

.Lspatial_next:
    add     x14, x14, #1
    b       .Lspatial_loop

.Lbatch_next:
    add     x10, x10, #1
    b       .Lbatch_loop

.Ldone_sme:
    smstop

.Ldone:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     d8,  d9,  [sp, #64]
    ldp     d10, d11, [sp, #80]
    ldp     d12, d13, [sp, #96]
    ldp     x29, x30, [sp], #112
    ret

.p2align 2
.Lneginf:
    .long  0xFF800000  // -inf
