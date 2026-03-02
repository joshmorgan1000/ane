// global_avg_pool_fp32.s — Global average pooling over spatial dimensions (H*W)
//
// void global_avg_pool_fp32(const float* input, float* output,
//                           long batch, long spatial, long channels)
//
// Input:  [batch, spatial, channels] flattened
// Output: [batch, channels]
//
// For each batch b and channel c, compute average over spatial positions:
// output[b,c] = (1/spatial) * sum_{s=0}^{spatial} input[b,s,c]
//
// AAPCS64: x0=input, x1=output, x2=batch, x3=spatial, x4=channels

.section __TEXT,__text,regular,pure_instructions
.global _global_avg_pool_fp32
.p2align 4

_global_avg_pool_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     d8,  d9,  [sp, #64]
    stp     d10, d11, [sp, #80]
    stp     d12, d13, [sp, #96]
    // [sp, #112] scratch for inv_spatial

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

    smstart sm

    ptrue   p0.s

    // Compute 1/spatial and broadcast
    scvtf   s8, x20                    // float(spatial)
    adr     x9, .Lone
    ld1rw   {z9.s}, p0/z, [x9]         // load 1.0
    fdiv    s8, s9, s8                 // inv_spatial = 1.0 / spatial
    mov     z13.s, s8                  // broadcast inv_spatial to all lanes

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

    // Initialize output to 0.0 for this batch
    mov     x14, #0
    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0

.Linit_loop:
    cmp     x14, x21
    b.ge    .Linit_done

    // Store zeros to output
    mov     x13, x14
    whilelt pn9.s, x13, x21, vlx4
    b.none  .Linit_done

.Linit_store:
    st1w    {z0.s, z1.s, z2.s, z3.s}, pn9, [x26, x13, lsl #2]
    incw    x13, all, mul #4
    whilelt pn9.s, x13, x21, vlx4
    b.first .Linit_store

.Linit_done:
    // Inner loop: for each spatial position
    mov     x15, #0                     // spatial index

.Lspatial_loop:
    cmp     x15, x20
    b.ge    .Lbatch_next

    // Compute input address for this spatial position
    mul     x16, x15, x21
    lsl     x16, x16, #2                // offset = s * channels * 4
    add     x27, x25, x16               // addr = input_base + offset

    // Channel loop: process all channels for this spatial position
    mov     x13, #0                     // channel offset
    whilelt pn9.s, x13, x21, vlx4

.Lchannel_loop:
    // Load input[b,s,c:c+64]
    ld1w    {z0.s, z1.s, z2.s, z3.s}, pn9/z, [x27, x13, lsl #2]
    // Load current sum output[b,c:c+64]
    ld1w    {z4.s, z5.s, z6.s, z7.s}, pn9/z, [x26, x13, lsl #2]
    // Accumulate
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s
    // Store back to output
    st1w    {z4.s, z5.s, z6.s, z7.s}, pn9, [x26, x13, lsl #2]

    incw    x13, all, mul #4
    whilelt pn9.s, x13, x21, vlx4
    b.first .Lchannel_loop

.Lspatial_next:
    add     x15, x15, #1
    b       .Lspatial_loop

.Lbatch_next:
    // Divide output[b,:] by spatial
    mov     x13, #0
    whilelt pn9.s, x13, x21, vlx4

.Ldiv_loop:
    ld1w    {z4.s, z5.s, z6.s, z7.s}, pn9/z, [x26, x13, lsl #2]
    fmul    z4.s, p0/m, z4.s, z13.s
    fmul    z5.s, p0/m, z5.s, z13.s
    fmul    z6.s, p0/m, z6.s, z13.s
    fmul    z7.s, p0/m, z7.s, z13.s
    st1w    {z4.s, z5.s, z6.s, z7.s}, pn9, [x26, x13, lsl #2]

    incw    x13, all, mul #4
    whilelt pn9.s, x13, x21, vlx4
    b.first .Ldiv_loop

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
.Lone:
    .long  0x3F800000  // 1.0
