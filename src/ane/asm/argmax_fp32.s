// argmax_fp32.s — Index of maximum element via SME2 streaming SVE
//
// int32_t argmax_fp32(const float *input, long n)
//
// AAPCS: x0=input, x1=n
// Returns: w0 = index of maximum element (0-based), 0 for empty array
//
// Uses 4-accumulator vlx4 processing (64 floats/iter on M4 Max).
// Tracks both max values and their indices; horizontal reduction at end.

.section __TEXT,__text,regular,pure_instructions
.global _argmax_fp32
.p2align 4

_argmax_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    stp     d14, d15, [sp, #96]

    cmp     x1, #0
    b.le    .Lzero

    mov     x19, x0             // save input
    mov     x20, x1             // save n

    smstart sm

    ptrue   p0.s

    // Load -inf for initialization
    adr     x9, .Lneginf
    ld1rw   {z20.s}, p0/z, [x9]

    // Initialize 4 max-value accumulators to -inf
    mov     z4.d, z20.d
    mov     z5.d, z20.d
    mov     z6.d, z20.d
    mov     z7.d, z20.d

    // Initialize 4 index accumulators to 0 (will be overwritten on first compare)
    mov     z8.d, #0
    mov     z9.d, #0
    mov     z10.d, #0
    mov     z11.d, #0

    // Build incrementing index template: [0, 1, 2, ..., VL-1]
    index   z24.s, #0, #1

    // VL constants
    cntw    x10                 // x10 = VL (floats per vector)
    lsl     x11, x10, #2       // x11 = 4*VL

    // Compute aligned count for vlx4 loop
    udiv    x12, x20, x11
    mul     x22, x12, x11      // x22 = aligned count

    mov     x8, #0
    cmp     x8, x22
    b.ge    .Ltail

    ptrue   pn9.s

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Build current index vectors for each of the 4 vectors
    mov     w9, w8
    dup     z12.s, w9
    add     z12.s, z12.s, z24.s     // indices for vector 0

    add     w9, w8, w10
    dup     z13.s, w9
    add     z13.s, z13.s, z24.s     // indices for vector 1

    add     w9, w9, w10
    dup     z14.s, w9
    add     z14.s, z14.s, z24.s     // indices for vector 2

    add     w9, w9, w10
    dup     z15.s, w9
    add     z15.s, z15.s, z24.s     // indices for vector 3

    // Compare and conditionally update max+index for each accumulator
    fcmgt   p1.s, p0/z, z0.s, z4.s
    sel     z4.s, p1, z0.s, z4.s
    sel     z8.s, p1, z12.s, z8.s

    fcmgt   p2.s, p0/z, z1.s, z5.s
    sel     z5.s, p2, z1.s, z5.s
    sel     z9.s, p2, z13.s, z9.s

    fcmgt   p3.s, p0/z, z2.s, z6.s
    sel     z6.s, p3, z2.s, z6.s
    sel     z10.s, p3, z14.s, z10.s

    fcmgt   p1.s, p0/z, z3.s, z7.s
    sel     z7.s, p1, z3.s, z7.s
    sel     z11.s, p1, z15.s, z11.s

    incw    x8, all, mul #4
    cmp     x8, x22
    b.lt    .Lloop

.Ltail:
    // Single-vector cleanup for remaining elements
    whilelt p1.s, x8, x20
    b.none  .Lreduce

.Ltail_loop:
    ld1w    {z0.s}, p1/z, [x19, x8, lsl #2]

    mov     w9, w8
    dup     z12.s, w9
    add     z12.s, z12.s, z24.s

    fcmgt   p2.s, p1/z, z0.s, z4.s
    sel     z4.s, p2, z0.s, z4.s
    sel     z8.s, p2, z12.s, z8.s

    incw    x8
    whilelt p1.s, x8, x20
    b.first .Ltail_loop

.Lreduce:
    // Merge 4 accumulators into z4/z8, comparing BEFORE merging to track indices
    // Merge acc1 into acc0
    fcmgt   p1.s, p0/z, z5.s, z4.s
    sel     z4.s, p1, z5.s, z4.s
    sel     z8.s, p1, z9.s, z8.s

    // Merge acc2 into acc0
    fcmgt   p1.s, p0/z, z6.s, z4.s
    sel     z4.s, p1, z6.s, z4.s
    sel     z8.s, p1, z10.s, z8.s

    // Merge acc3 into acc0
    fcmgt   p1.s, p0/z, z7.s, z4.s
    sel     z4.s, p1, z7.s, z4.s
    sel     z8.s, p1, z11.s, z8.s

    // Now z4 has per-lane max values, z8 has per-lane max indices
    // Find the horizontal maximum value
    fmaxv   s0, p0, z4.s

    // Broadcast max to all lanes for comparison
    mov     z0.s, s0

    // Find lanes that equal the global max
    fcmeq   p1.s, p0/z, z4.s, z0.s

    // Among matching lanes, find the one with the smallest index.
    // We want the FIRST true lane's index value from z8.
    // UMINV gives the minimum index among all matching lanes.
    // Set non-matching lanes to INT32_MAX so they lose the min
    mov     w9, #0x7FFFFFFF
    dup     z2.s, w9
    sel     z1.s, p1, z8.s, z2.s   // matching lanes keep index, others get MAX_INT
    uminv   s1, p0, z1.s           // s1 = minimum index among matching lanes

    // Save result before smstop
    fmov    w0, s1
    str     w0, [sp, #104]

    smstop

    ldr     w0, [sp, #104]
    b       .Ldone

.Lzero:
    mov     w0, #0

.Ldone:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     d8,  d9,  [sp, #48]
    ldp     d10, d11, [sp, #64]
    ldp     d12, d13, [sp, #80]
    ldp     d14, d15, [sp, #96]
    ldp     x29, x30, [sp], #112
    ret

.p2align 2
.Lneginf:
    .long   0xFF800000              // -inf
