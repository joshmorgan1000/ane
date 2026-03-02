// reduce_max_fp32.s — Max reduction via SME2 streaming SVE
//
// float reduce_max_fp32(const float *input, long n)
//
// Returns max(input[i]) for i in [0, n).
//
// Processes 4 vectors with 4 accumulators, plus single-vector tail for correctness.

.section __TEXT,__text,regular,pure_instructions
.global _reduce_max_fp32
.p2align 4

_reduce_max_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8,  d9,  [sp, #32]
    stp     d10, d11, [sp, #48]
    stp     d12, d13, [sp, #64]
    stp     d14, d15, [sp, #80]
    // [sp, #96] = scratch for result

    cmp     x1, #0
    b.le    .Lzero

    mov     x19, x1             // save n

    smstart sm

    ptrue   p0.s

    // Load -inf for initial max
    adr     x9, .Lneginf
    ld1rw   {z8.s}, p0/z, [x9]
    mov     z9.d, z8.d
    mov     z10.d, z8.d
    mov     z11.d, z8.d

    // Compute aligned count
    cntw    x10                 // VL
    lsl     x11, x10, #2        // 4*VL
    udiv    x12, x19, x11
    mul     x20, x12, x11       // aligned count

    mov     x8, #0
    cmp     x8, x20
    b.ge    .Ltail

    ptrue   pn9.s

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    fmax    z8.s, p0/m, z8.s, z0.s
    fmax    z9.s, p0/m, z9.s, z1.s
    fmax    z10.s, p0/m, z10.s, z2.s
    fmax    z11.s, p0/m, z11.s, z3.s
    incw    x8, all, mul #4
    cmp     x8, x20
    b.lt    .Lloop

.Ltail:
    // Single-vector cleanup for remaining elements
    whilelt p1.s, x8, x19
    b.none  .Lreduce

.Ltail_loop:
    ld1w    {z0.s}, p1/z, [x0, x8, lsl #2]
    fmax    z8.s, p1/m, z8.s, z0.s
    incw    x8
    whilelt p1.s, x8, x19
    b.first .Ltail_loop

.Lreduce:
    // Tree-reduce 4 accumulators
    fmax    z8.s, p0/m, z8.s, z9.s
    fmax    z10.s, p0/m, z10.s, z11.s
    fmax    z8.s, p0/m, z8.s, z10.s
    fmaxv   s0, p0, z8.s
    str     s0, [sp, #96]

    smstop

    ldr     s0, [sp, #96]
    b       .Ldone

.Lzero:
    adr     x9, .Lneginf
    ldr     s0, [x9]

.Ldone:
    ldp     x19, x20, [sp, #16]
    ldp     d8,  d9,  [sp, #32]
    ldp     d10, d11, [sp, #48]
    ldp     d12, d13, [sp, #64]
    ldp     d14, d15, [sp, #80]
    ldp     x29, x30, [sp], #112
    ret

.p2align 2
.Lneginf:
    .long  0xFF800000

