// unzip_fp32.s — Deinterleave FP32 array into two arrays via SME2 streaming SVE
//
// void unzip_fp32(const float *input, float *a, float *b, long n)
//
// Given input = [a0,b0,a1,b1,...], produces a[i] and b[i] for i in [0, n).
// Input length is 2*n, output arrays are each n elements.
//
// Uses uzp1/uzp2 to extract even/odd elements. Each iteration loads
// 2*SVLs input elements (32 on M4) and produces SVLs elements for each output.

.section __TEXT,__text,regular,pure_instructions
.global _unzip_fp32
.p2align 4

_unzip_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x3, .Ldone

    smstart sm

    cntw    x8                  // SVLs (16 on M4)
    lsl     x9, x3, #1         // 2*n = total input elements
    mov     x10, #0             // output index
    mov     x11, #0             // input index

.Lloop:
    whilelt p1.s, x10, x3
    b.none  .Lexit

    // Load 2 vectors from interleaved input
    whilelt p2.s, x11, x9
    ld1w    {z0.s}, p2/z, [x0, x11, lsl #2]
    add     x11, x11, x8

    whilelt p3.s, x11, x9
    ld1w    {z1.s}, p3/z, [x0, x11, lsl #2]
    add     x11, x11, x8

    // Deinterleave: even elements → a, odd elements → b
    uzp1    z2.s, z0.s, z1.s       // [a0,a1,a2,...,a15]
    uzp2    z3.s, z0.s, z1.s       // [b0,b1,b2,...,b15]

    // Store with output predicate (handles tail)
    st1w    {z2.s}, p1, [x1, x10, lsl #2]     // a
    st1w    {z3.s}, p1, [x2, x10, lsl #2]     // b

    incw    x10
    b       .Lloop

.Lexit:
    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
