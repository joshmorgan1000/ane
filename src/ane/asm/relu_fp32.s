// relu_fp32.s — FP32 ReLU via SME2 streaming SVE
//
// void relu_fp32(const float *input, float *output, long n)
//
// Computes output[i] = max(input[i], 0.0) for i in [0, n).
// For in-place operation, pass the same pointer for input and output.
//
// Uses streaming mode for 512-bit SVE vectors (64 bytes on M4) with
// 4-vector group operations (ld1w/fmax/st1w vlx4), processing
// 64 floats (256 bytes) per loop iteration.

.section __TEXT,__text,regular,pure_instructions
.global _relu_fp32
.p2align 4

_relu_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]

    cbz     x2, .Ldone              // early exit if n == 0

    smstart sm

    mov     z8.d, #0                // zero register for fmax broadcast
    mov     x8, #0                  // element counter

    whilelt pn9.s, x8, x2, vlx4    // predicate for 4-vector group

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    fmax    {z0.s-z3.s}, {z0.s-z3.s}, z8.s
    st1w    {z0.s-z3.s}, pn9, [x1, x8, lsl #2]
    incw    x8, all, mul #4         // x8 += 4*SVLs (64 on M4)
    whilelt pn9.s, x8, x2, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10,  d11,  [sp, #32]
    ldp     d12,  d13,  [sp, #48]
    ldp     d14,  d15,  [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
