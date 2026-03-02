// slice_fp32.s — FP32 contiguous slice extraction via SME2 streaming SVE
//
// void slice_fp32(const float *src, float *dst, long offset, long count)
//
// Copies count floats from src + offset to dst using streaming bandwidth.
// Equivalent to: memcpy(dst, src + offset, count * sizeof(float))
// but uses SME2 streaming mode for maximum memory throughput.
//
// AAPCS: x0=src, x1=dst, x2=offset, x3=count

.section __TEXT,__text,regular,pure_instructions
.global _slice_fp32
.p2align 4

_slice_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x3, .Ldone

    // Advance src by offset elements: src_ptr = src + offset
    add     x0, x0, x2, lsl #2     // x0 = src + offset * 4 (byte address)

    smstart sm

    mov     x8, #0                 // element counter
    whilelt pn9.s, x8, x3, vlx4

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    st1w    {z0.s-z3.s}, pn9, [x1, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
