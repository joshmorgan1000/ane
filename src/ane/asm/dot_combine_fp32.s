// dot_combine_fp32.s — Combines partial dot products from multiple workers
//
// float dot_combine_fp32(const float* partials, long num_workers)
//
// Each worker produces a single float partial dot product.
// This kernel sums all partial values and returns the final scalar dot product.
// partials: array of num_workers floats

.section __TEXT,__text,regular,pure_instructions
.global _dot_combine_fp32
.p2align 4

_dot_combine_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8, d9, [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]
    // [sp, #80-87] reserved for scalar result save across smstop

    // Return 0 if no workers
    cbz     x1, .Lcombine_zero

    smstart sm

    ptrue   p0.s

    // Load all partials into a vector and sum
    mov     x8, #0
    whilelt p1.s, x8, x1

    // Load the partials (predicated for num_workers elements)
    ld1w    {z0.s}, p1/z, [x0]

    // Horizontal sum to scalar
    faddv   s0, p1, z0.s
    str     s0, [sp, #80]      // save result BEFORE smstop (fixed offset, no SP modification)

    smstop

    ldr     s0, [sp, #80]      // reload result AFTER smstop
    b       .Lcombine_done

.Lcombine_zero:
    fmov    s0, wzr

.Lcombine_done:
    ldp     d8, d9, [sp, #16]
    ldp     d10,  d11,  [sp, #32]
    ldp     d12,  d13,  [sp, #48]
    ldp     d14,  d15,  [sp, #64]
    ldp     x29, x30, [sp], #96
    ret
