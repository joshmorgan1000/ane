// reduce_max_combine_fp32.s — Combines partial maxes from multiple workers
//
// float reduce_max_combine_fp32(const float* partials, long num_workers)
//
// Each worker produces a single float partial max.
// This kernel finds the max of all partial values and returns the final scalar max.
// partials: array of num_workers floats

.section __TEXT,__text,regular,pure_instructions
.global _reduce_max_combine_fp32
.p2align 4

_reduce_max_combine_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8, d9, [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]
    // #80 reserved for scalar result preservation across smstop

    // Return -inf if no workers
    cbz     x1, .Lcombine_neg_inf

    smstart sm

    ptrue   p0.s

    // Load all partials into a vector
    mov     x8, #0
    whilelt p1.s, x8, x1

    // Load the partials (predicated for num_workers elements)
    ld1w    {z0.s}, p1/z, [x0]

    // Horizontal max to scalar
    fmaxv   s0, p1, z0.s
    str     s0, [sp, #80]      // use reserved frame slot (not SP-modifying)

    smstop

    ldr     s0, [sp, #80]
    b       .Lcombine_done

.Lcombine_neg_inf:
    adr     x9, .Lneg_inf
    ldr     s0, [x9]
    b       .Lcombine_done

.Lcombine_done:
    ldp     d8, d9, [sp, #16]
    ldp     d10,  d11,  [sp, #32]
    ldp     d12,  d13,  [sp, #48]
    ldp     d14,  d15,  [sp, #64]
    ldp     x29, x30, [sp], #96
    ret

.p2align 2
.Lneg_inf:
    .long   0xFF800000  // -infinity
