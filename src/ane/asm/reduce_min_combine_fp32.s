// reduce_min_combine_fp32.s — Combines partial mins from multiple workers
//
// float reduce_min_combine_fp32(const float* partials, long num_workers)
//
// Each worker produces a single float partial min.
// This kernel finds the min of all partial values and returns the final scalar min.
// partials: array of num_workers floats

.section __TEXT,__text,regular,pure_instructions
.global _reduce_min_combine_fp32
.p2align 4

_reduce_min_combine_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8, d9, [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]

    // Return +inf if no workers
    cbz     x1, .Lcombine_pos_inf

    smstart sm

    ptrue   p0.s

    // Load all partials into a vector
    mov     x8, #0
    whilelt p1.s, x8, x1

    // Load the partials (predicated for num_workers elements)
    ld1w    {z0.s}, p1/z, [x0]

    // Horizontal min to scalar
    fminv   s0, p1, z0.s
    str     s0, [sp, #72]      // use unused frame slot (within d14/d15 save area)

    smstop

    ldr     s0, [sp, #72]
    b       .Lcombine_done

.Lcombine_pos_inf:
    adr     x9, .Lpos_inf
    ldr     s0, [x9]
    b       .Lcombine_done

.Lcombine_done:
    ldp     d8, d9, [sp, #16]
    ldp     d10,  d11,  [sp, #32]
    ldp     d12,  d13,  [sp, #48]
    ldp     d14,  d15,  [sp, #64]
    ldp     x29, x30, [sp], #80
    ret

.p2align 2
.Lpos_inf:
    .long   0x7F800000  // +infinity
