// argmax_combine_fp32.s — Combines worker argmax results to find global maximum
//
// long argmax_combine_fp32(const float* input, const long* worker_indices,
//                          long num_workers)
//
// AAPCS: x0=input, x1=worker_indices, x2=num_workers
// Returns: x0 = global index of maximum element
//
// Each worker_indices[i] contains the index (in input) of the max element
// in that worker's range. This kernel finds which worker has the global max
// by comparing the values at those indices.
// Pure scalar implementation (no SME streaming needed).

.section __TEXT,__text,regular,pure_instructions
.global _argmax_combine_fp32
.p2align 4

_argmax_combine_fp32:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Return 0 if no workers
    cbz     x2, .Lcombine_zero

    // Initialize: load first worker's value and index
    ldr     x9, [x1]                // first worker's index
    ldr     s0, [x0, x9, lsl #2]    // max_value = input[first_idx]
    mov     x10, x9                 // max_index = first_idx

    // Early exit if only one worker
    cmp     x2, #1
    b.le    .Lcombine_done

    // Iterate through remaining workers
    mov     x8, #1

.Lcombine_loop:
    cmp     x8, x2
    b.ge    .Lcombine_done

    // Load worker_indices[i]
    ldr     x12, [x1, x8, lsl #3]

    // Load input[worker_indices[i]]
    ldr     s1, [x0, x12, lsl #2]

    // Compare: if s1 > s0, update max
    fcmp    s1, s0
    b.le    .Lcombine_no_update     // if s1 <= s0, skip update

    // s1 > s0: update max value and index
    fmov    s0, s1
    mov     x10, x12

.Lcombine_no_update:
    add     x8, x8, #1
    b       .Lcombine_loop

.Lcombine_done:
    mov     x0, x10
    ldp     x29, x30, [sp], #16
    ret

.Lcombine_zero:
    mov     x0, #0
    ldp     x29, x30, [sp], #16
    ret
