// extract_below_u16.s
// ARM SME2/SVE2 kernel for stream compaction of u16 distance array
// Extracts indices where distance < threshold into output u32 array
//
// long extract_below_u16(
//     const uint16_t* distances,  // x0
//     long n,                     // x1
//     long threshold,             // x2 (u16 threshold value, passed as long)
//     uint32_t* out_indices       // x3 (output: indices where dist < threshold)
// );
// Returns: count of extracted candidates in x0

.section __TEXT,__text,regular,pure_instructions
.global _extract_below_u16
.p2align 4

_extract_below_u16:
    // Prologue: save frame pointer and link register
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8,  d9,  [sp, #32]
    stp     d10,  d11,  [sp, #48]
    stp     d12,  d13,  [sp, #64]
    stp     d14,  d15,  [sp, #80]

    // Early exit if n == 0
    cbz     x1, .Lreturn_zero

    // Save input parameters to callee-saved registers
    mov     x19, x0                     // distances pointer (const)
    mov     x20, x1                     // n (total elements)

    // Initialize output count
    mov     x4, #0                      // x4 = output count

    smstart sm

    // Set up predicates and broadcast threshold
    ptrue   p0.h                        // always-true predicate for halfword
    dup     z31.h, w2                   // broadcast threshold to all lanes

    // Main vectorized loop: process 32 u16 elements per iteration
    mov     x8, #0                      // x8 = scan offset (element index)

.Lloop_vector:
    // Check if we have at least 32 elements remaining
    add     x9, x8, #32
    cmp     x9, x20
    b.gt    .Ltail_setup

    // Load 32 u16 elements (64 bytes, one full z-register at .h granularity)
    ld1h    {z0.h}, p0/z, [x19, x8, lsl #1]

    // Compare: p1 = (distance < threshold), unsigned comparison
    cmplo   p1.h, p0/z, z0.h, z31.h

    // Count matches
    cntp    x9, p0, p1.h

    // Fast skip: if no matches, continue to next chunk
    cbz     x9, .Lnext_vector

    // Matches exist: use scalar extraction loop
    // We're already in streaming mode - scalar loads work fine
    mov     x10, #0                     // x10 = inner loop counter (0..31)

.Lextract_loop:
    // Load single u16 from current chunk
    add     x11, x8, x10                // x11 = absolute index
    ldrh    w12, [x19, x11, lsl #1]     // w12 = distances[x11]

    // Compare against threshold
    cmp     w12, w2
    b.hs    .Lextract_next              // skip if >= threshold (unsigned)

    // Store absolute index to output
    str     w11, [x3, x4, lsl #2]       // out_indices[x4] = x11
    add     x4, x4, #1                  // increment output count

.Lextract_next:
    add     x10, x10, #1                // inner loop counter++
    cmp     x10, #32
    b.lt    .Lextract_loop

.Lnext_vector:
    add     x8, x8, #32                 // advance by 32 elements
    b       .Lloop_vector

.Ltail_setup:
    // Handle remaining elements (< 32)
    cmp     x8, x20
    b.ge    .Ldone                      // no tail elements

    // Create predicate for remaining elements
    whilelt p2.h, x8, x20

    // Load tail elements
    ld1h    {z0.h}, p2/z, [x19, x8, lsl #1]

    // Compare: p1 = (distance < threshold), masked by p2
    cmplo   p1.h, p2/z, z0.h, z31.h

    // Count matches in tail
    cntp    x9, p2, p1.h

    // Skip if no matches
    cbz     x9, .Ldone

    // Extract matches from tail
    // Compute actual tail size
    sub     x10, x20, x8                // x10 = tail_size
    mov     x11, #0                     // x11 = inner loop counter

.Ltail_extract_loop:
    cmp     x11, x10
    b.ge    .Ldone

    // Load single u16 from tail
    add     x12, x8, x11                // x12 = absolute index
    ldrh    w13, [x19, x12, lsl #1]     // w13 = distances[x12]

    // Compare against threshold
    cmp     w13, w2
    b.hs    .Ltail_extract_next         // skip if >= threshold

    // Store absolute index to output
    str     w12, [x3, x4, lsl #2]       // out_indices[x4] = x12
    add     x4, x4, #1                  // increment output count

.Ltail_extract_next:
    add     x11, x11, #1
    b       .Ltail_extract_loop

.Ldone:
    smstop

    // Move output count to return register
    mov     x0, x4

    // Epilogue: restore callee-saved registers
    ldp     x19, x20, [sp, #16]
    ldp     d8,  d9,  [sp, #32]
    ldp     d10,  d11,  [sp, #48]
    ldp     d12,  d13,  [sp, #64]
    ldp     d14,  d15,  [sp, #80]
    ldp     x29, x30, [sp], #96
    ret

.Lreturn_zero:
    mov     x0, #0
    ldp     d14, d15, [sp, #80]
    ldp     d12, d13, [sp, #64]
    ldp     d10, d11, [sp, #48]
    ldp     d8,  d9,  [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret
