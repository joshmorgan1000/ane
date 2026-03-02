// concat_fp32.s — FP32 concatenation via SME2 streaming SVE
//
// void concat_fp32(const float *a, const float *b, float *out, long na, long nb)
// AAPCS: x0=a, x1=b, x2=out, x3=na, x4=nb
//
// Concatenates a[0..na-1] then b[0..nb-1] into out contiguously.
// Two consecutive copy loops: copy a, then copy b.
//
// Processing: 4 vectors (64 elements) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _concat_fp32
.p2align 4

_concat_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    // Early exit if both arrays are empty
    cbz     x3, .Lcopy_b
    cbz     x4, .Lcopy_a_only

    smstart sm

    // ===== Copy a[0..na-1] to out[0..na-1] =====
.Lcopy_a_only:
    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4

.Lcopy_a_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lcopy_a_loop

    // ===== Copy b[0..nb-1] to out[na..na+nb-1] =====
.Lcopy_b:
    cbz     x4, .Ldone

    mov     x9, #0                // b read offset
    mov     x10, x3               // out write offset = na
    whilelt pn9.s, x9, x4, vlx4

.Lcopy_b_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x1, x9, lsl #2]
    st1w    {z0.s-z3.s}, pn9, [x2, x10, lsl #2]
    incw    x9, all, mul #4
    add     x10, x10, #64         // increment out offset by 64 elements (256 bytes / 4 bytes)
    whilelt pn9.s, x9, x4, vlx4
    b.first .Lcopy_b_loop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
