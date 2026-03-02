// bf16_to_fp32.s — BF16 to FP32 conversion via SME2 streaming SVE
//
// void bf16_to_fp32(const uint16_t *input, float *output, long n)
//
// Converts bf16 to fp32 by left-shifting 16 bits into upper half.
// vlx4 loop: load 128 halfwords, widen to 64 words, shift left 16, store 64 floats.

.section __TEXT,__text,regular,pure_instructions
.global _bf16_to_fp32
.p2align 4

_bf16_to_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]

    cbz     x2, .Ldone

    smstart sm

    ptrue   p0.s
    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    // Load 64 BF16 values as halfwords (need to load with .s predicate for element count)
    // We use individual loads with scalar predicates since multi-vector ld1h requires
    // matching container types
    whilelt p1.s, x8, x2
    ld1h    {z0.s}, p1/z, [x0, x8, lsl #1]

    add     x9, x8, #16
    whilelt p2.s, x9, x2
    ld1h    {z1.s}, p2/z, [x0, x9, lsl #1]

    add     x10, x8, #32
    whilelt p3.s, x10, x2
    ld1h    {z2.s}, p3/z, [x0, x10, lsl #1]

    add     x11, x8, #48
    whilelt p4.s, x11, x2
    ld1h    {z3.s}, p4/z, [x0, x11, lsl #1]

    // Shift left 16 to place in upper half of FP32
    lsl     z0.s, p0/m, z0.s, #16
    lsl     z1.s, p0/m, z1.s, #16
    lsl     z2.s, p0/m, z2.s, #16
    lsl     z3.s, p0/m, z3.s, #16

    // Store as FP32
    st1w    {z0.s-z3.s}, pn9, [x1, x8, lsl #2]

    incw    x8, all, mul #4
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
