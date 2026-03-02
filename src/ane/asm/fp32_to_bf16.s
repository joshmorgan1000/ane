// fp32_to_bf16.s — FP32 to BF16 conversion via SME2 streaming SVE
//
// void fp32_to_bf16(const float *input, uint16_t *output, long n)
//
// Converts fp32 to bf16 by truncating low 16 bits (right shift 16).
// vlx4 loop: load 64 floats, shift, narrow to 64 halfwords, store.

.section __TEXT,__text,regular,pure_instructions
.global _fp32_to_bf16
.p2align 4

_fp32_to_bf16:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]
    cbz     x2, .Ldone
    smstart sm
    ptrue   p0.s

    mov     x8, #0              // element index
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    // Load 4 vectors of FP32 (64 elements total)
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]

    // BF16 = upper 16 bits of FP32 → logical shift right by 16
    lsr     z0.s, p0/m, z0.s, #16
    lsr     z1.s, p0/m, z1.s, #16
    lsr     z2.s, p0/m, z2.s, #16
    lsr     z3.s, p0/m, z3.s, #16

    // Narrow pairs: z0,z1 → z4 and z2,z3 → z5
    // uzp1 z.h takes even halfword lanes from both sources
    uzp1    z4.h, z0.h, z1.h
    uzp1    z5.h, z2.h, z3.h

    // Store 2 vectors of BF16 (64 halfwords total)
    // Need halfword predicates for the narrowed output
    // First 32 elements
    whilelt p1.h, x8, x2
    st1h    {z4.h}, p1, [x1, x8, lsl #1]

    // Second 32 elements (offset by 32)
    add     x9, x8, #32
    whilelt p2.h, x9, x2
    st1h    {z5.h}, p2, [x1, x9, lsl #1]

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
