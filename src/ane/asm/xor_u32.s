// xor_u32.s — Element-wise bitwise XOR via SME2 streaming SVE
//
// void xor_u32(const uint32_t *a, const uint32_t *b, uint32_t *c, long n)

.section __TEXT,__text,regular,pure_instructions
.global _xor_u32
.p2align 4

_xor_u32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]
    cbz     x3, .Ldone
    smstart sm
    ptrue   p0.s
    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4
.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]
    eor     z0.d, z0.d, z4.d
    eor     z1.d, z1.d, z5.d
    eor     z2.d, z2.d, z6.d
    eor     z3.d, z3.d, z7.d
    st1w    {z0.s-z3.s}, pn9, [x2, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x3, vlx4
    b.first .Lloop
    smstop
.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10,  d11,  [sp, #32]
    ldp     d12,  d13,  [sp, #48]
    ldp     d14,  d15,  [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
