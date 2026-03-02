// not_u32.s — Element-wise bitwise NOT via SME2 streaming SVE
//
// void not_u32(const uint32_t *input, uint32_t *output, long n)

.section __TEXT,__text,regular,pure_instructions
.global _not_u32
.p2align 4

_not_u32:
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
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    not     z0.d, p0/m, z0.d
    not     z1.d, p0/m, z1.d
    not     z2.d, p0/m, z2.d
    not     z3.d, p0/m, z3.d
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
