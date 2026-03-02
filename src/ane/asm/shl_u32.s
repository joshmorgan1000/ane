// shl_u32.s — Element-wise left shift via SME2 streaming SVE
//
// void shl_u32(const uint32_t *input, uint32_t *output, int shift, long n)
//
// Computes output[i] = input[i] << shift for i in [0, n).
// AAPCS: x0=input, x1=output, w2=shift, x3=n

.section __TEXT,__text,regular,pure_instructions
.global _shl_u32
.p2align 4

_shl_u32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]
    cbz     x3, .Ldone
    smstart sm
    ptrue   p0.s
    mov     z8.s, w2            // broadcast shift amount
    mov     x8, #0
    whilelt pn9.s, x8, x3, vlx4
.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]
    lsl     z0.s, p0/m, z0.s, z8.s
    lsl     z1.s, p0/m, z1.s, z8.s
    lsl     z2.s, p0/m, z2.s, z8.s
    lsl     z3.s, p0/m, z3.s, z8.s
    st1w    {z0.s-z3.s}, pn9, [x1, x8, lsl #2]
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
