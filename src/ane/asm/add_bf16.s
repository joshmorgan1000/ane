// add_bf16.s — Element-wise BF16 addition via SME2 streaming SVE
//
// void add_bf16(const __bf16 *a, const __bf16 *b, __bf16 *c, long n)
//
// Computes c[i] = a[i] + b[i] for i in [0, n).
//
// No native bf16 fadd on M4 (FEAT_SME_B16B16 absent), so we widen to fp32,
// add, and narrow back via bfcvt. Widening uses ld1h into .s (zero-extend)
// then LSL #16 to place bf16 bits into fp32 position.
//
// Processes 16 bf16 elements per iteration (limited by widening to .s).

.section __TEXT,__text,regular,pure_instructions
.global _add_bf16
.p2align 4

_add_bf16:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]

    cbz     x3, .Ldone

    smstart sm

    mov     x8, #0
    whilelt p0.s, x8, x3

.Lloop:
    ld1h    {z0.s}, p0/z, [x0, x8, lsl #1]     // a: bf16 → zero-extended .s
    ld1h    {z1.s}, p0/z, [x1, x8, lsl #1]     // b: bf16 → zero-extended .s
    lsl     z0.s, z0.s, #16                      // → fp32
    lsl     z1.s, z1.s, #16                      // → fp32
    fadd    z0.s, p0/m, z0.s, z1.s              // fp32 add
    bfcvt   z0.h, p0/m, z0.s                    // fp32 → bf16 (rounded)
    st1h    {z0.s}, p0, [x2, x8, lsl #1]       // store low .h of each .s
    incw    x8
    whilelt p0.s, x8, x3
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10,  d11,  [sp, #32]
    ldp     d12,  d13,  [sp, #48]
    ldp     d14,  d15,  [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
