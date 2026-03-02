// probe_za_fmla_vgx4.s — Probe for ZA multi-vector FMLA/FMLS (vgx4)
//
// int probe_za_fmla_vgx4(void)
// Returns 1 if both FMLA and FMLS za.s[w8, 0, vgx4] execute without SIGILL.
// Returns 0 if smstart itself works but should never reach here on failure
// (caller must install SIGILL handler).
//
// This is a standalone probe — the caller wraps it in a sigsetjmp/SIGILL handler.

.section __TEXT,__text,regular,pure_instructions
.global _probe_za_fmla_vgx4
.p2align 4

_probe_za_fmla_vgx4:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]

    smstart
    zero    {za}
    ptrue   p0.s
    mov     z0.d, #0
    mov     z1.d, #0
    mov     z2.d, #0
    mov     z3.d, #0
    mov     w8, #0

    // Probe FMLA za multi-vector element-wise
    fmla    za.s[w8, 0, vgx4], {z0.s-z3.s}, {z0.s-z3.s}

    // Probe FMLS za multi-vector element-wise
    fmls    za.s[w8, 0, vgx4], {z0.s-z3.s}, {z0.s-z3.s}

    smstop

    mov     w0, #1          // success
    ldp     d8,  d9,  [sp, #16]
    ldp     d10,  d11,  [sp, #32]
    ldp     d12,  d13,  [sp, #48]
    ldp     d14,  d15,  [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
