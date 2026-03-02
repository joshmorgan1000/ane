// relu_bf16.s — BF16 ReLU via SME2 streaming SVE
//
// void relu_bf16(const __bf16 *input, __bf16 *output, long n)
//
// Computes output[i] = max(input[i], 0.0) for i in [0, n).
// For in-place operation, pass the same pointer for input and output.
//
// Sign-bit trick: ASR #15 extracts sign → 0x0000 (positive) or 0xFFFF
// (negative). BIC clears all bits where mask is set → zeros negatives,
// preserves positives. 2 instructions per vector, no comparisons.
//
// Processes 128 bf16 values (256 bytes) per loop iteration on M4.

.section __TEXT,__text,regular,pure_instructions
.global _relu_bf16
.p2align 4

_relu_bf16:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10,  d11,  [sp, #32]
    stp     d12,  d13,  [sp, #48]
    stp     d14,  d15,  [sp, #64]

    cbz     x2, .Ldone

    smstart sm

    mov     x8, #0
    whilelt pn9.h, x8, x2, vlx4

.Lloop:
    ld1h    {z0.h-z3.h}, pn9/z, [x0, x8, lsl #1]

    // Extract sign masks: 0x0000 for positive, 0xFFFF for negative
    asr     z4.h, z0.h, #15
    asr     z5.h, z1.h, #15
    asr     z6.h, z2.h, #15
    asr     z7.h, z3.h, #15

    // Bit-clear: zero out negative values, preserve positive
    bic     z0.d, z0.d, z4.d
    bic     z1.d, z1.d, z5.d
    bic     z2.d, z2.d, z6.d
    bic     z3.d, z3.d, z7.d

    st1h    {z0.h-z3.h}, pn9, [x1, x8, lsl #1]
    inch    x8, all, mul #4
    whilelt pn9.h, x8, x2, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10,  d11,  [sp, #32]
    ldp     d12,  d13,  [sp, #48]
    ldp     d14,  d15,  [sp, #64]
    ldp     x29, x30, [sp], #80
    ret
