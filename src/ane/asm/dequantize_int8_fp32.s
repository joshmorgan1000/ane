// dequantize_int8_fp32.s — INT8 to FP32 dequantization via SME2 streaming SVE
//
// void dequantize_int8_fp32(const int8_t *input, float *output, float scale, long n)
//
// Computes output[i] = (float)input[i] * scale
// AAPCS: x0=input, x1=output, s0=scale, x2=n
//
// vlx4 pattern: 64 elements (64 bytes → 4×16 int32 → 4×16 FP32) per iteration

.section __TEXT,__text,regular,pure_instructions
.global _dequantize_int8_fp32
.p2align 4

_dequantize_int8_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x2, .Ldone

    // Save scale before smstart (smstart zeroes z-regs)
    fmov    w9, s0

    smstart sm

    ptrue   p0.s
    cntw    x10                 // x10 = 16 (SVLw, number of int32 elements per vector)

    // Restore and broadcast scale
    fmov    s0, w9
    mov     z8.s, s0            // broadcast scale across all lanes

    mov     x8, #0              // element index (for both input bytes and output floats)
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    // Load 64 signed bytes sign-extended to 4×16 int32
    // ld1sb with .s sign-extends each byte to 32-bit int
    // We need 4 separate loads with regular predicates (p1-p4)
    // since ld1sb doesn't support multi-vector predicates

    // Generate individual predicates for each of 4 vectors
    whilelt p1.s, x8, x2
    add     x11, x8, x10
    whilelt p2.s, x11, x2
    add     x12, x11, x10
    whilelt p3.s, x12, x2
    add     x13, x12, x10
    whilelt p4.s, x13, x2

    // Load 16 bytes per vector with sign extension
    ld1sb   {z0.s}, p1/z, [x0, x8]           // load 16 bytes at x8
    ld1sb   {z1.s}, p2/z, [x0, x11]          // load 16 bytes at x8+16
    ld1sb   {z2.s}, p3/z, [x0, x12]          // load 16 bytes at x8+32
    ld1sb   {z3.s}, p4/z, [x0, x13]          // load 16 bytes at x8+48

    // Convert int32 → float (4 vectors in parallel)
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s

    // Scale all 4 vectors
    fmul    z0.s, p0/m, z0.s, z8.s
    fmul    z1.s, p0/m, z1.s, z8.s
    fmul    z2.s, p0/m, z2.s, z8.s
    fmul    z3.s, p0/m, z3.s, z8.s

    // Store 4 vectors (64 FP32 values, 256 bytes)
    st1w    {z0.s-z3.s}, pn9, [x1, x8, lsl #2]

    // Advance by 64 elements (4 vectors × 16 elements)
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x2, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #96
    ret
