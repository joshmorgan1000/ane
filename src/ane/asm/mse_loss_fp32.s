// mse_loss_fp32.s — MSE loss via SME2 streaming SVE
//
// float mse_loss_fp32(const float *pred, const float *target, long n)
//
// Returns sum((pred[i] - target[i])^2) / n = Mean Squared Error
//
// Uses vlx4 loads for throughput, 4 independent accumulator vectors
// to hide fadd latency, then tree-reduces to a scalar via faddv,
// and finally divides by n for the mean.

.section __TEXT,__text,regular,pure_instructions
.global _mse_loss_fp32
.p2align 4

_mse_loss_fp32:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]
    // [sp, #80] = scratch for result float
    // [sp, #88] = scratch for n (float)

    // Return 0.0 for n <= 0
    cmp     x2, #0
    b.le    .Lzero

    smstart sm

    ptrue   p0.s

    // Zero 4 accumulator vectors
    mov     z8.d, #0
    mov     z9.d, #0
    mov     z10.d, #0
    mov     z11.d, #0

    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]   // load pred[i..i+63]
    ld1w    {z4.s-z7.s}, pn9/z, [x1, x8, lsl #2]   // load target[i..i+63]

    // Compute diff = pred - target
    fsub    z0.s, p0/m, z0.s, z4.s
    fsub    z1.s, p0/m, z1.s, z5.s
    fsub    z2.s, p0/m, z2.s, z6.s
    fsub    z3.s, p0/m, z3.s, z7.s

    // Square: z0-z3 = diff^2
    fmul    z0.s, p0/m, z0.s, z0.s
    fmul    z1.s, p0/m, z1.s, z1.s
    fmul    z2.s, p0/m, z2.s, z2.s
    fmul    z3.s, p0/m, z3.s, z3.s

    // Accumulate into 4 independent sums
    fadd    z8.s,  p0/m, z8.s,  z0.s
    fadd    z9.s,  p0/m, z9.s,  z1.s
    fadd    z10.s, p0/m, z10.s, z2.s
    fadd    z11.s, p0/m, z11.s, z3.s

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x2, vlx4
    b.first .Lloop

    // Tree-reduce 4 accumulators → 2 → 1
    fadd    z8.s, p0/m, z8.s, z9.s
    fadd    z10.s, p0/m, z10.s, z11.s
    fadd    z8.s, p0/m, z8.s, z10.s

    // Horizontal sum → scalar, store to stack before smstop
    faddv   s0, p0, z8.s
    str     s0, [sp, #80]

    smstop

    // Reload sum and compute mean = sum / n
    ldr     s0, [sp, #80]

    // Convert n to float and divide
    scvtf   s1, x2
    fdiv    s0, s0, s1

    b       .Ldone

.Lzero:
    fmov    s0, wzr

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #96
    ret
