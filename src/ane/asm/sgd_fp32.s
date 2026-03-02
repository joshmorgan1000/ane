// sgd_fp32.s — SGD parameter update with contiguous gradient averaging
//
// void sgd_fp32(
//     const float *params,       // x0: current parameters [n]
//     const float *gradients,    // x1: packed gradients [batch_size × n], contiguous
//     float *output,             // x2: output parameters [n] (may alias params)
//     long n,                    // x3: parameter count
//     float scale,               // s0: pre-computed learning_rate / batch_size
//     long batch_size             // x4: number of gradient rows
// );
//
// Computes:
//   sum[i] = gradients[0*n + i] + gradients[1*n + i] + ... + gradients[(batch_size-1)*n + i]
//   output[i] = params[i] - scale * sum[i]
//
// Gradients are laid out as batch_size contiguous rows of n floats each:
//   [ row0: g0_0, g0_1, ..., g0_{n-1}, row1: g1_0, g1_1, ..., g1_{n-1}, ... ]
//
// The C++ wrapper computes scale = learning_rate / batch_size, so the kernel
// just does: output = params - scale * sum(gradient_rows).
//
// Supports in-place update (output == params).
// Processes 64 floats per outer-loop iteration on M4.

.section __TEXT,__text,regular,pure_instructions
.global _sgd_fp32
.p2align 4

_sgd_fp32:
    stp     x29, x30, [sp, #-144]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     d8, d9, [sp, #80]
    stp     d10, d11, [sp, #96]
    stp     d12,  d13,  [sp, #112]
    stp     d14,  d15,  [sp, #128]

    cbz     x3, .Ldone
    cbz     x4, .Lcopy_params

    // Preserve args across smstart
    mov     x19, x0                 // params
    mov     x20, x1                 // gradients (contiguous)
    mov     x21, x2                 // output
    mov     x22, x3                 // n
    mov     x23, x4                 // batch_size
    fmov    w9, s0                  // scale bits (survives smstart)

    // Gradient row stride in bytes: n * sizeof(float) = n * 4
    lsl     x24, x22, #2           // x24 = stride = n * 4

    smstart sm

    // Restore scale into z8 (broadcast)
    fmov    s0, w9
    mov     z8.s, s0

    ptrue   p0.s
    mov     x8, #0                  // element offset
    whilelt pn9.s, x8, x22, vlx4

    // ── batch_size == 1 fast path ──────────────────────────────
    cmp     x23, #1
    b.eq    .Lloop_1

    // ── General loop: arbitrary batch_size ──────────────────────
.Lloop_general:
    // Load params
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Zero gradient accumulators
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0

    // Inner loop over gradient rows
    // x25 = current gradient row pointer (advances by stride each iteration)
    // x26 = rows remaining
    mov     x25, x20                // start at gradients[0]
    mov     x26, x23                // rows_remaining = batch_size

.Lgrad_row_loop:
    ld1w    {z12.s-z15.s}, pn9/z, [x25, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z12.s
    fadd    z5.s, p0/m, z5.s, z13.s
    fadd    z6.s, p0/m, z6.s, z14.s
    fadd    z7.s, p0/m, z7.s, z15.s
    add     x25, x25, x24           // advance to next row (stride = n * 4 bytes)
    subs    x26, x26, #1
    b.ne    .Lgrad_row_loop

    // output = params - scale * sum
    fmls    z0.s, p0/m, z4.s, z8.s
    fmls    z1.s, p0/m, z5.s, z8.s
    fmls    z2.s, p0/m, z6.s, z8.s
    fmls    z3.s, p0/m, z7.s, z8.s
    st1w    {z0.s-z3.s}, pn9, [x21, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lloop_general
    b       .Lexit

    // ── batch_size == 1: no accumulation needed ────────────────
.Lloop_1:
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x20, x8, lsl #2]
    fmls    z0.s, p0/m, z4.s, z8.s
    fmls    z1.s, p0/m, z5.s, z8.s
    fmls    z2.s, p0/m, z6.s, z8.s
    fmls    z3.s, p0/m, z7.s, z8.s
    st1w    {z0.s-z3.s}, pn9, [x21, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lloop_1
    b       .Lexit

    // ── batch_size == 0: copy params to output unchanged ───────
.Lcopy_params:
    mov     x19, x0
    mov     x21, x2
    mov     x22, x3

    smstart sm
    ptrue   p0.s
    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

.Lcopy_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]
    st1w    {z0.s-z3.s}, pn9, [x21, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lcopy_loop

.Lexit:
    smstop

.Ldone:
    ldp     d10, d11, [sp, #96]
    ldp     d8, d9, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     d12,  d13,  [sp, #112]
    ldp     d14,  d15,  [sp, #128]
    ldp     x29, x30, [sp], #144
    ret
