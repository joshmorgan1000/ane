// param_update_fp32.s — Fused aggregated parameter update via SME2 streaming SVE
//
// void param_update_fp32(
//     const float *params,              // x0: current parameters (read-only)
//     const float * const gradients[],  // x1: array of gradient buffer pointers
//     float *output,                    // x2: output buffer (new parameters)
//     long n,                           // x3: element count (same for all buffers)
//     float learning_rate,              // s0
//     float inv_updates,                // s1
//     int count                         // x4: number of gradient buffers (1..N)
// );
//
// Computes:
//   sum[i] = gradients[0][i] + gradients[1][i] + ... + gradients[count-1][i]
//   output[i] = params[i] - learning_rate * inv_updates * sum[i]
//
// All gradient buffers, params, and output have the same length n.
// Single smstart/smstop. Uses FMLS for the final fused multiply-subtract.
// Processes 64 floats per inner-loop iteration on M4.
//
// Handles arbitrary count:
//   count <= 0 → copy params to output unchanged
//   count 1-4  → optimized unrolled loops (fast path)
//   count > 4  → general loop iterating over gradient pointer array

.section __TEXT,__text,regular,pure_instructions
.global _param_update_fp32
.p2align 4

_param_update_fp32:
    stp     x29, x30, [sp, #-160]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8, d9, [sp, #96]
    stp     d10, d11, [sp, #112]
    stp     d12, d13, [sp, #128]
    stp     d14, d15, [sp, #144]

    cbz     x3, .Ldone

    // Preserve args across smstart
    mov     x19, x0                 // params
    mov     x20, x1                 // gradients[]
    mov     x21, x2                 // output
    mov     x22, x3                 // n
    mov     w23, w4                 // count
    fmov    w9, s0                  // lr bits
    fmov    w10, s1                 // inv_updates bits

    // count <= 0 → just copy params to output
    cmp     w23, #1
    b.lt    .Lcopy_params

    // Load gradient pointers (up to 4) into callee-saved regs for fast paths
    ldr     x24, [x20]             // grads[0] (always present)
    mov     x25, xzr
    mov     x26, xzr
    mov     x27, xzr
    cmp     w23, #2
    b.lt    .Lptrs_done
    ldr     x25, [x20, #8]         // grads[1]
    cmp     w23, #3
    b.lt    .Lptrs_done
    ldr     x26, [x20, #16]        // grads[2]
    cmp     w23, #4
    b.lt    .Lptrs_done
    ldr     x27, [x20, #24]        // grads[3]
.Lptrs_done:

    smstart sm

    // scale = lr * inv_updates, broadcast
    fmov    s0, w9
    fmov    s1, w10
    fmul    s8, s0, s1
    mov     z8.s, s8

    ptrue   p0.s
    mov     x8, #0
    whilelt pn9.s, x8, x22, vlx4

    // Branch to specialised loop based on count
    cmp     w23, #1
    b.eq    .Lloop_1
    cmp     w23, #2
    b.eq    .Lloop_2
    cmp     w23, #3
    b.eq    .Lloop_3
    cmp     w23, #4
    b.eq    .Lloop_4
    b       .Lloop_general

    // ── count=1: output = params - scale * g0 ────────────────
.Lloop_1:
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]   // params
    ld1w    {z4.s-z7.s}, pn9/z, [x24, x8, lsl #2]   // g0
    fmls    z0.s, p0/m, z4.s, z8.s
    fmls    z1.s, p0/m, z5.s, z8.s
    fmls    z2.s, p0/m, z6.s, z8.s
    fmls    z3.s, p0/m, z7.s, z8.s
    st1w    {z0.s-z3.s}, pn9, [x21, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lloop_1
    b       .Lexit

    // ── count=2: sum g0+g1, then fmls ────────────────────────
.Lloop_2:
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x24, x8, lsl #2]
    ld1w    {z12.s-z15.s}, pn9/z, [x25, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z12.s
    fadd    z5.s, p0/m, z5.s, z13.s
    fadd    z6.s, p0/m, z6.s, z14.s
    fadd    z7.s, p0/m, z7.s, z15.s
    fmls    z0.s, p0/m, z4.s, z8.s
    fmls    z1.s, p0/m, z5.s, z8.s
    fmls    z2.s, p0/m, z6.s, z8.s
    fmls    z3.s, p0/m, z7.s, z8.s
    st1w    {z0.s-z3.s}, pn9, [x21, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lloop_2
    b       .Lexit

    // ── count=3: sum g0+g1+g2, then fmls ────────────────────
.Lloop_3:
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x24, x8, lsl #2]
    ld1w    {z12.s-z15.s}, pn9/z, [x25, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z12.s
    fadd    z5.s, p0/m, z5.s, z13.s
    fadd    z6.s, p0/m, z6.s, z14.s
    fadd    z7.s, p0/m, z7.s, z15.s
    ld1w    {z12.s-z15.s}, pn9/z, [x26, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z12.s
    fadd    z5.s, p0/m, z5.s, z13.s
    fadd    z6.s, p0/m, z6.s, z14.s
    fadd    z7.s, p0/m, z7.s, z15.s
    fmls    z0.s, p0/m, z4.s, z8.s
    fmls    z1.s, p0/m, z5.s, z8.s
    fmls    z2.s, p0/m, z6.s, z8.s
    fmls    z3.s, p0/m, z7.s, z8.s
    st1w    {z0.s-z3.s}, pn9, [x21, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lloop_3
    b       .Lexit

    // ── count=4: sum g0+g1+g2+g3, then fmls ─────────────────
.Lloop_4:
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]
    ld1w    {z4.s-z7.s}, pn9/z, [x24, x8, lsl #2]
    ld1w    {z12.s-z15.s}, pn9/z, [x25, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z12.s
    fadd    z5.s, p0/m, z5.s, z13.s
    fadd    z6.s, p0/m, z6.s, z14.s
    fadd    z7.s, p0/m, z7.s, z15.s
    ld1w    {z12.s-z15.s}, pn9/z, [x26, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z12.s
    fadd    z5.s, p0/m, z5.s, z13.s
    fadd    z6.s, p0/m, z6.s, z14.s
    fadd    z7.s, p0/m, z7.s, z15.s
    ld1w    {z12.s-z15.s}, pn9/z, [x27, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z12.s
    fadd    z5.s, p0/m, z5.s, z13.s
    fadd    z6.s, p0/m, z6.s, z14.s
    fadd    z7.s, p0/m, z7.s, z15.s
    fmls    z0.s, p0/m, z4.s, z8.s
    fmls    z1.s, p0/m, z5.s, z8.s
    fmls    z2.s, p0/m, z6.s, z8.s
    fmls    z3.s, p0/m, z7.s, z8.s
    st1w    {z0.s-z3.s}, pn9, [x21, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lloop_4
    b       .Lexit

    // ── count > 4: general loop ─────────────────────────────
    // For each vlx4 chunk:
    //   1. Load params into z0-z3
    //   2. Zero accumulators z4-z7
    //   3. Loop over all count gradient pointers, loading + accumulating
    //   4. fmls params against accumulated sum * scale
    //   5. Store
    //
    // x20 = gradient pointer array base
    // w23 = count
    // x28 = inner loop counter for gradient index
.Lloop_general:
    // Load params
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // Zero gradient accumulators
    mov     z4.d, #0
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0

    // Inner loop over gradient pointers
    mov     w28, #0                 // grad_idx = 0
.Lgen_grad_loop:
    // Load gradient pointer: grad_ptr = gradients[grad_idx]
    ldr     x9, [x20, x28, lsl #3]  // x9 = gradients[grad_idx] (8-byte pointers)
    // Load gradient data
    ld1w    {z12.s-z15.s}, pn9/z, [x9, x8, lsl #2]
    // Accumulate
    fadd    z4.s, p0/m, z4.s, z12.s
    fadd    z5.s, p0/m, z5.s, z13.s
    fadd    z6.s, p0/m, z6.s, z14.s
    fadd    z7.s, p0/m, z7.s, z15.s
    // Next gradient
    add     w28, w28, #1
    cmp     w28, w23
    b.lt    .Lgen_grad_loop

    // Apply: output = params - scale * sum
    fmls    z0.s, p0/m, z4.s, z8.s
    fmls    z1.s, p0/m, z5.s, z8.s
    fmls    z2.s, p0/m, z6.s, z8.s
    fmls    z3.s, p0/m, z7.s, z8.s
    st1w    {z0.s-z3.s}, pn9, [x21, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x22, vlx4
    b.first .Lloop_general
    b       .Lexit

    // ── count <= 0: copy params to output unchanged ──────────
.Lcopy_params:
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
    ldp     d14, d15, [sp, #144]
    ldp     d12, d13, [sp, #128]
    ldp     d10, d11, [sp, #112]
    ldp     d8, d9, [sp, #96]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     x29, x30, [sp], #160
    ret
