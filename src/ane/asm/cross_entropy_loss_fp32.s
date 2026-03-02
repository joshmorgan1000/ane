// cross_entropy_loss_fp32.s — Cross-entropy loss via SME2 streaming SVE
//
// float cross_entropy_loss_fp32(const float *log_probs, const int32_t *targets, long batch_size, long num_classes)
//
// Returns: -(1/batch_size) * sum_{b=0}^{batch_size-1} log_probs[b * num_classes + targets[b]]

.section __TEXT,__text,regular,pure_instructions
.global _cross_entropy_loss_fp32
.p2align 4

_cross_entropy_loss_fp32:
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]

    cbz     x2, .Lzero              // batch_size == 0

    // Scalar accumulation (no SME needed — element-at-a-time)
    fmov    s8, wzr                 // sum = 0.0
    mov     x4, #0                  // batch index

.Lloop:
    cmp     x4, x2
    b.ge    .Lreduce

    // Load targets[batch_idx] → w5
    ldr     w5, [x1, x4, lsl #2]

    // Offset = batch_idx * num_classes + targets[batch_idx]
    mul     x6, x4, x3
    add     x6, x6, x5

    // Load log_probs[offset]
    ldr     s0, [x0, x6, lsl #2]

    // Negate and accumulate: sum += -log_probs[...]
    fneg    s0, s0
    fadd    s8, s8, s0

    add     x4, x4, #1
    b       .Lloop

.Lreduce:
    // Compute mean = sum / batch_size
    scvtf   s1, x2
    fdiv    s0, s8, s1

    b       .Ldone

.Lzero:
    fmov    s0, wzr

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     x29, x30, [sp], #64
    ret
