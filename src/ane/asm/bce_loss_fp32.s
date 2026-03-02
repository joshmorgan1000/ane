// bce_loss_fp32.s — Binary cross-entropy loss via SME2 streaming SVE
//
// float bce_loss_fp32(const float *pred, const float *target, long n)
//
// Returns: -(1/n) * sum(target[i]*log(pred[i]) + (1-target[i])*log(1-pred[i]))
// Uses scalar loop with proper IEEE754 log extraction.

.section __TEXT,__text,regular,pure_instructions
.global _bce_loss_fp32
.p2align 4

_bce_loss_fp32:
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]

    cmp     x2, #0
    b.le    .Lzero

    // Load constants
    adr     x9, .Lconst
    ldr     s12, [x9]               // epsilon = 1e-7
    ldr     s13, [x9, #4]           // 1 - epsilon
    ldr     s10, [x9, #8]           // 1.0
    ldr     s11, [x9, #12]          // ln(2)

    // Scalar accumulator
    fmov    s8, wzr                 // sum = 0.0
    mov     x6, #0

.Lloop:
    cmp     x6, x2
    b.ge    .Lreduce

    // Load pred and target
    ldr     s0, [x0, x6, lsl #2]
    ldr     s1, [x1, x6, lsl #2]

    // Clamp pred to [epsilon, 1-epsilon]
    fmax    s0, s0, s12
    fmin    s0, s0, s13

    // Compute 1 - pred
    fsub    s2, s10, s0
    fmax    s2, s2, s12
    fmin    s2, s2, s13

    // Compute log(pred):
    // Save raw bits, extract exponent and mantissa separately
    fmov    w3, s0                  // w3 = raw bits
    ubfx    w4, w3, #23, #8        // w4 = biased exponent (bits 30:23)
    sub     w4, w4, #127            // w4 = unbiased exponent e
    scvtf   s4, w4                  // s4 = e as float
    and     w5, w3, #0x007FFFFF     // w5 = mantissa bits
    orr     w5, w5, #0x3F800000     // w5 = mantissa in [1.0, 2.0)
    fmov    s3, w5
    // log(x) ≈ e*ln(2) + (m-1)*ln(2)
    fsub    s3, s3, s10             // m - 1
    fmul    s3, s3, s11             // (m-1)*ln(2)
    fmadd   s3, s4, s11, s3         // e*ln(2) + (m-1)*ln(2)

    // Compute log(1-pred) same way
    fmov    w3, s2
    ubfx    w4, w3, #23, #8
    sub     w4, w4, #127
    scvtf   s6, w4
    and     w5, w3, #0x007FFFFF
    orr     w5, w5, #0x3F800000
    fmov    s7, w5
    fsub    s7, s7, s10
    fmul    s7, s7, s11
    fmadd   s7, s6, s11, s7

    // loss = -[target*log(pred) + (1-target)*log(1-pred)]
    fmul    s4, s1, s3              // target * log(pred)
    fsub    s5, s10, s1             // 1 - target
    fmadd   s4, s5, s7, s4          // + (1-target)*log(1-pred)
    fneg    s4, s4

    // Accumulate
    fadd    s8, s8, s4

    add     x6, x6, #1
    b       .Lloop

.Lreduce:
    // mean = sum / n
    scvtf   s1, x2
    fdiv    s0, s8, s1
    b       .Ldone

.Lzero:
    fmov    s0, wzr

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     x29, x30, [sp], #64
    ret

.p2align 2
.Lconst:
    .long   0x33D6BF95  // 1e-7 (epsilon)
    .long   0x3F7FFF99  // 1 - 1e-7
    .long   0x3F800000  // 1.0
    .long   0x3F317218  // ln(2)
