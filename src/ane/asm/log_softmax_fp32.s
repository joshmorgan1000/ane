// log_softmax_fp32.s — Log-softmax via SME2 streaming SVE
//
// void log_softmax_fp32(const float *input, float *output, long n)
//
// Computes: output[i] = input[i] - log(sum(exp(input[j] for all j)))
// Algorithm:
//   1) Find max(input) via reduce_max (prevent overflow)
//   2) Compute exp(input[i] - max), store to output
//   3) Sum the exp values
//   4) Compute log_sum = log(sum) + max
//   5) output[i] -= log_sum

.section __TEXT,__text,regular,pure_instructions
.global _log_softmax_fp32
.p2align 4

_log_softmax_fp32:
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    stp     d14, d15, [sp, #96]

    cbz     x2, .Ldone

    mov     x19, x0
    mov     x20, x1
    mov     x21, x2

    smstart sm

    ptrue   p0.s

    // Load exp constants
    adr     x9, .Lconst_exp
    ld1rw   {z21.s}, p0/z, [x9]        // log2e
    ld1rw   {z22.s}, p0/z, [x9, #4]    // c6
    ld1rw   {z23.s}, p0/z, [x9, #8]    // c5
    ld1rw   {z24.s}, p0/z, [x9, #12]   // c4
    ld1rw   {z25.s}, p0/z, [x9, #16]   // c3
    ld1rw   {z26.s}, p0/z, [x9, #20]   // c2
    ld1rw   {z27.s}, p0/z, [x9, #24]   // c1 = ln2
    ld1rw   {z29.s}, p0/z, [x9, #28]   // -inf
    fmov    z28.s, #1.0                  // c0

    // ═════════════════════════════════════════════════════════════
    // Phase 1: find max (single-vector tail)
    // ═════════════════════════════════════════════════════════════
    mov     z4.d, z29.d         // max_acc = -inf
    mov     z5.d, z29.d
    mov     z6.d, z29.d
    mov     z7.d, z29.d

    mov     x8, #0
    whilelt p1.s, x8, x21
    b.none  .Lmax_reduce

.Lmax_loop:
    ld1w    {z0.s}, p1/z, [x19, x8, lsl #2]
    fmax    z4.s, p1/m, z4.s, z0.s
    incw    x8
    whilelt p1.s, x8, x21
    b.first .Lmax_loop

.Lmax_reduce:
    // Tree-reduce 4 accumulators → 1
    fmax    z4.s, p0/m, z4.s, z5.s
    fmax    z6.s, p0/m, z6.s, z7.s
    fmax    z4.s, p0/m, z4.s, z6.s
    fmaxv   s20, p0, z4.s
    mov     z20.s, s20

    // ═════════════════════════════════════════════════════════════
    // Phase 2: exp(input - max) → output
    // ═════════════════════════════════════════════════════════════
    mov     x8, #0
    whilelt pn9.s, x8, x21, vlx4

.Lexp_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]

    // x = input - max
    fsub    z0.s, p0/m, z0.s, z20.s
    fsub    z1.s, p0/m, z1.s, z20.s
    fsub    z2.s, p0/m, z2.s, z20.s
    fsub    z3.s, p0/m, z3.s, z20.s

    // z_scaled = x * log2e
    fmul    z0.s, p0/m, z0.s, z21.s
    fmul    z1.s, p0/m, z1.s, z21.s
    fmul    z2.s, p0/m, z2.s, z21.s
    fmul    z3.s, p0/m, z3.s, z21.s

    // n = round(z_scaled) → z8-z11
    movprfx z8, z0
    frintn  z8.s, p0/m, z0.s
    movprfx z9, z1
    frintn  z9.s, p0/m, z1.s
    movprfx z10, z2
    frintn  z10.s, p0/m, z2.s
    movprfx z11, z3
    frintn  z11.s, p0/m, z3.s

    // f = z_scaled - n → z0-z3
    fsub    z0.s, p0/m, z0.s, z8.s
    fsub    z1.s, p0/m, z1.s, z9.s
    fsub    z2.s, p0/m, z2.s, z10.s
    fsub    z3.s, p0/m, z3.s, z11.s

    // Convert n to integer for fscale
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s

    // Horner: 2^f = c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*(c5 + f*c6)))))
    mov     z16.d, z22.d        // p = c6
    mov     z17.d, z22.d
    mov     z18.d, z22.d
    mov     z19.d, z22.d

    fmad    z16.s, p0/m, z0.s, z23.s   // p = p*f + c5
    fmad    z17.s, p0/m, z1.s, z23.s
    fmad    z18.s, p0/m, z2.s, z23.s
    fmad    z19.s, p0/m, z3.s, z23.s

    fmad    z16.s, p0/m, z0.s, z24.s   // p = p*f + c4
    fmad    z17.s, p0/m, z1.s, z24.s
    fmad    z18.s, p0/m, z2.s, z24.s
    fmad    z19.s, p0/m, z3.s, z24.s

    fmad    z16.s, p0/m, z0.s, z25.s   // p = p*f + c3
    fmad    z17.s, p0/m, z1.s, z25.s
    fmad    z18.s, p0/m, z2.s, z25.s
    fmad    z19.s, p0/m, z3.s, z25.s

    fmad    z16.s, p0/m, z0.s, z26.s   // p = p*f + c2
    fmad    z17.s, p0/m, z1.s, z26.s
    fmad    z18.s, p0/m, z2.s, z26.s
    fmad    z19.s, p0/m, z3.s, z26.s

    fmad    z16.s, p0/m, z0.s, z27.s   // p = p*f + c1
    fmad    z17.s, p0/m, z1.s, z27.s
    fmad    z18.s, p0/m, z2.s, z27.s
    fmad    z19.s, p0/m, z3.s, z27.s

    fmad    z16.s, p0/m, z0.s, z28.s   // p = p*f + c0
    fmad    z17.s, p0/m, z1.s, z28.s
    fmad    z18.s, p0/m, z2.s, z28.s
    fmad    z19.s, p0/m, z3.s, z28.s

    // result = p * 2^n
    fscale  z16.s, p0/m, z16.s, z8.s
    fscale  z17.s, p0/m, z17.s, z9.s
    fscale  z18.s, p0/m, z18.s, z10.s
    fscale  z19.s, p0/m, z19.s, z11.s

    // Store (pn9 masks inactive lanes)
    st1w    {z16.s-z19.s}, pn9, [x20, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x21, vlx4
    b.first .Lexp_loop

    // ═════════════════════════════════════════════════════════════
    // Phase 3: sum exp values and compute log_sum
    // ═════════════════════════════════════════════════════════════
    mov     z4.d, #0            // sum_acc
    mov     z5.d, #0
    mov     z6.d, #0
    mov     z7.d, #0

    mov     x8, #0
    whilelt pn8.s, x8, x21, vlx4

.Lsum_loop:
    ld1w    {z0.s-z3.s}, pn8/z, [x20, x8, lsl #2]
    fadd    z4.s, p0/m, z4.s, z0.s
    fadd    z5.s, p0/m, z5.s, z1.s
    fadd    z6.s, p0/m, z6.s, z2.s
    fadd    z7.s, p0/m, z7.s, z3.s
    incw    x8, all, mul #4
    whilelt pn8.s, x8, x21, vlx4
    b.first .Lsum_loop

    // Tree-reduce 4 accumulators → 1 → scalar
    fadd    z4.s, p0/m, z4.s, z5.s
    fadd    z6.s, p0/m, z6.s, z7.s
    fadd    z4.s, p0/m, z4.s, z6.s
    faddv   s4, p0, z4.s

    // log(sum) via IEEE754 decomposition
    fmov    w11, s4              // w11 = raw bits of sum
    ubfx    w12, w11, #23, #8   // w12 = biased exponent
    sub     w12, w12, #127      // w12 = unbiased exponent e
    scvtf   s5, w12             // s5 = e as float
    and     w13, w11, #0x007FFFFF // w13 = mantissa bits
    orr     w13, w13, #0x3F800000
    fmov    s6, w13             // m in [1, 2)

    // ln(m) via atanh series: ln(m) = 2*atanh((m-1)/(m+1))
    //   u = (m-1)/(m+1), ln(m) = 2*u*(1 + u²/3 + u⁴/5 + u⁶/7)
    //   u ∈ [0, 1/3) → fast convergence, max error ~1e-3
    fsub    s6, s6, s28         // t = m - 1.0 (z28=1.0, still live)
    fmov    s9, #2.0
    fadd    s10, s6, s9          // t + 2
    fdiv    s6, s6, s10          // u = t / (t + 2)
    fmul    s10, s6, s6          // u²

    adr     x9, .Lconst_log
    ldr     s7, [x9]            // 1/7
    ldr     s11, [x9, #4]       // 1/5
    ldr     s12, [x9, #8]       // 1/3

    fmadd   s7, s7, s10, s11    // p = 1/7*u² + 1/5
    fmadd   s7, s7, s10, s12    // p = p*u² + 1/3
    fmadd   s7, s7, s10, s28    // p = p*u² + 1.0
    fmul    s7, s7, s6           // u * p
    fadd    s7, s7, s7           // 2*u*p = ln(m)

    // log(sum) = e*ln(2) + ln(m)
    fmadd   s8, s5, s27, s7     // e*ln(2) + ln(m) (s27 = ln2 from z27)

    // ═════════════════════════════════════════════════════════════
    // Phase 4: output[i] = input[i] - (log_sum + max)
    // ═════════════════════════════════════════════════════════════
    fadd    s8, s8, s20          // log_sum_total = log(sum_exp) + max (z20 preserved)
    mov     x8, #0
    whilelt pn9.s, x8, x21, vlx4
    mov     z30.s, s8            // broadcast log_sum_total

.Lnorm_loop:
    ld1w    {z0.s-z3.s}, pn9/z, [x19, x8, lsl #2]  // re-read input
    fsub    z0.s, p0/m, z0.s, z30.s
    fsub    z1.s, p0/m, z1.s, z30.s
    fsub    z2.s, p0/m, z2.s, z30.s
    fsub    z3.s, p0/m, z3.s, z30.s
    st1w    {z0.s-z3.s}, pn9, [x20, x8, lsl #2]
    incw    x8, all, mul #4
    whilelt pn9.s, x8, x21, vlx4
    b.first .Lnorm_loop

.Lexit:
    smstop

.Ldone:
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     d8,  d9,  [sp, #48]
    ldp     d10, d11, [sp, #64]
    ldp     d12, d13, [sp, #80]
    ldp     d14, d15, [sp, #96]
    ldp     x29, x30, [sp], #112
    ret

.p2align 2
.Lconst_exp:
    .float 1.4426950408889634       // log2e
    .float 0.00015403530393381609   // c6 = ln2^6/720
    .float 0.0013333558146428443    // c5 = ln2^5/120
    .float 0.009618129107628477     // c4 = ln2^4/24
    .float 0.05550410866482158      // c3 = ln2^3/6
    .float 0.24022650695910072      // c2 = ln2^2/2
    .float 0.6931471805599453       // c1 = ln2
    .long  0xFF800000               // -inf

.p2align 2
.Lconst_log:
    .float 0.142857143              // 1/7
    .float 0.2                      // 1/5
    .float 0.333333333              // 1/3
