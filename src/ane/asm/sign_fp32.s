// sign_fp32.s — Element-wise sign function via SME2 streaming SVE
//
// void sign_fp32(const float *input, float *output, long n)
// AAPCS: x0=input, x1=output, x2=n
//
// output[i] = -1.0  if input[i] < 0
// output[i] =  0.0  if input[i] == 0
// output[i] = +1.0  if input[i] > 0
//
// Strategy: load +1.0 and -1.0 constant vectors, then use sel to pick
// from them based on predicate comparisons, with a zero base vector.
//
// Processes 4 vectors (64 floats on M4) per iteration.

.section __TEXT,__text,regular,pure_instructions
.global _sign_fp32
.p2align 4

_sign_fp32:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     d8,  d9,  [sp, #16]
    stp     d10, d11, [sp, #32]
    stp     d12, d13, [sp, #48]
    stp     d14, d15, [sp, #64]

    cbz     x2, .Ldone

    smstart sm

    ptrue   p0.s

    // Load +1.0 into z16, -1.0 into z17
    adr     x9, .Lconst
    ld1rw   {z16.s}, p0/z, [x9]        // +1.0 = 0x3F800000
    ld1rw   {z17.s}, p0/z, [x9, #4]    // -1.0 = 0xBF800000

    mov     x8, #0
    whilelt pn9.s, x8, x2, vlx4

.Lloop:
    ld1w    {z0.s-z3.s}, pn9/z, [x0, x8, lsl #2]

    // --- z0 -> z4 ---
    // start with zero
    mov     z4.d, #0
    // where input > 0, select +1.0
    fcmgt   p1.s, p0/z, z0.s, #0.0
    sel     z4.s, p1, z16.s, z4.s
    // where input < 0, select -1.0
    fcmlt   p2.s, p0/z, z0.s, #0.0
    sel     z4.s, p2, z17.s, z4.s

    // --- z1 -> z5 ---
    mov     z5.d, #0
    fcmgt   p3.s, p0/z, z1.s, #0.0
    sel     z5.s, p3, z16.s, z5.s
    fcmlt   p4.s, p0/z, z1.s, #0.0
    sel     z5.s, p4, z17.s, z5.s

    // --- z2 -> z6 ---
    mov     z6.d, #0
    fcmgt   p5.s, p0/z, z2.s, #0.0
    sel     z6.s, p5, z16.s, z6.s
    fcmlt   p6.s, p0/z, z2.s, #0.0
    sel     z6.s, p6, z17.s, z6.s

    // --- z3 -> z7 ---
    mov     z7.d, #0
    fcmgt   p7.s, p0/z, z3.s, #0.0
    sel     z7.s, p7, z16.s, z7.s
    fcmlt   p1.s, p0/z, z3.s, #0.0
    sel     z7.s, p1, z17.s, z7.s

    st1w    {z4.s-z7.s}, pn9, [x1, x8, lsl #2]

    incw    x8, all, mul #4
    whilelt pn9.s, x8, x2, vlx4
    b.first .Lloop

    smstop

.Ldone:
    ldp     d8,  d9,  [sp, #16]
    ldp     d10, d11, [sp, #32]
    ldp     d12, d13, [sp, #48]
    ldp     d14, d15, [sp, #64]
    ldp     x29, x30, [sp], #80
    ret

.p2align 2
.Lconst:
    .long   0x3F800000  // +1.0f
    .long   0xBF800000  // -1.0f
