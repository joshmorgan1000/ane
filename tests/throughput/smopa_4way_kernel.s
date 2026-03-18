.section __TEXT,__text,regular,pure_instructions
.global _smopa_4way_loop
.p2align 4
// x0 = iteration count
_smopa_4way_loop:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    mov     x10, x0
    smstart
    ptrue   p0.b
    ptrue   p1.b
.Lloop_smopa:
    // 8× unroll: 2 rounds of all 4 za.s tiles = 8 SMOPA instructions
    // Round 1: different source pairs per tile to avoid RAW stalls
    smopa   za0.s, p0/m, p1/m, z0.b, z1.b
    smopa   za1.s, p0/m, p1/m, z2.b, z3.b
    smopa   za2.s, p0/m, p1/m, z4.b, z5.b
    smopa   za3.s, p0/m, p1/m, z6.b, z7.b
    // Round 2
    smopa   za0.s, p0/m, p1/m, z8.b, z9.b
    smopa   za1.s, p0/m, p1/m, z10.b, z11.b
    smopa   za2.s, p0/m, p1/m, z12.b, z13.b
    smopa   za3.s, p0/m, p1/m, z14.b, z15.b
    subs    x10, x10, #1
    b.ne    .Lloop_smopa
    smstop
    ldp     x29, x30, [sp], #16
    ret
