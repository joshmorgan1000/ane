// luti2_u8.s — 2-bit index LUT expand (scalar implementation)
//
// void luti2_u8(const uint8_t *lut4, const uint8_t *packed_indices,
//               uint8_t *output, long n);
//
// lut4:           4-byte table (4 entries, one byte each)
// packed_indices: n/4 bytes (4 crumbs per byte, bits [1:0] first)
// output:         n bytes of expanded values
// n:              number of output elements (must be multiple of 4)
//
// NOTE: This is a scalar-only implementation. The original used SME2 luti2
// instruction via ZT0, but luti2 causes SIGILL on Apple M4 (see asm_agent_guide.md
// Section 5e). This scalar fallback is functional but slower.
//
// TODO: Implement vectorized 2-bit lookup using tbl + bit manipulation:
//       1. Extract 2-bit indices via shifts and masks
//       2. Replicate 4-entry LUT to fill z-register (16x replication for .b)
//       3. Use tbl for parallel lookup
//
// Register allocation:
//   x0 = lut4 ptr
//   x1 = packed_indices ptr
//   x2 = output ptr
//   x3 = n (element count)
//   x8 = loop index
//   x9 = byte index (element / 4)
//   x10 = crumb position and shift amount
//   w11 = packed byte / extracted crumb index
//   w12 = LUT value

.section __TEXT,__text,regular,pure_instructions
.global _luti2_u8
.p2align 4

_luti2_u8:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Early exit for n <= 0
    cmp     x3, #0
    b.le    .Ldone

    // Scalar loop: process all elements one at a time
    // (luti2 instruction causes SIGILL on M4, so we use scalar fallback)
    mov     x8, #0                      // loop index

.Lloop:
    cmp     x8, x3
    b.ge    .Ldone

    // Compute byte offset and crumb position within that byte
    lsr     x9, x8, #2                  // byte index = element / 4
    and     x10, x8, #3                 // crumb position (0..3)
    ldrb    w11, [x1, x9]               // load packed byte

    // Extract 2-bit index: shift right by (crumb_pos * 2) and mask
    lsl     x10, x10, #1                // shift amount = crumb_pos * 2
    lsr     w11, w11, w10
    and     w11, w11, #0x3              // 2-bit index (0..3)

    // LUT uses stride-4 layout (matches lookup_table struct and ZT0 layout):
    // entry[i] is stored at byte offset i*4 within the 64-byte table.
    lsl     x11, x11, #2               // byte offset = index * 4
    ldrb    w12, [x0, x11]              // load lut[index * 4]
    strb    w12, [x2, x8]               // store to output

    add     x8, x8, #1
    b       .Lloop

.Ldone:
    ldp     x29, x30, [sp], #16
    ret
