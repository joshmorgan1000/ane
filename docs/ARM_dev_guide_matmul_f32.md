# ARM MatMul example (f32)

## Preprocessing step

The preprocess_l function rearranges the matLeft matrix so that blocks of SVLs (rows) x K (columns) are transposed by data tiling and contiguously stored to memory. This rearrangement is implemented by loading the matLeft matrix rows to horizontal slices of a 32-bit ZA tile and storing vertical slices of that 32-bit ZA tile contiguously to memory. The input matrix is zero-padded to a multiple of SVLs rows.

Rearranging the matrix data in memory in this way makes the subsequent memory accesses in the matrix multiplication calculation more efficient, because all data is then read from contiguous memory addresses:
```assembly
1.  preprocess_l:      // x0: M, x1: K, x2: &matLeft, x3: &matLeft_mod
2.      smstart
3. 
4.  // constants
5.      cntw    x4                      // SVLs
6.      mul     x11, x4, x1             // SVLs*K
7.      lsl     x14, x1, #1             // 2*K
8.      add     x15, x14, x1            // 3*K
9. 
10.     mul     x16, x4, x4             // SVLs*SVLs
11. 
12.     mov     x7, #0
13.     whilelt p0.s, x7, x0            // Tile predicate (M dimension)
14. 
15. .Loop_outer:
16.     mov     x8, x2                  // matLeft load base address
17.     mov     x9, x3                  // matLeft_mod store base address
18.     add     x5,  x2, x1, lsl #2     // Exit condition for inner loop
19. 
20.     add     x10, x9 , x11, lsl #2   // 32b Tile0 store predicate condition
21.     sub     x13, x10, x16, lsl #2   // 32b Tile1 store predicate condition
22.     whilelt pn8.b, x8, x5, vlx2     // Tile predicate-as-counter (K dimension)
23. 
24. .Loop_inner:
25.     mov     x6, x8                  // matLeft
26. 
27.     mov     w12, #0                 // Load_loop counter
28. 
29. .Load_loop:
30.     psel    pn10, pn8, p0.s[w12, 0]
31.     psel    pn11, pn8, p0.s[w12, 1]
32.     psel    pn12, pn8, p0.s[w12, 2]
33.     psel    pn13, pn8, p0.s[w12, 3]
34.     ld1w    {z20.s, z28.s}, pn10/z, [x6]                // matLeft
35.     ld1w    {z21.s, z29.s}, pn11/z, [x6, x1,  lsl #2]   // matLeft + K
36.     ld1w    {z22.s, z30.s}, pn12/z, [x6, x14, lsl #2]   // matLeft + K*2
37.     ld1w    {z23.s, z31.s}, pn13/z, [x6, x15, lsl #2]   // matLeft + K*3    
38.     mova    za0h.s[w12, 0:3], {z20.s-z23.s}
39.     mova    za1h.s[w12, 0:3], {z28.s-z31.s}
40. 
41.     add     x6, x6, x1, lsl #4      // matLeft+=4*K FP32 elements (bytes)
42.     add     w12, w12, #4            // Increment counter
43.     cmp     w12, w4
44.     b.mi    .Load_loop
45. 
46.     mov     w12, #0                 // Store_loop counter
47. 
48. .Store_loop:
49.     whilelt pn10.b, x9, x10, vlx4
50.     whilelt pn11.b, x9, x13, vlx4
51.     mova    {z0.s-z3.s}, za0v.s[w12, 0:3]
52.     mova    {z4.s-z7.s}, za1v.s[w12, 0:3]
53.     st1w    {z0.s-z3.s}, pn10, [x9] // Store 4 col vectors to matLeft_mod
54.     st1w    {z4.s-z7.s}, pn11, [x9, x16, lsl #2]  // matLeft_mod+SVLs*SVLs
55.     addvl   x9, x9, #4              // matLeft_mod += 4*SVLb (bytes)
56.     add     w12, w12, #4            // Increment counter
57.     cmp     w12, w4
58.     b.mi    .Store_loop
59. 
60.     add     x9, x9, x16, lsl #2
61.     addvl   x8, x8, #2              // matLeft+= 2*SVLb (bytes)
62.     whilelt pn8.b, x8, x5, vlx2
63.     b.first .Loop_inner
64. 
65.     add     x3, x3, x11, lsl #2     // matLeft_mod+= SVLs*K FP32 elms (bytes)
66.     add     x2, x2, x11, lsl #2     // matLeft+= SVLs*K FP32 elms (bytes]
67.     incw    x7
68. 
69.     whilelt p0.s, x7, x0
70.     b.first .Loop_outer
71. 
72.     smstop
73. 
74.     ret
```

The following shows the overall structure of the preprocess_l function:
```psuedo_code
Loop_outer:
  // Iterate over the rows of the input matrix, the M dimension.
  // Each iteration of Loop_outer rearranges SVLs rows of the input 
  // matrix. 

  Loop_inner:
    // Iterate over the columns of the input matrix, the K dimension.
    // Each iteration of Loop_inner deals with a group of 2*SVLs
    // columns from the original matrix.

    Load_loop:
      // Loads a segment of SVLs (rows) x 2*SVLs (columns) from 
      // the input matrix to two 32-bit ZA tiles.
      //
      // Each iteration of Load_loop loads 2*SVLs elements from
      // 4 rows horizontally into two 32-bit ZA tiles.
      // 
      // In Load_loop, input matrix loads are predicated using 
      // predicate-as-counter. This zeroes inactive elements in
      // the destination vector.

    Store_loop:
      // Stores vertical slices of the two 32-bit element ZA tiles to 
      // memory.
      //
      // Each iteration of Store_loop takes 4 vertical tile slices from
      // the 32-bit ZA tiles and stores the elements at consecutive 
      // memory locations.
      //
      // In Store_loop, output matrix stores are predicated using
      // predicate-as-counter. If the input matrix rows were 
      // zero-padded to a multiple of SVLs rows, those padded zeros
      // are also stored.
```
By repeatedly writing to horizontal slices of the ZA tile on load, but reading from vertical slices to store, the matrix is rearranged in blocks as shown in Rearranging the matLeft matrix:
```figure
   matLeft                                                                                                        matLeft_mod
Row-major memory array                                                                                    Rearranged memory array

0000 [1A]                                                                                                         [1A] 0000
0020 [1B]                                                                                                         [2A] 0020
0040 [1C]                                                                                                         [3A] 0040
0060 [1D]                                                                                                         [4A] 0060
0080 [1E]                                                                                                         [1B] 0080
00A0 [1F]                                                                                                         [2B] 00A0
00C0 [1G]                                                                                                         [3B] 00C0
00E0 [1H]                          [1A] [1B] [1C] [1D] [1E] [1F] [1G] [1H] [1I] [IJ] [  ] [  ]                    ...  ...
0100 [1I]                                                                                                         [4D] 01E0
0120 [1J]                          [2A] [2B] [2C] [2D] [2E] [2F] [2G] [2H] [2I] [2J] [  ] [  ]                    [1E] 0200
0140 [2A]                                                                                                         [2E] 0220
0160 [2B]                          [3A] [3B] [3C] [3D] [3E] [3F] [3G] [3H] [3I] [3J] [  ] [  ]                    [3E] 0240
0180 [2C]      Load to                                                                             Save from      [4E] 0260
01A0 [2D]   Horizontal slices      [4A] [4B] [4C] [4D] [4E] [4F] [4G] [4H] [4I] [4J] [  ] [  ]  Vertical Slices   [1F] 0280
...  [...]    ----------->                                                                        --------->      [2F] 02A0
...  [...]                         [5A] [5B] [5C] [5D] [5E] [5F] [5G] [5H] [5I] [5J] [  ] [  ]                    ...  ...
0500 [5A]                                                                                                         [4H] 03E0
0520 [5B]                          [6A] [6B] [6C] [6D] [6E] [6F] [6G] [6H] [6I] [6J] [  ] [  ]                    [1I] 0400
0540 [5C]                                                                                                         [2I] 0420
0560 [5D]                            0    0    0    0    0    0    0    0    0    0  [  ] [  ]                    [3I] 0440
0580 [5E]                                                                                                         [4I] 0460
05A0 [5F]                            0    0    0    0    0    0    0    0    0    0  [  ] [  ]                    ...  ...
05C0 [5G]                                                                                                         [4J] 04E0
05E0 [5H]                                                                                                         [5A] 0500
0600 [5I]                                                                                                         [6A] 0540
0620 [5J]                                                                                                         [ 0] 0560
0640 [6A]                                                                                                         [ 0] 0580
0660 [6B]                                                                                                         [5B] 05A0
0680 [6C]                                                                                                         [6B] 05C0
06A0 [6D]                                                                                                         ...  ...
06C0 [6E]                                                                                                         [ 0] 06E0
06E0 [6F]                                                                                                         [5E] 0700
0700 [6G]                                                                                                         [6E] 0720
0720 [6H]                                                                                                         [ 0] 0740
0740 [6I]                                                                                                         [ 0] 0760
0760 [6J]                                                                                                         [5F] 0780
                                                                                                                  [6F] 07A0
                                                                                                                  ... ...
                                                                                                                  [ 0] 07E0
                                                                                                                  [5I] 0900
                                                                                                                  [6I] 0920
                                                                                                                  [ 0] 0940
                                                                                                                  [ 0] 0960
                                                                                                                  [5J] 0980
                                                                                                                  [6J] 09A0
                                                                                                                  [ 0] 09C0
                                                                                                                  [ 0] 09E0
```
The left-hand side of the figure shows the row-major input matrix in consecutive memory address locations, with each element occupying 4 bytes. The right-hand side of the figure shows the rearranged data, after processing by the preprocess_l function. Rearranging the data in this way means that the matmul_opt function accesses matleft_mod data from contiguous memory addresses when performing the matrix multiplication, improving the efficiency of the algorithm.

The example in this figure assumes that SVL is 128b, with 4x4 32-bit elements per ZA tile and that the dimensions of the matLeft matrix are 6 rows by 10 columns.

## matmul_opt code
```assembly
1.   matmul_opt: // x0: M, x1: K, x2: N, x3: matLeft_mod, x4: matRight, x5: matResult
2.     stp     x19, x20, [sp, #-48]!
3.     stp     x21, x22, [sp, #16]
4.     stp     x23, x24, [sp, #32]
5. 
6.     smstart
7. 
8.     // constants
9.     cntw    x6                      // SVLs
10.    mul     x22, x6, x1             // SVLs*K
11.    mul     x23, x6, x2             // SVLs*N
12.    add     x25, x23, x2            // SVLs*N + N
13.    add     x11, x4, x2, lsl #2     // Exit condition for N loop
14.    mov     x12, #0
15.    cntb    x6                      // SVLb
16.    mov     x14, #0
17.    ptrue   pn10.b                  // Predicate for SME2 VLx2 (a_ptr loads)
18.    whilelt pn8.s, x12, x0, vlx2    // tiles predicate (M dimension)
19.    sub     w6, w6, #8              // SVLb-8
20. 
21.  .Loop_M:
22.    // Extract tile 0/1 and tile 2/3 predicates (M) from vlx2 predicate.
23.    pext    { p2.s, p3.s }, pn8[0]
24.    mov     x16, x4                 // b_base
25.    mov     x9, x5                  // c_base
26.    whilelt pn9.b, x16, x11, vlx2   // tiles predicate (N dimension)
27. 
28.  .Loop_N:
29.    mov     x7, x3                  // a_ptr = a_base
30.    mov     x17, x16                // b_ptr = b_base
31.    mov     x10, x9                 // c_ptr0 = c_base
32. 
33.    // Extract tile 0/2 and tile 1/3 predicates (N) from vlx2 predicate.
34.    pext    { p0.b, p1.b }, pn9[0]
35. 
36.    add     x8, x3, x22, lsl #2     // a_base + SVLs*K FP32 elms (bytes)
37.    addvl   x15, x8, #-1            // Exit condition for K loop
38.    ld1w    {z1.s}, p2/z, [x7]      // Load 1st vector from a_ptr
39. 
40.    zero    {za}
41.    ld1w    {z2.s-z3.s}, pn9/z, [x17]  // Load 2 vectors from b_ptr
42. 
43.    fmopa   za0.s, p2/m, p0/m, z1.s, z2.s // ZA0+=1st a_ptr vec OP 1st b_ptr vec
44.    ld1w    {z5.s}, p3/z, [x7, x22, lsl #2]  // Load 2nd vector from a_ptr
45.    addvl   x7, x7, #1                       // a_ptr += SVLb (bytes)
46. 
47.  .Loop_K:
48.    fmopa   za2.s, p3/m, p0/m, z5.s, z2.s // ZA2+=2nd a_ptr vec OP 1st b_ptr vec
49. 
50.    fmopa   za1.s, p2/m, p1/m, z1.s, z3.s // ZA1+=1st a_ptr vec OP 2nd b_ptr vec
51.    ld1w    {z0.s-z1.s}, pn10/z, [x7]     // Load next 2 vectors from a_ptr
52. 
53.    fmopa   za3.s, p3/m, p1/m, z5.s, z3.s // ZA3+=2nd a_ptr vec OP 2nd b_ptr vec
54.    ld1w    {z6.s-z7.s}, pn9/z, [x17, x2, lsl #2] // Load next 2 vecs from b_ptr
55. 
56.    fmopa   za0.s, p2/m, p0/m, z0.s, z6.s // ZA0+=1st a_ptr vec OP 1st b_ptr vec
57.    psel    pn11, pn10, p3.s[w14, 0]      // Select predicate-as-counter
58.    ld1w    {z4.s-z5.s}, pn11/z, [x7, x22, lsl #2] // Load next 2 vecs from a_ptr
59. 
60.    fmopa   za2.s, p3/m, p0/m, z4.s, z6.s // ZA2+=2nd a_ptr vec OP 1st b_ptr vec
61.    add     x17, x17, x2, lsl #3          // b_ptr += 2*N FP32 elms (bytes)
62. 
63.    fmopa   za1.s, p2/m, p1/m, z0.s, z7.s // ZA1+=1st a_ptr vec OP 2nd b_ptr vec
64. 
65.    fmopa   za3.s, p3/m, p1/m, z4.s, z7.s // ZA3+=2nd a_ptr vec OP 2nd b_ptr vec
66.    ld1w    {z2.s-z3.s}, pn9/z, [x17]     // Load next 2 vectors from b_ptr
67. 
68.    fmopa   za0.s, p2/m, p0/m, z1.s, z2.s // ZA0+=1st a_ptr vec OP 1st b_ptr vec
69.    addvl   x7, x7, #2                    // a_ptr += 2*SVLb (bytes)
70. 
71.    cmp     x7, x15
72.    b.mi    .Loop_K
73. 
74.    fmopa   za2.s, p3/m, p0/m, z5.s, z2.s // ZA2+=2nd a_ptr vec OP 1st b_ptr vec
75. 
76.    fmopa   za1.s, p2/m, p1/m, z1.s, z3.s // ZA1+=1st a_ptr vec OP 2nd b_ptr vec
77. 
78.    fmopa   za3.s, p3/m, p1/m, z5.s, z3.s // ZA3+=2nd a_ptr vec OP 2nd b_ptr vec
79.    add     x17, x17, x2, lsl #2          // b_ptr += 2*N FP32 elms (bytes)
80. 
81.    cmp     x7, x8
82.    b.pl    .Ktail_end
83. 
84.  .Ktail_start:
85.    ld1w    {z1.s}, p2/z, [x7]
86.    ld1w    {z2.s-z3.s}, pn9/z, [x17]
87. 
88.    fmopa   za0.s, p2/m, p0/m, z1.s, z2.s
89.    ld1w    {z5.s}, p3/z, [x7, x22, lsl #2]
90. 
91.    fmopa   za2.s, p3/m, p0/m, z5.s, z2.s
92. 
93.    fmopa   za1.s, p2/m, p1/m, z1.s, z3.s
94. 
95.    fmopa   za3.s, p3/m, p1/m, z5.s, z3.s
96. 
97.  .Ktail_end:
98.    mov     w13, #0
99.    psel    pn11, pn9, p2.b[w13, 0]
100.   psel    pn12, pn9, p3.b[w13, 0]
101.   // ZA tiles to vecs: z0 = za0h[1], z1 = za1h[1], z2 = za2h[1], z3 = za3h[1]
102.   mova    { z0.b-z3.b }, za0h.b[w13, 0:3]
103.   st1w    { z0.s-z1.s }, pn11, [x10]              // Store to c_ptr0
104.   st1w    { z2.s-z3.s }, pn12, [x10, x23, lsl #2] // Store to c_ptr0+(SVLs*N)
105. .Loop_store_ZA:
106.   psel    pn11, pn9, p2.b[w13, 4]
107.   psel    pn12, pn9, p3.b[w13, 4]
108.   mova    { z0.b-z3.b }, za0h.b[w13, 4:7]
109.   st1w    { z0.s-z1.s }, pn11, [x10, x2, lsl #2]  // Store to c_ptr0+N
110.   st1w    { z2.s-z3.s }, pn12, [x10, x25, lsl #2] // Store to c_ptr0+(SVLs+1)*N
111. 
112.   add     x10, x10, x2, lsl #3    // c_ptr0 += 2*N FP32 elms (bytes)
113.   add     w13, w13, #8
114. 
115.   psel    pn11, pn9, p2.b[w13, 0]
116.   psel    pn12, pn9, p3.b[w13, 0]
117.   mova    { z0.b-z3.b }, za0h.b[w13, 0:3]
118.   st1w    { z0.s-z1.s }, pn11, [x10]               // Store to c_ptr0
119.   st1w    { z2.s-z3.s }, pn12, [x10, x23, lsl #2]  // Store to c_ptr0+SVLs*N
120.   cmp     w13, w6
121.   b.mi    .Loop_store_ZA
122. 
123.   psel    pn11, pn9, p2.b[w13, 4]
124.   psel    pn12, pn9, p3.b[w13, 4]
125.   mova    { z0.b-z3.b }, za0h.b[w13, 4:7]
126.   st1w    { z0.s-z1.s }, pn11, [x10, x2, lsl #2]  // Store to c_ptr0+N
127.   st1w    { z2.s-z3.s }, pn12, [x10, x25, lsl #2] // Store to c_ptr0+(SVLs+1)*N
128. 
129.   addvl   x9, x9, #2
130.   addvl   x16, x16, #2            // b_base += 2*SVLb (bytes)
131.   whilelt pn9.b, x16, x11, vlx2   // tile predicate (N dimension)
132.   b.first .Loop_N
133. 
134.   add     x3, x3, x22, lsl #3     // a_base += 2*SVLs*K FP32 elms (bytes)
135.   add     x5, x5, x23, lsl #3     // c_base += 2*SVLs*N FP32 elms (bytes)
136.   incw    x12, all, mul #2        // M loop counter += 2* SVLs
137.   whilelt pn8.s, x12, x0, vlx2    // tiles predicate (M dimension)
138.   b.first .Loop_M
139. 
140.   smstop
141. 
142.   ldp     x23, x24, [sp, #32]
143.   ldp     x21, x22, [sp, #16]
144.   ldp     x19, x20, [sp], #48
145. 
146.   ret
```

The matmul_opt function does the following:

Iterates over the columns of the matLeft matrix (matLeft_mod buffer) and the rows of the matRight matrix (matRight buffer)
Calculates the outer products
Stores the result in matResult_opt.
The code in this example uses the fact that multiplying two matrices together is the same as summing the outer products for each row and column in turn. That is, given a matrix matLeft with dimensions M x K and a matrix matRight with dimensions K x N, the result of multiplying matLeft and matRight produces a matrix matResult_opt with dimensions M x N. The result is calculated as follows:
```psuedo_code
for k = 1 to K:
  // Partial products computation
  for m = 1 to M: // m+=2
    for n = 1 to N: // n+=2
      // Inner loops unrolled by 2
      // A OP B is equal to
      matResult_opt(m ,n )   += matLeft(m,k)   x matRight(k,n)
      matResult_opt(m+1,n )  += matLeft(m+1,k) x matRight(k,n)
      matResult_opt(m ,n+1)  += matLeft(m,k)   x matRight(k,n+1)
      matResult_opt(m+1,n+1) += matLeft(m+1,k) x matRight(k,n+1)
```
OP represents the outer product operation.

The matmul_opt function uses SME2 functionality that calculates the outer product of two vectors using a single instruction, storing the results in two-dimensional ZA matrix tiles as shown in Calculating outer product, 4 tiles at a time.

Figure 1. Calculating outer product, 4 tiles at a time
```figure
             matRight
 [b1] [  ] [  ] [  ] [b2] [  ] [  ] [  ]
 [  ] [  ] [  ] [  ] [  ] [  ] [  ] [  ]
 [  ] [  ] [  ] [  ] [  ] [  ] [  ] [  ]
 [  ] [  ] [  ] [  ] [  ] [  ] [  ] [  ]
 [  ] [  ] [  ] [  ] [  ] [  ] [  ] [  ]
 [  ] [  ] [  ] [  ] [  ] [  ] [  ] [  ]

         matLeft
 [a1] [  ] [  ] [  ] [  ] [  ]
 [  ] [  ] [  ] [  ] [  ] [  ]
 [  ] [  ] [  ] [  ] [  ] [  ]
 [  ] [  ] [  ] [  ] [  ] [  ]
 [a2] [  ] [  ] [  ] [  ] [  ]
 [  ] [  ] [  ] [  ] [  ] [  ]
 [  ] [  ] [  ] [  ] [  ] [  ]
 [  ] [  ] [  ] [  ] [  ] [  ]
 [  ] [  ] [  ] [  ] [  ] [  ]
 [  ] [  ] [  ] [  ] [  ] [  ]

       matResult_opt
 [ a1 OP b1 ] [ a1 OP b2 ]
 [ a2 OP b1 ] [ a2 OP b2 ]
 [          ] [          ]
 [          ] [          ]
```

matmul_opt function details

This section describes how the matmul_opt function operates, looking at sections of the code in turn.

Lines 2-4:
The code starts by saving registers x19 through x24 to the stack. These registers are restored at the end of the matmul_opt function. The Procedure Call Standard for the Arm 64-bit Architecture defines registers x19 through x28 as callee-saved registers, so the matmul_opt function must preserve the values of the registers it uses in this range.
Line 6:
Enters Streaming SVE mode and enables the ZA array storage.
Line 18:
Sets a 32-bit element predicate-as-counter pn8. If the numerical value of x12 is less than the value of base address of matLeft_mod with offset 2 * SVLs (because of vlx2), set the active counter of pn8 as the value of x12, otherwise set as vlx2.
Lines 21-26 and 134-138 (Loop_M)
Loop_M iterates over the M dimension in blocks of 2 x SVLs, that is rows in the original matLeft matrix.
The pext instruction in line 23 creates predicate-as-mask equivalent predicates, p2 and p3, from the predicate-as-counter register pn8. p2 corresponds to the first SVL, and p3 to the second SVL. These predicate registers control which columns in matLeft_mod are processed in each iteration.
At the end of Loop_M, lines 134-135 increment the pointers to the current rows in matLeft_mod and matResult by the length of the processed data.
Successive iterations of Loop_M update the predicate-as-counter pn8 until all M rows have been processed.
Note

Because the matLeft_mod is rearranged from matLeft, the columns of matLeft_mod are extracted from the rows of matLeft.
Lines 28-45 and 129-132 (Loop_N)
Loop_N iterates over the N dimension by a block of 2 * SVLs column. N is the number of columns in the matRight matrix.
The pext instruction sets a pair of predicate registers, p0 for the first SVLs columns and p1 for the second SVLs columns, from the predicate-as-counter register pn9. These predicate registers control which columns in matRight are processed in each iteration.
Each iteration of Loop_N clears the ZA array by zeroing all elements. The zero {za} instruction in line 40 does this, with za indicating that the whole ZA array is zeroed.
At the end of Loop_N, lines 129-130 increment the pointers to the current columns in matRight and matResult by the length of the processed data.
Successive iterations of Loop_N update the predicate-as-counter pn9 until all N columns in matRight have been processed.
Lines 47-104 (Loop_K)
Loop_K iterates over the K dimension, that is rows in the matRight and matLeft_mod matrices, computing a sub-block of the result matrix, measuring (2 * SVLs) x (2 * SVLs). These dimensions fit the four 32-bit ZA tiles used to store the results.
The code uses loop unrolling to improve efficiency. Each Loop_K iteration processes two k values, where k=1..K, so that two products are accumulated to each result for each loop iteration.
The ld1w instructions load matrix data from memory to Z vector registers as follows:
Z1 and Z0 contain the first SVLs elements from the M dimension of matLeft for k and k+1, Z0.s = matLeft[k, 0:SVLs-1] and Z1.s = matLeft[k+1, 0:SVLs-1].
Z5 and Z4 contain the second SVLs elements from the M dimension of matLeft for k and k+1, Z4.s = matLeft[SVLs:2*SVLs-1, k] and Z5.s = matLeft[SVLs:2*SVLs-1, k+1].
Z2 and Z3 contain the 2xSVLs elements from the N dimension of matRight for k, Z2.s = matRight[k, 0:SVLs-1] and Z3.s = matRight[k, SVLs:2*SVLs-1].
Z6 and Z7 contain the 2xSVLs elements from the N dimension of matRight for k+1, Z6.s = matRight[k+1, 0:SVLs-1] and Z7.s = matRight[k+1, SVLs:2*SVLs-1].
These ld1w instructions use predicated 2-vector loads to load two Z registers at a time.
The single-precision floating-point fmopa instructions operate on the 32-bit ZA0, ZA1, ZA2, and ZA3 tiles to compute the outer product of the left and right matrix subblocks as follows:
ZA0 contains the outer product: 1st SVLs from M (OP) 1st SVLs from N
ZA1 contains the outer product: 1st SVLs from M (OP) 2nd SVLs from N
ZA2 contains the outer product: 2nd SVLs from M (OP) 1st SVLs from N
ZA3 contains the outer product: 2nd SVLs from M (OP) 2nd SVLs from N
Each of these fmopa instructions are independently predicated, enabling 2D predication of tile results.
Loads are reused, so that 8 fmopa instructions consume 8 loaded vectors (2 for each of left and right matrix per one k), and the load-to-multiply ratio is perfectly balanced in the loop.
Accumulating these successive outer products takes advantage of the fact that multiplying two matrices together is the same as summing the outer products for each row and column in turn.
In line 81, note that the function does not use a dedicated loop counter. The code uses the value of left matrix pointer, x7, as both the left matrix load address, and also to determine the Loop_k exit condition.
Line 57
The psel pn11, pn10, p3.s[w14, 0] instruction sets the predicate-as-counter pn11 to either all false or all true, depending on the value of the first element of p3. If the first element of p3 is false, then the second SVLs of the matLeft column are all zero, so there is no need to load the data. This situation can occur in the final iteration if the number of rows is not exactly divisible by 2 x SVLs.
Lines 105-127 (Loop_store_ZA)
Finally, all that is required is to store the results to the matResult array for each ZA tile.
The data in the ZA tiles is first transferred from the four tiles to four Z vector registers, using the 8-bit indexed mova instruction. Results are stored to matResult by updating elements in a segment of memory with size ((2 * SVLs) x (2 * SVLs)). Each row in this segment contains 2 * SVLs elements, formed by combining consecutive slices from 2 tiles:
The first half of the memory segment contains rows formed by combining slices from the ZA0 and ZA1 tiles.
The second half of the memory segment contains rows formed by combining slices from the ZA2 and ZA3 tiles.
Consecutive vectors are stored to memory using the 2 vector store instruction st1w, predicated using the vlx2 predicate-as-counter.
The mova instruction at line 108 (and elsewhere, due to loop unrolling) illustrates an SME2 coding optimization. The results were obtained in a ZA array using an outer product instruction to 32-bit ZA tiles. However, the mova instructions operate on 8-bit tiles (ZA0.B). When 4 consecutive horizontal slices of an 8-bit tile ZA0.B are moved, these slices correspond to one horizontal slice from 4 different 32-bit tiles ZA0.S, ZA1.S, ZA2.S, ZA3.S.
The psel instructions at lines 106-107 (and elsewhere, due to loop unrolling) enable predication of the result matrix stores. This means that the Loop_store_ZA loop can cope with situations where the number of rows is not an exact multiple of SVLs by ignoring the leftover rows.
The Loop_store_ZA function iterates over SVLb horizontal row slices from the ZA0.b tile, with each iteration extracting four row slices corresponding to the four ZAx.S 32b tiles.
Line 140
Exit Streaming SVE mode, and disable the ZA storage.
Lines 142-146
Restore the callee-saved registers and return from the matmul_opt function.