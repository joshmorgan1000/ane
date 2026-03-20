#!/bin/bash
cat > smopa_test.s << 'EOF'
.text
// 8-bit SMOPA (I8 -> I32)
smopa za0.s, p0/m, p0/m, z0.b, z1.b
smopa za3.s, p0/m, p0/m, z0.b, z1.b

// Wide register MOVA ops (za0.b - za3.b)
// Tile slice moves (from SME1) - or maybe user means multi-vector MOVA/LUTI?
mova za0.b[w12, 0], p0/m, z0.b
mova za3.b[w12, 0], p0/m, z0.b

// 2-way and 4-way fmopa (FP32)
fmopa za.s[w8,0,vgx2], {z0.s-z1.s}, {z2.s-z3.s}
fmopa za.s[w8,0,vgx4], {z0.s-z3.s}, {z4.s-z7.s}

// 2-way and 4-way smopa (I8 -> I32)
smopa za.s[w8,0,vgx2], {z0.b-z1.b}, {z2.b-z3.b}
smopa za.s[w8,0,vgx4], {z0.b-z3.b}, {z4.b-z7.b}

// 2-way and 4-way umopa (I8 -> I32)
umopa za.s[w8,0,vgx2], {z0.b-z1.b}, {z2.b-z3.b}
umopa za.s[w8,0,vgx4], {z0.b-z3.b}, {z4.b-z7.b}

// Multi-vector moves to/from ZA
read  {z0.b-z3.b}, za[w8, 0]
write za[w8, 0], {z0.b-z3.b}
EOF
clang -c -target arm64-apple-macos smopa_test.s -mattr=+sme,+sme2
objdump -d smopa_test.o
