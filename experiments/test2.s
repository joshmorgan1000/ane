smopa za.s[w0, 0], p0/m, p1/m, {z0.b, z1.b}, {z2.b, z3.b}
smopa za.s[w0, 0], {z0.b - z1.b}, {z2.b - z3.b}
smopa za0.s, p0/m, p1/m, {z0.b-z1.b}, {z2.b-z3.b}
sdot za0.s, {z0.b-z1.b}, {z2.b-z3.b}
smlal za0.s, {z0.b-z3.b}, {z4.b-z7.b}
