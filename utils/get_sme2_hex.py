# In SME2, the encoding for 2-way and 4-way 8-bit smopa/umopa.
# Let's check some known bit patterns or invoke GNU assembler via a docker container if needed.
# Since we don't have gcc/gas here easily configured for SME2, let's look up the SME2 XML/reference or make a best guess based on the 32-bit FMOPA 2-way:
# We already found:
# fmopa 2v: c1 a2 18 18 -> 11000001 10100010 00011000 00011000
# sdot 2v (into ZA): c1 a2 14 00 -> 11000001 10100010 00010100 00000000

# SME2 SMOPA (multi-vector)
# smopa <ZAda>.s[<Wv>, <imm>, vgx2], { <Zn1>.b-<Zn2>.b }, { <Zm1>.b-<Zm2>.b }
# umopa ...
pass
