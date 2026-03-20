def decode(h):
    return bin(int(h, 16))[2:].zfill(32)

print("fmopa 2-way (guess?)", f"{decode('0x81220400')}")
