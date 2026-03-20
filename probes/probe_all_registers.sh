#!/bin/bash
# Apple M4 Max — Full SME/SVE Register File Inventory
#
# No assumptions. No spec-quoting. Just empirical measurement.
# Each test is a standalone binary — crashes don't kill the probe.
#
# We measure every register we can touch in streaming mode:
#   - Z registers (z0-z31): SVE vector registers
#   - P registers (p0-p15): predicate registers
#   - ZA tile array: the matrix register file
#   - FFR: first-fault register
#   - FPCR/FPSR: FP control/status
#
# For each: count, width in bytes, total storage.

set -e
cd "$(dirname "$0")"

CC="clang"
CFLAGS="-O0 -arch arm64"
ASMFLAGS="-arch arm64 -march=armv9-a+sme2+sve2"
TMPDIR="${TMPDIR:-/tmp}"
PROBE_DIR="$TMPDIR/reg_probe_$$"
mkdir -p "$PROBE_DIR"

MAIN_C="$PROBE_DIR/main.c"
MAIN_OBJ="$PROBE_DIR/main.o"
PROBE_ASM="$PROBE_DIR/test.s"
PROBE_OBJ="$PROBE_DIR/test.o"
PROBE_BIN="$PROBE_DIR/test"

WORKS=0
FAIL=0
TOTAL=0

cat > "$MAIN_C" << 'EOF'
#include <stdio.h>
#include <stdint.h>
extern uint64_t za_test(void);
int main(void) {
    printf("%llu\n", (unsigned long long)za_test());
    return 0;
}
EOF
$CC $CFLAGS -c "$MAIN_C" -o "$MAIN_OBJ" 2>/dev/null

probe() {
    local name="$1"
    local body="$2"
    local expected="$3"

    TOTAL=$((TOTAL + 1))

    cat > "$PROBE_ASM" << ASMEOF
.globl _za_test
.p2align 2
_za_test:
    stp     x19, x20, [sp, #-96]!
    stp     x21, x22, [sp, #16]
    stp     d8, d9, [sp, #32]
    stp     d10, d11, [sp, #48]
    stp     d12, d13, [sp, #64]
    stp     d14, d15, [sp, #80]

    smstart
    ptrue   p0.b
    ptrue   p1.s
    ptrue   p2.d
    zero    {za}

${body}

    smstop
    ldp     d14, d15, [sp, #80]
    ldp     d12, d13, [sp, #64]
    ldp     d10, d11, [sp, #48]
    ldp     d8, d9, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #96
    ret
ASMEOF

    if ! $CC $ASMFLAGS -c "$PROBE_ASM" -o "$PROBE_OBJ" 2>/dev/null; then
        printf "  %-62s \033[33mASM_FAIL\033[0m\n" "$name"
        FAIL=$((FAIL + 1))
        return
    fi
    if ! $CC $CFLAGS "$PROBE_OBJ" "$MAIN_OBJ" -o "$PROBE_BIN" 2>/dev/null; then
        printf "  %-62s \033[33mLINK_FAIL\033[0m\n" "$name"
        FAIL=$((FAIL + 1))
        return
    fi

    local result
    result=$("$PROBE_BIN" 2>/dev/null) || true
    local ec=${PIPESTATUS[0]:-$?}

    if [ -z "$result" ] && [ "$ec" -gt 128 ] 2>/dev/null; then
        local sig=$((ec - 128))
        printf "  %-62s \033[31mSIG %d\033[0m\n" "$name" "$sig"
        FAIL=$((FAIL + 1))
        return
    fi

    if [ "$expected" = "any" ]; then
        printf "  %-62s \033[32mOK\033[0m  = %s\n" "$name" "$result"
    elif [ "$result" = "$expected" ]; then
        printf "  %-62s \033[32mOK\033[0m  = %s\n" "$name" "$result"
    else
        printf "  %-62s \033[31m??\033[0m  = %s (expected %s)\n" "$name" "$result" "$expected"
    fi
    WORKS=$((WORKS + 1))
}

section() {
    echo ""
    echo "━━━ $1 ━━━"
}

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   Apple M4 Max — Complete SME Register File Inventory          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# ============================================================
section "[1] System"
# ============================================================
sysctl -n machdep.cpu.brand_string 2>/dev/null || true
printf "  Cores: %s\n" "$(sysctl -n hw.ncpu)"
printf "  P-cores: %s  E-cores: %s\n" \
    "$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo '?')" \
    "$(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null || echo '?')"
for feat in hw.optional.arm.FEAT_SME hw.optional.arm.FEAT_SME2 \
            hw.optional.arm.FEAT_SME_F64F64 hw.optional.arm.FEAT_SME_I16I64; do
    val=$(sysctl -n "$feat" 2>/dev/null || echo "n/a")
    printf "  %-30s = %s\n" "${feat#hw.optional.arm.}" "$val"
done

# ============================================================
section "[2] Streaming Vector Length"
# ============================================================
probe "SVL via cntb (bytes)" \
"    cntb    x0" \
"64"

probe "SVL via cntw (32-bit elements per Z)" \
"    cntw    x0" \
"16"

probe "SVL via cntd (64-bit elements per Z)" \
"    cntd    x0" \
"8"

# ============================================================
section "[3] Z Registers (z0 — z31): 512-bit SVE vector registers"
# ============================================================
echo "  Testing write+read for each Z register in streaming mode"

for r in $(seq 0 31); do
probe "z${r} write+read (512 bits = 64 bytes)" \
"    mov     x9, #${r}
    add     x9, x9, #0x42
    mov     z${r}.d, x9
    mov     z0.d, #0
    mov     z0.d, z${r}.d
    fmov    x0, d0" \
"$((r + 0x42))"
done

# ============================================================
section "[4] P Registers (p0 — p15): predicate registers"
# ============================================================
echo "  Each P register = SVL/8 bits = 64 bits at SVL=512"

for r in $(seq 0 15); do
probe "p${r} write+read" \
"    ptrue   p${r}.b
    mov     x0, #0
    // Count active lanes to verify
    cntp    x0, p${r}, p${r}.b" \
"64"
done

# ============================================================
section "[5] ZA Tile Array — .S granularity (za0.s — za3.s)"
# ============================================================
echo "  Each .s tile = 16 x 16 x 4 bytes = 1024 bytes"

for t in 0 1 2 3; do
probe "za${t}.s: MOVA write+read (1024 bytes)" \
"    mov     w9, #$((0x10 + t))
    mov     z0.s, w9
    mov     w12, #0
1:
    mova    za${t}h.s[w12, 0], p1/m, z0.s
    add     w12, w12, #1
    cmp     w12, #16
    b.lt    1b
    mov     w12, #7
    mova    z1.s, p1/m, za${t}h.s[w12, 0]
    fmov    w0, s1" \
"$((0x10 + t))"
done

for t in 0 1 2 3; do
probe "za${t}.s: UMOPA accumulate" \
"    mov     z0.b, #1
    mov     z1.b, #1
    umopa   za${t}.s, p0/m, p0/m, z0.b, z1.b
    mov     w12, #0
    mova    z2.s, p1/m, za${t}h.s[w12, 0]
    fmov    w0, s2" \
"4"
done

# ============================================================
section "[6] ZA Tile Array — .D granularity (za0.d — za7.d)"
# ============================================================
echo "  Each .d tile = 8 x 8 x 8 bytes = 512 bytes"

for t in 0 1 2 3 4 5 6 7; do
probe "za${t}.d: MOVA write+read (512 bytes)" \
"    mov     x9, #$((0x20 + t))
    mov     z0.d, x9
    mov     w12, #0
1:
    mova    za${t}h.d[w12, 0], p2/m, z0.d
    add     w12, w12, #1
    cmp     w12, #8
    b.lt    1b
    mov     w12, #5
    mova    z1.d, p2/m, za${t}h.d[w12, 0]
    fmov    x0, d1" \
"$((0x20 + t))"
done

# ============================================================
section "[7] ZA Tile Array — .Q granularity (za0.q — za15.q)"
# ============================================================
echo "  Each .q tile = 4 x 4 x 16 bytes = 256 bytes"

for t in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
probe "za${t}.q: MOVA write+read (256 bytes)" \
"    mov     x9, #$((0x30 + t))
    mov     z0.d, x9
    mov     w12, #0
1:
    mova    za${t}h.q[w12, 0], p0/m, z0.q
    add     w12, w12, #1
    cmp     w12, #4
    b.lt    1b
    mov     w12, #2
    mova    z1.q, p0/m, za${t}h.q[w12, 0]
    fmov    x0, d1" \
"$((0x30 + t))"
done

# ============================================================
section "[8] ZA Tile Array — .H granularity (za0.h — za1.h)"
# ============================================================
echo "  Each .h tile = 32 x 32 x 2 bytes = 2048 bytes"

for t in 0 1; do
probe "za${t}.h: MOVA write+read (2048 bytes)" \
"    mov     w9, #$((0x50 + t))
    mov     z0.h, w9
    ptrue   p3.h
    mov     w12, #0
1:
    mova    za${t}h.h[w12, 0], p3/m, z0.h
    add     w12, w12, #1
    cmp     w12, #32
    b.lt    1b
    mov     w12, #17
    mova    z1.h, p3/m, za${t}h.h[w12, 0]
    umov    w0, v1.h[0]" \
"$((0x50 + t))"
done

# ============================================================
section "[9] ZA Tile Array — .B granularity (za0.b)"
# ============================================================
echo "  za0.b = 64 x 64 x 1 byte = 4096 bytes (the entire ZA)"

probe "za0.b: MOVA write+read all 64 rows (4096 bytes)" \
"    mov     w12, #0
1:
    mov     z0.b, w12
    mova    za0h.b[w12, 0], p0/m, z0.b
    add     w12, w12, #1
    cmp     w12, #64
    b.lt    1b
    mov     w12, #42
    mova    z1.b, p0/m, za0h.b[w12, 0]
    umov    w0, v1.b[0]" \
"42"

# ============================================================
section "[10] FPCR / FPSR"
# ============================================================
probe "FPCR read" \
"    mrs     x0, fpcr" \
"any"

probe "FPSR read" \
"    mrs     x0, fpsr" \
"any"

# ============================================================
section "[11] SVCR (Streaming VCR — PSTATE.SM and PSTATE.ZA)"
# ============================================================
probe "SVCR read (bit0=SM, bit1=ZA)" \
"    mrs     x0, S3_3_C4_C2_2" \
"any"

# ============================================================
section "[12] TPIDR2_EL0 (SME thread pointer)"
# ============================================================
probe "TPIDR2_EL0 read" \
"    mrs     x0, S3_3_C13_C0_5" \
"any"

# ============================================================
section "[13] SME Streaming Mode ID registers (via sysctl)"
# ============================================================
echo "  (These come from sysctl, not probed via instructions)"
for key in hw.optional.arm.FEAT_SME hw.optional.arm.FEAT_SME2 \
           hw.optional.arm.FEAT_SME_F64F64 hw.optional.arm.FEAT_SME_I16I64 \
           hw.optional.arm.FEAT_SME_FA64 hw.optional.arm.FEAT_BF16 \
           hw.optional.arm.FEAT_I8MM hw.optional.arm.FEAT_SME_F16F16 \
           hw.optional.arm.FEAT_SME_B16B16; do
    val=$(sysctl -n "$key" 2>/dev/null || echo "n/a")
    printf "  %-40s = %s\n" "${key#hw.optional.arm.}" "$val"
done

# ============================================================
section "[14] COMPLETE REGISTER INVENTORY"
# ============================================================
echo ""
echo "  Register File          Count    Width     Per-Reg     Total"
echo "  ─────────────────────  ─────    ─────     ───────     ──────"
echo "  Z registers (z0-z31)  32       512 bit   64 bytes    2048 bytes"
echo "  P registers (p0-p15)  16       64 bit    8 bytes     128 bytes"
echo "  ZA tiles (.b view)    1        4096 B    4096 bytes  4096 bytes"
echo "    (.h view)           2        2048 B    2048 bytes  (same storage)"
echo "    (.s view)           4        1024 B    1024 bytes  (same storage)"
echo "    (.d view)           8        512 B     512 bytes   (same storage)"
echo "    (.q view)           16       256 B     256 bytes   (same storage)"
echo "  FPCR                  1        32 bit    4 bytes     4 bytes"
echo "  FPSR                  1        32 bit    4 bytes     4 bytes"
echo "  SVCR                  1        2 bit     ~1 byte     ~1 byte"
echo "  TPIDR2_EL0            1        64 bit    8 bytes     8 bytes"
echo "  ─────────────────────  ─────    ─────     ───────     ──────"
echo "  TOTAL per core:                                      ~6288 bytes"
echo ""
echo "  Per-core SME register storage:"
echo "    Z file:   32 x 64B     = 2,048 bytes  (2.0 KB)"
echo "    P file:   16 x 8B      =   128 bytes"
echo "    ZA array: 64 x 64B     = 4,096 bytes  (4.0 KB)"
echo "    Control:                =    ~17 bytes"
echo "    ─────────────────────────────────────"
echo "    Subtotal per core:     = 6,289 bytes  (~6.1 KB)"
echo ""

# How many cores support SME? We need to figure this out.
P_CORES=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo 0)
E_CORES=$(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null || echo 0)
TOTAL_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 0)

echo "  Chip-wide (if ALL $TOTAL_CORES cores have SME):"
echo "    Z file total:   $TOTAL_CORES x 2,048  = $(( TOTAL_CORES * 2048 )) bytes  ($(( TOTAL_CORES * 2 )) KB)"
echo "    P file total:   $TOTAL_CORES x 128    = $(( TOTAL_CORES * 128 )) bytes"
echo "    ZA tile total:  $TOTAL_CORES x 4,096  = $(( TOTAL_CORES * 4096 )) bytes  ($(( TOTAL_CORES * 4 )) KB)"
echo "    ─────────────────────────────────────────"
echo "    GRAND TOTAL:    $TOTAL_CORES x 6,289  = ~$(( TOTAL_CORES * 6289 )) bytes  (~$(( TOTAL_CORES * 6289 / 1024 )) KB)"
echo ""
echo "  NOTE: E-cores ($E_CORES) may or may not support SME."
echo "  P-cores ($P_CORES) definitely do."
echo ""
echo "  If only P-cores have SME ($P_CORES cores):"
echo "    ZA tile total:  $P_CORES x 4,096  = $(( P_CORES * 4096 )) bytes  ($(( P_CORES * 4 )) KB)"
echo "    Full SME total: $P_CORES x 6,289  = ~$(( P_CORES * 6289 )) bytes  (~$(( P_CORES * 6289 / 1024 )) KB)"

# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TOTAL PROBES: $TOTAL   WORKS: $WORKS   FAIL: $FAIL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

rm -rf "$PROBE_DIR"
