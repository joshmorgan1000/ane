#!/bin/bash
# Comprehensive SME/SVE2/SSVE Instruction Probe
# Intended for use with Apple Silicon M4 or M5 chips
#
# References:
#   https://www.scs.stanford.edu/~zyedidia/arm64/sveindex.html
#   https://www.scs.stanford.edu/~zyedidia/arm64/mortlachindex.html
#
# Usage: ./probe_instructions.sh [--skip ops] [--skip tp]
#   --skip ops    Skip instruction probes, run only throughput tests
#   --skip tp     Skip throughput tests, run only instruction probes
# Output: table of instruction -> WORKS / SIGILL / COMPILE_FAIL
#
# Author: Josh Morgan (@joshmorgan1000 on GitHub) with help from Claude and Gemini
# Released under the MIT License
SKIP_OPS=0
SKIP_TP=0
for arg in "$@"; do
    case "$arg" in
        ops) SKIP_OPS=1 ;;
        tp)  SKIP_TP=1 ;;
    esac
done
# Handle --skip X
while [ $# -gt 0 ]; do
    case "$1" in
        --skip)
            shift
            case "$1" in
                ops) SKIP_OPS=1 ;;
                tp)  SKIP_TP=1 ;;
            esac
            ;;
    esac
    shift
done
set +e
cd "$(dirname "$0")"
CYAN="\033[36m"
NC="\033[0m"
GREEN="\033[32m"
echo -e " "
echo -e "\033[35m______________________________________________________\033[0m"
echo -e "\033[36m     __     _____ ______  _______ __   _ _______\033[0m"
echo -e "\033[36m     |        |   |_____] |_____| | \  | |______\033[0m"
echo -e "\033[36m     |_____ __|__ |_____] |     | |  \_| |______\033[0m"
echo -e "\033[35m──────────────────────────────────────────────────────\033[0m"
echo -e "\033[34m        SME Instructions and Registers Probe\033[0m"
echo " "
echo " To skip the instruction probes and run only throughput tests, use: $0 --skip ops"
echo " To skip the throughput tests and run only instruction probes, use: $0 --skip tp"
set +e
cd "$(dirname "$0")"
CC="clang"
CFLAGS="-O0 -arch arm64"
ASMFLAGS="-arch arm64 -march=armv9-a+sme2+sve2+sme-lutv2+sme-f8f32+sme-f8f16+sme-f16f16+sme-i16i64+sme-f64f64+sme2p1+b16b16+fp8+i8mm+sve2-aes+sve2-sm4+sve2-sha3+sve2-bitperm+f32mm"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROBE_DIR="$SCRIPT_DIR/../build/probe"
mkdir -p "$PROBE_DIR"
MAIN_C="$PROBE_DIR/main.c"
MAIN_OBJ="$PROBE_DIR/main.o"
PROBE_ASM="$PROBE_DIR/test.s"
PROBE_OBJ="$PROBE_DIR/test.o"
PROBE_BIN="$PROBE_DIR/test"
cat > "$MAIN_C" << 'EOF'
#include <stdio.h>
extern void instr_probe(void);
int main(void) {
    instr_probe();
    return 0;
}
EOF
$CC $CFLAGS -c "$MAIN_C" -o "$MAIN_OBJ" 2>/dev/null
WORKS_COUNT=0
FAIL_COUNT=0
EXPECTED_FAIL_COUNT=0
COMPILE_FAIL_COUNT=0
TOTAL_COUNT=0
FAIL_LOG="$PROBE_DIR/failures.log"
rm -f "$FAIL_LOG"
# =============================================================================
# probe_one
# Params:
#   $1 = instruction body (e.g. "add z0.b, z1.b, z2.b")
#   $2 = mode ("sme" for streaming+ZA, "nosve" for bare SVE)
#   $3 = expected exit code (e.g. "132" for SIGILL), empty = expect ok
# Builds a small assembly program with the given instruction, as well as a
# small C harness to call it. Compiles the test, and then runs it as a
# subprocess, checking the exit code against the expected result. Logs results.
# =============================================================================
probe_one() {
    # Delete existing test
    rm -f "$PROBE_BIN" "$PROBE_BIN.s"
    local display="$1"
    local mode="${2:-sme}"  # sme = streaming+ZA (default), nosve = bare
    local expect="${3:-}"   # expected exit code (e.g. "132" for SIGILL), empty = expect ok
    # Strip parenthesized description to get clean assembly
    # Only strip trailing (description) — do NOT strip [x0] addressing modes
    local body
    body=$(echo "$1" | sed 's/ *([^)]*)$//')
    local comment
    comment=$(echo "$1" | grep -o '([^)]*)$' | sed 's/^(//;s/)$//')
    # For multi-line bodies, only show the last line (the actual instruction)
    local show_body
    show_body=$(echo "$body" | tail -1 | sed 's/^ *//')
    local mnemonic operands
    mnemonic=$(echo "$show_body" | awk '{print $1}')
    operands=$(echo "$show_body" | sed 's/^[^ ]* *//')
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    # [SME]/[OFF] tag
    if [ "$mode" = "sme" ]; then
        printf "[\033[32mSME\033[0m] "
    else
        printf "[\033[90mOFF\033[0m] "
    fi
    # Syntax highlight: mnemonic=bright yellow, registers=cyan, literals/indices=green
    printf "\033[93m%-10s\033[0m " "$mnemonic"
    python3 -c '
import re, sys
s = sys.argv[1]
C="\x1b[36m"; G="\x1b[32m"; R="\x1b[0m"
# Inside square brackets: registers cyan, literals green, punctuation plain
def color_bracket(m):
    inner = m.group(1)
    # x-registers inside brackets: x0, x30, xzr → cyan as whole token
    inner = re.sub(r"\b(x\d+|xzr)\b", C+r"\1"+R, inner)
    # w-registers: w8-w15 → cyan as whole token
    inner = re.sub(r"\b(w\d+)\b", C+r"\1"+R, inner)
    # VGx2/VGx4 → green
    inner = re.sub(r"(VGx[24])", G+r"\1"+R, inner)
    # Bare numbers and ranges (0, 0:3, 0:1) → green
    inner = re.sub(r"(?<=,)\s*([\d][\d:]*)", G+r"\1"+R, inner)
    inner = re.sub(r"^([\d][\d:]*)", G+r"\1"+R, inner)
    inner = re.sub(r"(?<= )([\d][\d:]*)", G+r"\1"+R, inner)
    return "["+inner+"]"
s = re.sub(r"\[([^\]]+)\]", color_bracket, s)
# Register groups {z0.s-z1.s} — names+suffix cyan, dots/dash/braces plain
s = re.sub(r"\{(z\d+)\.([bhdswq])-(z\d+)\.([bhdswq])\}", "{"+C+r"\1"+R+"."+C+r"\2"+R+"-"+C+r"\3"+R+"."+C+r"\4"+R+"}", s)
# ZA tile registers: za0h.s → name cyan, dot plain, suffix cyan
s = re.sub(r"\b(za\d*[hv]?)\.([bhdswq])", C+r"\1"+R+"."+C+r"\2"+R, s)
# Bare za (no suffix) → cyan
s = re.sub(r"\b(za)\b(?![\d.])", C+r"\1"+R, s)
# zt0 → cyan
s = re.sub(r"\b(zt0)", C+r"\1"+R, s)
# z-registers with suffix: z0.b → name cyan, dot plain, suffix cyan
s = re.sub(r"\b(z\d+)\.([bhdswq])", C+r"\1"+R+"."+C+r"\2"+R, s)
# Bare z-registers (no suffix): z2, z4 → cyan
s = re.sub(r"\b(z\d+)\b(?!\.)", C+r"\1"+R, s)
# Predicate registers: p0.b → name cyan, dot plain, suffix cyan; p0/m → name cyan, /m plain
s = re.sub(r"\b(pn?\d+)\.([bhdswq])", C+r"\1"+R+"."+C+r"\2"+R, s)
s = re.sub(r"\b(pn?\d+)(/[mz])", C+r"\1"+R+r"\2", s)
s = re.sub(r"\b(pn?\d+)\b(?![./])", C+r"\1"+R, s)
# GP registers outside brackets: x0-x30, xzr → cyan
s = re.sub(r"\b(x\d+|xzr)\b", C+r"\1"+R, s)
# w registers outside brackets: w0-w31 → cyan
s = re.sub(r"\b(w\d+)\b", C+r"\1"+R, s)
# Scalar FP/byte: s0, d0, h0, q0, b0 → cyan
s = re.sub(r"\b([bsdhq]\d+)\b", C+r"\1"+R, s)
# System registers: FPMR → cyan
s = re.sub(r"\b(FPMR)\b", C+r"\1"+R, s)
# Immediates: #N, #-N, #N.N → green
s = re.sub(r"(#-?[\d.]+)", G+r"\1"+R, s)
print(s, end="")
' "$operands"
    # Pad based on visible text length
    local visible_len=$(( 10 + 1 + ${#operands} ))
    local target=55
    local pad=$(( target - visible_len ))
    [ $pad -gt 0 ] && printf "%${pad}s" ""
    if [ -n "$comment" ]; then
        printf " \033[34m(%s)\033[0m" "$comment"
    fi
    printf " "
    # Write assembly test
    if [ "$mode" = "nosve" ]; then
        {
            echo '.globl _instr_probe'
            echo '.p2align 2'
            echo '_instr_probe:'
            echo "    $body"
            echo '    ret'
        } > "$PROBE_BIN.s"
    else
        {
            echo '.globl _instr_probe'
            echo '.p2align 2'
            echo '_instr_probe:'
            echo '    stp x19, x20, [sp, #-96]!'
            echo '    stp x21, x22, [sp, #16]'
            echo '    stp d8, d9, [sp, #32]'
            echo '    stp d10, d11, [sp, #48]'
            echo '    stp d12, d13, [sp, #64]'
            echo '    stp d14, d15, [sp, #80]'
            echo '    sub sp, sp, #4096'
            echo '    mov x0, sp'
            echo '    smstart'
            echo '    ptrue p0.b'
            echo '    ptrue p1.s'
            echo '    ptrue p2.d'
            echo '    zero {za}'
            echo "    $body"
            echo '    smstop'
            echo '    add sp, sp, #4096'
            echo '    ldp d14, d15, [sp, #80]'
            echo '    ldp d12, d13, [sp, #64]'
            echo '    ldp d10, d11, [sp, #48]'
            echo '    ldp d8, d9, [sp, #32]'
            echo '    ldp x21, x22, [sp, #16]'
            echo '    ldp x19, x20, [sp], #96'
            echo '    ret'
        } > "$PROBE_BIN.s"
    fi
    local compile_err
    compile_err=$($CC $ASMFLAGS -c "$PROBE_BIN.s" -o "$PROBE_OBJ" 2>&1)
    if [ $? -ne 0 ]; then
        local err_detail
        err_detail=$(echo "$compile_err" | grep -m1 "error:" | sed 's/.*error://' | head -c 80)
        printf "COMPILE_FAIL|%s\n" "$err_detail"
        COMPILE_FAIL_COUNT=$((COMPILE_FAIL_COUNT + 1))
        echo "COMPILE_FAIL | $display | $err_detail" >> "$FAIL_LOG"
        return
    fi
    compile_err=$($CC $CFLAGS -o "$PROBE_BIN" "$PROBE_OBJ" "$MAIN_OBJ" 2>&1)
    if [ $? -ne 0 ]; then
        printf "LINK_FAIL\n"
        COMPILE_FAIL_COUNT=$((COMPILE_FAIL_COUNT + 1))
        echo "LINK_FAIL    | $display" >> "$FAIL_LOG"
        return
    fi
    local run_output
    run_output=$("$PROBE_BIN" 2>&1)
    local exit_code=$?
    if [ -n "$expect" ]; then
        # We expect a specific exit code
        if [ $exit_code -eq "$expect" ]; then
            printf "\033[36mFAIL (expected, exit=%d)\033[0m\n" "$exit_code"
            EXPECTED_FAIL_COUNT=$((EXPECTED_FAIL_COUNT + 1))
        elif [ $exit_code -eq 0 ]; then
            printf "\033[33mUNEXPECTED OK\033[0m (expected exit=%s)\n" "$expect"
            echo "UNEXPECTED_OK | $display | expected exit=$expect" >> "$FAIL_LOG"
        else
            printf "\033[31mWRONG FAIL\033[0m (exit=%d, expected %s) %s\n" "$exit_code" "$expect" "$run_output"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "WRONG_FAIL   | $display | exit=$exit_code expected=$expect $run_output" >> "$FAIL_LOG"
        fi
    else
        # We expect success
        if [ $exit_code -eq 0 ]; then
            printf "[\033[92mOK\033[0m]\n"
            WORKS_COUNT=$((WORKS_COUNT + 1))
        else
            printf "\033[31mFAIL\033[0m (exit=%d) %s\n" "$exit_code" "$run_output"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "RUNTIME_FAIL | $display | exit=$exit_code $run_output" >> "$FAIL_LOG"
        fi
    fi
}
section() {
    echo ""
    echo -e "\033[35m━━━━━━━━\033[36m $1 \033[35m━━━━━━━━\033[0m"
}
# ============================================================
# [1] SYSTEM INFO
# ============================================================
section "System Info"
sysctl -n machdep.cpu.brand_string 2>/dev/null && true
echo "  Cores: $(sysctl -n hw.ncpu) total"
for feat in hw.optional.arm.FEAT_SME hw.optional.arm.FEAT_SME2 \
            hw.optional.arm.FEAT_SVE hw.optional.arm.FEAT_SVE2 \
            hw.optional.arm.FEAT_SME_F64F64 hw.optional.arm.FEAT_SME_I16I64 \
            hw.optional.arm.FEAT_SME_FA64 hw.optional.arm.FEAT_BF16 \
            hw.optional.arm.FEAT_I8MM hw.optional.arm.FEAT_SHA3 \
            hw.optional.arm.FEAT_AES hw.optional.arm.FEAT_SSVE_AES; do
    val=$(sysctl -n "$feat" 2>/dev/null || echo "n/a")
    label="${feat#hw.optional.arm.}"
    printf "  %-22s = %s\n" "$label" "$val"
done
if [ $SKIP_OPS -eq 0 ]; then

CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -o 'M[0-9]' | head -n 1)
# Some SME2.1 / B16B16 / F16F16 / LUT instructions are only on M5+
EXPECT_M5="132"
if [ "$CHIP" = "M5" ]; then
    EXPECT_M5=""
fi

section "Instruction Probes"
echo -e "The [\033[32mSME\033[0m] tag indicates instructions that require SME/ZA state (streaming mode)."
probe_one "abs z0.b, p0/m, z1.b (INT8 abs)"
probe_one "adclb z0.s, z1.s, z2.s (Add with carry long, bot)"
probe_one "adclt z0.s, z1.s, z2.s (Add with carry long, top)"
probe_one "add z0.b, z1.b, z2.b (INT8 add)"
probe_one "add z0.h, z1.h, z2.h (INT16 add)"
probe_one "add z0.s, z1.s, z2.s (INT32 add)"
probe_one "add z0.d, z1.d, z2.d (INT64 add)"
probe_one "add z0.s, z0.s, #1 (ADD imm)"
probe_one "add z0.s, p0/m, z0.s, z1.s (ADD pred)"
probe_one "add {z0.b-z1.b}, {z0.b-z1.b}, z2.b (2-vec add byte)"
probe_one "add {z0.b-z3.b}, {z0.b-z3.b}, z4.b (4-vec add byte)"
probe_one "add {z0.s-z1.s}, {z0.s-z1.s}, z2.s (2-vec add word)"
probe_one "add {z0.s-z3.s}, {z0.s-z3.s}, z4.s (4-vec add word)"
probe_one "add za.s[w8,0,VGx2], {z0.s-z1.s} (2v add to ZA)"
probe_one "add za.s[w8,0,VGx4], {z0.s-z3.s} (4v add to ZA)"
probe_one "add za.s[w8,0,VGx2], {z0.s-z1.s}, {z2.s-z3.s} (2v add arr result)"
probe_one "add za.s[w8,0,VGx4], {z0.s-z3.s}, {z4.s-z7.s} (4v add arr result)"
probe_one "add za.s[w8,0,VGx2], {z0.s-z1.s}, z2.s (2vx1 add arr result)"
probe_one "add za.s[w8,0,VGx4], {z0.s-z3.s}, z4.s (4vx1 add arr result)"
probe_one "add z0.b, z1.b, z2.b (NO SMSTART control)" nosve 132
probe_one "addha za0.s, p0/m, p0/m, z0.s (horiz add)"
probe_one "addha za0.d, p0/m, p0/m, z0.d (ADDHA 64b)"
probe_one "addhnb z0.b, z1.h, z2.h (narrow add hi lo)"
probe_one "addhnt z0.b, z1.h, z2.h (narrow add hi hi)"
probe_one "addp z0.s, p0/m, z0.s, z1.s (ADDP)"
probe_one "addpl x0, x0, #1 (Add multiple of pred length)"
probe_one "addspl x0, x0, #1 (Add multiple of streaming PL)" sm
probe_one "addsvl x0, x0, #1 (Add multiple of SVL)" sm
probe_one "addva za0.s, p0/m, p0/m, z0.s (vert add)"
probe_one "addva za0.d, p0/m, p0/m, z0.d (ADDVA 64b)"
probe_one "addvl x0, x0, #1 (Add multiple of vector length)"
probe_one "aesd z0.b, z0.b, z1.b (AES decrypt)" sme 132
probe_one "aese z0.b, z0.b, z1.b (AES encrypt)" sme 132
probe_one "aesimc z0.b, z0.b (AES inv mix)" sme 132
probe_one "aesmc z0.b, z0.b (AES mix col)" sme 132
probe_one "and z0.d, z1.d, z2.d (AND)"
probe_one "ands p0.b, p0/z, p0.b, p0.b (Pred AND, setting flags)"
probe_one "andv b0, p0, z1.b (Bitwise AND reduction)"
probe_one "asr z0.b, p0/m, z0.b, z1.b (arith right)"
probe_one "asr z0.s, z1.s, #1 (Arith shift right, immediate)"
probe_one "asrd z0.s, p0/m, z0.s, #1 (Arith shift right, divide)"
probe_one "asrr z0.s, p0/m, z0.s, z1.s (Arith shift right, reversed)"
probe_one "bcax z0.d, z0.d, z1.d, z2.d (bic-and-xor)"
probe_one "bdep z0.s, z1.s, z2.s (Bit deposit, scatter bits)" sme 132
probe_one "bext z0.s, z1.s, z2.s (Bit extract, gather bits)" sme 132
probe_one "bf1cvt {z0.h-z1.h}, z2.b (FP8->BF16 in-order)" sme 132
probe_one "bf1cvt {z0.h-z1.h}, z2.b (FP8->BF16 bf1cvt)" sme 132
probe_one "bf1cvtl {z0.h-z1.h}, z2.b (FP8->BF16 deinterleaved)" sme 132
probe_one "bf1cvtl {z0.h-z1.h}, z2.b (FP8->BF16 deintrl)" sme 132
probe_one "bf2cvt {z0.h-z1.h}, z2.b (FP8->BF16 in-order)" sme 132
probe_one "bf2cvt {z0.h-z1.h}, z2.b (FP8->BF16 bf2cvt)" sme 132
probe_one "bf2cvtl {z0.h-z1.h}, z2.b (FP8->BF16 deintrl2)" sme 132
probe_one "bfadd za.h[w8,0,VGx2], {z0.h-z1.h} (2v bfadd to ZA)" "sme" "$EXPECT_M5"
probe_one "bfadd za.h[w8,0,VGx4], {z0.h-z3.h} (4v bfadd to ZA)" "sme" "$EXPECT_M5"
probe_one "bfclamp {z0.h-z1.h}, z2.h, z3.h (2v bfclamp)" "sme" "$EXPECT_M5"
probe_one "bfclamp {z0.h-z3.h}, z2.h, z3.h (4v bfclamp)" "sme" "$EXPECT_M5"
probe_one "bfcvt z0.h, p0/m, z1.s (FP32->BF16)"
probe_one "bfcvt z0.b, {z0.h-z1.h} (BF16->FP8 cvt)" sme 132
probe_one "bfcvt z0.h, {z0.s-z1.s} (2v FP32->BF16)"
probe_one "bfcvtn z0.h, {z0.s-z1.s} (FP32->BF16 intrlv)"
probe_one "bfcvtnt z0.h, p0/m, z1.s (BF16 convert narrow top)"
probe_one "bfdot z0.s, z1.h, z2.h (BF16 dot)"
probe_one "bfdot z0.s, z1.h, z2.h[0] (BF16 dot product indexed)"
probe_one "bfdot za.s[w8,0,VGx2], {z0.h-z1.h}, z2.h (2vx1 bfdot ZA)"
probe_one "bfdot za.s[w8,0,VGx4], {z0.h-z3.h}, z4.h (4vx1 bfdot ZA)"
probe_one "bfdot za.s[w8,0,VGx2], {z0.h-z1.h}, {z2.h-z3.h} (2v bfdot ZA)"
probe_one "bfdot za.s[w8,0,VGx4], {z0.h-z3.h}, {z4.h-z7.h} (4v bfdot ZA)"
probe_one "bfdot za.s[w8,0,VGx2], {z0.h-z1.h}, z2.h[0] (2v bfdot idx)"
probe_one "bfdot za.s[w8,0,VGx4], {z0.h-z3.h}, z4.h[0] (4v bfdot idx)"
probe_one "bfmax {z0.h-z1.h}, {z0.h-z1.h}, z2.h (2v bfmax single)" "sme" "$EXPECT_M5"
probe_one "bfmax {z0.h-z3.h}, {z0.h-z3.h}, z4.h (4v bfmax single)" "sme" "$EXPECT_M5"
probe_one "bfmax {z0.h-z1.h}, {z0.h-z1.h}, {z2.h-z3.h} (2v bfmax)" "sme" "$EXPECT_M5"
probe_one "bfmax {z0.h-z3.h}, {z0.h-z3.h}, {z4.h-z7.h} (4v bfmax)" "sme" "$EXPECT_M5"
probe_one "bfmaxnm {z0.h-z1.h}, {z0.h-z1.h}, z2.h (2v bfmaxnm single)" "sme" "$EXPECT_M5"
probe_one "bfmaxnm {z0.h-z3.h}, {z0.h-z3.h}, z4.h (4v bfmaxnm single)" "sme" "$EXPECT_M5"
probe_one "bfmaxnm {z0.h-z1.h}, {z0.h-z1.h}, {z2.h-z3.h} (2v bfmaxnm)" "sme" "$EXPECT_M5"
probe_one "bfmaxnm {z0.h-z3.h}, {z0.h-z3.h}, {z4.h-z7.h} (4v bfmaxnm)" "sme" "$EXPECT_M5"
probe_one "bfmin {z0.h-z1.h}, {z0.h-z1.h}, z2.h (2v bfmin single)" "sme" "$EXPECT_M5"
probe_one "bfmin {z0.h-z3.h}, {z0.h-z3.h}, z4.h (4v bfmin single)" "sme" "$EXPECT_M5"
probe_one "bfmin {z0.h-z1.h}, {z0.h-z1.h}, {z2.h-z3.h} (2v bfmin)" "sme" "$EXPECT_M5"
probe_one "bfmin {z0.h-z3.h}, {z0.h-z3.h}, {z4.h-z7.h} (4v bfmin)" "sme" "$EXPECT_M5"
probe_one "bfminnm {z0.h-z1.h}, {z0.h-z1.h}, z2.h (2v bfminnm single)" "sme" "$EXPECT_M5"
probe_one "bfminnm {z0.h-z3.h}, {z0.h-z3.h}, z4.h (4v bfminnm single)" "sme" "$EXPECT_M5"
probe_one "bfminnm {z0.h-z1.h}, {z0.h-z1.h}, {z2.h-z3.h} (2v bfminnm)" "sme" "$EXPECT_M5"
probe_one "bfminnm {z0.h-z3.h}, {z0.h-z3.h}, {z4.h-z7.h} (4v bfminnm)" "sme" "$EXPECT_M5"
probe_one "bfmla za.h[w8,0,VGx2], {z0.h-z1.h}, z2.h (2vx1 bfmla ZA)" "sme" "$EXPECT_M5"
probe_one "bfmla za.h[w8,0,VGx4], {z0.h-z3.h}, z4.h (4vx1 bfmla ZA)" "sme" "$EXPECT_M5"
probe_one "bfmla za.h[w8,0,VGx2], {z0.h-z1.h}, {z2.h-z3.h} (2v bfmla ZA)" "sme" "$EXPECT_M5"
probe_one "bfmla za.h[w8,0,VGx4], {z0.h-z3.h}, {z4.h-z7.h} (4v bfmla ZA)" "sme" "$EXPECT_M5"
probe_one "bfmla za.h[w8,0,VGx2], {z0.h-z1.h}, z2.h[0] (2v bfmla idx)" "sme" "$EXPECT_M5"
probe_one "bfmla za.h[w8,0,VGx4], {z0.h-z3.h}, z4.h[0] (4v bfmla idx)" "sme" "$EXPECT_M5"
probe_one "bfmlal za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h (2vx1 bfmlal ZA)"
probe_one "bfmlal za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h (4vx1 bfmlal ZA)"
probe_one "bfmlal za.s[w8,0:1,VGx2], {z0.h-z1.h}, {z2.h-z3.h} (2v bfmlal ZA)"
probe_one "bfmlal za.s[w8,0:1,VGx4], {z0.h-z3.h}, {z4.h-z7.h} (4v bfmlal ZA)"
probe_one "bfmlal za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h[0] (2v bfmlal idx)"
probe_one "bfmlal za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h[0] (4v bfmlal idx)"
probe_one "bfmlalb z0.s, z1.h, z2.h (BF16 widening mul-add low)"
probe_one "bfmlalt z0.s, z1.h, z2.h (BF16 widening mul-add high)"
probe_one "bfmls za.h[w8,0,VGx2], {z0.h-z1.h}, z2.h (2vx1 bfmls ZA)" "sme" "$EXPECT_M5"
probe_one "bfmls za.h[w8,0,VGx4], {z0.h-z3.h}, z4.h (4vx1 bfmls ZA)" "sme" "$EXPECT_M5"
probe_one "bfmls za.h[w8,0,VGx2], {z0.h-z1.h}, {z2.h-z3.h} (2v bfmls ZA)" "sme" "$EXPECT_M5"
probe_one "bfmls za.h[w8,0,VGx4], {z0.h-z3.h}, {z4.h-z7.h} (4v bfmls ZA)" "sme" "$EXPECT_M5"
probe_one "bfmls za.h[w8,0,VGx2], {z0.h-z1.h}, z2.h[0] (2v bfmls idx)" "sme" "$EXPECT_M5"
probe_one "bfmls za.h[w8,0,VGx4], {z0.h-z3.h}, z4.h[0] (4v bfmls idx)" "sme" "$EXPECT_M5"
probe_one "bfmlsl za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h (2vx1 bfmlsl ZA)"
probe_one "bfmlsl za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h (4vx1 bfmlsl ZA)"
probe_one "bfmlsl za.s[w8,0:1,VGx2], {z0.h-z1.h}, {z2.h-z3.h} (2v bfmlsl ZA)"
probe_one "bfmlsl za.s[w8,0:1,VGx4], {z0.h-z3.h}, {z4.h-z7.h} (4v bfmlsl ZA)"
probe_one "bfmlsl za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h[0] (2v bfmlsl idx)"
probe_one "bfmlsl za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h[0] (4v bfmlsl idx)"
probe_one "bfmmla z0.s, z1.h, z2.h (BF16 matmul)" sme 132
probe_one "bfmopa za0.s, p0/m, p0/m, z0.h, z1.h (BF16 outer)"
probe_one "bfmopa za0.h, p0/m, p0/m, z0.h, z1.h (BF16 non-wid mopa)" "sme" "$EXPECT_M5"
probe_one "bfmops za0.s, p0/m, p0/m, z0.h, z1.h (BF16 mops)"
probe_one "bfmops za0.h, p0/m, p0/m, z0.h, z1.h (BF16 non-wid mops)" "sme" "$EXPECT_M5"
probe_one "bfsub za.h[w8,0,VGx2], {z0.h-z1.h} (2v bfsub from ZA)" "sme" "$EXPECT_M5"
probe_one "bfsub za.h[w8,0,VGx4], {z0.h-z3.h} (4v bfsub from ZA)" "sme" "$EXPECT_M5"
probe_one "bfvdot za.s[w8,0], {z0.h-z1.h}, z2.h[0] (BF16 vertical dot)"
probe_one "bgrp z0.s, z1.s, z2.s (Bit group, partition bits)" sme 132
probe_one "bic z0.d, z1.d, z2.d (AND NOT)"
probe_one "bic p0.b, p0/z, p0.b, p0.b (Pred bitwise AND-NOT)"
probe_one "bmopa za0.s, p0/m, p0/m, z0.s, z1.s (bitwise outer)"
probe_one "bmops za0.s, p0/m, p0/m, z0.s, z1.s (bitwise outer sub)"
probe_one "brka p0.b, p0/z, p0.b (Break after first active)"
probe_one "brkb p0.b, p0/z, p0.b (Break before first active)"
probe_one "brkn p0.b, p0/z, p0.b, p0.b (Propagate break to next part)"
probe_one "brkpa p0.b, p0/z, p0.b, p0.b (Break after from pair)"
probe_one "brkpb p0.b, p0/z, p0.b, p0.b (Break before from pair)"
probe_one "bsl z0.d, z0.d, z1.d, z2.d (bit select)"
probe_one "bsl1n z0.d, z0.d, z1.d, z2.d (Bit select, first negated)"
probe_one "bsl2n z0.d, z0.d, z1.d, z2.d (Bit select, second negated)"
probe_one "cadd z0.s, z0.s, z1.s, #90 (Complex integer add)"
probe_one "cdot z2.s, z0.b, z1.b, #0 (complex dot)"
probe_one "clasta z0.s, p0, z0.s, z1.s (Cond extract after last active)"
probe_one "clastb z0.s, p0, z0.s, z1.s (Cond extract last active)"
probe_one "cls z0.b, p0/m, z1.b (lead sign)"
probe_one "clz z0.b, p0/m, z1.b (lead zero)"
probe_one "cmla z0.s, z1.s, z2.s, #0 (Complex integer mul-add)"
probe_one "cmpeq p0.b, p0/z, z0.b, z1.b (==)"
probe_one "cmpge p0.b, p0/z, z0.b, z1.b (>= signed)"
probe_one "cmpgt p0.b, p0/z, z0.b, z1.b (>  signed)"
probe_one "cmphi p0.b, p0/z, z0.b, z1.b (>  unsigned)"
probe_one "cmphs p0.b, p0/z, z0.b, z1.b (>= unsigned)"
probe_one "cnot z0.s, p0/m, z1.s (CNOT)"
probe_one "cnt z0.b, p0/m, z1.b (popcount)"
probe_one "cntb x0 (count bytes)"
probe_one "cntd x0 (count dblwd)"
probe_one "cnth x0 (count halfs)"
probe_one "cntp x0, p0, p0.s (CNTP)"
probe_one "cntw x0 (count words)"
probe_one "compact z0.s, p0, z1.s (compact)" sme 132
probe_one "cpy z0.s, p0/m, #0 (copy imm)"
probe_one "ctermeq x0, x1 (Compare and terminate if eq)"
probe_one "ctermne x0, x1 (Compare and terminate if ne)"
probe_one "decb x0 (Decrement by byte count)"
probe_one "decd x0 (Decrement by doubleword count)"
probe_one "decd z0.d (Dec doubleword count, vector)"
probe_one "dech x0 (Decrement by halfword count)"
probe_one "dech z0.h (Dec halfword count, vector)"
probe_one "decp x0, p0.s (Dec by active pred count)"
probe_one "decw x0 (Decrement by word count)"
probe_one "decw z0.s (Dec word count, vector)"
probe_one "dup z0.s, #0 (broadcast 0)"
probe_one "dup z0.s, w0 (broadcast reg)"
probe_one "eor z0.d, z1.d, z2.d (XOR)"
probe_one "eor z0.d, z1.d, z2.d (NO SMSTART control)" nosve 132
probe_one "eor3 z0.d, z0.d, z1.d, z2.d (3-way XOR)"
probe_one "eorbt z0.s, z1.s, z2.s (XOR bottom with top)"
probe_one "eors p0.b, p0/z, p0.b, p0.b (Pred XOR, setting flags)"
probe_one "eortb z0.s, z1.s, z2.s (XOR top with bottom)"
probe_one "eorv b0, p0, z1.b (Bitwise XOR reduction)"
probe_one "ext z0.b, z0.b, z1.b, #0 (extract)"
probe_one "f1cvt {z0.h-z1.h}, z2.b (FP8->FP16 f1cvt)" sme 132
probe_one "f1cvtl {z0.h-z1.h}, z2.b (FP8->FP16 deintrl)" sme 132
probe_one "f2cvt {z0.h-z1.h}, z2.b (FP8->FP16 f2cvt)" sme 132
probe_one "f2cvtl {z0.h-z1.h}, z2.b (FP8->FP16 deintrl2)" sme 132
probe_one "fabd z0.s, p0/m, z0.s, z1.s (FP abs difference)"
probe_one "fabs z0.s, p0/m, z1.s (FP32 abs)"
probe_one "facge p0.s, p0/z, z0.s, z1.s (FP absolute compare GE)"
probe_one "facgt p0.s, p0/z, z0.s, z1.s (FP absolute compare GT)"
probe_one "fadd z0.s, z1.s, z2.s (FP32 add)"
probe_one "fadd z0.s, p0/m, z0.s, #0.5 (FADD imm)"
probe_one "fadd z0.h, z1.h, z2.h (FP16 vector add)"
probe_one "fadd za.s[w8,0,VGx2], {z0.s-z1.s} (2v fadd to ZA)"
probe_one "fadd za.s[w8,0,VGx4], {z0.s-z3.s} (4v fadd to ZA)"
probe_one "fadda s0, p0, s0, z1.s (ordered sum)" sme 132
probe_one "faddp z0.s, p0/m, z0.s, z1.s (pairwise add)"
probe_one "faddv s0, p0, z1.s (FP32 sum)"
probe_one "faddv h0, p0, z0.h (FP16 add reduction)"
probe_one "famax z0.s, p0/m, z0.s, z1.s (famax predicated)" sme 132
probe_one "famin z0.s, p0/m, z0.s, z1.s (famin predicated)" sme 132
probe_one "fcadd z0.s, p0/m, z0.s, z1.s, #90 (FP complex add)"
probe_one "fclamp z0.s, z1.s, z2.s (FP clamp to range)"
probe_one "fclamp z0.h, z1.h, z2.h (FP16 clamp)"
probe_one "fcmeq p0.s, p0/z, z0.s, z1.s (FP compare equal)"
probe_one "fcmeq p0.s, p0/z, z0.s, #0.0 (FP compare equal to zero)"
probe_one "fcmge p0.s, p0/z, z0.s, z1.s (FP compare greater/equal)"
probe_one "fcmge p0.s, p0/z, z0.s, #0.0 (FP compare GE zero)"
probe_one "fcmgt p0.s, p0/z, z0.s, z1.s (FP compare greater than)"
probe_one "fcmgt p0.s, p0/z, z0.s, #0.0 (FP compare GT zero)"
probe_one "fcmla z0.s, p0/m, z0.s, z1.s, #0 (FP complex multiply-add)"
probe_one "fcmla z0.s, z1.s, z2.s[0], #0 (FP complex MLA indexed)"
probe_one "fcmle p0.s, p0/z, z0.s, #0.0 (FP compare LE zero)"
probe_one "fcmlt p0.s, p0/z, z0.s, #0.0 (FP compare LT zero)"
probe_one "fcmne p0.s, p0/z, z0.s, z1.s (FP compare not equal)"
probe_one "fcmne p0.s, p0/z, z0.s, #0.0 (FP compare NE zero)"
probe_one "fcvt z0.s, p0/m, z1.h (FP16->FP32)"
probe_one "fcvt z0.h, p0/m, z1.s (FP32->FP16)"
probe_one "fcvt z0.d, p0/m, z1.s (FP32->FP64)"
probe_one "fcvt z0.s, p0/m, z0.h (FP16->FP32 convert)"
probe_one "fcvt z0.h, p0/m, z0.s (FP32->FP16 convert)"
probe_one "fcvt z0.b, {z0.h-z1.h} (FP16->FP8 convert)" sme 132
probe_one "fcvt {z0.s-z1.s}, z2.h (FP16->FP32 wid 2v)" "sme" "$EXPECT_M5"
probe_one "fcvt z0.b, {z0.s-z3.s} (FP32->FP8 4reg)" sme 132
probe_one "fcvtl {z0.s-z1.s}, z2.h (FP16->FP32 deintrl)" "sme" "$EXPECT_M5"
probe_one "fcvtn z0.h, {z0.s-z1.s} (Narrow FP32->FP16)"
probe_one "fcvtn z0.b, {z0.s-z3.s} (FP32->FP8 intrlv)" sme 132
probe_one "fcvtzs z0.s, p0/m, z1.s (FP32->INT32)"
probe_one "fcvtzs z0.s, p0/m, z1.d (FP64->I32)"
probe_one "fcvtzu z0.s, p0/m, z1.s (FP32->UINT32)"
probe_one "fdiv z0.s, p0/m, z0.s, z1.s (FP32 div)"
probe_one "fdivr z0.s, p0/m, z0.s, z1.s (FP divide reversed)"
probe_one "fdot z0.s, z1.h, z2.h (FP16 2-way)"
probe_one "fdot za.h[w8,0,VGx2], {z0.b-z1.b}, z2.b (2v FP8 fdot->fp16)" sme 132
probe_one "fdot za.h[w8,0,VGx4], {z0.b-z3.b}, z4.b (4v FP8 fdot->fp16)" sme 132
probe_one "fdot za.h[w8,0,VGx2], {z0.b-z1.b}, {z2.b-z3.b} (2v FP8 fdot mv fp16)" sme 132
probe_one "fdot za.h[w8,0,VGx4], {z0.b-z3.b}, {z4.b-z7.b} (4v FP8 fdot mv fp16)" sme 132
probe_one "fdot za.h[w8,0,VGx2], {z0.b-z1.b}, z2.b[0] (2v FP8 fdot idx fp16)" sme 132
probe_one "fdot za.h[w8,0,VGx4], {z0.b-z3.b}, z4.b[0] (4v FP8 fdot idx fp16)" sme 132
probe_one "fdot za.s[w8,0,VGx2], {z0.b-z1.b}, z2.b (2v FP8 fdot->fp32)" sme 132
probe_one "fdot za.s[w8,0,VGx4], {z0.b-z3.b}, z4.b (4v FP8 fdot->fp32)" sme 132
probe_one "fdot za.s[w8,0,VGx2], {z0.b-z1.b}, {z2.b-z3.b} (2v FP8 fdot mv fp32)" sme 132
probe_one "fdot za.s[w8,0,VGx4], {z0.b-z3.b}, {z4.b-z7.b} (4v FP8 fdot mv fp32)" sme 132
probe_one "fdot za.s[w8,0,VGx2], {z0.b-z1.b}, z2.b[0] (2v FP8 fdot idx fp32)" sme 132
probe_one "fdot za.s[w8,0,VGx4], {z0.b-z3.b}, z4.b[0] (4v FP8 fdot idx fp32)" sme 132
probe_one "fexpa z0.s, z1.s (2^x approx)" sme 132
probe_one "flogb z0.s, p0/m, z1.s (FP base-2 log of exponent)"
probe_one "fmad z0.s, p0/m, z1.s, z2.s (FP mul-add, destructive)"
probe_one "fmax z0.s, p0/m, z0.s, z1.s (FP32 max)"
probe_one "fmax z0.s, p0/m, z0.s, #0.0 (FMAX imm)"
probe_one "fmax z0.h, p0/m, z0.h, z1.h (FP16 maximum)"
probe_one "fmax {z0.s-z1.s}, {z0.s-z1.s}, {z2.s-z3.s} (2v fmax multi)"
probe_one "fmax {z0.s-z3.s}, {z0.s-z3.s}, {z4.s-z7.s} (4v fmax multi)"
probe_one "fmax {z0.s-z1.s}, {z0.s-z1.s}, z2.s (2v fmax single)"
probe_one "fmax {z0.s-z3.s}, {z0.s-z3.s}, z4.s (4v fmax single)"
probe_one "fmaxnm z0.s, p0/m, z0.s, z1.s (FP max, propagate number)"
probe_one "fmaxnm z0.s, p0/m, z0.s, #0.0 (FP max number, immediate)"
probe_one "fmaxnm {z0.s-z1.s}, {z0.s-z1.s}, z2.s (2v fmaxnm single)"
probe_one "fmaxnm {z0.s-z3.s}, {z0.s-z3.s}, z4.s (4v fmaxnm single)"
probe_one "fmaxnm {z0.s-z1.s}, {z0.s-z1.s}, {z2.s-z3.s} (2v fmaxnm mv)"
probe_one "fmaxnm {z0.s-z3.s}, {z0.s-z3.s}, {z4.s-z7.s} (4v fmaxnm mv)"
probe_one "fmaxnmp z0.s, p0/m, z0.s, z1.s (FP max number pairwise)"
probe_one "fmaxnmv s0, p0, z1.s (FP max number reduction)"
probe_one "fmaxp z0.s, p0/m, z0.s, z1.s (FP max pairwise)"
probe_one "fmaxv s0, p0, z1.s (FP32 max)"
probe_one "fmin z0.s, p0/m, z0.s, z1.s (FP32 min)"
probe_one "fmin z0.s, p0/m, z0.s, #0.0 (FMIN imm)"
probe_one "fmin z0.h, p0/m, z0.h, z1.h (FP16 minimum)"
probe_one "fmin {z0.s-z1.s}, {z0.s-z1.s}, {z2.s-z3.s} (2v fmin multi)"
probe_one "fmin {z0.s-z3.s}, {z0.s-z3.s}, {z4.s-z7.s} (4v fmin multi)"
probe_one "fminnm z0.s, p0/m, z0.s, z1.s (FP min, propagate number)"
probe_one "fminnm z0.s, p0/m, z0.s, #0.0 (FP min number, immediate)"
probe_one "fminnm {z0.s-z1.s}, {z0.s-z1.s}, z2.s (2v fminnm single)"
probe_one "fminnm {z0.s-z3.s}, {z0.s-z3.s}, z4.s (4v fminnm single)"
probe_one "fminnm {z0.s-z1.s}, {z0.s-z1.s}, {z2.s-z3.s} (2v fminnm mv)"
probe_one "fminnm {z0.s-z3.s}, {z0.s-z3.s}, {z4.s-z7.s} (4v fminnm mv)"
probe_one "fminnmp z0.s, p0/m, z0.s, z1.s (FP min number pairwise)"
probe_one "fminnmv s0, p0, z1.s (FP min number reduction)"
probe_one "fminp z0.s, p0/m, z0.s, z1.s (FP min pairwise)"
probe_one "fminv s0, p0, z1.s (FP32 min)"
probe_one "fmla z0.s, p0/m, z1.s, z2.s (FP32 fma)"
probe_one "fmla z0.s, z1.s, z2.s[0] (FMLA idx)"
probe_one "fmla z0.h, p0/m, z1.h, z2.h (FP16 fused mul-add)"
probe_one "fmla za.s[w8,0,VGx2], {z0.s-z1.s}, z2.s (2vx1 fmla ZA)"
probe_one "fmla za.s[w8,0,VGx4], {z0.s-z3.s}, z4.s (4vx1 fmla ZA)"
probe_one "fmla za.s[w8,0,VGx2], {z0.s-z1.s}, {z2.s-z3.s} (2vx2v fmla ZA)"
probe_one "fmla za.s[w8,0,VGx4], {z0.s-z3.s}, {z4.s-z7.s} (4vx4v fmla ZA)"
probe_one "fmla za.s[w8,0,VGx2], {z0.s-z1.s}, z2.s (2vx1 fmla ZA)"
probe_one "fmla za.s[w8,0,VGx4], {z0.s-z3.s}, z4.s (4vx1 fmla ZA)"
probe_one "fmla za.s[w8,0,VGx2], {z0.s-z1.s}, {z2.s-z3.s} (2vx2v fmla)"
probe_one "fmla za.s[w8,0,VGx4], {z0.s-z3.s}, {z4.s-z7.s} (4vx4v fmla)"
probe_one "fmla za.s[w8,0,VGx2], {z0.s-z1.s}, z2.s[0] (2v fmla idx ZA)"
probe_one "fmla za.s[w8,0,VGx4], {z0.s-z3.s}, z4.s[0] (4v fmla idx ZA)"
probe_one "fmlal za.h[w8,0:1], z0.b, z2.b (1v FP8 FMLAL fp16)" sme 132
probe_one "fmlal za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h (2vx1 fmlal FP16)"
probe_one "fmlal za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h (4vx1 fmlal FP16)"
probe_one "fmlal za.s[w8,0:1,VGx2], {z0.h-z1.h}, {z2.h-z3.h} (2v fmlal FP16)"
probe_one "fmlal za.s[w8,0:1,VGx4], {z0.h-z3.h}, {z4.h-z7.h} (4v fmlal FP16)"
probe_one "fmlal za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h[0] (2v fmlal idx)"
probe_one "fmlal za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h[0] (4v fmlal idx)"
probe_one "fmlal za.h[w8,0:1,VGx2], {z0.b-z1.b}, z2.b (2v FP8 fmlal fp16)" sme 132
probe_one "fmlal za.h[w8,0:1,VGx4], {z0.b-z3.b}, z4.b (4v FP8 fmlal fp16)" sme 132
probe_one "fmlal za.h[w8,0:1,VGx2], {z0.b-z1.b}, {z2.b-z3.b} (2v FP8 fmlal mv)" sme 132
probe_one "fmlal za.h[w8,0:1,VGx4], {z0.b-z3.b}, {z4.b-z7.b} (4v FP8 fmlal mv)" sme 132
probe_one "fmlal za.h[w8,0:1,VGx2], {z0.b-z1.b}, z2.b[0] (2v FP8 fmlal idx)" sme 132
probe_one "fmlal za.h[w8,0:1,VGx4], {z0.b-z3.b}, z4.b[0] (4v FP8 fmlal idx)" sme 132
probe_one "fmlalb z0.s, z1.h, z2.h (FP16 widen mul-add low)"
probe_one "fmlall za.s[w8,0:3], z0.b, z2.b (1v FP8 FMLALL)" sme 132
probe_one "fmlall za.s[w8,0:3,VGx2], {z0.b-z1.b}, z2.b (2v FP8 FMLALL)" sme 132
probe_one "fmlall za.s[w8,0:3,VGx4], {z0.b-z3.b}, z4.b (4v FP8 FMLALL)" sme 132
probe_one "fmlall za.s[w8,0:3,VGx2], {z0.b-z1.b}, z2.b[0] (2v FP8 fmlall idx)" sme 132
probe_one "fmlall za.s[w8,0:3,VGx4], {z0.b-z3.b}, z4.b[0] (4v FP8 fmlall idx)" sme 132
probe_one "fmlall za.s[w8,0:3,VGx2], {z0.b-z1.b}, {z2.b-z3.b} (2v FP8 fmlall mv)" sme 132
probe_one "fmlall za.s[w8,0:3,VGx4], {z0.b-z3.b}, {z4.b-z7.b} (4v FP8 fmlall mv)" sme 132
probe_one "fmlalt z0.s, z1.h, z2.h (FP16 widen mul-add high)"
probe_one "fmls z0.s, p0/m, z1.s, z2.s (FP32 fms)"
probe_one "fmls z0.s, z1.s, z2.s[0] (FMLS idx)"
probe_one "fmls za.s[w8,0,VGx2], {z0.s-z1.s}, z2.s (2vx1 fmls ZA)"
probe_one "fmls za.s[w8,0,VGx4], {z0.s-z3.s}, z4.s (4vx1 fmls ZA)"
probe_one "fmls za.s[w8,0,VGx2], {z0.s-z1.s}, z2.s[0] (2v fmls idx ZA)"
probe_one "fmls za.s[w8,0,VGx4], {z0.s-z3.s}, z4.s[0] (4v fmls idx ZA)"
probe_one "fmls za.s[w8,0,VGx2], {z0.s-z1.s}, {z2.s-z3.s} (2vx2v fmls ZA)"
probe_one "fmls za.s[w8,0,VGx4], {z0.s-z3.s}, {z4.s-z7.s} (4vx4v fmls ZA)"
probe_one "fmlsl za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h (2vx1 fmlsl FP16)"
probe_one "fmlsl za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h (4vx1 fmlsl FP16)"
probe_one "fmlsl za.s[w8,0:1,VGx2], {z0.h-z1.h}, {z2.h-z3.h} (2v fmlsl)"
probe_one "fmlsl za.s[w8,0:1,VGx4], {z0.h-z3.h}, {z4.h-z7.h} (4v fmlsl)"
probe_one "fmlsl za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h[0] (2v fmlsl idx)"
probe_one "fmlsl za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h[0] (4v fmlsl idx)"
probe_one "fmlslb z0.s, z1.h, z2.h (FP16 widen mul-sub low)"
probe_one "fmlslt z0.s, z1.h, z2.h (FP16 widen mul-sub high)"
probe_one "fmmla z0.s, z1.s, z2.s (FP32 matmul)" sme 132
probe_one "fmopa za0.s, p0/m, p0/m, z0.s, z1.s (FP32 outer)"
probe_one "fmopa za0.s, p0/m, p0/m, z0.h, z1.h (FP16->FP32)"
probe_one "fmopa za0.d, p0/m, p0/m, z0.d, z1.d (FP64 mopa)"
probe_one "fmopa za0.h, p0/m, p0/m, z0.b, z1.b (FP8->FP16 FMOPA)" sme 132
probe_one "fmopa za0.s, p0/m, p0/m, z0.b, z1.b (FP8->FP32 FMOPA)" sme 132
probe_one "fmops za0.s, p0/m, p0/m, z0.s, z1.s (FP32 sub)"
probe_one "fmops za0.d, p0/m, p0/m, z0.d, z1.d (FP64 mops)"
probe_one "fmsb z0.s, p0/m, z1.s, z2.s (FP mul-sub, destructive)"
probe_one "fmul z0.s, z1.s, z2.s (FP32 mul)"
probe_one "fmul z0.s, p0/m, z0.s, #2.0 (FMUL imm)"
probe_one "fmul z0.s, z1.s, z2.s[0] (FMUL idx)"
probe_one "fmul z0.h, z1.h, z2.h (FP16 vector mul)"
probe_one "fmulx z0.s, p0/m, z0.s, z1.s (FP multiply extended)"
probe_one "fneg z0.s, p0/m, z1.s (FP32 neg)"
probe_one "fnmad z0.s, p0/m, z1.s, z2.s (FP neg mul-add, destructive)"
probe_one "fnmla z0.s, p0/m, z1.s, z2.s (FP neg multiply-add)"
probe_one "fnmls z0.s, p0/m, z1.s, z2.s (FP neg multiply-subtract)"
probe_one "fnmsb z0.s, p0/m, z1.s, z2.s (FP neg mul-sub, destructive)"
probe_one "frecpe z0.s, z1.s (FP32 recip)"
probe_one "frecps z0.s, z0.s, z1.s (recip step)"
probe_one "frecpx z0.s, p0/m, z1.s (FP reciprocal exponent)"
probe_one "frinta z0.s, p0/m, z1.s (Round to nearest, ties away)"
probe_one "frinta {z0.s-z1.s}, {z0.s-z1.s} (2v frinta)"
probe_one "frinta {z0.s-z3.s}, {z0.s-z3.s} (4v frinta)"
probe_one "frinti z0.s, p0/m, z1.s (Round to int, current mode)"
probe_one "frintm z0.s, p0/m, z1.s (Round toward minus infinity)"
probe_one "frintm {z0.s-z1.s}, {z0.s-z1.s} (2v frintm)"
probe_one "frintm {z0.s-z3.s}, {z0.s-z3.s} (4v frintm)"
probe_one "frintn z0.s, p0/m, z1.s (Round to nearest, ties even)"
probe_one "frintn {z0.s-z1.s}, {z0.s-z1.s} (2v frintn)"
probe_one "frintn {z0.s-z3.s}, {z0.s-z3.s} (4v frintn)"
probe_one "frintp z0.s, p0/m, z1.s (Round toward plus infinity)"
probe_one "frintp {z0.s-z1.s}, {z0.s-z1.s} (2v frintp)"
probe_one "frintp {z0.s-z3.s}, {z0.s-z3.s} (4v frintp)"
probe_one "frintx z0.s, p0/m, z1.s (Round to int exact)"
probe_one "frintz z0.s, p0/m, z1.s (Round toward zero)"
probe_one "frsqrte z0.s, z1.s (FP32 rsqrt)"
probe_one "frsqrts z0.s, z0.s, z1.s (rsqrt step)"
probe_one "fscale z0.s, p0/m, z0.s, z1.s (scale 2^n)"
probe_one "fscale {z0.s-z1.s}, {z0.s-z1.s}, z2.s (2-vec fscale)" sme 132
probe_one "fscale {z0.s-z3.s}, {z0.s-z3.s}, z4.s (4-vec fscale)" sme 132
probe_one "fsqrt z0.s, p0/m, z1.s (FP square root)"
probe_one "fsub z0.s, z1.s, z2.s (FP32 sub)"
probe_one "fsub z0.s, p0/m, z0.s, #0.5 (FSUB imm)"
probe_one "fsub z0.h, z1.h, z2.h (FP16 vector sub)"
probe_one "fsub za.s[w8,0,VGx2], {z0.s-z1.s} (2v fsub from ZA)"
probe_one "fsub za.s[w8,0,VGx4], {z0.s-z3.s} (4v fsub from ZA)"
probe_one "fsubr z0.s, p0/m, z0.s, z1.s (FP subtract reversed)"
probe_one "fsubr z0.s, p0/m, z0.s, #0.5 (FP subtract reversed, imm)"
probe_one "ftmad z0.s, z0.s, z1.s, #0 (trig coeff)" sme 132
probe_one "ftsmul z0.s, z1.s, z2.s (FP trig select coefficient)" sme 132
probe_one "ftssel z0.s, z1.s, z2.s (FP trig start multiply)" sme 132
probe_one "fvdot za.s[w8,0], {z0.h-z1.h}, z2.h[0] (FP16 vertical dot)"
probe_one "fvdot za.h[w8,0], {z0.b-z1.b}, z2.b[0] (FP8 vdot fp16)" sme 132
probe_one "histcnt z0.s, p0/z, z1.s, z2.s (hist count)" sme 132
probe_one "histseg z0.b, z1.b, z2.b (hist segment)" sme 132
probe_one "incb x0 (Increment by byte count)"
probe_one "incd x0 (Increment by doubleword count)"
probe_one "incd z0.d (Inc doubleword count, vector)"
probe_one "inch x0 (Increment by halfword count)"
probe_one "inch z0.h (Inc halfword count, vector)"
probe_one "incp x0, p0.s (Inc by active pred count)"
probe_one "incw x0 (Increment by word count)"
probe_one "incw z0.s (Inc word count, vector)"
probe_one "index z0.s, #0, #1 (0,1,2,...)"
probe_one "insr z0.s, w0 (Insert scalar GPR at elem 0)"
probe_one "insr z0.s, s0 (Insert SIMD scalar at elem 0)"
probe_one "lasta w0, p0, z0.s (Extract after last active, GPR)"
probe_one "lasta s0, p0, z0.s (Extract after last active, SIMD)"
probe_one "lastb w0, p0, z0.s (Extract last active, GPR)"
probe_one "lastb s0, p0, z0.s (Extract last active, SIMD)"
probe_one "ld1b {z0.b}, p0/z, [x0] (INT8 load)"
probe_one "ld1b {z0.b-z1.b}, pn8/z, [x0] (2-vec byte load)" sm
probe_one "ld1b {z0.b-z3.b}, pn8/z, [x0] (4-vec byte load)" sm
probe_one "ld1b {z0.b,z8.b}, pn8/z, [x0] (ld1b strided 2)" sm
probe_one "ld1b {z0.b,z4.b,z8.b,z12.b}, pn8/z, [x0] (ld1b strided 4)" sm
probe_one "ld1d {z0.d}, p0/z, [x0] (Load 64-bit elements)"
probe_one "ld1d {z0.d,z8.d}, pn8/z, [x0] (ld1d strided 2)" sm
probe_one "ld1d {z0.d,z4.d,z8.d,z12.d}, pn8/z, [x0] (ld1d strided 4)" sm
probe_one "ld1h {z0.h}, p0/z, [x0] (INT16 load)"
probe_one "ld1h {z0.h,z8.h}, pn8/z, [x0] (ld1h strided 2)" sm
probe_one "ld1h {z0.h,z4.h,z8.h,z12.h}, pn8/z, [x0] (ld1h strided 4)" sm
probe_one "ld1rb z0.s, p0/z, [x0] (Load+broadcast byte)"
probe_one "ld1rd z0.d, p0/z, [x0] (Load+broadcast doubleword)"
probe_one "ld1rh z0.s, p0/z, [x0] (Load+broadcast halfword)"
probe_one "ld1rqb {z0.b}, p0/z, [x0] (Load+replicate 128b, bytes)"
probe_one "ld1rqd {z0.d}, p0/z, [x0] (Load+replicate 128b, dblwd)"
probe_one "ld1rqh {z0.h}, p0/z, [x0] (Load+replicate 128b, halfs)"
probe_one "ld1rqw {z0.s}, p0/z, [x0] (Load+replicate 128b, words)"
probe_one "ld1rw z0.s, p0/z, [x0] (Load+broadcast word)"
probe_one "ld1sb {z0.s}, p0/z, [x0] (Load sign-extended bytes)"
probe_one "ld1sh {z0.s}, p0/z, [x0] (Load sign-extended halfwords)"
probe_one "ld1sw {z0.d}, p0/z, [x0] (Load sign-extended words)"
probe_one "ld1w {z0.s}, p0/z, [x0] (FP32 load)"
probe_one "ld1w {z0.s-z1.s}, pn8/z, [x0] (2-vec word load)" sm
probe_one "ld1w {z0.s-z3.s}, pn8/z, [x0] (4-vec word load)" sm
probe_one "ld1w {z0.s,z8.s}, pn8/z, [x0] (ld1w strided 2)" sm
probe_one "ld1w {z0.s,z4.s,z8.s,z12.s}, pn8/z, [x0] (ld1w strided 4)" sm
probe_one "ld2b {z0.b, z1.b}, p0/z, [x0] (Load 2-struct bytes)"
probe_one "ld2d {z0.d, z1.d}, p0/z, [x0] (Load 2-struct doublewords)"
probe_one "ld2h {z0.h, z1.h}, p0/z, [x0] (Load 2-struct halfwords)"
probe_one "ld2w {z0.s, z1.s}, p0/z, [x0] (Load 2-struct words)"
probe_one "ld3b {z0.b, z1.b, z2.b}, p0/z, [x0] (Load 3-struct bytes)"
probe_one "ld3d {z0.d, z1.d, z2.d}, p0/z, [x0] (Load 3-struct doublewords)"
probe_one "ld3h {z0.h, z1.h, z2.h}, p0/z, [x0] (Load 3-struct halfwords)"
probe_one "ld3w {z0.s, z1.s, z2.s}, p0/z, [x0] (Load 3-struct words)"
probe_one "ld4b {z0.b, z1.b, z2.b, z3.b}, p0/z, [x0] (Load 4-struct bytes)"
probe_one "ld4d {z0.d, z1.d, z2.d, z3.d}, p0/z, [x0] (Load 4-struct doublewords)"
probe_one "ld4h {z0.h, z1.h, z2.h, z3.h}, p0/z, [x0] (Load 4-struct halfwords)"
probe_one "ld4w {z0.s, z1.s, z2.s, z3.s}, p0/z, [x0] (Load 4-struct words)"
probe_one "ldnt1b {z0.b}, p0/z, [x0] (Non-temporal load bytes)"
probe_one "ldnt1b {z0.b,z8.b}, pn8/z, [x0] (ldnt1b strided 2)" sm
probe_one "ldnt1b {z0.b,z4.b,z8.b,z12.b}, pn8/z, [x0] (ldnt1b strided 4)" sm
probe_one "ldnt1d {z0.d}, p0/z, [x0] (Non-temporal load dblwords)"
probe_one "ldnt1d {z0.d,z8.d}, pn8/z, [x0] (ldnt1d strided 2)" sm
probe_one "ldnt1h {z0.h}, p0/z, [x0] (Non-temporal load halfwords)"
probe_one "ldnt1h {z0.h,z8.h}, pn8/z, [x0] (ldnt1h strided 2)" sm
probe_one "ldnt1w {z0.s}, p0/z, [x0] (Non-temporal load words)"
probe_one "ldnt1w {z0.s,z8.s}, pn8/z, [x0] (ldnt1w strided 2)" sm
probe_one "ldr za[w12,0], [x0] (load ZA array vec)"
probe_one "ldr zt0, [x0] (load ZT0)" sme
probe_one "lsl z0.b, p0/m, z0.b, z1.b (shift left)"
probe_one "lsl z0.s, z1.s, #1 (Logical shift left, immediate)"
probe_one "lslr z0.s, p0/m, z0.s, z1.s (Logical shift left, reversed)"
probe_one "lsr z0.b, p0/m, z0.b, z1.b (shift right)"
probe_one "lsr z0.s, z1.s, #1 (Logical shift right, immediate)"
probe_one "lsrr z0.s, p0/m, z0.s, z1.s (Logical shift right, reversed)"
probe_one "luti2 z0.b, zt0, z1[0] (1v luti2 8b)" sme
probe_one "luti2 z0.h, zt0, z1[0] (1v luti2 16b)" sme
probe_one "luti2 z0.s, zt0, z1[0] (1v luti2 32b)" sme
probe_one "luti2 {z0.b-z1.b}, zt0, z2[0] (2v luti2 8b)" sme
probe_one "luti2 {z0.h-z1.h}, zt0, z2[0] (2v luti2 16b)" sme
probe_one "luti2 {z0.s-z1.s}, zt0, z2[0] (2v luti2 32b)" sme
probe_one "luti2 {z0.b-z3.b}, zt0, z2[0] (4v luti2 8b)" sme
probe_one "luti2 {z0.h-z3.h}, zt0, z2[0] (4v luti2 16b)" sme
probe_one "luti2 {z0.s-z3.s}, zt0, z2[0] (4v luti2 32b)" sme
probe_one "luti4 z0.b, zt0, z1[0] (1v luti4 8b)" sme
probe_one "luti4 z0.h, zt0, z1[0] (1v luti4 16b)" sme
probe_one "luti4 z0.s, zt0, z1[0] (1v luti4 32b)" sme
probe_one "luti4 {z0.b-z1.b}, zt0, z2[0] (2v luti4 8b)" sme
probe_one "luti4 {z0.h-z1.h}, zt0, z2[0] (2v luti4 16b)" sme
probe_one "luti4 {z0.s-z1.s}, zt0, z2[0] (2v luti4 32b)" sme
probe_one "luti4 {z0.h-z3.h}, zt0, z2[0] (4v luti4 16b)" sme
probe_one "luti4 {z0.s-z3.s}, zt0, z2[0] (4v luti4 32b)" sme
probe_one "mad z0.s, p0/m, z1.s, z2.s (MAD)"
probe_one "match p0.b, p0/z, z0.b, z1.b (set match)" sme 132
probe_one "mla z0.s, p0/m, z1.s, z2.s (MLA pred)"
probe_one "mla z0.s, p0/m, z1.s, z2.s (Integer mul-add)"
probe_one "mls z0.s, p0/m, z1.s, z2.s (MLS pred)"
probe_one "mls z0.s, p0/m, z1.s, z2.s (Integer mul-sub)"
probe_one "mov z0.s, #1 (immediate)"
probe_one "mov w12, #0
      mova   za0h.b[w12, 0], p0/m, z0.b (mova vec->tile byte)"
probe_one "mov w12, #0
      mova   za0h.s[w12, 0], p0/m, z0.s (mova vec->tile word)"
probe_one "mov w12, #0
      mova   za0h.d[w12, 0], p0/m, z0.d (mova vec->tile dword)"
probe_one "mov w8, #0
      zero   za.d[w8, 0:1] (double-vec zero pair)" sme
probe_one "mov w12, #0
      ld1b   za0h.b[w12,0], p0/z, [x0] (tile ld1b)"
probe_one "mov w12, #0
      ld1h   za0h.h[w12,0], p0/z, [x0] (tile ld1h)"
probe_one "mov w12, #0
      ld1w   za0h.s[w12,0], p0/z, [x0] (tile ld1w)"
probe_one "mov w12, #0
      ld1d   za0h.d[w12,0], p0/z, [x0] (tile ld1d)"
probe_one "mov w12, #0
      ld1q   za0h.q[w12,0], p0/z, [x0] (tile ld1q)"
probe_one "mov w12, #0
      st1b   za0h.b[w12,0], p0, [x0] (tile st1b)"
probe_one "mov w12, #0
      st1h   za0h.h[w12,0], p0, [x0] (tile st1h)"
probe_one "mov w12, #0
      st1w   za0h.s[w12,0], p0, [x0] (tile st1w)"
probe_one "mov w12, #0
      st1d   za0h.d[w12,0], p0, [x0] (tile st1d)"
probe_one "mov w12, #0
      st1q   za0h.q[w12,0], p0, [x0] (tile st1q)"
probe_one "mova z0.s, p0/m, za0h.s[w12, 0] (tile->vec h)"
probe_one "mova z0.s, p0/m, za0v.s[w12, 0] (tile->vec v)"
probe_one "mova za0h.s[w12, 0], p0/m, z0.s (vec->tile)"
probe_one "mova {z0.s-z1.s}, za0h.s[w12, 0:1] (2v arr->vec horiz)"
probe_one "mova {z0.s-z3.s}, za0h.s[w12, 0:3] (4v arr->vec horiz)"
probe_one "mova {z0.s-z1.s}, za0h.s[w12, 0:1] (2v tile->vec)"
probe_one "mova {z0.s-z3.s}, za0h.s[w12, 0:3] (4v tile->vec)"
probe_one "mova za0h.s[w12, 0:1], {z0.s-z1.s} (2v vec->arr)"
probe_one "mova za0h.s[w12, 0:3], {z0.s-z3.s} (4v vec->arr)"
probe_one "mova za0h.s[w12, 0:1], {z0.s-z1.s} (2v vec->tile)"
probe_one "mova za0h.s[w12, 0:3], {z0.s-z3.s} (4v vec->tile)"
probe_one "movaz z0.s, za0h.s[w12, 0] (move+zero)" "sme" "$EXPECT_M5"
probe_one "movaz {z0.s-z1.s}, za0h.s[w12, 0:1] (2v movaz arr->vec)" "sme" "$EXPECT_M5"
probe_one "movaz {z0.s-z3.s}, za0h.s[w12, 0:3] (4v movaz arr->vec)" "sme" "$EXPECT_M5"
probe_one "movaz {z0.s-z1.s}, za0h.s[w12, 0:1] (2v movaz tile->vec)" "sme" "$EXPECT_M5"
probe_one "movaz {z0.s-z3.s}, za0h.s[w12, 0:3] (4v movaz tile->vec)" "sme" "$EXPECT_M5"
probe_one "movprfx z0.s, p0/m, z1.s
      add z0.s, p0/m, z0.s, z1.s (move prefix)"
probe_one "movt x0, zt0[0] (read ZT0)" sme 132
probe_one "movt zt0[0], x0 (GPR->ZT0)" sme 132
probe_one "mrs x0, FPMR (Read FPMR)" nosve 132
probe_one "msb z0.s, p0/m, z1.s, z2.s (MSB)"
probe_one "msr FPMR, x0 (Write FPMR)" nosve 132
probe_one "mul z0.b, z1.b, z2.b (INT8 mul)"
probe_one "mul z0.h, z1.h, z2.h (INT16 mul)"
probe_one "mul z0.s, z1.s, z2.s (INT32 mul)"
probe_one "mul z0.s, z0.s, #1 (MUL imm)"
probe_one "mul z0.s, z1.s, z2.s[0] (Integer multiply indexed)"
probe_one "mul z0.b, z1.b, z2.b (NO SMSTART control)" nosve 132
probe_one "nand p0.b, p0/z, p0.b, p0.b (Pred bitwise NAND)"
probe_one "nbsl z0.d, z0.d, z1.d, z2.d (nand select)"
probe_one "neg z0.b, p0/m, z1.b (INT8 neg)"
probe_one "nmatch p0.b, p0/z, z0.b, z1.b (no match)" sme 132
probe_one "nor p0.b, p0/z, p0.b, p0.b (Pred bitwise NOR)"
probe_one "not z0.b, p0/m, z1.b (NOT)"
probe_one "orn p0.b, p0/z, p0.b, p0.b (Pred bitwise OR-NOT)"
probe_one "orr z0.d, z1.d, z2.d (OR)"
probe_one "orrs p0.b, p0/z, p0.b, p0.b (Pred OR, setting flags)"
probe_one "orv b0, p0, z1.b (Bitwise OR reduction)"
probe_one "pfalse p0.b (Set predicate to all-false)"
probe_one "pfirst p0.b, p0, p0.b (Set first active to true)"
probe_one "pmul z0.b, z1.b, z2.b (poly mul)"
probe_one "pmullb z0.h, z1.b, z2.b (Polynomial mul long, bot)"
probe_one "pmullt z0.h, z1.b, z2.b (Polynomial mul long, top)"
probe_one "pnext p0.s, p0, p0.s (Find next active element)"
probe_one "ptest p0, p0.b (Test predicate, set flags)"
probe_one "ptrue p0.b (all-true .b)"
probe_one "ptrue p0.h (all-true .h)"
probe_one "ptrue p0.s (all-true .s)"
probe_one "ptrue p0.d (all-true .d)"
probe_one "ptrues p0.s (All-true predicate, set flags)"
probe_one "punpkhi p0.h, p0.b (Pred unpack, high half->wide)"
probe_one "punpklo p0.h, p0.b (Pred unpack, low half->wide)"
probe_one "raddhnb z0.b, z1.h, z2.h (Rounding add narrow hi, bot)"
probe_one "raddhnt z0.b, z1.h, z2.h (Rounding add narrow hi, top)"
probe_one "rax1 z0.d, z1.d, z2.d (SHA3 rotate-and-XOR)" sme 132
probe_one "rbit z0.b, p0/m, z1.b (reverse bits)"
probe_one "rdffr p0.b (Read first-fault register)" sme 132
probe_one "rdsvl x0, #1 (Read streaming vector length)" sm
probe_one "rdvl x0, #1 (read VL)"
probe_one "rev z0.b, z1.b (reverse)"
probe_one "rev p0.s, p0.s (Reverse predicate elements)"
probe_one "revb z0.h, p0/m, z1.h (byte reverse)"
probe_one "revh z0.s, p0/m, z1.s (Reverse halfwords in element)"
probe_one "revw z0.d, p0/m, z1.d (Reverse words in element)"
probe_one "rshrnb z0.b, z1.h, #1 (Rounding shift narrow, bot)"
probe_one "rshrnt z0.b, z1.h, #1 (Rounding shift narrow, top)"
probe_one "rsubhnb z0.b, z1.h, z2.h (Rounding sub narrow hi, bot)"
probe_one "rsubhnt z0.b, z1.h, z2.h (Rounding sub narrow hi, top)"
probe_one "saba z0.s, z1.s, z2.s (Signed abs diff accumulate)"
probe_one "sabalb z0.s, z1.h, z2.h (Signed abs diff acc, wid lo)"
probe_one "sabalt z0.s, z1.h, z2.h (Signed abs diff acc, wid hi)"
probe_one "sabd z0.s, p0/m, z0.s, z1.s (Signed abs difference)"
probe_one "sabdlb z0.s, z1.h, z2.h (Signed abs diff, widen lo)"
probe_one "sabdlt z0.s, z1.h, z2.h (Signed abs diff, widen hi)"
probe_one "sadalp z0.s, p0/m, z1.h (Signed add+accum long pairwise)"
probe_one "saddlb z0.h, z1.b, z2.b (widen add lo s)"
probe_one "saddlbt z0.s, z1.h, z2.h (Signed add long, bot+top)"
probe_one "saddlt z0.h, z1.b, z2.b (widen add hi s)"
probe_one "saddv d0, p0, z1.b (INT8 widen sum)"
probe_one "saddwb z0.s, z0.s, z1.h (Signed add wide, bottom)"
probe_one "saddwt z0.s, z0.s, z1.h (Signed add wide, top)"
probe_one "sbclb z0.s, z1.s, z2.s (Sub with borrow long, bot)"
probe_one "sbclt z0.s, z1.s, z2.s (Sub with borrow long, top)"
probe_one "sclamp z0.s, z1.s, z2.s (signed clamp)"
probe_one "scvtf z0.s, p0/m, z1.s (INT32->FP32)"
probe_one "scvtf z0.d, p0/m, z1.s (I32->FP64)"
probe_one "scvtf {z0.s-z1.s}, {z0.s-z1.s} (2-vec scvtf)"
probe_one "scvtf {z0.s-z3.s}, {z0.s-z3.s} (4-vec scvtf)"
probe_one "sdiv z0.s, p0/m, z0.s, z1.s (INT32 sdiv)"
probe_one "sdivr z0.s, p0/m, z0.s, z1.s (SDIVR)"
probe_one "sdot z2.s, z0.b, z1.b (INT8 sdot)"
probe_one "sdot z0.s, z0.b, z1.b[0] (indexed sdot)"
probe_one "sdot za.s[w8,0], {z0.b-z1.b}, z2.b (2v sdot)"
probe_one "sdot z0.s, z1.h, z2.h (Signed 16b->32b dot product)"
probe_one "sdot za.s[w8,0], {z0.b-z1.b}, z2.b (2v x 1 sdot)"
probe_one "sdot za.s[w8,0], {z0.b-z3.b}, z4.b (4v x 1 sdot)"
probe_one "sdot za.s[w8,0], {z0.b-z1.b}, {z2.b-z3.b} (2v x 2v sdot)"
probe_one "sdot za.s[w8,0], {z0.b-z3.b}, {z4.b-z7.b} (4v x 4v sdot)"
probe_one "sdot za.s[w8,0,VGx2], {z0.h-z1.h}, z2.h[0] (2v 2way sdot idx)"
probe_one "sdot za.s[w8,0,VGx4], {z0.h-z3.h}, z4.h[0] (4v 2way sdot idx)"
probe_one "sdot za.s[w8,0,VGx2], {z0.b-z1.b}, z2.b[0] (2v 4way sdot idx)"
probe_one "sdot za.s[w8,0,VGx4], {z0.b-z3.b}, z4.b[0] (4v 4way sdot idx)"
probe_one "sdot z2.s, z0.b, z1.b (NO SMSTART control)" nosve 132
probe_one "sel z0.s, p0, z1.s, z2.s (select)"
probe_one "sel p0.b, p0, p0.b, p0.b (Select predicate elements)"
probe_one "sel {z0.s-z1.s}, pn8, {z0.s-z1.s}, {z2.s-z3.s} (2v sel)"
probe_one "sel {z0.s-z3.s}, pn8, {z0.s-z3.s}, {z4.s-z7.s} (4v sel)"
probe_one "setffr (Set first-fault register)" sme 132
probe_one "shadd z0.s, p0/m, z0.s, z1.s (Signed halving add)"
probe_one "shrnb z0.b, z1.h, #1 (Shift right narrow, bottom)"
probe_one "shrnt z0.b, z1.h, #1 (Shift right narrow, top)"
probe_one "shsub z0.s, p0/m, z0.s, z1.s (Signed halving subtract)"
probe_one "shsubr z0.s, p0/m, z0.s, z1.s (Signed halving sub reversed)"
probe_one "sli z0.b, z1.b, #0 (shift left ins)"
probe_one "sm4e z0.s, z0.s, z1.s (SM4 encrypt)" sme 132
probe_one "sm4ekey z0.s, z1.s, z2.s (SM4 key expansion)" sme 132
probe_one "smax z0.b, p0/m, z0.b, z1.b (signed max)"
probe_one "smax {z0.s-z1.s}, {z0.s-z1.s}, {z2.s-z3.s} (2v smax mv)"
probe_one "smax {z0.s-z3.s}, {z0.s-z3.s}, {z4.s-z7.s} (4v smax mv)"
probe_one "smaxp z0.s, p0/m, z0.s, z1.s (SMAXP)"
probe_one "smaxv b0, p0, z1.b (INT8 max)"
probe_one "smin z0.b, p0/m, z0.b, z1.b (signed min)"
probe_one "smin {z0.s-z1.s}, {z0.s-z1.s}, {z2.s-z3.s} (2v smin mv)"
probe_one "smin {z0.s-z3.s}, {z0.s-z3.s}, {z4.s-z7.s} (4v smin mv)"
probe_one "sminp z0.s, p0/m, z0.s, z1.s (SMINP)"
probe_one "sminv b0, p0, z1.b (INT8 min)"
probe_one "smlal za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h (2vx1 smlal)"
probe_one "smlal za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h (4vx1 smlal)"
probe_one "smlal za.s[w8,0:1,VGx2], {z0.h-z1.h}, {z2.h-z3.h} (2v smlal mv)"
probe_one "smlal za.s[w8,0:1,VGx4], {z0.h-z3.h}, {z4.h-z7.h} (4v smlal mv)"
probe_one "smlal za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h[0] (2v smlal idx)"
probe_one "smlal za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h[0] (4v smlal idx)"
probe_one "smlalb z0.h, z1.b, z2.b (widen mla lo s)"
probe_one "smlall za.s[w8,0:3], z0.b, z2.b (1v SMLALL I8->I32)"
probe_one "smlall za.s[w8,0:3,VGx2], {z0.b-z1.b}, z2.b (2v SMLALL single)"
probe_one "smlall za.s[w8,0:3,VGx4], {z0.b-z3.b}, z4.b (4v SMLALL single)"
probe_one "smlall za.s[w8,0:3,VGx2], {z0.b-z1.b}, z2.b[0] (2v smlall idx)"
probe_one "smlall za.s[w8,0:3,VGx4], {z0.b-z3.b}, z4.b[0] (4v smlall idx)"
probe_one "smlall za.s[w8,0:3,VGx2], {z0.b-z1.b}, {z2.b-z3.b} (2v smlall mv)"
probe_one "smlall za.s[w8,0:3,VGx4], {z0.b-z3.b}, {z4.b-z7.b} (4v smlall mv)"
probe_one "smlalt z0.h, z1.b, z2.b (widen mla hi s)"
probe_one "smlsl za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h (2vx1 smlsl)"
probe_one "smlsl za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h (4vx1 smlsl)"
probe_one "smlsl za.s[w8,0:1,VGx2], {z0.h-z1.h}, {z2.h-z3.h} (2v smlsl mv)"
probe_one "smlsl za.s[w8,0:1,VGx4], {z0.h-z3.h}, {z4.h-z7.h} (4v smlsl mv)"
probe_one "smlsl za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h[0] (2v smlsl idx)"
probe_one "smlsl za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h[0] (4v smlsl idx)"
probe_one "smlsll za.s[w8,0:3], z0.b, z2.b (1v smlsll)"
probe_one "smlsll za.s[w8,0:3,VGx2], {z0.b-z1.b}, z2.b (2v smlsll sgl)"
probe_one "smlsll za.s[w8,0:3,VGx4], {z0.b-z3.b}, z4.b (4v smlsll sgl)"
probe_one "smlsll za.s[w8,0:3,VGx2], {z0.b-z1.b}, z2.b[0] (2v smlsll idx)"
probe_one "smlsll za.s[w8,0:3,VGx4], {z0.b-z3.b}, z4.b[0] (4v smlsll idx)"
probe_one "smlsll za.s[w8,0:3,VGx2], {z0.b-z1.b}, {z2.b-z3.b} (2v smlsll mv)"
probe_one "smlsll za.s[w8,0:3,VGx4], {z0.b-z3.b}, {z4.b-z7.b} (4v smlsll mv)"
probe_one "smmla z0.s, z1.b, z2.b (Signed 8b matrix mul accum)" sme 132
probe_one "smopa za0.s, p0/m, p0/m, z0.b, z1.b (INT8 outer s)"
probe_one "smopa za0.s, p0/m, p1/m, z0.b, z1.b (I8->I32)"
probe_one "smopa za1.s, p0/m, p1/m, z0.b, z1.b (I8->I32 tile 1?)"
probe_one "smopa za2.s, p0/m, p1/m, z0.b, z1.b (I8->I32 tile 2?)"
probe_one "smopa za3.s, p0/m, p1/m, z0.b, z1.b (I8->I32 tile 3?)"
probe_one "smopa za0.d, p0/m, p0/m, z0.h, z1.h (I16->I64)"
probe_one "smopa za0.s, p0/m, p0/m, z0.h, z1.h (I16->I32 signed OPA)"
probe_one "smops za0.s, p0/m, p0/m, z0.b, z1.b (INT8 sub s)"
probe_one "smops za0.s, p0/m, p0/m, z0.h, z1.h (I16 2way smops)"
probe_one "smulh z0.b, z1.b, z2.b (INT8 mulhi s)"
probe_one "smullb z0.h, z1.b, z2.b (widen mul lo s)"
probe_one "smullt z0.h, z1.b, z2.b (widen mul hi s)"
probe_one "splice z0.b, p0, z0.b, z1.b (splice)"
probe_one "sqabs z0.s, p0/m, z1.s (Saturating absolute value)"
probe_one "sqadd z0.b, z1.b, z2.b (sat add s)"
probe_one "sqadd z0.s, p0/m, z0.s, z1.s (Sat add signed, predicated)"
probe_one "sqcadd z0.s, z0.s, z1.s, #90 (Sat complex integer add)"
probe_one "sqcvt z0.h, {z0.s-z1.s} (sat narrow s)"
probe_one "sqcvt z0.b, {z0.s-z3.s} (4reg sqcvt)"
probe_one "sqcvtn z0.b, {z0.s-z3.s} (sqcvtn interleave)"
probe_one "sqcvtu z0.h, {z0.s-z1.s} (sat narrow su)"
probe_one "sqcvtu z0.b, {z0.s-z3.s} (4reg sqcvtu)"
probe_one "sqcvtun z0.b, {z0.s-z3.s} (sqcvtun interleave)"
probe_one "sqdecb x0 (Sat dec by byte count, signed)"
probe_one "sqdecp x0, p0.s (Sat dec by pred count, signed)"
probe_one "sqdecw x0 (Sat dec by word count, signed)"
probe_one "sqdmlalb z0.s, z1.h, z2.h (Sat dbl mul-add long, bot)"
probe_one "sqdmlalbt z0.s, z1.h, z2.h (Sat dbl mul-add long, bxt)"
probe_one "sqdmlalt z0.s, z1.h, z2.h (Sat dbl mul-add long, top)"
probe_one "sqdmlslb z0.s, z1.h, z2.h (Sat dbl mul-sub long, bot)"
probe_one "sqdmlslbt z0.s, z1.h, z2.h (Sat dbl mul-sub long, bxt)"
probe_one "sqdmlslt z0.s, z1.h, z2.h (Sat dbl mul-sub long, top)"
probe_one "sqdmulh z0.b, z1.b, z2.b (sat dbl mul)"
probe_one "sqdmulh {z0.s-z1.s}, {z0.s-z1.s}, z2.s (2v sqdmulh sgl)"
probe_one "sqdmulh {z0.s-z3.s}, {z0.s-z3.s}, z4.s (4v sqdmulh sgl)"
probe_one "sqdmullb z0.s, z1.h, z2.h (Sat doubling mul long, bot)"
probe_one "sqdmullt z0.s, z1.h, z2.h (Sat doubling mul long, top)"
probe_one "sqincb x0 (Sat inc by byte count, signed)"
probe_one "sqincp x0, p0.s (Sat inc by pred count, signed)"
probe_one "sqincw x0 (Sat inc by word count, signed)"
probe_one "sqneg z0.s, p0/m, z1.s (Saturating negate)"
probe_one "sqrdcmlah z0.s, z1.s, z2.s, #0 (Sat rnd cmplx dbl mul-add)"
probe_one "sqrdmlah z0.s, z1.s, z2.s (Sat rnd dbl mul-add high)"
probe_one "sqrdmlsh z0.s, z1.s, z2.s (Sat rnd dbl mul-sub high)"
probe_one "sqrdmulh z0.b, z1.b, z2.b (sat rnd mul)"
probe_one "sqrshl z0.s, p0/m, z0.s, z1.s (Sat rounding shift left, s)"
probe_one "sqrshlr z0.s, p0/m, z0.s, z1.s (Sat rounding shift s, rev)"
probe_one "sqrshr z0.h, {z0.s-z1.s}, #1 (2reg sqrshr)"
probe_one "sqrshr z0.b, {z0.s-z3.s}, #1 (4reg sqrshr)"
probe_one "sqrshrn z0.b, {z0.s-z3.s}, #1 (sqrshrn interleave)"
probe_one "sqrshrnb z0.h, z1.s, #1 (Sat rnd shift narrow s, bot)"
probe_one "sqrshrnt z0.h, z1.s, #1 (Sat rnd shift narrow s, top)"
probe_one "sqrshru z0.h, {z0.s-z1.s}, #1 (2reg sqrshru)"
probe_one "sqrshru z0.b, {z0.s-z3.s}, #1 (4reg sqrshru)"
probe_one "sqrshrun z0.b, {z0.s-z3.s}, #1 (sqrshrun intrlv)"
probe_one "sqrshrunb z0.h, z1.s, #1 (Sat rnd shift nar s->u, bot)"
probe_one "sqrshrunt z0.h, z1.s, #1 (Sat rnd shift nar s->u, top)"
probe_one "sqshl z0.s, p0/m, z0.s, z1.s (Sat shift left signed)"
probe_one "sqshl z0.s, p0/m, z0.s, #1 (Sat shift left signed, imm)"
probe_one "sqshlr z0.s, p0/m, z0.s, z1.s (Sat shift left signed, rev)"
probe_one "sqshlu z0.s, p0/m, z0.s, #1 (Sat shift left uns from signed)"
probe_one "sqshrnb z0.b, z1.h, #1 (sat shift narrow)"
probe_one "sqshrnb z0.h, z1.s, #1 (Sat shift right narrow s, bot)"
probe_one "sqshrnt z0.h, z1.s, #1 (Sat shift right narrow s, top)"
probe_one "sqshrunb z0.h, z1.s, #1 (Sat shift right nar s->u, bot)"
probe_one "sqshrunt z0.h, z1.s, #1 (Sat shift right nar s->u, top)"
probe_one "sqsub z0.b, z1.b, z2.b (sat sub s)"
probe_one "sqsub z0.s, p0/m, z0.s, z1.s (Sat sub signed, predicated)"
probe_one "sqsubr z0.s, p0/m, z0.s, z1.s (Sat sub signed, reversed)"
probe_one "sqxtnb z0.b, z1.h (sat narrow lo s)"
probe_one "sqxtnt z0.b, z1.h (Sat narrow signed, top)"
probe_one "sqxtunb z0.b, z1.h (Sat narrow signed->uns, bot)"
probe_one "sqxtunt z0.b, z1.h (Sat narrow signed->uns, top)"
probe_one "srhadd z0.s, p0/m, z0.s, z1.s (Signed rounding halving add)"
probe_one "sri z0.b, z1.b, #8 (shift right ins)"
probe_one "srshl z0.s, p0/m, z0.s, z1.s (Signed rounding shift left)"
probe_one "srshl {z0.s-z1.s}, {z0.s-z1.s}, z2.s (2v srshl single)"
probe_one "srshl {z0.s-z3.s}, {z0.s-z3.s}, z4.s (4v srshl single)"
probe_one "srshl {z0.s-z1.s}, {z0.s-z1.s}, {z2.s-z3.s} (2v srshl mv)"
probe_one "srshl {z0.s-z3.s}, {z0.s-z3.s}, {z4.s-z7.s} (4v srshl mv)"
probe_one "srshlr z0.s, p0/m, z0.s, z1.s (Signed rounding shift, rev)"
probe_one "srshr z0.s, p0/m, z0.s, #1 (Signed rounding shift right)"
probe_one "srsra z0.s, z1.s, #1 (Signed rnd shift+accumulate)"
probe_one "sshllb z0.s, z1.h, #0 (Signed shift left long, bot)"
probe_one "sshllt z0.s, z1.h, #0 (Signed shift left long, top)"
probe_one "ssra z0.s, z1.s, #1 (Signed shift right+accumulate)"
probe_one "ssublb z0.s, z1.h, z2.h (Signed subtract long bottom)"
probe_one "ssublbt z0.s, z1.h, z2.h (Signed sub long, bot-top)"
probe_one "ssublt z0.s, z1.h, z2.h (Signed subtract long top)"
probe_one "ssubltb z0.s, z1.h, z2.h (Signed sub long, top-bot)"
probe_one "ssubwb z0.s, z0.s, z1.h (Signed sub wide, bottom)"
probe_one "ssubwt z0.s, z0.s, z1.h (Signed sub wide, top)"
probe_one "st1b {z0.b}, p0, [x0] (INT8 store)"
probe_one "st1b {z0.b-z1.b}, pn8, [x0] (2-vec byte store)" sm
probe_one "st1b {z0.b-z3.b}, pn8, [x0] (4-vec byte store)" sm
probe_one "st1b {z0.b,z8.b}, pn8, [x0] (st1b strided 2)" sm
probe_one "st1b {z0.b,z4.b,z8.b,z12.b}, pn8, [x0] (st1b strided 4)" sm
probe_one "st1d {z0.d}, p0, [x0] (Store 64-bit elements)"
probe_one "st1d {z0.d,z8.d}, pn8, [x0] (st1d strided 2)" sm
probe_one "st1h {z0.h}, p0, [x0] (INT16 store)"
probe_one "st1h {z0.h,z8.h}, pn8, [x0] (st1h strided 2)" sm
probe_one "st1w {z0.s}, p0, [x0] (FP32 store)"
probe_one "st1w {z0.s-z1.s}, pn8, [x0] (2-vec word store)" sm
probe_one "st1w {z0.s-z3.s}, pn8, [x0] (4-vec word store)" sm
probe_one "st1w {z0.s,z8.s}, pn8, [x0] (st1w strided 2)" sm
probe_one "st2b {z0.b, z1.b}, p0, [x0] (Store 2-struct bytes)"
probe_one "st2d {z0.d, z1.d}, p0, [x0] (Store 2-struct doublewords)"
probe_one "st2h {z0.h, z1.h}, p0, [x0] (Store 2-struct halfwords)"
probe_one "st2w {z0.s, z1.s}, p0, [x0] (Store 2-struct words)"
probe_one "st3b {z0.b, z1.b, z2.b}, p0, [x0] (Store 3-struct bytes)"
probe_one "st3d {z0.d, z1.d, z2.d}, p0, [x0] (Store 3-struct doublewords)"
probe_one "st3h {z0.h, z1.h, z2.h}, p0, [x0] (Store 3-struct halfwords)"
probe_one "st3w {z0.s, z1.s, z2.s}, p0, [x0] (Store 3-struct words)"
probe_one "st4b {z0.b, z1.b, z2.b, z3.b}, p0, [x0] (Store 4-struct bytes)"
probe_one "st4d {z0.d, z1.d, z2.d, z3.d}, p0, [x0] (Store 4-struct doublewords)"
probe_one "st4h {z0.h, z1.h, z2.h, z3.h}, p0, [x0] (Store 4-struct halfwords)"
probe_one "st4w {z0.s, z1.s, z2.s, z3.s}, p0, [x0] (Store 4-struct words)"
probe_one "stnt1b {z0.b}, p0, [x0] (Non-temporal store bytes)"
probe_one "stnt1b {z0.b,z8.b}, pn8, [x0] (stnt1b strided 2)" sm
probe_one "stnt1d {z0.d}, p0, [x0] (Non-temporal store dblwords)"
probe_one "stnt1d {z0.d,z8.d}, pn8, [x0] (stnt1d strided 2)" sm
probe_one "stnt1h {z0.h}, p0, [x0] (Non-temporal store halfwords)"
probe_one "stnt1h {z0.h,z8.h}, pn8, [x0] (stnt1h strided 2)" sm
probe_one "stnt1w {z0.s}, p0, [x0] (Non-temporal store words)"
probe_one "stnt1w {z0.s,z8.s}, pn8, [x0] (stnt1w strided 2)" sm
probe_one "str za[w12,0], [x0] (store ZA array vec)"
probe_one "str zt0, [x0] (store ZT0)" sme
probe_one "sub z0.b, z1.b, z2.b (INT8 sub)"
probe_one "sub z0.h, z1.h, z2.h (INT16 sub)"
probe_one "sub z0.s, z1.s, z2.s (INT32 sub)"
probe_one "sub z0.d, z1.d, z2.d (INT64 sub)"
probe_one "sub z0.s, z0.s, #1 (SUB imm)"
probe_one "sub z0.s, p0/m, z0.s, z1.s (SUB pred)"
probe_one "sub za.s[w8,0,VGx2], {z0.s-z1.s} (2v sub from ZA)"
probe_one "sub za.s[w8,0,VGx4], {z0.s-z3.s} (4v sub from ZA)"
probe_one "sub za.s[w8,0,VGx2], {z0.s-z1.s}, {z2.s-z3.s} (2v sub arr result)"
probe_one "sub za.s[w8,0,VGx4], {z0.s-z3.s}, {z4.s-z7.s} (4v sub arr result)"
probe_one "sub za.s[w8,0,VGx2], {z0.s-z1.s}, z2.s (2vx1 sub arr result)"
probe_one "sub za.s[w8,0,VGx4], {z0.s-z3.s}, z4.s (4vx1 sub arr result)"
probe_one "subhnb z0.b, z1.h, z2.h (Subtract narrow high, bot)"
probe_one "subhnt z0.b, z1.h, z2.h (Subtract narrow high, top)"
probe_one "subr z0.s, z0.s, #1 (SUBR imm)"
probe_one "subr z0.s, p0/m, z0.s, z1.s (SUBR pred)"
probe_one "sudot z0.s, z1.b, z2.b[0] (Signed-unsigned dot indexed)"
probe_one "sudot za.s[w8,0,VGx2], {z0.b-z1.b}, z2.b (2v sudot single)"
probe_one "sudot za.s[w8,0,VGx4], {z0.b-z3.b}, z4.b (4v sudot single)"
probe_one "sumlall za.s[w8,0:3,VGx2], {z0.b-z1.b}, z2.b (2v sumlall sgl)"
probe_one "sumlall za.s[w8,0:3,VGx4], {z0.b-z3.b}, z4.b (4v sumlall sgl)"
probe_one "sumopa za0.s, p0/m, p0/m, z0.b, z1.b (s*u outer)"
probe_one "sumopa za0.s, p0/m, p1/m, z0.b, z1.b (S8*U8->I32)"
probe_one "sumops za0.s, p0/m, p0/m, z0.b, z1.b (Signed*unsigned outer prod sub)"
probe_one "sunpk {z0.h-z1.h}, z2.b (2-vec sunpk b->h)"
probe_one "sunpk {z0.s-z1.s}, z2.h (2-vec sunpk h->s)"
probe_one "sunpk {z0.h-z3.h}, {z2.b-z3.b} (4-vec sunpk b->h)"
probe_one "sunpk {z0.s-z3.s}, {z2.h-z3.h} (4-vec sunpk h->s)"
probe_one "sunpkhi z0.h, z1.b (unpack hi s)"
probe_one "sunpklo z0.h, z1.b (unpack lo s)"
probe_one "suqadd z0.s, p0/m, z0.s, z1.s (Signed sat add unsigned)"
probe_one "suvdot za.s[w8,0], {z0.b-z3.b}, z4.b[0] (Signedxunsigned vdot)"
probe_one "svdot za.s[w8,0], {z0.b-z3.b}, z4.b[0] (Signed vertical dot)"
probe_one "svdot za.s[w8,0], {z0.h-z1.h}, z2.h[0] (2way signed vdot)"
probe_one "sxtb z0.h, p0/m, z1.h (sign ext b)"
probe_one "sxth z0.s, p0/m, z1.s (Sign-extend halfword)"
probe_one "sxtw z0.d, p0/m, z1.d (Sign-extend word)"
probe_one "tbl z0.b, {z1.b}, z2.b (64-byte lookup)"
probe_one "tbl z0.b, {z1.b}, z2.b (NO SMSTART control)" nosve 132
probe_one "tblq z0.b, {z1.b}, z2.b (per-128 lookup)" "sme" "$EXPECT_M5"
probe_one "tbx z0.b, z1.b, z2.b (merge lookup)"
probe_one "tbxq z0.b, z1.b, z2.b (per-128 merge)" "sme" "$EXPECT_M5"
probe_one "trn1 z0.b, z1.b, z2.b (transpose lo)"
probe_one "trn1 p0.s, p0.s, p1.s (Transpose predicates, low)"
probe_one "trn2 z0.b, z1.b, z2.b (transpose hi)"
probe_one "trn2 p0.s, p0.s, p1.s (Transpose predicates, high)"
probe_one "uaba z0.s, z1.s, z2.s (Unsigned abs diff accumulate)"
probe_one "uabalb z0.s, z1.h, z2.h (Unsigned abs diff acc, wid lo)"
probe_one "uabalt z0.s, z1.h, z2.h (Unsigned abs diff acc, wid hi)"
probe_one "uabd z0.s, p0/m, z0.s, z1.s (Unsigned abs difference)"
probe_one "uabdlb z0.s, z1.h, z2.h (Unsigned abs diff, widen lo)"
probe_one "uabdlt z0.s, z1.h, z2.h (Unsigned abs diff, widen hi)"
probe_one "uadalp z0.s, p0/m, z1.h (Unsigned add+accum long pw)"
probe_one "uaddlb z0.h, z1.b, z2.b (widen add lo u)"
probe_one "uaddlt z0.h, z1.b, z2.b (widen add hi u)"
probe_one "uaddv d0, p0, z1.s (INT32 reduce)"
probe_one "uaddv d0, p0, z1.b (UINT8 widen sum)"
probe_one "uaddwb z0.s, z0.s, z1.h (Unsigned add wide, bottom)"
probe_one "uaddwt z0.s, z0.s, z1.h (Unsigned add wide, top)"
probe_one "uclamp z0.s, z1.s, z2.s (unsigned clamp)"
probe_one "ucvtf z0.s, p0/m, z1.s (UINT32->FP32)"
probe_one "ucvtf {z0.s-z1.s}, {z0.s-z1.s} (2-vec ucvtf)"
probe_one "ucvtf {z0.s-z3.s}, {z0.s-z3.s} (4-vec ucvtf)"
probe_one "udiv z0.s, p0/m, z0.s, z1.s (INT32 udiv)"
probe_one "udivr z0.s, p0/m, z0.s, z1.s (UDIVR)"
probe_one "udot z2.s, z0.b, z1.b (UINT8 udot)"
probe_one "udot za.s[w8,0], {z0.b-z1.b}, z2.b (2v udot)"
probe_one "udot z0.s, z1.h, z2.h (Unsigned 16b->32b dot product)"
probe_one "udot za.s[w8,0], {z0.b-z1.b}, z2.b (2v x 1 udot)"
probe_one "udot za.s[w8,0], {z0.b-z3.b}, z4.b (4v x 1 udot)"
probe_one "udot za.s[w8,0], {z0.b-z1.b}, {z2.b-z3.b} (2v x 2v udot)"
probe_one "udot za.s[w8,0], {z0.b-z3.b}, {z4.b-z7.b} (4v x 4v udot)"
probe_one "udot za.s[w8,0,VGx2], {z0.h-z1.h}, z2.h[0] (2v 2way udot idx)"
probe_one "udot za.s[w8,0,VGx4], {z0.h-z3.h}, z4.h[0] (4v 2way udot idx)"
probe_one "udot za.s[w8,0,VGx2], {z0.b-z1.b}, z2.b[0] (2v 4way udot idx)"
probe_one "udot za.s[w8,0,VGx4], {z0.b-z3.b}, z4.b[0] (4v 4way udot idx)"
probe_one "uhadd z0.s, p0/m, z0.s, z1.s (Unsigned halving add)"
probe_one "uhsub z0.s, p0/m, z0.s, z1.s (Unsigned halving subtract)"
probe_one "uhsubr z0.s, p0/m, z0.s, z1.s (Unsigned halving sub reversed)"
probe_one "umax z0.b, p0/m, z0.b, z1.b (unsigned max)"
probe_one "umax {z0.s-z1.s}, {z0.s-z1.s}, {z2.s-z3.s} (2v umax mv)"
probe_one "umax {z0.s-z3.s}, {z0.s-z3.s}, {z4.s-z7.s} (4v umax mv)"
probe_one "umaxp z0.s, p0/m, z0.s, z1.s (UMAXP)"
probe_one "umaxv b0, p0, z1.b (UINT8 max)"
probe_one "umin z0.b, p0/m, z0.b, z1.b (unsigned min)"
probe_one "umin {z0.s-z1.s}, {z0.s-z1.s}, {z2.s-z3.s} (2v umin mv)"
probe_one "umin {z0.s-z3.s}, {z0.s-z3.s}, {z4.s-z7.s} (4v umin mv)"
probe_one "uminp z0.s, p0/m, z0.s, z1.s (UMINP)"
probe_one "uminv b0, p0, z1.b (UINT8 min)"
probe_one "umlal za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h (2vx1 umlal)"
probe_one "umlal za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h (4vx1 umlal)"
probe_one "umlal za.s[w8,0:1,VGx2], {z0.h-z1.h}, {z2.h-z3.h} (2v umlal mv)"
probe_one "umlal za.s[w8,0:1,VGx4], {z0.h-z3.h}, {z4.h-z7.h} (4v umlal mv)"
probe_one "umlal za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h[0] (2v umlal idx)"
probe_one "umlal za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h[0] (4v umlal idx)"
probe_one "umlalb z0.h, z1.b, z2.b (widen mla lo u)"
probe_one "umlall za.s[w8,0:3], z0.b, z2.b (1v umlall)"
probe_one "umlall za.s[w8,0:3,VGx2], {z0.b-z1.b}, z2.b (2v umlall sgl)"
probe_one "umlall za.s[w8,0:3,VGx4], {z0.b-z3.b}, z4.b (4v umlall sgl)"
probe_one "umlall za.s[w8,0:3,VGx2], {z0.b-z1.b}, z2.b[0] (2v umlall idx)"
probe_one "umlall za.s[w8,0:3,VGx4], {z0.b-z3.b}, z4.b[0] (4v umlall idx)"
probe_one "umlall za.s[w8,0:3,VGx2], {z0.b-z1.b}, {z2.b-z3.b} (2v umlall mv)"
probe_one "umlall za.s[w8,0:3,VGx4], {z0.b-z3.b}, {z4.b-z7.b} (4v umlall mv)"
probe_one "umlalt z0.h, z1.b, z2.b (widen mla hi u)"
probe_one "umlsl za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h (2vx1 umlsl)"
probe_one "umlsl za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h (4vx1 umlsl)"
probe_one "umlsl za.s[w8,0:1,VGx2], {z0.h-z1.h}, {z2.h-z3.h} (2v umlsl mv)"
probe_one "umlsl za.s[w8,0:1,VGx4], {z0.h-z3.h}, {z4.h-z7.h} (4v umlsl mv)"
probe_one "umlsl za.s[w8,0:1,VGx2], {z0.h-z1.h}, z2.h[0] (2v umlsl idx)"
probe_one "umlsl za.s[w8,0:1,VGx4], {z0.h-z3.h}, z4.h[0] (4v umlsl idx)"
probe_one "umlsll za.s[w8,0:3], z0.b, z2.b (1v umlsll)"
probe_one "umlsll za.s[w8,0:3,VGx2], {z0.b-z1.b}, z2.b (2v umlsll sgl)"
probe_one "umlsll za.s[w8,0:3,VGx4], {z0.b-z3.b}, z4.b (4v umlsll sgl)"
probe_one "umlsll za.s[w8,0:3,VGx2], {z0.b-z1.b}, z2.b[0] (2v umlsll idx)"
probe_one "umlsll za.s[w8,0:3,VGx4], {z0.b-z3.b}, z4.b[0] (4v umlsll idx)"
probe_one "umlsll za.s[w8,0:3,VGx2], {z0.b-z1.b}, {z2.b-z3.b} (2v umlsll mv)"
probe_one "umlsll za.s[w8,0:3,VGx4], {z0.b-z3.b}, {z4.b-z7.b} (4v umlsll mv)"
probe_one "ummla z0.s, z1.b, z2.b (Unsigned 8b matrix mul accum)" sme 132
probe_one "umopa za0.s, p0/m, p0/m, z0.b, z1.b (INT8 outer u)"
probe_one "umopa za0.s, p0/m, p1/m, z0.b, z1.b (U8->I32)"
probe_one "umopa za0.d, p0/m, p0/m, z0.h, z1.h (I16->I64)"
probe_one "umopa za0.s, p0/m, p0/m, z0.h, z1.h (I16->I32 unsigned OPA)"
probe_one "umops za0.s, p0/m, p0/m, z0.b, z1.b (INT8 sub u)"
probe_one "umops za0.s, p0/m, p0/m, z0.h, z1.h (I16 2way umops)"
probe_one "umulh z0.b, z1.b, z2.b (INT8 mulhi u)"
probe_one "umullb z0.h, z1.b, z2.b (widen mul lo u)"
probe_one "umullt z0.h, z1.b, z2.b (widen mul hi u)"
probe_one "uqadd z0.b, z1.b, z2.b (sat add u)"
probe_one "uqadd z0.s, p0/m, z0.s, z1.s (Sat add unsigned, predicated)"
probe_one "uqcvt z0.h, {z0.s-z1.s} (sat narrow u)"
probe_one "uqcvt z0.b, {z0.s-z3.s} (4reg uqcvt)"
probe_one "uqcvtn z0.b, {z0.s-z3.s} (uqcvtn interleave)"
probe_one "uqdecb x0 (Sat dec by byte count, unsign)"
probe_one "uqdecp x0, p0.s (Sat dec by pred count, unsign)"
probe_one "uqdecw x0 (Sat dec by word count, unsign)"
probe_one "uqincb x0 (Sat inc by byte count, unsign)"
probe_one "uqincp x0, p0.s (Sat inc by pred count, unsign)"
probe_one "uqincw x0 (Sat inc by word count, unsign)"
probe_one "uqrshl z0.s, p0/m, z0.s, z1.s (Sat rounding shift left, u)"
probe_one "uqrshlr z0.s, p0/m, z0.s, z1.s (Sat rounding shift u, rev)"
probe_one "uqrshr z0.h, {z0.s-z1.s}, #1 (2reg uqrshr)"
probe_one "uqrshr z0.b, {z0.s-z3.s}, #1 (4reg uqrshr)"
probe_one "uqrshrn z0.b, {z0.s-z3.s}, #1 (uqrshrn intrlv)"
probe_one "uqrshrnb z0.h, z1.s, #1 (Sat rnd shift narrow u, bot)"
probe_one "uqrshrnt z0.h, z1.s, #1 (Sat rnd shift narrow u, top)"
probe_one "uqshl z0.s, p0/m, z0.s, z1.s (Sat shift left unsigned)"
probe_one "uqshl z0.s, p0/m, z0.s, #1 (Sat shift left unsigned, imm)"
probe_one "uqshlr z0.s, p0/m, z0.s, z1.s (Sat shift left unsigned, rev)"
probe_one "uqshrnb z0.b, z1.h, #1 (sat shift narrow)"
probe_one "uqshrnb z0.h, z1.s, #1 (Sat shift right narrow u, bot)"
probe_one "uqshrnt z0.h, z1.s, #1 (Sat shift right narrow u, top)"
probe_one "uqsub z0.b, z1.b, z2.b (sat sub u)"
probe_one "uqsub z0.s, p0/m, z0.s, z1.s (Sat sub unsigned, predicated)"
probe_one "uqsubr z0.s, p0/m, z0.s, z1.s (Sat sub unsigned, reversed)"
probe_one "uqxtnb z0.b, z1.h (sat narrow lo u)"
probe_one "uqxtnt z0.b, z1.h (Sat narrow unsigned, top)"
probe_one "urecpe z0.s, p0/m, z1.s (Unsigned reciprocal estimate)"
probe_one "urhadd z0.s, p0/m, z0.s, z1.s (Unsigned rounding halving add)"
probe_one "urshl z0.s, p0/m, z0.s, z1.s (Unsigned rounding shift left)"
probe_one "urshl {z0.s-z1.s}, {z0.s-z1.s}, z2.s (2v urshl single)"
probe_one "urshl {z0.s-z3.s}, {z0.s-z3.s}, z4.s (4v urshl single)"
probe_one "urshl {z0.s-z1.s}, {z0.s-z1.s}, {z2.s-z3.s} (2v urshl mv)"
probe_one "urshl {z0.s-z3.s}, {z0.s-z3.s}, {z4.s-z7.s} (4v urshl mv)"
probe_one "urshlr z0.s, p0/m, z0.s, z1.s (Unsigned rounding shift, rev)"
probe_one "urshr z0.s, p0/m, z0.s, #1 (Unsigned rounding shift right)"
probe_one "ursqrte z0.s, p0/m, z1.s (Unsigned recip sqrt estimate)"
probe_one "ursra z0.s, z1.s, #1 (Unsigned rnd shift+accumulate)"
probe_one "usdot z2.s, z0.b, z1.b (mixed usdot)"
probe_one "usdot z0.s, z1.b, z2.b[0] (Unsigned-signed dot indexed)"
probe_one "usdot za.s[w8,0,VGx2], {z0.b-z1.b}, z2.b (2v usdot single)"
probe_one "usdot za.s[w8,0,VGx4], {z0.b-z3.b}, z4.b (4v usdot single)"
probe_one "usdot za.s[w8,0,VGx2], {z0.b-z1.b}, z2.b[0] (2v usdot idx)"
probe_one "usdot za.s[w8,0,VGx4], {z0.b-z3.b}, z4.b[0] (4v usdot idx)"
probe_one "usdot za.s[w8,0,VGx2], {z0.b-z1.b}, {z2.b-z3.b} (2v usdot mv)"
probe_one "usdot za.s[w8,0,VGx4], {z0.b-z3.b}, {z4.b-z7.b} (4v usdot mv)"
probe_one "ushllb z0.s, z1.h, #0 (Unsigned shift left long, bot)"
probe_one "ushllt z0.s, z1.h, #0 (Unsigned shift left long, top)"
probe_one "usmlall za.s[w8,0:3,VGx2], {z0.b-z1.b}, z2.b (2v usmlall sgl)"
probe_one "usmlall za.s[w8,0:3,VGx4], {z0.b-z3.b}, z4.b (4v usmlall sgl)"
probe_one "usmlall za.s[w8,0:3,VGx2], {z0.b-z1.b}, {z2.b-z3.b} (2v usmlall mv)"
probe_one "usmlall za.s[w8,0:3,VGx4], {z0.b-z3.b}, {z4.b-z7.b} (4v usmlall mv)"
probe_one "usmmla z0.s, z1.b, z2.b (Mixed 8b matrix mul accum)" sme 132
probe_one "usmopa za0.s, p0/m, p0/m, z0.b, z1.b (u*s outer)"
probe_one "usmopa za0.s, p0/m, p1/m, z0.b, z1.b (U8*S8->I32)"
probe_one "usmops za0.s, p0/m, p0/m, z0.b, z1.b (Unsigned*signed outer prod sub)"
probe_one "usqadd z0.s, p0/m, z0.s, z1.s (Unsigned sat add signed)"
probe_one "usra z0.s, z1.s, #1 (Unsigned shift right+accum)"
probe_one "usublb z0.s, z1.h, z2.h (Unsigned sub long bottom)"
probe_one "usublt z0.s, z1.h, z2.h (Unsigned sub long top)"
probe_one "usubwb z0.s, z0.s, z1.h (Unsigned sub wide, bottom)"
probe_one "usubwt z0.s, z0.s, z1.h (Unsigned sub wide, top)"
probe_one "usvdot za.s[w8,0], {z0.b-z3.b}, z4.b[0] (Unsignedxsigned vdot)"
probe_one "uunpk {z0.h-z1.h}, z2.b (2v uunpk b->h)"
probe_one "uunpk {z0.s-z1.s}, z2.h (2v uunpk h->s)"
probe_one "uunpk {z0.h-z3.h}, {z2.b-z3.b} (4v uunpk b->h)"
probe_one "uunpk {z0.s-z3.s}, {z2.h-z3.h} (4v uunpk h->s)"
probe_one "uunpkhi z0.h, z1.b (unpack hi u)"
probe_one "uunpklo z0.h, z1.b (unpack lo u)"
probe_one "uvdot za.s[w8,0], {z0.b-z3.b}, z4.b[0] (Unsigned vertical dot)"
probe_one "uvdot za.s[w8,0], {z0.h-z1.h}, z2.h[0] (2way unsigned vdot)"
probe_one "uxtb z0.h, p0/m, z1.h (zero ext b)"
probe_one "uxth z0.s, p0/m, z1.s (Zero-extend halfword)"
probe_one "uxtw z0.d, p0/m, z1.d (Zero-extend word)"
probe_one "uzp {z0.s-z1.s}, z0.s, z1.s (2v uzp)"
probe_one "uzp {z0.s-z3.s}, {z0.s-z3.s} (4v uzp)"
probe_one "uzp1 z0.b, z1.b, z2.b (deintrlv lo)"
probe_one "uzp1 p0.s, p0.s, p1.s (Deinterleave preds, low)"
probe_one "uzp2 z0.b, z1.b, z2.b (deintrlv hi)"
probe_one "uzp2 p0.s, p0.s, p1.s (Deinterleave preds, high)"
probe_one "whilege p0.s, x0, x1 (while >=)"
probe_one "whilegt p0.s, x0, x1 (while >)"
probe_one "whilehi p0.s, x0, x1 (Pred while higher, unsigned)"
probe_one "whilehs p0.s, x0, x1 (Pred while higher/same, uns)"
probe_one "whilele p0.s, x0, x1 (Pred while less-or-equal)"
probe_one "whilelo p0.s, x0, x1 (Pred while lower, unsigned)"
probe_one "whilels p0.s, x0, x1 (Pred while lower/same, uns)"
probe_one "whilelt p0.s, xzr, x0 (while lt)"
probe_one "whilerw p0.s, x0, x1 (Pred while read-after-write)"
probe_one "whilewr p0.s, x0, x1 (Pred while write-after-read)"
probe_one "wrffr p0.b (Write first-fault register)" sme 132
probe_one "xar z0.b, z0.b, z1.b, #1 (xor-and-rotate)"
probe_one "zero {za} (zero ZA)" sme
probe_one "zero za.d[w8, 0:1] (double-vec zero)" "sme" "$EXPECT_M5" sme
probe_one "zero za.d[w8, 0:3] (quad-vec zero)" "sme" "$EXPECT_M5" sme
probe_one "zero {zt0} (zero ZT0)" sme
probe_one "zip {z0.s-z1.s}, z0.s, z1.s (2v zip)"
probe_one "zip {z0.s-z3.s}, {z0.s-z3.s} (4v zip)"
probe_one "zip1 z0.b, z1.b, z2.b (interleave lo)"
probe_one "zip1 p0.s, p0.s, p1.s (Interleave predicates, low)"
probe_one "zip2 z0.b, z1.b, z2.b (interleave hi)"
probe_one "zip2 p0.s, p0.s, p1.s (Interleave predicates, high)"
fi # end SKIP_OPS
# ===========================================================================================================
# [6S] STRESS TESTING — Throughput & Parallelism
# ===========================================================================================================
stress_throughput() {
    local name="$1"
    local instr="$2"
    local iters="${3:-1000000000}"  # 1B iterations
    rm -f "$PROBE_BIN" "$PROBE_BIN.s"
    cat > "$PROBE_BIN.s" <<EOF
.globl _stress_loop
.p2align 2
_stress_loop:
    // x0 = iterations
    smstart
    ptrue p0.b
    ptrue p1.s
    zero {za}
1:
    ${instr}
    ${instr}
    ${instr}
    ${instr}
    ${instr}
    ${instr}
    ${instr}
    ${instr}
    subs x0, x0, #8
    b.gt 1b
    smstop
    ret
EOF
    cat > "$PROBE_DIR/stress_main.c" <<EOF
#include <stdio.h>
#include <stdint.h>
#include <mach/mach_time.h>
extern void stress_loop(uint64_t iters);
int main(void) {
    uint64_t iters = ${iters};
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    uint64_t t0 = mach_absolute_time();
    stress_loop(iters);
    uint64_t t1 = mach_absolute_time();
    double ns = (double)(t1-t0) * tb.numer / tb.denom;
    double gops = iters / (ns / 1e9) / 1e9;
    printf("  %-42s %8.2f Gops/s (%6.3f s)\n", "${name}", gops, ns/1e9);
    return 0;
}
EOF
    if ! $CC $ASMFLAGS -c "$PROBE_BIN.s" -o "$PROBE_OBJ" 2>/dev/null; then
        printf "  %-42s COMPILE_FAIL\n" "$name"
        return
    fi
    if ! $CC $CFLAGS -c "$PROBE_DIR/stress_main.c" -o "$PROBE_DIR/stress_main.o" 2>/dev/null; then
        printf "  %-42s COMPILE_FAIL (main)\n" "$name"
        return
    fi
    if ! $CC $CFLAGS -o "$PROBE_BIN" "$PROBE_OBJ" "$PROBE_DIR/stress_main.o" 2>/dev/null; then
        printf "  %-42s LINK_FAIL\n" "$name"
        return
    fi
    "$PROBE_BIN" 2>/dev/null || printf "  %-42s FAILED\n" "$name"
}
# ===========================================================================================================
# Stress test: Parallel FMOPA in SME ZA mode (requires FEAT_SME_F16FMOPA or FEAT_SME_F32FMOPA)
# ===========================================================================================================
stress_parallel_fmopa() {
    local threads="$1"
    local iters="${2:-1000000000}"  # 1B per thread
    rm -f "$PROBE_BIN" "$PROBE_BIN.s"
    cat > "$PROBE_BIN.s" <<EOF
.globl _fmopa_loop
.p2align 2
_fmopa_loop:
    smstart
    ptrue p0.b
    ptrue p1.s
    zero {za}
1:
    fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    fmopa za1.s, p0/m, p1/m, z0.s, z1.s
    fmopa za2.s, p0/m, p1/m, z0.s, z1.s
    fmopa za3.s, p0/m, p1/m, z0.s, z1.s
    subs x0, x0, #4
    b.gt 1b
    smstop
    ret
EOF
    cat > "$PROBE_DIR/par_main.c" <<EOF
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <mach/mach_time.h>
static uint64_t g_iters;
extern void fmopa_loop(uint64_t n);
static void* thread_fn(void* arg) { fmopa_loop(g_iters); return NULL; }
int main(void) {
    int nthreads = ${threads};
    g_iters = ${iters};
    pthread_t tids[64];
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    uint64_t t0 = mach_absolute_time();
    for (int i = 1; i < nthreads; i++)
        pthread_create(&tids[i], NULL, thread_fn, NULL);
    fmopa_loop(g_iters);
    for (int i = 1; i < nthreads; i++)
        pthread_join(tids[i], NULL);
    uint64_t t1 = mach_absolute_time();
    double secs = (double)(t1-t0) * tb.numer / tb.denom / 1e9;
    double total_ops = (double)g_iters * nthreads * 512.0;
    double tflops = total_ops / secs / 1e12;
    printf("  FMOPA %2d threads: %8.3f TFLOPS (%6.3f s, %llu ops/thread)\n",
           nthreads, tflops, secs, (unsigned long long)g_iters);
    return 0;
}
EOF
    if ! $CC $ASMFLAGS -c "$PROBE_BIN.s" -o "$PROBE_OBJ" 2>/dev/null; then
        printf "  FMOPA %2d threads: COMPILE_FAIL\n" "$threads"
        return
    fi
    if ! $CC $CFLAGS -c "$PROBE_DIR/par_main.c" -o "$PROBE_DIR/par_main.o" 2>/dev/null; then
        printf "  FMOPA %2d threads: COMPILE_FAIL (main)\n" "$threads"
        return
    fi
    if ! $CC $CFLAGS -lpthread -o "$PROBE_BIN" "$PROBE_OBJ" "$PROBE_DIR/par_main.o" 2>/dev/null; then
        printf "  FMOPA %2d threads: LINK_FAIL\n" "$threads"
        return
    fi
    "$PROBE_BIN" 2>/dev/null || printf "  FMOPA %2d threads: FAILED\n" "$threads"
}
if [ $SKIP_TP -eq 0 ]; then
section "Single-core instruction throughput (1B iters, 8x unroll)"
stress_throughput "FADD z.s (FP32 vector add)"       "fadd z0.s, z1.s, z2.s"
stress_throughput "FMUL z.s (FP32 vector mul)"       "fmul z0.s, z1.s, z2.s"
stress_throughput "FMLA z.s (FP32 fused mul-add)"    "fmla z0.s, p0/m, z1.s, z2.s"
stress_throughput "ADD z.b (INT8 vector add)"         "add z0.b, z1.b, z2.b"
stress_throughput "MUL z.s (INT32 vector mul)"        "mul z0.s, z1.s, z2.s"
stress_throughput "SDOT z.s (INT8 dot product)"       "sdot z0.s, z1.b, z2.b"
stress_throughput "FDOT z.s (FP16 2-way dot)"         "fdot z0.s, z1.h, z2.h"
section "Single-core matrix throughput (1B iters, 8x unroll)"
stress_throughput "FMOPA za0.s (FP32 outer product)"  "fmopa za0.s, p0/m, p1/m, z0.s, z1.s"
stress_throughput "SMOPA za0.s (INT8 outer product)"  "smopa za0.s, p0/m, p1/m, z0.b, z1.b"
stress_throughput "SMLALL 1v (INT8 long-long)"         "smlall za.s[w8,0:3], z0.b, z2.b"
stress_throughput "SMLALL VGx2 (INT8 long-long 2v)"   "smlall za.s[w8,0:3,VGx2], {z0.b-z1.b}, {z2.b-z3.b}"
stress_throughput "SMLALL VGx4 (INT8 long-long 4v)"   "smlall za.s[w8,0:3,VGx4], {z0.b-z3.b}, {z4.b-z7.b}"
stress_throughput "BFMOPA za0.s (BF16 outer product)" "bfmopa za0.s, p0/m, p1/m, z0.h, z1.h"
stress_throughput "FMOPA za0.d (FP64 outer product)"  "fmopa za0.d, p0/m, p1/m, z0.d, z1.d"
section "Multi-threaded FMOPA (FP32) scaling"
for t in 1 2 4 6 8 12 16 18; do
    stress_parallel_fmopa "$t"
done
echo ""
echo "  Note: FMOPA TFLOPS = iterations * threads * 512ops / time"
echo "  (Each FMOPA za.s: 16x16x2 = 512 FP32 ops)"

stress_parallel_smlall() {
    local threads="$1"
    local iters="${2:-1000000000}"
    rm -f "$PROBE_BIN" "$PROBE_BIN.s"
    {
        echo '.globl _smlall_loop'
        echo '.p2align 2'
        echo '_smlall_loop:'
        echo '    smstart'
        echo '    ptrue p0.b'
        echo '    zero {za}'
        echo '1:'
        echo '    smlall za.s[w8,0:3], z0.b, z1.b'
        echo '    smlall za.s[w8,0:3], z2.b, z3.b'
        echo '    smlall za.s[w8,0:3], z4.b, z5.b'
        echo '    smlall za.s[w8,0:3], z6.b, z7.b'
        echo '    subs x0, x0, #4'
        echo '    b.gt 1b'
        echo '    smstop'
        echo '    ret'
    } > "$PROBE_BIN.s"
    cat > "$PROBE_DIR/par_smlall.c" <<CEOF
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <mach/mach_time.h>
static uint64_t g_iters;
extern void smlall_loop(uint64_t n);
static void* thread_fn(void* arg) { smlall_loop(g_iters); return NULL; }
int main(void) {
    int nthreads = ${threads};
    g_iters = ${iters};
    pthread_t tids[64];
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    uint64_t t0 = mach_absolute_time();
    for (int i = 1; i < nthreads; i++)
        pthread_create(&tids[i], NULL, thread_fn, NULL);
    smlall_loop(g_iters);
    for (int i = 1; i < nthreads; i++)
        pthread_join(tids[i], NULL);
    uint64_t t1 = mach_absolute_time();
    double secs = (double)(t1-t0) * tb.numer / tb.denom / 1e9;
    double total_ops = (double)g_iters * nthreads * 1024.0;
    double tops = total_ops / secs / 1e12;
    printf("  SMLALL %2d threads: %8.3f TOPS  (%6.3f s, %llu ops/thread)\n",
           nthreads, tops, secs, (unsigned long long)g_iters);
    return 0;
}
CEOF
    if ! $CC $ASMFLAGS -c "$PROBE_BIN.s" -o "$PROBE_OBJ" 2>/dev/null; then
        printf "  SMLALL %2d threads: COMPILE_FAIL\n" "$threads"
        return
    fi
    if ! $CC $CFLAGS -c "$PROBE_DIR/par_smlall.c" -o "$PROBE_DIR/par_smlall.o" 2>/dev/null; then
        printf "  SMLALL %2d threads: COMPILE_FAIL (main)\n" "$threads"
        return
    fi
    if ! $CC $CFLAGS -lpthread -o "$PROBE_BIN" "$PROBE_OBJ" "$PROBE_DIR/par_smlall.o" 2>/dev/null; then
        printf "  SMLALL %2d threads: LINK_FAIL\n" "$threads"
        return
    fi
    "$PROBE_BIN" 2>/dev/null || printf "  SMLALL %2d threads: FAILED\n" "$threads"
}
section "Multi-threaded SMLALL (INT8 long-long) scaling"
for t in 1 2 4 6 8 12 16 18; do
    stress_parallel_smlall "$t"
done
echo ""
echo "  Note: SMLALL TOPS = iterations * threads * 1024ops / time"
echo "  (Each SMLALL za.s: 16x64 = 1024 INT8 multiply-accumulate ops)"
stress_parallel_smopa_i8() {
    local threads="$1"
    local iters="${2:-1000000000}"
    rm -f "$PROBE_BIN" "$PROBE_BIN.s"
    {
        echo '.globl _smopa_i8_loop'
        echo '.p2align 2'
        echo '_smopa_i8_loop:'
        echo '    smstart'
        echo '    ptrue p0.b'
        echo '    zero {za}'
        echo '1:'
        echo '    smopa za0.s, p0/m, p0/m, z0.b, z1.b'
        echo '    smopa za1.s, p0/m, p0/m, z2.b, z3.b'
        echo '    smopa za2.s, p0/m, p0/m, z4.b, z5.b'
        echo '    smopa za3.s, p0/m, p0/m, z6.b, z7.b'
        echo '    subs x0, x0, #4'
        echo '    b.gt 1b'
        echo '    smstop'
        echo '    ret'
    } > "$PROBE_BIN.s"
    cat > "$PROBE_DIR/par_smopa.c" <<CEOF
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <mach/mach_time.h>
static uint64_t g_iters;
extern void smopa_i8_loop(uint64_t n);
static void* thread_fn(void* arg) { smopa_i8_loop(g_iters); return NULL; }
int main(void) {
    int nthreads = ${threads};
    g_iters = ${iters};
    pthread_t tids[64];
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    uint64_t t0 = mach_absolute_time();
    for (int i = 1; i < nthreads; i++)
        pthread_create(&tids[i], NULL, thread_fn, NULL);
    smopa_i8_loop(g_iters);
    for (int i = 1; i < nthreads; i++)
        pthread_join(tids[i], NULL);
    uint64_t t1 = mach_absolute_time();
    double secs = (double)(t1-t0) * tb.numer / tb.denom / 1e9;
    double total_ops = (double)g_iters * nthreads * 2048.0;
    double tops = total_ops / secs / 1e12;
    printf("  SMOPA i8 %2d threads: %8.3f TOPS  (%6.3f s, %llu ops/thread)\n",
           nthreads, tops, secs, (unsigned long long)g_iters);
    return 0;
}
CEOF
    if ! $CC $ASMFLAGS -c "$PROBE_BIN.s" -o "$PROBE_OBJ" 2>/dev/null; then
        printf "  SMOPA i8 %2d threads: COMPILE_FAIL\n" "$threads"
        return
    fi
    if ! $CC $CFLAGS -c "$PROBE_DIR/par_smopa.c" -o "$PROBE_DIR/par_smopa.o" 2>/dev/null; then
        printf "  SMOPA i8 %2d threads: COMPILE_FAIL (main)\n" "$threads"
        return
    fi
    if ! $CC $CFLAGS -lpthread -o "$PROBE_BIN" "$PROBE_OBJ" "$PROBE_DIR/par_smopa.o" 2>/dev/null; then
        printf "  SMOPA i8 %2d threads: LINK_FAIL\n" "$threads"
        return
    fi
    "$PROBE_BIN" 2>/dev/null || printf "  SMOPA i8 %2d threads: FAILED\n" "$threads"
}
section "Multi-threaded SMOPA INT8 scaling"
for t in 1 2 4 6 8 12 16 18; do
    stress_parallel_smopa_i8 "$t"
done
echo ""
echo "  Note: SMOPA TOPS = iterations * threads * 2048ops / time"
echo "  (Each SMOPA za.s i8: 16x16x4x2 = 2048 INT8 ops via dot4)"
stress_parallel_luti() {
    local threads="$1"
    local instr="$2"
    local name="$3"
    local iters="${4:-1000000000}"
    rm -f "$PROBE_BIN" "$PROBE_BIN.s"
    {
        echo '.globl _luti_loop'
        echo '.p2align 2'
        echo '_luti_loop:'
        echo '    smstart'
        echo '    ptrue p0.b'
        echo '    zero {za}'
        echo '1:'
        echo "    $instr"
        echo "    $instr"
        echo "    $instr"
        echo "    $instr"
        echo "    $instr"
        echo "    $instr"
        echo "    $instr"
        echo "    $instr"
        echo '    subs x0, x0, #8'
        echo '    b.gt 1b'
        echo '    smstop'
        echo '    ret'
    } > "$PROBE_BIN.s"
    cat > "$PROBE_DIR/par_luti.c" <<CEOF
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <mach/mach_time.h>
static uint64_t g_iters;
extern void luti_loop(uint64_t n);
static void* thread_fn(void* arg) { luti_loop(g_iters); return NULL; }
int main(void) {
    int nthreads = ${threads};
    g_iters = ${iters};
    pthread_t tids[64];
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    uint64_t t0 = mach_absolute_time();
    for (int i = 1; i < nthreads; i++)
        pthread_create(&tids[i], NULL, thread_fn, NULL);
    luti_loop(g_iters);
    for (int i = 1; i < nthreads; i++)
        pthread_join(tids[i], NULL);
    uint64_t t1 = mach_absolute_time();
    double secs = (double)(t1-t0) * tb.numer / tb.denom / 1e9;
    double gops = (double)g_iters * nthreads / secs / 1e9;
    printf("  %-24s %2d threads: %8.2f Gops/s  (%6.3f s)\n",
           "${name}", nthreads, gops, secs);
    return 0;
}
CEOF
    if ! $CC $ASMFLAGS -c "$PROBE_BIN.s" -o "$PROBE_OBJ" 2>/dev/null; then
        printf "  %-24s %2d threads: COMPILE_FAIL\n" "$name" "$threads"
        return
    fi
    if ! $CC $CFLAGS -c "$PROBE_DIR/par_luti.c" -o "$PROBE_DIR/par_luti.o" 2>/dev/null; then
        printf "  %-24s %2d threads: COMPILE_FAIL (main)\n" "$name" "$threads"
        return
    fi
    if ! $CC $CFLAGS -lpthread -o "$PROBE_BIN" "$PROBE_OBJ" "$PROBE_DIR/par_luti.o" 2>/dev/null; then
        printf "  %-24s %2d threads: LINK_FAIL\n" "$name" "$threads"
        return
    fi
    "$PROBE_BIN" 2>/dev/null || printf "  %-24s %2d threads: FAILED\n" "$name" "$threads"
}
section "Multi-threaded LUTI throughput scaling"
for t in 1 2 4 6 8 12 16 18; do
    stress_parallel_luti "$t" "luti2 z0.b, zt0, z1[0]" "LUTI2 1v 8b"
done
echo ""
for t in 1 2 4 6 8 12 16 18; do
    stress_parallel_luti "$t" "luti4 z0.b, zt0, z1[0]" "LUTI4 1v 8b"
done
echo ""
for t in 1 2 4 6 8 12 16 18; do
    stress_parallel_luti "$t" "luti4 {z0.h-z1.h}, zt0, z2[0]" "LUTI4 2v 16b"
done
fi # end SKIP_TP
# ============================================================
# [7] SUMMARY
# ============================================================
# Cleanup
rm -f "$PROBE_BIN" "$PROBE_BIN.c" "$SIGILL_LOG"
section "Summary"
echo "  Total probed:     $TOTAL_COUNT"
echo "  Total working:    $WORKS_COUNT"
echo "  Expected fails:   $EXPECTED_FAIL_COUNT"
echo "  Unexpected fails: $FAIL_COUNT"
echo " "
echo -e "\033[35m        ------\033[36m Probe Completed. \033[35m------\033[0m"
echo " "
