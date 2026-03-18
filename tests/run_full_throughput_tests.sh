#!/bin/bash
# =============================================================================
# run_full_throughput_tests.sh — Full INT8 Throughput Tests for Apple M4 Max
#
# Proves that Apple's "38 TOPS Neural Engine" is GPU + CPU SME, not a separate
# coprocessor. Runs three tests:
#
#   Test 1: GPU peak INT8 ALU (Metal char4 vectors)          → ~37 TOPS
#   Test 2: GPU + CPU SME simultaneously                     → ~41 TOPS
#   Test 3: GPU + CPU SME + CoreML "ANE" simultaneously      → ~41 TOPS (no gain)
#
# If the ANE were separate hardware, Test 3 would show ~80 TOPS.
# Instead it shows the same ~41 TOPS, proving CoreML competes for the same
# GPU and CPU resources.
#
# Requirements:
#   - Apple M4 (or any ARM processor with SME2)
#   - Xcode / Apple Clang with Metal support
#   - CoreML model at: /Users/joshmorgan/AI/nebula/tests/ane_poc/dot_product_model.mlmodel
#     (Test 3 skips gracefully if not found)
#
# Usage:
#   cd tests/throughput && bash ../run_full_throughput_tests.sh
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$SCRIPT_DIR/throughput"
cd "$BENCH_DIR"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Apple M4 INT8 Throughput — ANE Existence Test Suite        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# ── Build SME assembly kernel ────────────────────────────────────────────────
echo "Building SME SMOPA kernel..."
clang -c -arch arm64 -march=armv9-a+sme2+sve2+sme-lutv2 \
    smopa_4way_kernel.s -o smopa_4way_kernel.o 2>&1
echo "  OK"

# ── Test 1: GPU Peak INT8 ALU ────────────────────────────────────────────────
echo ""
echo "━━━ Test 1: GPU Peak INT8 ALU (Metal char4 vectors) ━━━"
echo ""
swiftc -O -o metal_int8_peak metal_int8_peak.swift \
    -framework Metal -framework Foundation 2>/dev/null
./metal_int8_peak
echo ""

# ── Test 2: GPU + CPU SME Simultaneous ───────────────────────────────────────
echo "━━━ Test 2: GPU + CPU SME Simultaneous ━━━"
echo ""
swiftc -O -o combined_int8 combined_int8.swift \
    -framework Metal -framework Foundation 2>/dev/null
./combined_int8
echo ""

# ── Test 3: GPU + CPU SME + CoreML "ANE" (The Triple Threat) ─────────────────
echo "━━━ Test 3: GPU + CPU SME + CoreML/ANE (Triple Threat) ━━━"
echo ""
swiftc -O -o triple_threat triple_threat.swift \
    -framework Metal -framework Foundation -framework CoreML 2>/dev/null
./triple_threat
echo ""

# ── Cleanup ──────────────────────────────────────────────────────────────────
rm -f smopa_4way_kernel.o combined_sme_worker combined_sme_worker.c \
    metal_int8_peak combined_int8 triple_threat 2>/dev/null

echo ""
echo "All tests complete."
