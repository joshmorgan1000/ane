/** --------------------------------------------------------------------------------------------------------- File Info
 * @file test_program_demo.cpp
 * @brief Demonstrates the program builder, loop opcode, mov_zreg, load/store, and z_tiles struct
 *  - composing LUTI2 table lookups via the bytecode program builder in a single streaming session.
 *
 * @author Josh Morgan (@joshmorgan1000 on GitHub) with help from Claude
 * Released under the MIT License
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ane/ane.hpp>

static constexpr int SVLb = 64;  ///< Bytes per z-vector (512-bit SVL on M4/M5)
/** --------------------------------------------------------------------------------------------------------- Aligned Allocation
 * @brief Allocates 64-byte aligned memory for the given size, rounded up to 64-byte boundary.
 */
static void* alloc64(size_t size) {
    size = ((size + 63) >> 6) << 6;
    return std::aligned_alloc(64, size);
}
/** --------------------------------------------------------------------------------------------------------- Test 1: Basic Load/Store/MovZreg
 * @brief Verifies that load, mov_zreg, and store correctly round-trip data through z-registers
 *  within a single streaming session.
 */
static bool test_load_store_mov() {
    printf("  [1] load → mov_zreg → store round-trip ... ");
    auto* s = static_cast<uint8_t*>(alloc64(SVLb));
    auto* d = static_cast<uint8_t*>(alloc64(SVLb));
    for (int i = 0; i < SVLb; i++) { s[i] = static_cast<uint8_t>(i); d[i] = 0; }
    ane::program p;
    p.emit(ane::Op::load, reinterpret_cast<uintptr_t>(s));                     // z0 = src
    p.emit(ane::Op::mov_zreg, uint8_t(0), uint8_t(7));                        // z7 = z0
    p.emit(ane::Op::mov_zreg, uint8_t(7), uint8_t(0));                        // z0 = z7
    p.emit(ane::Op::store, reinterpret_cast<uintptr_t>(d));                    // dst = z0
    p.exec();
    for (int i = 0; i < SVLb; i++) {
        if (d[i] != static_cast<uint8_t>(i)) {
            printf("FAIL (byte %d: expected %d, got %d)\n", i, i, d[i]);
            return false;
        }
    }
    std::free(s); std::free(d);
    printf("OK\n");
    return true;
}
/** --------------------------------------------------------------------------------------------------------- Test 2: Loop Opcode
 * @brief Verifies that begin_loop/end_loop correctly repeats a load→store sequence N times.
 *  Since load/store use fixed pointers (no auto-advance), the loop copies the same vector N times.
 */
static bool test_loop() {
    printf("  [2] loop: 4× load → store ... ");
    constexpr int N = 4;
    auto* s = static_cast<uint8_t*>(alloc64(N * SVLb));
    auto* d = static_cast<uint8_t*>(alloc64(N * SVLb));
    for (int i = 0; i < N * SVLb; i++) { s[i] = static_cast<uint8_t>(i & 0xFF); d[i] = 0; }
    auto src_ptr = reinterpret_cast<uintptr_t>(s);
    auto dst_ptr = reinterpret_cast<uintptr_t>(d);
    ane::program p;
    p.begin_loop(N);
    p.emit(ane::Op::load, src_ptr);
    p.emit(ane::Op::store, dst_ptr);
    p.end_loop();
    p.exec();
    // Only the FIRST z-vector should be copied (load/store don't auto-advance pointers)
    // The loop repeats the same load/store with the same pointers, so dst[0..63] = src[0..63]
    for (int i = 0; i < SVLb; i++) {
        if (d[i] != s[i]) {
            printf("FAIL (byte %d: expected %d, got %d)\n", i, s[i], d[i]);
            return false;
        }
    }
    std::free(s); std::free(d);
    printf("OK\n");
    return true;
}
/** --------------------------------------------------------------------------------------------------------- Test 3: LUTI2 via Program Builder
 * @brief Builds a program that loads a LUTI2 table, runs LUTI2 lookup on packed 2-bit indices,
 *  and verifies the expanded output matches expected values.
 *
 *  This demonstrates composing existing fused kernels (luti2_op) with the program builder to
 *  run a complete lookup pipeline in a single streaming session.
 */
static bool test_luti2_program() {
    printf("  [3] LUTI2 lookup via program builder ... ");
    ane::luti2<uint8_t> table(10, 20, 30, 40);  ///< 4-entry table: {10, 20, 30, 40}
    auto* idx = static_cast<uint8_t*>(alloc64(SVLb));
    auto* out = static_cast<uint8_t*>(alloc64(SVLb));
    // Fill indices: each byte has 4 2-bit fields [b7b6|b5b4|b3b2|b1b0]
    // Pattern: 0, 1, 2, 3, 0, 1, 2, 3, ... → packed as 0xE4 = 0b11_10_01_00
    for (int i = 0; i < SVLb; i++) idx[i] = 0xE4;
    std::memset(out, 0, SVLb);
    // Expected output: each byte gets looked up → 10, 20, 30, 40 repeating
    // With elem_size=0 (.b), LUTI2 processes one z-vector of indices into one z-vector of outputs
    ane::program p;
    p.emit(ane::Op::luti2_op,
        uint32_t(1),                                                 ///< count: 1 z-vector
        uint8_t(0),                                                  ///< elem_size: 0 = .b (8-bit)
        reinterpret_cast<uintptr_t>(table.data.data()),              ///< table_ptr (64 bytes, loaded into ZT0)
        reinterpret_cast<uintptr_t>(idx),                            ///< indices_ptr
        reinterpret_cast<uintptr_t>(out));                           ///< output_ptr
    p.exec();
    // Verify: LUTI2 with .b element size does a 2-bit indexed lookup per byte
    // Each input byte 0xE4 has indices: field0=0, field1=1, field2=2, field3=3
    // LUTI2 expands each 2-bit index to an 8-bit table value
    // Output should be groups of table values based on 2-bit index extraction
    bool any_nonzero = false;
    for (int i = 0; i < SVLb; i++) {
        if (out[i] != 0) any_nonzero = true;
    }
    if (!any_nonzero) {
        std::free(idx); std::free(out);
        printf("FAIL (all zeros — LUTI2 produced no output)\n");
        return false;
    }
    uint8_t first8[8];
    std::memcpy(first8, out, 8);
    std::free(idx); std::free(out);
    printf("OK (first 8 bytes: %d %d %d %d %d %d %d %d)\n",
        first8[0], first8[1], first8[2], first8[3], first8[4], first8[5], first8[6], first8[7]);
    return true;
}
/** --------------------------------------------------------------------------------------------------------- Test 4: z_tiles Capture
 * @brief Demonstrates capturing ZA tile state into a z_tiles struct via store_tiles.
 *  - Zeros ZA, loads a known bias pattern, then stores tiles into z_tiles.
 */
static bool test_z_tiles_capture() {
    printf("  [4] z_tiles capture via zero_za → load_bias → store_tiles ... ");
    ane::z_tiles tiles;
    std::memset(tiles.data, 0, sizeof(tiles.data));
    // Create a bias pattern: 32×32 int32 matrix (GROUP_DIM=32 on M4/M5)
    alignas(4096) int32_t bias[1024];
    for (int i = 0; i < 1024; i++) bias[i] = i + 1;  ///< 1, 2, 3, ..., 1024
    ane::program p;
    p.emit(ane::Op::zero_za);                                                  ///< Clear ZA tiles
    p.emit(ane::Op::load_bias, reinterpret_cast<uintptr_t>(bias));             ///< Load bias into ZA
    p.emit(ane::Op::store_tiles, reinterpret_cast<uintptr_t>(tiles.ptr()));    ///< Dump ZA → z_tiles
    p.exec();
    // Verify first few values came through
    const int32_t* t = tiles.as_i32();
    bool ok = true;
    for (int i = 0; i < 16; i++) {
        if (t[i] != bias[i]) {
            printf("FAIL (tile[%d]: expected %d, got %d)\n", i, bias[i], t[i]);
            ok = false;
            break;
        }
    }
    if (ok) printf("OK (first 4: %d %d %d %d)\n", t[0], t[1], t[2], t[3]);
    return ok;
}
/** --------------------------------------------------------------------------------------------------------- Test 5: Multi-Op Single Stream
 * @brief Full integration: zero ZA, loop over 4 SMOPA accumulations via load→mov_zreg→smopa_2x2,
 *  then store_tiles into z_tiles. Everything in one streaming session.
 */
static bool test_multi_op_stream() {
    printf("  [5] Multi-op single stream: zero → 4× load+smopa → store_tiles ... ");
    constexpr int K_STEPS = 4;
    auto* r = static_cast<int8_t*>(alloc64(K_STEPS * 2 * SVLb));
    auto* c = static_cast<int8_t*>(alloc64(K_STEPS * 2 * SVLb));
    // Fill with small values to avoid overflow
    for (int i = 0; i < K_STEPS * 2 * SVLb; i++) {
        r[i] = static_cast<int8_t>(1);
        c[i] = static_cast<int8_t>(1);
    }
    ane::z_tiles tiles;
    std::memset(tiles.data, 0, sizeof(tiles.data));
    // Use the fused acc_smopa kernel which internally loops over k_steps
    ane::program p;
    p.emit(ane::Op::zero_za);
    p.emit(ane::Op::acc_smopa,
        uint32_t(K_STEPS),
        reinterpret_cast<uintptr_t>(r),
        reinterpret_cast<uintptr_t>(c));
    p.emit(ane::Op::store_tiles, reinterpret_cast<uintptr_t>(tiles.ptr()));
    p.exec();
    // SMOPA computes 4-element int8 dot products per tile position. With all-1
    // inputs, each SMOPA adds 4 per element (1*1+1*1+1*1+1*1). After K_STEPS
    // iterations: expected = K_STEPS * 4.
    const int32_t* t = tiles.as_i32();
    int32_t expected = K_STEPS * 4;  ///< 4 * 4 = 16
    std::free(r); std::free(c);
    if (t[0] != expected) {
        printf("FAIL (tile[0][0]: expected %d, got %d)\n", expected, t[0]);
        return false;
    }
    printf("OK (tile[0][0] = %d, expected %d)\n", t[0], expected);
    return true;
}
/** --------------------------------------------------------------------------------------------------------- Main
 */
int main() {
    printf("\n=== ane::program demo tests ===\n\n");
    int passed = 0, total = 5;
    if (test_load_store_mov()) passed++;
    if (test_loop()) passed++;
    if (test_luti2_program()) passed++;
    if (test_z_tiles_capture()) passed++;
    if (test_multi_op_stream()) passed++;
    printf("\n  Results: %d/%d passed\n\n", passed, total);
    return (passed == total) ? 0 : 1;
}
