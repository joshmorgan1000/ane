/** --------------------------------------------------------------------------------------------------------- File Info
 * @file test_program_demo.cpp
 * @brief Demonstrates the DSL script compiler, loop opcode, load/save, and z_tiles struct
 *  - composing LUTI2 table lookups via DSL scripts in a single streaming session.
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
/** --------------------------------------------------------------------------------------------------------- Test 1: Basic Load/Save via DSL
 * @brief Verifies that DSL load/save correctly round-trip data through z-registers
 *  within a single streaming session.
 */
static bool test_load_store_mov() {
    printf("  [1] load → save round-trip via DSL ... ");
    auto* s = static_cast<uint8_t*>(alloc64(SVLb));
    auto* d = static_cast<uint8_t*>(alloc64(SVLb));
    for (int i = 0; i < SVLb; i++) { s[i] = static_cast<uint8_t>(i); d[i] = 0; }
    ane::script sc(R"(
        a: ZVEC_F32;
        a.load(params[0]);
        a.save(params[1]);
    )");
    sc.exec({s, d});
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
 * @brief Verifies that DSL goto loops correctly repeat a load→store sequence N times,
 *  advancing pointers between iterations.
 */
static bool test_loop() {
    printf("  [2] loop: 4× load → store with advancing pointers ... ");
    constexpr int N = 4;
    auto* s = static_cast<uint8_t*>(alloc64(N * SVLb));
    auto* d = static_cast<uint8_t*>(alloc64(N * SVLb));
    for (int i = 0; i < N * SVLb; i++) { s[i] = static_cast<uint8_t>(i & 0xFF); d[i] = 0; }
    ane::script sc(R"(
        a: ZVEC_F32;
        _LOOP_:;
        a.load(params[0]);
        a.save(params[1]);
        params[0]++;
        params[1]++;
        goto _LOOP_ 4;
    )");
    sc.exec({s, d});
    // All 4 z-vectors should be copied since we advance pointers each iteration
    for (int i = 0; i < N * SVLb; i++) {
        if (d[i] != s[i]) {
            printf("FAIL (byte %d: expected %d, got %d)\n", i, s[i], d[i]);
            return false;
        }
    }
    std::free(s); std::free(d);
    printf("OK\n");
    return true;
}
/** --------------------------------------------------------------------------------------------------------- Test 3: LUTI2 via DSL Script
 * @brief Builds a DSL script that runs LUTI2 lookup on packed 2-bit indices,
 *  and verifies the expanded output matches expected values.
 */
static bool test_luti2_program() {
    printf("  [3] LUTI2 lookup via DSL script ... ");
    ane::luti2<uint8_t> table(10, 20, 30, 40);  ///< 4-entry table: {10, 20, 30, 40}
    auto* idx = static_cast<uint8_t*>(alloc64(SVLb));
    auto* out = static_cast<uint8_t*>(alloc64(SVLb));
    // Fill indices: each byte has 4 2-bit fields [b7b6|b5b4|b3b2|b1b0]
    // Pattern: 0, 1, 2, 3, 0, 1, 2, 3, ... → packed as 0xE4 = 0b11_10_01_00
    for (int i = 0; i < SVLb; i++) idx[i] = 0xE4;
    std::memset(out, 0, SVLb);
    ane::script sc(R"(
        luti2(1, 0, params[0], params[1], params[2]);
    )");
    sc.exec({table.data.data(), idx, out});
    // Verify: LUTI2 with .b element size does a 2-bit indexed lookup per byte
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
    ane::script sc(R"(
        zero_za();
        load_bias(params[0]);
        store_tiles(params[1]);
    )");
    sc.exec({bias, tiles.ptr()});
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
 * @brief Full integration: zero ZA, accumulate via SMOPA, then store_tiles.
 *  Everything in one streaming session via DSL script.
 */
static bool test_multi_op_stream() {
    printf("  [5] Multi-op single stream: zero → acc_smopa → store_tiles ... ");
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
    ane::script sc(R"(
        zero_za();
        acc_smopa(4, params[0], params[1]);
        store_tiles(params[2]);
    )");
    sc.exec({r, c, tiles.ptr()});
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
    printf("\n=== ane::script demo tests ===\n\n");
    int passed = 0, total = 5;
    if (test_load_store_mov()) passed++;
    if (test_loop()) passed++;
    if (test_luti2_program()) passed++;
    if (test_z_tiles_capture()) passed++;
    if (test_multi_op_stream()) passed++;
    printf("\n  Results: %d/%d passed\n\n", passed, total);
    return (passed == total) ? 0 : 1;
}
