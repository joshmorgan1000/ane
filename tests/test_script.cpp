/** --------------------------------------------------------------------------------------------------------- File Info
 * @file test_script.cpp
 * @brief Tests the ane::script DSL compiler — variable declarations, param loads/stores,
 *  element-wise addition, pointer advancement, and counted loops.
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

static constexpr int SVLs = 16;  ///< float32 elements per z-vector on M4/M5
static constexpr int VL   = 64;  ///< Bytes per z-vector
static void* alloc64(size_t size) {
    return std::aligned_alloc(64, ((size + 63) >> 6) << 6);
}
/** --------------------------------------------------------------------------------------------------------- Test 1: Simple Addition
 * @brief Loads two float32 z-vectors, adds them, stores the result.
 *  Single iteration, no loop.
 */
static bool test_simple_add() {
    printf("  [1] script: a = a + b (single vector) ... ");
    auto* a = static_cast<float*>(alloc64(VL));
    auto* b = static_cast<float*>(alloc64(VL));
    auto* c = static_cast<float*>(alloc64(VL));
    for (int i = 0; i < SVLs; i++) { a[i] = static_cast<float>(i); b[i] = 100.0f; c[i] = 0.0f; }
    ane::script s(R"(
        x: ZVEC_F32;
        y: ZVEC_F32;
        x.load(params[0]);
        y.load(params[1]);
        x = x + y;
        x.save(params[2]);
    )");
    s.exec({a, b, c});
    bool ok = true;
    for (int i = 0; i < SVLs; i++) {
        float expected = static_cast<float>(i) + 100.0f;
        if (std::fabs(c[i] - expected) > 0.01f) {
            printf("FAIL (c[%d]: expected %.1f, got %.1f)\n", i, expected, c[i]);
            ok = false;
            break;
        }
    }
    if (ok) printf("OK (c[0]=%.0f c[15]=%.0f)\n", c[0], c[15]);
    std::free(a); std::free(b); std::free(c);
    return ok;
}
/** --------------------------------------------------------------------------------------------------------- Test 2: Loop with Pointer Advance
 * @brief Adds 4 z-vectors element-wise in a loop, advancing all three pointers each iteration.
 */
static bool test_loop_add() {
    printf("  [2] script: loop 4× add with ptr advance ... ");
    constexpr int N = 4;
    auto* a = static_cast<float*>(alloc64(N * VL));
    auto* b = static_cast<float*>(alloc64(N * VL));
    auto* c = static_cast<float*>(alloc64(N * VL));
    for (int i = 0; i < N * SVLs; i++) {
        a[i] = static_cast<float>(i);
        b[i] = 1000.0f;
        c[i] = 0.0f;
    }
    ane::script s(R"(
        x: ZVEC_F32;
        y: ZVEC_F32;
        _LOOP_:;
        x.load(params[0]);
        y.load(params[1]);
        x = x + y;
        x.save(params[2]);
        params[0]++;
        params[1]++;
        params[2]++;
        goto _LOOP_ 4;
    )");
    s.exec({a, b, c});
    bool ok = true;
    for (int i = 0; i < N * SVLs; i++) {
        float expected = static_cast<float>(i) + 1000.0f;
        if (std::fabs(c[i] - expected) > 0.01f) {
            printf("FAIL (c[%d]: expected %.1f, got %.1f)\n", i, expected, c[i]);
            ok = false;
            break;
        }
    }
    if (ok) printf("OK (c[0]=%.0f c[63]=%.0f)\n", c[0], c[63]);
    std::free(a); std::free(b); std::free(c);
    return ok;
}
/** --------------------------------------------------------------------------------------------------------- Main
 */
int main() {
    printf("\n=== ane::script DSL tests ===\n\n");
    int passed = 0, total = 2;
    if (test_simple_add()) passed++;
    if (test_loop_add()) passed++;
    printf("\n  Results: %d/%d passed\n\n", passed, total);
    return (passed == total) ? 0 : 1;
}
