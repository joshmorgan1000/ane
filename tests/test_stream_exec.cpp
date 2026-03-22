#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include <simd/simd.h>
#include "../include/ane/streamexec/bytecodes.hpp"
#include "../include/ane/streamexec/compiler.hpp"

extern "C" void stream_exec(const void* compiled_program);

// ── Reference i8 matmul (naive, matches SME 2x2 tile layout) ────────────────
// A is (GROUP_DIM × K) stored as row panels: each k-step is 2*VL bytes
// B is (K × GROUP_DIM) stored as col panels: each k-step is 2*VL bytes
// C is (GROUP_DIM × GROUP_DIM) int32
//
// SME smopa does: za[i,j] += Σ_d (a_row[4i+d] * b_col[4j+d])  for d=0..3
// So each k-step consumes 4 K-elements (dot4).

static constexpr int SVLs = 16;          // M4: 16 int32 lanes
static constexpr int SVLb = 64;          // M4: 64 byte lanes
static constexpr int GROUP_DIM = 2 * SVLs; // 32

// Reference: C[i][j] = Σ_k A[i][k] * B[k][j], all int8 → int32
static void ref_matmul_i8(const int8_t* A, const int8_t* B, int32_t* C,
                           int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)A[i * K + j] * (int32_t)B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

// Pack A into row panels for SME: each k-step loads 2*SVLb contiguous bytes
// Panel layout: for k-step t, bytes [t*128 .. t*128+127] contain:
//   z0: rows 0..15, 4 bytes each (dot4 group for columns 4t..4t+3)
//   z1: rows 16..31, 4 bytes each
static void pack_rows_i8(const int8_t* A, int8_t* packed, int M, int K) {
    int k_steps = K / 4;
    for (int t = 0; t < k_steps; t++) {
        int8_t* dst = packed + t * 2 * SVLb;
        // z0: rows 0..SVLs-1
        for (int r = 0; r < SVLs; r++) {
            for (int d = 0; d < 4; d++) {
                dst[r * 4 + d] = A[r * K + t * 4 + d];
            }
        }
        // z1: rows SVLs..2*SVLs-1
        for (int r = 0; r < SVLs; r++) {
            for (int d = 0; d < 4; d++) {
                dst[SVLb + r * 4 + d] = A[(SVLs + r) * K + t * 4 + d];
            }
        }
    }
}

// Pack B into col panels for SME: each k-step loads 2*SVLb contiguous bytes
// Panel layout: for k-step t, bytes [t*128 .. t*128+127] contain:
//   z2: cols 0..15, 4 bytes each
//   z3: cols 16..31, 4 bytes each
static void pack_cols_i8(const int8_t* B, int8_t* packed, int N, int K) {
    int k_steps = K / 4;
    for (int t = 0; t < k_steps; t++) {
        int8_t* dst = packed + t * 2 * SVLb;
        // z2: cols 0..SVLs-1
        for (int c = 0; c < SVLs; c++) {
            for (int d = 0; d < 4; d++) {
                dst[c * 4 + d] = B[(t * 4 + d) * N + c];
            }
        }
        // z3: cols SVLs..2*SVLs-1
        for (int c = 0; c < SVLs; c++) {
            for (int d = 0; d < 4; d++) {
                dst[SVLb + c * 4 + d] = B[(t * 4 + d) * N + SVLs + c];
            }
        }
    }
}

static bool check_result(const int32_t* got, const int32_t* expected, int count,
                          const char* test_name) {
    int errors = 0;
    for (int i = 0; i < count; i++) {
        if (got[i] != expected[i]) {
            if (errors < 5) {
                printf("  MISMATCH [%d]: got %d, expected %d\n", i, got[i], expected[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("  PASS: %s\n", test_name);
        return true;
    } else {
        printf("  FAIL: %s (%d errors)\n", test_name, errors);
        return false;
    }
}

// ── Test 1: Single tile, signed i8 matmul ────────────────────────────────────
static bool test_single_tile_smopa() {
    printf("\n=== Test: single tile smopa (32x32 × 32xK, K=16) ===\n");

    const int M = GROUP_DIM, N = GROUP_DIM, K = 16;
    const int k_steps = K / 4;

    // Allocate source matrices
    std::vector<int8_t> A(M * K), B(K * N);
    for (int i = 0; i < M * K; i++) A[i] = (int8_t)((i % 7) - 3);  // -3..3
    for (int i = 0; i < K * N; i++) B[i] = (int8_t)((i % 5) - 2);  // -2..2

    // Reference result
    std::vector<int32_t> C_ref(M * N, 0);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                C_ref[i * N + j] += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];

    // Pack for SME
    auto* packed_rows = (int8_t*)aligned_alloc(64, k_steps * 2 * SVLb);
    auto* packed_cols = (int8_t*)aligned_alloc(64, k_steps * 2 * SVLb);
    pack_rows_i8(A.data(), packed_rows, M, K);
    pack_cols_i8(B.data(), packed_cols, N, K);

    // Output: 32 rows × 32 cols = 1024 int32 = 64 × simd_int16
    auto* output = (simd_int16*)aligned_alloc(64, GROUP_DIM * GROUP_DIM * sizeof(int32_t));
    memset(output, 0, GROUP_DIM * GROUP_DIM * sizeof(int32_t));

    // Compile & execute
    std::vector<ane::ByteCode> program = {
        ane::Zero{},
        ane::Accumulate{reinterpret_cast<simd_uchar64*>(packed_rows), reinterpret_cast<simd_uchar64*>(packed_cols), (uint32_t)k_steps, ane::MopaType::signed_i8},
    };
    auto compiled = ane::compile(program, 1, output);
    stream_exec(&compiled);

    // Check
    bool ok = check_result(reinterpret_cast<int32_t*>(output), C_ref.data(), M * N, "single_tile_smopa");

    free(packed_rows);
    free(packed_cols);
    free(output);
    return ok;
}

// ── Test 2: Multi-tile loop ──────────────────────────────────────────────────
static bool test_multi_tile() {
    printf("\n=== Test: multi-tile loop (4 tiles, K=8) ===\n");

    const int M = GROUP_DIM, N = GROUP_DIM, K = 8;
    const int k_steps = K / 4;
    const int num_tiles = 4;

    std::vector<int8_t> A(M * K), B(K * N);
    for (int i = 0; i < M * K; i++) A[i] = (int8_t)(i % 3);
    for (int i = 0; i < K * N; i++) B[i] = (int8_t)(i % 4);

    // Reference: same matmul repeated num_tiles times to same output
    // Each tile gets the same data, output = num_tiles × single result
    std::vector<int32_t> C_single(M * N, 0);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                C_single[i * N + j] += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];

    auto* packed_rows = (int8_t*)aligned_alloc(64, k_steps * 2 * SVLb);
    auto* packed_cols = (int8_t*)aligned_alloc(64, k_steps * 2 * SVLb);
    pack_rows_i8(A.data(), packed_rows, M, K);
    pack_cols_i8(B.data(), packed_cols, N, K);

    // For multi-tile: each loop iteration does zero+acc+store
    // The operand tape needs entries for each iteration
    // With loop_count, halt resets bytecode IP but operand ptr advances
    // So we need num_tiles copies of the operands in the tape
    size_t tile_bytes = GROUP_DIM * GROUP_DIM * sizeof(int32_t);
    auto* output = (simd_int16*)aligned_alloc(64, num_tiles * tile_bytes);
    memset(output, 0, num_tiles * tile_bytes);

    // Build raw CompiledProgram — multi-tile needs separate store per tile
    ane::CompiledProgram<simd_int16> cp;
    cp.loop_count = 1;
    cp.output = output;

    auto* rows_simd = reinterpret_cast<simd_uchar64*>(packed_rows);
    auto* cols_simd = reinterpret_cast<simd_uchar64*>(packed_cols);

    for (int t = 0; t < num_tiles; t++) {
        cp.bytecodes.push_back(ane::op::zero_za);
        cp.bytecodes.push_back(ane::op::acc_smopa);
        uint32_t ks = k_steps;
        uint8_t buf[4];
        memcpy(buf, &ks, 4);
        cp.bytecodes.insert(cp.bytecodes.end(), buf, buf + 4);
        cp.operands.push_back(rows_simd);
        cp.operands.push_back(cols_simd);
        cp.bytecodes.push_back(ane::op::store_tiles);
        auto* tile_out = reinterpret_cast<simd_int16*>(
            reinterpret_cast<uint8_t*>(output) + t * tile_bytes);
        cp.operands.push_back(tile_out);
    }
    cp.bytecodes.push_back(ane::op::halt);

    stream_exec(&cp);

    bool ok = true;
    for (int t = 0; t < num_tiles; t++) {
        auto* result = reinterpret_cast<int32_t*>(output) + t * (tile_bytes / sizeof(int32_t));
        char name[64];
        snprintf(name, sizeof(name), "multi_tile[%d]", t);
        ok &= check_result(result, C_single.data(), M * N, name);
    }

    free(packed_rows);
    free(packed_cols);
    free(output);
    return ok;
}

int main() {
    printf("stream_exec bytecode interpreter tests\n");
    printf("SVLs=%d, SVLb=%d, GROUP_DIM=%d\n", SVLs, SVLb, GROUP_DIM);

    bool all_pass = true;
    all_pass &= test_single_tile_smopa();
    all_pass &= test_multi_tile();

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
