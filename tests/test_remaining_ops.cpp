/** --------------------------------------------------------------------------------------------------------- File Info
 * @file test_remaining_ops.cpp
 * @brief Tests for remaining untested kernels: cblas_bfgemm, cblas_igemm, get_rows_fp32,
 * get_rows_q8_0, get_rows_q4_0, flash_attention_fp32.
 * Each test compares ANE output against a CPU scalar reference computed in double precision.
 *
 * @author Josh Morgan (@joshmorgan1000 on GitHub) with help from Claude
 * Released under the MIT License
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <ane/ane.hpp>

static int tests_passed = 0;
static int tests_total = 0;
static void* alloc(size_t n) { return std::aligned_alloc(64, ((n + 63) / 64) * 64); }
static bool check(const char* name, const float* got, const float* expected, int n, float tol) {
    tests_total++;
    float max_err = 0;
    int worst_idx = 0;
    for (int i = 0; i < n; i++) {
        float err = std::fabs(got[i] - expected[i]);
        if (err > max_err) { max_err = err; worst_idx = i; }
    }
    if (max_err <= tol) {
        printf("  [PASS] %s (max_err=%.2e at [%d])\n", name, max_err, worst_idx);
        tests_passed++;
        return true;
    }
    printf("  [FAIL] %s (max_err=%.2e at [%d]: got %.6f, expected %.6f)\n",
        name, max_err, worst_idx, got[worst_idx], expected[worst_idx]);
    return false;
}
static bool check_scalar(const char* name, float got, float expected, float tol) {
    tests_total++;
    float err = std::fabs(got - expected);
    if (err <= tol) {
        printf("  [PASS] %s (got=%.6f expected=%.6f err=%.2e)\n", name, got, expected, err);
        tests_passed++;
        return true;
    }
    printf("  [FAIL] %s (got=%.6f expected=%.6f err=%.2e)\n", name, got, expected, err);
    return false;
}
/** --------------------------------------------------------------------------------------------------------- BF16 Helpers
 */
static uint16_t float_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return static_cast<uint16_t>(bits >> 16);
}
static float bf16_to_float(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
/** --------------------------------------------------------------------------------------------------------- FP16 Helper
 */
static uint16_t float_to_fp16(float f) {
    __fp16 h = (__fp16)f;
    uint16_t bits;
    std::memcpy(&bits, &h, 2);
    return bits;
}
/** --------------------------------------------------------------------------------------------------------- cblas_bfgemm Tests
 */
static void test_cblas_bfgemm() {
    printf("\n-- cblas_bfgemm --\n");
    const int M = 16, N = 32, K = 32;
    const float alpha = 1.0f, beta = 0.0f;
    auto* A_fp32 = static_cast<float*>(alloc(M * K * 4));
    auto* B_fp32 = static_cast<float*>(alloc(K * N * 4));
    auto* A_bf16 = static_cast<uint16_t*>(alloc(M * K * 2));
    auto* B_bf16 = static_cast<uint16_t*>(alloc(K * N * 2));
    auto* C      = static_cast<float*>(alloc(M * N * 4));
    auto* ref    = static_cast<float*>(alloc(M * N * 4));
    for (int i = 0; i < M * K; i++) {
        A_fp32[i] = 0.01f * (i % 37 - 18);
        A_bf16[i] = float_to_bf16(A_fp32[i]);
    }
    for (int i = 0; i < K * N; i++) {
        B_fp32[i] = 0.01f * (i % 41 - 20);
        B_bf16[i] = float_to_bf16(B_fp32[i]);
    }
    // CPU reference using bf16-truncated values (double precision accumulation)
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < K; k++)
                sum += (double)bf16_to_float(A_bf16[i * K + k]) * (double)bf16_to_float(B_bf16[k * N + j]);
            ref[i * N + j] = (float)(alpha * sum + beta);
        }
    std::memset(C, 0, M * N * 4);
    ane::dispatch(ane::Op::cblas_bfgemm,
        uint8_t(0), uint32_t(M), uint32_t(N), uint32_t(K),
        uint32_t(K), uint32_t(N), uint32_t(N),
        alpha, beta,
        reinterpret_cast<uintptr_t>(A_bf16),
        reinterpret_cast<uintptr_t>(B_bf16),
        reinterpret_cast<uintptr_t>(C));
    check("cblas_bfgemm 16x32 K=32", C, ref, M * N, 1e-1f);
    // Test with non-zero beta
    for (int i = 0; i < M * N; i++) C[i] = 0.5f;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < K; k++)
                sum += (double)bf16_to_float(A_bf16[i * K + k]) * (double)bf16_to_float(B_bf16[k * N + j]);
            ref[i * N + j] = (float)(1.0 * sum + 2.0 * 0.5);
        }
    for (int i = 0; i < M * N; i++) C[i] = 0.5f;
    ane::dispatch(ane::Op::cblas_bfgemm,
        uint8_t(0), uint32_t(M), uint32_t(N), uint32_t(K),
        uint32_t(K), uint32_t(N), uint32_t(N),
        1.0f, 2.0f,
        reinterpret_cast<uintptr_t>(A_bf16),
        reinterpret_cast<uintptr_t>(B_bf16),
        reinterpret_cast<uintptr_t>(C));
    check("cblas_bfgemm alpha=1 beta=2", C, ref, M * N, 1e-1f);
    std::free(A_fp32); std::free(B_fp32); std::free(A_bf16); std::free(B_bf16);
    std::free(C); std::free(ref);
}
/** --------------------------------------------------------------------------------------------------------- cblas_igemm Tests
 */
static void test_cblas_igemm() {
    printf("\n-- cblas_igemm --\n");
    const int M = 16, N = 32, K = 64;
    const float alpha = 1.0f, beta = 0.0f;
    auto* A = static_cast<int8_t*>(alloc(M * K));
    auto* B = static_cast<int8_t*>(alloc(K * N));
    auto* C = static_cast<float*>(alloc(M * N * 4));
    auto* ref = static_cast<float*>(alloc(M * N * 4));
    for (int i = 0; i < M * K; i++) A[i] = static_cast<int8_t>((i % 11) - 5);
    for (int i = 0; i < K * N; i++) B[i] = static_cast<int8_t>((i % 13) - 6);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < K; k++) sum += (int)A[i * K + k] * (int)B[k * N + j];
            ref[i * N + j] = (float)sum * alpha;
        }
    std::memset(C, 0, M * N * 4);
    ane::dispatch(ane::Op::cblas_igemm,
        uint8_t(0), uint32_t(M), uint32_t(N), uint32_t(K),
        uint32_t(K), uint32_t(N), uint32_t(N),
        alpha, beta,
        reinterpret_cast<uintptr_t>(A),
        reinterpret_cast<uintptr_t>(B),
        reinterpret_cast<uintptr_t>(C));
    check("cblas_igemm 16x32 K=64 notransB", C, ref, M * N, 1e-3f);
    // Test alpha scaling: output should scale proportionally
    auto* C2 = static_cast<float*>(alloc(M * N * 4));
    std::memset(C2, 0, M * N * 4);
    ane::dispatch(ane::Op::cblas_igemm,
        uint8_t(0), uint32_t(M), uint32_t(N), uint32_t(K),
        uint32_t(K), uint32_t(N), uint32_t(N),
        0.5f, beta,
        reinterpret_cast<uintptr_t>(A),
        reinterpret_cast<uintptr_t>(B),
        reinterpret_cast<uintptr_t>(C2));
    tests_total++;
    float max_ratio_err = 0;
    int worst_idx = 0;
    for (int i = 0; i < M * N; i++) {
        if (std::fabs(C[i]) > 1e-6f) {
            float expected = C[i] * 0.5f;
            float err = std::fabs(C2[i] - expected);
            if (err > max_ratio_err) { max_ratio_err = err; worst_idx = i; }
        }
    }
    if (max_ratio_err <= 1e-3f) {
        printf("  [PASS] cblas_igemm alpha scaling consistent (max_err=%.2e)\n", max_ratio_err);
        tests_passed++;
    } else {
        printf("  [FAIL] cblas_igemm alpha scaling inconsistent (max_err=%.2e at [%d])\n",
            max_ratio_err, worst_idx);
    }
    std::free(A); std::free(B); std::free(C); std::free(C2); std::free(ref);
}
/** --------------------------------------------------------------------------------------------------------- get_rows_fp32 Tests
 */
static void test_get_rows_fp32() {
    printf("\n-- get_rows_fp32 --\n");
    const int vocab = 8, dim = 16, n_rows = 3;
    auto* table   = static_cast<float*>(alloc(vocab * dim * 4));
    auto* indices = static_cast<uint32_t*>(alloc(n_rows * 4));
    auto* out     = static_cast<float*>(alloc(n_rows * dim * 4));
    auto* ref     = static_cast<float*>(alloc(n_rows * dim * 4));
    for (int i = 0; i < vocab * dim; i++) table[i] = 0.1f * (i % 53 - 26);
    indices[0] = 2;
    indices[1] = 5;
    indices[2] = 0;
    // CPU reference
    for (int r = 0; r < n_rows; r++)
        for (int d = 0; d < dim; d++)
            ref[r * dim + d] = table[indices[r] * dim + d];
    std::memset(out, 0, n_rows * dim * 4);
    ane::dispatch(ane::Op::get_rows_fp32,
        uint32_t(n_rows), uint32_t(dim),
        reinterpret_cast<uintptr_t>(table),
        reinterpret_cast<uintptr_t>(indices),
        reinterpret_cast<uintptr_t>(out));
    check("get_rows_fp32 3 rows from 8x16", out, ref, n_rows * dim, 1e-6f);
    // Test with larger dim (multiple of 16)
    const int dim2 = 48;
    auto* table2   = static_cast<float*>(alloc(vocab * dim2 * 4));
    auto* out2     = static_cast<float*>(alloc(n_rows * dim2 * 4));
    auto* ref2     = static_cast<float*>(alloc(n_rows * dim2 * 4));
    for (int i = 0; i < vocab * dim2; i++) table2[i] = 0.05f * (i % 97 - 48);
    for (int r = 0; r < n_rows; r++)
        for (int d = 0; d < dim2; d++)
            ref2[r * dim2 + d] = table2[indices[r] * dim2 + d];
    std::memset(out2, 0, n_rows * dim2 * 4);
    ane::dispatch(ane::Op::get_rows_fp32,
        uint32_t(n_rows), uint32_t(dim2),
        reinterpret_cast<uintptr_t>(table2),
        reinterpret_cast<uintptr_t>(indices),
        reinterpret_cast<uintptr_t>(out2));
    check("get_rows_fp32 3 rows from 8x48", out2, ref2, n_rows * dim2, 1e-6f);
    std::free(table); std::free(indices); std::free(out); std::free(ref);
    std::free(table2); std::free(out2); std::free(ref2);
}
/** --------------------------------------------------------------------------------------------------------- get_rows_q8_0 Tests
 */
static void test_get_rows_q8_0() {
    printf("\n-- get_rows_q8_0 --\n");
    const int vocab = 4, dim = 32, n_rows = 2;
    const int blocks_per_row = dim / 32;        ///< 1
    const int block_bytes = 34;                 ///< fp16 scale (2) + 32 int8 quants
    const int row_bytes = blocks_per_row * block_bytes;
    auto* table   = static_cast<uint8_t*>(alloc(vocab * row_bytes));
    auto* indices = static_cast<uint32_t*>(alloc(n_rows * 4));
    auto* out     = static_cast<float*>(alloc(n_rows * dim * 4));
    auto* ref     = static_cast<float*>(alloc(n_rows * dim * 4));
    // Build Q8_0 table manually: each block is {fp16_scale, 32 x int8}
    for (int v = 0; v < vocab; v++) {
        uint8_t* row = table + v * row_bytes;
        for (int b = 0; b < blocks_per_row; b++) {
            uint8_t* block = row + b * block_bytes;
            float scale = 0.1f * (v + 1);
            uint16_t scale_fp16 = float_to_fp16(scale);
            std::memcpy(block, &scale_fp16, 2);
            for (int q = 0; q < 32; q++) {
                int8_t val = static_cast<int8_t>((q + v * 7) % 21 - 10);
                block[2 + q] = static_cast<uint8_t>(val);
            }
        }
    }
    indices[0] = 1;
    indices[1] = 3;
    // CPU reference: dequantize
    for (int r = 0; r < n_rows; r++) {
        uint32_t idx = indices[r];
        const uint8_t* row = table + idx * row_bytes;
        for (int b = 0; b < blocks_per_row; b++) {
            const uint8_t* block = row + b * block_bytes;
            uint16_t scale_bits;
            std::memcpy(&scale_bits, block, 2);
            __fp16 scale_fp16;
            std::memcpy(&scale_fp16, &scale_bits, 2);
            float scale = (float)scale_fp16;
            for (int q = 0; q < 32; q++) {
                int8_t qval = static_cast<int8_t>(block[2 + q]);
                ref[r * dim + b * 32 + q] = (float)qval * scale;
            }
        }
    }
    std::memset(out, 0, n_rows * dim * 4);
    ane::dispatch(ane::Op::get_rows_q8_0,
        uint32_t(n_rows), uint32_t(dim),
        reinterpret_cast<uintptr_t>(table),
        reinterpret_cast<uintptr_t>(indices),
        reinterpret_cast<uintptr_t>(out));
    check("get_rows_q8_0 2 rows dim=32", out, ref, n_rows * dim, 1e-4f);
    // Test with dim=64 (2 blocks per row)
    const int dim2 = 64;
    const int bpr2 = dim2 / 32;
    const int rb2 = bpr2 * block_bytes;
    auto* table2   = static_cast<uint8_t*>(alloc(vocab * rb2));
    auto* out2     = static_cast<float*>(alloc(n_rows * dim2 * 4));
    auto* ref2     = static_cast<float*>(alloc(n_rows * dim2 * 4));
    for (int v = 0; v < vocab; v++) {
        uint8_t* row = table2 + v * rb2;
        for (int b = 0; b < bpr2; b++) {
            uint8_t* block = row + b * block_bytes;
            float scale = 0.05f * (v + b + 1);
            uint16_t scale_fp16 = float_to_fp16(scale);
            std::memcpy(block, &scale_fp16, 2);
            for (int q = 0; q < 32; q++) {
                int8_t val = static_cast<int8_t>((q + v * 3 + b * 5) % 19 - 9);
                block[2 + q] = static_cast<uint8_t>(val);
            }
        }
    }
    for (int r = 0; r < n_rows; r++) {
        uint32_t idx = indices[r];
        const uint8_t* row = table2 + idx * rb2;
        for (int b = 0; b < bpr2; b++) {
            const uint8_t* block = row + b * block_bytes;
            uint16_t scale_bits;
            std::memcpy(&scale_bits, block, 2);
            __fp16 scale_fp16;
            std::memcpy(&scale_fp16, &scale_bits, 2);
            float scale = (float)scale_fp16;
            for (int q = 0; q < 32; q++) {
                int8_t qval = static_cast<int8_t>(block[2 + q]);
                ref2[r * dim2 + b * 32 + q] = (float)qval * scale;
            }
        }
    }
    std::memset(out2, 0, n_rows * dim2 * 4);
    ane::dispatch(ane::Op::get_rows_q8_0,
        uint32_t(n_rows), uint32_t(dim2),
        reinterpret_cast<uintptr_t>(table2),
        reinterpret_cast<uintptr_t>(indices),
        reinterpret_cast<uintptr_t>(out2));
    check("get_rows_q8_0 2 rows dim=64", out2, ref2, n_rows * dim2, 1e-4f);
    std::free(table); std::free(indices); std::free(out); std::free(ref);
    std::free(table2); std::free(out2); std::free(ref2);
}
/** --------------------------------------------------------------------------------------------------------- get_rows_q4_0 Tests
 */
static void test_get_rows_q4_0() {
    printf("\n-- get_rows_q4_0 --\n");
    const int vocab = 4, dim = 32, n_rows = 2;
    const int blocks_per_row = dim / 32;        ///< 1
    const int block_bytes = 18;                 ///< fp16 scale (2) + 16 packed bytes
    const int row_bytes = blocks_per_row * block_bytes;
    auto* table   = static_cast<uint8_t*>(alloc(vocab * row_bytes));
    auto* indices = static_cast<uint32_t*>(alloc(n_rows * 4));
    auto* out     = static_cast<float*>(alloc(n_rows * dim * 4));
    auto* ref     = static_cast<float*>(alloc(n_rows * dim * 4));
    // Build Q4_0 table: each block is {fp16_scale, 16 packed bytes}
    // Low nibble = element i (0..15), high nibble = element i+16 (16..31)
    // Dequant: (nibble - 8) * scale
    for (int v = 0; v < vocab; v++) {
        uint8_t* row = table + v * row_bytes;
        for (int b = 0; b < blocks_per_row; b++) {
            uint8_t* block = row + b * block_bytes;
            float scale = 0.2f * (v + 1);
            uint16_t scale_fp16 = float_to_fp16(scale);
            std::memcpy(block, &scale_fp16, 2);
            for (int i = 0; i < 16; i++) {
                uint8_t lo_nib = static_cast<uint8_t>((i + v) % 16);
                uint8_t hi_nib = static_cast<uint8_t>((i + v + 3) % 16);
                block[2 + i] = (hi_nib << 4) | lo_nib;
            }
        }
    }
    indices[0] = 0;
    indices[1] = 2;
    // CPU reference: dequantize Q4_0
    for (int r = 0; r < n_rows; r++) {
        uint32_t idx = indices[r];
        const uint8_t* row = table + idx * row_bytes;
        for (int b = 0; b < blocks_per_row; b++) {
            const uint8_t* block = row + b * block_bytes;
            uint16_t scale_bits;
            std::memcpy(&scale_bits, block, 2);
            __fp16 scale_fp16;
            std::memcpy(&scale_fp16, &scale_bits, 2);
            float scale = (float)scale_fp16;
            for (int i = 0; i < 16; i++) {
                uint8_t packed = block[2 + i];
                int lo = (packed & 0x0F) - 8;
                int hi = ((packed >> 4) & 0x0F) - 8;
                ref[r * dim + b * 32 + i]      = (float)lo * scale;
                ref[r * dim + b * 32 + i + 16] = (float)hi * scale;
            }
        }
    }
    std::memset(out, 0, n_rows * dim * 4);
    ane::dispatch(ane::Op::get_rows_q4_0,
        uint32_t(n_rows), uint32_t(dim),
        reinterpret_cast<uintptr_t>(table),
        reinterpret_cast<uintptr_t>(indices),
        reinterpret_cast<uintptr_t>(out));
    check("get_rows_q4_0 2 rows dim=32", out, ref, n_rows * dim, 1e-4f);
    // Test with dim=64 (2 blocks per row)
    const int dim2 = 64;
    const int bpr2 = dim2 / 32;
    const int rb2 = bpr2 * block_bytes;
    auto* table2   = static_cast<uint8_t*>(alloc(vocab * rb2));
    auto* out2     = static_cast<float*>(alloc(n_rows * dim2 * 4));
    auto* ref2     = static_cast<float*>(alloc(n_rows * dim2 * 4));
    for (int v = 0; v < vocab; v++) {
        uint8_t* row = table2 + v * rb2;
        for (int b = 0; b < bpr2; b++) {
            uint8_t* block = row + b * block_bytes;
            float scale = 0.15f * (v + b + 1);
            uint16_t scale_fp16 = float_to_fp16(scale);
            std::memcpy(block, &scale_fp16, 2);
            for (int i = 0; i < 16; i++) {
                uint8_t lo_nib = static_cast<uint8_t>((i + v + b * 2) % 16);
                uint8_t hi_nib = static_cast<uint8_t>((i + v + b * 2 + 5) % 16);
                block[2 + i] = (hi_nib << 4) | lo_nib;
            }
        }
    }
    for (int r = 0; r < n_rows; r++) {
        uint32_t idx = indices[r];
        const uint8_t* row = table2 + idx * rb2;
        for (int b = 0; b < bpr2; b++) {
            const uint8_t* block = row + b * block_bytes;
            uint16_t scale_bits;
            std::memcpy(&scale_bits, block, 2);
            __fp16 scale_fp16;
            std::memcpy(&scale_fp16, &scale_bits, 2);
            float scale = (float)scale_fp16;
            for (int i = 0; i < 16; i++) {
                uint8_t packed = block[2 + i];
                int lo = (packed & 0x0F) - 8;
                int hi = ((packed >> 4) & 0x0F) - 8;
                ref2[r * dim2 + b * 32 + i]      = (float)lo * scale;
                ref2[r * dim2 + b * 32 + i + 16] = (float)hi * scale;
            }
        }
    }
    std::memset(out2, 0, n_rows * dim2 * 4);
    ane::dispatch(ane::Op::get_rows_q4_0,
        uint32_t(n_rows), uint32_t(dim2),
        reinterpret_cast<uintptr_t>(table2),
        reinterpret_cast<uintptr_t>(indices),
        reinterpret_cast<uintptr_t>(out2));
    check("get_rows_q4_0 2 rows dim=64", out2, ref2, n_rows * dim2, 1e-4f);
    std::free(table); std::free(indices); std::free(out); std::free(ref);
    std::free(table2); std::free(out2); std::free(ref2);
}
/** --------------------------------------------------------------------------------------------------------- flash_attention_fp32 Tests
 */
static void ref_flash_attention(const float* Q, const float* K, const float* V,
                                float* out, int N, int d, bool causal) {
    auto* scores = static_cast<float*>(alloc(N * N * 4));
    auto* probs  = static_cast<float*>(alloc(N * N * 4));
    double rsqrt_d = 1.0 / std::sqrt((double)d);
    // scores = Q @ K^T / sqrt(d)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < d; k++) sum += (double)Q[i * d + k] * (double)K[j * d + k];
            scores[i * N + j] = (float)(sum * rsqrt_d);
        }
    // Causal mask: upper triangle -> -inf
    if (causal) {
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++)
                scores[i * N + j] = -INFINITY;
    }
    // Softmax per row
    for (int i = 0; i < N; i++) {
        double max_val = scores[i * N];
        for (int j = 1; j < N; j++) if (scores[i * N + j] > max_val) max_val = scores[i * N + j];
        double sum_exp = 0;
        for (int j = 0; j < N; j++) {
            probs[i * N + j] = (float)std::exp((double)scores[i * N + j] - max_val);
            sum_exp += probs[i * N + j];
        }
        for (int j = 0; j < N; j++) probs[i * N + j] = (float)((double)probs[i * N + j] / sum_exp);
    }
    // out = probs @ V
    for (int i = 0; i < N; i++)
        for (int j = 0; j < d; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++) sum += (double)probs[i * N + k] * (double)V[k * d + j];
            out[i * d + j] = (float)sum;
        }
    std::free(scores);
    std::free(probs);
}
static void test_flash_attention_fp32() {
    printf("\n-- flash_attention_fp32 --\n");
    const int N = 16, d = 16;
    auto* Q   = static_cast<float*>(alloc(N * d * 4));
    auto* K   = static_cast<float*>(alloc(N * d * 4));
    auto* V   = static_cast<float*>(alloc(N * d * 4));
    auto* out = static_cast<float*>(alloc(N * d * 4));
    auto* ref = static_cast<float*>(alloc(N * d * 4));
    // Initialize with small values to keep softmax numerically stable
    for (int i = 0; i < N * d; i++) {
        Q[i] = 0.1f * ((i % 29) - 14);
        K[i] = 0.1f * ((i % 31) - 15);
        V[i] = 0.1f * ((i % 37) - 18);
    }
    // Test non-causal (flags=0)
    ref_flash_attention(Q, K, V, ref, N, d, false);
    std::memset(out, 0, N * d * 4);
    ane::dispatch(ane::Op::flash_attention_fp32,
        uint32_t(N), uint32_t(d), uint8_t(0),
        reinterpret_cast<uintptr_t>(Q),
        reinterpret_cast<uintptr_t>(K),
        reinterpret_cast<uintptr_t>(V),
        reinterpret_cast<uintptr_t>(out));
    check("flash_attention non-causal 16x16", out, ref, N * d, 1e-2f);
    // Test causal (flags=1)
    ref_flash_attention(Q, K, V, ref, N, d, true);
    std::memset(out, 0, N * d * 4);
    ane::dispatch(ane::Op::flash_attention_fp32,
        uint32_t(N), uint32_t(d), uint8_t(1),
        reinterpret_cast<uintptr_t>(Q),
        reinterpret_cast<uintptr_t>(K),
        reinterpret_cast<uintptr_t>(V),
        reinterpret_cast<uintptr_t>(out));
    check("flash_attention causal 16x16", out, ref, N * d, 1e-2f);
    // Verify causal: first row should only attend to first key
    // (softmax of single non-masked element = 1.0, so out[0] = V[0])
    tests_total++;
    bool first_row_ok = true;
    for (int j = 0; j < d; j++) {
        float err = std::fabs(out[j] - V[j]);
        if (err > 5e-2f) { first_row_ok = false; break; }
    }
    if (first_row_ok) {
        printf("  [PASS] flash_attention causal row0 == V[0]\n");
        tests_passed++;
    } else {
        printf("  [FAIL] flash_attention causal row0 != V[0]\n");
    }
    std::free(Q); std::free(K); std::free(V); std::free(out); std::free(ref);
}
/** --------------------------------------------------------------------------------------------------------- Main
 */
int main() {
    printf("=== ANE Remaining Ops Test Suite ===\n");
    test_cblas_bfgemm();
    test_cblas_igemm();
    test_get_rows_fp32();
    test_get_rows_q8_0();
    test_get_rows_q4_0();
    test_flash_attention_fp32();
    printf("\n===============================\n");
    printf("  Results: %d / %d passed\n", tests_passed, tests_total);
    printf("===============================\n");
    return (tests_passed == tests_total) ? 0 : 1;
}
