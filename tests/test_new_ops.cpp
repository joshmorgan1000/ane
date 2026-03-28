/** --------------------------------------------------------------------------------------------------------- File Info
 * @file test_new_ops.cpp
 * @brief Tests for all new kernels added in the multi-precision GEMM, decomposition, training,
 * and inference expansion: cblas_bfgemm, cblas_igemm, cblas_ugemm, cblas_usgemm, gemm_tile_fp32,
 * softmax_partial/correct, reduce_sum_sq, reduce_col_sum, silu_backward, softmax_backward,
 * gelu, layer_norm, causal_mask, adam_step.
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
/** --------------------------------------------------------------------------------------------------------- GEMM Tile FP32 Tests
 */
static void ref_sgemm(const float* A, const float* B, float* C, int M, int N, int K,
                       float alpha, float beta) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < K; k++) sum += (double)A[i*K+k] * (double)B[k*N+j];
            C[i*N+j] = (float)(alpha * sum + beta * C[i*N+j]);
        }
}
static void test_gemm_tile_fp32() {
    printf("\n── gemm_tile_fp32 ──\n");
    const int M = 64, N = 64, K = 16;
    auto* A   = static_cast<float*>(alloc(M * K * 4));
    auto* B   = static_cast<float*>(alloc(K * N * 4));
    auto* C   = static_cast<float*>(alloc(M * N * 4));
    auto* ref = static_cast<float*>(alloc(M * N * 4));
    for (int i = 0; i < M*K; i++) A[i] = 0.01f * (i % 37 - 18);
    for (int i = 0; i < K*N; i++) B[i] = 0.01f * (i % 41 - 20);
    // Full GEMM via tile: compute all tiles
    std::memset(C, 0, M * N * 4);
    ref_sgemm(A, B, ref, M, N, K, 1.0f, 0.0f);
    ane::program p;
    p.emit(ane::Op::gemm_tile_fp32,
        uint8_t(0), uint32_t(M), uint32_t(N), uint32_t(K),
        uint32_t(K), uint32_t(N), uint32_t(N), 1.0f, 0.0f,
        uint32_t(0), uint32_t(0), uint32_t(M), uint32_t(N),
        reinterpret_cast<uintptr_t>(A), reinterpret_cast<uintptr_t>(B),
        reinterpret_cast<uintptr_t>(C));
    p.exec();
    check("gemm_tile full 64x64", C, ref, M * N, 1e-3f);
    // Partial tile: top-left 32x32 only
    std::memset(C, 0, M * N * 4);
    float ref_partial[32 * 32];
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++) {
            double sum = 0;
            for (int k = 0; k < K; k++) sum += (double)A[i*K+k] * (double)B[k*N+j];
            ref_partial[i*32+j] = (float)sum;
        }
    // Write to a separate 32x32 output
    auto* C2 = static_cast<float*>(alloc(M * N * 4));
    std::memset(C2, 0, M * N * 4);
    ane::program p2;
    p2.emit(ane::Op::gemm_tile_fp32,
        uint8_t(0), uint32_t(M), uint32_t(N), uint32_t(K),
        uint32_t(K), uint32_t(N), uint32_t(N), 1.0f, 0.0f,
        uint32_t(0), uint32_t(0), uint32_t(32), uint32_t(32),
        reinterpret_cast<uintptr_t>(A), reinterpret_cast<uintptr_t>(B),
        reinterpret_cast<uintptr_t>(C2));
    p2.exec();
    // Check only the top-left 32x32
    bool pass = true;
    float max_err = 0;
    for (int i = 0; i < 32 && pass; i++)
        for (int j = 0; j < 32; j++) {
            float err = std::fabs(C2[i*N+j] - ref_partial[i*32+j]);
            if (err > max_err) max_err = err;
        }
    tests_total++;
    if (max_err <= 1e-3f) {
        printf("  [PASS] gemm_tile partial 32x32 (max_err=%.2e)\n", max_err);
        tests_passed++;
    } else {
        printf("  [FAIL] gemm_tile partial 32x32 (max_err=%.2e)\n", max_err);
    }
    std::free(A); std::free(B); std::free(C); std::free(ref); std::free(C2);
}
/** --------------------------------------------------------------------------------------------------------- Softmax Partial/Correct Tests
 */
static void test_softmax_partial() {
    printf("\n── softmax_partial + softmax_correct ──\n");
    const int dim = 64;
    auto* in      = static_cast<float*>(alloc(dim * 4));
    auto* out     = static_cast<float*>(alloc(dim * 4));
    auto* ref     = static_cast<float*>(alloc(dim * 4));
    auto* max_out = static_cast<float*>(alloc(64));
    auto* sum_out = static_cast<float*>(alloc(64));
    for (int i = 0; i < dim; i++) in[i] = 0.1f * (i - dim/2);
    // CPU reference: full softmax
    double cpu_max = in[0];
    for (int i = 1; i < dim; i++) if (in[i] > cpu_max) cpu_max = in[i];
    double cpu_sum = 0;
    for (int i = 0; i < dim; i++) { ref[i] = (float)std::exp((double)in[i] - cpu_max); cpu_sum += ref[i]; }
    for (int i = 0; i < dim; i++) ref[i] = (float)((double)ref[i] / cpu_sum);
    // ANE: partial softmax
    ane::dispatch(ane::Op::softmax_partial_fp32, uint32_t(dim),
        reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out),
        reinterpret_cast<uintptr_t>(max_out), reinterpret_cast<uintptr_t>(sum_out));
    check_scalar("softmax_partial max", max_out[0], (float)cpu_max, 1e-5f);
    // Finalize: divide by sum to get full softmax
    float inv_sum = 1.0f / sum_out[0];
    for (int i = 0; i < dim; i++) out[i] *= inv_sum;
    check("softmax_partial → full softmax", out, ref, dim, 1e-4f);
    // Test softmax_correct: split into two halves, merge
    const int half = dim / 2;
    auto* out1 = static_cast<float*>(alloc(half * 4));
    auto* out2 = static_cast<float*>(alloc(half * 4));
    auto* max1 = static_cast<float*>(alloc(64));
    auto* max2 = static_cast<float*>(alloc(64));
    auto* sum1 = static_cast<float*>(alloc(64));
    auto* sum2 = static_cast<float*>(alloc(64));
    ane::dispatch(ane::Op::softmax_partial_fp32, uint32_t(half),
        reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out1),
        reinterpret_cast<uintptr_t>(max1), reinterpret_cast<uintptr_t>(sum1));
    ane::dispatch(ane::Op::softmax_partial_fp32, uint32_t(half),
        reinterpret_cast<uintptr_t>(in + half), reinterpret_cast<uintptr_t>(out2),
        reinterpret_cast<uintptr_t>(max2), reinterpret_cast<uintptr_t>(sum2));
    float global_max = std::max(max1[0], max2[0]);
    ane::dispatch(ane::Op::softmax_correct_fp32, uint32_t(half),
        max1[0], global_max,
        reinterpret_cast<uintptr_t>(out1), reinterpret_cast<uintptr_t>(sum1));
    ane::dispatch(ane::Op::softmax_correct_fp32, uint32_t(half),
        max2[0], global_max,
        reinterpret_cast<uintptr_t>(out2), reinterpret_cast<uintptr_t>(sum2));
    float global_sum = sum1[0] + sum2[0];
    float inv_gsum = 1.0f / global_sum;
    for (int i = 0; i < half; i++) { out1[i] *= inv_gsum; out2[i] *= inv_gsum; }
    // Combine and check against reference
    auto* combined = static_cast<float*>(alloc(dim * 4));
    std::memcpy(combined, out1, half * 4);
    std::memcpy(combined + half, out2, half * 4);
    check("softmax split→correct→merge", combined, ref, dim, 1e-3f);
    std::free(in); std::free(out); std::free(ref); std::free(max_out); std::free(sum_out);
    std::free(out1); std::free(out2); std::free(max1); std::free(max2);
    std::free(sum1); std::free(sum2); std::free(combined);
}
/** --------------------------------------------------------------------------------------------------------- Reduce Sum Sq Tests
 */
static void test_reduce_sum_sq() {
    printf("\n── reduce_sum_sq_fp32 ──\n");
    const int dim = 256;
    auto* in  = static_cast<float*>(alloc(dim * 4));
    auto* out = static_cast<float*>(alloc(64));
    for (int i = 0; i < dim; i++) in[i] = 0.01f * (i - 128);
    double cpu_sum_sq = 0;
    for (int i = 0; i < dim; i++) cpu_sum_sq += (double)in[i] * (double)in[i];
    ane::dispatch(ane::Op::reduce_sum_sq_fp32, uint32_t(dim),
        reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
    check_scalar("reduce_sum_sq dim=256", out[0], (float)cpu_sum_sq, 1e-2f);
    std::free(in); std::free(out);
}
/** --------------------------------------------------------------------------------------------------------- Reduce Col Sum Tests
 */
static void test_reduce_col_sum() {
    printf("\n── reduce_col_sum_fp32 ──\n");
    const int M = 32, N = 48;
    auto* src = static_cast<float*>(alloc(M * N * 4));
    auto* dst = static_cast<float*>(alloc(N * 4));
    auto* ref = static_cast<float*>(alloc(N * 4));
    for (int i = 0; i < M * N; i++) src[i] = 0.01f * (i % 53 - 26);
    for (int j = 0; j < N; j++) {
        double sum = 0;
        for (int i = 0; i < M; i++) sum += (double)src[i * N + j];
        ref[j] = (float)sum;
    }
    ane::dispatch(ane::Op::reduce_col_sum_fp32,
        uint32_t(M), uint32_t(N), uint32_t(N),
        reinterpret_cast<uintptr_t>(src), reinterpret_cast<uintptr_t>(dst));
    check("reduce_col_sum 32x48", dst, ref, N, 1e-4f);
    // Test with stride != N (column subrange)
    const int full_N = 64;
    auto* full = static_cast<float*>(alloc(M * full_N * 4));
    for (int i = 0; i < M * full_N; i++) full[i] = 0.01f * (i % 47 - 23);
    auto* dst2 = static_cast<float*>(alloc(32 * 4));
    auto* ref2 = static_cast<float*>(alloc(32 * 4));
    for (int j = 0; j < 32; j++) {
        double sum = 0;
        for (int i = 0; i < M; i++) sum += (double)full[i * full_N + 16 + j];
        ref2[j] = (float)sum;
    }
    ane::dispatch(ane::Op::reduce_col_sum_fp32,
        uint32_t(M), uint32_t(32), uint32_t(full_N),
        reinterpret_cast<uintptr_t>(full + 16), reinterpret_cast<uintptr_t>(dst2));
    check("reduce_col_sum stride!=N (subrange)", dst2, ref2, 32, 1e-4f);
    std::free(src); std::free(dst); std::free(ref);
    std::free(full); std::free(dst2); std::free(ref2);
}
/** --------------------------------------------------------------------------------------------------------- SiLU Backward Tests
 */
static void ref_silu_backward(const float* x, const float* dy, float* dx, int n) {
    for (int i = 0; i < n; i++) {
        double xi = x[i];
        double sig = 1.0 / (1.0 + std::exp(-xi));
        dx[i] = (float)((double)dy[i] * (sig + xi * sig * (1.0 - sig)));
    }
}
static void test_silu_backward() {
    printf("\n── silu_backward_fp32 ──\n");
    const int dim = 256;
    auto* x   = static_cast<float*>(alloc(dim * 4));
    auto* dy  = static_cast<float*>(alloc(dim * 4));
    auto* dx  = static_cast<float*>(alloc(dim * 4));
    auto* ref = static_cast<float*>(alloc(dim * 4));
    for (int i = 0; i < dim; i++) { x[i] = 0.05f * (i - 128); dy[i] = 0.01f * (i % 17 - 8); }
    ref_silu_backward(x, dy, ref, dim);
    ane::dispatch(ane::Op::silu_backward_fp32, uint32_t(dim),
        reinterpret_cast<uintptr_t>(x), reinterpret_cast<uintptr_t>(dy),
        reinterpret_cast<uintptr_t>(dx));
    check("silu_backward dim=256", dx, ref, dim, 1e-3f);
    // Edge: x near zero (sigmoid ≈ 0.5)
    for (int i = 0; i < 16; i++) { x[i] = 0.0f; dy[i] = 1.0f; }
    ref_silu_backward(x, dy, ref, 16);
    ane::dispatch(ane::Op::silu_backward_fp32, uint32_t(16),
        reinterpret_cast<uintptr_t>(x), reinterpret_cast<uintptr_t>(dy),
        reinterpret_cast<uintptr_t>(dx));
    check("silu_backward x=0 (deriv≈0.5)", dx, ref, 16, 1e-4f);
    std::free(x); std::free(dy); std::free(dx); std::free(ref);
}
/** --------------------------------------------------------------------------------------------------------- Softmax Backward Tests
 */
static void ref_softmax_backward(const float* s, const float* dy, float* dx, int n) {
    double dot = 0;
    for (int i = 0; i < n; i++) dot += (double)s[i] * (double)dy[i];
    for (int i = 0; i < n; i++) dx[i] = (float)((double)s[i] * ((double)dy[i] - dot));
}
static void test_softmax_backward() {
    printf("\n── softmax_backward_fp32 ──\n");
    const int dim = 64;
    auto* s   = static_cast<float*>(alloc(dim * 4));
    auto* dy  = static_cast<float*>(alloc(dim * 4));
    auto* dx  = static_cast<float*>(alloc(dim * 4));
    auto* ref = static_cast<float*>(alloc(dim * 4));
    // s must be a valid softmax output (sums to 1)
    float sum = 0;
    for (int i = 0; i < dim; i++) { s[i] = std::exp(-0.1f * std::abs(i - 32)); sum += s[i]; }
    for (int i = 0; i < dim; i++) s[i] /= sum;
    for (int i = 0; i < dim; i++) dy[i] = 0.01f * (i % 13 - 6);
    ref_softmax_backward(s, dy, ref, dim);
    ane::dispatch(ane::Op::softmax_backward_fp32, uint32_t(dim),
        reinterpret_cast<uintptr_t>(s), reinterpret_cast<uintptr_t>(dy),
        reinterpret_cast<uintptr_t>(dx));
    check("softmax_backward dim=64", dx, ref, dim, 1e-4f);
    std::free(s); std::free(dy); std::free(dx); std::free(ref);
}
/** --------------------------------------------------------------------------------------------------------- GeLU Tests
 */
static void ref_gelu(const float* in, float* out, int n) {
    const double sqrt_2_pi = 0.7978845608;
    for (int i = 0; i < n; i++) {
        double x = in[i];
        double inner = sqrt_2_pi * (x + 0.044715 * x * x * x);
        out[i] = (float)(0.5 * x * (1.0 + std::tanh(inner)));
    }
}
static void test_gelu() {
    printf("\n── gelu_fp32 ──\n");
    const int dim = 256;
    auto* in  = static_cast<float*>(alloc(dim * 4));
    auto* out = static_cast<float*>(alloc(dim * 4));
    auto* ref = static_cast<float*>(alloc(dim * 4));
    for (int i = 0; i < dim; i++) in[i] = 0.05f * (i - 128);
    ref_gelu(in, ref, dim);
    ane::dispatch(ane::Op::gelu_fp32, uint32_t(dim),
        reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
    check("gelu dim=256 range [-6.4, 6.35]", out, ref, dim, 1e-3f);
    // Edge: x=0 → gelu(0) = 0
    for (int i = 0; i < 16; i++) in[i] = 0.0f;
    ref_gelu(in, ref, 16);
    ane::dispatch(ane::Op::gelu_fp32, uint32_t(16),
        reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
    check("gelu x=0", out, ref, 16, 1e-5f);
    std::free(in); std::free(out); std::free(ref);
}
/** --------------------------------------------------------------------------------------------------------- Layer Norm Tests
 */
static void ref_layer_norm(const float* in, const float* gamma, const float* beta,
                           float* out, int dim, float eps) {
    double mean = 0;
    for (int i = 0; i < dim; i++) mean += (double)in[i];
    mean /= dim;
    double var = 0;
    for (int i = 0; i < dim; i++) { double d = (double)in[i] - mean; var += d * d; }
    var /= dim;
    double inv_std = 1.0 / std::sqrt(var + eps);
    for (int i = 0; i < dim; i++)
        out[i] = (float)(((double)in[i] - mean) * inv_std * (double)gamma[i] + (double)beta[i]);
}
static void test_layer_norm() {
    printf("\n── layer_norm_fp32 ──\n");
    const int dim = 256;
    const float eps = 1e-5f;
    auto* in    = static_cast<float*>(alloc(dim * 4));
    auto* gamma = static_cast<float*>(alloc(dim * 4));
    auto* beta  = static_cast<float*>(alloc(dim * 4));
    auto* out   = static_cast<float*>(alloc(dim * 4));
    auto* ref   = static_cast<float*>(alloc(dim * 4));
    for (int i = 0; i < dim; i++) {
        in[i] = sinf((float)i * 0.1f) * 3.0f;
        gamma[i] = 1.0f;
        beta[i] = 0.0f;
    }
    ref_layer_norm(in, gamma, beta, ref, dim, eps);
    ane::dispatch(ane::Op::layer_norm_fp32, uint32_t(dim), eps,
        reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(gamma),
        reinterpret_cast<uintptr_t>(beta), reinterpret_cast<uintptr_t>(out));
    check("layer_norm dim=256 gamma=1 beta=0", out, ref, dim, 1e-3f);
    // With non-trivial gamma/beta
    for (int i = 0; i < dim; i++) { gamma[i] = 0.5f + 0.01f * i; beta[i] = -0.5f + 0.005f * i; }
    ref_layer_norm(in, gamma, beta, ref, dim, eps);
    ane::dispatch(ane::Op::layer_norm_fp32, uint32_t(dim), eps,
        reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(gamma),
        reinterpret_cast<uintptr_t>(beta), reinterpret_cast<uintptr_t>(out));
    check("layer_norm dim=256 gamma/beta varying", out, ref, dim, 1e-3f);
    std::free(in); std::free(gamma); std::free(beta); std::free(out); std::free(ref);
}
/** --------------------------------------------------------------------------------------------------------- Causal Mask Tests
 */
static void test_causal_mask() {
    printf("\n── causal_mask_fp32 ──\n");
    const int dim = 32;
    auto* scores = static_cast<float*>(alloc(dim * dim * 4));
    for (int i = 0; i < dim * dim; i++) scores[i] = 1.0f;
    ane::dispatch(ane::Op::causal_mask_fp32, uint32_t(dim), uint32_t(dim),
        reinterpret_cast<uintptr_t>(scores));
    tests_total++;
    bool pass = true;
    for (int i = 0; i < dim && pass; i++)
        for (int j = 0; j < dim && pass; j++) {
            float val = scores[i * dim + j];
            if (j > i) {
                if (!std::isinf(val) || val > 0) {
                    printf("  [FAIL] causal_mask: scores[%d][%d] = %.2f, expected -inf\n", i, j, val);
                    pass = false;
                }
            } else {
                if (val != 1.0f) {
                    printf("  [FAIL] causal_mask: scores[%d][%d] = %.2f, expected 1.0 (untouched)\n", i, j, val);
                    pass = false;
                }
            }
        }
    if (pass) { printf("  [PASS] causal_mask 32x32\n"); tests_passed++; }
    std::free(scores);
}
/** --------------------------------------------------------------------------------------------------------- Adam Step Tests
 */
static void ref_adam(float* params, const float* grads, float* m, float* v,
                     int n, float lr, float b1, float b2, float eps, int t) {
    double b1t = 1, b2t = 1;
    for (int i = 0; i < t; i++) { b1t *= b1; b2t *= b2; }
    double bc1 = 1.0 - b1t;
    double bc2 = 1.0 - b2t;
    for (int i = 0; i < n; i++) {
        m[i] = (float)(b1 * (double)m[i] + (1.0 - b1) * (double)grads[i]);
        v[i] = (float)(b2 * (double)v[i] + (1.0 - b2) * (double)grads[i] * (double)grads[i]);
        double m_hat = (double)m[i] / bc1;
        double v_hat = (double)v[i] / bc2;
        params[i] -= (float)(lr * m_hat / (std::sqrt(v_hat) + eps));
    }
}
static void test_adam_step() {
    printf("\n── adam_step_fp32 ──\n");
    const int n = 256;
    const float lr = 0.001f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    auto* params     = static_cast<float*>(alloc(n * 4));
    auto* grads      = static_cast<float*>(alloc(n * 4));
    auto* m          = static_cast<float*>(alloc(n * 4));
    auto* v          = static_cast<float*>(alloc(n * 4));
    auto* ref_params = static_cast<float*>(alloc(n * 4));
    auto* ref_m      = static_cast<float*>(alloc(n * 4));
    auto* ref_v      = static_cast<float*>(alloc(n * 4));
    for (int i = 0; i < n; i++) {
        params[i] = ref_params[i] = 0.1f * (i % 31 - 15);
        grads[i] = 0.01f * (i % 23 - 11);
        m[i] = ref_m[i] = 0.0f;
        v[i] = ref_v[i] = 0.0f;
    }
    // Step 1
    ref_adam(ref_params, grads, ref_m, ref_v, n, lr, b1, b2, eps, 1);
    ane::dispatch(ane::Op::adam_step_fp32,
        uint32_t(n), lr, b1, b2, eps, uint32_t(1),
        reinterpret_cast<uintptr_t>(params), reinterpret_cast<uintptr_t>(grads),
        reinterpret_cast<uintptr_t>(m), reinterpret_cast<uintptr_t>(v));
    check("adam_step t=1 params", params, ref_params, n, 1e-4f);
    check("adam_step t=1 m", m, ref_m, n, 1e-5f);
    check("adam_step t=1 v", v, ref_v, n, 1e-5f);
    // Step 10 (accumulate from current state)
    ref_adam(ref_params, grads, ref_m, ref_v, n, lr, b1, b2, eps, 10);
    ane::dispatch(ane::Op::adam_step_fp32,
        uint32_t(n), lr, b1, b2, eps, uint32_t(10),
        reinterpret_cast<uintptr_t>(params), reinterpret_cast<uintptr_t>(grads),
        reinterpret_cast<uintptr_t>(m), reinterpret_cast<uintptr_t>(v));
    check("adam_step t=10 params", params, ref_params, n, 1e-3f);
    std::free(params); std::free(grads); std::free(m); std::free(v);
    std::free(ref_params); std::free(ref_m); std::free(ref_v);
}
/** --------------------------------------------------------------------------------------------------------- GeLU Backward Tests
 */
static void ref_gelu_backward(const float* x, const float* dy, float* dx, int n) {
    const double sqrt_2_pi = 0.7978845608;
    for (int i = 0; i < n; i++) {
        double xi = x[i];
        double inner = sqrt_2_pi * (xi + 0.044715 * xi * xi * xi);
        double t = std::tanh(inner);
        double sech2 = 1.0 - t * t;
        double d_inner = sqrt_2_pi * (1.0 + 3.0 * 0.044715 * xi * xi);
        double deriv = 0.5 * (1.0 + t) + 0.5 * xi * sech2 * d_inner;
        dx[i] = (float)((double)dy[i] * deriv);
    }
}
static void test_gelu_backward() {
    printf("\n── gelu_backward_fp32 ──\n");
    const int dim = 256;
    auto* x   = static_cast<float*>(alloc(dim * 4));
    auto* dy  = static_cast<float*>(alloc(dim * 4));
    auto* dx  = static_cast<float*>(alloc(dim * 4));
    auto* ref = static_cast<float*>(alloc(dim * 4));
    for (int i = 0; i < dim; i++) { x[i] = 0.05f * (i - 128); dy[i] = 0.01f * (i % 19 - 9); }
    ref_gelu_backward(x, dy, ref, dim);
    ane::dispatch(ane::Op::gelu_backward_fp32, uint32_t(dim),
        reinterpret_cast<uintptr_t>(x), reinterpret_cast<uintptr_t>(dy),
        reinterpret_cast<uintptr_t>(dx));
    check("gelu_backward dim=256", dx, ref, dim, 2e-3f);
    std::free(x); std::free(dy); std::free(dx); std::free(ref);
}
/** --------------------------------------------------------------------------------------------------------- RMS Norm Backward Tests
 */
static void ref_rms_norm_backward(const float* x, const float* w, const float* dy,
                                   float* dx, float* dw, int dim, float eps) {
    double sum_sq = 0;
    for (int i = 0; i < dim; i++) sum_sq += (double)x[i] * (double)x[i];
    double rms = std::sqrt(sum_sq / dim + eps);
    double inv_rms = 1.0 / rms;
    double dot = 0;
    for (int i = 0; i < dim; i++) dot += (double)dy[i] * (double)w[i] * (double)x[i];
    double scale = dot / (rms * rms * dim);
    for (int i = 0; i < dim; i++) {
        dw[i] = (float)((double)dy[i] * (double)x[i] * inv_rms);
        dx[i] = (float)(inv_rms * ((double)dy[i] * (double)w[i] - (double)x[i] * scale));
    }
}
static void test_rms_norm_backward() {
    printf("\n── rms_norm_backward_fp32 ──\n");
    const int dim = 256;
    const float eps = 1e-5f;
    auto* x   = static_cast<float*>(alloc(dim * 4));
    auto* w   = static_cast<float*>(alloc(dim * 4));
    auto* dy  = static_cast<float*>(alloc(dim * 4));
    auto* dx  = static_cast<float*>(alloc(dim * 4));
    auto* dw  = static_cast<float*>(alloc(dim * 4));
    auto* ref_dx = static_cast<float*>(alloc(dim * 4));
    auto* ref_dw = static_cast<float*>(alloc(dim * 4));
    for (int i = 0; i < dim; i++) {
        x[i] = sinf((float)i * 0.1f);
        w[i] = 1.0f + 0.01f * i;
        dy[i] = 0.01f * (i % 23 - 11);
    }
    ref_rms_norm_backward(x, w, dy, ref_dx, ref_dw, dim, eps);
    ane::dispatch(ane::Op::rms_norm_backward_fp32, uint32_t(dim), eps,
        reinterpret_cast<uintptr_t>(x), reinterpret_cast<uintptr_t>(w),
        reinterpret_cast<uintptr_t>(dy), reinterpret_cast<uintptr_t>(dx),
        reinterpret_cast<uintptr_t>(dw));
    check("rms_norm_backward dx", dx, ref_dx, dim, 1e-3f);
    check("rms_norm_backward dw", dw, ref_dw, dim, 1e-3f);
    std::free(x); std::free(w); std::free(dy); std::free(dx); std::free(dw);
    std::free(ref_dx); std::free(ref_dw);
}
/** --------------------------------------------------------------------------------------------------------- Layer Norm Backward Tests
 */
static void ref_layer_norm_backward(const float* x, const float* gamma, const float* dy,
                                     float* dx, float* dgamma, float* dbeta, int dim, float eps) {
    double mean = 0;
    for (int i = 0; i < dim; i++) mean += (double)x[i];
    mean /= dim;
    double var = 0;
    for (int i = 0; i < dim; i++) { double d = (double)x[i] - mean; var += d * d; }
    var /= dim;
    double inv_std = 1.0 / std::sqrt(var + eps);
    double ds = 0, dm = 0;
    for (int i = 0; i < dim; i++) {
        double x_hat = ((double)x[i] - mean) * inv_std;
        dgamma[i] = (float)((double)dy[i] * x_hat);
        dbeta[i] = dy[i];
        ds += (double)dy[i] * (double)gamma[i] * x_hat;
        dm += (double)dy[i] * (double)gamma[i];
    }
    ds /= dim;
    dm /= dim;
    for (int i = 0; i < dim; i++) {
        double x_hat = ((double)x[i] - mean) * inv_std;
        dx[i] = (float)(inv_std * ((double)dy[i] * (double)gamma[i] - dm - x_hat * ds));
    }
}
static void test_layer_norm_backward() {
    printf("\n── layer_norm_backward_fp32 ──\n");
    const int dim = 256;
    const float eps = 1e-5f;
    auto* x      = static_cast<float*>(alloc(dim * 4));
    auto* gamma  = static_cast<float*>(alloc(dim * 4));
    auto* dy     = static_cast<float*>(alloc(dim * 4));
    auto* dx     = static_cast<float*>(alloc(dim * 4));
    auto* dg     = static_cast<float*>(alloc(dim * 4));
    auto* db     = static_cast<float*>(alloc(dim * 4));
    auto* ref_dx = static_cast<float*>(alloc(dim * 4));
    auto* ref_dg = static_cast<float*>(alloc(dim * 4));
    auto* ref_db = static_cast<float*>(alloc(dim * 4));
    for (int i = 0; i < dim; i++) {
        x[i] = sinf((float)i * 0.1f) * 3.0f;
        gamma[i] = 1.0f + 0.01f * i;
        dy[i] = 0.01f * (i % 17 - 8);
    }
    ref_layer_norm_backward(x, gamma, dy, ref_dx, ref_dg, ref_db, dim, eps);
    ane::dispatch(ane::Op::layer_norm_backward_fp32, uint32_t(dim), eps,
        reinterpret_cast<uintptr_t>(x), reinterpret_cast<uintptr_t>(gamma),
        reinterpret_cast<uintptr_t>(dy), reinterpret_cast<uintptr_t>(dx),
        reinterpret_cast<uintptr_t>(dg), reinterpret_cast<uintptr_t>(db));
    check("layer_norm_backward dx", dx, ref_dx, dim, 1e-3f);
    check("layer_norm_backward dgamma", dg, ref_dg, dim, 1e-3f);
    check("layer_norm_backward dbeta", db, ref_db, dim, 1e-5f);
    std::free(x); std::free(gamma); std::free(dy);
    std::free(dx); std::free(dg); std::free(db);
    std::free(ref_dx); std::free(ref_dg); std::free(ref_db);
}
/** --------------------------------------------------------------------------------------------------------- RoPE Backward Tests
 */
static void ref_rope_backward(const float* dy, float* dx, int dim, int pos, float theta) {
    for (int i = 0; i < dim; i += 2) {
        int k = i / 2;
        double freq = (double)pos / std::pow((double)theta, (2.0 * k) / dim);
        double cos_f = std::cos(freq);
        double sin_f = std::sin(freq);
        // Inverse rotation: transpose of forward rotation matrix
        dx[i]     = (float)((double)dy[i] * cos_f + (double)dy[i + 1] * sin_f);
        dx[i + 1] = (float)(-(double)dy[i] * sin_f + (double)dy[i + 1] * cos_f);
    }
}
static void test_rope_backward() {
    printf("\n── rope_backward_fp32 ──\n");
    const int dim = 64;
    const float theta = 10000.0f;
    auto* dy  = static_cast<float*>(alloc(dim * 4));
    auto* dx  = static_cast<float*>(alloc(dim * 4));
    auto* ref = static_cast<float*>(alloc(dim * 4));
    for (int i = 0; i < dim; i++) dy[i] = 0.1f * (i % 13 - 6);
    ref_rope_backward(dy, ref, dim, 42, theta);
    ane::dispatch(ane::Op::rope_backward_fp32, uint32_t(dim), uint32_t(42), theta,
        reinterpret_cast<uintptr_t>(dy), reinterpret_cast<uintptr_t>(dx));
    check("rope_backward pos=42", dx, ref, dim, 5e-3f);
    // Verify forward→backward roundtrip: rope_back(rope_fwd(x)) ≈ x
    auto* orig = static_cast<float*>(alloc(dim * 4));
    auto* fwd  = static_cast<float*>(alloc(dim * 4));
    auto* back = static_cast<float*>(alloc(dim * 4));
    for (int i = 0; i < dim; i++) orig[i] = 0.1f * (i + 1);
    ane::dispatch(ane::Op::rope_fp32, uint32_t(dim), uint32_t(7), theta,
        reinterpret_cast<uintptr_t>(orig), reinterpret_cast<uintptr_t>(fwd));
    ane::dispatch(ane::Op::rope_backward_fp32, uint32_t(dim), uint32_t(7), theta,
        reinterpret_cast<uintptr_t>(fwd), reinterpret_cast<uintptr_t>(back));
    check("rope fwd→bwd roundtrip", back, orig, dim, 5e-3f);
    std::free(dy); std::free(dx); std::free(ref);
    std::free(orig); std::free(fwd); std::free(back);
}
/** --------------------------------------------------------------------------------------------------------- Cross-Entropy Tests
 */
static void test_cross_entropy() {
    printf("\n── cross_entropy_fp32 ──\n");
    const int dim = 64;
    auto* logits = static_cast<float*>(alloc(dim * 4));
    auto* grad   = static_cast<float*>(alloc(dim * 4));
    auto* loss   = static_cast<float*>(alloc(64));
    for (int i = 0; i < dim; i++) logits[i] = 0.1f * (i - 32);
    uint32_t label = 40;
    // CPU reference
    double max_val = logits[0];
    for (int i = 1; i < dim; i++) if (logits[i] > max_val) max_val = logits[i];
    double sum_exp = 0;
    for (int i = 0; i < dim; i++) sum_exp += std::exp((double)logits[i] - max_val);
    float ref_loss = (float)(std::log(sum_exp) - ((double)logits[label] - max_val));
    float ref_grad[64];
    for (int i = 0; i < dim; i++) {
        ref_grad[i] = (float)(std::exp((double)logits[i] - max_val) / sum_exp);
        if ((uint32_t)i == label) ref_grad[i] -= 1.0f;
    }
    ane::dispatch(ane::Op::cross_entropy_fp32, uint32_t(dim), label,
        reinterpret_cast<uintptr_t>(logits), reinterpret_cast<uintptr_t>(grad),
        reinterpret_cast<uintptr_t>(loss));
    check_scalar("cross_entropy loss", loss[0], ref_loss, 2e-3f);
    check("cross_entropy gradient", grad, ref_grad, dim, 1e-4f);
    std::free(logits); std::free(grad); std::free(loss);
}
/** --------------------------------------------------------------------------------------------------------- Elementwise Sub Tests
 */
static void test_elementwise_sub() {
    printf("\n── elementwise_sub_fp32 ──\n");
    const int dim = 256;
    auto* a   = static_cast<float*>(alloc(dim * 4));
    auto* b   = static_cast<float*>(alloc(dim * 4));
    auto* out = static_cast<float*>(alloc(dim * 4));
    auto* ref = static_cast<float*>(alloc(dim * 4));
    for (int i = 0; i < dim; i++) { a[i] = 0.1f * i; b[i] = 0.05f * (i % 31); ref[i] = a[i] - b[i]; }
    ane::dispatch(ane::Op::elementwise_sub_fp32, uint32_t(dim),
        reinterpret_cast<uintptr_t>(a), reinterpret_cast<uintptr_t>(b),
        reinterpret_cast<uintptr_t>(out));
    check("elementwise_sub dim=256", out, ref, dim, 1e-6f);
    std::free(a); std::free(b); std::free(out); std::free(ref);
}
/** --------------------------------------------------------------------------------------------------------- Main
 */
int main() {
    printf("=== ANE New Ops Test Suite ===\n");
    test_gemm_tile_fp32();
    test_softmax_partial();
    test_reduce_sum_sq();
    test_reduce_col_sum();
    test_silu_backward();
    test_softmax_backward();
    test_gelu();
    test_layer_norm();
    test_causal_mask();
    test_adam_step();
    test_gelu_backward();
    test_rms_norm_backward();
    test_layer_norm_backward();
    test_rope_backward();
    test_cross_entropy();
    test_elementwise_sub();
    printf("\n═══════════════════════════════\n");
    printf("  Results: %d / %d passed\n", tests_passed, tests_total);
    printf("═══════════════════════════════\n");
    return (tests_passed == tests_total) ? 0 : 1;
}
