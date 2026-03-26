/** --------------------------------------------------------------------------------------------------------- File Info
 * @file test_transformer.cpp
 * @brief Comprehensive tests for the transformer block kernels: RMS normalization, softmax,
 *  SiLU activation, and RoPE. Each test compares ANE output against a CPU scalar reference
 *  implementation computed in double precision.
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
#include <numeric>
#include <ane/ane.hpp>

static int tests_passed = 0;
static int tests_total = 0;
static void* alloc(size_t n) { return std::aligned_alloc(64, ((n + 63) / 64) * 64); }
/** --------------------------------------------------------------------------------------------------------- Test Helper
 * @brief Compares two float arrays element-wise. Prints PASS or FAIL with details.
 */
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
/** --------------------------------------------------------------------------------------------------------- CPU Reference: RMS Norm
 */
static void ref_rms_norm(const float* in, const float* weight, float* out, int dim, float eps) {
    double sum_sq = 0;
    for (int i = 0; i < dim; i++) sum_sq += (double)in[i] * (double)in[i];
    double rms = std::sqrt(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = (float)((double)in[i] / rms * (double)weight[i]);
}
/** --------------------------------------------------------------------------------------------------------- CPU Reference: Softmax
 */
static void ref_softmax(const float* in, float* out, int dim) {
    double max_val = in[0];
    for (int i = 1; i < dim; i++) if (in[i] > max_val) max_val = in[i];
    double sum = 0;
    for (int i = 0; i < dim; i++) { out[i] = (float)std::exp((double)in[i] - max_val); sum += out[i]; }
    for (int i = 0; i < dim; i++) out[i] = (float)((double)out[i] / sum);
}
/** --------------------------------------------------------------------------------------------------------- CPU Reference: SiLU
 */
static void ref_silu(const float* in, float* out, int dim) {
    for (int i = 0; i < dim; i++) {
        double x = in[i];
        out[i] = (float)(x / (1.0 + std::exp(-x)));
    }
}
/** --------------------------------------------------------------------------------------------------------- CPU Reference: RoPE
 */
static void ref_rope(const float* in, float* out, int dim, int pos, float theta) {
    for (int i = 0; i < dim; i += 2) {
        int k = i / 2;
        double freq = (double)pos / std::pow((double)theta, (2.0 * k) / dim);
        double cos_f = std::cos(freq);
        double sin_f = std::sin(freq);
        out[i]     = (float)((double)in[i] * cos_f - (double)in[i + 1] * sin_f);
        out[i + 1] = (float)((double)in[i] * sin_f + (double)in[i + 1] * cos_f);
    }
}
/** --------------------------------------------------------------------------------------------------------- RMS Norm Tests
 */
static void test_rms_norm() {
    printf("\n── RMS Normalization ──\n");
    const float eps = 1e-5f;
    // Test 1: Sequential values [1..16]
    {
        const int dim = 16;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* wt  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) { in[i] = (float)(i + 1); wt[i] = 1.0f; }
        ref_rms_norm(in, wt, ref, dim, eps);
        ane::dispatch(ane::Op::rms_norm_fp32, uint32_t(dim), eps,
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(wt),
            reinterpret_cast<uintptr_t>(out));
        check("rms_norm [1..16] weight=1", out, ref, dim, 1e-4f);
        std::free(in); std::free(wt); std::free(out); std::free(ref);
    }
    // Test 2: With non-uniform weights
    {
        const int dim = 16;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* wt  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) { in[i] = (float)(i + 1); wt[i] = 0.5f + 0.1f * i; }
        ref_rms_norm(in, wt, ref, dim, eps);
        ane::dispatch(ane::Op::rms_norm_fp32, uint32_t(dim), eps,
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(wt),
            reinterpret_cast<uintptr_t>(out));
        check("rms_norm [1..16] weight=[0.5..2.0]", out, ref, dim, 1e-4f);
        std::free(in); std::free(wt); std::free(out); std::free(ref);
    }
    // Test 3: Constant array → normalized to 1.0 * weight
    {
        const int dim = 16;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* wt  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) { in[i] = 5.0f; wt[i] = 1.0f; }
        ref_rms_norm(in, wt, ref, dim, eps);
        ane::dispatch(ane::Op::rms_norm_fp32, uint32_t(dim), eps,
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(wt),
            reinterpret_cast<uintptr_t>(out));
        check("rms_norm constant=5.0", out, ref, dim, 1e-4f);
        std::free(in); std::free(wt); std::free(out); std::free(ref);
    }
    // Test 4: Near-zero input (eps test)
    {
        const int dim = 16;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* wt  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) { in[i] = 1e-8f; wt[i] = 1.0f; }
        ref_rms_norm(in, wt, ref, dim, eps);
        ane::dispatch(ane::Op::rms_norm_fp32, uint32_t(dim), eps,
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(wt),
            reinterpret_cast<uintptr_t>(out));
        check("rms_norm near-zero input (eps)", out, ref, dim, 1e-2f);
        std::free(in); std::free(wt); std::free(out); std::free(ref);
    }
    // Test 5: Larger dimension (256 = 16 z-vectors)
    {
        const int dim = 256;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* wt  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) { in[i] = sinf((float)i * 0.1f); wt[i] = 1.0f; }
        ref_rms_norm(in, wt, ref, dim, eps);
        ane::dispatch(ane::Op::rms_norm_fp32, uint32_t(dim), eps,
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(wt),
            reinterpret_cast<uintptr_t>(out));
        check("rms_norm dim=256 sinusoidal", out, ref, dim, 1e-3f);
        std::free(in); std::free(wt); std::free(out); std::free(ref);
    }
}
/** --------------------------------------------------------------------------------------------------------- Softmax Tests
 */
static void test_softmax() {
    printf("\n── Softmax ──\n");
    // Test 1: Uniform input → uniform output
    {
        const int dim = 16;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) in[i] = 1.0f;
        ref_softmax(in, ref, dim);
        ane::dispatch(ane::Op::softmax_fp32, uint32_t(dim),
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check("softmax uniform → 1/16", out, ref, dim, 1e-5f);
        std::free(in); std::free(out); std::free(ref);
    }
    // Test 2: One large value (near one-hot)
    {
        const int dim = 16;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) in[i] = 0.0f;
        in[7] = 100.0f;
        ref_softmax(in, ref, dim);
        ane::dispatch(ane::Op::softmax_fp32, uint32_t(dim),
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check("softmax one-hot (in[7]=100)", out, ref, dim, 1e-5f);
        std::free(in); std::free(out); std::free(ref);
    }
    // Test 3: Negative inputs
    {
        const int dim = 16;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) in[i] = (float)(i - 8);  // -8..+7
        ref_softmax(in, ref, dim);
        ane::dispatch(ane::Op::softmax_fp32, uint32_t(dim),
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check("softmax [-8..+7]", out, ref, dim, 1e-4f);
        std::free(in); std::free(out); std::free(ref);
    }
    // Test 4: Sum = 1.0 check
    {
        const int dim = 16;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) in[i] = (float)i * 0.5f;
        ane::dispatch(ane::Op::softmax_fp32, uint32_t(dim),
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        float sum = 0; for (int i = 0; i < dim; i++) sum += out[i];
        check_scalar("softmax sum=1.0", sum, 1.0f, 1e-5f);
        std::free(in); std::free(out);
    }
    // Test 5: Large values (numerical stability)
    {
        const int dim = 16;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) in[i] = 1000.0f + (float)i;
        ref_softmax(in, ref, dim);
        ane::dispatch(ane::Op::softmax_fp32, uint32_t(dim),
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check("softmax large values [1000..1015]", out, ref, dim, 1e-4f);
        float sum = 0; for (int i = 0; i < dim; i++) sum += out[i];
        check_scalar("softmax large values sum=1.0", sum, 1.0f, 1e-4f);
        std::free(in); std::free(out); std::free(ref);
    }
    // Test 6: Larger dimension (256)
    {
        const int dim = 256;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) in[i] = sinf((float)i * 0.05f) * 3.0f;
        ref_softmax(in, ref, dim);
        ane::dispatch(ane::Op::softmax_fp32, uint32_t(dim),
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check("softmax dim=256 sinusoidal", out, ref, dim, 1e-3f);
        std::free(in); std::free(out); std::free(ref);
    }
}
/** --------------------------------------------------------------------------------------------------------- SiLU Tests
 */
static void test_silu() {
    printf("\n── SiLU Activation ──\n");
    // Test 1: Known values across range
    {
        const int dim = 16;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) in[i] = (float)(i - 8);  // -8..+7
        ref_silu(in, ref, dim);
        ane::dispatch(ane::Op::silu_fp32, uint32_t(dim),
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check("silu [-8..+7]", out, ref, dim, 5e-3f);
        std::free(in); std::free(out); std::free(ref);
    }
    // Test 2: SiLU(0) = 0
    {
        auto* in  = static_cast<float*>(alloc(64));
        auto* out = static_cast<float*>(alloc(64));
        for (int i = 0; i < 16; i++) in[i] = 0.0f;
        ane::dispatch(ane::Op::silu_fp32, uint32_t(16),
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check_scalar("silu(0) = 0", out[0], 0.0f, 1e-5f);
        std::free(in); std::free(out);
    }
    // Test 3: Large positive → x (sigmoid→1)
    {
        auto* in  = static_cast<float*>(alloc(64));
        auto* out = static_cast<float*>(alloc(64));
        for (int i = 0; i < 16; i++) in[i] = 10.0f;
        ane::dispatch(ane::Op::silu_fp32, uint32_t(16),
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check_scalar("silu(10) ≈ 10", out[0], 10.0f, 5e-2f);
        std::free(in); std::free(out);
    }
    // Test 4: Large negative → 0 (sigmoid→0)
    {
        auto* in  = static_cast<float*>(alloc(64));
        auto* out = static_cast<float*>(alloc(64));
        for (int i = 0; i < 16; i++) in[i] = -10.0f;
        ane::dispatch(ane::Op::silu_fp32, uint32_t(16),
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check_scalar("silu(-10) ≈ 0", out[0], 0.0f, 5e-3f);
        std::free(in); std::free(out);
    }
    // Test 5: Fine-grained accuracy check across [-5, +5]
    {
        const int dim = 16;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) in[i] = -5.0f + (10.0f * i / (dim - 1));
        ref_silu(in, ref, dim);
        ane::dispatch(ane::Op::silu_fp32, uint32_t(dim),
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check("silu [-5..+5] fine-grained", out, ref, dim, 5e-3f);
        std::free(in); std::free(out); std::free(ref);
    }
    // Test 6: Larger array (256 elements)
    {
        const int dim = 256;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) in[i] = -10.0f + (20.0f * i / (dim - 1));
        ref_silu(in, ref, dim);
        ane::dispatch(ane::Op::silu_fp32, uint32_t(dim),
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check("silu dim=256 [-10..+10]", out, ref, dim, 5e-3f);
        std::free(in); std::free(out); std::free(ref);
    }
}
/** --------------------------------------------------------------------------------------------------------- RoPE Tests
 */
static void test_rope() {
    printf("\n── Rotary Position Embedding ──\n");
    const float theta = 10000.0f;
    // Test 1: Position 0 → identity (no rotation)
    {
        const int dim = 16;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) in[i] = (float)(i + 1);
        ane::dispatch(ane::Op::rope_fp32, uint32_t(dim), uint32_t(0), theta,
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check("rope pos=0 (identity)", out, in, dim, 1e-4f);
        std::free(in); std::free(out);
    }
    // Test 2: Known rotation at pos=1, dim=4
    {
        const int dim = 16;  // padded to 16 for SVE
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) in[i] = 0.0f;
        in[0] = 1.0f; in[1] = 0.0f;  // pair 0: (1, 0)
        in[2] = 0.0f; in[3] = 1.0f;  // pair 1: (0, 1)
        ref_rope(in, ref, dim, 1, theta);
        ane::dispatch(ane::Op::rope_fp32, uint32_t(dim), uint32_t(1), theta,
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check("rope pos=1 dim=16", out, ref, dim, 1e-3f);
        std::free(in); std::free(out); std::free(ref);
    }
    // Test 3: Larger position
    {
        const int dim = 16;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) in[i] = (float)(i + 1);
        ref_rope(in, ref, dim, 42, theta);
        ane::dispatch(ane::Op::rope_fp32, uint32_t(dim), uint32_t(42), theta,
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check("rope pos=42 dim=16", out, ref, dim, 1e-2f);
        std::free(in); std::free(out); std::free(ref);
    }
    // Test 4: Larger dimension (128 — typical head dim)
    {
        const int dim = 128;
        auto* in  = static_cast<float*>(alloc(dim * 4));
        auto* out = static_cast<float*>(alloc(dim * 4));
        auto* ref = static_cast<float*>(alloc(dim * 4));
        for (int i = 0; i < dim; i++) in[i] = sinf((float)i * 0.1f);
        ref_rope(in, ref, dim, 100, theta);
        ane::dispatch(ane::Op::rope_fp32, uint32_t(dim), uint32_t(100), theta,
            reinterpret_cast<uintptr_t>(in), reinterpret_cast<uintptr_t>(out));
        check("rope pos=100 dim=128", out, ref, dim, 5e-2f);
        std::free(in); std::free(out); std::free(ref);
    }
}
/** --------------------------------------------------------------------------------------------------------- Horizontal Reduce + Reciprocal Sqrt
 */
static void test_reduce_and_rsqrt() {
    printf("\n── Horizontal Reduce + Reciprocal Sqrt ──\n");
    // faddv: sum of [1..16] = 136
    {
        auto* in  = static_cast<float*>(alloc(64));
        auto* out = static_cast<float*>(alloc(64));
        for (int i = 0; i < 16; i++) in[i] = (float)(i + 1);
        ane::program p;
        p.emit(ane::Op::load, reinterpret_cast<uintptr_t>(in));
        p.emit(ane::Op::mov_zreg, uint8_t(0), uint8_t(5));
        p.emit(ane::Op::faddv_zreg, uint8_t(5), uint8_t(5));
        p.emit(ane::Op::mov_zreg, uint8_t(5), uint8_t(0));
        p.emit(ane::Op::store, reinterpret_cast<uintptr_t>(out));
        p.exec();
        check_scalar("faddv [1..16] = 136", out[0], 136.0f, 1e-3f);
        // All 16 lanes should have the same broadcast sum
        bool all_same = true;
        for (int i = 1; i < 16; i++) if (out[i] != out[0]) all_same = false;
        tests_total++;
        if (all_same) { printf("  [PASS] faddv broadcasts to all lanes\n"); tests_passed++; }
        else printf("  [FAIL] faddv not broadcast: [0]=%.1f [1]=%.1f\n", out[0], out[1]);
        std::free(in); std::free(out);
    }
    // frsqrt: rsqrt([1, 4, 9, 16, ...])
    {
        auto* in  = static_cast<float*>(alloc(64));
        auto* out = static_cast<float*>(alloc(64));
        auto* ref = static_cast<float*>(alloc(64));
        for (int i = 0; i < 16; i++) { float v = (float)((i + 1) * (i + 1)); in[i] = v; ref[i] = 1.0f / sqrtf(v); }
        ane::program p;
        p.emit(ane::Op::load, reinterpret_cast<uintptr_t>(in));
        p.emit(ane::Op::mov_zreg, uint8_t(0), uint8_t(5));
        p.emit(ane::Op::frsqrt_zreg, uint8_t(5), uint8_t(5));
        p.emit(ane::Op::mov_zreg, uint8_t(5), uint8_t(0));
        p.emit(ane::Op::store, reinterpret_cast<uintptr_t>(out));
        p.exec();
        check("frsqrt [1,4,9,...256]", out, ref, 16, 1e-5f);
        std::free(in); std::free(out); std::free(ref);
    }
}
/** --------------------------------------------------------------------------------------------------------- Broadcast + Scale
 */
static void test_broadcast_scale() {
    printf("\n── Broadcast Scalar + Scale ──\n");
    {
        auto* out = static_cast<float*>(alloc(64));
        ane::program p;
        p.emit(ane::Op::broadcast_scalar_zreg, uint8_t(5), 3.14f);
        p.emit(ane::Op::mov_zreg, uint8_t(5), uint8_t(0));
        p.emit(ane::Op::store, reinterpret_cast<uintptr_t>(out));
        p.exec();
        float ref[16]; for (int i = 0; i < 16; i++) ref[i] = 3.14f;
        check("broadcast 3.14 to all lanes", out, ref, 16, 1e-5f);
        std::free(out);
    }
    {
        auto* in  = static_cast<float*>(alloc(64));
        auto* out = static_cast<float*>(alloc(64));
        auto* ref = static_cast<float*>(alloc(64));
        for (int i = 0; i < 16; i++) { in[i] = (float)(i + 1); ref[i] = (float)(i + 1) * 0.5f; }
        ane::program p;
        p.emit(ane::Op::load, reinterpret_cast<uintptr_t>(in));
        p.emit(ane::Op::mov_zreg, uint8_t(0), uint8_t(5));
        p.emit(ane::Op::fscale_zreg, uint8_t(5), uint8_t(5), 0.5f);
        p.emit(ane::Op::mov_zreg, uint8_t(5), uint8_t(0));
        p.emit(ane::Op::store, reinterpret_cast<uintptr_t>(out));
        p.exec();
        check("fscale [1..16] * 0.5", out, ref, 16, 1e-5f);
        std::free(in); std::free(out); std::free(ref);
    }
}
/** --------------------------------------------------------------------------------------------------------- Main
 */
int main() {
    printf("\n====== Transformer Kernel Tests ======\n");
    test_rms_norm();
    test_softmax();
    test_silu();
    test_rope();
    test_reduce_and_rsqrt();
    test_broadcast_scale();
    printf("\n====== Results: %d/%d passed ======\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
