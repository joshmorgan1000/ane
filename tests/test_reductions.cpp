// test_reductions.cpp — Tests for reduction operations
#include "test_common.hpp"
#include <cmath>
#include <numeric>

static const size_t SIZES[] = {1, 16, 64, 128, 1000, 10000};
static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

static bool test_reduce_sum(size_t n) {
    tap::AlignedBuffer<float> a(n);
    a.fill_random(42);
    double ref = 0.0;
    for (size_t i = 0; i < n; ++i) ref += a[i];

    float result = ane::kernel::reduce_sum_fp32(a.data(), n);
    double err = std::fabs((double)result - ref);
    // Allow relative tolerance that scales with n
    double tol = std::fabs(ref) * 1e-4 + n * 1e-5;
    auto name = tap::test_name("reduce_sum_fp32", n);
    if (err < tol) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "err=%.2e ref=%.6g got=%.6g", err, ref, (double)result); return false; }
}

static bool test_reduce_max(size_t n) {
    tap::AlignedBuffer<float> a(n);
    a.fill_random(42);
    float ref = a[0];
    for (size_t i = 1; i < n; ++i) if (a[i] > ref) ref = a[i];

    float result = ane::kernel::reduce_max_fp32(a.data(), n);
    auto name = tap::test_name("reduce_max_fp32", n);
    if (result == ref) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "expected=%.6g got=%.6g", ref, result); return false; }
}

static bool test_reduce_min(size_t n) {
    tap::AlignedBuffer<float> a(n);
    a.fill_random(42);
    float ref = a[0];
    for (size_t i = 1; i < n; ++i) if (a[i] < ref) ref = a[i];

    float result = ane::kernel::reduce_min_fp32(a.data(), n);
    auto name = tap::test_name("reduce_min_fp32", n);
    if (result == ref) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "expected=%.6g got=%.6g", ref, result); return false; }
}

static bool test_dot(size_t n) {
    tap::AlignedBuffer<float> a(n), b(n);
    a.fill_random(42); b.fill_random(99);
    double ref = 0.0;
    for (size_t i = 0; i < n; ++i) ref += (double)a[i] * (double)b[i];

    float result = ane::kernel::dot_fp32(a.data(), b.data(), n);
    double err = std::fabs((double)result - ref);
    double tol = std::fabs(ref) * 1e-3 + n * 1e-4;
    auto name = tap::test_name("dot_fp32", n);
    if (err < tol) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "err=%.2e ref=%.6g got=%.6g", err, ref, (double)result); return false; }
}

static bool test_argmax(size_t n) {
    tap::AlignedBuffer<float> a(n);
    a.fill_random(42);
    int32_t ref = 0;
    for (size_t i = 1; i < n; ++i) if (a[i] > a[ref]) ref = (int32_t)i;

    int32_t result = ane::kernel::argmax_fp32(a.data(), n);
    auto name = tap::test_name("argmax_fp32", n);
    if (result == ref) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "expected=%d got=%d", ref, result); return false; }
}

int main() {
    // 5 ops x 6 sizes = 30 tests
    TAP_PLAN(5 * NUM_SIZES);

    for (int i = 0; i < NUM_SIZES; ++i) {
        test_reduce_sum(SIZES[i]);
        test_reduce_max(SIZES[i]);
        test_reduce_min(SIZES[i]);
        test_dot(SIZES[i]);
        test_argmax(SIZES[i]);
    }

    TAP_DIAG("Passed: %d, Failed: %d", TAP_PASSED(), TAP_FAILED());
    return TAP_EXIT();
}
