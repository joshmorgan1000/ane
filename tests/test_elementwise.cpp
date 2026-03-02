// test_elementwise.cpp — Tests for basic elementwise operations
#include "test_common.hpp"

static const size_t SIZES[] = {1, 16, 64, 128, 1000, 10000};
static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

static bool test_add(size_t n) {
    tap::AlignedBuffer<float> a(n), b(n), out(n), ref(n);
    a.fill_random(42); b.fill_random(99); out.zero();
    for (size_t i = 0; i < n; ++i) ref[i] = a[i] + b[i];

    ane::kernel::add_fp32(a.data(), b.data(), out.data(), n);
    double err = tap::max_abs_error(out.data(), ref.data(), n);
    auto name = tap::test_name("add_fp32", n);
    if (err < 1e-6) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e", err); return false; }
}

static bool test_sub(size_t n) {
    tap::AlignedBuffer<float> a(n), b(n), out(n), ref(n);
    a.fill_random(42); b.fill_random(99); out.zero();
    for (size_t i = 0; i < n; ++i) ref[i] = a[i] - b[i];

    ane::kernel::sub_fp32(a.data(), b.data(), out.data(), n);
    double err = tap::max_abs_error(out.data(), ref.data(), n);
    auto name = tap::test_name("sub_fp32", n);
    if (err < 1e-6) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e", err); return false; }
}

static bool test_mul(size_t n) {
    tap::AlignedBuffer<float> a(n), b(n), out(n), ref(n);
    a.fill_random(42); b.fill_random(99); out.zero();
    for (size_t i = 0; i < n; ++i) ref[i] = a[i] * b[i];

    ane::kernel::mul_fp32(a.data(), b.data(), out.data(), n);
    double err = tap::max_abs_error(out.data(), ref.data(), n);
    auto name = tap::test_name("mul_fp32", n);
    if (err < 1e-6) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e", err); return false; }
}

static bool test_neg(size_t n) {
    tap::AlignedBuffer<float> a(n), out(n), ref(n);
    a.fill_random(42); out.zero();
    for (size_t i = 0; i < n; ++i) ref[i] = -a[i];

    ane::kernel::neg_fp32(a.data(), out.data(), n);
    double err = tap::max_abs_error(out.data(), ref.data(), n);
    auto name = tap::test_name("neg_fp32", n);
    if (err < 1e-6) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e", err); return false; }
}

static bool test_abs(size_t n) {
    tap::AlignedBuffer<float> a(n), out(n), ref(n);
    a.fill_random(42); out.zero();
    for (size_t i = 0; i < n; ++i) ref[i] = std::fabs(a[i]);

    ane::kernel::abs_fp32(a.data(), out.data(), n);
    double err = tap::max_abs_error(out.data(), ref.data(), n);
    auto name = tap::test_name("abs_fp32", n);
    if (err < 1e-6) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e", err); return false; }
}

static bool test_scalar_mul(size_t n) {
    tap::AlignedBuffer<float> a(n), out(n), ref(n);
    a.fill_random(42); out.zero();
    float s = 3.14f;
    for (size_t i = 0; i < n; ++i) ref[i] = a[i] * s;

    ane::kernel::scalar_mul_fp32(a.data(), s, out.data(), n);
    double err = tap::max_abs_error(out.data(), ref.data(), n);
    auto name = tap::test_name("scalar_mul_fp32", n);
    if (err < 1e-5) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e", err); return false; }
}

int main() {
    // 6 ops x 6 sizes = 36 tests
    TAP_PLAN(6 * NUM_SIZES);

    for (int i = 0; i < NUM_SIZES; ++i) {
        test_add(SIZES[i]);
        test_sub(SIZES[i]);
        test_mul(SIZES[i]);
        test_neg(SIZES[i]);
        test_abs(SIZES[i]);
        test_scalar_mul(SIZES[i]);
    }

    TAP_DIAG("Passed: %d, Failed: %d", TAP_PASSED(), TAP_FAILED());
    return TAP_EXIT();
}
