// test_softmax.cpp — Tests for softmax and log_softmax
#include "test_common.hpp"
#include <cmath>

static const size_t SIZES[] = {4, 16, 64, 128, 256, 1000};
static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

static void ref_softmax(const float* in, float* out, long n) {
    float max_val = in[0];
    for (long i = 1; i < n; ++i) if (in[i] > max_val) max_val = in[i];
    double sum = 0.0;
    for (long i = 0; i < n; ++i) { out[i] = std::exp(in[i] - max_val); sum += out[i]; }
    for (long i = 0; i < n; ++i) out[i] = (float)(out[i] / sum);
}

static void ref_log_softmax(const float* in, float* out, long n) {
    float max_val = in[0];
    for (long i = 1; i < n; ++i) if (in[i] > max_val) max_val = in[i];
    double sum = 0.0;
    for (long i = 0; i < n; ++i) sum += std::exp(in[i] - max_val);
    double log_sum = std::log(sum);
    for (long i = 0; i < n; ++i) out[i] = (float)(in[i] - max_val - log_sum);
}

static bool test_softmax(size_t n) {
    tap::AlignedBuffer<float> in(n), out(n), ref(n);
    srand(42);
    for (size_t i = 0; i < n; ++i) in[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
    out.zero();

    ref_softmax(in.data(), ref.data(), n);
    ane::kernel::softmax_fp32(in.data(), out.data(), n);

    double err = tap::max_abs_error(out.data(), ref.data(), n);
    auto name = tap::test_name("softmax_fp32", n);
    if (err < 5e-4) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e", err); return false; }
}

static bool test_log_softmax(size_t n) {
    tap::AlignedBuffer<float> in(n), out(n), ref(n);
    srand(42);
    for (size_t i = 0; i < n; ++i) in[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
    out.zero();

    ref_log_softmax(in.data(), ref.data(), n);
    ane::kernel::log_softmax_fp32(in.data(), out.data(), n);

    double err = tap::max_abs_error(out.data(), ref.data(), n);
    auto name = tap::test_name("log_softmax_fp32", n);
    if (err < 5e-3) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e", err); return false; }
}

int main() {
    TAP_PLAN(2 * NUM_SIZES);

    for (int i = 0; i < NUM_SIZES; ++i) {
        test_softmax(SIZES[i]);
        test_log_softmax(SIZES[i]);
    }

    TAP_DIAG("Passed: %d, Failed: %d", TAP_PASSED(), TAP_FAILED());
    return TAP_EXIT();
}
