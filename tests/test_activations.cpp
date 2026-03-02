// test_activations.cpp — Tests for activation functions
#include "test_common.hpp"
#include <cmath>

static const size_t SIZES[] = {16, 64, 128, 256, 1000, 10000};
static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

static void fill_moderate(float* data, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++)
        data[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
}

// Reference implementations
static void ref_relu(const float* in, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}
static void ref_sigmoid(const float* in, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = 1.0f / (1.0f + std::exp(-in[i]));
}
static void ref_tanh(const float* in, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = std::tanh(in[i]);
}
static void ref_gelu(const float* in, float* out, long n) {
    for (long i = 0; i < n; i++)
        out[i] = 0.5f * in[i] * (1.0f + std::erf(in[i] / std::sqrt(2.0f)));
}
static void ref_silu(const float* in, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = in[i] / (1.0f + std::exp(-in[i]));
}
static void ref_elu(const float* in, float* out, float alpha, long n) {
    for (long i = 0; i < n; i++)
        out[i] = in[i] > 0.0f ? in[i] : alpha * (std::exp(in[i]) - 1.0f);
}
static void ref_softplus(const float* in, float* out, long n) {
    for (long i = 0; i < n; i++) {
        if (in[i] > 20.0f) out[i] = in[i];
        else if (in[i] < -20.0f) out[i] = std::exp(in[i]);
        else out[i] = std::log(1.0f + std::exp(in[i]));
    }
}

// Generic test runner
using ActFn = void(*)(const float*, float*, long);
static bool test_act(const char* kernel_name, ActFn kernel, ActFn reference, size_t n, double tol) {
    tap::AlignedBuffer<float> in(n), out(n), ref(n);
    fill_moderate(in.data(), n, 42);
    out.zero();
    reference(in.data(), ref.data(), n);
    kernel(in.data(), out.data(), n);
    double err = tap::max_abs_error(out.data(), ref.data(), n);
    auto name = tap::test_name(kernel_name, n);
    if (err < tol) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e (tol=%.2e)", err, tol); return false; }
}

static bool test_elu(size_t n) {
    tap::AlignedBuffer<float> in(n), out(n), ref(n);
    fill_moderate(in.data(), n, 42);
    out.zero();
    ref_elu(in.data(), ref.data(), 1.0f, n);
    ane::kernel::elu_fp32(in.data(), out.data(), 1.0f, n);
    double err = tap::max_abs_error(out.data(), ref.data(), n);
    auto name = tap::test_name("elu_fp32", n);
    if (err < 5e-2) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e", err); return false; }
}

int main() {
    // 7 activations x 6 sizes = 42 tests
    TAP_PLAN(7 * NUM_SIZES);

    for (int i = 0; i < NUM_SIZES; ++i) {
        size_t n = SIZES[i];
        test_act("relu_fp32",    ane::kernel::relu_fp32,    ref_relu,     n, 1e-6);
        test_act("sigmoid_fp32", ane::kernel::sigmoid_fp32, ref_sigmoid,  n, 5e-4);
        test_act("tanh_fp32",    ane::kernel::tanh_fp32,    ref_tanh,     n, 5e-4);
        test_act("gelu_fp32",    ane::kernel::gelu_fp32,    ref_gelu,     n, 5e-2);
        test_act("silu_fp32",    ane::kernel::silu_fp32,    ref_silu,     n, 5e-4);
        test_act("softplus_fp32",ane::kernel::softplus_fp32,ref_softplus, n, 1e-1);
        test_elu(n);
    }

    TAP_DIAG("Passed: %d, Failed: %d", TAP_PASSED(), TAP_FAILED());
    return TAP_EXIT();
}
