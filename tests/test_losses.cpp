// test_losses.cpp — Tests for loss functions
#include "test_common.hpp"
#include <cmath>

static const size_t SIZES[] = {16, 64, 128, 256, 1000};
static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

static bool test_mse_loss(size_t n) {
    tap::AlignedBuffer<float> pred(n), target(n);
    srand(42);
    for (size_t i = 0; i < n; ++i) {
        pred[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        target[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    double ref = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)pred[i] - (double)target[i];
        ref += d * d;
    }
    ref /= n;

    float result = ane::kernel::mse_loss_fp32(pred.data(), target.data(), n);
    double err = std::fabs((double)result - ref);
    double tol = std::fabs(ref) * 1e-3 + 1e-5;
    auto name = tap::test_name("mse_loss_fp32", n);
    if (err < tol) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "err=%.2e ref=%.6g got=%.6g", err, ref, (double)result); return false; }
}

static bool test_mae_loss(size_t n) {
    tap::AlignedBuffer<float> pred(n), target(n);
    srand(42);
    for (size_t i = 0; i < n; ++i) {
        pred[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        target[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    double ref = 0.0;
    for (size_t i = 0; i < n; ++i)
        ref += std::fabs((double)pred[i] - (double)target[i]);
    ref /= n;

    float result = ane::kernel::mae_loss_fp32(pred.data(), target.data(), n);
    double err = std::fabs((double)result - ref);
    double tol = std::fabs(ref) * 1e-3 + 1e-5;
    auto name = tap::test_name("mae_loss_fp32", n);
    if (err < tol) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "err=%.2e ref=%.6g got=%.6g", err, ref, (double)result); return false; }
}

int main() {
    TAP_PLAN(2 * NUM_SIZES);

    for (int i = 0; i < NUM_SIZES; ++i) {
        test_mse_loss(SIZES[i]);
        test_mae_loss(SIZES[i]);
    }

    TAP_DIAG("Passed: %d, Failed: %d", TAP_PASSED(), TAP_FAILED());
    return TAP_EXIT();
}
