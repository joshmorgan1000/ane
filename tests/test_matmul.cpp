// test_matmul.cpp — Tests for matrix multiply kernels
#include "test_common.hpp"

struct MatSize { long M, N, K; };

static const MatSize SIZES[] = {
    {1, 1, 1},
    {4, 4, 4},
    {16, 16, 16},
    {32, 64, 16},
    {64, 32, 128},
    {128, 128, 128},
};
static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

static void fill_small(float* data, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i)
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

static void ref_matmul(const float* A, const float* B, float* C, long M, long N, long K) {
    for (long i = 0; i < M; ++i)
        for (long j = 0; j < N; ++j) {
            double sum = 0.0;
            for (long k = 0; k < K; ++k)
                sum += (double)A[i * K + k] * (double)B[k * N + j];
            C[i * N + j] = (float)sum;
        }
}

static void ref_matmul_tn(const float* A, const float* B, float* C, long M, long N, long K) {
    // C = A^T * B, A stored KxM
    for (long i = 0; i < M; ++i)
        for (long j = 0; j < N; ++j) {
            double sum = 0.0;
            for (long k = 0; k < K; ++k)
                sum += (double)A[k * M + i] * (double)B[k * N + j];
            C[i * N + j] = (float)sum;
        }
}

static bool test_matmul(long M, long N, long K) {
    tap::AlignedBuffer<float> A(M * K), B(K * N), C(M * N), ref(M * N);
    fill_small(A.data(), M * K, 42);
    fill_small(B.data(), K * N, 99);
    C.zero(); ref.zero();

    ref_matmul(A.data(), B.data(), ref.data(), M, N, K);
    ane::kernel::matmul_fp32(A.data(), B.data(), C.data(), M, N, K);

    double max_err = tap::max_abs_error(C.data(), ref.data(), M * N);
    double tol = K * 5e-4;
    auto name = tap::test_name("matmul_fp32", M, N, K);
    if (max_err < tol) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e (tol=%.2e)", max_err, tol); return false; }
}

static bool test_matmul_tn(long M, long N, long K) {
    tap::AlignedBuffer<float> A(K * M), B(K * N), C(M * N), ref(M * N);
    fill_small(A.data(), K * M, 42);
    fill_small(B.data(), K * N, 99);
    C.zero(); ref.zero();

    ref_matmul_tn(A.data(), B.data(), ref.data(), M, N, K);
    ane::kernel::matmul_fp32_tn(A.data(), B.data(), C.data(), M, N, K);

    double max_err = tap::max_abs_error(C.data(), ref.data(), M * N);
    double tol = K * 5e-4;
    auto name = tap::test_name("matmul_fp32_tn", M, N, K);
    if (max_err < tol) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e (tol=%.2e)", max_err, tol); return false; }
}

int main() {
    TAP_PLAN(2 * NUM_SIZES);

    for (int i = 0; i < NUM_SIZES; ++i) {
        test_matmul(SIZES[i].M, SIZES[i].N, SIZES[i].K);
        test_matmul_tn(SIZES[i].M, SIZES[i].N, SIZES[i].K);
    }

    TAP_DIAG("Passed: %d, Failed: %d", TAP_PASSED(), TAP_FAILED());
    return TAP_EXIT();
}
