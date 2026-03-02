// test_memory_ops.cpp — Tests for copy, fill, slice, reshape, concat, transpose
#include "test_common.hpp"

static const size_t SIZES[] = {1, 16, 64, 128, 1000, 10000};
static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

static bool test_copy(size_t n) {
    tap::AlignedBuffer<float> src(n), dst(n);
    src.fill_random(42); dst.zero();

    ane::kernel::copy_fp32(src.data(), dst.data(), n);
    double err = tap::max_abs_error(dst.data(), src.data(), n);
    auto name = tap::test_name("copy_fp32", n);
    if (err == 0.0) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e", err); return false; }
}

static bool test_fill(size_t n) {
    tap::AlignedBuffer<float> buf(n);
    buf.zero();
    float val = 42.0f;

    ane::kernel::fill_fp32(buf.data(), val, n);
    bool ok = true;
    for (size_t i = 0; i < n; ++i) if (buf[i] != val) { ok = false; break; }
    auto name = tap::test_name("fill_fp32", n);
    if (ok) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "mismatch"); return false; }
}

static bool test_concat(size_t n) {
    size_t na = n, nb = n / 2 + 1;
    tap::AlignedBuffer<float> a(na), b(nb), out(na + nb);
    a.fill_random(42); b.fill_random(99); out.zero();

    ane::kernel::concat_fp32(a.data(), b.data(), out.data(), na, nb);

    bool ok = true;
    for (size_t i = 0; i < na && ok; ++i) if (out[i] != a[i]) ok = false;
    for (size_t i = 0; i < nb && ok; ++i) if (out[na + i] != b[i]) ok = false;
    auto name = tap::test_name("concat_fp32", n);
    if (ok) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "mismatch"); return false; }
}

static bool test_transpose(size_t n) {
    long rows = 16, cols = (long)(n / 16);
    if (cols < 1) cols = 1;
    size_t total = rows * cols;
    tap::AlignedBuffer<float> src(total), dst(total), ref(total);
    src.fill_random(42); dst.zero();

    for (long i = 0; i < rows; ++i)
        for (long j = 0; j < cols; ++j)
            ref[j * rows + i] = src[i * cols + j];

    ane::kernel::transpose_fp32(src.data(), dst.data(), rows, cols);
    double err = tap::max_abs_error(dst.data(), ref.data(), total);
    auto name = tap::test_name("transpose_fp32", total);
    if (err == 0.0) { TAP_OK(name.c_str()); return true; }
    else { TAP_FAIL(name.c_str(), "max_err=%.2e", err); return false; }
}

int main() {
    TAP_PLAN(4 * NUM_SIZES);

    for (int i = 0; i < NUM_SIZES; ++i) {
        test_copy(SIZES[i]);
        test_fill(SIZES[i]);
        test_concat(SIZES[i]);
        test_transpose(SIZES[i]);
    }

    TAP_DIAG("Passed: %d, Failed: %d", TAP_PASSED(), TAP_FAILED());
    return TAP_EXIT();
}
