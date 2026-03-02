// test_common.hpp — Shared test utilities with TAP output format
#pragma once

#include <ane/ane.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

namespace tap {

struct State {
    int test_number = 0;
    int passed = 0;
    int failed = 0;
    int planned = 0;
};

inline State& state() {
    static State s;
    return s;
}

} // namespace tap

#define TAP_PLAN(n) do { \
    printf("TAP version 14\n"); \
    printf("1..%d\n", (n)); \
    tap::state().planned = (n); \
} while(0)

#define TAP_OK(description) do { \
    tap::state().test_number++; \
    tap::state().passed++; \
    printf("ok %d - %s\n", tap::state().test_number, (description)); \
} while(0)

#define TAP_FAIL(description, ...) do { \
    tap::state().test_number++; \
    tap::state().failed++; \
    printf("not ok %d - %s\n", tap::state().test_number, (description)); \
    printf("# "); printf(__VA_ARGS__); printf("\n"); \
} while(0)

#define TAP_DIAG(...) do { printf("# "); printf(__VA_ARGS__); printf("\n"); } while(0)

#define TAP_EXIT() (tap::state().failed > 0 ? 1 : 0)
#define TAP_PASSED() (tap::state().passed)
#define TAP_FAILED() (tap::state().failed)

#define TAP_CHECK_NEAR(actual, expected, tol, description) do { \
    double _a = static_cast<double>(actual); \
    double _e = static_cast<double>(expected); \
    if (std::fabs(_a - _e) <= static_cast<double>(tol)) { TAP_OK(description); } \
    else { TAP_FAIL(description, "expected=%.9g actual=%.9g delta=%.2e", _e, _a, std::fabs(_a - _e)); } \
} while(0)

namespace tap {

template<typename T>
T* aligned_alloc(size_t n) {
    T* p = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&p), 64, n * sizeof(T)) != 0)
        return nullptr;
    return p;
}

template<typename T>
class AlignedBuffer {
public:
    explicit AlignedBuffer(size_t n) : size_(n) { ptr_ = aligned_alloc<T>(n); }
    ~AlignedBuffer() { if (ptr_) free(ptr_); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return size_; }
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }

    void fill(T value) { for (size_t i = 0; i < size_; ++i) ptr_[i] = value; }
    void fill_random(unsigned seed = 42) {
        srand(seed);
        for (size_t i = 0; i < size_; ++i) {
            if constexpr (std::is_floating_point_v<T>)
                ptr_[i] = static_cast<T>((double)rand() / RAND_MAX * 2.0 - 1.0);
            else
                ptr_[i] = static_cast<T>(rand());
        }
    }
    void zero() { memset(ptr_, 0, size_ * sizeof(T)); }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

template<typename T>
double max_abs_error(const T* actual, const T* expected, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double err = std::fabs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        if (err > max_err) max_err = err;
    }
    return max_err;
}

inline std::string test_name(const char* base, size_t n) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s: n=%zu", base, n);
    return buf;
}

inline std::string test_name(const char* base, size_t m, size_t n, size_t k) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s: %zux%zu x %zux%zu", base, m, k, k, n);
    return buf;
}

using Clock = std::chrono::high_resolution_clock;

template<typename Func>
double time_ns(Func&& fn, int iters = 10) {
    // warmup
    for (int i = 0; i < 3; ++i) fn();
    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        times[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    return times[iters / 2]; // median
}

} // namespace tap
