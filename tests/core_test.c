/** --------------------------------------------------------------------------------------------------------- core_test
 * @file core_test.c
 * @brief A simple benchmark to test the per-core saturation of CBLAS SGEMM on Apple Silicon.
 * 
 * @author Josh Morgan (@joshmorgan1000 on GitHub) with help from Claude and Gemini
 * Released under the MIT License
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <Accelerate/Accelerate.h>

// Uses 2048×2,048 matrices to saturate the cores
#define N 2048
// Each thread runs for 5 seconds to get a stable measurement of sustained performance
#define ITERS 5
/// @brief Global variables for thread synchronization and results storage
static volatile int g_go = 0;
static volatile int g_stop = 0;
static double g_results[18];
/** --------------------------------------------------------------------------------------------------------- now_sec
 * @brief Helper function to get the current time in seconds with high resolution, used for benchmarking.
 * @return The current time in seconds as a double, with high resolution.
 */
static double now_sec() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}
/** --------------------------------------------------------------------------------------------------------- worker
 * @brief Worker function for each thread. Each thread performs SGEMM operations in a loop until the global
 * stop flag is set.
 * @param arg The thread ID passed as a void pointer, which is cast to an integer to identify the thread's
 * index in the results array.
 */
static void *worker(void *arg) {
    int id = (int)(intptr_t)arg;
    float *A = malloc(N * N * sizeof(float));
    float *B = malloc(N * N * sizeof(float));
    float *C = malloc(N * N * sizeof(float));
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(i % 17) * 0.1f;
        B[i] = (float)(i % 13) * 0.1f;
    }
    // warmup
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        N, N, N, 1.0f, A, N, B, N, 0.0f, C, N
    );
    while (!g_go) {} // spin until all threads ready
    double t0 = now_sec();
    int count = 0;
    while (!g_stop) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N, 1.0f, A, N, B, N, 0.0f, C, N
        );
        count++;
    }
    double dt = now_sec() - t0;
    double flops = 2.0 * N * N * N * count;
    g_results[id] = flops / dt / 1e9; // GFLOPS
    free(A); free(B); free(C);
    return NULL;
}
/** --------------------------------------------------------------------------------------------------------- main
 * @brief Main function to run the per-core saturation test for CBLAS SGEMM on Apple Silicon.
 */
int main() {
    printf("SGEMM per-core saturation test (2048x2048, 5s each)\n\n");
    printf("  %-8s  %8s  %8s  %8s\n", "Threads", "GFLOPS", "TFLOPS", "per-thr");
    printf("  %-8s  %8s  %8s  %8s\n", "-------", "------", "------", "-------");
    for (int nt = 1; nt <= 18; nt++) {
        g_go = 0;
        g_stop = 0;
        for (int i = 0; i < 18; i++) {
            g_results[i] = 0;
        }
        pthread_t threads[18];
        for (int i = 0; i < nt; i++) {
            pthread_create(&threads[i], NULL, worker, (void*)(intptr_t)i);
        }
        usleep(100000); // let all threads reach spin loop
        g_go = 1;
        struct timespec req = {5, 0};
        nanosleep(&req, NULL);
        g_stop = 1;
        for (int i = 0; i < nt; i++) {
            pthread_join(threads[i], NULL);
        }
        double total = 0;
        for (int i = 0; i < nt; i++) {
            total += g_results[i];
        }
        printf("  %-8d  %8.1f  %8.3f  %8.1f\n", nt, total, total/1000.0, total/nt);
    }
    return 0;
}
