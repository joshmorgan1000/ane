#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

static double now_sec() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main() {
    int sizes[] = {512, 1024, 2048, 4096};
    int nsizes = 4;
    
    printf("CBLAS SGEMM (FP32) — Accelerate framework\n");
    printf("Apple will use AMX/SME/NEON internally as it sees fit\n\n");
    printf("  %-8s  %8s  %8s  %8s\n", "N", "GFLOPS", "TOPS", "Time(s)");
    printf("  %-8s  %8s  %8s  %8s\n", "--------", "--------", "--------", "--------");
    
    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        float *A = malloc(N * N * sizeof(float));
        float *B = malloc(N * N * sizeof(float));
        float *C = malloc(N * N * sizeof(float));
        
        for (int i = 0; i < N * N; i++) {
            A[i] = (float)(i % 17) * 0.1f;
            B[i] = (float)(i % 13) * 0.1f;
            C[i] = 0;
        }
        
        // Warmup
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
        
        // Benchmark
        double flops = 2.0 * N * N * N;
        int iters = (N <= 1024) ? 20 : (N <= 2048) ? 5 : 2;
        
        double t0 = now_sec();
        for (int i = 0; i < iters; i++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
        }
        double dt = now_sec() - t0;
        
        double gflops = (flops * iters) / dt / 1e9;
        printf("  %-8d  %8.1f  %8.3f  %8.4f\n", N, gflops, gflops/1000.0, dt/iters);
        
        free(A); free(B); free(C);
    }
    
    // Now INT8 via vDSP if available, or manual CBLAS with int
    printf("\n\nCBLAS DGEMM (FP64) — Accelerate framework\n\n");
    printf("  %-8s  %8s  %8s  %8s\n", "N", "GFLOPS", "TOPS", "Time(s)");
    printf("  %-8s  %8s  %8s  %8s\n", "--------", "--------", "--------", "--------");
    
    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        double *A = malloc(N * N * sizeof(double));
        double *B = malloc(N * N * sizeof(double));
        double *C = malloc(N * N * sizeof(double));
        
        for (int i = 0; i < N * N; i++) {
            A[i] = (double)(i % 17) * 0.1;
            B[i] = (double)(i % 13) * 0.1;
            C[i] = 0;
        }
        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0, A, N, B, N, 0.0, C, N);
        
        double flops = 2.0 * N * N * N;
        int iters = (N <= 1024) ? 20 : (N <= 2048) ? 5 : 2;
        
        double t0 = now_sec();
        for (int i = 0; i < iters; i++) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        N, N, N, 1.0, A, N, B, N, 0.0, C, N);
        }
        double dt = now_sec() - t0;
        
        double gflops = (flops * iters) / dt / 1e9;
        printf("  %-8d  %8.1f  %8.3f  %8.4f\n", N, gflops, gflops/1000.0, dt/iters);
        
        free(A); free(B); free(C);
    }
    
    return 0;
}
