#!/bin/bash
# =============================================================================
# run_full_throughput_tests.sh — Apple Silicon Concurrent Throughput Measurement
#
# Fully self-contained. Generates all source, builds, runs, analyzes.
# Each worker logs heartbeats to stdout: <epoch_ns> <cumulative_ops>
# Analysis finds the proven-concurrent window and computes ops/sec from
# interior heartbeats only.
#
# Workers:
#   GPU  (Metal): char4 INT8 multiply-accumulate, 1M threads × 10K iters
#   SME  (Nt thr): ARM SMOPA int8 outer product on ZA tiles
#   BNNS (Nt thr): Apple Accelerate INT8 FullyConnected (int8×int8→int32)
#   NEON (Nt thr): ARM NEON FP32 fused multiply-add
#
# Thread counts auto-detected from hardware topology.
#
# Usage: bash run_full_throughput_tests.sh
# Requirements: Apple Silicon (SME2), Xcode/clang, Swift, python3
#
# Author: Josh Morgan (@joshmorgan1000 on GitHub) with help from Claude and Gemini Pro
# Released under MIT License
# =============================================================================
set -u
WORK=$(mktemp -d /tmp/ane_test.XXXXXX)
trap "rm -rf $WORK" EXIT
cd "$WORK"
DURATION=10
# ── Auto-detect core topology ────────────────────────────────────────────────
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
PCORES=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo 4)
ECORES=$(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null || echo 0)
TOTAL_CORES=$(sysctl -n hw.logicalcpu 2>/dev/null || echo "$((PCORES + ECORES))")
# SME runs on P-cores (Super/Performance cores with SME units)
SME_THREADS=$PCORES
# BNNS benefits from P-cores
BNNS_THREADS=$PCORES
# CBLAS SGEMM benefits from P-cores (AMX dispatch)
CBLAS_THREADS=$PCORES
# NEON can run on all cores; leave 1 for OS overhead
NEON_THREADS=$((TOTAL_CORES - 1))
[ "$NEON_THREADS" -lt 1 ] && NEON_THREADS=1
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "   ${CHIP} — Concurrent Throughput Measurement"
echo "   Each worker runs ${DURATION}s, logging timestamped heartbeats."
echo "═══════════════════════════════════════════════════════════════════════"
echo "   Cores: ${PCORES} P-cores + ${ECORES} E-cores = ${TOTAL_CORES} total"
echo "   SME threads: ${SME_THREADS}  BNNS threads: ${BNNS_THREADS}  CBLAS threads: ${CBLAS_THREADS}  NEON threads: ${NEON_THREADS}"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
# =============================================================================
# Generate source files
# =============================================================================
# ── Metal shader ─────────────────────────────────────────────────────────────
cat > int8_peak.metal << 'METAL'
#include <metal_stdlib>
using namespace metal;
kernel void int8_peak(
    device int* out [[buffer(0)]],
    device const char4* seed [[buffer(1)]],
    constant uint& iters [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    char4 s = seed[gid & 0xFF];
    int4 a0=int4(0),a1=int4(0),a2=int4(0),a3=int4(0);
    int4 a4=int4(0),a5=int4(0),a6=int4(0),a7=int4(0);
    char4 x0=s,x1=s+char4(1),x2=s+char4(2),x3=s+char4(3);
    char4 y0=s+char4(4),y1=s+char4(5),y2=s+char4(6),y3=s+char4(7);
    for (uint i = 0; i < iters; i++) {
        a0+=int4(x0)*int4(y0); a1+=int4(x1)*int4(y1);
        a2+=int4(x2)*int4(y2); a3+=int4(x3)*int4(y3);
        a4+=int4(x0)*int4(y1); a5+=int4(x1)*int4(y2);
        a6+=int4(x2)*int4(y3); a7+=int4(x3)*int4(y0);
    }
    out[gid]=a0.x+a1.x+a2.x+a3.x+a4.x+a5.x+a6.x+a7.x;
}
METAL
# ── GPU worker (Swift) ──────────────────────────────────────────────────────
cat > gpu_worker.swift << SWIFT
import Foundation
import Metal
let device=MTLCreateSystemDefaultDevice()!
let queue=device.makeCommandQueue()!
let src=try! String(contentsOfFile:"int8_peak.metal",encoding:.utf8)
let lib=try! device.makeLibrary(source:src,options:nil)
let pipe=try! device.makeComputePipelineState(function:lib.makeFunction(name:"int8_peak")!)
let nT=1024*1024; var it:UInt32=10_000
let opsPerDispatch=Double(nT)*Double(it)*32.0*2.0
let ob=device.makeBuffer(length:nT*4,options:.storageModeShared)!
let sb=device.makeBuffer(length:256*4,options:.storageModeShared)!
let sp=sb.contents().bindMemory(to:Int8.self,capacity:1024)
for i in 0..<1024{sp[i]=Int8(truncatingIfNeeded:i&*7&+3)}
let tg=MTLSize(width:256,height:1,depth:1);let gr=MTLSize(width:nT,height:1,depth:1)
for _ in 0..<3{let c=queue.makeCommandBuffer()!;let e=c.makeComputeCommandEncoder()!
e.setComputePipelineState(pipe);e.setBuffer(ob,offset:0,index:0)
e.setBuffer(sb,offset:0,index:1);e.setBytes(&it,length:4,index:2)
e.dispatchThreads(gr,threadsPerThreadgroup:tg);e.endEncoding();c.commit();c.waitUntilCompleted()}
var cumOps=0.0;let t0=CFAbsoluteTimeGetCurrent();var dispatches=0
while CFAbsoluteTimeGetCurrent()-t0<Double($DURATION){
let c=queue.makeCommandBuffer()!;let e=c.makeComputeCommandEncoder()!
e.setComputePipelineState(pipe);e.setBuffer(ob,offset:0,index:0)
e.setBuffer(sb,offset:0,index:1);e.setBytes(&it,length:4,index:2)
e.dispatchThreads(gr,threadsPerThreadgroup:tg);e.endEncoding();c.commit()
dispatches+=1
if dispatches%10==0{c.waitUntilCompleted();cumOps+=opsPerDispatch*10
let ns=UInt64(Date().timeIntervalSince1970*1e9)
print("\(ns) \(String(format:"%.0f",cumOps))")}}
let fc=queue.makeCommandBuffer()!;fc.commit();fc.waitUntilCompleted()
SWIFT
# ── SME assembly kernel ─────────────────────────────────────────────────────
cat > sme_kern.s << 'ASM'
.section __TEXT,__text,regular,pure_instructions
.global _smopa_loop
.p2align 4
_smopa_loop:
    stp x29,x30,[sp,#-16]!
    mov x29,sp
    mov x10,x0
    smstart
    ptrue p0.b
    ptrue p1.b
1:  smopa za0.s,p0/m,p1/m,z0.b,z1.b
    smopa za1.s,p0/m,p1/m,z2.b,z3.b
    smopa za2.s,p0/m,p1/m,z4.b,z5.b
    smopa za3.s,p0/m,p1/m,z6.b,z7.b
    smopa za0.s,p0/m,p1/m,z8.b,z9.b
    smopa za1.s,p0/m,p1/m,z10.b,z11.b
    smopa za2.s,p0/m,p1/m,z12.b,z13.b
    smopa za3.s,p0/m,p1/m,z14.b,z15.b
    subs x10,x10,#1
    b.ne 1b
    smstop
    ldp x29,x30,[sp],#16
    ret
ASM
# ── SME worker (C, 4 threads) ───────────────────────────────────────────────
cat > sme_worker.c << 'CSRC'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>
extern void smopa_loop(uint64_t iters);
#define CHUNK 10000000ULL
#define OPS_PER_ITER 16384.0
#ifndef NT
#define NT 4
#endif
static volatile int g_stop = 0;
static volatile double g_cumops = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static void *worker(void *a) {
    (void)a;
    while (!g_stop) {
        smopa_loop(CHUNK);
        pthread_mutex_lock(&g_lock);
        g_cumops += CHUNK * OPS_PER_ITER;
        pthread_mutex_unlock(&g_lock);
    }
    return NULL;
}
int main(int argc, char **argv) {
    int duration = argc > 1 ? atoi(argv[1]) : 10;
    pthread_t t[NT];
    for (int i = 0; i < NT; i++) pthread_create(&t[i], NULL, worker, NULL);
    struct timespec start, now;
    clock_gettime(CLOCK_REALTIME, &start);
    double prev_ops = 0;
    setbuf(stdout, NULL);
    while (1) {
        struct timespec req = {0, 100000000};
        nanosleep(&req, NULL);
        clock_gettime(CLOCK_REALTIME, &now);
        double elapsed = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
        pthread_mutex_lock(&g_lock);
        double ops = g_cumops;
        pthread_mutex_unlock(&g_lock);
        if (ops > prev_ops) {
            uint64_t ns = (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
            printf("%llu %.0f\n", (unsigned long long)ns, ops);
            prev_ops = ops;
        }
        if (elapsed >= duration) break;
    }
    g_stop = 1;
    for (int i = 0; i < NT; i++) pthread_join(t[i], NULL);
    return 0;
}
CSRC
# ── BNNS INT8 worker (C, 4 threads) ─────────────────────────────────────────
cat > bnns_worker.c << 'CSRC'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>
#include <Accelerate/Accelerate.h>
#define K_DIM 4096
#define N_DIM 1024
#define BATCH 512
#ifndef NT
#define NT 4
#endif
#define OPS_PER_CALL ((double)BATCH * N_DIM * K_DIM * 2.0)
static int8_t *g_weights;
static volatile int g_stop = 0;
static volatile double g_cumops = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static void *worker(void *a) {
    (void)a;
    BNNSLayerParametersFullyConnected params;
    memset(&params, 0, sizeof(params));
    params.i_desc.layout = BNNSDataLayoutVector;
    params.i_desc.size[0] = K_DIM; params.i_desc.stride[0] = 1;
    params.i_desc.data_type = BNNSDataTypeInt8; params.i_desc.data_scale = 1.0f;
    params.o_desc.layout = BNNSDataLayoutVector;
    params.o_desc.size[0] = N_DIM; params.o_desc.stride[0] = 1;
    params.o_desc.data_type = BNNSDataTypeInt32; params.o_desc.data_scale = 1.0f;
    params.w_desc.layout = BNNSDataLayoutRowMajorMatrix;
    params.w_desc.size[0] = K_DIM; params.w_desc.size[1] = N_DIM;
    params.w_desc.stride[0] = 1; params.w_desc.stride[1] = K_DIM;
    params.w_desc.data_type = BNNSDataTypeInt8; params.w_desc.data = g_weights;
    params.w_desc.data_scale = 1.0f;
    BNNSFilterParameters fp; memset(&fp, 0, sizeof(fp));
    BNNSFilter f = BNNSFilterCreateLayerFullyConnected(&params, &fp);
    if (!f) return NULL;
    int8_t *input = calloc(BATCH * K_DIM, 1);
    int32_t *output = calloc(BATCH * N_DIM, 4);
    for (int i = 0; i < BATCH * K_DIM; i++) input[i] = (i % 5) - 2;
    while (!g_stop) {
        BNNSFilterApplyBatch(f, BATCH, input, K_DIM, output, N_DIM * 4);
        pthread_mutex_lock(&g_lock);
        g_cumops += OPS_PER_CALL;
        pthread_mutex_unlock(&g_lock);
    }
    BNNSFilterDestroy(f); free(input); free(output);
    return NULL;
}
int main(int argc, char **argv) {
    int duration = argc > 1 ? atoi(argv[1]) : 10;
    g_weights = calloc(K_DIM * N_DIM, 1);
    for (int i = 0; i < K_DIM * N_DIM; i++) g_weights[i] = (i % 7) - 3;
    pthread_t t[NT];
    for (int i = 0; i < NT; i++) pthread_create(&t[i], NULL, worker, NULL);
    struct timespec start, now;
    clock_gettime(CLOCK_REALTIME, &start);
    double prev_ops = 0;
    setbuf(stdout, NULL);
    while (1) {
        struct timespec req = {0, 100000000};
        nanosleep(&req, NULL);
        clock_gettime(CLOCK_REALTIME, &now);
        double elapsed = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
        pthread_mutex_lock(&g_lock);
        double ops = g_cumops;
        pthread_mutex_unlock(&g_lock);
        if (ops > prev_ops) {
            uint64_t ns = (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
            printf("%llu %.0f\n", (unsigned long long)ns, ops);
            prev_ops = ops;
        }
        if (elapsed >= duration) break;
    }
    g_stop = 1;
    for (int i = 0; i < NT; i++) pthread_join(t[i], NULL);
    free(g_weights);
    return 0;
}
CSRC
# ── CBLAS SGEMM worker (C, Nt threads) ─────────────────────────────────────
cat > cblas_worker.c << 'CSRC'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <Accelerate/Accelerate.h>
#define MAT_N 2048
#define OPS_PER_GEMM (2.0 * MAT_N * MAT_N * MAT_N)
#ifndef NT
#define NT 4
#endif
static volatile int g_stop = 0;
static volatile double g_cumops = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static void *worker(void *a) {
    (void)a;
    float *A = malloc(MAT_N * MAT_N * sizeof(float));
    float *B = malloc(MAT_N * MAT_N * sizeof(float));
    float *C = malloc(MAT_N * MAT_N * sizeof(float));
    for (int i = 0; i < MAT_N * MAT_N; i++) {
        A[i] = (float)(i % 17) * 0.1f;
        B[i] = (float)(i % 13) * 0.1f;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                MAT_N, MAT_N, MAT_N, 1.0f, A, MAT_N, B, MAT_N, 0.0f, C, MAT_N);
    while (!g_stop) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    MAT_N, MAT_N, MAT_N, 1.0f, A, MAT_N, B, MAT_N, 0.0f, C, MAT_N);
        pthread_mutex_lock(&g_lock);
        g_cumops += OPS_PER_GEMM;
        pthread_mutex_unlock(&g_lock);
    }
    free(A); free(B); free(C);
    return NULL;
}
int main(int argc, char **argv) {
    int duration = argc > 1 ? atoi(argv[1]) : 10;
    pthread_t t[NT];
    for (int i = 0; i < NT; i++) pthread_create(&t[i], NULL, worker, NULL);
    struct timespec start, now;
    clock_gettime(CLOCK_REALTIME, &start);
    double prev_ops = 0;
    setbuf(stdout, NULL);
    while (1) {
        struct timespec req = {0, 100000000};
        nanosleep(&req, NULL);
        clock_gettime(CLOCK_REALTIME, &now);
        double elapsed = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
        pthread_mutex_lock(&g_lock);
        double ops = g_cumops;
        pthread_mutex_unlock(&g_lock);
        if (ops > prev_ops) {
            uint64_t ns = (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
            printf("%llu %.0f\n", (unsigned long long)ns, ops);
            prev_ops = ops;
        }
        if (elapsed >= duration) break;
    }
    g_stop = 1;
    for (int i = 0; i < NT; i++) pthread_join(t[i], NULL);
    return 0;
}
CSRC
# ── NEON FMA kernel (assembly) ──────────────────────────────────────────────
cat > neon_kern.s << 'ASM'
.section __TEXT,__text,regular,pure_instructions
.global _neon_fma_loop
.p2align 4
_neon_fma_loop:
    fmov v2.4s, #1.0
    fmov v3.4s, #2.0
    fmov v4.4s, #1.0
    fmov v5.4s, #2.0
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    movi v24.4s, #0
    movi v25.4s, #0
    movi v26.4s, #0
    movi v27.4s, #0
1:  fmla v8.4s, v2.4s, v3.4s
    fmla v9.4s, v4.4s, v5.4s
    fmla v10.4s, v2.4s, v3.4s
    fmla v11.4s, v4.4s, v5.4s
    fmla v12.4s, v2.4s, v3.4s
    fmla v13.4s, v4.4s, v5.4s
    fmla v14.4s, v2.4s, v3.4s
    fmla v15.4s, v4.4s, v5.4s
    fmla v16.4s, v2.4s, v3.4s
    fmla v17.4s, v4.4s, v5.4s
    fmla v18.4s, v2.4s, v3.4s
    fmla v19.4s, v4.4s, v5.4s
    fmla v20.4s, v2.4s, v3.4s
    fmla v21.4s, v4.4s, v5.4s
    fmla v22.4s, v2.4s, v3.4s
    fmla v23.4s, v4.4s, v5.4s
    fmla v24.4s, v2.4s, v3.4s
    fmla v25.4s, v4.4s, v5.4s
    fmla v26.4s, v2.4s, v3.4s
    fmla v27.4s, v4.4s, v5.4s
    subs x0, x0, #1
    b.ne 1b
    ret
ASM
# ── NEON worker (C, 7 threads) ──────────────────────────────────────────────
cat > neon_worker.c << 'CSRC'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
extern void neon_fma_loop(uint64_t iters);
#define CHUNK 50000000ULL
#define OPS_PER_ITER 160.0
#ifndef NT
#define NT 7
#endif
static volatile int g_stop = 0;
static volatile double g_cumops = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static void *worker(void *a) {
    (void)a;
    while (!g_stop) {
        neon_fma_loop(CHUNK);
        pthread_mutex_lock(&g_lock);
        g_cumops += CHUNK * OPS_PER_ITER;
        pthread_mutex_unlock(&g_lock);
    }
    return NULL;
}
int main(int argc, char **argv) {
    int duration = argc > 1 ? atoi(argv[1]) : 10;
    pthread_t t[NT];
    for (int i = 0; i < NT; i++) pthread_create(&t[i], NULL, worker, NULL);
    struct timespec start, now;
    clock_gettime(CLOCK_REALTIME, &start);
    double prev_ops = 0;
    setbuf(stdout, NULL);
    while (1) {
        struct timespec req = {0, 100000000};
        nanosleep(&req, NULL);
        clock_gettime(CLOCK_REALTIME, &now);
        double elapsed = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
        pthread_mutex_lock(&g_lock);
        double ops = g_cumops;
        pthread_mutex_unlock(&g_lock);
        if (ops > prev_ops) {
            uint64_t ns = (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
            printf("%llu %.0f\n", (unsigned long long)ns, ops);
            prev_ops = ops;
        }
        if (elapsed >= duration) break;
    }
    g_stop = 1;
    for (int i = 0; i < NT; i++) pthread_join(t[i], NULL);
    return 0;
}
CSRC
# =============================================================================
# Build
# =============================================================================
echo "Building..."
clang -c -arch arm64 -march=armv9-a+sme2+sve2+sme-lutv2 sme_kern.s -o sme_kern.o 2>/dev/null || { echo "FAIL: sme asm"; exit 1; }
clang -O0 -arch arm64 -DNT=$SME_THREADS -o _sme sme_worker.c sme_kern.o -lpthread 2>/dev/null || { echo "FAIL: sme"; exit 1; }
clang -c -arch arm64 neon_kern.s -o neon_kern.o 2>/dev/null || { echo "FAIL: neon asm"; exit 1; }
clang -O0 -arch arm64 -DNT=$NEON_THREADS -o _neon neon_worker.c neon_kern.o -lpthread 2>/dev/null || { echo "FAIL: neon"; exit 1; }
clang -O2 -arch arm64 -DNT=$BNNS_THREADS -o _bnns bnns_worker.c -framework Accelerate -lpthread 2>/dev/null || { echo "FAIL: bnns"; exit 1; }
clang -O2 -arch arm64 -DNT=$CBLAS_THREADS -o _cblas cblas_worker.c -framework Accelerate -lpthread 2>/dev/null || { echo "FAIL: cblas"; exit 1; }
swiftc -O -o _gpu gpu_worker.swift -framework Metal -framework Foundation 2>/dev/null || { echo "FAIL: gpu"; exit 1; }
echo "  Done."
echo ""
echo "  GPU:  Metal char4 INT8 (1M threads × 10K iters per dispatch)"
echo "  SME:  SMOPA int8 outer product (${SME_THREADS} threads)"
echo "  BNNS: Accelerate INT8 FullyConnected [512,4096]×[4096,1024] (${BNNS_THREADS} threads)"
echo "  CBLAS: Accelerate SGEMM FP32 2048×2048 (${CBLAS_THREADS} threads)"
echo "  NEON: FP32 FMLA (${NEON_THREADS} threads)"
echo ""
# =============================================================================
# Analyzer
# =============================================================================
cat > analyze.py << 'PYANALYZE'
import sys, os, json
def read_log(path):
    entries = []
    if not os.path.exists(path): return entries
    for line in open(path):
        parts = line.strip().split()
        if len(parts) == 2:
            try: entries.append((int(parts[0]), float(parts[1])))
            except ValueError: pass
    return entries
def compute_throughput(logs):
    active = {k: v for k, v in logs.items() if len(v) >= 3}
    if not active:
        print("  No valid log data."); return None
    t_start = max(v[0][0] for v in active.values())
    t_end = min(v[-1][0] for v in active.values())
    if t_end <= t_start:
        print("  No overlapping window found."); return None
    window_sec = (t_end - t_start) / 1e9
    print(f"  Concurrent window: {window_sec:.2f}s")
    print(f"  {'Worker':<12s}  {'Ops/sec':>14s}  {'TOPS':>8s}  {'Heartbeats':>10s}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*8}  {'-'*10}")
    result_tops = {}; total_ops_sec = 0
    for name, entries in sorted(active.items()):
        interior = [(t, o) for t, o in entries if t > t_start and t < t_end]
        if len(interior) < 2:
            print(f"  {name:<12s}  {'(insufficient)':>14s}"); continue
        dt = (interior[-1][0] - interior[0][0]) / 1e9
        dops = interior[-1][1] - interior[0][1]
        if dt <= 0: continue
        ops_sec = dops / dt; tops = ops_sec / 1e12
        total_ops_sec += ops_sec; result_tops[name] = tops
        print(f"  {name:<12s}  {ops_sec:>14,.0f}  {tops:>8.3f}  {len(interior):>10d}")
    print(f"  {'':12s}  {'':>14s}  {'─'*8}")
    print(f"  {'TOTAL':<12s}  {total_ops_sec:>14,.0f}  {total_ops_sec/1e12:>8.3f}")
    return result_tops
if sys.argv[1] == "--summary":
    results = json.loads(sys.argv[2])
    workers = sorted(set(w for r in results for w in r["tops"]))
    hdr = f"  {'Test':<35s}"
    for w in workers: hdr += f"  {w:>8s}"
    hdr += f"  {'TOTAL':>8s}"
    print("\n" + "=" * len(hdr))
    print("  SUMMARY (all values in TOPS)")
    print("=" * len(hdr))
    print(hdr)
    print("  " + "-" * (len(hdr)-2))
    for r in results:
        line = f"  {r['label']:<35s}"; total = 0
        for w in workers:
            v = r["tops"].get(w)
            if v is not None: line += f"  {v:>8.3f}"; total += v
            else: line += f"  {'--':>8s}"
        line += f"  {total:>8.3f}"
        print(line)
    print("=" * len(hdr))
else:
    label = sys.argv[1]; log_dir = sys.argv[2]; log_files = sys.argv[3:]
    print(f"\n━━━ {label} ━━━")
    logs = {}
    for lf in log_files:
        name = os.path.splitext(os.path.basename(lf))[0]
        logs[name] = read_log(os.path.join(log_dir, lf))
    result = compute_throughput(logs)
    if result: print(f"__RESULT__:{json.dumps(result)}")
PYANALYZE
# =============================================================================
# Run tests
# =============================================================================
run_test() {
    local label="$1"; shift
    local workers=("$@")
    rm -f "$WORK"/*.log 2>/dev/null
    local pids=()
    for w in "${workers[@]}"; do
        case "$w" in
            gpu)  ./_gpu  > "$WORK/gpu.log"  2>/dev/null & pids+=($!) ;;
            sme)  ./_sme  "$DURATION" > "$WORK/sme.log"  2>/dev/null & pids+=($!) ;;
            bnns)  ./_bnns  "$DURATION" > "$WORK/bnns.log"  2>/dev/null & pids+=($!) ;;
            cblas) ./_cblas "$DURATION" > "$WORK/cblas.log" 2>/dev/null & pids+=($!) ;;
            neon)  ./_neon  "$DURATION" > "$WORK/neon.log"  2>/dev/null & pids+=($!) ;;
        esac
    done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done
    local logfiles=()
    for w in "${workers[@]}"; do logfiles+=("${w}.log"); done
    local output
    output=$(python3 analyze.py "$label" "$WORK" "${logfiles[@]}")
    echo "$output" | grep -v "^__RESULT__:"
    local json_line
    json_line=$(echo "$output" | grep "^__RESULT__:" | sed 's/^__RESULT__://')
    if [ -n "$json_line" ]; then
        echo "{\"label\":\"$label\",\"tops\":$json_line}" >> "$WORK/all_results.jsonl"
    fi
    echo ""
}
> "$WORK/all_results.jsonl"
echo "Running tests (${DURATION}s each)..."
echo ""
# Solo baselines
run_test "GPU alone"                  gpu
run_test "SME alone (${SME_THREADS}t)"             sme
run_test "BNNS INT8 alone (${BNNS_THREADS}t)"       bnns
run_test "CBLAS SGEMM alone (${CBLAS_THREADS}t)"    cblas
run_test "NEON alone (${NEON_THREADS}t)"            neon
# Pairs
run_test "GPU + SME"                  gpu sme
run_test "GPU + CBLAS"                gpu cblas
run_test "GPU + BNNS"                 gpu bnns
run_test "GPU + NEON"                 gpu neon
run_test "SME + CBLAS"                sme cblas
run_test "SME + BNNS"                 sme bnns
run_test "SME + NEON"                 sme neon
run_test "CBLAS + BNNS"               cblas bnns
run_test "CBLAS + NEON"               cblas neon
run_test "BNNS + NEON"                bnns neon
# Triples
run_test "GPU + SME + CBLAS"          gpu sme cblas
run_test "GPU + SME + NEON"           gpu sme neon
run_test "GPU + CBLAS + NEON"         gpu cblas neon
run_test "GPU + BNNS + NEON"          gpu bnns neon
run_test "GPU + SME + BNNS"           gpu sme bnns
# Everything
run_test "GPU + SME + CBLAS + NEON"   gpu sme cblas neon
run_test "GPU + SME + BNNS + NEON"    gpu sme bnns neon
run_test "GPU + SME + CBLAS + BNNS + NEON" gpu sme cblas bnns neon
# Summary
RESULTS_JSON=$(python3 -c "
import json
results = [json.loads(line) for line in open('$WORK/all_results.jsonl') if line.strip()]
print(json.dumps(results))
" 2>/dev/null)
if [ -n "$RESULTS_JSON" ]; then
    python3 analyze.py --summary "$RESULTS_JSON"
fi
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " All tests complete."
echo "═══════════════════════════════════════════════════════════════"
