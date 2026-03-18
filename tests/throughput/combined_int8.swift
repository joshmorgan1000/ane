/**
 * combined_int8.swift — Simultaneous GPU + CPU SME INT8 throughput
 *
 * Runs GPU Metal char4 peak ALU test AND CPU SME SMOPA concurrently,
 * measures combined wall time, reports individual + total TOPS.
 *
 * Goal: beat Apple's "38 TOPS" marketing by running both at once.
 *
 * Build:
 *   # First ensure smopa_4way_kernel.o exists (from earlier benchmark)
 *   swiftc -O -o combined_int8 combined_int8.swift smopa_4way_kernel.o \
 *       -framework Metal -framework Foundation -import-objc-header bridge.h
 *
 * Or simpler: use system() to launch the SME benchmark in parallel.
 */
import Foundation
import Metal

// ============================================================================
// GPU Setup
// ============================================================================
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
let queue = device.makeCommandQueue()!

let shaderPath = (ProcessInfo.processInfo.environment["PWD"] ?? ".") + "/int8_peak.metal"
let src = try! String(contentsOfFile: shaderPath, encoding: .utf8)
let lib = try! device.makeLibrary(source: src, options: nil)
let pipe = try! device.makeComputePipelineState(function: lib.makeFunction(name: "int8_peak")!)

print("Device: \(device.name)")
print("=============================================================")
print(" Combined GPU + CPU SME INT8 Throughput Test — Apple M4 Max")
print("=============================================================\n")

let nThreads = 1024 * 1024
let innerIters: UInt32 = 10_000
let gpuOpsPerThread = Double(innerIters) * 32.0 * 2.0

let outBuf = device.makeBuffer(length: nThreads * 4, options: .storageModeShared)!
let seedBuf = device.makeBuffer(length: 256 * 4, options: .storageModeShared)!
let sp = seedBuf.contents().bindMemory(to: Int8.self, capacity: 1024)
for i in 0..<1024 { sp[i] = Int8(truncatingIfNeeded: i &* 7 &+ 3) }
var iters = innerIters
let tpg = MTLSize(width: 256, height: 1, depth: 1)
let grid = MTLSize(width: nThreads, height: 1, depth: 1)

// GPU warmup
for _ in 0..<2 {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(outBuf, offset: 0, index: 0)
    enc.setBuffer(seedBuf, offset: 0, index: 1)
    enc.setBytes(&iters, length: 4, index: 2)
    enc.dispatchThreads(grid, threadsPerThreadgroup: tpg)
    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
}

// ============================================================================
// CPU SME: launch 4 threads running SMOPA via the assembly kernel
// We call it via system() since linking the .o with Swift bridging is complex
// ============================================================================
let gpuRuns = 10  // Balanced: GPU ~0.18s matches SME ~0.18s
let smeIters = 20_000_000  // matches our earlier benchmark
let smeThreads = 4
let smeOpsPerThread = Double(smeIters) * 8.0 * 2048.0  // 8 SMOPA × 2048 ops each

// Build a tiny C program that runs SME in a thread
let smeProg = """
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
extern void smopa_4way_loop(uint64_t iters);
#define ITERS \(smeIters)ULL
#define NTHREADS \(smeThreads)
static void *worker(void *arg) { (void)arg; smopa_4way_loop(ITERS); return NULL; }
int main() {
    pthread_t t[\(smeThreads)];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < NTHREADS; i++) pthread_create(&t[i], NULL, worker, NULL);
    for (int i = 0; i < NTHREADS; i++) pthread_join(t[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double s = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    double ops = (double)NTHREADS * ITERS * 8.0 * 2048.0;
    printf("%.6f %.3f\\n", s, ops/s/1e12);
    return 0;
}
"""
let smeC = (ProcessInfo.processInfo.environment["PWD"] ?? ".") + "/combined_sme_worker.c"
try! smeProg.write(toFile: smeC, atomically: true, encoding: .utf8)

// Compile the SME worker (links against pre-built smopa_4way_kernel.o)
let pwd = ProcessInfo.processInfo.environment["PWD"] ?? "."
let buildCmd = "clang -O0 -arch arm64 -o \(pwd)/combined_sme_worker \(pwd)/combined_sme_worker.c \(pwd)/smopa_4way_kernel.o -lpthread 2>&1"
let buildResult = Process()
buildResult.executableURL = URL(fileURLWithPath: "/bin/zsh")
buildResult.arguments = ["-c", buildCmd]
try! buildResult.run(); buildResult.waitUntilExit()
if buildResult.terminationStatus != 0 {
    print("Failed to build SME worker")
    exit(1)
}

// ============================================================================
// Run both simultaneously
// ============================================================================
print("Starting GPU (5 dispatches, 1M threads × 10K iters each)")
print("Starting CPU SME (4 threads × 20M SMOPA iterations)")
print("Running simultaneously...\n")

// Launch SME process
let smeProc = Process()
smeProc.executableURL = URL(fileURLWithPath: "\(pwd)/combined_sme_worker")
let smePipe = Pipe()
smeProc.standardOutput = smePipe
try! smeProc.run()

// Simultaneously run GPU
var lastCB: MTLCommandBuffer?
let wallStart = CFAbsoluteTimeGetCurrent()
for _ in 0..<gpuRuns {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(outBuf, offset: 0, index: 0)
    enc.setBuffer(seedBuf, offset: 0, index: 1)
    enc.setBytes(&iters, length: 4, index: 2)
    enc.dispatchThreads(grid, threadsPerThreadgroup: tpg)
    enc.endEncoding()
    cb.commit()
    lastCB = cb
}
lastCB!.waitUntilCompleted()
let gpuDone = CFAbsoluteTimeGetCurrent()

// Wait for SME to finish
smeProc.waitUntilExit()
let allDone = CFAbsoluteTimeGetCurrent()

let gpuWall = gpuDone - wallStart
let totalWall = allDone - wallStart

// Parse SME output: "time tops"
let smeOutput = String(data: smePipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)!.trimmingCharacters(in: .whitespacesAndNewlines)
let smeParts = smeOutput.split(separator: " ")
let smeTime = Double(smeParts[0])!
let smeTops = Double(smeParts[1])!

let gpuTotalOps = Double(nThreads) * gpuOpsPerThread * Double(gpuRuns)
let gpuTops = gpuTotalOps / gpuWall / 1e12

// Combined: both ran during the max(gpuWall, smeTime) window
let overlapWindow = max(gpuWall, smeTime)
let gpuOpsInWindow = gpuTotalOps  // GPU finished in gpuWall
let smeOpsInWindow = Double(smeThreads) * smeOpsPerThread  // SME finished in smeTime
let combinedTops = (gpuOpsInWindow + smeOpsInWindow) / overlapWindow / 1e12

print("Results:")
print("─────────────────────────────────────────────")
print(String(format: "  GPU INT8:         %.2f TOPS  (%.3f s)", gpuTops, gpuWall))
print(String(format: "  CPU SME INT8:     %.2f TOPS  (%.3f s)", smeTops, smeTime))
print(String(format: "  Overlap window:   %.3f s", overlapWindow))
print(String(format: "  "))
print(String(format: "  COMBINED INT8:    %.2f TOPS", combinedTops))
print("─────────────────────────────────────────────")
print(String(format: "  Apple claims:     38 TOPS"))
print(String(format: "  We measured:      %.1f TOPS (GPU + CPU simultaneously)", combinedTops))
