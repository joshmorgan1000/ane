/**
 * metal_int8_peak.swift — Host harness for INT8 peak throughput measurement
 *
 * Dispatches 1M threads on the GPU, each running 10K iterations of 8 int4 MADs.
 * Pipelined command buffer submission (5 runs, wait only on last) to minimize
 * CPU-side overhead and keep the GPU fully saturated.
 *
 * Result: ~37 TOPS on Apple M4 Max GPU with native char4 vector arithmetic.
 *
 * Build:  swiftc -O -o metal_int8_peak metal_int8_peak.swift -framework Metal -framework Foundation
 * Run:    ./metal_int8_peak
 */
import Foundation
import Metal

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
let queue = device.makeCommandQueue()!

let shaderPath = (ProcessInfo.processInfo.environment["PWD"] ?? ".") + "/int8_peak.metal"
let src = try! String(contentsOfFile: shaderPath, encoding: .utf8)
let lib = try! device.makeLibrary(source: src, options: nil)
let pipe = try! device.makeComputePipelineState(function: lib.makeFunction(name: "int8_peak")!)

print("Device: \(device.name)")
print("Max threads/threadgroup: \(pipe.maxTotalThreadsPerThreadgroup)")
print("SIMD width: \(pipe.threadExecutionWidth)")

let nThreads = 1024 * 1024          // 1M GPU threads
let innerIters: UInt32 = 10_000     // 10K MAD iterations per thread
let opsPerThread = Double(innerIters) * 32.0 * 2.0  // 8 int4 MADs × 4 lanes × 2 ops(mul+acc)

let outBuf = device.makeBuffer(length: nThreads * 4, options: .storageModeShared)!

// 256 char4 seeds — data-dependent to prevent compiler optimization
let seedBuf = device.makeBuffer(length: 256 * 4, options: .storageModeShared)!
let sp = seedBuf.contents().bindMemory(to: Int8.self, capacity: 1024)
for i in 0..<1024 { sp[i] = Int8(truncatingIfNeeded: i &* 7 &+ 3) }

var iters = innerIters
let tpg = MTLSize(width: 256, height: 1, depth: 1)
let grid = MTLSize(width: nThreads, height: 1, depth: 1)

// Warmup: 2 full dispatches to get GPU clocks up
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

// Benchmark: 5 pipelined submissions, wait only on last
let runs = 5
var lastCB: MTLCommandBuffer?
let t0 = CFAbsoluteTimeGetCurrent()
for _ in 0..<runs {
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
let wall = CFAbsoluteTimeGetCurrent() - t0

let totalOps = Double(nThreads) * opsPerThread * Double(runs)
let tops = totalOps / wall / 1e12

print(String(format: "\n%dK threads × %d inner iters × %d runs", nThreads/1024, innerIters, runs))
print(String(format: "Wall time: %.3f s", wall))
print(String(format: "Peak GPU INT8: %.2f TOPS", tops))
print(String(format: "\nCPU SME INT8:  ~8.4  TOPS (SMOPA, 4 threads)"))
print(String(format: "GPU INT8:      %.1f TOPS (this test)", tops))
print(String(format: "Combined:      %.1f TOPS (theoretical simultaneous)", tops + 8.4))
