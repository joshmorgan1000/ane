-/**
 * triple_threat.swift — All-Combinations ANE Existence Test
 *
 * All concurrent workloads run as separate processes to avoid Swift data races.
 * CoreML → coreml_worker, SME → sme_worker_gen, GPU → in-process Metal.
 *
 * Build:
 *   swiftc -O -o triple_threat triple_threat.swift -framework Metal -framework Foundation
 *   swiftc -O -o coreml_worker coreml_worker.swift -framework CoreML -framework Foundation
 */
import Foundation
import Metal

let GPU_RUNS = 100
let GPU_THREADS = 1024 * 1024
let GPU_INNER_ITERS: UInt32 = 10_000
let GPU_OPS_PER_RUN = Double(GPU_THREADS) * Double(GPU_INNER_ITERS) * 32.0 * 2.0
let SME_TOTAL_OPS = 4.0 * 100_000_000.0 * 8.0 * 2048.0
let pwd = ProcessInfo.processInfo.environment["PWD"] ?? "."

print("=================================================================")
print(" ALL-COMBINATIONS ANE TEST — Apple M4 Max")
print("=================================================================\n")

// ── GPU Setup ────────────────────────────────────────────────────────────────
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let src = try! String(contentsOfFile: pwd + "/int8_peak.metal", encoding: .utf8)
let lib = try! device.makeLibrary(source: src, options: nil)
let pipe = try! device.makeComputePipelineState(function: lib.makeFunction(name: "int8_peak")!)
let outBuf = device.makeBuffer(length: GPU_THREADS * 4, options: .storageModeShared)!
let seedBuf = device.makeBuffer(length: 256 * 4, options: .storageModeShared)!
let sp = seedBuf.contents().bindMemory(to: Int8.self, capacity: 1024)
for i in 0..<1024 { sp[i] = Int8(truncatingIfNeeded: i &* 7 &+ 3) }
var gpuIters = GPU_INNER_ITERS
let tpg = MTLSize(width: 256, height: 1, depth: 1)
let grid = MTLSize(width: GPU_THREADS, height: 1, depth: 1)
for _ in 0..<3 {
    let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe); enc.setBuffer(outBuf, offset:0, index:0)
    enc.setBuffer(seedBuf, offset:0, index:1); enc.setBytes(&gpuIters, length:4, index:2)
    enc.dispatchThreads(grid, threadsPerThreadgroup: tpg)
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
}

func runGPU() -> (tops: Double, time: Double) {
    var last: MTLCommandBuffer?
    let t0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<GPU_RUNS {
        let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe); enc.setBuffer(outBuf, offset:0, index:0)
        enc.setBuffer(seedBuf, offset:0, index:1); enc.setBytes(&gpuIters, length:4, index:2)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tpg)
        enc.endEncoding(); cb.commit(); last = cb
    }
    last!.waitUntilCompleted()
    let t = CFAbsoluteTimeGetCurrent() - t0
    return (Double(GPU_RUNS) * GPU_OPS_PER_RUN / t / 1e12, t)
}

// ── Subprocess helpers ───────────────────────────────────────────────────────
func launchSME() -> Process {
    let p = Process(); p.executableURL = URL(fileURLWithPath: "\(pwd)/sme_worker_gen")
    let o = Pipe(); p.standardOutput = o; try! p.run(); return p
}
func collectSME(_ p: Process) -> (tops: Double, time: Double) {
    p.waitUntilExit()
    let pipe = p.standardOutput as! Pipe
    let s = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)!
        .trimmingCharacters(in: .whitespacesAndNewlines).split(separator: " ")
    return (Double(s[1])!, Double(s[0])!)
}
func launchCoreML(duration: Double) -> Process {
    let p = Process(); p.executableURL = URL(fileURLWithPath: "\(pwd)/coreml_worker")
    p.arguments = [String(duration)]
    let o = Pipe(); p.standardOutput = o; try! p.run(); return p
}
func collectCoreML(_ p: Process) -> (count: Int, rate: Double) {
    p.waitUntilExit()
    let pipe = p.standardOutput as! Pipe
    let s = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)!
        .trimmingCharacters(in: .whitespacesAndNewlines).split(separator: " ")
    return (Int(s[0])!, Double(s[1])!)
}

// ── Tests ────────────────────────────────────────────────────────────────────
struct R { let name: String; let gpu: Double?; let sme: Double?; let cmRate: Double?; let time: Double }
var results: [R] = []

print("[1/7] CoreML alone (3s sustained)...")
let cm1p = launchCoreML(duration: 3.0); let cm1 = collectCoreML(cm1p)
results.append(R(name: "CoreML alone", gpu: nil, sme: nil, cmRate: cm1.rate, time: 3.0))
print(String(format: "  %d inferences, %.0f infer/s", cm1.count, cm1.rate))

print("\n[2/7] GPU alone...")
let g2 = runGPU()
results.append(R(name: "GPU alone", gpu: g2.tops, sme: nil, cmRate: nil, time: g2.time))
print(String(format: "  %.1f TOPS (%.2fs)", g2.tops, g2.time))

print("\n[3/7] CPU SME alone...")
let s3 = collectSME(launchSME())
results.append(R(name: "SME alone", gpu: nil, sme: s3.tops, cmRate: nil, time: s3.time))
print(String(format: "  %.1f TOPS (%.2fs)", s3.tops, s3.time))

print("\n[4/7] GPU + CPU SME...")
let s4p = launchSME(); let g4 = runGPU(); let s4 = collectSME(s4p)
let w4 = max(g4.time, s4.time); let c4 = (Double(GPU_RUNS)*GPU_OPS_PER_RUN + SME_TOTAL_OPS)/w4/1e12
results.append(R(name: "GPU + SME", gpu: g4.tops, sme: s4.tops, cmRate: nil, time: w4))
print(String(format: "  GPU: %.1f  SME: %.1f  Combined: %.1f TOPS", g4.tops, s4.tops, c4))

print("\n[5/7] CoreML + GPU...")
let cm5p = launchCoreML(duration: 5.0); Thread.sleep(forTimeInterval: 0.5)  // Let CoreML warm
let g5 = runGPU(); let cm5 = collectCoreML(cm5p)
results.append(R(name: "CoreML + GPU", gpu: g5.tops, sme: nil, cmRate: cm5.rate, time: g5.time))
print(String(format: "  GPU: %.1f TOPS  CoreML: %.0f infer/s", g5.tops, cm5.rate))

print("\n[6/7] CoreML + CPU SME...")
let cm6p = launchCoreML(duration: 5.0); Thread.sleep(forTimeInterval: 0.5)
let s6p = launchSME(); let s6 = collectSME(s6p); let cm6 = collectCoreML(cm6p)
results.append(R(name: "CoreML + SME", gpu: nil, sme: s6.tops, cmRate: cm6.rate, time: s6.time))
print(String(format: "  SME: %.1f TOPS  CoreML: %.0f infer/s", s6.tops, cm6.rate))

print("\n[7/7] GPU + CPU SME + CoreML (TRIPLE THREAT)...")
let cm7p = launchCoreML(duration: 5.0); Thread.sleep(forTimeInterval: 0.5)
let s7p = launchSME(); let g7 = runGPU(); let s7 = collectSME(s7p); let cm7 = collectCoreML(cm7p)
let w7 = max(g7.time, s7.time); let c7 = (Double(GPU_RUNS)*GPU_OPS_PER_RUN + SME_TOTAL_OPS)/w7/1e12
results.append(R(name: "ALL THREE", gpu: g7.tops, sme: s7.tops, cmRate: cm7.rate, time: w7))
print(String(format: "  GPU: %.1f  SME: %.1f  CoreML: %.0f/s  Combined: %.1f TOPS", g7.tops, s7.tops, cm7.rate, c7))

// ── Summary ──────────────────────────────────────────────────────────────────
print("\n===================================================================")
print(String(format: " %-20s  %8s  %8s  %10s  %7s", "Test", "GPU", "SME", "CoreML", "Time"))
print(String(format: " %-20s  %8s  %8s  %10s  %7s", "", "TOPS", "TOPS", "infer/s", ""))
print(" -------------------------------------------------------------------")
for r in results {
    let g = r.gpu != nil ? String(format: "%6.1f", r.gpu!) : "  --  "
    let s = r.sme != nil ? String(format: "%6.1f", r.sme!) : "  --  "
    let c = r.cmRate != nil ? String(format: "%8.0f", r.cmRate!) : "   --   "
    print(String(format: " %-20s  %8s  %8s  %10s  %5.2fs", r.name, g, s, c, r.time))
}
print("===================================================================\n")
let cmBase = results[0].cmRate!; let gpuBase = results[1].gpu!; let smeBase = results[2].sme!
print(String(format: " CoreML baseline:      %.0f infer/s", cmBase))
print(String(format: " CoreML + GPU:         %.0f infer/s  (%+.0f%%)", results[4].cmRate!, (results[4].cmRate!/cmBase - 1)*100))
print(String(format: " CoreML + SME:         %.0f infer/s  (%+.0f%%)", results[5].cmRate!, (results[5].cmRate!/cmBase - 1)*100))
print(String(format: " CoreML + ALL:         %.0f infer/s  (%+.0f%%)", results[6].cmRate!, (results[6].cmRate!/cmBase - 1)*100))
print(String(format: " GPU baseline:         %.1f TOPS", gpuBase))
print(String(format: " GPU + CoreML:         %.1f TOPS     (%+.0f%%)", results[4].gpu!, (results[4].gpu!/gpuBase - 1)*100))
print(String(format: " SME baseline:         %.1f TOPS", smeBase))
print(String(format: " SME + CoreML:         %.1f TOPS     (%+.0f%%)", results[5].sme!, (results[5].sme!/smeBase - 1)*100))
print("\n===================================================================")
