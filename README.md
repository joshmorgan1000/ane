# ane

Hand-written ARM SME2 assembly kernels for the Apple Neural Engine (M4).

A demonstration of Apple Silicon's SME (Scalable Matrix Extension) capabilities, featuring a custom bytecode interpreter and a fully functional multi-threaded MNIST training example bypassing standard ML frameworks. All code targets `armv9-a+sme2+sve2+sme-lutv2`.

## Why do you call it the Neural Engine?

Oh boy. This has become an interesting topic, but I should have seen it coming.

### **What exactly is the "Neural Engine"?**

#### Is it separate compute unit on the CPU that is completely undocumented?

Well, yes, it is a "separate unit", but no it is not "completely undocumented".

#### Is it a CoreML API/framework?

Yes. A lot of what is referred to as "the Neural Engine" is in fact software, which I will get into in a bit. But this is not the whole story.

#### Is it something that can only be accessed through CoreML?

That depends on your definition of "Neural Engine". If part of that includes the software stack, then yes. If you're just referring to the hardware, then no, as this repository demonstrates.

### The Discovery

I recently gained access to some ASUS GX10's, which incidentally have ARM CPU's in them along with unified memory... but with a CUDA GPU. Familiar with Accelerate, I was looking for an equivalent library, which led me to ARM's website, where I found out about the **SME2** extension in ARMv9.2. So I had Claude write some assembly code with every intention on running on a GX10 to see what kind of numbers I could get.

Instead, Claude built the tests on my M4. Which ran. Cool.

So I ran a full probe (well, mostly full) which can be [found here](probes/probe.sh).

I did some performance benchmarks and found:
- `bfloat16` `fmopa` operations were the exact same speed as 32-bit float `fmopa` operations. This shouldn't be the case since they take up half the memory bandwidth?
- Many of the *basic arithmetic* operations for the `za` tile (`add`, `mul`, `sub`, etc) didn't work. Weird, but oh well, `fmla` works which does the same thing.
- None of the `FP8` operations worked. Zero.
- The `int8` and `uint8` MOPA operations *did* work.
- The four isolated 4096-byte `za` arrays are quite a lot of CPU-integrated compute to keep hidden, without even a mention. I haven't seen any of thi in any of the marketing material I've read. Seems kind of strange that they wouldn't advertise this CPU unit that is 1/4 of the processing power that the GPU possesses, except without the shader dispatch tax. Unless....

That's when I started to get deja-vu. I had heard of something that was some sort of fast 8-bit matrix multiply unit, wasn't really optimized for FP16, and had some weird quirks. That's when it hit me. The four separate 4096-byte `za` matrix arrays I was driving across the CPU cores were functioning identically to Apple's software stacks that target the Neural Engine.

### Further Testing

> **Note:** I currently have an open Apple Support ticket investigating performance discrepancies I'm seeing where my M4 actually outperforms my newer M5 chip in certain SME throughput benchmarks. If it weren't for the deeply integrated hardware probes in this project, I doubt I would have ever noticed the discrepancy. I'll update this repository when I hear back from their hardware engineering team.

We know that Apple advertises the "Neural Engine" as being able to do 38 TOPS of compute. We ran a comprehensive benchmark measuring four independent compute paths — GPU (Metal `char4` INT8), CPU SME (`smopa` INT8), Apple's BNNS INT8 (`BNNSFilterCreateLayerFullyConnected` with `BNNSDataTypeInt8`), and NEON FP32 FMA — both in isolation and in every combination, for 10 seconds each, with heartbeat logging and proven-concurrent overlap windows.

Results on Apple M4 Max (selected from full table):

| Test | GPU | SME | BNNS INT8 | NEON | Total |
|---|---|---|---|---|---|
| GPU alone | 37.4 | | | | 37.4 |
| SME alone (4 threads) | | 8.5 | | | 8.5 |
| BNNS INT8 alone (4 threads) | | | 4.2 | | 4.2 |
| GPU + SME | 37.4 | 8.5 | | | **45.9** |
| GPU + BNNS | 37.3 | | 4.0 | | 41.3 |
| **SME + BNNS** | | **5.2** | **1.6** | | **6.9** |
| GPU + SME + BNNS | 37.0 | 5.2 | 1.6 | | 43.8 |
| GPU + SME + BNNS + NEON | 35.8 | 4.9 | 1.7 | 0.7 | 43.2 |

The critical result is **SME + BNNS**: when both run simultaneously, SME drops from 8.5 to 5.2 TOPS (39% drop) and BNNS drops from 4.2 to 1.6 TOPS (62% drop). They are fighting for the same hardware. Meanwhile, the GPU is completely unaffected by any CPU-side workload.

This proves that Apple's BNNS INT8 library — the same library that CoreML dispatches to for "Neural Engine" inference — uses the SME `za` matrix tiles. There is no separate coprocessor.

Additionally, the raw SME `smopa` kernels in this repository achieve **8.5 TOPS** on INT8 operations — more than **double** the 4.2 TOPS that Apple's own BNNS framework achieves on the same hardware. Direct access to the hardware is significantly more efficient than going through Apple's software stack.

Run the tests yourself:

```bash
bash tests/run_full_throughput_tests.sh
```

The test script is fully self-contained — it generates all source code inline, builds in a temp directory, and cleans up after itself. Each worker process self-times with nanosecond-precision epoch timestamps. The analyzer finds the proven-concurrent window (where all processes in a test are confirmed running) and computes throughput from interior heartbeats only, discarding startup and shutdown artifacts.

## Conclusion

The "Neural Engine" is the ARM SME2 `za` matrix tiles plus a software scheduling layer. Apple's "38 TOPS" comes from the GPU (~37 TOPS via native Metal `char4` INT8 vectors) plus the CPU SME tiles (~8.5 TOPS via `smopa`). BNNS and CoreML dispatch to the same SME hardware, achieving ~4.2 TOPS through their software abstraction.

The evidence:
1. **SME and BNNS INT8 contend for the same hardware.** Running them simultaneously drops both — total throughput is less than either alone. If BNNS used separate silicon, both would maintain full speed.
2. **GPU is fully independent.** No throughput drop when paired with any CPU workload — it's genuinely separate hardware.
3. **Same limitations.** The "ANE" has the same data type support as SME (no FP8, same int8/bf16/fp32 types).
4. **Direct SME access outperforms the "ANE" by 2x.** Raw `smopa` at 8.5 TOPS vs BNNS at 4.2 TOPS on the same hardware, same data types.
5. **Same official documentation:** none.

## So What?

For 32-bit floating point workloads, `cblas_sgemm` from Accelerate is likely your best bet — it uses `fmopa` under the hood and Apple is very good at scheduling it. Language models are familiar with it so you don't have to argue with them about how to use it properly.

For 8-bit integer workloads however, this repository's `smopa` kernels achieve over twice the throughput of Apple's own BNNS INT8 path. The `luti4` instructions can be used for efficient quantized inference or training. Working examples are included (though not fully implemented yet).

## Is this production ready?

Absolutely not. Test and use at your own risk. I will do my best to make them easier to use and to call, but since we are dealing directly with machine code that is undocumented I can't make any guarantees.

## Requirements

- Apple M4 (or any ARM processor with SME2)
- CMake 3.19+
- C++20 compiler (Apple Clang recommended)
- Node.js (20+) and npm (for generating the UI Dashboard)
- Python 3.10+ (with `torch` and `torchvision` installed if you wish to run the PyTorch benchmark comparisons)

## Building the Live Hardware UI Dashboard

This repository features a React-based UI single-file dashboard that dynamically compiles and runs probing payloads to uncover your CPU's hardware matrix limits. 

It also runs a live MNIST training benchmark comparing our custom SME C++ engine against PyTorch CPU (Eager) to demonstrate the raw throughput (samples/sec) achievable by bypassing standard framework overhead.

To build and open the dashboard:

```bash
./dashboard.sh
```

This will automatically:
1. Run local Apple Clang compilation and benchmark passes against your SME hardware.
2. Build the Vite React application down into a single `dist/index.html` static file.
3. Serve it directly in your browser.

## Quick Start: Training MNIST with SME

For a fully working implementation that showcases how to use the SME bytecode interpreter to build a multi-threaded matrix multiplication and training loop from scratch, check out the [`tests/test_mnist.cpp`](tests/test_mnist.cpp) file. 

To build and run the test suite manually:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja
cmake --build .
./bin/test_mnist
```

## Adding `ane` To Your Project

`ane` is a **header-only** C++20 library. The entire machine-code generation and interpretation process happens inline.

You can install it globally:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
sudo cmake --install .
```

And link it via CMake:

```cmake
find_package(ane REQUIRED)
target_link_libraries(my_app PRIVATE ane::ane)
```

## Usage (Bytecode Dispatch API)

All neural engine operations are now encoded as custom `ane::Op` bytecodes which are passed to the hardware-accelerated interpreter via `ane::dispatch`.

Here is an example showing a fused INT8 8-bit quantized matrix multiplication pass:

```cpp
#include <ane/ane.hpp>
#include <cstring>

int main() {
    int batch_size = 50;
    int HIDDEN_DIM = 128;
    int INPUT_DIM = 784;

    // All memory should be 64-byte aligned and padded to multiples of 16 for SME vector bounds
    float* hidden = ... // allocation
    float* xi     = ... // input data
    float* W1     = ... // model weights

    // Forward Pass: Matrix multiply (dense fused 8-bit quantization)
    memset(hidden, 0, batch_size * HIDDEN_DIM * sizeof(float));

    ane::dispatch(
        ane::Op::dense_fused_i8, 
        batch_size,     // Output rows (M)
        HIDDEN_DIM,     // Output cols (N)
        INPUT_DIM,      // Inner dim (K)
        1.0f,           // Float scaling
        true,           // Apply Fused-ReLU Activation
        xi,             // Matrix A (batch_size × INPUT_DIM)
        W1,             // Matrix B (INPUT_DIM × HIDDEN_DIM)
        hidden          // Matrix C (Output)
    );

    return 0;
}
```

## AI Assistants & Contributors

Thanks to Anthropic's Claude models (Haiku, Sonnet, and Opus 4.5 and 4.6 were all used at some point).
Thanks to Gemini 3.1 pro for assembling the React dashboards, benchmark integration, and keeping things clean!

## License

MIT. See [LICENSE](LICENSE) for details.