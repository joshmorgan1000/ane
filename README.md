# ane

Bare-metal ARM SME2 bytecode interpreter for Apple Silicon (M4/M5).

A C++20 static library providing 28 custom opcodes executed by a hand-written ARM64 bytecode interpreter written in assembly. All matrix operations run directly on the SME2 `za` tiles — the same hardware that Apple's "Neural Engine" software stack targets — bypassing CoreML and BNNS entirely. For the full story on that discovery, see [docs/DISCOVERY.md](docs/DISCOVERY.md).

---

## Benchmarks

Raw SME `smopa` achieves **2x the throughput** of Apple's own BNNS INT8 path on the same hardware:

| Test | GPU | SME | BNNS INT8 | Total |
|---|---|---|---|---|
| SME alone | | 8.5 TOPS | | 8.5 |
| BNNS alone | | | 4.2 TOPS | 4.2 |
| **SME + BNNS** | | **5.2** | **1.6** | **6.9** |
| GPU + SME | 37.4 | 8.5 | | 45.9 |

When SME and BNNS run simultaneously, both drop — proving they share hardware. The GPU is unaffected. [Full results and methodology](docs/DISCOVERY.md#contention-benchmark).

```bash
bash tests/run_full_throughput_tests.sh    # Run it yourself
```

---

## Bytecode Interpreter?

The `smstart` and `smstop` come with a little bit of overhead. They call it streaming for a reason, it is designed so that several streaming operations happen within a single streaming session. How many exactly? I'm not sure yet. Right now the bytecode programs are limited to 1 loop that can only repeat 255 times. We'll see how that goes.

---

## Quick Start

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja
cmake --build .
./bin/test_mnist
```

This builds the library and runs a multi-threaded MNIST training demo (784 -> 128 -> 10, Hogwild SGD) with all compute dispatched through the SME bytecode interpreter. See [`tests/test_mnist.cpp`](tests/test_mnist.cpp) for the full implementation.

---

## Installation

`ane` is a C++20 static library. The header (`ane.hpp`) provides the dispatch template and helper types; the assembly interpreter (`bytecode_interpreter.s`) is compiled into `libane.a`.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
sudo cmake --install .
```

Then link via CMake:

```cmake
find_package(ane REQUIRED)
target_link_libraries(my_app PRIVATE ane::ane)
```

---

## Usage

All operations are encoded as `ane::Op` bytecodes and executed by the assembly interpreter via `ane::dispatch`:

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
    float* W1     = ... // model parameters

    // Forward pass: dense matmul with fused 8-bit quantization and ReLU
    memset(hidden, 0, batch_size * HIDDEN_DIM * sizeof(float));

    ane::dispatch(
        ane::Op::dense_fused_i8,
        batch_size,     // Output rows (M)
        HIDDEN_DIM,     // Output cols (N)
        INPUT_DIM,      // Inner dim (K)
        1.0f,           // Float scaling
        true,           // Fuse ReLU
        xi,             // Matrix A (batch_size x INPUT_DIM)
        W1,             // Matrix B (INPUT_DIM x HIDDEN_DIM)
        hidden          // Matrix C (Output)
    );

    return 0;
}
```

See [docs/API.md](docs/API.md) for the full opcode reference.

---

## Dashboard

An interactive React dashboard that runs live hardware probes and benchmarks against your SME hardware:

```bash
./dashboard.sh
```

See [docs/TOOLS.md](docs/TOOLS.md) for details on the probe scripts, throughput benchmarks, and dashboard development.

---

## Requirements

- Apple M4 or M5 (or any ARM processor with SME2)
- CMake 3.19+
- C++20 compiler (Apple Clang recommended)
- Node.js 20+ and npm (optional, for the dashboard)
- Python 3.10+ with `torch` and `torchvision` (optional, for PyTorch comparisons)

---

## Documentation

| | |
|---|---|
| [docs/API.md](docs/API.md) | Opcode reference, dispatch API, helper types |
| [docs/DISCOVERY.md](docs/DISCOVERY.md) | How we found that SME2 is the hardware behind Apple's "Neural Engine" |
| [docs/TOOLS.md](docs/TOOLS.md) | Hardware probes, throughput benchmarks, and dashboard |

---

## Contributors

Thanks to Anthropic's Claude models (Haiku, Sonnet, and Opus 4.5 and 4.6 were all used at some point).
Thanks to Gemini 3.1 Pro for assembling the React dashboards, benchmark integration, and keeping things clean!

## License

MIT. See [LICENSE](LICENSE) for details.
