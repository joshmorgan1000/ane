# Comprehensive Test Plan

Before wiring anything into llama.cpp, every kernel and every code path needs to be verified independently. The tests are organized from simple to complex — each layer builds confidence for the next.

## Level 1: Composable Primitives

These are the atomic building blocks. If any of these fail, nothing built on top of them can be trusted.

### 1.1 Data Movement
- [ ] `load` / `store`: round-trip 64 bytes through v0 *(existing test_program_demo #1)*
- [ ] `mov_zreg`: copy between all register pairs (v2→v31, v31→v2, v0→v15, etc.)
- [ ] `load_param` / `store_param`: read/write through all 8 param slots (0-7)
- [ ] `advance_param`: verify pointer advances by exactly 64 bytes each call
- [ ] `set_param`: verify pointer is stored correctly (load back and compare)

### 1.2 Control Flow
- [ ] `loop_begin` / `loop_end`: verify exact iteration count (1, 2, 127, 255)
- [ ] Loop with pointer advance: verify 4th iteration reads data at offset 192 (3×64)
- [ ] Loop counter = 1: body executes exactly once
- [ ] Loop counter = 255: maximum iteration count works

### 1.3 Arithmetic (float32)
- [ ] `fadd_zreg`: [1,2,3,...16] + [100,100,...] = [101,102,...116]
- [ ] `fsub_zreg`: [100,100,...] - [1,2,...16] = [99,98,...84]
- [ ] `fmul_zreg`: [1,2,3,...16] × [2,2,...] = [2,4,6,...32]
- [ ] `fmla_zreg`: accumulate 4 iterations of [1,1,...] × [1,1,...] = [4,4,...4]
- [ ] `fmla_zreg` with different dst/src: verify accumulator is read AND written
- [ ] Arithmetic where dst == src1: `x = x + y` in-place
- [ ] Arithmetic where dst == src2: `x = y + x`
- [ ] Arithmetic where all three are the same: `x = x + x` (doubling)

### 1.4 Bitwise
- [ ] `and_zreg`: 0xFF00FF00 AND 0x0F0F0F0F = 0x0F000F00
- [ ] `orr_zreg`: 0xF0 OR 0x0F = 0xFF
- [ ] `eor_zreg`: 0xFF XOR 0xFF = 0x00, 0xFF XOR 0x00 = 0xFF
- [ ] `not_zreg`: NOT 0x00 = 0xFF (all bits flipped)
- [ ] `lsl_zreg`: shift left by 1, 4, 8, 32, 63
- [ ] `lsr_zreg`: shift right unsigned
- [ ] `asr_zreg`: shift right with sign extension (test negative values)

### 1.5 Clamp/Max/Min
- [ ] `fclamp_zreg` flags=3: clamp [-10..10] to [-3, 3]
- [ ] `fclamp_zreg` flags=1: max(x, 0) — ReLU behavior
- [ ] `fclamp_zreg` flags=2: min(x, 5)
- [ ] Clamp with lo > hi: verify defined behavior
- [ ] Clamp with NaN inputs: verify no crash

### 1.6 Scalar Operations
- [ ] `broadcast_scalar_zreg`: fill with 3.14, verify all 16 lanes
- [ ] `fscale_zreg`: [1,2,...16] × 0.5 = [0.5, 1.0, ... 8.0]
- [ ] `fscale_zreg` with scale = 0.0: all zeros
- [ ] `fscale_zreg` with scale = -1.0: negation

### 1.7 Horizontal Reduce
- [ ] `faddv_zreg`: [1,1,...1] → [16,16,...16] (broadcast sum of 16 ones)
- [ ] `faddv_zreg`: [1,2,3,...16] → [136,136,...136] (sum 1..16 = 136)
- [ ] `faddv_zreg` with negative values: [1,-1,1,-1,...] → [0,0,...0]

### 1.8 Reciprocal Sqrt
- [ ] `frsqrt_zreg`: rsqrt(4.0) ≈ 0.5 (within 1e-5)
- [ ] `frsqrt_zreg`: rsqrt(1.0) ≈ 1.0
- [ ] `frsqrt_zreg`: rsqrt(0.25) ≈ 2.0
- [ ] `frsqrt_zreg` array: [1, 4, 9, 16, ...] → [1.0, 0.5, 0.333, 0.25, ...]

---

## Level 2: Lookup Tables

### 2.1 Composable LUT
- [ ] `load_zt0`: load a known 64-byte table, verify by running a lookup
- [ ] `luti2_zreg`: 2-bit indices {0,1,2,3} → correct table values
- [ ] `luti4_zreg`: 4-bit indices {0,1,...15} → correct table values
- [ ] LUTI2 with repeated indices: all same index → all same output
- [ ] LUTI4 round-trip: encode values into table, indices are identity map

### 2.2 Fused LUT
- [ ] `luti2_op`: memory-to-memory, 4 z-vectors, byte output
- [ ] `luti4_op`: memory-to-memory, 4 z-vectors, byte output
- [ ] Fused vs composable: verify same results for identical inputs

---

## Level 3: Matrix Scratchpad

### 3.1 ZA Lifecycle
- [ ] `zero_za` → `store_tiles`: all 4096 bytes are zero
- [ ] `load_bias` → `store_tiles`: round-trip known pattern (1,2,3,...1024)
- [ ] `zero_za` → `load_bias` → `store_tiles`: bias values loaded correctly

### 3.2 Composable Outer Products
- [ ] `smopa_zreg` tile 0: all-ones → 4 per element (4-element dot in int8)
- [ ] `smopa_zreg` tiles 0-3: verify each tile accumulates independently
- [ ] `fmopa_zreg` tile 0: known fp32 vectors → verify outer product values
- [ ] Multiple `fmopa_zreg` calls: verify accumulation (not overwrite)
- [ ] `umopa_zreg`: unsigned int8 outer product
- [ ] `usmopa_zreg`: mixed signedness

### 3.3 Fused Accumulators
- [ ] `acc_smopa`: k_steps=1, verify matches single smopa_zreg
- [ ] `acc_smopa`: k_steps=4, verify correct accumulated result
- [ ] `smopa_2x2`: pre-loaded z0-z3, verify 4 tiles match manual computation

---

## Level 4: Transformer Kernels

These must be tested against known-good reference implementations (CPU scalar math).

### 4.1 RMS Normalization
- [ ] Identity case: constant array → all outputs = 1.0 (or the constant itself normalized)
- [ ] Known vector: [1,2,3,...16] → compare to CPU reference (`x / rms(x)`)
- [ ] With epsilon: verify eps prevents division by zero for all-zero input
- [ ] With weights: non-uniform weight array
- [ ] Multiple rows: verify each row normalized independently
- [ ] Large dim (512, 4096): verify no accumulation drift

### 4.2 Softmax
- [ ] Uniform input [1,1,...1]: output should be [1/16, 1/16, ...]
- [ ] One-hot: [0,0,...,100,...0] → output ≈ [0,...,1,...,0]
- [ ] Negative inputs: [-10, -9, ..., 5] → outputs sum to 1.0
- [ ] Large values: [1000, 1001, 1002, ...] → numerically stable (no overflow)
- [ ] All zeros: → uniform distribution [1/16, ...]
- [ ] Multiple rows: each row independently softmax'd
- [ ] Verify sum = 1.0 ± 1e-5 for every row

### 4.3 SiLU Activation
- [ ] SiLU(0) = 0
- [ ] SiLU(large positive) ≈ x (sigmoid→1)
- [ ] SiLU(large negative) ≈ 0 (sigmoid→0)
- [ ] SiLU reference: compare against `x / (1 + exp(-x))` for [-10..10]
- [ ] Accuracy: within 1e-3 of reference for |x| < 10

### 4.4 RoPE
- [ ] Position 0: output equals input (rotation by 0)
- [ ] Known rotation: dim=4, pos=1, theta=10000 → compute by hand
- [ ] Verify pairs rotate independently (element 2i and 2i+1)
- [ ] Increasing position: output changes predictably
- [ ] Large dim (128): verify no frequency aliasing

---

## Level 5: Quantized GEMV

The critical inference path. Test against scalar reference dequant+matmul.

### 5.1 Q8_0 GEMV
- [ ] Single block (K=32): all-ones input × all-ones weights → 32
- [ ] Multi-block (K=64, K=128): verify accumulation across blocks
- [ ] Scale factor: scale=0.5 → output halved
- [ ] Negative quants: verify sign handling
- [ ] Multi-row (M=1, 4, 16, 128): verify each row independent
- [ ] Reference test: generate random Q8_0 data, dequant on CPU, matmul on CPU,
  compare ANE output within tolerance (1e-3 per element)
- [ ] Large dimensions: M=4096, K=4096 (realistic LLM layer size)

### 5.2 Q4_0 GEMV
- [ ] All nibbles = 8 (zero after bias): output = 0
- [ ] Low nibble = 15 (+7), high nibble = 0 (-8): verify asymmetric
- [ ] Scale factor test: same quants, different scales
- [ ] Multi-block: verify 4-bit extraction across block boundaries
- [ ] Reference test: random Q4_0 data, CPU dequant+matmul, compare to ANE
- [ ] Large dimensions: M=4096, K=4096

### 5.3 Cross-validation
- [ ] Same weight matrix, fp32 GEMM vs Q8_0 GEMV: outputs within quantization error
- [ ] Same weight matrix, fp32 GEMM vs Q4_0 GEMV: outputs within expected error bounds

---

## Level 6: BLAS GEMM

### 6.1 cblas_sgemm
- [ ] Square: 16×16 identity matrix × vector = vector
- [ ] Rectangular: 100×300 × 300×200 → 100×200
- [ ] Alpha/beta: verify C = 2.0*A*B + 0.5*C
- [ ] TransA: A^T × B, compare to manual transpose+multiply
- [ ] TransB: A × B^T
- [ ] Both transposed: A^T × B^T
- [ ] Leading dimensions ≠ matrix dimensions (padded/strided layouts)
- [ ] Edge: M=1 (GEMV case), N=1, K=1
- [ ] Reference: compare against Accelerate's cblas_sgemm for random matrices

---

## Level 7: ggml Backend Integration

### 7.1 Operation Dispatch
- [ ] For each supported op (MUL_MAT, ADD, MUL, SCALE, RMS_NORM, SOFT_MAX, SILU, ROPE):
  create a minimal ggml graph with one node, run through `ane_graph_compute`,
  verify output matches ggml CPU backend output

### 7.2 Multi-Op Graphs
- [ ] Linear layer: MUL_MAT + ADD (bias) → verify end-to-end
- [ ] Attention: softmax(Q×K^T/√d) × V → compare to CPU
- [ ] Feed-forward: x × W1 → SiLU → × W2 → compare to CPU
- [ ] Full transformer block: RMS_NORM → attention → ADD → RMS_NORM → FFN → ADD

### 7.3 Quantized Inference
- [ ] Q8_0 MUL_MAT through ggml backend: random weights, compare to CPU
- [ ] Q4_0 MUL_MAT through ggml backend: random weights, compare to CPU
- [ ] Mixed graph: Q4_0 matmul + fp32 add + RMS_NORM + softmax

### 7.4 Unsupported Op Fallback
- [ ] Graph with unsupported op returns GGML_STATUS_FAILED
- [ ] `supports_op` correctly returns false for unsupported ops/types
- [ ] `supports_op` returns true for all claimed supported ops

---

## Level 8: End-to-End Inference

### 8.1 Tiny Model Test
- [ ] Create a 2-layer transformer with random weights (dim=64, heads=4)
- [ ] Run one token through both ANE and CPU backends
- [ ] Compare logits within tolerance

### 8.2 Real Model Test
- [ ] Load a small GGUF model (TinyLlama Q4_0)
- [ ] Generate 1 token with ANE backend
- [ ] Compare logits to CPU backend output
- [ ] Generate 10 tokens: verify coherent output

---

## Test Infrastructure

Each test file should:
1. Print PASS/FAIL per test case with the actual vs expected values
2. Return 0 on all-pass, 1 on any failure
3. Be runnable independently (`./build/bin/test_name`)
4. Be added to CTest via CMakeLists.txt

### Proposed test files:
| File | Covers |
|------|--------|
| `test_program_demo.cpp` | Level 1.1, 1.2, 2.2, 3.1 *(existing, expand)* |
| `test_script.cpp` | DSL compiler *(existing, expand)* |
| `test_primitives.cpp` | Level 1.3-1.8 (arithmetic, bitwise, scalar, reduce) |
| `test_lut.cpp` | Level 2 (composable + fused LUT) |
| `test_scratchpad.cpp` | Level 3 (ZA lifecycle, outer products) |
| `test_transformer.cpp` | Level 4 (RMS norm, softmax, SiLU, RoPE) |
| `test_gemv.cpp` | Level 5 (Q8_0, Q4_0 GEMV with reference comparison) |
| `test_gemm.cpp` | Level 6 (CBLAS SGEMM with Accelerate reference) |
| `test_ggml_ops.cpp` | Level 7.1-7.4 (ggml backend op dispatch) |
| `test_inference.cpp` | Level 8 (end-to-end with tiny model) |
