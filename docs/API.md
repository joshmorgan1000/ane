# API Reference

## Overview

`ane` is a C++20 static library built around a custom bytecode interpreter written in ARM64 assembly. Operations are encoded as single-byte opcodes with inline arguments and executed inside SME streaming mode.

The execution flow:

1. **`ane::dispatch(Op, args...)`** — encodes the opcode and arguments into a compact byte stream on the stack.
2. **`interpreter::stream_exec()`** — the assembly entry point. Enters SME streaming mode (`smstart`), dispatches via a PC-relative jump table, and exits (`smstop`) when the byte stream is consumed.

Each `dispatch()` call is a self-contained operation — it enters streaming mode, executes one opcode, and exits. There is no persistent VM state between calls.

## `ane::dispatch()`

```cpp
template<typename... Args>
inline void dispatch(Op op, Args... args);
```

Encodes `op` as the first byte, followed by each argument (`uint32_t`, `int`, `float`, `uint8_t`, `bool`, or pointer) in the order they appear. The resulting byte stream is passed to the assembly interpreter.

## Memory Requirements

- All data pointers passed to `dispatch()` must be **64-byte aligned**.
- Buffers should be padded to multiples of **16 elements** (one SVL width on Apple Silicon).
- The ZA tile is 4096 bytes (64x64 bytes, or 16x16 int32/float elements).

## Helper Types

### `z_stream<T>`

RAII memory manager for 4KB-aligned buffers sized in units of 512-bit z-vectors. Useful for allocating SME-compatible memory.

```cpp
ane::z_stream<float> buf(64);       // 64 z-vectors = 4096 bytes
float* raw = buf.ptr<float>();      // typed pointer to the buffer
auto copy = buf.clone();            // deep copy
```

### `luti4<T>` / `luti2<T>`

Lookup table wrappers for the `LUTI4` (16-element, 4-bit index) and `LUTI2` (4-element, 2-bit index) instructions. Both are 64-byte aligned and accept construction from arrays, vectors, initializer lists, or individual values.

```cpp
ane::luti4<float> table = {0.0f, 0.1f, 0.2f, ..., 1.5f};  // 16 entries
ane::luti2<int8_t> small = {-1, 0, 1, 2};                   // 4 entries
```

## Opcodes

28 implemented opcodes (0x01 through 0x1C), grouped by category.

### Tile Management

| Opcode | Hex | Description |
|--------|-----|-------------|
| `zero_za` | 0x01 | Clear all ZA tiles to zero |
| `store_tiles` | 0x06 | Extract and write a 32x32 int32 tile from ZA to memory |
| `load_bias` | 0x0C | Load int32 data from memory into ZA tiles (reverse of store) |
| `scatter_tile_fp32` | 0x17 | Scatter a tile into a matrix at a given row/column offset |

### Matrix Outer Products

| Opcode | Hex | Description |
|--------|-----|-------------|
| `acc_smopa` | 0x02 | Fused load + signed int8 outer product accumulate (SMOPA) |
| `acc_umopa` | 0x03 | Fused load + unsigned int8 outer product accumulate (UMOPA) |
| `acc_usmopa` | 0x04 | Fused load + unsigned-by-signed outer product (USMOPA) |
| `acc_sumopa` | 0x05 | Signed-by-unsigned outer product (USMOPA with swapped operands) |
| `smopa_2x2` | 0x09 | 2x2 block of four SMOPA operations (no load, uses pre-loaded z-regs) |
| `umopa_2x2` | 0x0A | 2x2 block of four UMOPA operations |
| `usmopa_2x2` | 0x0B | 2x2 block of four USMOPA operations |

### Fused Dense Operations

| Opcode | Hex | Description |
|--------|-----|-------------|
| `dense_fused_i8` | 0x0E | End-to-end: fp32 quantize + int8 pack + SMOPA matmul + dequantize, with optional fused ReLU |
| `dense_scale_i8` | 0x0F | Pre-packed int8 matmul + dequantize to fp32 |
| `dense_fp32` | 0x1C | Full fp32 matmul via FMOPA with optional fused ReLU |

### Data Movement

| Opcode | Hex | Description |
|--------|-----|-------------|
| `load_rows_i8` | 0x07 | Load 128 bytes into z0, z1 (row operands for outer product) |
| `load_cols_i8` | 0x08 | Load 128 bytes into z2, z3 (column operands for outer product) |
| `transpose_fp32` | 0x18 | Transpose an M x N fp32 matrix |

### Elementwise

| Opcode | Hex | Description |
|--------|-----|-------------|
| `elementwise_add_fp32` | 0x10 | `out = a + b` |
| `elementwise_scaled_add_fp32` | 0x11 | `out = a + scale * b` |
| `elementwise_mul_fp32` | 0x12 | `out = a * b` |
| `relu_backward_fp32` | 0x13 | `out = (a > 0) ? b : 0` (gradient masking) |

### Quantization and Packing

| Opcode | Hex | Description |
|--------|-----|-------------|
| `quantize_fp32_i8` | 0x14 | Saturating fp32 to int8 conversion, clamped to [-127, 127] |
| `pack_rows_i8` | 0x15 | Pack int8 rows into dot4 format for SMOPA |
| `pack_cols_i8` | 0x16 | Pack int8 columns into dot4 format for SMOPA |
| `scale_store` | 0x0D | Convert ZA int32 to fp32, multiply by scale factor, store to memory |

### Table Lookup

| Opcode | Hex | Description |
|--------|-----|-------------|
| `luti4_op` | 0x1A | 4-bit indexed lookup via ZT0 register |
| `luti2_op` | 0x1B | 2-bit indexed lookup via ZT0 register |

### Classification

| Opcode | Hex | Description |
|--------|-----|-------------|
| `softmax_argmax_fp32` | 0x19 | Fused batched softmax + cross-entropy backward + argmax |
