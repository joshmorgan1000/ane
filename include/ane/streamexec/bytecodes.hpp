#pragma once
/** --------------------------------------------------------------------------------------------------------- ByteCodes
 * @file bytecodes.hpp
 * @brief Two-layer bytecode architecture for SME streaming operations.
 *
 * Layer 1: C++ DSL — high-level operations (Zero, OuterProduct, Accumulate)
 * Layer 2: Compiled bytecode — opcode stream + operand tape, consumed by stream_exec()
 *
 * Users write programs as std::vector<ByteCode>, then compile() lowers them to a
 * CompiledProgram that the assembly interpreter dispatches via O(1) jump table.
 */
#include <cstdint>
#include <variant>
#include <vector>
#include <simd/simd.h>
#include "../intrinsics/common.hpp"

namespace ane {
/** --------------------------------------------------------------------------------------------------------- Internal Opcodes
 * @namespace op
 * @brief Internal opcodes for the assembly interpreter. These are not exposed to users directly,
 * but are emitted by the compile() function based on the high-level ByteCode variants.
 * Each opcode corresponds to a specific assembly sequence that performs the desired operation.
 */
namespace op {
    constexpr uint8_t halt         = 0x00;
    constexpr uint8_t zero_za      = 0x01;
    constexpr uint8_t acc_smopa    = 0x02;  // +u32 k_steps, 2 operands
    constexpr uint8_t acc_umopa    = 0x03;
    constexpr uint8_t acc_usmopa   = 0x04;
    constexpr uint8_t acc_sumopa   = 0x05;
    constexpr uint8_t store_tiles  = 0x06;  // 1 operand (output ptr)
    constexpr uint8_t load_rows_i8 = 0x07;  // 1 operand
    constexpr uint8_t load_cols_i8 = 0x08;  // 1 operand
    constexpr uint8_t smopa_2x2    = 0x09;  // no operands
    constexpr uint8_t umopa_2x2    = 0x0A;
    constexpr uint8_t usmopa_2x2   = 0x0B;
    constexpr uint8_t load_bias    = 0x0C;  // load int32 bias into ZA, 1 operand
    constexpr uint8_t scale_store  = 0x0D;  // int32→float×scale→store, +f32 imm, 1 operand
    constexpr uint8_t NUM_OPCODES  = 0x0E;
} // namespace op
/** --------------------------------------------------------------------------------------------------------- MopaType Enumeration
 * @enum MopaType
 * @brief Enumeration of supported data types for the MOPA (Matrix Outer Product Accumulate) operations.
 * Each enumerator corresponds to a specific combination of input and output types for the MOPA instructions.
 * The values are chosen to allow for easy mapping to the corresponding opcodes in the assembly interpreter.
 */
enum class MopaType : uint8_t {
    signed_i8     = 0,  // smopa  (i8 × i8 → i32)
    unsigned_u8   = 1,  // umopa  (u8 × u8 → i32)
    mixed_us      = 2,  // usmopa (u8 × i8 → i32)
    mixed_su      = 3,  // sumopa (i8 × u8 → i32)
    signed_i16    = 4,  // smopa  (i16 × i16 → i32)
    unsigned_u16  = 5,  // umopa  (u16 × u16 → u32)
    mixed_us_16   = 6,  // usmopa (u16 × i16 → u32)
    mixed_su_16   = 7,  // sumopa (i16 × u16 → i32)
    signed_i32    = 8,  // smopa  (i32 × i32 → i64)
    unsigned_u32  = 9,  // umopa  (u32 × u32 → u64)
    mixed_us_32   = 10, // usmopa (u32 × i32 → u64)
    mixed_su_32   = 11, // sumopa (i32 × u32 → i64)
    float_f32     = 12, // fmopa  (f32 × f32 → f32)
    bfloat16      = 13, // fmopa  (f16 × f16 → f16)
};
/** --------------------------------------------------------------------------------------------------------- Opcode Mapping Functions
 * @struct Zero
 * @brief Represents a zeroing operation in the high-level bytecode. When compiled, this will emit an opcode
 * that zeroes out the ZA register tile, which is typically used to initialize the accumulator state before a
 * series of outer product operations.
 */
struct Zero {};
/** --------------------------------------------------------------------------------------------------------- OuterProduct Struct
 * @struct OuterProduct
 * @brief Represents an outer product operation in the high-level bytecode.
 */
struct OuterProduct {
    simd_uchar64* a;  // row data (2 × simd_uchar64 = z0, z1)
    simd_uchar64* b;  // col data (2 × simd_uchar64 = z2, z3)
    MopaType type = MopaType::signed_i8;
};
/** --------------------------------------------------------------------------------------------------------- Accumulate Struct
 * @struct Accumulate
 * @brief Represents an accumulation operation in the high-level bytecode, which performs a series of
 * outer product and accumulate steps.
 */
struct Accumulate {
    simd_uchar64* rows;  // contiguous row data (k_steps × 2 simd_uchar64)
    simd_uchar64* cols;  // contiguous col data (k_steps × 2 simd_uchar64)
    uint32_t k_steps;    // number of iterations
    MopaType type = MopaType::signed_i8;
};
/// @brief The user-facing bytecode variant
using ByteCode = std::variant<Zero, OuterProduct, Accumulate>;
/** --------------------------------------------------------------------------------------------------------- CompiledProgram Struct
 * @struct CompiledProgram
 * @brief Represents a compiled program consisting of a sequence of bytecodes and an operand tape.
 */
template<SIMDVectorEquivalent OutputType = simd_int16>
struct CompiledProgram {
    std::vector<uint8_t> bytecodes;   // low-level opcode stream (ends with store+halt)
    std::vector<void*>   operands;    // flat tape, consumed left-to-right
    uint32_t             loop_count;  // outer loop iterations
    OutputType*          output;      // output buffer (64-byte aligned, type carries layout info)
};
/** --------------------------------------------------------------------------------------------------------- Opcode Mapping Functions
 * @brief Compile a high-level program to bytecodes + operand tape.
 * @param program A vector of high-level ByteCode variants representing the operations to be performed.
 * @param loop_count The number of iterations for the outer loop (for streaming scenarios)
 * @param output A pointer to the output z_vector buffer where results will be stored (should be
 * pre-allocated by the caller)
 * @return A CompiledProgram instance containing the emitted bytecodes and operand tape ready for execution
 * by the assembly interpreter.
 */
template<SIMDVectorEquivalent OutputType = simd_int16>
CompiledProgram<OutputType> compile(
    const std::vector<ByteCode>& program,
    uint32_t loop_count,
    OutputType* output
);
} // namespace ane
