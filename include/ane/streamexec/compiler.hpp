#pragma once
/** --------------------------------------------------------------------------------------------------------- Compiler
 * @file compiler.hpp
 * @brief Compiles high-level ByteCode programs into CompiledProgram for stream_exec().
 *
 * Walks the variant vector, emits low-level opcodes into the bytecode stream and
 * pushes data pointers onto the operand tape. Loads and stores are implicit —
 * derived from the operation type and output buffer.
 */
#include "bytecodes.hpp"
#include <cstring>
#include "../intrinsics/common.hpp"
#include <simd/simd.h>

namespace ane {
namespace detail {
/** --------------------------------------------------------------------------------------------------------- Accumulate Opcode for MopaType
 * @brief Accumulate opcode for a given MopaType. This function maps the high-level MopaType to the
 * corresponding low-level opcode used by the assembly interpreter for accumulation operations.
 * @param type The MopaType for which to get the accumulate opcode
 * @return The opcode corresponding to the given MopaType for accumulation operations
 */
inline uint8_t acc_opcode_for(MopaType type) {
    switch (type) {
        case MopaType::signed_i8:  return op::acc_smopa;
        case MopaType::unsigned_u8: return op::acc_umopa;
        case MopaType::mixed_us:   return op::acc_usmopa;
        case MopaType::mixed_su:   return op::acc_sumopa;
        default: return op::acc_smopa;
    }
}
/** --------------------------------------------------------------------------------------------------------- Mopa_2x2 Opcode for MopaType
 * @brief Mopa_2x2 opcode for a given MopaType. This function maps the high-level MopaType to the
 * corresponding low-level opcode used by the assembly interpreter for 2x2 outer product operations.
 * @param type The MopaType for which to get the mopa_2x2 opcode
 * @return The opcode corresponding to the given MopaType for 2x2 outer product operations
 */
inline uint8_t mopa_2x2_opcode_for(MopaType type) {
    switch (type) {
        case MopaType::signed_i8:  return op::smopa_2x2;
        case MopaType::unsigned_u8: return op::umopa_2x2;
        case MopaType::mixed_us:   return op::usmopa_2x2;
        case MopaType::mixed_su:   return op::smopa_2x2; // sumopa uses smopa_2x2 with swapped operands
        default: return op::smopa_2x2;
    }
}
/** --------------------------------------------------------------------------------------------------------- Emit uint32_t to Bytecode Stream
 * @brief Helper function to emit a uint32_t value into the bytecode stream as 4 bytes in little-endian
 * order. This is used for encoding immediate values (like k_steps) into the bytecode stream.
 * @param bc The bytecode stream vector to which the bytes will be appended
 * @param val The uint32_t value to emit into the bytecode stream
 */
inline void emit_u32(std::vector<uint8_t>& bc, uint32_t val) {
    uint8_t buf[4];
    std::memcpy(buf, &val, 4);
    bc.insert(bc.end(), buf, buf + 4);
}
} // namespace detail
/** --------------------------------------------------------------------------------------------------------- Compile Function
 * @brief Compile a high-level program to bytecodes + operand tape.
 * For each iteration of the outer loop, the operand tape gets its own set of pointer entries. For
 * loop_count=1, pointers are used as-is. For loop_count>1, the caller is responsible for pre-tiling the data
 * so that the same pointers work across iterations (or the caller can set loop_count=1 and handle tiling in
 * the program vector itself).
 * @param program A vector of high-level ByteCode variants representing the operations to be performed.
 * @param loop_count The number of iterations for the outer loop (for streaming scenarios)
 * @param output A pointer to the output z_vector buffer where results will be stored (should be
 * pre-allocated by the caller)
 * @return A CompiledProgram instance containing the emitted bytecodes and operand tape ready for execution
 * by the assembly interpreter.
 */
template<SIMDVectorEquivalent OutputType>
inline CompiledProgram<OutputType> compile(
    const std::vector<ByteCode>& program,
    uint32_t loop_count,
    OutputType* output
) {
    CompiledProgram<OutputType> cp;
    cp.loop_count = loop_count;
    cp.output = output;
    // Emit bytecodes for one loop body
    for (const auto& bc : program) {
        std::visit([&](const auto& op_val) {
            using T = std::decay_t<decltype(op_val)>;
            if constexpr (std::is_same_v<T, Zero>) {
                cp.bytecodes.push_back(op::zero_za);
            } else if constexpr (std::is_same_v<T, OuterProduct>) {
                // load_rows_i8 + load_cols_i8 + mopa_2x2
                cp.bytecodes.push_back(op::load_rows_i8);
                cp.operands.push_back(op_val.a);
                cp.bytecodes.push_back(op::load_cols_i8);
                cp.operands.push_back(op_val.b);
                cp.bytecodes.push_back(detail::mopa_2x2_opcode_for(op_val.type));
            } else if constexpr (std::is_same_v<T, Accumulate>) {
                cp.bytecodes.push_back(detail::acc_opcode_for(op_val.type));
                detail::emit_u32(cp.bytecodes, op_val.k_steps);
                cp.operands.push_back(op_val.rows);
                cp.operands.push_back(op_val.cols);
            }
        }, bc);
    }
    // Emit store + halt
    cp.bytecodes.push_back(op::store_tiles);
    cp.operands.push_back(static_cast<void*>(output));
    cp.bytecodes.push_back(op::halt);
    return cp;
}
} // namespace ane
