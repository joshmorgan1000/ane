/** --------------------------------------------------------------------------------------------------------- Psyne Parser
 * @file psyne.cpp
 * @brief Parser and validator for the standalone Psyne language IR.
 */
#include <ane/ane.hpp>
#include <ane/psyne_lang.hpp>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace ane {
namespace psyne {
uint32_t bit_width(element_type type) {
    switch (type) {
        case element_type::i2: {
            return 2;
        }
        case element_type::i4: {
            return 4;
        }
        case element_type::f8:
        case element_type::i8:
        case element_type::u8: {
            return 8;
        }
        case element_type::f16:
        case element_type::bf16:
        case element_type::i16:
        case element_type::u16: {
            return 16;
        }
        case element_type::f32:
        case element_type::i32:
        case element_type::u32: {
            return 32;
        }
        case element_type::f64:
        case element_type::i64:
        case element_type::u64: {
            return 64;
        }
    }
    return 0;
}
uint32_t logical_bytes(element_type type, uint32_t element_count) {
    const uint64_t bits = static_cast<uint64_t>(bit_width(type)) * element_count;
    return static_cast<uint32_t>((bits + 7) >> 3);
}
uint32_t round_up_64(uint32_t byte_count) {
    return (byte_count + 63U) & ~63U;
}
uint32_t vector_elements(element_type type) {
    return VECTOR_BITS / bit_width(type);
}
uint32_t tile_elements(element_type type) {
    const uint32_t lanes = vector_elements(type);
    return lanes * lanes;
}
const char* element_type_name(element_type type) {
    switch (type) {
        case element_type::i2: {
            return "i2";
        }
        case element_type::i4: {
            return "i4";
        }
        case element_type::f8: {
            return "f8";
        }
        case element_type::i8: {
            return "i8";
        }
        case element_type::u8: {
            return "u8";
        }
        case element_type::f16: {
            return "f16";
        }
        case element_type::bf16: {
            return "bf16";
        }
        case element_type::i16: {
            return "i16";
        }
        case element_type::u16: {
            return "u16";
        }
        case element_type::f32: {
            return "f32";
        }
        case element_type::i32: {
            return "i32";
        }
        case element_type::u32: {
            return "u32";
        }
        case element_type::f64: {
            return "f64";
        }
        case element_type::i64: {
            return "i64";
        }
        case element_type::u64: {
            return "u64";
        }
    }
    return "unknown";
}
const char* operation_kind_name(operation_kind kind) {
    switch (kind) {
        case operation_kind::element_add: {
            return "element_add";
        }
        case operation_kind::element_sub: {
            return "element_sub";
        }
        case operation_kind::element_mul: {
            return "element_mul";
        }
        case operation_kind::element_exp: {
            return "element_exp";
        }
        case operation_kind::element_log: {
            return "element_log";
        }
        case operation_kind::element_pow: {
            return "element_pow";
        }
        case operation_kind::element_abs: {
            return "element_abs";
        }
        case operation_kind::element_sqrt: {
            return "element_sqrt";
        }
        case operation_kind::element_rsqrt: {
            return "element_rsqrt";
        }
        case operation_kind::element_square: {
            return "element_square";
        }
        case operation_kind::element_relu: {
            return "element_relu";
        }
        case operation_kind::element_div: {
            return "element_div";
        }
        case operation_kind::convert: {
            return "convert";
        }
        case operation_kind::matmul_accumulate: {
            return "matmul_accumulate";
        }
        case operation_kind::quantized_matmul_accumulate: {
            return "quantized_matmul_accumulate";
        }
        case operation_kind::reduce_add: {
            return "reduce_add";
        }
        case operation_kind::reduce_max: {
            return "reduce_max";
        }
        case operation_kind::reduce_mul: {
            return "reduce_mul";
        }
    }
    return "unknown";
}
void program::add_declaration(const declaration& decl) {
    declarations_.push_back(decl);
}
void program::add_operation(const operation& op) {
    operations_.push_back(op);
}
const declaration* program::find_declaration(std::string_view name) const {
    for (const declaration& decl : declarations_) {
        if (decl.name == name) {
            return &decl;
        }
    }
    return nullptr;
}
const std::vector<declaration>& program::declarations() const {
    return declarations_;
}
const std::vector<operation>& program::operations() const {
    return operations_;
}
namespace {
uint32_t declaration_index(const program& prog, const std::string& name, uint32_t line) {
    const std::vector<declaration>& declarations = prog.declarations();
    for (uint32_t i = 0; i < declarations.size(); i++) {
        if (declarations[i].name == name) {
            return i;
        }
    }
    throw std::runtime_error("Undeclared symbol '" + name + "' at line " + std::to_string(line));
}
void require_lowerable_element_target(const declaration& decl, uint32_t line) {
    if (decl.type != element_type::f32) {
        throw std::runtime_error("Executable Psyne currently lowers f32 element targets only at line " + std::to_string(line));
    }
    if (decl.kind == value_kind::stream) {
        const uint32_t lanes = vector_elements(decl.type);
        if ((decl.element_count % lanes) != 0) {
            throw std::runtime_error("Executable Psyne currently requires whole-vector f32 streams at line " + std::to_string(line));
        }
    } else if (decl.kind != value_kind::scalar) {
        throw std::runtime_error("Executable Psyne element target must be scalar or stream at line " + std::to_string(line));
    }
}
void require_lowerable_binary_element_target(const declaration& decl, operation_kind kind, uint32_t line) {
    if (decl.type == element_type::f16) {
        if (kind != operation_kind::element_add && kind != operation_kind::element_sub && kind != operation_kind::element_mul) {
            throw std::runtime_error("Executable Psyne currently lowers f16 add, sub, and mul only at line " + std::to_string(line));
        }
        if (decl.kind != value_kind::stream) {
            throw std::runtime_error("Executable Psyne currently lowers f16 element stream targets only at line " + std::to_string(line));
        }
        const uint32_t lanes = vector_elements(decl.type);
        if ((decl.element_count % lanes) != 0) {
            throw std::runtime_error("Executable Psyne currently requires whole-vector f16 streams at line " + std::to_string(line));
        }
        return;
    }
    require_lowerable_element_target(decl, line);
}
bool is_binary_element_operation(operation_kind kind) {
    return kind == operation_kind::element_add || kind == operation_kind::element_sub || kind == operation_kind::element_mul || kind == operation_kind::element_pow || kind == operation_kind::element_div;
}
bool is_unary_element_operation(operation_kind kind) {
    return kind == operation_kind::element_exp || kind == operation_kind::element_log || kind == operation_kind::element_abs || kind == operation_kind::element_sqrt || kind == operation_kind::element_rsqrt || kind == operation_kind::element_square || kind == operation_kind::element_relu;
}
void require_lowerable_element_source(const declaration& target, const declaration& source, uint32_t line) {
    if (source.type != element_type::f32) {
        throw std::runtime_error("Executable Psyne currently lowers f32 element sources only at line " + std::to_string(line));
    }
    if (source.kind == value_kind::stream) {
        const uint32_t lanes = vector_elements(source.type);
        if ((source.element_count % lanes) != 0) {
            throw std::runtime_error("Executable Psyne currently requires whole-vector f32 streams at line " + std::to_string(line));
        }
        if (source.element_count != target.element_count) {
            throw std::runtime_error("Executable Psyne source stream size does not match target at line " + std::to_string(line));
        }
    } else if (source.kind != value_kind::scalar) {
        throw std::runtime_error("Executable Psyne element source must be scalar or stream at line " + std::to_string(line));
    }
}
void require_lowerable_binary_element_source(const declaration& target, const declaration& source, operation_kind kind, uint32_t line) {
    if (target.type == element_type::f16) {
        if (kind != operation_kind::element_add && kind != operation_kind::element_sub && kind != operation_kind::element_mul) {
            throw std::runtime_error("Executable Psyne currently lowers f16 add, sub, and mul only at line " + std::to_string(line));
        }
        if (source.type != element_type::f16) {
            throw std::runtime_error("Executable Psyne f16 element sources must be f16 at line " + std::to_string(line));
        }
        if (source.kind == value_kind::stream) {
            const uint32_t lanes = vector_elements(source.type);
            if ((source.element_count % lanes) != 0) {
                throw std::runtime_error("Executable Psyne currently requires whole-vector f16 streams at line " + std::to_string(line));
            }
            if (source.element_count != target.element_count) {
                throw std::runtime_error("Executable Psyne source stream size does not match target at line " + std::to_string(line));
            }
        } else if (source.kind != value_kind::scalar) {
            throw std::runtime_error("Executable Psyne f16 element source must be scalar or stream at line " + std::to_string(line));
        }
        return;
    }
    require_lowerable_element_source(target, source, line);
}
bool is_matmul_accumulate_shape(const declaration& target, const declaration& lhs, const declaration& rhs) {
    if (target.kind != value_kind::stream || lhs.kind != value_kind::stream || rhs.kind != value_kind::stream) {
        return false;
    }
    if (target.shape != stream_shape::tiles || lhs.shape != stream_shape::vectors || rhs.shape != stream_shape::vectors) {
        return false;
    }
    if (target.type != element_type::f32 || lhs.type != rhs.type) {
        return false;
    }
    if (lhs.type != element_type::f32 && lhs.type != element_type::f16 && lhs.type != element_type::bf16) {
        return false;
    }
    if (target.element_count != tile_elements(element_type::f32)) {
        return false;
    }
    const uint32_t lhs_lanes = vector_elements(lhs.type);
    const uint32_t panel_vectors = lhs.type == element_type::f32 ? vector_elements(element_type::f32) : vector_elements(element_type::f32) / 2U;
    if (lhs.element_count != lhs_lanes * panel_vectors) {
        return false;
    }
    return rhs.element_count == lhs.element_count;
}
bool is_low_bit_type(element_type type) {
    return type == element_type::i2 || type == element_type::i4;
}
uint32_t low_bit_table_element_count(element_type type) {
    return type == element_type::i2 ? 4U : 16U;
}
bool is_supported_quantized_table_type(element_type type) {
    return type == element_type::f32 || type == element_type::i8 || type == element_type::u8;
}
bool is_convert_operation_shape(const declaration& target, const declaration& source) {
    if (target.kind != value_kind::stream || source.kind != value_kind::stream) {
        return false;
    }
    if (target.element_count != source.element_count) {
        return false;
    }
    const bool f32_to_f16 = source.type == element_type::f32 && target.type == element_type::f16;
    const bool f16_to_f32 = source.type == element_type::f16 && target.type == element_type::f32;
    const bool f32_to_bf16 = source.type == element_type::f32 && target.type == element_type::bf16;
    const bool bf16_to_f32 = source.type == element_type::bf16 && target.type == element_type::f32;
    if (!f32_to_f16 && !f16_to_f32 && !f32_to_bf16 && !bf16_to_f32) {
        return false;
    }
    return (target.element_count % vector_elements(element_type::f16)) == 0;
}
uint32_t convert_chunk_count(const declaration& target, const declaration& source) {
    const element_type narrow_type = target.type == element_type::f32 ? source.type : target.type;
    return target.element_count / vector_elements(narrow_type);
}
bool is_quantized_matmul_accumulate_shape(const program& prog, const declaration& target, const declaration& lhs, const declaration& rhs) {
    if (target.kind != value_kind::stream || lhs.kind != value_kind::stream || rhs.kind != value_kind::stream) {
        return false;
    }
    if (target.shape != stream_shape::tiles) {
        return false;
    }
    if (!is_low_bit_type(lhs.type) || lhs.type != rhs.type) {
        return false;
    }
    if (target.element_count != tile_elements(element_type::f32)) {
        return false;
    }
    if (lhs.element_count != target.element_count || rhs.element_count != target.element_count) {
        return false;
    }
    if (lhs.quantization_table.empty() || rhs.quantization_table.empty()) {
        return false;
    }
    const declaration* lhs_table = prog.find_declaration(lhs.quantization_table);
    const declaration* rhs_table = prog.find_declaration(rhs.quantization_table);
    if (lhs_table == nullptr || rhs_table == nullptr) {
        return false;
    }
    if (lhs_table->kind != value_kind::stream || rhs_table->kind != value_kind::stream) {
        return false;
    }
    if (lhs_table->element_count != low_bit_table_element_count(lhs.type) || rhs_table->element_count != low_bit_table_element_count(rhs.type)) {
        return false;
    }
    if (!is_supported_quantized_table_type(lhs_table->type) || lhs_table->type != rhs_table->type) {
        return false;
    }
    if (target.type == element_type::f32) {
        return lhs_table->type == element_type::f32;
    }
    if (target.type == element_type::i32) {
        return lhs_table->type == element_type::i8 || lhs_table->type == element_type::u8;
    }
    return false;
}
bool is_tile_stack_reduce_add_shape(const declaration& target, const declaration& source) {
    if (target.kind != value_kind::stream || source.kind != value_kind::stream) {
        return false;
    }
    if (target.shape != stream_shape::tiles || source.shape != stream_shape::tiles) {
        return false;
    }
    if (target.type != element_type::f32 || source.type != element_type::f32) {
        return false;
    }
    const uint32_t tile_count = tile_elements(target.type);
    const uint32_t lanes = vector_elements(target.type);
    if (target.element_count != tile_count) {
        return false;
    }
    return source.element_count == tile_count * lanes;
}
uint32_t tile_stack_reduce_count(const declaration& target, const declaration& source) {
    return source.element_count / target.element_count;
}
bool is_stream_vector_reduce_shape(const declaration& target, const declaration& source) {
    if (source.kind != value_kind::stream) {
        return false;
    }
    if (source.shape == stream_shape::tiles) {
        return false;
    }
    if (target.kind != value_kind::stream && target.kind != value_kind::scalar) {
        return false;
    }
    if (target.type != element_type::f32 || source.type != element_type::f32) {
        return false;
    }
    const uint32_t lanes = vector_elements(source.type);
    if ((source.element_count % lanes) != 0) {
        return false;
    }
    return target.element_count == (source.element_count / lanes);
}
uint32_t stream_vector_reduce_count(const declaration& source) {
    return source.element_count / vector_elements(source.type);
}
void validate_lowerable_operation(const program& prog, const operation& op) {
    const declaration* target = prog.find_declaration(op.target);
    const declaration* lhs = prog.find_declaration(op.lhs);
    if (target == nullptr || lhs == nullptr) {
        throw std::runtime_error("Executable Psyne operation references an undeclared stream at line " + std::to_string(op.line));
    }
    if (is_binary_element_operation(op.kind)) {
        const declaration* rhs = prog.find_declaration(op.rhs);
        if (rhs == nullptr) {
            throw std::runtime_error("Executable Psyne operation references an undeclared stream at line " + std::to_string(op.line));
        }
        require_lowerable_binary_element_target(*target, op.kind, op.line);
        require_lowerable_binary_element_source(*target, *lhs, op.kind, op.line);
        require_lowerable_binary_element_source(*target, *rhs, op.kind, op.line);
    } else if (is_unary_element_operation(op.kind)) {
        require_lowerable_element_target(*target, op.line);
        require_lowerable_element_source(*target, *lhs, op.line);
    } else if (op.kind == operation_kind::convert) {
        if (!is_convert_operation_shape(*target, *lhs)) {
            throw std::runtime_error("Executable Psyne conversion requires matching f32/f16 or f32/bf16 stream windows at line " + std::to_string(op.line));
        }
    } else if (op.kind == operation_kind::matmul_accumulate) {
        const declaration* rhs = prog.find_declaration(op.rhs);
        if (rhs == nullptr) {
            throw std::runtime_error("Executable Psyne operation references an undeclared stream at line " + std::to_string(op.line));
        }
        if (!is_matmul_accumulate_shape(*target, *lhs, *rhs)) {
            throw std::runtime_error("Executable Psyne matmul requires vector-panel sources and tile target at line " + std::to_string(op.line));
        }
    } else if (op.kind == operation_kind::quantized_matmul_accumulate) {
        const declaration* rhs = prog.find_declaration(op.rhs);
        if (rhs == nullptr) {
            throw std::runtime_error("Executable Psyne operation references an undeclared stream at line " + std::to_string(op.line));
        }
        if (!is_quantized_matmul_accumulate_shape(prog, *target, *lhs, *rhs)) {
            throw std::runtime_error("Executable Psyne quantized matmul requires packed low-bit panels and f32 lookup tables at line " + std::to_string(op.line));
        }
    } else if (op.kind == operation_kind::reduce_add) {
        if (!is_tile_stack_reduce_add_shape(*target, *lhs) && !is_stream_vector_reduce_shape(*target, *lhs)) {
            throw std::runtime_error("Executable Psyne reduce add requires f32 tile stack or whole-vector stream source at line " + std::to_string(op.line));
        }
    } else if (op.kind == operation_kind::reduce_max) {
        if (!is_stream_vector_reduce_shape(*target, *lhs)) {
            throw std::runtime_error("Executable Psyne reduce max requires a whole-vector f32 stream source at line " + std::to_string(op.line));
        }
    } else {
        throw std::runtime_error("Executable Psyne currently lowers conversions, element ops, matmul partial tiles, and scheduled f32 reductions only at line " + std::to_string(op.line));
    }
}
void validate_lowerable_program(const program& prog) {
    if (prog.declarations().size() > 8) {
        throw std::runtime_error("Executable Psyne currently supports at most 8 bound declarations");
    }
    for (const operation& op : prog.operations()) {
        validate_lowerable_operation(prog, op);
    }
}
void emit_load_element_value(const declaration& decl, uint32_t idx, uint8_t reg, ane::program& bytecode) {
    if (decl.kind == value_kind::scalar) {
        if (decl.type == element_type::f16) {
            bytecode.emit(Op::load_scalar_param_f16, static_cast<uint8_t>(idx));
        } else {
            bytecode.emit(Op::load_scalar_param_f32, static_cast<uint8_t>(idx));
        }
    } else {
        bytecode.emit(Op::load_param, static_cast<uint8_t>(idx));
    }
    bytecode.emit(Op::mov_zreg, uint8_t(0), reg);
}
void emit_advance_element_value(const declaration& decl, uint32_t idx, ane::program& bytecode) {
    if (decl.kind == value_kind::stream) {
        bytecode.emit(Op::advance_param, static_cast<uint8_t>(idx));
    }
}
void emit_store_element_value(const declaration& decl, uint32_t idx, ane::program& bytecode) {
    if (decl.kind == value_kind::scalar) {
        bytecode.emit(Op::store_scalar_param_f32, static_cast<uint8_t>(idx));
    } else {
        bytecode.emit(Op::store_param, static_cast<uint8_t>(idx));
    }
}
void emit_lowered_operation(const program& ir, const operation& op, ane::program& bytecode) {
    const declaration& target = *ir.find_declaration(op.target);
    const uint32_t target_idx = declaration_index(ir, op.target, op.line);
    const uint32_t lhs_idx = declaration_index(ir, op.lhs, op.line);
    if (op.kind == operation_kind::reduce_add) {
        const declaration& source = *ir.find_declaration(op.lhs);
        if (is_stream_vector_reduce_shape(target, source)) {
            bytecode.emit(
                Op::reduce_stream_sum_f32,
                static_cast<uint8_t>(lhs_idx),
                static_cast<uint8_t>(target_idx),
                static_cast<uint16_t>(stream_vector_reduce_count(source))
            );
            return;
        }
        bytecode.emit(
            Op::reduce_tile_stack_f32,
            static_cast<uint8_t>(lhs_idx),
            static_cast<uint8_t>(target_idx),
            static_cast<uint8_t>(tile_stack_reduce_count(target, source))
        );
        return;
    }
    if (op.kind == operation_kind::reduce_max) {
        const declaration& source = *ir.find_declaration(op.lhs);
        bytecode.emit(
            Op::reduce_stream_max_f32,
            static_cast<uint8_t>(lhs_idx),
            static_cast<uint8_t>(target_idx),
            static_cast<uint16_t>(stream_vector_reduce_count(source))
        );
        return;
    }
    if (op.kind == operation_kind::convert) {
        const declaration& source = *ir.find_declaration(op.lhs);
        Op opcode = Op::convert_f32_f16;
        if (source.type == element_type::f16 && target.type == element_type::f32) {
            opcode = Op::convert_f16_f32;
        } else if (source.type == element_type::f32 && target.type == element_type::bf16) {
            opcode = Op::convert_f32_bf16;
        } else if (source.type == element_type::bf16 && target.type == element_type::f32) {
            opcode = Op::convert_bf16_f32;
        }
        bytecode.emit(
            opcode,
            static_cast<uint8_t>(lhs_idx),
            static_cast<uint8_t>(target_idx),
            static_cast<uint16_t>(convert_chunk_count(target, source))
        );
        return;
    }
    if (is_unary_element_operation(op.kind)) {
        const declaration& lhs = *ir.find_declaration(op.lhs);
        const uint32_t chunks = target.kind == value_kind::scalar ? 1U : target.element_count / vector_elements(target.type);
        for (uint32_t i = 0; i < chunks; i++) {
            emit_load_element_value(lhs, lhs_idx, uint8_t(2), bytecode);
            if (op.kind == operation_kind::element_exp) {
                bytecode.emit(Op::fexp_zreg, uint8_t(4), uint8_t(2));
            } else if (op.kind == operation_kind::element_log) {
                bytecode.emit(Op::flog_zreg, uint8_t(4), uint8_t(2));
            } else if (op.kind == operation_kind::element_abs) {
                bytecode.emit(Op::fabs_zreg, uint8_t(4), uint8_t(2));
            } else if (op.kind == operation_kind::element_sqrt) {
                bytecode.emit(Op::fsqrt_zreg, uint8_t(4), uint8_t(2));
            } else if (op.kind == operation_kind::element_rsqrt) {
                bytecode.emit(Op::frsqrt_zreg, uint8_t(4), uint8_t(2));
            } else if (op.kind == operation_kind::element_square) {
                bytecode.emit(Op::fmul_zreg, uint8_t(4), uint8_t(2), uint8_t(2));
            } else {
                bytecode.emit(Op::fclamp_zreg, uint8_t(1), uint8_t(4), uint8_t(2), 0.0f, 0.0f);
            }
            bytecode.emit(Op::mov_zreg, uint8_t(4), uint8_t(0));
            emit_store_element_value(target, target_idx, bytecode);
            emit_advance_element_value(lhs, lhs_idx, bytecode);
            emit_advance_element_value(target, target_idx, bytecode);
        }
        return;
    }
    const uint32_t rhs_idx = declaration_index(ir, op.rhs, op.line);
    if (op.kind == operation_kind::matmul_accumulate) {
        const declaration& lhs = *ir.find_declaration(op.lhs);
        Op opcode = Op::matmul_partial_tile_f32;
        if (lhs.type == element_type::f16) {
            opcode = Op::matmul_partial_tile_f16_f32;
        } else if (lhs.type == element_type::bf16) {
            opcode = Op::matmul_partial_tile_bf16_f32;
        }
        bytecode.emit(
            opcode,
            static_cast<uint8_t>(lhs_idx),
            static_cast<uint8_t>(rhs_idx),
            static_cast<uint8_t>(target_idx)
        );
        return;
    }
    if (op.kind == operation_kind::quantized_matmul_accumulate) {
        const declaration& lhs = *ir.find_declaration(op.lhs);
        const declaration& rhs = *ir.find_declaration(op.rhs);
        const uint32_t lhs_table_idx = declaration_index(ir, lhs.quantization_table, op.line);
        const uint32_t rhs_table_idx = declaration_index(ir, rhs.quantization_table, op.line);
        const uint8_t bits = lhs.type == element_type::i2 ? uint8_t(2) : uint8_t(4);
        bytecode.emit(
            Op::matmul_lut_partial_tile_f32,
            static_cast<uint8_t>(lhs_idx),
            static_cast<uint8_t>(rhs_idx),
            static_cast<uint8_t>(lhs_table_idx),
            static_cast<uint8_t>(rhs_table_idx),
            static_cast<uint8_t>(target_idx),
            bits
        );
        return;
    }
    const declaration& lhs = *ir.find_declaration(op.lhs);
    const declaration& rhs = *ir.find_declaration(op.rhs);
    const uint32_t chunks = target.kind == value_kind::scalar ? 1U : target.element_count / vector_elements(target.type);
    for (uint32_t i = 0; i < chunks; i++) {
        emit_load_element_value(lhs, lhs_idx, uint8_t(2), bytecode);
        emit_load_element_value(rhs, rhs_idx, uint8_t(3), bytecode);
        if (target.type == element_type::f16 && op.kind == operation_kind::element_add) {
            bytecode.emit(Op::fadd_zreg_f16, uint8_t(4), uint8_t(2), uint8_t(3));
        } else if (target.type == element_type::f16 && op.kind == operation_kind::element_sub) {
            bytecode.emit(Op::fsub_zreg_f16, uint8_t(4), uint8_t(2), uint8_t(3));
        } else if (target.type == element_type::f16 && op.kind == operation_kind::element_mul) {
            bytecode.emit(Op::fmul_zreg_f16, uint8_t(4), uint8_t(2), uint8_t(3));
        } else if (op.kind == operation_kind::element_add) {
            bytecode.emit(Op::fadd_zreg, uint8_t(4), uint8_t(2), uint8_t(3));
        } else if (op.kind == operation_kind::element_sub) {
            bytecode.emit(Op::fsub_zreg, uint8_t(4), uint8_t(2), uint8_t(3));
        } else if (op.kind == operation_kind::element_div) {
            bytecode.emit(Op::fdiv_zreg, uint8_t(4), uint8_t(2), uint8_t(3));
        } else if (op.kind == operation_kind::element_pow) {
            bytecode.emit(Op::mov_zreg, uint8_t(3), uint8_t(15));
            bytecode.emit(Op::flog_zreg, uint8_t(4), uint8_t(2));
            bytecode.emit(Op::fmul_zreg, uint8_t(5), uint8_t(4), uint8_t(15));
            bytecode.emit(Op::fexp_zreg, uint8_t(4), uint8_t(5));
        } else {
            bytecode.emit(Op::fmul_zreg, uint8_t(4), uint8_t(2), uint8_t(3));
        }
        bytecode.emit(Op::mov_zreg, uint8_t(4), uint8_t(0));
        emit_store_element_value(target, target_idx, bytecode);
        emit_advance_element_value(lhs, lhs_idx, bytecode);
        emit_advance_element_value(rhs, rhs_idx, bytecode);
        emit_advance_element_value(target, target_idx, bytecode);
    }
}
void emit_lowered_body(const program& ir, ane::program& bytecode) {
    for (const operation& op : ir.operations()) {
        emit_lowered_operation(ir, op, bytecode);
    }
}
} // namespace
executable::executable(std::string_view source) : ir_(compiler().compile(source)) {
    validate_lowerable_program(ir_);
    const std::vector<declaration>& declarations = ir_.declarations();
    binding_offsets_.reserve(declarations.size());
    for (uint8_t i = 0; i < declarations.size(); i++) {
        binding_offsets_.push_back(bytecode_.mark() + 2);
        bytecode_.emit(Op::set_param, i, uint64_t(0));
    }
    emit_lowered_body(ir_, bytecode_);
}
const program& executable::ir() const {
    return ir_;
}
void executable::exec(std::initializer_list<const void*> bindings) const {
    auto binding = bindings.begin();
    for (size_t i = 0; i < binding_offsets_.size(); i++) {
        bytecode_.patch_u64(binding_offsets_[i], reinterpret_cast<uintptr_t>(*binding));
        ++binding;
    }
    bytecode_.exec();
}
matmul_tile_f32_plan make_matmul_tile_f32_plan(uint32_t reduction_chunks) {
    const uint32_t lanes = vector_elements(element_type::f32);
    if (reduction_chunks != lanes) {
        throw std::runtime_error("Matmul tile wrapper currently requires one f32 vector of partial chunks");
    }
    const uint32_t tile_count = tile_elements(element_type::f32);
    const uint32_t partial_count = tile_count * reduction_chunks;
    matmul_tile_f32_plan plan = {};
    plan.tile_element_count = tile_count;
    plan.partial_element_count = partial_count;
    plan.chunk_source =
        "a_panel: f32[16v] ro;\n"
        "b_panel: f32[16v] ro;\n"
        "partial_tile: f32[1t] wo;\n"
        "partial_tile = a_panel * b_panel;\n";
    plan.reduction_source =
        "partial_tiles: f32[" + std::to_string(reduction_chunks) + "t] ro;\n"
        "output_tile: f32[1t] wo;\n"
        "output_tile <+ partial_tiles;\n";
    compiler c;
    plan.chunk_program = c.compile(plan.chunk_source);
    plan.reduction_program = c.compile(plan.reduction_source);
    return plan;
}
namespace {
/** --------------------------------------------------------------------------------------------------------- Token Kind
 * @enum token_kind
 * @brief Lexer token kinds for Psyne source.
 */
enum class token_kind : uint8_t {
    identifier,
    number,
    colon,
    semicolon,
    lbracket,
    rbracket,
    equals,
    hash_add,
    hash_sub,
    hash_mul,
    hash_div,
    star,
    reduce_add,
    reduce_max,
    reduce_mul,
    end,
};
/** --------------------------------------------------------------------------------------------------------- Token
 * @struct token
 * @brief One token view into the source.
 */
struct token {
    token_kind kind;                   ///< Token kind
    std::string_view text;             ///< Token text
    uint32_t line;                     ///< One-based source line
};
/** --------------------------------------------------------------------------------------------------------- Extent Unit
 * @enum extent_unit
 * @brief Units accepted in stream range declarations.
 */
enum class extent_unit : uint8_t {
    elements,
    bytes,
    vectors,
    tiles,
};
stream_shape shape_from_unit(extent_unit unit) {
    switch (unit) {
        case extent_unit::elements: {
            return stream_shape::elements;
        }
        case extent_unit::bytes: {
            return stream_shape::bytes;
        }
        case extent_unit::vectors: {
            return stream_shape::vectors;
        }
        case extent_unit::tiles: {
            return stream_shape::tiles;
        }
    }
    return stream_shape::elements;
}
/** --------------------------------------------------------------------------------------------------------- Extent
 * @struct extent
 * @brief Parsed numeric range endpoint.
 */
struct extent {
    uint32_t value;                    ///< Numeric endpoint
    extent_unit unit;                  ///< Endpoint unit
};
/** --------------------------------------------------------------------------------------------------------- Lexer
 * @class lexer
 * @brief Whitespace/comment skipping lexer for Psyne v1.
 */
class lexer {
private:
    std::string_view source_;          ///< Source buffer
    size_t position_;                  ///< Current byte offset
    uint32_t line_;                    ///< Current one-based line
    void skip_ignored() {
        while (position_ < source_.size()) {
            const char c = source_[position_];
            if (c == '\n') {
                position_++;
                line_++;
            } else if (std::isspace(static_cast<unsigned char>(c)) != 0) {
                position_++;
            } else if (position_ + 1 < source_.size() && c == '/' && source_[position_ + 1] == '/') {
                position_ += 2;
                while (position_ < source_.size() && source_[position_] != '\n') {
                    position_++;
                }
            } else if (position_ + 1 < source_.size() && c == '/' && source_[position_ + 1] == '*') {
                position_ += 2;
                while (position_ + 1 < source_.size() && !(source_[position_] == '*' && source_[position_ + 1] == '/')) {
                    if (source_[position_] == '\n') {
                        line_++;
                    }
                    position_++;
                }
                if (position_ + 1 < source_.size()) {
                    position_ += 2;
                }
            } else {
                return;
            }
        }
    }
    token make_single(token_kind kind) {
        const uint32_t token_line = line_;
        position_++;
        return {kind, source_.substr(position_ - 1, 1), token_line};
    }
public:
    explicit lexer(std::string_view source) : source_(source), position_(0), line_(1) {}
    token next() {
        skip_ignored();
        if (position_ >= source_.size()) {
            return {token_kind::end, {}, line_};
        }
        const uint32_t token_line = line_;
        const char c = source_[position_];
        if (position_ + 1 < source_.size() && c == '#' && source_[position_ + 1] == '+') {
            position_ += 2;
            return {token_kind::hash_add, source_.substr(position_ - 2, 2), token_line};
        }
        if (position_ + 1 < source_.size() && c == '#' && source_[position_ + 1] == '-') {
            position_ += 2;
            return {token_kind::hash_sub, source_.substr(position_ - 2, 2), token_line};
        }
        if (position_ + 1 < source_.size() && c == '#' && source_[position_ + 1] == '*') {
            position_ += 2;
            return {token_kind::hash_mul, source_.substr(position_ - 2, 2), token_line};
        }
        if (position_ + 1 < source_.size() && c == '#' && source_[position_ + 1] == '/') {
            position_ += 2;
            return {token_kind::hash_div, source_.substr(position_ - 2, 2), token_line};
        }
        if (position_ + 1 < source_.size() && c == '<' && source_[position_ + 1] == '+') {
            position_ += 2;
            return {token_kind::reduce_add, source_.substr(position_ - 2, 2), token_line};
        }
        if (position_ + 4 <= source_.size() && c == '<' && source_.substr(position_, 4) == "<max") {
            position_ += 4;
            return {token_kind::reduce_max, source_.substr(position_ - 4, 4), token_line};
        }
        if (position_ + 1 < source_.size() && c == '<' && source_[position_ + 1] == '*') {
            position_ += 2;
            return {token_kind::reduce_mul, source_.substr(position_ - 2, 2), token_line};
        }
        switch (c) {
            case ':': {
                return make_single(token_kind::colon);
            }
            case ';': {
                return make_single(token_kind::semicolon);
            }
            case '[': {
                return make_single(token_kind::lbracket);
            }
            case ']': {
                return make_single(token_kind::rbracket);
            }
            case '=': {
                return make_single(token_kind::equals);
            }
            case '*': {
                return make_single(token_kind::star);
            }
            default: {
                break;
            }
        }
        if (std::isdigit(static_cast<unsigned char>(c)) != 0) {
            const size_t start = position_;
            while (position_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[position_])) != 0) {
                position_++;
            }
            if (position_ < source_.size() && (source_[position_] == 'b' || source_[position_] == 'v' || source_[position_] == 't')) {
                position_++;
            }
            return {token_kind::number, source_.substr(start, position_ - start), token_line};
        }
        if (std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_') {
            const size_t start = position_;
            while (position_ < source_.size()) {
                const char ident_char = source_[position_];
                if (std::isalnum(static_cast<unsigned char>(ident_char)) != 0 || ident_char == '_') {
                    position_++;
                } else {
                    break;
                }
            }
            return {token_kind::identifier, source_.substr(start, position_ - start), token_line};
        }
        throw std::runtime_error(std::string("Unexpected character '") + c + "' at line " + std::to_string(token_line));
    }
};
/** --------------------------------------------------------------------------------------------------------- Parser
 * @class parser
 * @brief Recursive-descent parser for declarations and single-operation statements.
 */
class parser {
private:
    lexer lexer_;                                      ///< Token source
    token current_;                                    ///< Current lookahead token
    token previous_;                                   ///< Last consumed token
    program program_;                                  ///< Program being constructed
    std::unordered_map<std::string, size_t> symbols_;  ///< Declaration name to declaration index
    void advance() {
        previous_ = current_;
        current_ = lexer_.next();
    }
    bool accept(token_kind kind) {
        if (current_.kind == kind) {
            advance();
            return true;
        }
        return false;
    }
    void expect(token_kind kind, const char* message) {
        if (current_.kind != kind) {
            throw std::runtime_error(std::string(message) + " at line " + std::to_string(current_.line));
        }
        advance();
    }
    std::string parse_identifier(const char* message) {
        if (current_.kind != token_kind::identifier) {
            throw std::runtime_error(std::string(message) + " at line " + std::to_string(current_.line));
        }
        std::string result(current_.text);
        advance();
        return result;
    }
    uint32_t parse_uint_text(std::string_view text) {
        const std::string owned(text);
        return static_cast<uint32_t>(std::strtoul(owned.c_str(), nullptr, 10));
    }
    extent parse_extent() {
        if (current_.kind != token_kind::number) {
            throw std::runtime_error("Expected range extent at line " + std::to_string(current_.line));
        }
        extent result = {0, extent_unit::elements};
        std::string_view text = current_.text;
        const char suffix = text.empty() ? '\0' : text[text.size() - 1];
        if (suffix == 'b' || suffix == 'v' || suffix == 't') {
            text.remove_suffix(1);
            if (suffix == 'b') {
                result.unit = extent_unit::bytes;
            } else if (suffix == 'v') {
                result.unit = extent_unit::vectors;
            } else {
                result.unit = extent_unit::tiles;
            }
        }
        result.value = parse_uint_text(text);
        advance();
        return result;
    }
    uint32_t extent_to_elements(element_type type, const extent& ext, uint32_t line) {
        switch (ext.unit) {
            case extent_unit::elements: {
                return ext.value;
            }
            case extent_unit::bytes: {
                const uint64_t bits = static_cast<uint64_t>(ext.value) * 8U;
                const uint32_t width = bit_width(type);
                if ((bits % width) != 0) {
                    throw std::runtime_error("Byte range does not align to " + std::string(element_type_name(type)) + " at line " + std::to_string(line));
                }
                return static_cast<uint32_t>(bits / width);
            }
            case extent_unit::vectors: {
                return ext.value * vector_elements(type);
            }
            case extent_unit::tiles: {
                return ext.value * tile_elements(type);
            }
        }
        return 0;
    }
    bool parse_type_name(std::string_view name, element_type& type) {
        if (name == "i2") {
            type = element_type::i2;
        } else if (name == "i4") {
            type = element_type::i4;
        } else if (name == "f8") {
            type = element_type::f8;
        } else if (name == "i8") {
            type = element_type::i8;
        } else if (name == "u8") {
            type = element_type::u8;
        } else if (name == "f16") {
            type = element_type::f16;
        } else if (name == "bf16") {
            type = element_type::bf16;
        } else if (name == "i16") {
            type = element_type::i16;
        } else if (name == "u16") {
            type = element_type::u16;
        } else if (name == "f32") {
            type = element_type::f32;
        } else if (name == "i32") {
            type = element_type::i32;
        } else if (name == "u32") {
            type = element_type::u32;
        } else if (name == "f64") {
            type = element_type::f64;
        } else if (name == "i64") {
            type = element_type::i64;
        } else if (name == "u64") {
            type = element_type::u64;
        } else {
            return false;
        }
        return true;
    }
    const declaration& require_symbol(const std::string& name, uint32_t line) const {
        const declaration* decl = program_.find_declaration(name);
        if (decl == nullptr) {
            throw std::runtime_error("Undeclared symbol '" + name + "' at line " + std::to_string(line));
        }
        return *decl;
    }
    void require_writable(const declaration& decl, uint32_t line) const {
        if (decl.access == access_mode::read_only) {
            throw std::runtime_error("Cannot write to read-only stream '" + decl.name + "' at line " + std::to_string(line));
        }
    }
    void require_readable(const declaration& decl, uint32_t line) const {
        if (decl.access == access_mode::write_only) {
            throw std::runtime_error("Cannot read from write-only stream '" + decl.name + "' at line " + std::to_string(line));
        }
    }
    uint32_t operation_element_count(const declaration& lhs, const declaration& rhs, uint32_t line) const {
        if (lhs.kind == value_kind::scalar && rhs.kind == value_kind::scalar) {
            return 1;
        }
        if (lhs.kind == value_kind::scalar) {
            return rhs.element_count;
        }
        if (rhs.kind == value_kind::scalar) {
            return lhs.element_count;
        }
        if (lhs.element_count != rhs.element_count) {
            throw std::runtime_error("Stream element counts do not match at line " + std::to_string(line));
        }
        return lhs.element_count;
    }
    bool is_matmul_accumulate(const declaration& target, const declaration& lhs, const declaration& rhs) const {
        return is_matmul_accumulate_shape(target, lhs, rhs);
    }
    bool is_quantized_matmul_accumulate(const declaration& target, const declaration& lhs, const declaration& rhs) const {
        return is_quantized_matmul_accumulate_shape(program_, target, lhs, rhs);
    }
    void validate_target_count(const declaration& target, uint32_t expected_count, uint32_t line) const {
        if (target.element_count != expected_count) {
            throw std::runtime_error("Target element count does not match operation result at line " + std::to_string(line));
        }
    }
    void parse_declaration(const std::string& name) {
        const uint32_t line = previous_.line;
        if (symbols_.find(name) != symbols_.end()) {
            throw std::runtime_error("Duplicate declaration '" + name + "' at line " + std::to_string(line));
        }
        if (current_.kind == token_kind::colon) {
            advance();
        }
        const std::string type_name = parse_identifier("Expected type name");
        element_type type = element_type::f32;
        if (!parse_type_name(type_name, type)) {
            throw std::runtime_error("Unknown Psyne type '" + type_name + "' at line " + std::to_string(line));
        }
        uint32_t element_count = 1;
        value_kind kind = value_kind::scalar;
        stream_shape shape = stream_shape::scalar;
        if (accept(token_kind::lbracket)) {
            kind = value_kind::stream;
            const extent first = parse_extent();
            if (accept(token_kind::colon)) {
                const extent last = parse_extent();
                shape = shape_from_unit(last.unit);
                const uint32_t first_elements = extent_to_elements(type, first, line);
                const uint32_t last_elements = extent_to_elements(type, last, line);
                if (last_elements <= first_elements) {
                    throw std::runtime_error("Stream range must have a positive length at line " + std::to_string(line));
                }
                element_count = last_elements - first_elements;
            } else {
                shape = shape_from_unit(first.unit);
                element_count = extent_to_elements(type, first, line);
                if (element_count == 0) {
                    throw std::runtime_error("Stream range must have a positive length at line " + std::to_string(line));
                }
            }
            expect(token_kind::rbracket, "Expected ']'");
        }
        access_mode access = access_mode::read_write;
        std::string quantization_table;
        while (current_.kind == token_kind::identifier) {
            if (current_.text == "ro") {
                access = access_mode::read_only;
                advance();
            } else if (current_.text == "wo") {
                access = access_mode::write_only;
                advance();
            } else if (current_.text == "rw") {
                access = access_mode::read_write;
                advance();
            } else if (current_.text == "table") {
                advance();
                quantization_table = parse_identifier("Expected quantization table name");
            } else {
                break;
            }
        }
        const uint32_t bytes = logical_bytes(type, element_count);
        if (bytes > MAX_LOGICAL_BYTES) {
            throw std::runtime_error("Stream exceeds 64 KiB logical byte limit at line " + std::to_string(line));
        }
        const uint32_t physical_bytes = round_up_64(bytes);
        declaration decl = {name, type, kind, shape, access, element_count, bytes, physical_bytes, quantization_table};
        symbols_[name] = program_.declarations().size();
        program_.add_declaration(decl);
    }
    void parse_element_operation(const std::string& target_name) {
        const uint32_t line = previous_.line;
        expect(token_kind::equals, "Expected '='");
        const std::string lhs_name = parse_identifier("Expected first source");
        if (lhs_name == "exp" || lhs_name == "log" || lhs_name == "abs" || lhs_name == "sqrt" || lhs_name == "rsqrt" || lhs_name == "square" || lhs_name == "relu") {
            const std::string source_name = parse_identifier("Expected unary source");
            if (current_.kind != token_kind::semicolon && current_.kind != token_kind::end) {
                throw std::runtime_error("Only one operation is allowed per statement at line " + std::to_string(current_.line));
            }
            const declaration& target = require_symbol(target_name, line);
            const declaration& source = require_symbol(source_name, line);
            require_writable(target, line);
            require_readable(source, line);
            const uint32_t count = source.kind == value_kind::scalar ? 1U : source.element_count;
            validate_target_count(target, count, line);
            operation_kind kind = operation_kind::element_exp;
            if (lhs_name == "log") {
                kind = operation_kind::element_log;
            } else if (lhs_name == "abs") {
                kind = operation_kind::element_abs;
            } else if (lhs_name == "sqrt") {
                kind = operation_kind::element_sqrt;
            } else if (lhs_name == "rsqrt") {
                kind = operation_kind::element_rsqrt;
            } else if (lhs_name == "square") {
                kind = operation_kind::element_square;
            } else if (lhs_name == "relu") {
                kind = operation_kind::element_relu;
            }
            program_.add_operation({kind, target_name, source_name, {}, count, line});
            return;
        }
        if (lhs_name == "pow") {
            const std::string base_name = parse_identifier("Expected pow base source");
            const std::string exponent_name = parse_identifier("Expected pow exponent source");
            if (current_.kind != token_kind::semicolon && current_.kind != token_kind::end) {
                throw std::runtime_error("Only one operation is allowed per statement at line " + std::to_string(current_.line));
            }
            const declaration& target = require_symbol(target_name, line);
            const declaration& base = require_symbol(base_name, line);
            const declaration& exponent = require_symbol(exponent_name, line);
            require_writable(target, line);
            require_readable(base, line);
            require_readable(exponent, line);
            const uint32_t count = operation_element_count(base, exponent, line);
            validate_target_count(target, count, line);
            program_.add_operation({operation_kind::element_pow, target_name, base_name, exponent_name, count, line});
            return;
        }
        if (current_.kind == token_kind::semicolon || current_.kind == token_kind::end) {
            const declaration& target = require_symbol(target_name, line);
            const declaration& source = require_symbol(lhs_name, line);
            require_writable(target, line);
            require_readable(source, line);
            if (!is_convert_operation_shape(target, source)) {
                throw std::runtime_error("Assignment requires compatible f32/f16 or f32/bf16 stream conversion at line " + std::to_string(line));
            }
            program_.add_operation({operation_kind::convert, target_name, lhs_name, {}, target.element_count, line});
            return;
        }
        operation_kind kind = operation_kind::element_add;
        if (accept(token_kind::hash_add)) {
            kind = operation_kind::element_add;
        } else if (accept(token_kind::hash_sub)) {
            kind = operation_kind::element_sub;
        } else if (accept(token_kind::hash_mul)) {
            kind = operation_kind::element_mul;
        } else if (accept(token_kind::hash_div)) {
            kind = operation_kind::element_div;
        } else if (accept(token_kind::star)) {
            kind = operation_kind::matmul_accumulate;
        } else {
            throw std::runtime_error("Expected '#+', '#-', '#*', '#/', or '*' at line " + std::to_string(current_.line));
        }
        const std::string rhs_name = parse_identifier("Expected second source");
        if (current_.kind != token_kind::semicolon && current_.kind != token_kind::end) {
            throw std::runtime_error("Only one operation is allowed per statement at line " + std::to_string(current_.line));
        }
        const declaration& target = require_symbol(target_name, line);
        const declaration& lhs = require_symbol(lhs_name, line);
        const declaration& rhs = require_symbol(rhs_name, line);
        require_writable(target, line);
        require_readable(lhs, line);
        require_readable(rhs, line);
        uint32_t count = 0;
        if (kind == operation_kind::matmul_accumulate) {
            if (is_matmul_accumulate(target, lhs, rhs)) {
                count = target.element_count;
            } else if (is_quantized_matmul_accumulate(target, lhs, rhs)) {
                kind = operation_kind::quantized_matmul_accumulate;
                count = target.element_count;
            } else {
                throw std::runtime_error("Matrix multiply requires vector-panel sources and tile target at line " + std::to_string(line));
            }
        } else {
            count = operation_element_count(lhs, rhs, line);
        }
        validate_target_count(target, count, line);
        program_.add_operation({kind, target_name, lhs_name, rhs_name, count, line});
    }
    void parse_reduction_operation(const std::string& target_name, operation_kind kind) {
        const uint32_t line = previous_.line;
        advance();
        const std::string source_name = parse_identifier("Expected reduction source");
        if (current_.kind != token_kind::semicolon && current_.kind != token_kind::end) {
            throw std::runtime_error("Only one operation is allowed per statement at line " + std::to_string(current_.line));
        }
        const declaration& target = require_symbol(target_name, line);
        const declaration& source = require_symbol(source_name, line);
        require_writable(target, line);
        require_readable(source, line);
        if (source.kind != value_kind::stream) {
            throw std::runtime_error("Reduction source must be a stream at line " + std::to_string(line));
        }
        const uint32_t lanes = vector_elements(source.type);
        const uint32_t expected = (source.element_count + lanes - 1U) / lanes;
        if (target.element_count < expected) {
            throw std::runtime_error("Not enough elements in target for reduction at line " + std::to_string(line));
        }
        if (target.element_count > expected) {
            throw std::runtime_error("Too many elements in target for reduction at line " + std::to_string(line));
        }
        program_.add_operation({kind, target_name, source_name, {}, expected, line});
    }
    void parse_statement() {
        const std::string name = parse_identifier("Expected declaration or operation");
        if (current_.kind == token_kind::colon) {
            parse_declaration(name);
        } else if (current_.kind == token_kind::identifier) {
            element_type ignored = element_type::f32;
            if (!parse_type_name(current_.text, ignored)) {
                throw std::runtime_error("Expected operation after '" + name + "' at line " + std::to_string(current_.line));
            }
            parse_declaration(name);
        } else if (current_.kind == token_kind::equals) {
            parse_element_operation(name);
        } else if (current_.kind == token_kind::reduce_add) {
            parse_reduction_operation(name, operation_kind::reduce_add);
        } else if (current_.kind == token_kind::reduce_max) {
            parse_reduction_operation(name, operation_kind::reduce_max);
        } else if (current_.kind == token_kind::reduce_mul) {
            parse_reduction_operation(name, operation_kind::reduce_mul);
        } else {
            throw std::runtime_error("Expected declaration or operation after '" + name + "' at line " + std::to_string(current_.line));
        }
    }
public:
    explicit parser(std::string_view source) : lexer_(source), current_({token_kind::end, {}, 1}), previous_({token_kind::end, {}, 1}) {
        advance();
    }
    program parse() {
        while (current_.kind != token_kind::end) {
            parse_statement();
            expect(token_kind::semicolon, "Expected ';'");
        }
        return program_;
    }
};
} // namespace
program compiler::compile(std::string_view source) const {
    parser p(source);
    return p.parse();
}
} // namespace psyne
} // namespace ane
