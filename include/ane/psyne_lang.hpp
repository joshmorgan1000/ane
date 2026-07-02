#pragma once
/** --------------------------------------------------------------------------------------------------------- Psyne Language
 * @file psyne_lang.hpp
 * @brief Compile-time parser and validated IR for the Psyne stream language.
 *
 * Psyne source is intentionally separate from the legacy script parser. This header exposes the
 * declaration and operation model without binding it to the bytecode interpreter yet.
 */
#include <ane/ane.hpp>
#include <cstdint>
#include <cstddef>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace ane {
namespace psyne {
static constexpr uint32_t VECTOR_BYTES = 64;  ///< One architectural vector window in bytes
static constexpr uint32_t VECTOR_BITS = VECTOR_BYTES * 8;  ///< One architectural vector window in bits
static constexpr uint32_t MAX_LOGICAL_BYTES = 65536;  ///< Per-stream logical byte limit
/** --------------------------------------------------------------------------------------------------------- Element Type
 * @enum element_type
 * @brief Primitive scalar storage types accepted by Psyne declarations.
 */
enum class element_type : uint8_t {
    i2,
    i4,
    f8,
    i8,
    u8,
    f16,
    bf16,
    i16,
    u16,
    f32,
    i32,
    u32,
    f64,
    i64,
    u64,
};
/** --------------------------------------------------------------------------------------------------------- Access Mode
 * @enum access_mode
 * @brief Declaration access contract checked by the compiler.
 */
enum class access_mode : uint8_t {
    read_write,
    read_only,
    write_only,
};
/** --------------------------------------------------------------------------------------------------------- Value Kind
 * @enum value_kind
 * @brief Psyne values are either one element or a stream of elements.
 */
enum class value_kind : uint8_t {
    scalar,
    stream,
};
/** --------------------------------------------------------------------------------------------------------- Stream Shape
 * @enum stream_shape
 * @brief Source-level declaration shape used to disambiguate stream operations.
 */
enum class stream_shape : uint8_t {
    scalar,
    elements,
    bytes,
    vectors,
    tiles,
};
/** --------------------------------------------------------------------------------------------------------- Operation Kind
 * @enum operation_kind
 * @brief One-operation-per-statement operations accepted by Psyne v1.
 */
enum class operation_kind : uint8_t {
    element_add,
    element_sub,
    element_mul,
    element_exp,
    element_log,
    element_pow,
    element_abs,
    element_sqrt,
    element_rsqrt,
    element_square,
    element_relu,
    element_div,
    convert,
    matmul_accumulate,
    quantized_matmul_accumulate,
    reduce_add,
    reduce_max,
    reduce_mul,
};
/** --------------------------------------------------------------------------------------------------------- Declaration
 * @struct declaration
 * @brief A scalar or stream declaration after range and padding validation.
 */
struct declaration {
    std::string name;                  ///< User-facing symbol name
    element_type type;                 ///< Declared primitive type
    value_kind kind;                   ///< Scalar or stream
    stream_shape shape;                ///< Source-level scalar/stream/vector/tile shape
    access_mode access;                ///< Compile-time access contract
    uint32_t element_count;            ///< Logical element count
    uint32_t logical_byte_count;       ///< Logical bytes requested by the source
    uint32_t physical_byte_count;      ///< Caller ABI bytes after 64-byte rounding
    std::string quantization_table;    ///< Optional lookup table symbol for packed low-bit streams
};
/** --------------------------------------------------------------------------------------------------------- Operation
 * @struct operation
 * @brief A validated single operation from the program body.
 */
struct operation {
    operation_kind kind;               ///< Element-wise or reduction operation
    std::string target;                ///< Destination symbol
    std::string lhs;                   ///< First source symbol
    std::string rhs;                   ///< Second source symbol, empty for reductions
    uint32_t output_element_count;     ///< Logical result elements written by the operation
    uint32_t line;                     ///< Source line for diagnostics
};
/** --------------------------------------------------------------------------------------------------------- Program
 * @class program
 * @brief Validated Psyne source represented as declarations plus operations.
 */
class program {
private:
    std::vector<declaration> declarations_;  ///< Declaration table in source order
    std::vector<operation> operations_;      ///< Operation list in source order
public:
    void add_declaration(const declaration& decl);
    void add_operation(const operation& op);
    const declaration* find_declaration(std::string_view name) const;
    const std::vector<declaration>& declarations() const;
    const std::vector<operation>& operations() const;
};
/** --------------------------------------------------------------------------------------------------------- Compiler
 * @class compiler
 * @brief Parses Psyne source and validates stream sizing, access, and reduction shape.
 */
class compiler {
public:
    program compile(std::string_view source) const;
};
/** --------------------------------------------------------------------------------------------------------- Executable
 * @class executable
 * @brief Compiles the currently lowerable Psyne subset and executes it with declaration-order bindings.
 */
class executable {
private:
    program ir_;                       ///< Validated Psyne IR
    mutable ane::program bytecode_;    ///< Pre-baked pointer preamble plus lowered body
    std::vector<size_t> binding_offsets_;  ///< U64 pointer slots patched by declaration order
public:
    explicit executable(std::string_view source);
    const program& ir() const;
    void exec(std::initializer_list<const void*> bindings) const;
};
/** --------------------------------------------------------------------------------------------------------- Matmul Tile F32 Plan
 * @struct matmul_tile_f32_plan
 * @brief Generated two-program PSL wrapper for one f32 output tile.
 */
struct matmul_tile_f32_plan {
    std::string chunk_source;          ///< PSL source for one outer-product chunk
    std::string reduction_source;      ///< PSL source for reducing chunk partials
    program chunk_program;             ///< Parsed chunk program
    program reduction_program;         ///< Parsed reduction program
    uint32_t tile_element_count;       ///< Elements in one output tile
    uint32_t partial_element_count;    ///< Elements consumed by the reduction program
};
/** --------------------------------------------------------------------------------------------------------- Make Matmul Tile F32 Plan
 * @brief Generates chunk and reduction PSL programs for one f32 tile matmul wrapper.
 * @param reduction_chunks Number of chunk partials reduced into one output tile
 * @return Generated and parsed two-program wrapper
 */
matmul_tile_f32_plan make_matmul_tile_f32_plan(uint32_t reduction_chunks);
uint32_t bit_width(element_type type);
uint32_t logical_bytes(element_type type, uint32_t element_count);
uint32_t round_up_64(uint32_t byte_count);
uint32_t vector_elements(element_type type);
uint32_t tile_elements(element_type type);
const char* element_type_name(element_type type);
const char* operation_kind_name(operation_kind kind);
} // namespace psyne
} // namespace ane
