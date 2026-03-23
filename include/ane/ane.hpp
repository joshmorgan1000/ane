#pragma once
/** --------------------------------------------------------------------------------------------------------- Apple Neural Engine
 * @file ane.hpp
 * @brief Header file for the ane library.
 * Provides declarations for the lookup_table, nibble_pack, and crumb_pack utility structs, as well as
 * includes the assembly-optimized kernel function declarations from asm.hpp. This file serves as the main
 * public interface for the ane library, allowing users to access the optimized kernels and utility functions.
 * 
 * @author Josh Morgan
 * https://github.com/joshmorgan1000/ane
 * Released under the MIT License
 */
#if defined(__aarch64__) && defined(__APPLE__)
#include <cstdint>
#include <cstddef>
#include <array>
#include <concepts>
#include <type_traits>
#include <simd/simd.h>
#include <vector>

namespace ane {
namespace interpreter {
/** --------------------------------------------------------------------------------------------------------- Interpreter Namespace
 * @brief The interpreter namespace contains the assembly code for the bytecode interpreter.
 * @param data Pointer to the bytecode stream to execute
 * @param size Size of the bytecode stream in bytes
 */
extern "C" {
void stream_exec(const uint8_t* data, size_t size);
} // extern "C"
} // namespace interpreter
/** --------------------------------------------------------------------------------------------------------- Internal Opcodes
 * @enum Op
 * @brief Internal opcodes for the assembly interpreter. These are not exposed to users directly,
 * but are emitted by the compile() function based on the high-level ByteCode variants.
 * Each opcode corresponds to a specific assembly sequence that performs the desired operation.
 */
enum class Op : uint8_t {
    reserved_0x00                = 0x00,  ///< Slot 0 reserved (dispatch exits at bytecodes end)
    zero_za                      = 0x01,
    acc_smopa                    = 0x02,  // +u32 k_steps, 2 operands
    acc_umopa                    = 0x03,
    acc_usmopa                   = 0x04,
    acc_sumopa                   = 0x05,
    store_tiles                  = 0x06,  // 1 operand (output ptr)
    load_rows_i8                 = 0x07,  // 1 operand
    load_cols_i8                 = 0x08,  // 1 operand
    smopa_2x2                    = 0x09,  // no operands
    umopa_2x2                    = 0x0A,
    usmopa_2x2                   = 0x0B,
    load_bias                    = 0x0C,  // load int32 bias into ZA, 1 operand
    scale_store                  = 0x0D,  // int32→float×scale→store, +f32 imm, 1 operand
    dense_fused_i8               = 0x0E,  // fused quantize+pack+matmul+dequant(+relu)
    dense_scale_i8               = 0x0F,  // pre-packed matmul+dequant
    elementwise_add_fp32         = 0x10,  // fused elementwise add
    elementwise_scaled_add_fp32  = 0x11,  // out = a + scale*b
    elementwise_mul_fp32         = 0x12,  // out = a * b
    relu_backward_fp32           = 0x13,  // out = (a>0)?b:0
    quantize_fp32_i8             = 0x14,  // quantize fp32→i8
    pack_rows_i8                 = 0x15,  // pack rows to dot4
    pack_cols_i8                 = 0x16,  // pack cols to dot4
    scatter_tile_fp32            = 0x17,  // scatter tile to matrix
    transpose_fp32               = 0x18,  // transpose M×N
    softmax_argmax_fp32          = 0x19,  // batched softmax + argmax
    luti4_op                     = 0x1A,  // 4-bit table lookup via ZT0, [count:u32][elem_size:u8], 3 ops
    luti2_op                     = 0x1B,  // 2-bit table lookup via ZT0, [count:u32][elem_size:u8], 3 ops
    dense_fp32                   = 0x1C,  // full fp32 matmul via FMOPA (+optional relu)
    NUM_OPCODES                  = 0x1D,
};
/** --------------------------------------------------------------------------------------------------------- Opcode Dispatch
 * @brief Dispatches an operation to the assembly interpreter by encoding the opcode and its arguments into a
 * byte stream.
 * @param op The opcode to execute
 * @param args Immediates (uint32_t, int, float, uint8_t, bool) and operand pointers in encoding order
 */
template<typename... Args>
inline void dispatch(Op op, Args... args) {
    constexpr size_t size_needed = 1 + (sizeof(Args) + ...);
    alignas(8) uint8_t bytecodes[size_needed] = {};
    bytecodes[0] = static_cast<uint8_t>(op);
    size_t offset = 1;
    auto emit = [&](auto arg) {
        std::memcpy(&bytecodes[offset], &arg, sizeof(arg));
        offset += sizeof(arg);
    };
    (emit(args), ...);
    asm volatile("" ::: "memory");  // prevent dead store elimination of bytecodes
    interpreter::stream_exec(bytecodes, size_needed);
}
/** --------------------------------------------------------------------------------------------------------- ValidZType Concept
 * @brief Concept to constrain valid types for z_stream operations. This includes all the standard
 * integer and floating-point types that the Apple Neural Engine supports for its vectorized
 */
template<typename T>
concept ValidZType = std::same_as<T, float>
    || std::same_as<T, int32_t>
    || std::same_as<T, uint32_t>
    || std::same_as<T, int16_t>
    || std::same_as<T, uint16_t>
    || std::same_as<T, int8_t>
    || std::same_as<T, uint8_t>
    || std::same_as<T, bfloat16_t>
    || std::same_as<T, double>
    || std::same_as<T, int64_t>
    || std::same_as<T, uint64_t>;
/** --------------------------------------------------------------------------------------------------------- SIMDVectorEquivalent Concept
 * @brief Concept to constrain types that have a corresponding SIMD vector type in simd.h.
 * Makes sure memory is aligned and adds a lot of stuff so we don't have to
 */
template<typename T>
concept SIMDVectorEquivalent = std::same_as<T, simd_float16>
    || std::same_as<T, simd_uchar64>
    || std::same_as<T, simd_char64>
    || std::same_as<T, simd_ushort32>
    || std::same_as<T, simd_short32>
    || std::same_as<T, simd_half32>
    || std::same_as<T, simd_int16>
    || std::same_as<T, simd_uint16>
    || std::same_as<T, simd_long8>
    || std::same_as<T, simd_ulong8>
    || std::same_as<T, simd_double8>;
/** --------------------------------------------------------------------------------------------------------- LUTCompatible Concept
 * @brief Concept to constrain types that are compatible with the LUTI2 or LUTI4 instructions.
 */
template<typename T>
concept LUTCompatible = std::same_as<T, uint8_t>
    || std::same_as<T, int8_t>
    || std::same_as<T, uint16_t>
    || std::same_as<T, int16_t>
    || std::same_as<T, bfloat16_t>
    || std::same_as<T, uint32_t>
    || std::same_as<T, int32_t>
    || std::same_as<T, float>;
/** --------------------------------------------------------------------------------------------------------- luti4
 * @struct luti4
 * @brief A helper utility struct for working with the LUTI4 instruction, which performs a 4-bit indexed
 * lookup into a 4-element or 16-element table.
 */
template<LUTCompatible T>
struct alignas(64) luti4 {
    /// @brief The underlying data array for the lookup table, aligned to 64 bytes.
    std::array<T, 16> data;
    /** --------------------------------------------------------------------------------- Default Constructor
     * @brief Default constructor initializes the lookup table with zeros
     */
    luti4(T init_val = 0) {
        data.fill(init_val);
    }
    /** --------------------------------------------------------------------------------- Construct from Array
     * @brief Constructor that initializes the lookup table with the provided array of
     * values. The input array must have exactly 16 elements, which will be copied into
     * the internal data array.
     * @param init_data The array of values to initialize the lookup table with.
     */
    luti4(const std::array<T, 16>& init_data) : data(init_data) {}
    /**
     * @brief Constructor that initializes the lookup table with the provided vector of
     * values. The input vector must have exactly 16 elements, which will be copied into
     * the internal data array.
     * @param init_data The vector of values to initialize the lookup table with
     * (must have 16 elements)
     */
    luti4(const std::vector<T>& init_data) {
        if (init_data.size() != 16) [[unlikely]] {
            throw std::invalid_argument("Initializer vector must have exactly"
                " 16 elements for luti4");
        }
        std::memcpy(data.data(), init_data.data(), 16 * sizeof(T));
    }
    /** --------------------------------------------------------------------------------- Construct from Initializer List
     * @brief Constructor that initializes the lookup table with the provided initializer
     * list of values. The input initializer list must have exactly 16 elements, which
     * will be copied into the internal data array.
     * @param init_list The initializer list of values to initialize the lookup table
     * with (must have 16 elements)
     */
    luti4(std::initializer_list<T> init_list) {
        if (init_list.size() != 16) [[unlikely]] {
            throw std::invalid_argument("Initializer list must have exactly"
                " 16 elements for luti4");
        }
        std::copy(init_list.begin(), init_list.end(), data.begin());
    }
    /** --------------------------------------------------------------------------------- Construct from Individual Values
     * @brief Constructor that initializes the lookup table with the provided individual
     * values. The constructor takes 16 individual values as parameters, which will be
     * copied into the internal data array.
     * @param val0 The first value to initialize the lookup table with
     * @param val1 The second value to initialize the lookup table with
     * @param val2 The third value to initialize the lookup table with
     * @param val3 The fourth value to initialize the lookup table with
     * @param val4 The fifth value to initialize the lookup table with
     * @param val5 The sixth value to initialize the lookup table with
     * @param val6 The seventh value to initialize the lookup table with
     * @param val7 The eighth value to initialize the lookup table with
     * @param val8 The ninth value to initialize the lookup table with
     * @param val9 The tenth value to initialize the lookup table with
     * @param val10 The eleventh value to initialize the lookup table with
     * @param val11 The twelfth value to initialize the lookup table with
     * @param val12 The thirteenth value to initialize the lookup table with
     * @param val13 The fourteenth value to initialize the lookup table with
     * @param val14 The fifteenth value to initialize the lookup table with
     * @param val15 The sixteenth value to initialize the lookup table with
     */
    luti4(
        T val0, T val1, T val2, T val3, T val4, T val5, T val6, T val7,
        T val8, T val9, T val10, T val11, T val12, T val13, T val14, T val15
    ) : data{val0, val1, val2, val3, val4, val5, val6, val7, val8, val9,
        val10, val11, val12, val13, val14, val15} {}
    /** --------------------------------------------------------------------------------- Get Value
     * @brief Gets a reference to the value at the specified index in the lookup table.
     * The index should be in the range [0, 15], and the function will return a reference
     * to the corresponding value in the internal data array.
     * @param index The index of the value to get (should be in the range [0, 15])
     * @return A reference to the value at the specified index in the lookup table
     */
    T& operator[](size_t index) {
        if (index >= data.size()) [[unlikely]] {
            throw std::out_of_range("Index out of range for luti4");
        }
        return data[index];
    }
    /** --------------------------------------------------------------------------------- Get Value (const)
     * @brief Gets a const reference to the value at the specified index in the lookup
     * table. The index should be in the range [0, 15], and the function will return a
     * const reference to the corresponding value in the internal data array.
     * @param index The index of the value to get (should be in the range [0, 15])
     * @return A const reference to the value at the specified index in the lookup table
     */
    const T& operator[](size_t index) const {
        if (index >= data.size()) [[unlikely]] {
            throw std::out_of_range("Index out of range for luti4");
        }
        return data[index];
    }
};
/** --------------------------------------------------------------------------------------------------------- luti2
 * @struct luti2
 * @brief A helper utility struct for working with the LUTI2 instruction, which performs a 2-bit indexed
 * lookup into a 4-element table.
 */
template<LUTCompatible T>
struct luti2 {
    /// @brief The underlying data array for the lookup table, aligned to 64 bytes.
    std::array<T, 4> data;
    /** --------------------------------------------------------------------------------- Default Constructor
     * @brief Default constructor initializes the lookup table with zeros
     */
    luti2(T init_val = 0) {
        data.fill(init_val);
    }
    /** --------------------------------------------------------------------------------- Construct from Array
     * @brief Constructor that initializes the lookup table with the provided array of
     * values. The input array must have exactly 4 elements, which will be copied into
     * the internal data array.
     * @param init_data The array of values to initialize the lookup table with.
     */
    luti2(const std::array<T, 4>& init_data) : data(init_data) {}
    /**
     * @brief Constructor that initializes the lookup table with the provided vector of
     * values. The input vector must have exactly 4 elements, which will be copied into
     * the internal data array.
     * @param init_data The vector of values to initialize the lookup table with
     * (must have 4 elements)
     */
    luti2(const std::vector<T>& init_data) {
        if (init_data.size() != 4) [[unlikely]] {
            throw std::invalid_argument("Initializer vector must have exactly"
                " 4 elements for luti2");
        }
        std::memcpy(data.data(), init_data.data(), 4 * sizeof(T));
    }
    /** --------------------------------------------------------------------------------- Construct from Initializer List
     * @brief Constructor that initializes the lookup table with the provided initializer
     * list of values. The input initializer list must have exactly 4 elements, which
     * will be copied into the internal data array.
     * @param init_list The initializer list of values to initialize the lookup table
     * with (must have 4 elements)
     */
    luti2(std::initializer_list<T> init_list) {
        if (init_list.size() != 4) [[unlikely]] {
            throw std::invalid_argument("Initializer list must have exactly"
                " 4 elements for luti2");
        }
        std::copy(init_list.begin(), init_list.end(), data.begin());
    }
    /** --------------------------------------------------------------------------------- Construct from Individual Values
     * @brief Constructor that initializes the lookup table with the provided individual
     * values. The constructor takes 4 individual values as parameters, which will be
     * copied into the internal data array.
     * @param val0 The first value to initialize the lookup table with
     * @param val1 The second value to initialize the lookup table with
     * @param val2 The third value to initialize the lookup table with
     * @param val3 The fourth value to initialize the lookup table with
     */
    luti2(T val0, T val1, T val2, T val3) : data{val0, val1, val2, val3} {}
    /** --------------------------------------------------------------------------------- Get Value
     * @brief Gets a reference to the value at the specified index in the lookup table.
     * The index should be in the range [0, 3], and the function will return a reference
     * to the corresponding value in the internal data array.
     * @param index The index of the value to get (should be in the range [0, 3])
     * @return A reference to the value at the specified index in the lookup table
     */
    T& operator[](size_t index) {
        if (index >= data.size()) [[unlikely]] {
            throw std::out_of_range("Index out of range for luti2");
        }
        return data[index];
    }
    /** --------------------------------------------------------------------------------- Get Value (const)
     * @brief Gets a const reference to the value at the specified index in the lookup
     * table. The index should be in the range [0, 3], and the function will return a const reference to the corresponding value in the internal data array.
     * @param index The index of the value to get (should be in the range [0, 3])
     * @return A const reference to the value at the specified index in the lookup table
     */
    const T& operator[](size_t index) const {
        if (index >= data.size()) [[unlikely]] {
            throw std::out_of_range("Index out of range for luti2");
        }
        return data[index];
    }
};
/** --------------------------------------------------------------------------------------------------------- z_stream 
 * @class z_stream
 * @brief A helper class to manage a stream of z[n] registers for use in assembly kernels.
 * This class abstracts away the details of allocating and managing the aligned memory needed 
 * for streaming data into the z[n] registers, and provides a convenient interface for working
 * with streams of vectors of various types and various operations.
 */
template<ValidZType T>
class z_stream {
private:
    /// @brief Type-erased pointer to allocate the aligned memory for the z_stream
    T* data_ = nullptr;
    /// @brief Number of zvecs (512-bit vectors) allocated in the stream
    size_t num_zvecs_ = 0;
    /** --------------------------------------------------------------------------------- Aligned Memory Allocation
     * @brief Allocates aligned memory for the z_stream based on the number of zvecs
     * needed. Each zvec is 64 bytes, so the total size allocated is num_zvecs * 64
     * bytes, rounded up to the nearest multiple of 64 for alignment.
     * @param size The total size in bytes needed for the z_stream (should be
     * num_zvecs * 64) 
     */
    void alignedAlloc(size_t size) {
        size <<= 6; // Multiply by 64 to get the total size in bytes
        // Align to the size of the full za.b tile (4096 bytes) to ensure optimal
        // access patterns for streaming
        data_ = static_cast<T*>(std::aligned_alloc(4096, size));
        if (!data_) [[unlikely]] {
            std::string error_msg = "Failed to allocate memory for z_stream:"
                    " requested size " + std::to_string(size) + " bytes";
            throw std::runtime_error(error_msg);
        }
    }
public:
    /** --------------------------------------------------------------------------------- Constructor
     * @brief Constructor for z_stream. Allocates memory for the specified number of
     * zvecs, each of which is 512 bits (64 bytes).
     * @param num_zvecs The number of 512-bit vectors (zvecs) to allocate in the stream
     */
    z_stream(size_t num_zvecs) : num_zvecs_(num_zvecs) {
        alignedAlloc(num_zvecs_);
    }
    /** --------------------------------------------------------------------------------- Destructor
     * @brief Destructor for z_stream. Frees the allocated memory for the stream.
     */
    ~z_stream() {
        if (data_) {
            std::free(data_);
        }
    }
    // Delete copy constructor and copy assignment operator to prevent accidental copying
    z_stream(const z_stream&) = delete;
    z_stream& operator=(const z_stream&) = delete;
    /** --------------------------------------------------------------------------------- Get Pointer
     * @brief Gets a pointer to the raw data of the z_stream. This pointer can be used
     * to load data into the zvecs for streaming into assembly kernels. The pointer is
     * typed as T* for convenience, but it actually points to the aligned memory
     * allocated for the z_stream, which is suitable for use with the z[n] registers.
     * @tparam T The SIMD vector type of the z_stream (e.g., simd_float16, simd_uchar64,
     * etc.).
     */
    template<SIMDVectorEquivalent U = T>
        requires std::same_as<U, T> || std::is_convertible_v<T, U>
    U* ptr() {
        return data_;
    }
    /** --------------------------------------------------------------------------------- Get Const Pointer
     * @brief Gets a const pointer to the raw data of the z_stream. This can be used
     * when the z_stream is only being read from (e.g., for streaming data into kernels
     * without modifying it).
     * @tparam T The SIMD vector type of the z_stream (e.g., simd_float16, simd_uchar64,
     * etc.).
     */
    template<SIMDVectorEquivalent U = T>
        requires std::same_as<U, T> || std::is_convertible_v<T, U>
    const U* ptr() const {
        return data_;
    }
    /** --------------------------------------------------------------------------------- Clone
     * @brief Creates a clone of the z_stream. This is useful if you want to create a
     * copy of the stream with the same number of zvecs, optionally copying the data
     * as well.
     * @param copy_bytes If true, the data from the original stream will be copied to the
     * new stream. If false, the clone will be pointed to the same data without copying
     * (use with caution). This is useful for cases where you want to have multiple
     * z_stream instances that share the same underlying data without the overhead of
     * copying, but it should be used carefully to avoid unintended side effects from
     * modifying the shared data.
     */
    z_stream clone(bool copy_bytes = false) const {
        z_stream copy(num_zvecs_);
        if (copy_bytes) {
            std::memcpy(copy.data_, data_, num_zvecs_ << 6); // num_zvecs * 64 bytes
        } else {
            std::free(copy.data_);
            copy.data_ = data_;
        }
        return copy;
    }
};
} // namespace ane
#else
#error "This library is only supported on Apple Silicon (aarch64 architecture)."
#endif // defined(__aarch64__) && defined(__APPLE__)
