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
void stream_exec(const uint8_t* data, size_t size);
}
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
/** --------------------------------------------------------------------------------------------------------- Interpreter Namespace
 * @brief The interpreter namespace contains the assembly code for the bytecode interpreter.
 * @param data Pointer to the bytecode stream to execute
 * @param size Size of the bytecode stream in bytes
 */
namespace interpreter {
__attribute__((naked, used, noinline))
void stream_exec(const uint8_t* data, size_t size) {
    asm volatile(R"(
        .arch armv9-a+sme2+sve2+sme-lutv2
        stp     x29, x30, [sp, #-128]!
        mov     x29, sp
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        stp     x23, x24, [sp, #48]
        stp     x25, x26, [sp, #64]
        stp     d8,  d9,  [sp, #80]
        stp     d10, d11, [sp, #96]
        stp     d12, d13, [sp, #112]
        // ── x0 = data, x1 = size ──
        mov     x19, x0                // instruction pointer
        add     x21, x0, x1            // end = data + size
        // ── Enter streaming mode ──
        smstart
        // ── Load jump table base (must be after smstart, adrp is fine in streaming) ──
        adrp    x25, Ljump_table@PAGE
        add     x25, x25, Ljump_table@PAGEOFF
        // ── Dispatch ──
    Ldispatch:
        cmp     x19, x21               // IP >= bytecodes end?
        b.hs    Lexit                  // done — no explicit halt needed
        ldrb    w9, [x19], #1          // fetch opcode, advance IP
        ldrsw   x10, [x25, x9, lsl #2] // load relative offset (32-bit signed)
        add     x10, x25, x10          // absolute target = table_base + offset
        br      x10
    // ================================================================
    // Jump Table (PC-relative offsets, avoids text relocations on macOS)
    // ================================================================
    .p2align 2
    Ljump_table:
        .long   Lexit          - Ljump_table  // 0x00 (reserved)
        .long   Lop_zero_za    - Ljump_table  // 0x01
        .long   Lop_acc_smopa  - Ljump_table  // 0x02
        .long   Lop_acc_umopa  - Ljump_table  // 0x03
        .long   Lop_acc_usmopa - Ljump_table  // 0x04
        .long   Lop_acc_sumopa - Ljump_table  // 0x05
        .long   Lop_store_tiles - Ljump_table // 0x06
        .long   Lop_load_rows_i8 - Ljump_table // 0x07
        .long   Lop_load_cols_i8 - Ljump_table // 0x08
        .long   Lop_smopa_2x2  - Ljump_table // 0x09
        .long   Lop_umopa_2x2  - Ljump_table // 0x0A
        .long   Lop_usmopa_2x2 - Ljump_table // 0x0B
        .long   Lop_load_bias  - Ljump_table // 0x0C
        .long   Lop_scale_store - Ljump_table // 0x0D
        .long   Lop_dense_scale_relu_i8 - Ljump_table // 0x0E
        .long   Lop_dense_scale_i8        - Ljump_table // 0x0F
        .long   Lop_elementwise_add_fp32  - Ljump_table // 0x10
        .long   Lop_elementwise_scaled_add_fp32 - Ljump_table // 0x11
        .long   Lop_elementwise_mul_fp32 - Ljump_table // 0x12
        .long   Lop_relu_backward_fp32 - Ljump_table // 0x13
        .long   Lop_quantize_fp32_i8 - Ljump_table // 0x14
        .long   Lop_pack_rows_i8 - Ljump_table // 0x15
        .long   Lop_pack_cols_i8 - Ljump_table // 0x16
        .long   Lop_scatter_tile_fp32 - Ljump_table // 0x17
        .long   Lop_transpose_fp32 - Ljump_table // 0x18
        .long   Lop_softmax_argmax_fp32 - Ljump_table // 0x19
        .long   Lop_luti4 - Ljump_table // 0x1A
        .long   Lop_luti2 - Ljump_table // 0x1B
        .long   Lop_dense_fp32 - Ljump_table // 0x1C
    // ================================================================
    // EXIT — reached end of bytecodes
    // ================================================================
    Lexit:
        smstop
        ldp     d12, d13, [sp, #112]
        ldp     d10, d11, [sp, #96]
        ldp     d8,  d9,  [sp, #80]
        ldp     x25, x26, [sp, #64]
        ldp     x23, x24, [sp, #48]
        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #128
        ret
    // ================================================================
    // ZERO_ZA (0x01)
    // ================================================================
    Lop_zero_za:
        zero    {za}
        b       Ldispatch
    // ================================================================
    // ACC_SMOPA (0x02) — fused ld1b+smopa loop
    // Encoding: [0x02] [k_steps: 4 bytes LE]
    // Operands: row_ptr, col_ptr
    // ================================================================
    Lop_acc_smopa:
        ldr     w22, [x19]             // k_steps (4 bytes LE)
        add     x19, x19, #4           // advance IP past immediate
        ldr     x8, [x19], #8          // consume row_ptr
        ldr     x11, [x19], #8         // consume col_ptr
        ptrue   p0.b
        cbz     w22, Ldispatch
    Lacc_smopa_loop:
        ld1b    {z0.b}, p0/z, [x8]
        ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
        ld1b    {z2.b}, p0/z, [x11]
        ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
        smopa   za0.s, p0/m, p0/m, z0.b, z2.b
        smopa   za1.s, p0/m, p0/m, z0.b, z3.b
        smopa   za2.s, p0/m, p0/m, z1.b, z2.b
        smopa   za3.s, p0/m, p0/m, z1.b, z3.b
        addvl   x8, x8, #2
        addvl   x11, x11, #2
        sub     w22, w22, #1
        cbnz    w22, Lacc_smopa_loop
        b       Ldispatch
    // ================================================================
    // ACC_UMOPA (0x03) — fused ld1b+umopa loop
    // ================================================================
    Lop_acc_umopa:
        ldr     w22, [x19]
        add     x19, x19, #4
        ldr     x8, [x19], #8
        ldr     x11, [x19], #8
        ptrue   p0.b
        cbz     w22, Ldispatch
    Lacc_umopa_loop:
        ld1b    {z0.b}, p0/z, [x8]
        ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
        ld1b    {z2.b}, p0/z, [x11]
        ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
        umopa   za0.s, p0/m, p0/m, z0.b, z2.b
        umopa   za1.s, p0/m, p0/m, z0.b, z3.b
        umopa   za2.s, p0/m, p0/m, z1.b, z2.b
        umopa   za3.s, p0/m, p0/m, z1.b, z3.b
        addvl   x8, x8, #2
        addvl   x11, x11, #2
        sub     w22, w22, #1
        cbnz    w22, Lacc_umopa_loop
        b       Ldispatch
    // ================================================================
    // ACC_USMOPA (0x04) — fused ld1b+usmopa loop
    // ================================================================
    Lop_acc_usmopa:
        ldr     w22, [x19]
        add     x19, x19, #4
        ldr     x8, [x19], #8
        ldr     x11, [x19], #8
        ptrue   p0.b
        cbz     w22, Ldispatch
    Lacc_usmopa_loop:
        ld1b    {z0.b}, p0/z, [x8]
        ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
        ld1b    {z2.b}, p0/z, [x11]
        ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
        usmopa  za0.s, p0/m, p0/m, z0.b, z2.b
        usmopa  za1.s, p0/m, p0/m, z0.b, z3.b
        usmopa  za2.s, p0/m, p0/m, z1.b, z2.b
        usmopa  za3.s, p0/m, p0/m, z1.b, z3.b
        addvl   x8, x8, #2
        addvl   x11, x11, #2
        sub     w22, w22, #1
        cbnz    w22, Lacc_usmopa_loop
        b       Ldispatch
    // ================================================================
    // ACC_SUMOPA (0x05) — fused ld1b+usmopa loop, signed rows × unsigned cols
    // No hardware sumopa exists. We use usmopa (unsigned × signed) with
    // swapped operand registers: cols (unsigned) as first arg, rows (signed)
    // as second. This transposes the tile accumulation — the caller must
    // account for the transposed layout when reading ZA tiles.
    // ================================================================
    Lop_acc_sumopa:
        ldr     w22, [x19]
        add     x19, x19, #4
        ldr     x8, [x19], #8
        ldr     x11, [x19], #8
        ptrue   p0.b
        cbz     w22, Ldispatch
    Lacc_sumopa_loop:
        ld1b    {z0.b}, p0/z, [x8]
        ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
        ld1b    {z2.b}, p0/z, [x11]
        ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
        usmopa  za0.s, p0/m, p0/m, z2.b, z0.b
        usmopa  za1.s, p0/m, p0/m, z3.b, z0.b
        usmopa  za2.s, p0/m, p0/m, z2.b, z1.b
        usmopa  za3.s, p0/m, p0/m, z3.b, z1.b
        addvl   x8, x8, #2
        addvl   x11, x11, #2
        sub     w22, w22, #1
        cbnz    w22, Lacc_sumopa_loop
        b       Ldispatch
    // ================================================================
    // STORE_TILES (0x06)
    // Stores the 2x2 tile group as (2*SVLs) × (2*SVLs) row-major int32.
    // Output is z_vector* — each z_vector is 64 bytes = 16 int32s = one SVLs row.
    // A full row of output is 2 z_vectors wide (left half + right half).
    // Consumes one operand: destination z_vector pointer.
    //
    // Layout (M4, SVLs=16, GROUP_DIM=32):
    //   Row  0: z_vec[0]=za0_row0_left, z_vec[1]=za1_row0_right
    //   Row  1: z_vec[2]=za0_row1_left, z_vec[3]=za1_row1_right
    //   ...
    //   Row 15: z_vec[30]=za0_row15_left, z_vec[31]=za1_row15_right
    //   Row 16: z_vec[32]=za2_row0_left, z_vec[33]=za3_row0_right
    //   ...
    //   Row 31: z_vec[62]=za2_row15_left, z_vec[63]=za3_row15_right
    // ================================================================
    Lop_store_tiles:
        ldr     x8, [x19], #8          // consume destination pointer
        ptrue   p0.s
        cntw    x9                     // SVLs (16 on M4)
        lsl     x10, x9, #3           // row stride in bytes: 2*SVLs*4 = SVLs*8
        // ---- Upper half: za0 (left) + za1 (right), SVLs rows ----
        mov     w12, #0
    Lse_store_upper:
        mova    {z4.s-z7.s}, za0h.s[w12, 0:3]
        mova    {z8.s-z11.s}, za1h.s[w12, 0:3]
        st1w    {z4.s}, p0, [x8]
        st1w    {z8.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        st1w    {z5.s}, p0, [x8]
        st1w    {z9.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        st1w    {z6.s}, p0, [x8]
        st1w    {z10.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        st1w    {z7.s}, p0, [x8]
        st1w    {z11.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        add     w12, w12, #4
        cmp     w12, w9
        b.lt    Lse_store_upper
        // ---- Lower half: za2 (left) + za3 (right), SVLs rows ----
        mov     w12, #0
    Lse_store_lower:
        mova    {z4.s-z7.s}, za2h.s[w12, 0:3]
        mova    {z8.s-z11.s}, za3h.s[w12, 0:3]
        st1w    {z4.s}, p0, [x8]
        st1w    {z8.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        st1w    {z5.s}, p0, [x8]
        st1w    {z9.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        st1w    {z6.s}, p0, [x8]
        st1w    {z10.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        st1w    {z7.s}, p0, [x8]
        st1w    {z11.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        add     w12, w12, #4
        cmp     w12, w9
        b.lt    Lse_store_lower
        b       Ldispatch
    // ================================================================
    // LOAD_ROWS_I8 (0x07) — load 128 bytes into z0, z1
    // ================================================================
    Lop_load_rows_i8:
        ldr     x8, [x19], #8
        ptrue   p0.b
        ld1b    {z0.b}, p0/z, [x8]
        cntb    x9
        ld1b    {z1.b}, p0/z, [x8, x9]
        b       Ldispatch
    // ================================================================
    // LOAD_COLS_I8 (0x08) — load 128 bytes into z2, z3
    // ================================================================
    Lop_load_cols_i8:
        ldr     x8, [x19], #8
        ptrue   p0.b
        ld1b    {z2.b}, p0/z, [x8]
        cntb    x9
        ld1b    {z3.b}, p0/z, [x8, x9]
        b       Ldispatch
    // ================================================================
    // SMOPA_2x2 (0x09) — 4× smopa, no load
    // ================================================================
    Lop_smopa_2x2:
        ptrue   p0.b
        smopa   za0.s, p0/m, p0/m, z0.b, z2.b
        smopa   za1.s, p0/m, p0/m, z0.b, z3.b
        smopa   za2.s, p0/m, p0/m, z1.b, z2.b
        smopa   za3.s, p0/m, p0/m, z1.b, z3.b
        b       Ldispatch
    // ================================================================
    // UMOPA_2x2 (0x0A) — 4× umopa, no load
    // ================================================================
    Lop_umopa_2x2:
        ptrue   p0.b
        umopa   za0.s, p0/m, p0/m, z0.b, z2.b
        umopa   za1.s, p0/m, p0/m, z0.b, z3.b
        umopa   za2.s, p0/m, p0/m, z1.b, z2.b
        umopa   za3.s, p0/m, p0/m, z1.b, z3.b
        b       Ldispatch
    // ================================================================
    // USMOPA_2x2 (0x0B) — 4× usmopa, no load
    // ================================================================
    Lop_usmopa_2x2:
        ptrue   p0.b
        usmopa  za0.s, p0/m, p0/m, z0.b, z2.b
        usmopa  za1.s, p0/m, p0/m, z0.b, z3.b
        usmopa  za2.s, p0/m, p0/m, z1.b, z2.b
        usmopa  za3.s, p0/m, p0/m, z1.b, z3.b
        b       Ldispatch
    // ================================================================
    // LOAD_BIAS (0x0C)
    // Loads int32 bias data from memory into ZA tiles (reverse of store_tiles).
    // Same layout as store: (2*SVLs) × (2*SVLs) row-major int32.
    // Consumes one operand: source pointer.
    // This replaces zero_za when you want to accumulate on top of bias.
    // ================================================================
    Lop_load_bias:
        ldr     x8, [x19], #8          // consume source pointer
        ptrue   p0.s
        cntw    x9                     // SVLs (16 on M4)
        lsl     x10, x9, #3           // row stride in bytes: 2*SVLs*4
        // ---- Upper half: → za0 (left) + za1 (right), SVLs rows ----
        mov     w12, #0
    Lse_loadbias_upper:
        ld1w    {z4.s}, p0/z, [x8]
        ld1w    {z8.s}, p0/z, [x8, x9, lsl #2]
        add     x8, x8, x10
        ld1w    {z5.s}, p0/z, [x8]
        ld1w    {z9.s}, p0/z, [x8, x9, lsl #2]
        add     x8, x8, x10
        ld1w    {z6.s}, p0/z, [x8]
        ld1w    {z10.s}, p0/z, [x8, x9, lsl #2]
        add     x8, x8, x10
        ld1w    {z7.s}, p0/z, [x8]
        ld1w    {z11.s}, p0/z, [x8, x9, lsl #2]
        add     x8, x8, x10
        mova    za0h.s[w12, 0:3], {z4.s-z7.s}
        mova    za1h.s[w12, 0:3], {z8.s-z11.s}
        add     w12, w12, #4
        cmp     w12, w9
        b.lt    Lse_loadbias_upper
        // ---- Lower half: → za2 (left) + za3 (right), SVLs rows ----
        mov     w12, #0
    Lse_loadbias_lower:
        ld1w    {z4.s}, p0/z, [x8]
        ld1w    {z8.s}, p0/z, [x8, x9, lsl #2]
        add     x8, x8, x10
        ld1w    {z5.s}, p0/z, [x8]
        ld1w    {z9.s}, p0/z, [x8, x9, lsl #2]
        add     x8, x8, x10
        ld1w    {z6.s}, p0/z, [x8]
        ld1w    {z10.s}, p0/z, [x8, x9, lsl #2]
        add     x8, x8, x10
        ld1w    {z7.s}, p0/z, [x8]
        ld1w    {z11.s}, p0/z, [x8, x9, lsl #2]
        add     x8, x8, x10
        mova    za2h.s[w12, 0:3], {z4.s-z7.s}
        mova    za3h.s[w12, 0:3], {z8.s-z11.s}
        add     w12, w12, #4
        cmp     w12, w9
        b.lt    Lse_loadbias_lower
        b       Ldispatch
    // ================================================================
    // SCALE_STORE (0x0D)
    // Reads ZA tiles (int32), converts to float32, multiplies by a scalar
    // scale factor, and stores as float32.
    // Encoding: [0x0D] [scale: 4 bytes LE float]
    // Consumes one operand: destination pointer.
    //
    // This is the dequantization step: int32_accum × scale → float32 output.
    // The scale is a float immediate in the bytecode stream.
    // ================================================================
    Lop_scale_store:
        ldr     s16, [x19]             // load float scale into s16 (bottom of z16)
        add     x19, x19, #4           // advance IP past float immediate
        ldr     x8, [x19], #8          // consume destination pointer
        ptrue   p0.s
        cntw    x9                     // SVLs
        lsl     x10, x9, #3           // row stride
        // Broadcast scale into z16 for fmul
        mov     z16.s, s16
        // ---- Upper half: za0 (left) + za1 (right) ----
        mov     w12, #0
    Lse_scalestore_upper:
        // Extract 4 rows from za0, za1
        mova    {z4.s-z7.s}, za0h.s[w12, 0:3]
        mova    {z8.s-z11.s}, za1h.s[w12, 0:3]
        // Convert int32 → float32 (scvtf) and multiply by scale
        scvtf   z4.s, p0/m, z4.s
        fmul    z4.s, z4.s, z16.s
        scvtf   z5.s, p0/m, z5.s
        fmul    z5.s, z5.s, z16.s
        scvtf   z6.s, p0/m, z6.s
        fmul    z6.s, z6.s, z16.s
        scvtf   z7.s, p0/m, z7.s
        fmul    z7.s, z7.s, z16.s
        scvtf   z8.s, p0/m, z8.s
        fmul    z8.s, z8.s, z16.s
        scvtf   z9.s, p0/m, z9.s
        fmul    z9.s, z9.s, z16.s
        scvtf   z10.s, p0/m, z10.s
        fmul    z10.s, z10.s, z16.s
        scvtf   z11.s, p0/m, z11.s
        fmul    z11.s, z11.s, z16.s
        // Store float32 results
        st1w    {z4.s}, p0, [x8]
        st1w    {z8.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        st1w    {z5.s}, p0, [x8]
        st1w    {z9.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        st1w    {z6.s}, p0, [x8]
        st1w    {z10.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        st1w    {z7.s}, p0, [x8]
        st1w    {z11.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        add     w12, w12, #4
        cmp     w12, w9
        b.lt    Lse_scalestore_upper
        // ---- Lower half: za2 (left) + za3 (right) ----
        mov     w12, #0
    Lse_scalestore_lower:
        mova    {z4.s-z7.s}, za2h.s[w12, 0:3]
        mova    {z8.s-z11.s}, za3h.s[w12, 0:3]
        scvtf   z4.s, p0/m, z4.s
        fmul    z4.s, z4.s, z16.s
        scvtf   z5.s, p0/m, z5.s
        fmul    z5.s, z5.s, z16.s
        scvtf   z6.s, p0/m, z6.s
        fmul    z6.s, z6.s, z16.s
        scvtf   z7.s, p0/m, z7.s
        fmul    z7.s, z7.s, z16.s
        scvtf   z8.s, p0/m, z8.s
        fmul    z8.s, z8.s, z16.s
        scvtf   z9.s, p0/m, z9.s
        fmul    z9.s, z9.s, z16.s
        scvtf   z10.s, p0/m, z10.s
        fmul    z10.s, z10.s, z16.s
        scvtf   z11.s, p0/m, z11.s
        fmul    z11.s, z11.s, z16.s
        st1w    {z4.s}, p0, [x8]
        st1w    {z8.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        st1w    {z5.s}, p0, [x8]
        st1w    {z9.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        st1w    {z6.s}, p0, [x8]
        st1w    {z10.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        st1w    {z7.s}, p0, [x8]
        st1w    {z11.s}, p0, [x8, x9, lsl #2]
        add     x8, x8, x10
        add     w12, w12, #4
        cmp     w12, w9
        b.lt    Lse_scalestore_lower
        b       Ldispatch
    // ================================================================
    // DENSE_FUSED_I8 (0x0E) — fully fused fp32→i8 quantize+pack+smopa+dequant(+relu)→fp32
    // Full-matrix version with internal tiling (arbitrary M, N, K).
    // Bytecode: [0x0E][M:u32][N:u32][K:u32][scale:f32][flags:u8] = 17 bytes
    //   M, N, K = full matrix dimensions (arbitrary, not limited to 32)
    //   scale = dequant scale (typically 1.0)
    //   flags bit 0: 1=apply ReLU, 0=no ReLU
    // Operands: A (M×K row-major, stride K), B (K×N row-major, stride N), C (M×N row-major, stride N)
    // Stack layout (dynamic):
    //   [sp+0   .. +128):   context block (saved ptrs, dims, scales)
    //   [sp+128 .. +384):   packed scratch (256 bytes: 128 rows + 128 cols)
    //   [sp+384 .. +4480):  tile_out scratch (4096 bytes = 32×32×4)
    //   [sp+4480 .. +4480+M_pad*K_pad): qa (quantized A, int8)
    //   [sp+4480+M_pad*K_pad .. +4480+M_pad*K_pad+K_pad*N_pad): qb (quantized B, int8)
    // ================================================================
    Lop_dense_scale_relu_i8:
        // ── Parse immediates + operands ──
        ldr     w0, [x19]              // M
        ldr     w1, [x19, #4]          // N
        ldr     w2, [x19, #8]          // K
        ldr     s16, [x19, #12]        // scale (f32)
        ldrb    w18, [x19, #16]        // flags (bit 0 = relu)
        add     x19, x19, #17
        ldr     x5, [x19], #8          // A
        ldr     x6, [x19], #8          // B
        ldr     x7, [x19], #8          // C
        // ── Compute derived values ──
        add     w14, w2, #3
        and     w14, w14, #0xFFFFFFFC  // K_pad = (K+3) & ~3
        lsr     w15, w14, #2           // k_steps = K_pad / 4
        add     w3, w0, #31
        and     w3, w3, #0xFFFFFFE0   // M_pad = (M+31) & ~31
        add     w4, w1, #31
        and     w4, w4, #0xFFFFFFE0   // N_pad = (N+31) & ~31
        ptrue   p0.s
        cntw    x9                     // SVLs = 16
        // ── Allocate stack frame ──
        //   fixed: 128 (context) + 256 (packed) + 4096 (tile_out) = 4480
        //   dynamic: M_pad*K_pad (qa) + K_pad*N_pad (qb)
        mul     w10, w3, w14           // M_pad * K_pad
        mul     w11, w14, w4           // K_pad * N_pad
        add     w10, w10, w11          // qa + qb total
        add     x10, x10, #4096       // + fixed (4480 = 4096 + 384)
        add     x10, x10, #384
        add     x10, x10, #63
        and     x10, x10, #~63        // round up to 64-byte alignment
        sub     sp, sp, x10
        // Save context to [sp+0..127]
        stp     x5, x6, [sp, #0]      // [0] A, [8] B
        str     x7, [sp, #16]         // [16] C
        stp     w0, w1, [sp, #24]     // [24] M, [28] N
        stp     w2, w14, [sp, #32]    // [32] K, [36] K_pad
        stp     w4, w3, [sp, #40]     // [40] N_pad, [44] M_pad
        stp     w15, w18, [sp, #48]   // [48] k_steps, [52] flags
        str     s16, [sp, #56]        // [56] scale
        str     x10, [sp, #72]        // [72] frame_size
        // ── Compute buffer pointers ──
        add     x11, sp, #4096        // qa_base = sp + 4480
        add     x11, x11, #384
        mul     w16, w3, w14          // M_pad * K_pad
        add     x13, x11, x16         // qb_base = qa_base + M_pad*K_pad
        // Zero qa and qb
        add     x8, x16, x11         // end of qa  (recompute: qa_base + M_pad*K_pad = qb_base)
        mul     w17, w14, w4          // K_pad * N_pad
        add     x8, x16, x17         // total bytes to zero = M_pad*K_pad + K_pad*N_pad
        mov     x16, x11             // cursor at qa_base
    Lfuse_zero_scratch:
        cbz     x8, Lfuse_zero_done
        stp     xzr, xzr, [x16], #16
        sub     x8, x8, #16
        cbnz    x8, Lfuse_zero_scratch
    Lfuse_zero_done:
        // ════════════════════════════════════════════════════════════
        // Phase 1: Absmax of A (M rows × K cols, contiguous stride K)
        // ════════════════════════════════════════════════════════════
        fmov    z18.s, #0.0            // running absmax
        mov     x8, x5                 // A cursor
        mul     x16, x0, x2            // M * K total elements
        mov     x17, xzr               // index
        whilelt p1.s, xzr, x16
    Lfuse_absmax_a:
        ld1w    {z0.s}, p1/z, [x8]
        fabs    z0.s, p1/m, z0.s
        fmax    z18.s, p1/m, z18.s, z0.s
        add     x8, x8, x9, lsl #2
        incw    x17
        whilelt p1.s, x17, x16
        b.first Lfuse_absmax_a
        fmaxv   s18, p0, z18.s        // absmax_a → s18
        // Compute scale_a = 127/absmax_a, inv_a = absmax_a/127
        fmov    s17, #1.0
        fcmp    s18, #0.0
        fcsel   s18, s17, s18, eq
        movz    w8, #0x42FE, lsl #16   // 127.0f
        fmov    s17, w8
        fdiv    s28, s17, s18          // s28 = 127/absmax_a (scale_a)
        fdiv    s29, s18, s17          // s29 = absmax_a/127 (inv_a)
        str     s29, [sp, #60]         // save inv_a at [60]
        // ════════════════════════════════════════════════════════════
        // Phase 2: Absmax of B (K rows × N cols, contiguous stride N)
        // ════════════════════════════════════════════════════════════
        fmov    z18.s, #0.0
        ldr     x6, [sp, #8]          // reload B
        ldr     w2, [sp, #32]         // K
        ldr     w1, [sp, #28]         // N
        lsl     x3, x1, #2            // stride N in bytes
        mov     w12, #0
    Lfuse_absmax_b:
        cmp     w12, w2
        b.ge    Lfuse_absmax_b_done
        mov     x8, x6
        mov     x17, xzr
        whilelt p1.s, xzr, x1
    Lfuse_absmax_b_inner:
        ld1w    {z0.s}, p1/z, [x8]
        fabs    z0.s, p1/m, z0.s
        fmax    z18.s, p1/m, z18.s, z0.s
        add     x8, x8, x9, lsl #2
        incw    x17
        whilelt p1.s, x17, x1
        b.first Lfuse_absmax_b_inner
        add     x6, x6, x3            // next B row (stride N*4 bytes)
        add     w12, w12, #1
        b       Lfuse_absmax_b
    Lfuse_absmax_b_done:
        fmaxv   s18, p0, z18.s
        fmov    s17, #1.0
        fcmp    s18, #0.0
        fcsel   s18, s17, s18, eq
        movz    w8, #0x42FE, lsl #16
        fmov    s17, w8
        fdiv    s30, s17, s18          // s30 = 127/absmax_b (scale_b)
        fdiv    s17, s18, s17          // s17 = inv_b
        str     s17, [sp, #64]         // save inv_b at [64]
        // ════════════════════════════════════════════════════════════
        // Phase 3: Quantize A → qa (M_pad rows × K_pad cols, zero-padded)
        // A is M rows of K contiguous floats, rows beyond M stay zero.
        // ════════════════════════════════════════════════════════════
        ldr     x5, [sp, #0]          // A
        ldr     w0, [sp, #24]         // M
        ldr     w2, [sp, #32]         // K
        ldr     w14, [sp, #36]        // K_pad
        add     x11, sp, #4096        // qa_base = sp + 4480
        add     x11, x11, #384
        mov     z16.s, s28             // broadcast scale_a
        mov     w12, #0
    Lfuse_quant_a_row:
        cmp     w12, w0
        b.ge    Lfuse_quant_a_done
        mov     x8, x5                 // src cursor (row start in A)
        mov     x13, x11               // dst cursor (row start in qa)
        mov     x17, xzr
        whilelt p1.s, xzr, x2
    Lfuse_quant_a_inner:
        ld1w    {z0.s}, p1/z, [x8]
        fmul    z0.s, z0.s, z16.s
        frinti  z0.s, p1/m, z0.s
        fcvtzs  z0.s, p1/m, z0.s
        mov     z1.s, #127
        mov     z2.s, #-127
        smin    z0.s, p1/m, z0.s, z1.s
        smax    z0.s, p1/m, z0.s, z2.s
        st1b    {z0.s}, p1, [x13]
        add     x8, x8, x9, lsl #2
        add     x13, x13, x9
        incw    x17
        whilelt p1.s, x17, x2
        b.first Lfuse_quant_a_inner
        lsl     x8, x2, #2            // K * 4 bytes
        add     x5, x5, x8            // next A row
        add     x11, x11, x14         // next qa row (stride K_pad)
        add     w12, w12, #1
        b       Lfuse_quant_a_row
    Lfuse_quant_a_done:
        // ════════════════════════════════════════════════════════════
        // Phase 4: Quantize B → qb (K_pad rows × N_pad cols, zero-padded)
        // B is K rows of N contiguous floats, stride N.
        // ════════════════════════════════════════════════════════════
        ldr     x6, [sp, #8]          // B
        ldr     w2, [sp, #32]         // K
        ldr     w1, [sp, #28]         // N
        ldr     w14, [sp, #36]        // K_pad
        ldr     w4, [sp, #40]         // N_pad
        lsl     x3, x1, #2            // N * 4 bytes (B row stride)
        ldr     w16, [sp, #44]        // M_pad
        mul     w10, w16, w14         // M_pad * K_pad = qa size
        add     x11, sp, #4096         // qa_base = sp + 4480
        add     x11, x11, #384
        add     x11, x11, x10         // qb_base
        mov     z16.s, s30             // broadcast scale_b
        mov     w12, #0
    Lfuse_quant_b_row:
        cmp     w12, w2
        b.ge    Lfuse_quant_b_done
        mov     x8, x6                 // src cursor
        mov     x13, x11               // dst cursor
        mov     x17, xzr
        whilelt p1.s, xzr, x1
    Lfuse_quant_b_inner:
        ld1w    {z0.s}, p1/z, [x8]
        fmul    z0.s, z0.s, z16.s
        frinti  z0.s, p1/m, z0.s
        fcvtzs  z0.s, p1/m, z0.s
        mov     z1.s, #127
        mov     z2.s, #-127
        smin    z0.s, p1/m, z0.s, z1.s
        smax    z0.s, p1/m, z0.s, z2.s
        st1b    {z0.s}, p1, [x13]
        add     x8, x8, x9, lsl #2
        add     x13, x13, x9
        incw    x17
        whilelt p1.s, x17, x1
        b.first Lfuse_quant_b_inner
        add     x6, x6, x3            // next B row (stride N*4 bytes)
        add     x11, x11, x4          // next qb row (stride N_pad bytes)
        add     w12, w12, #1
        b       Lfuse_quant_b_row
    Lfuse_quant_b_done:
        // ════════════════════════════════════════════════════════════
        // Phase 5: Compute combined dequant scale
        // ════════════════════════════════════════════════════════════
        ldr     s28, [sp, #60]         // inv_a
        ldr     s29, [sp, #64]         // inv_b
        ldr     s30, [sp, #56]         // user scale
        fmul    s28, s28, s29
        fmul    s28, s28, s30          // combined = inv_a * inv_b * scale
        str     s28, [sp, #56]         // overwrite scale with combined
        // ════════════════════════════════════════════════════════════
        // Phase 6: Tile loop — for each 32×32 tile block
        //   x0 = ti (row tile offset), x1 = tj (col tile offset)
        //   Iterates ti = 0..M_pad-1 step 32, tj = 0..N_pad-1 step 32
        // ════════════════════════════════════════════════════════════
        ldr     w3, [sp, #44]          // M_pad
        ldr     w4, [sp, #40]          // N_pad
        ldr     w14, [sp, #36]         // K_pad
        ldr     w15, [sp, #48]         // k_steps
        mov     w0, #0                 // ti = 0
    Lfuse_tile_row:
        cmp     w0, w3                 // ti < M_pad
        b.ge    Lfuse_tile_done
        mov     w1, #0                 // tj = 0
    Lfuse_tile_col:
        cmp     w1, w4                 // tj < N_pad
        b.ge    Lfuse_tile_row_next
        // ── Save tile coords to context ──
        stp     w0, w1, [sp, #80]     // [80] ti, [84] tj
        // ── Compute qa_tile = qa_base + ti*K_pad ──
        add     x11, sp, #4096        // qa_base = sp + 4480
        add     x11, x11, #384
        mul     w10, w0, w14          // ti * K_pad
        add     x11, x11, x10         // qa_tile = qa_base + ti*K_pad
        // ── Compute qb_tile = qb_base + tj (column offset in qb row) ──
        ldr     w16, [sp, #44]        // M_pad
        mul     w10, w16, w14         // M_pad * K_pad
        add     x13, sp, #4096
        add     x13, x13, #384
        add     x13, x13, x10         // qb_base
        add     x13, x13, x1          // qb_tile = qb_base + tj
        // ── Zero ZA for this tile ──
        zero    {za}
        ptrue   p0.b
        add     x16, sp, #128         // packed scratch
        mov     w12, #0               // t = 0 (k-step counter)
        cbz     w15, Lfuse_tile_smopa_done
    Lfuse_tile_kstep:
        // ── Pack rows: gather 4 bytes from each of 32 qa rows into packed scratch ──
        mov     x8, x16               // dst = packed_rows scratch
        mov     w17, #0               // row counter
    Lfuse_tile_pr:
        madd    x10, x17, x14, x11    // x10 = qa_tile + row*K_pad
        add     x10, x10, x12, lsl #2 // + t*4
        ldr     w22, [x10]
        str     w22, [x8], #4
        add     w17, w17, #1
        cmp     w17, #32
        b.lt    Lfuse_tile_pr
        // ── Pack cols: load 4 qb rows at k-step t, zip interleave ──
        //   qb row r starts at qb_tile + r * N_pad
        //   We need rows t*4+0 .. t*4+3
        lsl     w17, w12, #2           // t*4
        mul     x10, x17, x4          // (t*4) * N_pad
        add     x8, x13, x10          // qb_tile + (t*4)*N_pad
        ld1b    {z0.b}, p0/z, [x8]
        add     x8, x8, x4            // += N_pad
        ld1b    {z1.b}, p0/z, [x8]
        add     x8, x8, x4
        ld1b    {z2.b}, p0/z, [x8]
        add     x8, x8, x4
        ld1b    {z3.b}, p0/z, [x8]
        zip1    z4.b, z0.b, z2.b
        zip1    z5.b, z1.b, z3.b
        zip1    z6.b, z4.b, z5.b
        zip2    z7.b, z4.b, z5.b
        add     x8, x16, #128         // packed_cols scratch
        st1b    {z6.b}, p0, [x8]
        add     x10, x8, #64
        st1b    {z7.b}, p0, [x10]
        // ── Load packed data into z0-z3 and run smopa ──
        ld1b    {z0.b}, p0/z, [x16]
        add     x8, x16, #64
        ld1b    {z1.b}, p0/z, [x8]
        add     x8, x16, #128
        ld1b    {z2.b}, p0/z, [x8]
        add     x8, x16, #192
        ld1b    {z3.b}, p0/z, [x8]
        smopa   za0.s, p0/m, p0/m, z0.b, z2.b
        smopa   za1.s, p0/m, p0/m, z0.b, z3.b
        smopa   za2.s, p0/m, p0/m, z1.b, z2.b
        smopa   za3.s, p0/m, p0/m, z1.b, z3.b
        add     w12, w12, #1
        cmp     w12, w15
        b.lt    Lfuse_tile_kstep
    Lfuse_tile_smopa_done:
        // ── Dequant + optional ReLU → tile_out scratch (32×32 floats) ──
        ldr     s28, [sp, #56]         // combined dequant scale
        mov     z16.s, s28             // broadcast
        ldr     w18, [sp, #52]         // flags
        and     w18, w18, #1           // isolate relu
        ptrue   p0.s
        cntw    x9
        fmov    z17.s, #0.0            // for ReLU
        add     x7, sp, #384          // tile_out scratch base
        mov     x10, #128              // tile_out row stride = 32*4 = 128 bytes
        // Store upper half: za0 (left 16 cols) + za1 (right 16 cols) → rows 0..15
        mov     w12, #0
    Lfuse_tile_store_upper:
        mova    {z4.s-z7.s}, za0h.s[w12, 0:3]
        mova    {z8.s-z11.s}, za1h.s[w12, 0:3]
        scvtf   z4.s, p0/m, z4.s
        fmul    z4.s, z4.s, z16.s
        scvtf   z5.s, p0/m, z5.s
        fmul    z5.s, z5.s, z16.s
        scvtf   z6.s, p0/m, z6.s
        fmul    z6.s, z6.s, z16.s
        scvtf   z7.s, p0/m, z7.s
        fmul    z7.s, z7.s, z16.s
        scvtf   z8.s, p0/m, z8.s
        fmul    z8.s, z8.s, z16.s
        scvtf   z9.s, p0/m, z9.s
        fmul    z9.s, z9.s, z16.s
        scvtf   z10.s, p0/m, z10.s
        fmul    z10.s, z10.s, z16.s
        scvtf   z11.s, p0/m, z11.s
        fmul    z11.s, z11.s, z16.s
        cbz     w18, Lfuse_tile_su_norelu
        fmax    z4.s, p0/m, z4.s, z17.s
        fmax    z5.s, p0/m, z5.s, z17.s
        fmax    z6.s, p0/m, z6.s, z17.s
        fmax    z7.s, p0/m, z7.s, z17.s
        fmax    z8.s, p0/m, z8.s, z17.s
        fmax    z9.s, p0/m, z9.s, z17.s
        fmax    z10.s, p0/m, z10.s, z17.s
        fmax    z11.s, p0/m, z11.s, z17.s
    Lfuse_tile_su_norelu:
        st1w    {z4.s}, p0, [x7]
        st1w    {z8.s}, p0, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z5.s}, p0, [x7]
        st1w    {z9.s}, p0, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z6.s}, p0, [x7]
        st1w    {z10.s}, p0, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z7.s}, p0, [x7]
        st1w    {z11.s}, p0, [x7, x9, lsl #2]
        add     x7, x7, x10
        add     w12, w12, #4
        cmp     w12, w9
        b.lt    Lfuse_tile_store_upper
        // Store lower half: za2 (left) + za3 (right) → rows 16..31
        mov     w12, #0
    Lfuse_tile_store_lower:
        mova    {z4.s-z7.s}, za2h.s[w12, 0:3]
        mova    {z8.s-z11.s}, za3h.s[w12, 0:3]
        scvtf   z4.s, p0/m, z4.s
        fmul    z4.s, z4.s, z16.s
        scvtf   z5.s, p0/m, z5.s
        fmul    z5.s, z5.s, z16.s
        scvtf   z6.s, p0/m, z6.s
        fmul    z6.s, z6.s, z16.s
        scvtf   z7.s, p0/m, z7.s
        fmul    z7.s, z7.s, z16.s
        scvtf   z8.s, p0/m, z8.s
        fmul    z8.s, z8.s, z16.s
        scvtf   z9.s, p0/m, z9.s
        fmul    z9.s, z9.s, z16.s
        scvtf   z10.s, p0/m, z10.s
        fmul    z10.s, z10.s, z16.s
        scvtf   z11.s, p0/m, z11.s
        fmul    z11.s, z11.s, z16.s
        cbz     w18, Lfuse_tile_sl_norelu
        fmax    z4.s, p0/m, z4.s, z17.s
        fmax    z5.s, p0/m, z5.s, z17.s
        fmax    z6.s, p0/m, z6.s, z17.s
        fmax    z7.s, p0/m, z7.s, z17.s
        fmax    z8.s, p0/m, z8.s, z17.s
        fmax    z9.s, p0/m, z9.s, z17.s
        fmax    z10.s, p0/m, z10.s, z17.s
        fmax    z11.s, p0/m, z11.s, z17.s
    Lfuse_tile_sl_norelu:
        st1w    {z4.s}, p0, [x7]
        st1w    {z8.s}, p0, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z5.s}, p0, [x7]
        st1w    {z9.s}, p0, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z6.s}, p0, [x7]
        st1w    {z10.s}, p0, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z7.s}, p0, [x7]
        st1w    {z11.s}, p0, [x7, x9, lsl #2]
        add     x7, x7, x10
        add     w12, w12, #4
        cmp     w12, w9
        b.lt    Lfuse_tile_store_lower
        // ── Copy valid portion of tile_out → C ──
        // C_tile = C_base + ti*N + tj (in floats)
        // Copy min(32, M-ti) rows, each min(32, N-tj) floats
        ldr     x7, [sp, #16]         // C_base
        ldp     w0, w1, [sp, #80]     // ti, tj
        ldr     w2, [sp, #24]         // M
        ldr     w3, [sp, #28]         // N
        // rows_valid = min(32, M - ti)
        sub     w5, w2, w0             // M - ti
        mov     w8, #32
        cmp     w5, w8
        csel    w5, w5, w8, lt         // w5 = min(M-ti, 32)
        // cols_valid = min(32, N - tj)
        sub     w6, w3, w1             // N - tj
        cmp     w6, w8
        csel    w6, w6, w8, lt         // w6 = min(N-tj, 32)
        // C_tile = C + (ti * N + tj) * 4
        mul     w8, w0, w3             // ti * N
        add     w8, w8, w1             // + tj
        add     x7, x7, x8, lsl #2    // C_tile ptr
        lsl     x10, x3, #2           // N * 4 = C row stride in bytes
        add     x11, sp, #384         // tile_out base
        // Generate col predicate: whilelt for cols_valid
        whilelt p2.s, xzr, x6         // p2 masks valid columns
        mov     w12, #0               // row counter
    Lfuse_tile_copy:
        cmp     w12, w5
        b.ge    Lfuse_tile_copy_done
        // Load from tile_out row: left 16 + right 16
        ld1w    {z0.s}, p2/z, [x11]
        // Check if cols_valid > 16: need right half too
        cmp     w6, #16
        b.le    Lfuse_tile_copy_left_only
        // For the right half, create offset predicate
        sub     w17, w6, #16
        whilelt p3.s, xzr, x17
        add     x8, x11, x9, lsl #2   // tile_out + 16*4
        ld1w    {z1.s}, p3/z, [x8]
        st1w    {z0.s}, p2, [x7]
        st1w    {z1.s}, p3, [x7, x9, lsl #2]
        b       Lfuse_tile_copy_next
    Lfuse_tile_copy_left_only:
        st1w    {z0.s}, p2, [x7]
    Lfuse_tile_copy_next:
        add     x7, x7, x10           // next C row
        add     x11, x11, #128        // next tile_out row
        add     w12, w12, #1
        b       Lfuse_tile_copy
    Lfuse_tile_copy_done:
        // ── Advance to next tile column ──
        ldp     w0, w1, [sp, #80]     // reload ti, tj
        ldr     w4, [sp, #40]         // N_pad
        ldr     w14, [sp, #36]        // K_pad
        ldr     w15, [sp, #48]        // k_steps
        ldr     w3, [sp, #44]         // M_pad
        add     w1, w1, #32           // tj += 32
        b       Lfuse_tile_col
    Lfuse_tile_row_next:
        add     w0, w0, #32           // ti += 32
        b       Lfuse_tile_row
    Lfuse_tile_done:
        // ── Deallocate stack and dispatch ──
        ldr     x10, [sp, #72]         // frame size
        add     sp, sp, x10
        b       Ldispatch
    // ================================================================
    // DENSE_SCALE_I8 (0x0F)
    // Fused: zero → smopa accumulate → scvtf → fmul(scale) → store fp32
    // Bytecode: [0x0F][k_steps:u32][scale:f32]
    // Operands: rows_ptr, cols_ptr, output_ptr
    // ================================================================
    Lop_dense_scale_i8:
        ldr     w22, [x19]
        ldr     s16, [x19, #4]
        add     x19, x19, #8
        ldr     x8, [x19], #8
        ldr     x11, [x19], #8
        ldr     x13, [x19], #8
        zero    {za}
        ptrue   p0.b
        cbz     w22, Lscale_dequant
    Lscale_acc:
        ld1b    {z0.b}, p0/z, [x8]
        ld1b    {z1.b}, p0/z, [x8, #1, mul vl]
        ld1b    {z2.b}, p0/z, [x11]
        ld1b    {z3.b}, p0/z, [x11, #1, mul vl]
        smopa   za0.s, p0/m, p0/m, z0.b, z2.b
        smopa   za1.s, p0/m, p0/m, z0.b, z3.b
        smopa   za2.s, p0/m, p0/m, z1.b, z2.b
        smopa   za3.s, p0/m, p0/m, z1.b, z3.b
        addvl   x8, x8, #2
        addvl   x11, x11, #2
        sub     w22, w22, #1
        cbnz    w22, Lscale_acc
    Lscale_dequant:
        ptrue   p0.s
        mov     z16.s, s16
        cntw    x9
        lsl     x10, x9, #3
        mov     w12, #0
    Lscale_upper:
        mova    {z4.s-z7.s}, za0h.s[w12, 0:3]
        mova    {z8.s-z11.s}, za1h.s[w12, 0:3]
        scvtf   z4.s, p0/m, z4.s
        fmul    z4.s, z4.s, z16.s
        scvtf   z8.s, p0/m, z8.s
        fmul    z8.s, z8.s, z16.s
        st1w    {z4.s}, p0, [x13]
        st1w    {z8.s}, p0, [x13, x9, lsl #2]
        add     x13, x13, x10
        scvtf   z5.s, p0/m, z5.s
        fmul    z5.s, z5.s, z16.s
        scvtf   z9.s, p0/m, z9.s
        fmul    z9.s, z9.s, z16.s
        st1w    {z5.s}, p0, [x13]
        st1w    {z9.s}, p0, [x13, x9, lsl #2]
        add     x13, x13, x10
        scvtf   z6.s, p0/m, z6.s
        fmul    z6.s, z6.s, z16.s
        scvtf   z10.s, p0/m, z10.s
        fmul    z10.s, z10.s, z16.s
        st1w    {z6.s}, p0, [x13]
        st1w    {z10.s}, p0, [x13, x9, lsl #2]
        add     x13, x13, x10
        scvtf   z7.s, p0/m, z7.s
        fmul    z7.s, z7.s, z16.s
        scvtf   z11.s, p0/m, z11.s
        fmul    z11.s, z11.s, z16.s
        st1w    {z7.s}, p0, [x13]
        st1w    {z11.s}, p0, [x13, x9, lsl #2]
        add     x13, x13, x10
        add     w12, w12, #4
        cmp     w12, w9
        b.lt    Lscale_upper
        mov     w12, #0
    Lscale_lower:
        mova    {z4.s-z7.s}, za2h.s[w12, 0:3]
        mova    {z8.s-z11.s}, za3h.s[w12, 0:3]
        scvtf   z4.s, p0/m, z4.s
        fmul    z4.s, z4.s, z16.s
        scvtf   z8.s, p0/m, z8.s
        fmul    z8.s, z8.s, z16.s
        st1w    {z4.s}, p0, [x13]
        st1w    {z8.s}, p0, [x13, x9, lsl #2]
        add     x13, x13, x10
        scvtf   z5.s, p0/m, z5.s
        fmul    z5.s, z5.s, z16.s
        scvtf   z9.s, p0/m, z9.s
        fmul    z9.s, z9.s, z16.s
        st1w    {z5.s}, p0, [x13]
        st1w    {z9.s}, p0, [x13, x9, lsl #2]
        add     x13, x13, x10
        scvtf   z6.s, p0/m, z6.s
        fmul    z6.s, z6.s, z16.s
        scvtf   z10.s, p0/m, z10.s
        fmul    z10.s, z10.s, z16.s
        st1w    {z6.s}, p0, [x13]
        st1w    {z10.s}, p0, [x13, x9, lsl #2]
        add     x13, x13, x10
        scvtf   z7.s, p0/m, z7.s
        fmul    z7.s, z7.s, z16.s
        scvtf   z11.s, p0/m, z11.s
        fmul    z11.s, z11.s, z16.s
        st1w    {z7.s}, p0, [x13]
        st1w    {z11.s}, p0, [x13, x9, lsl #2]
        add     x13, x13, x10
        add     w12, w12, #4
        cmp     w12, w9
        b.lt    Lscale_lower
        b       Ldispatch
    // ================================================================
    // ELEMENTWISE_ADD_FP32 (0x10)
    // Pure SVE: load a, load b, fadd, store
    // Bytecode: [0x10][count:u32]
    // Operands: a_ptr, b_ptr, output_ptr
    // ================================================================
    Lop_elementwise_add_fp32:
        ldr     w22, [x19]
        add     x19, x19, #4
        ldr     x8, [x19], #8
        ldr     x11, [x19], #8
        ldr     x13, [x19], #8
        ptrue   p0.s
        cntw    x9
        cbz     w22, Ldispatch
    Ladd_loop:
        ld1w    {z0.s}, p0/z, [x8]
        ld1w    {z1.s}, p0/z, [x11]
        fadd    z0.s, z0.s, z1.s
        st1w    {z0.s}, p0, [x13]
        add     x8, x8, x9, lsl #2
        add     x11, x11, x9, lsl #2
        add     x13, x13, x9, lsl #2
        sub     w22, w22, w9
        cbnz    w22, Ladd_loop
        b       Ldispatch
    // ================================================================
    // ELEMENTWISE_SCALED_ADD_FP32 (0x11)
    // out = a + scale * b  (used for SGD: W = W + (-lr) * grad)
    // Bytecode: [0x11][count:u32][scale:f32]
    // Operands: a_ptr, b_ptr, output_ptr
    // ================================================================
    Lop_elementwise_scaled_add_fp32:
        ldr     w22, [x19]
        ldr     s16, [x19, #4]
        add     x19, x19, #8
        ldr     x8, [x19], #8
        ldr     x11, [x19], #8
        ldr     x13, [x19], #8
        ptrue   p0.s
        mov     z16.s, s16
        cntw    x9
        cbz     w22, Ldispatch
    Lscadd_loop:
        ld1w    {z0.s}, p0/z, [x8]
        ld1w    {z1.s}, p0/z, [x11]
        fmla    z0.s, p0/m, z1.s, z16.s
        st1w    {z0.s}, p0, [x13]
        add     x8, x8, x9, lsl #2
        add     x11, x11, x9, lsl #2
        add     x13, x13, x9, lsl #2
        sub     w22, w22, w9
        cbnz    w22, Lscadd_loop
        b       Ldispatch
    // ================================================================
    // ELEMENTWISE_MUL_FP32 (0x12)
    // out = a * b  (used for ReLU backward mask)
    // Bytecode: [0x12][count:u32]
    // Operands: a_ptr, b_ptr, output_ptr
    // ================================================================
    Lop_elementwise_mul_fp32:
        ldr     w22, [x19]
        add     x19, x19, #4
        ldr     x8, [x19], #8
        ldr     x11, [x19], #8
        ldr     x13, [x19], #8
        ptrue   p0.s
        cntw    x9
        cbz     w22, Ldispatch
    Lmul_loop:
        ld1w    {z0.s}, p0/z, [x8]
        ld1w    {z1.s}, p0/z, [x11]
        fmul    z0.s, z0.s, z1.s
        st1w    {z0.s}, p0, [x13]
        add     x8, x8, x9, lsl #2
        add     x11, x11, x9, lsl #2
        add     x13, x13, x9, lsl #2
        sub     w22, w22, w9
        cbnz    w22, Lmul_loop
        b       Ldispatch
    // ================================================================
    // RELU_BACKWARD_FP32 (0x13)
    // out[i] = (hidden[i] > 0) ? grad[i] : 0
    // Bytecode: [0x13][count:u32]
    // Operands: hidden_ptr, grad_ptr, output_ptr
    // ================================================================
    Lop_relu_backward_fp32:
        ldr     w22, [x19]
        add     x19, x19, #4
        ldr     x8, [x19], #8          // hidden (post-relu forward values)
        ldr     x11, [x19], #8         // grad (incoming gradient)
        ldr     x13, [x19], #8         // output
        ptrue   p0.s
        fmov    z17.s, #0.0
        cntw    x9
        cbz     w22, Ldispatch
    Lrelubw_loop:
        ld1w    {z0.s}, p0/z, [x8]     // hidden
        ld1w    {z1.s}, p0/z, [x11]    // grad
        fcmgt   p1.s, p0/z, z0.s, z17.s // p1 = (hidden > 0)
        mov     z2.s, #0
        sel     z2.s, p1, z1.s, z2.s   // z2 = p1 ? grad : 0
        st1w    {z2.s}, p0, [x13]
        add     x8, x8, x9, lsl #2
        add     x11, x11, x9, lsl #2
        add     x13, x13, x9, lsl #2
        sub     w22, w22, w9
        cbnz    w22, Lrelubw_loop
        b       Ldispatch
    // ================================================================
    // QUANTIZE_FP32_I8 (0x14)
    // Quantize fp32 array to int8: find absmax, scale to [-127,127], output i8
    // Writes inverse_scale (1/scale) to a float pointer for later dequant
    // Bytecode: [0x14][count:u32]
    // Operands: src_fp32_ptr, dst_i8_ptr, inv_scale_ptr
    // ================================================================
    Lop_quantize_fp32_i8:
        ldr     w22, [x19]             // count (fp32 elements)
        add     x19, x19, #4
        ldr     x8, [x19], #8          // src fp32
        ldr     x11, [x19], #8         // dst i8
        ldr     x13, [x19], #8         // inv_scale output (float*)
        ptrue   p0.s
        cntw    x9                     // SVLs
        // Pass 1: find absmax across all elements
        fmov    z16.s, #0.0            // running max = 0
        mov     x14, x8                // save src ptr
        mov     w15, w22               // save count
        cbz     w22, Lquant_store_scale
    Lquant_absmax:
        ld1w    {z0.s}, p0/z, [x8]
        fabs    z0.s, p0/m, z0.s
        fmax    z16.s, p0/m, z16.s, z0.s
        add     x8, x8, x9, lsl #2
        sub     w22, w22, w9
        cbnz    w22, Lquant_absmax
        // Horizontal max reduce z16 → s16
        fmaxv   s16, p0, z16.s
        // scale = 127.0 / absmax, handle near-zero
        fmov    s17, #1.0
        movz    w3, #0x42FE, lsl #16    // 127.0 as IEEE754 = 0x42FE0000
        fmov    s18, w3
        fcmp    s16, #0.0
        fccmp   s16, s17, #0, ne       // if absmax < 1e-8 treat as 1e-8
        fcsel   s16, s17, s16, lt      // clamp absmax to at least ~1.0 for safety
        fdiv    s17, s18, s16          // s17 = 127.0 / absmax = scale
        fdiv    s18, s16, s18          // s18 = absmax / 127.0 = inv_scale
        // Write inv_scale
        str     s18, [x13]
        // Pass 2: quantize — dst[i] = clamp(round(src[i] * scale), -127, 127)
        mov     z16.s, s17             // broadcast scale
        mov     x8, x14                // restore src ptr
        mov     w22, w15               // restore count
        cntw    x9
    Lquant_scale:
        ld1w    {z0.s}, p0/z, [x8]
        fmul    z0.s, z0.s, z16.s      // src * scale
        frinti  z0.s, p0/m, z0.s       // round to nearest int
        fcvtzs  z0.s, p0/m, z0.s       // convert to int32
        // Clamp to [-127, 127]
        mov     z17.s, #127
        mov     z18.s, #-127
        smin    z0.s, p0/m, z0.s, z17.s
        smax    z0.s, p0/m, z0.s, z18.s
        // Store low byte of each int32 element: st1b with .s element size
        // This truncates each 32-bit lane to 8 bits and stores contiguously
        st1b    {z0.s}, p0, [x11]
        add     x8, x8, x9, lsl #2
        add     x11, x11, x9           // advance by SVLs bytes (not *4)
        sub     w22, w22, w9
        cbnz    w22, Lquant_scale
    Lquant_store_scale:
        b       Ldispatch
    // ================================================================
    // PACK_ROWS_I8 (0x15)
    // Pack int8 row-major matrix into SME dot4 row panels.
    // Each k-step: z0 = rows 0..15 × 4 bytes, z1 = rows 16..31 × 4 bytes
    // Bytecode: [0x15][M:u32][K:u32]
    // Operands: src_i8_ptr, dst_packed_ptr
    // ================================================================
    Lop_pack_rows_i8:
        ldr     w22, [x19]             // M (must be GROUP_DIM=32)
        ldr     w3, [x19, #4]          // K (stride between rows in src)
        add     x19, x19, #8
        ldr     x8, [x19], #8          // src
        ldr     x11, [x19], #8         // dst
        lsr     w4, w3, #2             // k_steps = K / 4
        // Apple Silicon: no SVE gather in any mode. Use 4-byte loads per row.
        // smstop not needed — ldr w/str w work in streaming mode.
        mov     w14, #0
        cbz     w4, Ldispatch
    Lpack_rows_t:
        mov     x15, x11
        mov     w12, #0
    Lpr_upper:
        madd    x16, x12, x3, x8
        add     x16, x16, x14, lsl #2
        ldr     w17, [x16]
        str     w17, [x15], #4
        add     w12, w12, #1
        cmp     w12, #16
        b.lt    Lpr_upper
        mov     w12, #16
    Lpr_lower:
        madd    x16, x12, x3, x8
        add     x16, x16, x14, lsl #2
        ldr     w17, [x16]
        str     w17, [x15], #4
        add     w12, w12, #1
        cmp     w12, #32
        b.lt    Lpr_lower
        add     x11, x11, #128
        add     w14, w14, #1
        cmp     w14, w4
        b.lt    Lpack_rows_t
        b       Ldispatch
    // ================================================================
    // PACK_COLS_I8 (0x16)
    // Pack int8 row-major matrix into SME dot4 column panels.
    // Each k-step: z2 = cols 0..15 × 4 bytes, z3 = cols 16..31 × 4 bytes
    // Bytecode: [0x16][N:u32][K:u32]
    // Operands: src_i8_ptr, dst_packed_ptr
    // ================================================================
    Lop_pack_cols_i8:
        ldr     w22, [x19]             // N (stride between rows in src)
        ldr     w3, [x19, #4]          // K
        add     x19, x19, #8
        ldr     x8, [x19], #8          // src (K×N row-major)
        ldr     x11, [x19], #8         // dst packed
        lsr     w4, w3, #2             // k_steps = K/4
        ptrue   p0.b
        mov     w14, #0                // t = 0
        cbz     w4, Ldispatch
    Lpack_cols_t:
        // For k-step t, load 4 consecutive rows of N bytes each:
        //   row0 = B[(t*4+0)*N ..], row1 = B[(t*4+1)*N ..], etc.
        // Then interleave columns into dot4 format:
        //   dst[c*4+d] = row_d[c]
        lsl     w5, w14, #2            // t*4
        madd    x15, x5, x22, x8      // x15 = src + (t*4)*N = row0 base
        // Load 4 rows (each N bytes, but we only need 32 cols = first 32 bytes)
        // With SVLb=64, ld1b loads 64 bytes — first 32 are our columns, rest is fine to load
        ld1b    {z0.b}, p0/z, [x15]           // row t*4+0
        add     x15, x15, x22
        ld1b    {z1.b}, p0/z, [x15]           // row t*4+1
        add     x15, x15, x22
        ld1b    {z2.b}, p0/z, [x15]           // row t*4+2
        add     x15, x15, x22
        ld1b    {z3.b}, p0/z, [x15]           // row t*4+3
        // 4-way byte interleave: [r0c0,r1c0,r2c0,r3c0, r0c1,r1c1,r2c1,r3c1, ...]
        // Step 1: zip pairs at byte granularity
        zip1    z4.b, z0.b, z2.b      // z4 = [r0_0,r2_0, r0_1,r2_1, r0_2,r2_2, ...]
        zip1    z5.b, z1.b, z3.b      // z5 = [r1_0,r3_0, r1_1,r3_1, r1_2,r3_2, ...]
        // Step 2: zip the pairs to get 4-way interleave
        zip1    z6.b, z4.b, z5.b      // z6 = [r0_0,r1_0,r2_0,r3_0, r0_1,r1_1,r2_1,r3_1, ...]
        zip2    z7.b, z4.b, z5.b      // z7 = upper half columns
        // z6 has cols 0..15 in dot4 format (first 64 bytes)
        // z7 has cols 16..31 in dot4 format (next 64 bytes)
        st1b    {z6.b}, p0, [x11]
        add     x16, x11, #64
        st1b    {z7.b}, p0, [x16]
        add     x11, x11, #128
        add     w14, w14, #1
        cmp     w14, w4
        b.lt    Lpack_cols_t
        b       Ldispatch
    // ================================================================
    // SCATTER_TILE_FP32 (0x17)
    // Copy GROUP_DIM×GROUP_DIM tile into a strided output matrix.
    // Bytecode: [0x17][dst_row_offset:u32][dst_col_offset:u32][dst_stride_cols:u32]
    // Operands: tile_src_ptr, dst_matrix_ptr
    // tile_src is GROUP_DIM×GROUP_DIM contiguous, dst is strided.
    // ================================================================
    Lop_scatter_tile_fp32:
        ldr     w22, [x19]             // dst_row_offset
        ldr     w3, [x19, #4]          // dst_col_offset
        ldr     w4, [x19, #8]          // dst_stride_cols (N of the full matrix)
        add     x19, x19, #12
        ldr     x8, [x19], #8          // tile src (GROUP_DIM × GROUP_DIM fp32)
        ldr     x11, [x19], #8         // dst matrix base
        ptrue   p0.s
        cntw    x9                     // SVLs = 16
        // dst_start = dst + (row_offset * stride + col_offset) * 4
        madd    w5, w22, w4, w3        // row_off * stride + col_off
        lsl     x5, x5, #2            // offset * 4 bytes
        add     x11, x11, x5           // dst + offset
        lsl     x10, x4, #2           // dst row stride in bytes = stride_cols * 4
        mov     w12, #0                // row counter
    Lscatter_row:
        // Load 32 fp32 from tile (2 z-registers worth)
        ld1w    {z0.s}, p0/z, [x8]
        ld1w    {z1.s}, p0/z, [x8, x9, lsl #2]
        // Store to strided dst (only up to GROUP_DIM cols)
        st1w    {z0.s}, p0, [x11]
        st1w    {z1.s}, p0, [x11, x9, lsl #2]
        add     x8, x8, #128          // next tile row = GROUP_DIM * 4 = 128 bytes
        add     x11, x11, x10         // next dst row = stride * 4 bytes
        add     w12, w12, #1
        cmp     w12, #32               // GROUP_DIM rows
        b.lt    Lscatter_row
        b       Ldispatch
    // ================================================================
    // TRANSPOSE_FP32 (0x18)
    // Transpose M×N fp32 matrix → N×M fp32 matrix.
    // For each src row i, scatter-store as column i of dst.
    // dst[j*M + i] = src[i*N + j]
    // Bytecode: [0x18][M:u32][N:u32]
    // Operands: src_ptr, dst_ptr
    // ================================================================
    Lop_transpose_fp32:
        ldr     w22, [x19]             // M (rows of src)
        ldr     w3, [x19, #4]          // N (cols of src)
        add     x19, x19, #8
        ldr     x8, [x19], #8          // src (M×N fp32 row-major)
        ldr     x11, [x19], #8         // dst (N×M fp32 row-major)
        // dst[j*M + i] = src[i*N + j]
        // Apple Silicon: no scatter stores. Use scalar str per element.
        lsl     x5, x22, #2           // M * 4 = dst row stride
        lsl     x15, x3, #2           // N * 4 = src row stride
        mov     w12, #0
    Ltr_row:
        cmp     w12, w22
        b.ge    Ltr_done
        lsl     x16, x12, #2
        add     x16, x11, x16         // dst_col_base = dst + i*4
        mov     x17, x8               // src cursor
        mov     w14, #0
    Ltr_elem:
        cmp     w14, w3
        b.ge    Ltr_next_row
        ldr     s0, [x17], #4
        str     s0, [x16]
        add     x16, x16, x5          // next dst row
        add     w14, w14, #1
        b       Ltr_elem
    Ltr_next_row:
        add     x8, x8, x15
        add     w12, w12, #1
        b       Ltr_row
    Ltr_done:
        b       Ldispatch
    // SOFTMAX_ARGMAX_FP32 (0x19)
    // Fused softmax + cross-entropy backward + argmax. Processes batch rows.
    // Per row: (1) find max, (2) exp(x-max)+sum, (3) normalize, (4) g_out=probs-one_hot, (5) argmax.
    // Bytecode: [0x19][batch:u32][cols:u32]
    // Operands: logits, probs, labels (uint8_t[batch]), g_out (batch×cols), argmax (int32[batch])
    Lop_softmax_argmax_fp32:
        ldr     w22, [x19]             // batch
        ldr     w3, [x19, #4]          // cols (elements per row, must be multiple of SVLs)
        add     x19, x19, #8
        ldr     x8, [x19], #8          // src base (logits, batch × cols fp32)
        ldr     x11, [x19], #8         // dst base (probs, batch × cols fp32)
        ldr     x16, [x19], #8         // labels base (uint8_t[batch])
        ldr     x17, [x19], #8         // g_out base (batch × cols fp32)
        ldr     x13, [x19], #8         // argmax base (int32_t[batch])
        ptrue   p0.s
        cntw    x9
        lsl     x10, x3, #2           // row stride = cols * 4 bytes
        // ── Load exp constants (once for all rows) ──
        movz    w4, #0x3FB8, lsl #16
        movk    w4, #0xAA3B
        fmov    s28, w4
        mov     z28.s, s28              // log2(e)
        movz    w4, #0x3C1D, lsl #16
        movk    w4, #0x955A
        fmov    s29, w4
        mov     z29.s, s29              // c4
        movz    w4, #0x3D63, lsl #16
        movk    w4, #0x5847
        fmov    s30, w4
        mov     z30.s, s30              // c3
        movz    w4, #0x3E75, lsl #16
        movk    w4, #0xFDF0
        fmov    s31, w4
        mov     z31.s, s31              // c2
        movz    w4, #0x3F31, lsl #16
        movk    w4, #0x7218
        fmov    s27, w4
        mov     z27.s, s27              // c1 = ln(2)
        fmov    z26.s, #1.0
        // ── Batch loop ──
        mov     w18, #0
    Lsmb_row:
        cmp     w18, w22
        b.ge    Ldispatch
        mov     x14, x8                // save row src
        mov     x15, x11               // save row dst
        // ── Pass 1: find max ──
        movz    w4, #0xFF80, lsl #16
        fmov    s16, w4
        mov     z16.s, s16
        mov     w12, w3
    Lsmb_max:
        cbz     w12, Lsmb_max_done
        ld1w    {z0.s}, p0/z, [x8]
        fmax    z16.s, p0/m, z16.s, z0.s
        add     x8, x8, x9, lsl #2
        sub     w12, w12, w9
        cbnz    w12, Lsmb_max
    Lsmb_max_done:
        fmaxv   s16, p0, z16.s
        mov     z16.s, s16
        // ── Pass 2: exp(x - max) + sum ──
        fmov    z17.s, #0.0
        mov     x8, x14
        mov     x11, x15
        mov     w12, w3
    Lsmb_exp:
        cbz     w12, Lsmb_exp_done
        ld1w    {z0.s}, p0/z, [x8]
        fsub    z0.s, z0.s, z16.s
        fmul    z1.s, z0.s, z28.s
        frintm  z2.s, p0/m, z1.s
        fsub    z3.s, z1.s, z2.s
        fmul    z4.s, z29.s, z3.s
        fadd    z4.s, z4.s, z30.s
        fmul    z4.s, z4.s, z3.s
        fadd    z4.s, z4.s, z31.s
        fmul    z4.s, z4.s, z3.s
        fadd    z4.s, z4.s, z27.s
        fmul    z4.s, z4.s, z3.s
        fadd    z4.s, z4.s, z26.s
        fcvtzs  z5.s, p0/m, z2.s
        mov     z6.s, #-127
        smax    z5.s, p0/m, z5.s, z6.s
        add     z5.s, z5.s, #127
        lsl     z5.s, z5.s, #23
        fmul    z4.s, z4.s, z5.s
        st1w    {z4.s}, p0, [x11]
        fadd    z17.s, z17.s, z4.s
        add     x8, x8, x9, lsl #2
        add     x11, x11, x9, lsl #2
        sub     w12, w12, w9
        b       Lsmb_exp
    Lsmb_exp_done:
        // ── Pass 3: normalize ──
        faddv   s17, p0, z17.s
        fmov    s18, #1.0
        fdiv    s17, s18, s17
        mov     z17.s, s17
        mov     x11, x15
        mov     w12, w3
    Lsmb_div:
        cbz     w12, Lsmb_ce_backward
        ld1w    {z0.s}, p0/z, [x11]
        fmul    z0.s, z0.s, z17.s
        st1w    {z0.s}, p0, [x11]
        add     x11, x11, x9, lsl #2
        sub     w12, w12, w9
        b       Lsmb_div
    Lsmb_ce_backward:
        // ── Pass 4: cross-entropy backward: g_out = probs - one_hot(label) ──
        ld1w    {z0.s}, p0/z, [x15]     // reload normalized probs
        ldrb    w4, [x16], #1           // label for this row (advance labels ptr)
        index   z3.s, #0, #1            // [0, 1, 2, ..., 15]
        mov     z5.s, w4                // broadcast label index
        cmpeq   p3.s, p0/z, z3.s, z5.s // p3 = one active lane at label
        fmov    z6.s, #1.0
        fsub    z0.s, p3/m, z0.s, z6.s  // probs[label] -= 1.0
        st1w    {z0.s}, p0, [x17]       // store g_out
    Lsmb_argmax:
        // ── Pass 5: argmax ──
        whilelt p1.s, xzr, x3
        ld1w    {z0.s}, p1/z, [x15]
        movz    w4, #0xFF80, lsl #16
        fmov    s1, w4
        mov     z1.s, s1
        sel     z0.s, p1, z0.s, z1.s
        fmaxv   s2, p0, z0.s
        mov     z2.s, s2
        fcmeq   p2.s, p0/z, z0.s, z2.s
        index   z3.s, #0, #1
        mov     z4.s, #15
        sel     z4.s, p2, z3.s, z4.s
        uminv   s4, p0, z4.s
        fmov    w4, s4
        str     w4, [x13]
        // ── Advance to next row ──
        add     x8, x14, x10
        add     x11, x15, x10
        add     x17, x17, x10          // g_out += row stride
        add     x13, x13, #4           // argmax += sizeof(int32)
        add     w18, w18, #1
        b       Lsmb_row
    // ================================================================
    // LUTI4 (0x1A)
    // 4-bit table lookup using the SME LUTI4 instruction via ZT0.
    // Loads the 64-byte table into ZT0, then processes count z-vectors
    // of packed nibble indices, producing output z-vectors of looked-up values.
    // Bytecode: [0x1A][count:u32][elem_size:u8]
    //   elem_size: 0=8-bit (.b), 1=16-bit (.h), 2=32-bit (.s)
    // Operands: table_ptr (64 bytes), indices_ptr, output_ptr
    // For .b: each input byte has 2 nibbles → 2 output bytes (1v→1v, indices reinterpreted)
    // For .h: luti4 {z_dst.h}, zt0, z_idx[imm] — 16-bit output per 4-bit index
    // For .s: luti4 {z_dst.s}, zt0, z_idx[imm] — 32-bit output per 4-bit index
    // ================================================================
    Lop_luti4:
        ldr     w22, [x19]             // count (number of z-vectors to process)
        ldrb    w3, [x19, #4]          // elem_size (0=.b, 1=.h, 2=.s)
        add     x19, x19, #5
        ldr     x8, [x19], #8          // table_ptr (64 bytes, loaded into ZT0)
        ldr     x11, [x19], #8         // indices_ptr
        ldr     x13, [x19], #8         // output_ptr
        // Load table into ZT0
        ldr     zt0, [x8]
        ptrue   p0.b
        cntb    x9                     // SVLb = 64
        cbz     w22, Ldispatch
        // Dispatch on element size
        cmp     w3, #1
        b.eq    Lluti4_h
        cmp     w3, #2
        b.eq    Lluti4_s
        // Default: 8-bit lookup
    Lluti4_b:
        ld1b    {z0.b}, p0/z, [x11]
        luti4   z1.b, zt0, z0[0]
        st1b    {z1.b}, p0, [x13]
        add     x11, x11, x9
        add     x13, x13, x9
        sub     w22, w22, #1
        cbnz    w22, Lluti4_b
        b       Ldispatch
    Lluti4_h:
        ptrue   p0.h
        ld1b    {z0.b}, p0/z, [x11]
        luti4   z1.h, zt0, z0[0]
        st1h    {z1.h}, p0, [x13]
        add     x11, x11, x9
        cntw    x10
        lsl     x10, x10, #1           // SVLh * 2 bytes
        add     x13, x13, x10
        sub     w22, w22, #1
        cbnz    w22, Lluti4_h
        b       Ldispatch
    Lluti4_s:
        ptrue   p0.s
        ld1b    {z0.b}, p0/z, [x11]
        luti4   z1.s, zt0, z0[0]
        st1w    {z1.s}, p0, [x13]
        add     x11, x11, x9
        cntw    x10
        lsl     x10, x10, #2           // SVLs * 4 bytes
        add     x13, x13, x10
        sub     w22, w22, #1
        cbnz    w22, Lluti4_s
        b       Ldispatch
    // ================================================================
    // LUTI2 (0x1B)
    // 2-bit table lookup using the SME LUTI2 instruction via ZT0.
    // Bytecode: [0x1B][count:u32][elem_size:u8]
    //   elem_size: 0=8-bit (.b), 1=16-bit (.h), 2=32-bit (.s)
    // Operands: table_ptr (64 bytes), indices_ptr, output_ptr
    // ================================================================
    Lop_luti2:
        ldr     w22, [x19]
        ldrb    w3, [x19, #4]
        add     x19, x19, #5
        ldr     x8, [x19], #8          // table_ptr
        ldr     x11, [x19], #8         // indices_ptr
        ldr     x13, [x19], #8         // output_ptr
        ldr     zt0, [x8]
        ptrue   p0.b
        cntb    x9
        cbz     w22, Ldispatch
        cmp     w3, #1
        b.eq    Lluti2_h
        cmp     w3, #2
        b.eq    Lluti2_s
    Lluti2_b:
        ld1b    {z0.b}, p0/z, [x11]
        luti2   z1.b, zt0, z0[0]
        st1b    {z1.b}, p0, [x13]
        add     x11, x11, x9
        add     x13, x13, x9
        sub     w22, w22, #1
        cbnz    w22, Lluti2_b
        b       Ldispatch
    Lluti2_h:
        ptrue   p0.h
        ld1b    {z0.b}, p0/z, [x11]
        luti2   z1.h, zt0, z0[0]
        st1h    {z1.h}, p0, [x13]
        add     x11, x11, x9
        cntw    x10
        lsl     x10, x10, #1
        add     x13, x13, x10
        sub     w22, w22, #1
        cbnz    w22, Lluti2_h
        b       Ldispatch
    Lluti2_s:
        ptrue   p0.s
        ld1b    {z0.b}, p0/z, [x11]
        luti2   z1.s, zt0, z0[0]
        st1w    {z1.s}, p0, [x13]
        add     x11, x11, x9
        cntw    x10
        lsl     x10, x10, #2
        add     x13, x13, x10
        sub     w22, w22, #1
        cbnz    w22, Lluti2_s
        b       Ldispatch
    // ================================================================
    // DENSE_FP32 (0x1C)
    // Full fp32 matmul via FMOPA: C = scale * (A @ B) [+ relu]
    // Bytecode: [0x1C][M:u32][N:u32][K:u32][scale:f32][flags:u8]
    //   flags bit 0: apply ReLU after matmul+scale
    // Operands: A (M×K row-major fp32), B (K×N row-major fp32), C (M×N row-major fp32)
    // 16×32 output tiles: za0+za1 accumulate, za2 inline transpose.
    //   Caller must pad output buffer rows to ((M+31)&~31).
    //   Caller must pad A columns (K dim) to ((K+15)&~15) with zeros.
    // ================================================================
    Lop_dense_fp32:
        // ── Parse immediates + operands ──
        ldr     w0, [x19]              // M
        ldr     w1, [x19, #4]          // N
        ldr     w2, [x19, #8]          // K
        ldr     s20, [x19, #12]        // scale (f32)
        ldrb    w18, [x19, #16]        // flags (bit 0 = relu)
        add     x19, x19, #17
        ldr     x5, [x19], #8          // A
        ldr     x6, [x19], #8          // B
        ldr     x7, [x19], #8          // C
        // ── Derived values ──
        add     w3, w0, #15
        and     w3, w3, #0xFFFFFFF0    // M_pad = (M+15) & ~15 (tile rows = 16)
        add     w4, w1, #31
        and     w4, w4, #0xFFFFFFE0    // N_pad = (N+31) & ~31 (tile cols = 32)
        lsr     w15, w2, #4            // k_blocks = K / 16
        ptrue   p0.s
        cntw    x9                     // SVLs = 16
        lsl     x17, x2, #2           // K * 4 = A row stride in bytes
        // ── Branchless ReLU threshold: z21 = relu ? 0.0 : -FLT_MAX ──
        mov     z20.s, s20             // broadcast scale
        and     w18, w18, #1
        movz    w10, #0xFF7F, lsl #16
        movk    w10, #0xFFFF           // -FLT_MAX
        cmp     w18, #0
        csel    w10, w10, wzr, eq      // relu=0 → -FLT_MAX; relu=1 → 0.0
        fmov    s21, w10
        mov     z21.s, s21
        // ── Fixed 128-byte stack frame (context only, no scratch) ──
        sub     sp, sp, #128
        stp     x5, x6, [sp, #0]      // [0] A, [8] B
        str     x7, [sp, #16]         // [16] C
        stp     w0, w1, [sp, #24]     // [24] M, [28] N
        str     w2, [sp, #32]         // [32] K
        stp     w4, w3, [sp, #40]     // [40] N_pad, [44] M_pad  (store before x3 is reused)
        str     w15, [sp, #48]        // [48] k_blocks
        lsl     x3, x1, #2            // N * 4 = B/C row stride in bytes  (now safe to clobber w3)
        stp     x3, x17, [sp, #56]    // [56] B_stride, [64] A_row_stride
        // ── Tile row loop: ti = 0, 16, ... ──
        mov     w0, #0                 // ti = 0
    Lfp32_tile_row:
        str     w0, [sp, #80]          // save ti
        // A_tile_base = A + ti * K * 4
        ldr     x5, [sp, #0]
        ldr     w2, [sp, #32]
        mul     w10, w0, w2
        add     x5, x5, x10, lsl #2   // x5 = A + ti*K*4
        // ── Tile column loop: tj = 0, 32, ... ──
        mov     w1, #0                 // tj = 0
    Lfp32_tile_col:
        str     w1, [sp, #84]          // save tj
        // Zero accumulators
        zero    {za}
        // B_tile = B + tj * 4
        ldr     x6, [sp, #8]
        add     x6, x6, x1, lsl #2
        ldr     x3, [sp, #56]          // B_stride = N*4
        ldr     x17, [sp, #64]         // A_row_stride = K*4
        ldr     w15, [sp, #48]         // k_blocks
        ptrue   p0.s
        cntw    x9
        // x13 = ck byte offset into A rows (advances by 64 per k-block)
        mov     x13, xzr
        // ── K-block loop: process 16 columns of A per iteration ──
        cbz     w15, Lfp32_kblock_done
    Lfp32_kblock:
        // ── Load 16 A rows into za2 for transpose ──
        zero    {za2.s}
        add     x8, x5, x13           // A_tile_base + ck*4
        mov     w12, #0
        ld1w    {z0.s}, p0/z, [x8]
        add     x8, x8, x17
        ld1w    {z1.s}, p0/z, [x8]
        add     x8, x8, x17
        ld1w    {z2.s}, p0/z, [x8]
        add     x8, x8, x17
        ld1w    {z3.s}, p0/z, [x8]
        add     x8, x8, x17
        mova    za2h.s[w12, 0:3], {z0.s-z3.s}
        mov     w12, #4
        ld1w    {z0.s}, p0/z, [x8]
        add     x8, x8, x17
        ld1w    {z1.s}, p0/z, [x8]
        add     x8, x8, x17
        ld1w    {z2.s}, p0/z, [x8]
        add     x8, x8, x17
        ld1w    {z3.s}, p0/z, [x8]
        add     x8, x8, x17
        mova    za2h.s[w12, 0:3], {z0.s-z3.s}
        mov     w12, #8
        ld1w    {z0.s}, p0/z, [x8]
        add     x8, x8, x17
        ld1w    {z1.s}, p0/z, [x8]
        add     x8, x8, x17
        ld1w    {z2.s}, p0/z, [x8]
        add     x8, x8, x17
        ld1w    {z3.s}, p0/z, [x8]
        add     x8, x8, x17
        mova    za2h.s[w12, 0:3], {z0.s-z3.s}
        mov     w12, #12
        ld1w    {z0.s}, p0/z, [x8]
        add     x8, x8, x17
        ld1w    {z1.s}, p0/z, [x8]
        add     x8, x8, x17
        ld1w    {z2.s}, p0/z, [x8]
        add     x8, x8, x17
        ld1w    {z3.s}, p0/z, [x8]
        mova    za2h.s[w12, 0:3], {z0.s-z3.s}
        // ── Extract 16 columns from za2, FMOPA into za0+za1 ──
        // Cols 0-3
        mov     w12, #0
        mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z0.s, z5.s
        add     x6, x6, x3
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
        add     x6, x6, x3
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z2.s, z5.s
        add     x6, x6, x3
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z3.s, z5.s
        add     x6, x6, x3
        // Cols 4-7
        mov     w12, #4
        mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z0.s, z5.s
        add     x6, x6, x3
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
        add     x6, x6, x3
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z2.s, z5.s
        add     x6, x6, x3
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z3.s, z5.s
        add     x6, x6, x3
        // Cols 8-11
        mov     w12, #8
        mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z0.s, z5.s
        add     x6, x6, x3
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
        add     x6, x6, x3
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z2.s, z5.s
        add     x6, x6, x3
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z3.s, z5.s
        add     x6, x6, x3
        // Cols 12-15
        mov     w12, #12
        mova    {z0.s-z3.s}, za2v.s[w12, 0:3]
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z0.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z0.s, z5.s
        add     x6, x6, x3
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z1.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z1.s, z5.s
        add     x6, x6, x3
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z2.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z2.s, z5.s
        add     x6, x6, x3
        ld1w    {z4.s}, p0/z, [x6]
        ld1w    {z5.s}, p0/z, [x6, x9, lsl #2]
        fmopa   za0.s, p0/m, p0/m, z3.s, z4.s
        fmopa   za1.s, p0/m, p0/m, z3.s, z5.s
        add     x6, x6, x3
        // ── Advance k-block ──
        add     x13, x13, #64         // ck += 16 elements (64 bytes)
        subs    w15, w15, #1
        b.ne    Lfp32_kblock
    Lfp32_kblock_done:
        // ── Store: extract za0+za1 → scale → clamp → C ──
        ldr     x7, [sp, #16]         // C
        ldr     w0, [sp, #80]         // ti
        ldr     w1, [sp, #84]         // tj
        ldr     w14, [sp, #28]        // N
        lsl     x10, x14, #2          // C row stride = N*4
        mul     w8, w0, w14
        add     w8, w8, w1
        add     x7, x7, x8, lsl #2   // C_tile = C + (ti*N + tj)*4
        // Column predicates
        sub     w6, w14, w1            // N - tj
        mov     w8, #32
        cmp     w6, w8
        csel    w6, w6, w8, lt
        whilelt p2.s, xzr, x6         // left cols
        sub     w8, w6, #16
        cmp     w8, #0
        csel    w8, wzr, w8, lt
        whilelt p3.s, xzr, x8         // right cols (0 if N-tj <= 16)
        ptrue   p0.s
        cntw    x9
        // 4 groups of 4 rows = 16 rows
        // Scale via fmul, relu via multi-vector fmax
        mov     z14.d, z21.d           // relu threshold into low reg (z0-z15 required)
        mov     w12, #0
        // Group 0
        mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
        mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
        fmul    z0.s, z0.s, z20.s
        fmul    z1.s, z1.s, z20.s
        fmul    z2.s, z2.s, z20.s
        fmul    z3.s, z3.s, z20.s
        fmul    z4.s, z4.s, z20.s
        fmul    z5.s, z5.s, z20.s
        fmul    z6.s, z6.s, z20.s
        fmul    z7.s, z7.s, z20.s
        fmax    {z0.s-z3.s}, {z0.s-z3.s}, z14.s
        fmax    {z4.s-z7.s}, {z4.s-z7.s}, z14.s
        st1w    {z0.s}, p2, [x7]
        st1w    {z4.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z1.s}, p2, [x7]
        st1w    {z5.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z2.s}, p2, [x7]
        st1w    {z6.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z3.s}, p2, [x7]
        st1w    {z7.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        // Group 1
        mov     w12, #4
        mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
        mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
        fmul    z0.s, z0.s, z20.s
        fmul    z1.s, z1.s, z20.s
        fmul    z2.s, z2.s, z20.s
        fmul    z3.s, z3.s, z20.s
        fmul    z4.s, z4.s, z20.s
        fmul    z5.s, z5.s, z20.s
        fmul    z6.s, z6.s, z20.s
        fmul    z7.s, z7.s, z20.s
        fmax    {z0.s-z3.s}, {z0.s-z3.s}, z14.s
        fmax    {z4.s-z7.s}, {z4.s-z7.s}, z14.s
        st1w    {z0.s}, p2, [x7]
        st1w    {z4.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z1.s}, p2, [x7]
        st1w    {z5.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z2.s}, p2, [x7]
        st1w    {z6.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z3.s}, p2, [x7]
        st1w    {z7.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        // Group 2
        mov     w12, #8
        mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
        mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
        fmul    z0.s, z0.s, z20.s
        fmul    z1.s, z1.s, z20.s
        fmul    z2.s, z2.s, z20.s
        fmul    z3.s, z3.s, z20.s
        fmul    z4.s, z4.s, z20.s
        fmul    z5.s, z5.s, z20.s
        fmul    z6.s, z6.s, z20.s
        fmul    z7.s, z7.s, z20.s
        fmax    {z0.s-z3.s}, {z0.s-z3.s}, z14.s
        fmax    {z4.s-z7.s}, {z4.s-z7.s}, z14.s
        st1w    {z0.s}, p2, [x7]
        st1w    {z4.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z1.s}, p2, [x7]
        st1w    {z5.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z2.s}, p2, [x7]
        st1w    {z6.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z3.s}, p2, [x7]
        st1w    {z7.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        // Group 3
        mov     w12, #12
        mova    {z0.s-z3.s}, za0h.s[w12, 0:3]
        mova    {z4.s-z7.s}, za1h.s[w12, 0:3]
        fmul    z0.s, z0.s, z20.s
        fmul    z1.s, z1.s, z20.s
        fmul    z2.s, z2.s, z20.s
        fmul    z3.s, z3.s, z20.s
        fmul    z4.s, z4.s, z20.s
        fmul    z5.s, z5.s, z20.s
        fmul    z6.s, z6.s, z20.s
        fmul    z7.s, z7.s, z20.s
        fmax    {z0.s-z3.s}, {z0.s-z3.s}, z14.s
        fmax    {z4.s-z7.s}, {z4.s-z7.s}, z14.s
        st1w    {z0.s}, p2, [x7]
        st1w    {z4.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z1.s}, p2, [x7]
        st1w    {z5.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z2.s}, p2, [x7]
        st1w    {z6.s}, p3, [x7, x9, lsl #2]
        add     x7, x7, x10
        st1w    {z3.s}, p2, [x7]
        st1w    {z7.s}, p3, [x7, x9, lsl #2]
        // ── Advance tile column ──
        ldr     w1, [sp, #84]          // tj
        ldr     w4, [sp, #40]          // N_pad
        add     w1, w1, #32
        cmp     w1, w4
        b.lt    Lfp32_tile_col
        // ── Advance tile row ──
        ldr     w0, [sp, #80]          // ti
        ldr     w3, [sp, #44]          // M_pad
        add     w0, w0, #16
        cmp     w0, w3
        b.lt    Lfp32_tile_row
    Lfp32_tile_done:
        add     sp, sp, #128
        b       Ldispatch
    )");
}
} // namespace interpreter
} // namespace ane
#else
#error "This library is only supported on Apple Silicon (aarch64 architecture)."
#endif // defined(__aarch64__) && defined(__APPLE__)
