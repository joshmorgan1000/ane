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
#include <cstring>
#include <array>
#include <concepts>
#include <type_traits>
#include <initializer_list>
#include <string>
#include <simd/simd.h>
#include <vector>
#include <ane/tiles/concepts.hpp>
#include <ane/tiles/z_vector.hpp>
#include <ane/tiles/z_stream.hpp>
#include <ane/tiles/z_tiles.hpp>
#include <ane/tiles/lut.hpp>

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
    zero_za                      = 0x01,  ///< Zero all ZA tiles
    acc_smopa                    = 0x02,  ///< Fused ld1b+smopa loop, +u32 k_steps, 2 operands
    acc_umopa                    = 0x03,  ///< Fused ld1b+umopa loop
    acc_usmopa                   = 0x04,  ///< Fused ld1b+usmopa loop
    acc_sumopa                   = 0x05,  ///< Fused ld1b+usmopa (swapped operands) loop
    store_tiles                  = 0x06,  ///< Store 2x2 tile group as row-major int32, 1 operand
    smopa_2x2                    = 0x07,  ///< 4x smopa on pre-loaded z0-z3, no operands
    umopa_2x2                    = 0x08,  ///< 4x umopa on pre-loaded z0-z3
    usmopa_2x2                   = 0x09,  ///< 4x usmopa on pre-loaded z0-z3
    load_bias                    = 0x0A,  ///< Load int32 bias into ZA tiles, 1 operand
    scale_store                  = 0x0B,  ///< int32->float*scale->store, +f32 imm, 1 operand
    elementwise_add_fp32         = 0x0C,  ///< out = a + b, [count:u32], 3 operands
    elementwise_scaled_add_fp32  = 0x0D,  ///< out = a + scale*b, [count:u32][scale:f32], 3 operands
    elementwise_mul_fp32         = 0x0E,  ///< out = a * b, [count:u32], 3 operands
    relu_backward_fp32           = 0x0F,  ///< out = (a>0)?b:0, [count:u32], 3 operands
    scatter_tile_fp32            = 0x10,  ///< Scatter GROUP_DIM tile to strided matrix
    transpose_fp32               = 0x11,  ///< Transpose M*N fp32 matrix
    softmax_argmax_fp32          = 0x12,  ///< Batched softmax + cross-entropy backward + argmax
    luti4_op                     = 0x13,  ///< 4-bit table lookup via ZT0, [count:u32][elem_size:u8]
    luti2_op                     = 0x14,  ///< 2-bit table lookup via ZT0, [count:u32][elem_size:u8]
    dense_fp32                   = 0x15,  ///< Full fp32 matmul via FMOPA (+optional relu)
    count_matches                = 0x16,  ///< Compare int32 pred[] vs uint8 labels[], count matches
    reduce_sum_fp32              = 0x17,  ///< Horizontal sum of fp32 array
    dense_i8                     = 0x18,  ///< INT8 matmul via SMOPA: C = dequant(A_i8 @ B_i8) (+relu)
    quantize_fp32_i8             = 0x19,  ///< Quantize fp32 to signed int8 per-tensor symmetric
    dequantize_i8_fp32           = 0x1A,  ///< Dequantize signed int8 to fp32
    pack_b_i8                    = 0x1B,  ///< Pack K*N row-major i8 into SMOPA panel format
    quantize_fp32_i8_channelwise = 0x1C,  ///< Per-row symmetric quantize fp32 to i8
    transpose_i8                 = 0x1D,  ///< Transpose M*N int8 matrix to N*M
    dense_u8s8                   = 0x1E,  ///< UINT8×INT8 matmul via USMOPA: C = dequant(A_u8 @ B_i8) (+relu)
    load                         = 0x1F,  ///< Load one z-vector from ptr into z0, 1 operand
    store                        = 0x20,  ///< Store one z-vector from z0 to ptr, 1 operand
    l2_squared_fp32              = 0x21,  ///< Fused L2 squared distance fp32: sum((a-b)^2)
    l2_squared_bf16              = 0x22,  ///< Fused L2 squared distance bf16 inputs, fp32 accum
    l2_squared_f64               = 0x23,  ///< Fused L2 squared distance f64
    cosine_dist_fp32             = 0x24,  ///< Cosine distance fp32: 1 - dot(a,b)/(||a||*||b||)
    cosine_dist_bf16             = 0x25,  ///< Cosine distance bf16 inputs, fp32 accum
    cosine_dist_f64              = 0x26,  ///< Cosine distance f64
    normalize_fp32               = 0x27,  ///< In-place unit normalize fp32 vector
    dct2_forward_fp32            = 0x28,  ///< H.264 4-point integer butterfly DCT-II forward, groups of 4 fp32
    dct2_inverse_fp32            = 0x29,  ///< H.264 4-point integer butterfly DCT-II inverse, groups of 4 fp32
    threshold_bitmap_fp32        = 0x2A,  ///< Compare fp32 > threshold, produce packed bit output
    welford_stats_fp32           = 0x2B,  ///< Online Welford mean/stddev/maxabs/scale across n_vectors of dim fp32
    quantize_pack_4bit_fp32      = 0x2C,  ///< Quantize fp32 to signed 4-bit SoA packed nibbles (dual src)
    threshold_8bit               = 0x2D,  ///< Reconstruct 8-bit counters from bitplanes, threshold to bitmap
    quantize_accum_2bit          = 0x2E,  ///< 2-bit ternary {-1,0,+1} decode, scale by bf16, accumulate bf16
    accum_8bit                   = 0x2F,  ///< INT8 scale-accumulate: accum[i] += data[i] * scale[i] in bf16
    soa_sub_scale_bf16           = 0x30,  ///< SoA quantized L2 partial: accum[i] += bf16((src[i]*scale-scalar)^2)
    soa_luti2_accum              = 0x31,  ///< LUTI2 expand 2-bit indices via ZT0, accumulate into bf16
    soa_luti4_accum              = 0x32,  ///< LUTI4 expand 4-bit indices via ZT0, accumulate into bf16
    bitmap_score_pipeline        = 0x33,  ///< Ripple-carry bitmap accumulate, threshold, extract candidate IDs
    mov_zreg                     = 0x34,  ///< Move z{src} to z{dst}, [src:u8][dst:u8]
    loop_begin                   = 0x35,  ///< Set loop counter, [count:u8] (max 255 iterations)
    loop_end                     = 0x36,  ///< Decrement counter, jump back by [offset:u16] bytes
    set_param                    = 0x37,  ///< Set param table entry, [idx:u8][ptr:u64]
    load_param                   = 0x38,  ///< Load one z-vector from param[idx] into z0, [idx:u8]
    store_param                  = 0x39,  ///< Store z0 to param[idx], [idx:u8]
    advance_param                = 0x3A,  ///< Advance param[idx] pointer by VL bytes, [idx:u8]
    fadd_zreg                    = 0x3B,  ///< z{dst}.s = z{src1}.s + z{src2}.s, [dst:u8][src1:u8][src2:u8]
    fsub_zreg                    = 0x3C,  ///< z{dst}.s = z{src1}.s - z{src2}.s, [dst:u8][src1:u8][src2:u8]
    fmul_zreg                    = 0x3D,  ///< z{dst}.s = z{src1}.s * z{src2}.s, [dst:u8][src1:u8][src2:u8]
    fmla_zreg                    = 0x3E,  ///< z{dst}.s += z{src1}.s * z{src2}.s, [dst:u8][src1:u8][src2:u8]
    and_zreg                     = 0x3F,  ///< z{dst} = z{src1} AND z{src2}, [dst:u8][src1:u8][src2:u8]
    orr_zreg                     = 0x40,  ///< z{dst} = z{src1} OR z{src2}, [dst:u8][src1:u8][src2:u8]
    eor_zreg                     = 0x41,  ///< z{dst} = z{src1} XOR z{src2}, [dst:u8][src1:u8][src2:u8]
    not_zreg                     = 0x42,  ///< z{dst} = NOT z{src}, [dst:u8][src:u8]
    lsl_zreg                     = 0x43,  ///< z{dst} = z{src} << amount, [dst:u8][src:u8][amount:u8]
    lsr_zreg                     = 0x44,  ///< z{dst} = z{src} >> amount (logical), [dst:u8][src:u8][amount:u8]
    asr_zreg                     = 0x45,  ///< z{dst} = z{src} >> amount (arithmetic), [dst:u8][src:u8][amount:u8]
    load_zt0                     = 0x46,  ///< Load 64 bytes into ZT0 table register, [ptr:u64]
    luti2_zreg                   = 0x47,  ///< z{dst}.b = luti2(zt0, z{src}[0]), [dst:u8][src:u8]
    luti4_zreg                   = 0x48,  ///< z{dst}.b = luti4(zt0, z{src}[0]), [dst:u8][src:u8]
    smopa_zreg                   = 0x49,  ///< za{tile}.s += z{src1}.b * z{src2}.b (signed), [tile:u8][src1:u8][src2:u8]
    umopa_zreg                   = 0x4A,  ///< za{tile}.s += z{src1}.b * z{src2}.b (unsigned), [tile:u8][src1:u8][src2:u8]
    usmopa_zreg                  = 0x4B,  ///< za{tile}.s += z{src1}.b * z{src2}.b (u*s), [tile:u8][src1:u8][src2:u8]
    fmopa_zreg                   = 0x4C,  ///< za{tile}.s += z{src1}.s * z{src2}.s (fp32), [tile:u8][src1:u8][src2:u8]
    cblas_sgemm                  = 0x4D,  ///< C = alpha*op(A)*op(B) + beta*C, BLAS GEMM with strides
    fclamp_zreg                  = 0x4E,  ///< clamp/max/min: [flags:u8][dst:u8][src:u8][lo:f32][hi:f32]
    faddv_zreg                   = 0x4F,  ///< Horizontal sum: dst = broadcast(sum(src)), [dst:u8][src:u8]
    frsqrt_zreg                  = 0x50,  ///< Reciprocal sqrt: dst[i] = 1/sqrt(src[i]), [dst:u8][src:u8]
    rms_norm_fp32                = 0x51,  ///< RMS normalization: out = in * rsqrt(mean(in^2)+eps) * weight
    broadcast_scalar_zreg        = 0x52,  ///< Fill dst with scalar value, [dst:u8][value:f32]
    fscale_zreg                  = 0x53,  ///< dst = src * scalar, [dst:u8][src:u8][scalar:f32]
    silu_fp32                    = 0x54,  ///< SiLU: out[i] = in[i]*sigmoid(in[i]), [count:u32][input_ptr:u64][output_ptr:u64]
    rope_fp32                    = 0x55,  ///< RoPE: rotary position embedding, [dim:u32][pos:u32][theta:f32][in:u64][out:u64]
    softmax_fp32                 = 0x56,  ///< Standalone softmax: out = softmax(in), [dim:u32][in:u64][out:u64]
    q8_0_gemv                    = 0x57,  ///< Q8_0 quantized GEMV: out = dequant(W_q8) * input, [M:u32][K:u32][in:u64][W:u64][out:u64]
    q4_0_gemv                    = 0x58,  ///< Q4_0 quantized GEMV: out = dequant(W_q4) * input, [M:u32][K:u32][in:u64][W:u64][out:u64]
    fdot_zreg                    = 0x59,  ///< Dot product: z0 = broadcast(dot(a, b)), [width:u8] w1: z0·z1, w2: z0:z1·z2:z3, w4: z0:z3·z4:z7
    fmla_wide_zreg               = 0x5A,  ///< Wide FMA accumulate into z0, [width:u8] w1: z0+=z1*z2, w2: z0+=z1*z3+z2*z4, w4: z0+=z1*z5+...+z4*z8
    fadd_wide_zreg               = 0x5B,  ///< Wide ADD: [width:u8][dst:u8][s1:u8][s2:u8] dst..dst+w-1 = s1..s1+w-1 + s2..s2+w-1
    fsub_wide_zreg               = 0x5C,  ///< Wide SUB: same encoding as fadd_wide
    fmul_wide_zreg               = 0x5D,  ///< Wide MUL: same encoding as fadd_wide
    load_wide_param              = 0x5E,  ///< Wide load from param: [width:u8][param_idx:u8] loads width z-regs into z0..z(w-1), advances param
    store_wide_param             = 0x5F,  ///< Wide store to param: [width:u8][param_idx:u8] stores z0..z(w-1) to param, advances param
    cblas_bfgemm                 = 0x60,  ///< C = alpha*op(A_bf16)*op(B_bf16) + beta*C, BFMOPA widening GEMM
    cblas_igemm                  = 0x61,  ///< C = alpha*scvtf(A_i8 @ B_i8) + beta*C, SMOPA signed integer GEMM
    cblas_ugemm                  = 0x62,  ///< C = alpha*ucvtf(A_u8 @ B_u8) + beta*C, UMOPA unsigned integer GEMM
    cblas_usgemm                 = 0x63,  ///< C = alpha*scvtf(A_u8 @ B_i8) + beta*C, USMOPA mixed-sign GEMM
    gemm_tile_fp32               = 0x64,  ///< Tile-range FMOPA GEMM: compute output tiles [ti_start..ti_end, tj_start..tj_end]
    softmax_partial_fp32         = 0x65,  ///< Partial softmax: out=exp(in-max), returns (max, sum_exp) for cross-shard merging
    softmax_correct_fp32         = 0x66,  ///< Softmax correction: out *= exp(local_max - global_max), updates partial sum
    reduce_sum_sq_fp32           = 0x67,  ///< Partial sum of squares: result = sum(x[i]^2), for decomposed RMS norm
    reduce_col_sum_fp32          = 0x68,  ///< Column-wise sum: dst[j] = sum(src[i][j]) for i in 0..M-1, for bias gradients
    silu_backward_fp32           = 0x69,  ///< SiLU backward: dx = dy * sigmoid(x) * (1 + x*(1-sigmoid(x)))
    softmax_backward_fp32        = 0x6A,  ///< Softmax backward: dx = s*(dy - sum(s*dy))
    gelu_fp32                    = 0x6B,  ///< GeLU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    layer_norm_fp32              = 0x6C,  ///< Layer norm: (x-mean)/sqrt(var+eps)*gamma+beta
    causal_mask_fp32             = 0x6D,  ///< Causal mask: set scores[i][j] = -inf where j > i
    adam_step_fp32               = 0x6E,  ///< Fused Adam optimizer step over params/grads/m/v arrays
    gelu_backward_fp32           = 0x6F,  ///< GeLU backward: dx = dy * gelu'(x)
    rms_norm_backward_fp32       = 0x70,  ///< RMS norm backward: dx and dw gradients
    layer_norm_backward_fp32     = 0x71,  ///< Layer norm backward: dx, dgamma, dbeta gradients
    rope_backward_fp32           = 0x72,  ///< RoPE backward: inverse rotation (sin negated)
    cross_entropy_fp32           = 0x73,  ///< Cross-entropy loss + gradient: loss scalar + d_logits
    elementwise_sub_fp32         = 0x74,  ///< out[i] = a[i] - b[i]
    q4_k_gemv                    = 0x75,  ///< Q4_K quantized GEMV: super-blocks of 256, 4-bit quants + 6-bit scales
    q2_k_gemv                    = 0x76,  ///< Q2_K quantized GEMV: super-blocks of 256, 2-bit quants + 4-bit scales
    q3_k_gemv                    = 0x77,  ///< Q3_K quantized GEMV: super-blocks of 256, 3-bit quants (2+1 bit planes)
    q5_k_gemv                    = 0x78,  ///< Q5_K quantized GEMV: super-blocks of 256, 5-bit quants (4+1 bit planes)
    q6_k_gemv                    = 0x79,  ///< Q6_K quantized GEMV: super-blocks of 256, 6-bit quants (4+2 bit planes)
    flash_attention_fp32         = 0x7A,  ///< Fused flash attention: Q@K^T→scale→mask→softmax→@V, tiled online softmax
    get_rows_fp32                = 0x7B,  ///< Embedding lookup: gather rows by index from fp32 table
    get_rows_q8_0                = 0x7C,  ///< Embedding lookup + dequant from Q8_0 table
    get_rows_q4_0                = 0x7D,  ///< Embedding lookup + dequant from Q4_0 table
    dense_strided_fp32           = 0x7E,  ///< Fused matmul+bias+ReLU with explicit strides (lda, ldb, ldc)
    advance_param_stride         = 0x7F,  ///< Advance param[idx] by stride bytes, [idx:u8][stride:u32]
    NUM_OPCODES                  = 0x80,
};
/** --------------------------------------------------------------------------------------------------------- IsZVector / IsZStream helpers
 * @brief Type trait helpers to detect z_vector<T> and z_stream<T> without requiring T::value_type
 * on non-class types. Uses partial specialization to avoid SFINAE failures on scalars.
 */
template<typename T> struct is_z_vector : std::false_type {};
template<ValidZType T> struct is_z_vector<z_vector<T>> : std::true_type {};
template<typename T> struct is_z_stream : std::false_type {};
template<ValidZType T> struct is_z_stream<z_stream<T>> : std::true_type {};
/** --------------------------------------------------------------------------------------------------------- AllowedParameterTypes Concept
 * @brief Concept to constrain valid parameter types for dispatch() and program::emit().
 *  - ValidZType scalars (uint8_t, uint32_t, float, etc.) for immediates
 *  - z_vector<T> and z_stream<T> for typed pointer operands
 *  - aligned_pointer for raw aligned pointer operands
 *  - Op for opcode values
 */
template<typename T>
concept AllowedParameterTypes = ValidZType<T>
    || std::same_as<T, bool>
    || std::is_pointer_v<T>
    || is_z_vector<T>::value
    || is_z_stream<T>::value
    || std::is_same_v<T, aligned_pointer>;
/** --------------------------------------------------------------------------------------------------------- Opcode Dispatch
 * @brief Dispatches an operation to the assembly interpreter by encoding the opcode and its arguments into a
 * byte stream.
 * @param op The opcode to execute
 * @param args Immediates (uint32_t, int, float, uint8_t, bool) and operand pointers in encoding order
 */
template<typename... Args>
    requires (AllowedParameterTypes<Args> && ...)
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
/** --------------------------------------------------------------------------------------------------------- Program
 * @class program
 * @brief Builds a multi-opcode bytecode program that executes in a single streaming session.
 *  - Use the DSL script compiler (ane::script) to build programs from source text.
 *  - Call exec() to execute the entire program in one smstart/smstop pair.
 *  - Z-registers and ZA tiles persist across opcodes within the same program.
 */
class Compiler;  // Forward declaration — defined in parser.cpp
class program {
    friend class Compiler;  ///< DSL compiler needs raw emit/patch access
    friend class script;    ///< script::exec() needs patch/append access
    friend class prepared;  ///< prepared::exec() needs patch access
private:
    std::vector<uint8_t> bytecodes_;  ///< Accumulated bytecode buffer
    std::vector<size_t> loop_marks_;  ///< Stack of loop body start offsets for begin_loop/end_loop
    /** --------------------------------------------------------------------------------------------- Emit Raw Bytes
     * @brief Appends raw bytes to the bytecode buffer. Used by the DSL compiler for intrinsic
     * function calls where argument types are determined at parse time.
     */
    void emit_raw(const void* data, size_t len) {
        const auto* p = reinterpret_cast<const uint8_t*>(data);
        bytecodes_.insert(bytecodes_.end(), p, p + len);
    }
    void emit_raw_u8(uint8_t val) { bytecodes_.push_back(val); }
    /** --------------------------------------------------------------------------------------------- Mark
     * @brief Returns the current bytecode offset, for use as a loop target or jump reference.
     * @return Current offset in bytes from the start of the program
     */
    size_t mark() const { return bytecodes_.size(); }
    /** --------------------------------------------------------------------------------------------- Patch
     * @brief Patches a value at a specific byte offset in the bytecodes buffer.
     *  - Used by the DSL compiler for fixups (e.g., patching loop counts at label sites).
     * @param offset Byte offset into the bytecodes buffer
     * @param value The value to write
     */
    void patch_u8(size_t offset, uint8_t value) { bytecodes_[offset] = value; }
    void patch_u32(size_t offset, uint32_t value) {
        std::memcpy(&bytecodes_[offset], &value, sizeof(value));
    }
    void patch_u64(size_t offset, uint64_t value) {
        std::memcpy(&bytecodes_[offset], &value, sizeof(value));
    }
    /** --------------------------------------------------------------------------------------------- Append
     * @brief Appends another program's bytecodes to this one.
     * @param other The program whose bytecodes to append
     */
    void append(const program& other) {
        bytecodes_.insert(bytecodes_.end(), other.bytecodes_.begin(), other.bytecodes_.end());
    }
public:
    /** --------------------------------------------------------------------------------------------- Emit
     * @brief Appends a single opcode and its arguments to the program buffer.
     * @param op The opcode to emit
     * @param args Immediates and operand pointers in encoding order
     * @return Reference to this program for chaining
     */
    template<typename... Args>
        requires (AllowedParameterTypes<Args> && ...)
    program& emit(Op op, Args... args) {
        bytecodes_.push_back(static_cast<uint8_t>(op));
        auto append_arg = [&](auto arg) {
            const auto* p = reinterpret_cast<const uint8_t*>(&arg);
            bytecodes_.insert(bytecodes_.end(), p, p + sizeof(arg));
        };
        (append_arg(args), ...);
        return *this;
    }
    /** --------------------------------------------------------------------------------------------- Exec
     * @brief Executes the entire program in a single streaming session (one smstart/smstop pair).
     *  - All z-registers and ZA tiles persist across opcodes within this call.
     *  - After exec(), the program buffer is NOT cleared — call clear() to reuse.
     */
    void exec() {
        if (bytecodes_.empty()) return;
        asm volatile("" ::: "memory");
        interpreter::stream_exec(bytecodes_.data(), bytecodes_.size());
    }
    /** --------------------------------------------------------------------------------------------- Exec Tiles
     * @brief Executes the program and returns the scratchpad (ZA) state.
     *  - Appends a store_tiles instruction, runs everything, returns the captured tiles.
     *  - The appended instruction is removed afterward so the program can be reused.
     * @param tiles Reference to a z_tiles struct to receive the scratchpad state
     */
    void exec_into(z_tiles& tiles) {
        size_t saved_size = bytecodes_.size();
        emit(Op::store_tiles, reinterpret_cast<uintptr_t>(tiles.ptr()));
        exec();
        bytecodes_.resize(saved_size);
    }
    /** --------------------------------------------------------------------------------------------- Exec Returning Variable
     * @brief Executes the program and copies a specific variable slot to an output buffer.
     *  - Appends mov_zreg + store instructions, runs everything, removes the appended instructions.
     * @param var_slot Which variable slot to read (0-31)
     * @param out_ptr Where to write the 64 bytes (must be 64-byte aligned)
     */
    void exec_returning(uint8_t var_slot, void* out_ptr) {
        size_t saved_size = bytecodes_.size();
        emit(Op::mov_zreg, var_slot, uint8_t(0));
        emit(Op::store, reinterpret_cast<uintptr_t>(out_ptr));
        exec();
        bytecodes_.resize(saved_size);
    }
    /** --------------------------------------------------------------------------------------------- Clear
     * @brief Clears the bytecode buffer so the program can be rebuilt.
     */
    void clear() { bytecodes_.clear(); loop_marks_.clear(); }
    /** --------------------------------------------------------------------------------------------- Size
     * @brief Returns the current size of the bytecode buffer in bytes.
     */
    size_t size() const { return bytecodes_.size(); }
    /** --------------------------------------------------------------------------------------------- Begin Loop
     * @brief Emits a loop_begin opcode and records the body start for end_loop().
     * @param count Number of iterations
     * @return Reference to this program for chaining
     */
    program& begin_loop(uint8_t count) {
        emit(Op::loop_begin, count);
        loop_marks_.push_back(bytecodes_.size());
        return *this;
    }
    /** --------------------------------------------------------------------------------------------- End Loop
     * @brief Emits loop_end with the correct rewind offset.
     * @return Reference to this program for chaining
     */
    program& end_loop() {
        size_t body_start = loop_marks_.back();
        loop_marks_.pop_back();
        uint16_t offset = static_cast<uint16_t>(bytecodes_.size() - body_start + 3);
        emit(Op::loop_end, offset);
        return *this;
    }
};
/** --------------------------------------------------------------------------------------------------------- script
 * @class script
 * @brief Compiles a DSL source string into a reusable bytecode program.
 *  - Variables (ZVEC_F32) are auto-assigned to z-registers.
 *  - params[N] references map to the param table. Pointers are passed at exec() time.
 *  - Labels and goto provide counted loops.
 *  - Arithmetic operators (+, -, *) compile to register-addressed bytecode ops.
 *
 * Example:
 * @code
 *   ane::script s(R"(
 *       a: ZVEC_F32;
 *       b: ZVEC_F32;
 *       _LOOP_:;
 *       a.load(params[0]);
 *       b.load(params[1]);
 *       a = a + b;
 *       a.save(params[2]);
 *       params[0]++;
 *       params[1]++;
 *       params[2]++;
 *       goto _LOOP_ 64;
 *   )");
 *   s.exec({src_a_ptr, src_b_ptr, dst_ptr});
 * @endcode
 */
/** --------------------------------------------------------------------------------------------------------- PtrPatch
 * @struct PtrPatch
 * @brief Records a bytecode offset where a params[N] pointer placeholder needs to be patched
 * with the actual pointer value at exec() time. Used by DSL intrinsic function calls that emit
 * opcodes with inline pointer arguments (e.g., cblas_sgemm, softmax_partial_fp32).
 */
struct PtrPatch {
    size_t bytecode_offset;  ///< Offset within compiled bytecodes where the u64 placeholder sits
    uint8_t param_idx;       ///< Which params[N] to fill in (0-7)
};
/** --------------------------------------------------------------------------------------------------------- U32Patch
 * @struct U32Patch
 * @brief Records a bytecode offset where a params[N] runtime u32 placeholder needs to be patched
 * with the actual scalar value at exec() time. Used when intrinsic U32 arguments reference
 * params[N] instead of literal values, enabling runtime-configurable dimensions and strides.
 */
struct U32Patch {
    size_t bytecode_offset;  ///< Offset within compiled bytecodes where the u32 placeholder sits
    uint8_t param_idx;       ///< Which scalar params[N] to fill in (0-7)
};
/** --------------------------------------------------------------------------------------------------------- U8Patch
 * @struct U8Patch
 * @brief Records a bytecode offset where a params[N] runtime u8 placeholder needs to be patched.
 * The scalar value from exec() is truncated to 8 bits. Used for loop counts in goto statements.
 */
struct U8Patch {
    size_t bytecode_offset;  ///< Offset within compiled bytecodes where the u8 placeholder sits
    uint8_t param_idx;       ///< Which scalar params[N] to fill in (0-7)
};
/** --------------------------------------------------------------------------------------------------------- f32 Bit-Cast Helper
 * @brief Bit-casts a float to uint32_t for passing F32 runtime params through the scalar exec() API.
 * @param val The float value to encode
 * @return The same bits reinterpreted as uint32_t
 */
inline uint32_t f32(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, 4);
    return bits;
}
class script {
public:
    std::string source;      ///< DSL source text
private:
    program compiled_;        ///< Compiled bytecodes (without set_param preamble)
    std::vector<PtrPatch> patches_;      ///< Pointer fixups for intrinsic function calls
    std::vector<U32Patch> u32_patches_;  ///< Scalar u32 fixups for runtime intrinsic arguments
    std::vector<U8Patch> u8_patches_;    ///< Scalar u8 fixups for runtime loop counts
    bool compiled_valid_ = false;        ///< Whether compiled_ is up to date
public:
    script(const char* src);
    script(const std::string& src);
    /** --------------------------------------------------------------------------------------------- Compile
     * @brief Compiles the DSL source into bytecodes. Called automatically by exec() if needed.
     */
    void compile();
    /** --------------------------------------------------------------------------------------------- Exec
     * @brief Executes the script with the given parameter pointers in a single streaming session.
     *  - Emits set_param for each pointer, appends the compiled body, and runs it all.
     *  - Pointers should be z_stream-compatible (64-byte aligned).
     * @param params Ordered list of parameter pointers (matches params[0], params[1], etc.)
     */
    void exec(std::initializer_list<const void*> params);
    /** --------------------------------------------------------------------------------------------- Exec with Scalars
     * @brief Executes the script with separate pointer and scalar parameter lists.
     *  - Pointer params map to params[N] in PTR positions (intrinsic pointer arguments).
     *  - Scalar params map to params[N] in U32 positions (runtime-configurable dimensions/strides).
     * @param ptrs Ordered list of parameter pointers
     * @param scalars Ordered list of runtime u32 scalar values (dimensions, strides, counts)
     */
    void exec(std::initializer_list<const void*> ptrs,
              std::initializer_list<uint32_t> scalars);
};
/** --------------------------------------------------------------------------------------------------------- prepared
 * @class prepared
 * @brief A compile-once, execute-many program with zero-copy parameter updates.
 *
 * Unlike script::exec() which copies bytecodes and patches on every call, a prepared program
 * pre-bakes the set_param preamble and compiled body into a single buffer. On each exec(),
 * it patches parameter pointers and scalars in-place and runs — no allocation, no copy.
 *
 * Usage:
 * @code
 *   ane::prepared p = ane::prepared::from(R"(
 *       silu(params[0], params[1], params[2]);
 *   )", 2, 1);
 *
 *   // Hot loop — no compile, no copy, patches in-place
 *   p.exec({in1, out1}, {128});
 *   p.exec({in2, out2}, {256});
 * @endcode
 */
class prepared {
private:
    program prog_;                          ///< Pre-baked bytecodes (preamble + body)
    uint8_t num_ptrs_;                      ///< Number of pointer params in the preamble
    std::vector<size_t> ptr_preamble_;      ///< Offsets of u64 values in set_param instructions
    std::vector<PtrPatch> ptr_patches_;     ///< Inline PTR patch locations (relative to body start)
    std::vector<U32Patch> u32_patches_;     ///< Inline U32/F32 patch locations (relative to body start)
    std::vector<U8Patch> u8_patches_;       ///< Inline U8 patch locations (loop counts, relative to body start)
    size_t preamble_size_;                  ///< Byte offset where the compiled body starts
public:
    /** --------------------------------------------------------------------------------------------- from
     * @brief Compile a DSL source into a prepared program.
     * @param source DSL source text
     * @param num_ptrs Number of pointer parameters (used for set_param preamble)
     * @param num_scalars Number of scalar parameters (unused at prepare time, validated at exec)
     */
    static prepared from(const char* source, uint8_t num_ptrs, uint8_t num_scalars = 0);
    static prepared from(const std::string& source, uint8_t num_ptrs, uint8_t num_scalars = 0);
    /** --------------------------------------------------------------------------------------------- exec (ptrs only)
     * @brief Execute with pointer parameters only. Patches in-place, runs, no copy.
     */
    void exec(std::initializer_list<const void*> ptrs);
    /** --------------------------------------------------------------------------------------------- exec (ptrs + scalars)
     * @brief Execute with pointer and scalar parameters. Patches in-place, runs, no copy.
     */
    void exec(std::initializer_list<const void*> ptrs,
              std::initializer_list<uint32_t> scalars);
};
} // namespace ane
#else
#error "This library is only supported on Apple Silicon (aarch64 architecture)."
#endif // defined(__aarch64__) && defined(__APPLE__)
