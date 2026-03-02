#pragma once
#if defined(__APPLE__)
#include <cstdint>
#include <cstring>
#if defined(__has_include) && __has_include(<arm_neon.h>)
#include <arm_neon.h>
#else
struct alignas(2) bfloat16_t {
    uint16_t value;
    bfloat16_t() : value(0) {}
    explicit bfloat16_t(float f) {
        value = static_cast<uint16_t>(reinterpret_cast<uint32_t&>(f) >> 16);
    }
    operator float() const {
        uint32_t temp = static_cast<uint32_t>(value) << 16;
        return reinterpret_cast<float&>(temp);
    }
};
#endif

namespace ane {
namespace kernel {
extern "C" {
void add_fp32(const float* a, const float* b, float* c, long n);
void sub_fp32(const float* a, const float* b, float* c, long n);
void mul_fp32(const float* a, const float* b, float* c, long n);
void div_fp32(const float* a, const float* b, float* c, long n);
void max_fp32(const float* a, const float* b, float* c, long n);
void min_fp32(const float* a, const float* b, float* c, long n);
void fma_fp32(const float* a, const float* b, const float* c, float* out, long n);
void scalar_mul_fp32(const float* input, float scalar, float* output, long n);
void scalar_add_fp32(const float* input, float scalar, float* output, long n);
void clamp_fp32(const float* input, float lo, float hi, float* output, long n);
void fill_fp32(float* output, float value, long n);
void copy_fp32(const float* src, float* dst, long n);
void slice_fp32(const float* src, float* dst, long offset, long count);
void reshape_fp32(const float* src, float* dst, long n);
void neg_fp32(const float* input, float* output, long n);
void abs_fp32(const float* input, float* output, long n);
void sqrt_fp32(const float* input, float* output, long n);
void rsqrt_fp32(const float* input, float* output, long n);
void exp_fp32(const float* input, float* output, long n);
void log_fp32(const float* input, float* output, long n);
void relu_fp32(const float* input, float* output, long n);
void sigmoid_fp32(const float* input, float* output, long n);
void silu_fp32(const float* input, float* output, long n);
void silu_bwd_fp32(const float* x, const float* dy, float* dx, long n);
void relu_backward_fp32(const float* dy, const float* x, float* dx, long n);
void sigmoid_backward_fp32(const float* dy, const float* y, float* dx, long n);
void tanh_backward_fp32(const float* dy, const float* y, float* dx, long n);
void gelu_backward_fp32(const float* dy, const float* x, float* dx, long n);
void tanh_fp32(const float* input, float* output, long n);
void gelu_fp32(const float* input, float* output, long n);
void leaky_relu_fp32(const float* input, float* output, float alpha, long n);
void elu_fp32(const float* input, float* output, float alpha, long n);
void selu_fp32(const float* input, float* output, long n);
void softplus_fp32(const float* input, float* output, long n);
void mish_fp32(const float* input, float* output, long n);
void pow_fp32(const float* base, const float* exponent, float* output, long n);
void softmax_fp32(const float* input, float* output, long n);
void softmax_batch_fp32(const float* input, float* output, long batch_size, long seq_len);
void softmax_backward_fp32(const float* dy, const float* y, float* dx, long n);
float reduce_sum_fp32(const float* input, long n);
float reduce_max_fp32(const float* input, long n);
float reduce_min_fp32(const float* input, long n);
void global_max_pool_fp32(const float* input, float* output, long batch, long spatial, long channels);
void global_avg_pool_fp32(const float* input, float* output, long batch, long spatial, long channels);
float dot_fp32(const float* a, const float* b, long n);
float sumsqr_fp32(const float* input, long n);
float mse_loss_fp32(const float* pred, const float* target, long n);
float mae_loss_fp32(const float* pred, const float* target, long n);
float cross_entropy_loss_fp32(const float* log_probs, const int32_t* targets, long batch_size, long num_classes);
float bce_loss_fp32(const float* pred, const float* target, long n);
void log_softmax_fp32(const float* input, float* output, long n);
int32_t argmax_fp32(const float* input, long n);
void where_fp32(const uint32_t* cond, const float* a, const float* b, float* out, long n);
void concat_fp32(const float* a, const float* b, float* out, long na, long nb);
void prelu_fp32(const float* input, const float* alpha, float* output, long n);
void zip_fp32(const float* a, const float* b, float* out, long n);
void unzip_fp32(const float* input, float* a, float* b, long n);
void matmul_fp32(const float* A, const float* B, float* C, long M, long N, long K);
void matmul_fp32_nt(const float* A, const float* B, float* C, long M, long N, long K);
void matmul_fp32_tn(const float* A, const float* B, float* C, long M, long N, long K);
void matmul_int8(const int8_t* A, const int8_t* B, int32_t* C, long M, long N, long K, void* a_work, void* b_work);
void matmul_uint8(const uint8_t* A, const uint8_t* B, uint32_t* C, long M, long N, long K, void* a_work, void* b_work);
void matmul_bfp16(const bfloat16_t* A, const bfloat16_t* B, float* C, long M, long N, long K, void* a_work, void* b_work);
void matmul_bfp16_nt(const bfloat16_t* A, const bfloat16_t* B, float* C, long M, long N, long K);
void matmul_bfp16_tn(const bfloat16_t* A, const bfloat16_t* B, float* C, long M, long N, long K);
void transpose_fp32(const float* src, float* dst, long rows, long cols);
void fp32_to_bf16(const float* input, bfloat16_t* output, long n);
void bf16_to_fp32(const bfloat16_t* input, float* output, long n);
void fp32_to_int32(const float* input, int32_t* output, long n);
void int32_to_fp32(const int32_t* input, float* output, long n);
void quantize_fp32_int8(const float* input, int8_t* output, float scale, long n);
void quantize_fp32_int4(const float* input, uint8_t* output, float scale, float zero_point, long n);
void dequantize_int8_fp32(const int8_t* input, float* output, float scale, long n);
void dequantize_int4_fp32(const uint8_t* input, float* output, float scale, float zero_point, long n);
void q8_0_matvec(const int8_t* A_quants, const float* A_scales, const int8_t* B_quants_bm, const float* B_scales_bm, float* C, long N, long K);
void quantize_act_fp32_int8(const float* input, int8_t* output, float* block_scales, long K);
void add_bf16(const bfloat16_t* a, const bfloat16_t* b, bfloat16_t* c, long n);
void relu_bf16(const bfloat16_t* input, bfloat16_t* output, long n);
void and_u32(const uint32_t* a, const uint32_t* b, uint32_t* c, long n);
void or_u32(const uint32_t* a, const uint32_t* b, uint32_t* c, long n);
void xor_u32(const uint32_t* a, const uint32_t* b, uint32_t* c, long n);
void not_u32(const uint32_t* input, uint32_t* output, long n);
void shl_u32(const uint32_t* input, uint32_t* output, int shift, long n);
void shr_u32(const uint32_t* input, uint32_t* output, int shift, long n);
void param_update_fp32(const float* params, const float* const gradients[], float* output, long n, float learning_rate, float inv_updates, int count);
void sgd_fp32(const float* params, const float* gradients, float* output, long n, float scale, long batch_size);
void adam_fp32(float* params, const float* grads, float* m, float* v, long n, float lr_corrected, float beta1, float beta2, float eps);
void l2sq_batch_fp32(const float* query, const float* data, float* output, long dim, long n_vectors);
void broadcast_kv_fp32(const float* in, float* out, long seq_len, long head_dim, long n_kv_heads, long n_heads);
void causal_mask_fp32(float* data, long seq_len, long num_heads);
void rope_fp32(float* q, float* k, const float* cos_cache, const float* sin_cache, long num_q_heads, long num_kv_heads, long head_dim);
int probe_za_fmla_vgx4(void);
void softmax_fp32_reduce(float*, const float*, long, long);
void identity_reduce(void);
float dot_combine_fp32(const float*, long);
float sumsqr_combine_fp32(const float*, long);
long argmax_combine_fp32(const float*, const long*, long);
void matmul_fp32_pack_b(const float*, float*, long, long);
void matmul_bfp16_pack_b(const bfloat16_t*, bfloat16_t*, long, long);
void fused_rms_norm_scale_fp32(const float* input, const float* weight, float* out, float inv_rms, long n);
void layernorm_fp32(const float* x, const float* gamma, const float* beta, float* out, float* mean_out, float* inv_std_out, float eps, long n);
void layernorm_bwd_fp32(const float* dy, const float* x, const float* gamma, float mean, float inv_std, float* dx, float* dgamma, float* dbeta, long n);
void fused_silu_gate_mul_fp32(const float* gate, const float* up, float* out, long n);
void fused_weighted_add_fp32(const float* acc, const float* expert_out, float* out, float w, long n);
void fused_ssm_gate_fp32(const float* gate, const float* y, float* out, long n);
void fused_residual_sumsqr_fp32(const float* residual, const float* x, float* hidden, float* ss_out, long n);
void conv2d_fp32(const float* input_NHWC, const float* weight_HWIO, float* output_NHWC, long N, long H, long W, long C_in, long KH, long KW, long C_out, long stride, long pad);
void conv2d_backward_input_fp32(const float* grad_out_NHWC, const float* weight_HWIO, float* grad_in_NHWC, long N, long H, long W, long C_in, long KH, long KW, long C_out, long stride, long pad);
void conv2d_backward_weight_fp32(const float* input_NHWC, const float* grad_out_NHWC, float* grad_weight_HWIO, float* grad_bias, long N, long H, long W, long C_in, long KH, long KW, long C_out, long stride, long pad);
void conv2d_bias_fp32(const float* input_NHWC, const float* weight_HWIO, const float* bias, float* output_NHWC, long N, long H, long W, long C_in, long KH, long KW, long C_out, long stride, long pad);
void conv2d_bias_relu_fp32(const float* input_NHWC, const float* weight_HWIO, const float* bias_C, float* output_NHWC, long N, long H, long W, long C_in, long KH, long KW, long C_out, long stride, long pad);
void conv2d_bias_relu6_fp32(const float* input_NHWC, const float* weight_HWIO, const float* bias_C, float* output_NHWC, long N, long H, long W, long C_in, long KH, long KW, long C_out, long stride, long pad);
void conv2d_relu_fp32(const float* input_NHWC, const float* weight_HWIO, float* output_NHWC, long N, long H, long W, long C_in, long KH, long KW, long C_out, long stride, long pad);
void conv2d_bn_fp32(const float* input_NHWC, const float* weight_HWIO, const float* bn_scale, const float* bn_shift, float* output_NHWC, long N, long H, long W, long C_in, long KH, long KW, long C_out, long stride, long pad);
void conv2d_bn_relu_fp32(const float* input_NHWC, const float* weight_HWIO, const float* bn_scale, const float* bn_shift, float* output_NHWC, long N, long H, long W, long C_in, long KH, long KW, long C_out, long stride, long pad);
void conv2d_swish_fp32( const float* input_NHWC, const float* weight_HWIO, float* output_NHWC, long N, long H, long W, long C_in, long KH, long KW, long C_out, long stride, long pad);
void conv2d_gelu_fp32(const float* input_NHWC, const float* weight_HWIO, float* output_NHWC, long N, long H, long W, long C_in, long KH, long KW, long C_out, long stride, long pad);
void dense_relu_fp32(const float* input_MK, const float* weight_KN, float* output_MN, long M, long N, long K);
void dense_softmax_fp32(const float* input_MK, const float* weight_KN, float* output_MN, long M, long N, long K);
void dense_bias_fp32(const float* input_MK, const float* weight_KN, const float* bias_N, float* output_MN, long M, long N, long K);
void dense_bias_relu_fp32(const float* input_MK, const float* weight_KN, const float* bias_N, float* output_MN, long M, long N, long K);
void dense_gelu_fp32(const float* input_MK, const float* weight_KN, float* output_MN, long M, long N, long K);
void dense_silu_fp32(const float* input_MK, const float* weight_KN, float* output_MN, long M, long N, long K);
void dense_tanh_fp32(const float* input_MK, const float* weight_KN, float* output_MN, long M, long N, long K);
void dense_sigmoid_fp32(const float* input_MK, const float* weight_KN, float* output_MN, long M, long N, long K);
void dense_bias_gelu_fp32(const float* input_MK, const float* weight_KN, const float* bias_N, float* output_MN, long M, long N, long K);
void dense_bias_silu_fp32(const float* input_MK, const float* weight_KN, const float* bias_N, float* output_MN, long M, long N, long K);
void dense_residual_fp32(const float* input_MK, const float* weight_KN, const float* residual_MN, float* output_MN, long M, long N, long K);
void dense_bias_residual_fp32(const float* input_MK, const float* weight_KN, const float* bias_N, const float* residual_MN, float* output_MN, long M, long N, long K);
void tbl_u8(const uint8_t* table, const uint8_t* indices, uint8_t* output, long n_table, long n);
void tbl2_u8(const uint8_t* table, const uint8_t* indices, uint8_t* output, long n_table, long n);
void tbx_u8(const uint8_t* table, const uint8_t* indices, const uint8_t* fallback, uint8_t* output, long n_table, long n);
void luti4_u8(const uint8_t* lut64, const uint8_t* packed_indices, uint8_t* output, long n);
void luti2_u8(const uint8_t* lut64, const uint8_t* packed_indices, uint8_t* output, long n);
void fused_dense_layernorm_fp32(const float* W, int m, int n, const float* x, const float* gamma, const float* beta, float eps, float* out);
void fused_dense_leaky_relu_fp32(const float* W, int m, int n, const float* x, const float* bias, float alpha, float* out);
void fused_dense_mish_fp32(const float* W, int m, int n, const float* x, const float* bias, float* out);
void fused_dense_elu_fp32(const float* W, int m, int n, const float* x, const float* bias, float alpha, float* out);
void fused_batchnorm_relu_fp32(const float* x, long n, float mean, float inv_std, const float* gamma, const float* beta, float* out);
void fused_glu_fp32(const float* x, long n, float* out);
void argmax_fp32_st(const float* in, long n, long* out_idx, long wid, long nw);
void topk_threshold_fp32_st(const float* in, long n, float threshold, float* out_values, long* out_indices, long* out_count, long wid, long nw);
void cross_attn_prefill_fp32(const float* q, const float* k, const float* v, float* output, long q_len, long kv_len, long num_heads, long head_dim, float scale);
void cross_attn_decode_cached_fp32(const float* q, const float* k, const float* v, float* output, long kv_len, long num_heads, long head_dim, float scale);
void sdp_attn_backward_fp32(const float* dO, const float* q, const float* k, const float* v, const float* attn_weights, float* dQ, float* dK, float* dV, long seq_len, long num_heads, long head_dim, float scale);
void sdp_attn_prefill_causal_fp32(const float* qkv, float* output, long seq_len, long num_heads, long head_dim, float scale);
void sdp_attn_prefill_noncausal_fp32(const float* qkv, float* output, long seq_len, long num_heads, long head_dim, float scale);
void sdp_attn_decode_cached_fp32(const float* q, const float* k_cache, const float* v_cache, float* output, long cache_len, long num_heads, long head_dim, float scale);
void kv_cache_append_fp32(float* k_cache, float* v_cache, const float* new_k, const float* new_v, long pos, long num_heads, long head_dim);
void gqa_attn_prefill_causal_fp32(const float* qkv, float* output, long seq_len, long n_heads, long n_kv_heads, long head_dim, float scale);
void gqa_attn_prefill_noncausal_fp32(const float* qkv, float* output, long seq_len, long n_heads, long n_kv_heads, long head_dim, float scale);
void gqa_attn_decode_cached_fp32(const float* q, const float* k_cache, const float* v_cache, float* output, long cache_len, long n_heads, long n_kv_heads, long head_dim, float scale);
void flash_attn_prefill_causal_fp32(const float* qkv, float* output, long seq_len, long num_heads, long head_dim, float scale);
void flash_attn_prefill_noncausal_fp32(const float* qkv, float* output, long seq_len, long num_heads, long head_dim, float scale);
void flash_attn_decode_cached_fp32(const float* q, const float* k_cache, const float* v_cache, float* output, long cache_len, long num_heads, long head_dim, float scale);
void fused_attn_layernorm_fp32(const float* qkv, float* output, const float* gamma, const float* beta, long seq_len, long num_heads, long head_dim, float scale, float eps);
void fused_attn_residual_fp32(const float* qkv, const float* residual, float* output, long seq_len, long num_heads, long head_dim, float scale);
void dropout_fp32(const float* input, const float* mask, float* output, float inv_keep_prob, long n);
void gaussian_noise_fp32(const float* input, float* output, float stddev, long n, uint64_t seed);
void log1p_fp32(const float* input, float* output, long n);
void sign_fp32(const float* input, float* output, long n);
void hard_sigmoid_fp32(const float* input, float* output, long n);
} // extern "C"
} // namespace kernel
} // namespace ane
#endif 