/** --------------------------------------------------------------------------------------------------------- GGML ANE Backend
 * @file ggml-ane.cpp
 * @brief GGML backend implementation for Apple Silicon SME via the ane bytecode interpreter.
 *
 * Implements the ggml backend interface: buffer management, operation support checks, and
 * graph compute dispatch. All tensor math is dispatched through ane::program or ane::dispatch
 * to run on the SME matrix unit.
 *
 * @author Josh Morgan (@joshmorgan1000 on GitHub)
 * Released under the MIT License
 */
#include "ggml-ane.h"
#include "ggml.h"
#include "ggml-backend-impl.h"
#include <ane/ane.hpp>
#include <cstdlib>
#include <cstring>
#include <cstdio>

/** --------------------------------------------------------------------------------------------------------- Buffer Context
 * @struct ane_buffer_context
 * @brief Holds the allocated memory for an ANE buffer. 4KB-aligned for ZA tile compatibility.
 */
struct ane_buffer_context {
    void* data;
    size_t size;
};
/** --------------------------------------------------------------------------------------------------------- Buffer Type Functions
 */
static const char* ane_buft_get_name(ggml_backend_buffer_type_t buft) {
    return "ANE";
    (void)buft;
}
static ggml_backend_buffer_t ane_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    size = ((size + 4095) / 4096) * 4096;  ///< Round up to 4KB for ZA alignment
    auto* ctx = new ane_buffer_context{std::aligned_alloc(4096, size), size};
    if (!ctx->data) {
        delete ctx;
        return nullptr;
    }
    std::memset(ctx->data, 0, size);
    static struct ggml_backend_buffer_i iface = {
        /* free_buffer   */ [](ggml_backend_buffer_t buffer) {
            auto* c = static_cast<ane_buffer_context*>(buffer->context);
            std::free(c->data);
            delete c;
        },
        /* get_base      */ [](ggml_backend_buffer_t buffer) -> void* {
            return static_cast<ane_buffer_context*>(buffer->context)->data;
        },
        /* init_tensor   */ nullptr,
        /* memset_tensor */ [](ggml_backend_buffer_t buffer, struct ggml_tensor* tensor,
                              uint8_t value, size_t offset, size_t size) {
            std::memset(static_cast<char*>(tensor->data) + offset, value, size);
            (void)buffer;
        },
        /* set_tensor    */ [](ggml_backend_buffer_t buffer, struct ggml_tensor* tensor,
                              const void* data, size_t offset, size_t size) {
            std::memcpy(static_cast<char*>(tensor->data) + offset, data, size);
            (void)buffer;
        },
        /* get_tensor    */ [](ggml_backend_buffer_t buffer, const struct ggml_tensor* tensor,
                              void* data, size_t offset, size_t size) {
            std::memcpy(data, static_cast<const char*>(tensor->data) + offset, size);
            (void)buffer;
        },
        /* cpy_tensor    */ nullptr,
        /* clear         */ [](ggml_backend_buffer_t buffer, uint8_t value) {
            auto* c = static_cast<ane_buffer_context*>(buffer->context);
            std::memset(c->data, value, c->size);
        },
        /* reset         */ nullptr,
    };
    return ggml_backend_buffer_init(buft, iface, ctx, size);
}
static size_t ane_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;  ///< 64-byte alignment for z-vector operations
    (void)buft;
}
static bool ane_buft_is_host(ggml_backend_buffer_type_t buft) {
    return true;  ///< ANE buffers are in host memory (shared with CPU)
    (void)buft;
}
static struct ggml_backend_buffer_type ane_buffer_type = {
    /* iface */ {
        ane_buft_get_name,
        ane_buft_alloc_buffer,
        ane_buft_get_alignment,
        /* get_max_size  */ nullptr,
        /* get_alloc_size */ nullptr,
        ane_buft_is_host,
    },
    /* device  */ nullptr,  ///< Set during device init
    /* context */ nullptr,
};
/** --------------------------------------------------------------------------------------------------------- Backend Functions
 */
static const char* ane_backend_get_name(ggml_backend_t backend) {
    return "ANE";
    (void)backend;
}
static void ane_backend_free(ggml_backend_t backend) {
    delete backend;
}
/** --------------------------------------------------------------------------------------------------------- Operation Dispatch
 * @brief The core compute dispatch. Walks the compute graph and runs each supported operation
 * via the ane bytecode interpreter.
 */
static bool ane_compute_forward(const struct ggml_tensor* tensor) {
    const struct ggml_tensor* src0 = tensor->src[0];
    const struct ggml_tensor* src1 = tensor->src[1];
    const int64_t ne0  = tensor->ne[0];  ///< innermost dimension
    const int64_t ne1  = tensor->ne[1];
    const int64_t ne00 = src0 ? src0->ne[0] : 0;
    const int64_t ne01 = src0 ? src0->ne[1] : 0;
    const int64_t ne10 = src1 ? src1->ne[0] : 0;
    const int64_t ne11 = src1 ? src1->ne[1] : 0;
    switch (tensor->op) {
        case GGML_OP_MUL_MAT: {
            // src0 = weights (possibly quantized), src1 = input (fp32), dst = output (fp32)
            const enum ggml_type type0 = src0->type;
            float* dst_data  = static_cast<float*>(tensor->data);
            float* src1_data = static_cast<float*>(src1->data);
            const int M = static_cast<int>(ne1);   ///< output rows (batch or sequence)
            const int N = static_cast<int>(ne0);   ///< output cols (= src0 rows for matmul)
            const int K = static_cast<int>(ne00);  ///< shared dimension
            if (type0 == GGML_TYPE_F32) {
                float* src0_data = static_cast<float*>(src0->data);
                ane::dispatch(ane::Op::cblas_sgemm,
                    uint8_t(0),                    ///< flags: no transpose
                    uint32_t(N), uint32_t(M), uint32_t(K),
                    uint32_t(K), uint32_t(K), uint32_t(N),
                    1.0f, 0.0f,
                    reinterpret_cast<uintptr_t>(src0_data),
                    reinterpret_cast<uintptr_t>(src1_data),
                    reinterpret_cast<uintptr_t>(dst_data));
            } else if (type0 == GGML_TYPE_Q8_0) {
                // Q8_0 quantized matmul via GEMV per output column
                ane::dispatch(ane::Op::q8_0_gemv,
                    uint32_t(N), uint32_t(K),
                    reinterpret_cast<uintptr_t>(src1_data),
                    reinterpret_cast<uintptr_t>(src0->data),
                    reinterpret_cast<uintptr_t>(dst_data));
            } else if (type0 == GGML_TYPE_Q4_0) {
                ane::dispatch(ane::Op::q4_0_gemv,
                    uint32_t(N), uint32_t(K),
                    reinterpret_cast<uintptr_t>(src1_data),
                    reinterpret_cast<uintptr_t>(src0->data),
                    reinterpret_cast<uintptr_t>(dst_data));
            } else {
                return false;  ///< Unsupported quantization format
            }
            return true;
        }
        case GGML_OP_ADD: {
            float* dst_data  = static_cast<float*>(tensor->data);
            float* src0_data = static_cast<float*>(src0->data);
            float* src1_data = static_cast<float*>(src1->data);
            int count = static_cast<int>(ggml_nelements(tensor));
            int padded = ((count + 15) / 16) * 16;
            ane::dispatch(ane::Op::elementwise_add_fp32,
                uint32_t(padded),
                reinterpret_cast<uintptr_t>(src0_data),
                reinterpret_cast<uintptr_t>(src1_data),
                reinterpret_cast<uintptr_t>(dst_data));
            return true;
        }
        case GGML_OP_MUL: {
            float* dst_data  = static_cast<float*>(tensor->data);
            float* src0_data = static_cast<float*>(src0->data);
            float* src1_data = static_cast<float*>(src1->data);
            int count = static_cast<int>(ggml_nelements(tensor));
            int padded = ((count + 15) / 16) * 16;
            ane::dispatch(ane::Op::elementwise_mul_fp32,
                uint32_t(padded),
                reinterpret_cast<uintptr_t>(src0_data),
                reinterpret_cast<uintptr_t>(src1_data),
                reinterpret_cast<uintptr_t>(dst_data));
            return true;
        }
        case GGML_OP_SCALE: {
            float* dst_data  = static_cast<float*>(tensor->data);
            float* src0_data = static_cast<float*>(src0->data);
            float scale;
            std::memcpy(&scale, tensor->op_params, sizeof(float));
            int count = static_cast<int>(ggml_nelements(tensor));
            // Scale is just mul by broadcast scalar — copy src0 to dst, then scale in place
            // For now, use elementwise_scaled_add with a=0 trick, or just memcpy + scale
            if (dst_data != src0_data) {
                std::memcpy(dst_data, src0_data, ggml_nbytes(tensor));
            }
            // TODO: add a dedicated scale kernel, for now use CPU fallback
            for (int i = 0; i < count; i++) dst_data[i] *= scale;
            return true;
        }
        case GGML_OP_RMS_NORM: {
            float* dst_data  = static_cast<float*>(tensor->data);
            float* src0_data = static_cast<float*>(src0->data);
            float eps;
            std::memcpy(&eps, tensor->op_params, sizeof(float));
            int dim = static_cast<int>(ne00);
            int rows = static_cast<int>(ne01);
            int padded_dim = ((dim + 15) / 16) * 16;
            // RMS norm is applied per-row. Our kernel processes one row at a time.
            // Weight = all 1.0 for raw RMS norm (ggml applies weight separately via MUL)
            // TODO: use a static all-ones buffer
            alignas(64) float ones[4096];
            for (int i = 0; i < padded_dim; i++) ones[i] = 1.0f;
            for (int r = 0; r < rows; r++) {
                ane::dispatch(ane::Op::rms_norm_fp32,
                    uint32_t(padded_dim), eps,
                    reinterpret_cast<uintptr_t>(src0_data + r * ne00),
                    reinterpret_cast<uintptr_t>(ones),
                    reinterpret_cast<uintptr_t>(dst_data + r * ne0));
            }
            return true;
        }
        case GGML_OP_SOFT_MAX: {
            float* dst_data  = static_cast<float*>(tensor->data);
            float* src0_data = static_cast<float*>(src0->data);
            int dim = static_cast<int>(ne00);
            int rows = static_cast<int>(ne01);
            int padded_dim = ((dim + 15) / 16) * 16;
            for (int r = 0; r < rows; r++) {
                ane::dispatch(ane::Op::softmax_fp32,
                    uint32_t(padded_dim),
                    reinterpret_cast<uintptr_t>(src0_data + r * ne00),
                    reinterpret_cast<uintptr_t>(dst_data + r * ne0));
            }
            return true;
        }
        case GGML_OP_UNARY: {
            const enum ggml_unary_op uop = ggml_get_unary_op(tensor);
            if (uop == GGML_UNARY_OP_SILU) {
                float* dst_data  = static_cast<float*>(tensor->data);
                float* src0_data = static_cast<float*>(src0->data);
                int count = static_cast<int>(ggml_nelements(tensor));
                int padded = ((count + 15) / 16) * 16;
                ane::dispatch(ane::Op::silu_fp32,
                    uint32_t(padded),
                    reinterpret_cast<uintptr_t>(src0_data),
                    reinterpret_cast<uintptr_t>(dst_data));
                return true;
            }
            return false;  ///< Other unary ops not yet supported
        }
        case GGML_OP_ROPE: {
            float* dst_data  = static_cast<float*>(tensor->data);
            float* src0_data = static_cast<float*>(src0->data);
            // RoPE params are packed in op_params
            // src1 contains the position indices
            const int n_dims   = static_cast<int>(ne00);
            const int n_heads  = static_cast<int>(ne01);
            int32_t* pos_data = static_cast<int32_t*>(src1->data);
            float theta_base = 10000.0f;  ///< Default for most models
            // Extract theta from op_params if available
            if (tensor->op_params[6] != 0) {
                std::memcpy(&theta_base, &tensor->op_params[6], sizeof(float));
            }
            for (int h = 0; h < n_heads; h++) {
                int pos = pos_data[h % src1->ne[0]];
                ane::dispatch(ane::Op::rope_fp32,
                    uint32_t(n_dims), uint32_t(pos), theta_base,
                    reinterpret_cast<uintptr_t>(src0_data + h * n_dims),
                    reinterpret_cast<uintptr_t>(dst_data + h * n_dims));
            }
            return true;
        }
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP: {
            if (tensor->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) {
                std::memcpy(tensor->data, src0->data, ggml_nbytes(tensor));
                return true;
            }
            return false;
        }
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;  ///< These are metadata-only ops, no compute needed
        default:
            return false;
    }
}
static enum ggml_status ane_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (ggml_is_empty(node)) continue;
        if (!ane_compute_forward(node)) {
            fprintf(stderr, "ANE backend: unsupported op %s (type %d)\n",
                ggml_op_name(node->op), node->op);
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
    (void)backend;
}
/** --------------------------------------------------------------------------------------------------------- Device Functions
 */
static const char* ane_dev_get_name(ggml_backend_dev_t dev) {
    return "ANE0";
    (void)dev;
}
static const char* ane_dev_get_description(ggml_backend_dev_t dev) {
    return "Apple Silicon SME (Scalable Matrix Extension)";
    (void)dev;
}
static void ane_dev_get_memory(ggml_backend_dev_t dev, size_t* free, size_t* total) {
    *free  = 0;  ///< Shared memory, no dedicated pool
    *total = 0;
    (void)dev;
}
static enum ggml_backend_dev_type ane_dev_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    (void)dev;
}
static void ane_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props* props) {
    props->name        = ane_dev_get_name(dev);
    props->description = ane_dev_get_description(dev);
    props->type        = ane_dev_get_type(dev);
    props->memory_free  = 0;
    props->memory_total = 0;
    props->caps = {
        /* async     */ false,
        /* host_buffer */ false,
        /* buffer_from_host_ptr */ true,
        /* events    */ false,
    };
}
static ggml_backend_t ane_dev_init_backend(ggml_backend_dev_t dev, const char* params) {
    static struct ggml_backend_i ane_backend_iface = {
        /* get_name           */ ane_backend_get_name,
        /* free               */ ane_backend_free,
        /* set_tensor_async   */ nullptr,
        /* get_tensor_async   */ nullptr,
        /* cpy_tensor_async   */ nullptr,
        /* synchronize        */ nullptr,
        /* graph_plan_create  */ nullptr,
        /* graph_plan_free    */ nullptr,
        /* graph_plan_update  */ nullptr,
        /* graph_plan_compute */ nullptr,
        /* graph_compute      */ ane_graph_compute,
        /* event_record       */ nullptr,
        /* event_wait         */ nullptr,
        /* graph_optimize     */ nullptr,
    };
    static ggml_guid guid = {};
    auto* backend = new ggml_backend{};
    backend->guid    = guid;
    backend->iface   = ane_backend_iface;
    backend->device  = dev;
    backend->context = nullptr;
    return backend;
    (void)params;
}
static ggml_backend_buffer_type_t ane_dev_get_buffer_type(ggml_backend_dev_t dev) {
    ane_buffer_type.device = dev;
    return &ane_buffer_type;
    (void)dev;
}
static bool ane_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT: {
            enum ggml_type t = op->src[0]->type;
            return t == GGML_TYPE_F32 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0;
        }
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_SCALE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_UNARY:
            return ggml_get_unary_op(op) == GGML_UNARY_OP_SILU && op->type == GGML_TYPE_F32;
        case GGML_OP_ROPE:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
    (void)dev;
}
static bool ane_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == &ane_buffer_type;
    (void)dev;
}
/** --------------------------------------------------------------------------------------------------------- Device & Registry
 */
static struct ggml_backend_device_i ane_device_iface = {
    /* get_name             */ ane_dev_get_name,
    /* get_description      */ ane_dev_get_description,
    /* get_memory           */ ane_dev_get_memory,
    /* get_type             */ ane_dev_get_type,
    /* get_props            */ ane_dev_get_props,
    /* init_backend         */ ane_dev_init_backend,
    /* get_buffer_type      */ ane_dev_get_buffer_type,
    /* get_host_buffer_type */ nullptr,
    /* buffer_from_host_ptr */ nullptr,
    /* supports_op          */ ane_dev_supports_op,
    /* supports_buft        */ ane_dev_supports_buft,
    /* offload_op           */ nullptr,
    /* event_new            */ nullptr,
    /* event_free           */ nullptr,
    /* event_synchronize    */ nullptr,
};
static struct ggml_backend_device ane_device = {
    /* iface   */ ane_device_iface,
    /* reg     */ nullptr,  ///< Set during registration
    /* context */ nullptr,
};
static const char* ane_reg_get_name(ggml_backend_reg_t reg) {
    return "ANE";
    (void)reg;
}
static size_t ane_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;  ///< One ANE device per system
    (void)reg;
}
static ggml_backend_dev_t ane_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index == 0) {
        ane_device.reg = reg;
        return &ane_device;
    }
    return nullptr;
    (void)reg;
}
static struct ggml_backend_reg_i ane_reg_iface = {
    /* get_name         */ ane_reg_get_name,
    /* get_device_count */ ane_reg_get_device_count,
    /* get_device       */ ane_reg_get_device,
    /* get_proc_address */ nullptr,
};
static struct ggml_backend_reg ane_reg = {
    /* iface   */ ane_reg_iface,
    /* context */ nullptr,
};
/** --------------------------------------------------------------------------------------------------------- Public API
 */
ggml_backend_reg_t ggml_backend_ane_reg(void) {
    return &ane_reg;
}
