/** --------------------------------------------------------------------------------------------------------- GGML ANE Backend
 * @file ggml-ane.h
 * @brief Public C API for the ANE (Apple Neural Engine / SME) backend for ggml/llama.cpp.
 *
 * This backend uses Apple Silicon's Scalable Matrix Extension (SME) hardware via the ane
 * bytecode interpreter for LLM inference operations: quantized GEMV, RMS normalization,
 * softmax, SiLU activation, RoPE, and element-wise ops.
 *
 * @author Josh Morgan (@joshmorgan1000 on GitHub)
 * Released under the MIT License
 */
#pragma once
#include "ggml-backend.h"
#ifdef __cplusplus
extern "C" {
#endif
/** --------------------------------------------------------------------------------------------- Registration
 * @brief Returns the ANE backend registry entry. Called by ggml's backend loader.
 */
ggml_backend_reg_t ggml_backend_ane_reg(void);
#ifdef __cplusplus
}
#endif
