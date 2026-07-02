/** --------------------------------------------------------------------------------------------------------- Psyne Parser Tests
 * @file test_psyne_parser.cpp
 * @brief Contract tests for the standalone Psyne parser and validation model.
 */
#include <ane/psyne_lang.hpp>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

using ane::psyne::access_mode;
using ane::psyne::compiler;
using ane::psyne::executable;
using ane::psyne::make_matmul_tile_f32_plan;
using ane::psyne::operation_kind;
using ane::psyne::stream_shape;
using ane::psyne::value_kind;
static constexpr uint32_t F32_VECTOR_ELEMENTS = 16;  ///< Apple M4/M5 f32 lanes per 64-byte z-vector
static constexpr uint32_t F16_VECTOR_ELEMENTS = 32;  ///< Half-precision lanes per 64-byte z-vector
/** --------------------------------------------------------------------------------------------------------- Aligned Allocation
 * @brief Allocates caller-ABI-compatible memory rounded up to a 64-byte boundary.
 */
static void* alloc64(size_t size) {
    return std::aligned_alloc(64, ((size + 63U) >> 6U) << 6U);
}
/** --------------------------------------------------------------------------------------------------------- Expect Error
 * @brief Returns true when compiling source throws and the message contains the expected text.
 */
static bool expect_error(const char* source, const char* expected_text) {
    compiler c;
    try {
        (void)c.compile(source);
    } catch (const std::exception& ex) {
        return std::string(ex.what()).find(expected_text) != std::string::npos;
    }
    return false;
}
/** --------------------------------------------------------------------------------------------------------- Check Close
 * @brief Verifies all elements are within a fixed absolute tolerance.
 */
static bool check_close(const char* name, const float* got, const float* expected, uint32_t count, float tolerance) {
    for (uint32_t i = 0; i < count; i++) {
        if (std::fabs(got[i] - expected[i]) > tolerance) {
            std::printf("    %s mismatch [%u]: got %.6f expected %.6f\n", name, i, got[i], expected[i]);
            return false;
        }
    }
    return true;
}
/** --------------------------------------------------------------------------------------------------------- BF16 Helpers
 * @brief Converts test values through the same truncated BF16 storage contract as the kernels.
 */
static uint16_t float_to_bf16(float value) {
    return static_cast<uint16_t>(std::bit_cast<uint32_t>(value) >> 16);
}
static float bf16_to_float(uint16_t value) {
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}
/** --------------------------------------------------------------------------------------------------------- Low-Bit Test Packing
 * @brief Packs logical i4/i2 panel indices for LUT-backed executable matmul tests.
 */
static void set_i4_index(uint8_t* panel, uint32_t index, uint8_t value) {
    const uint32_t byte_index = index >> 1U;
    if ((index & 1U) == 0U) {
        panel[byte_index] = static_cast<uint8_t>((panel[byte_index] & 0xF0U) | (value & 0x0FU));
    } else {
        panel[byte_index] = static_cast<uint8_t>((panel[byte_index] & 0x0FU) | ((value & 0x0FU) << 4U));
    }
}
static uint8_t get_i4_index(const uint8_t* panel, uint32_t index) {
    const uint8_t packed = panel[index >> 1U];
    if ((index & 1U) == 0U) {
        return packed & 0x0FU;
    }
    return (packed >> 4U) & 0x0FU;
}
static void set_i2_index(uint8_t* panel, uint32_t index, uint8_t value) {
    const uint32_t byte_index = index >> 2U;
    const uint32_t shift = (index & 3U) << 1U;
    const uint8_t mask = static_cast<uint8_t>(0x03U << shift);
    panel[byte_index] = static_cast<uint8_t>((panel[byte_index] & ~mask) | ((value & 0x03U) << shift));
}
static uint8_t get_i2_index(const uint8_t* panel, uint32_t index) {
    const uint32_t shift = (index & 3U) << 1U;
    return static_cast<uint8_t>((panel[index >> 2U] >> shift) & 0x03U);
}
/** --------------------------------------------------------------------------------------------------------- Declaration Contract
 * @brief Verifies scalar/stream declarations, range units, and caller-aligned physical bytes.
 */
static bool test_declarations() {
    compiler c;
    auto program = c.compile(R"(
        input: f32[27] ro;
        scale: f32 ro;
        scratch: f32[0b:16v];
        tile: f32[1t] wo;
        q4: f32[16] ro;
        packed i4[256] table q4;
    )");
    const auto* input = program.find_declaration("input");
    const auto* scale = program.find_declaration("scale");
    const auto* scratch = program.find_declaration("scratch");
    const auto* tile = program.find_declaration("tile");
    const auto* q4 = program.find_declaration("q4");
    const auto* packed = program.find_declaration("packed");
    if (input == nullptr || scale == nullptr || scratch == nullptr || tile == nullptr || q4 == nullptr || packed == nullptr) {
        return false;
    }
    if (input->element_count != 27 || input->logical_byte_count != 108 || input->physical_byte_count != 128) {
        return false;
    }
    if (scale->kind != value_kind::scalar || scale->logical_byte_count != 4 || scale->physical_byte_count != 64) {
        return false;
    }
    if (scratch->element_count != 256 || scratch->logical_byte_count != 1024 || scratch->shape != stream_shape::vectors) {
        return false;
    }
    if (tile->element_count != 256 || tile->access != access_mode::write_only || tile->shape != stream_shape::tiles) {
        return false;
    }
    if (q4->element_count != 16 || q4->logical_byte_count != 64 || q4->physical_byte_count != 64) {
        return false;
    }
    if (packed->element_count != 256 || packed->logical_byte_count != 128 || packed->physical_byte_count != 128 || packed->quantization_table != "q4") {
        return false;
    }
    return true;
}
/** --------------------------------------------------------------------------------------------------------- Element Operation Contract
 * @brief Verifies one operation per line and scalar broadcasting for stream ops.
 */
static bool test_element_operations() {
    compiler c;
    auto program = c.compile(R"(
        input: f32[27] ro;
        scale: f32 ro;
        output: f32[27] wo;
        output = input #* scale;
    )");
    if (program.operations().size() != 1) {
        return false;
    }
    const auto& op = program.operations()[0];
    if (op.kind != operation_kind::element_mul || op.output_element_count != 27) {
        return false;
    }
    auto elementwise_program = c.compile(R"(
        left: f32[16v] ro;
        right: f32[16v] ro;
        output: f32[1t] wo;
        output = left #* right;
    )");
    if (elementwise_program.operations()[0].kind != operation_kind::element_mul) {
        return false;
    }
    auto subtract_program = c.compile(R"(
        left: f32[16v] ro;
        right: f32[16v] ro;
        output: f32[16v] wo;
        output = left #- right;
    )");
    if (subtract_program.operations()[0].kind != operation_kind::element_sub) {
        return false;
    }
    auto divide_program = c.compile(R"(
        left: f32[16v] ro;
        right: f32[16v] ro;
        output: f32[16v] wo;
        output = left #/ right;
    )");
    if (divide_program.operations()[0].kind != operation_kind::element_div) {
        return false;
    }
    auto unary_program = c.compile(R"(
        input: f32[16v] ro;
        output: f32[16v] wo;
        output = exp input;
    )");
    if (unary_program.operations()[0].kind != operation_kind::element_exp) {
        return false;
    }
    auto sqrt_program = c.compile(R"(
        input: f32[16v] ro;
        output: f32[16v] wo;
        output = sqrt input;
    )");
    if (sqrt_program.operations()[0].kind != operation_kind::element_sqrt) {
        return false;
    }
    auto pow_program = c.compile(R"(
        input: f32[16v] ro;
        exponent: f32 ro;
        output: f32[16v] wo;
        output = pow input exponent;
    )");
    if (pow_program.operations()[0].kind != operation_kind::element_pow) {
        return false;
    }
    return expect_error(R"(
        input: f32[16] ro;
        scale: f32 ro;
        output: f32[16] wo;
        output = input #* scale #+ input;
    )", "Only one operation");
}
/** --------------------------------------------------------------------------------------------------------- Reduction Contract
 * @brief Verifies reduction target size uses ceil(source elements / source vector elements).
 */
static bool test_reductions() {
    compiler c;
    auto program = c.compile(R"(
        something: f32[250] ro;
        something_else: f32[16] wo;
        something_else <+ something;
    )");
    if (program.operations().size() != 1) {
        return false;
    }
    const auto& op = program.operations()[0];
    if (op.kind != operation_kind::reduce_add || op.output_element_count != 16) {
        return false;
    }
    auto max_program = c.compile(R"(
        something: f32[16v] ro;
        partial_max: f32[16] wo;
        partial_max <max something;
    )");
    if (max_program.operations().size() != 1 || max_program.operations()[0].kind != operation_kind::reduce_max) {
        return false;
    }
    return expect_error(R"(
        something f32[256] ro;
        something_else f32[8] wo;
        something_else <+ something;
    )", "Not enough elements in target for reduction");
}
/** --------------------------------------------------------------------------------------------------------- Access Contract
 * @brief Verifies read-only and write-only declaration errors.
 */
static bool test_access_contract() {
    const bool write_error = expect_error(R"(
        input: f32[16] ro;
        scale: f32 ro;
        input = input #* scale;
    )", "Cannot write to read-only");
    const bool read_error = expect_error(R"(
        input: f32[16] wo;
        scale: f32 ro;
        output: f32[16] wo;
        output = input #* scale;
    )", "Cannot read from write-only");
    return write_error && read_error;
}
/** --------------------------------------------------------------------------------------------------------- Executable Stream Multiply
 * @brief Verifies Psyne lowers and executes one f32 vector multiply through the bytecode interpreter.
 */
static bool test_executable_stream_multiply() {
    auto* input = static_cast<float*>(alloc64(F32_VECTOR_ELEMENTS * sizeof(float)));
    auto* weight = static_cast<float*>(alloc64(F32_VECTOR_ELEMENTS * sizeof(float)));
    auto* output = static_cast<float*>(alloc64(F32_VECTOR_ELEMENTS * sizeof(float)));
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        input[i] = static_cast<float>(i + 1U);
        weight[i] = static_cast<float>(2U * i + 1U);
        output[i] = 0.0f;
    }
    executable program(R"(
        input: f32[16] ro;
        weight: f32[16] ro;
        output: f32[16] wo;
        output = input #* weight;
    )");
    program.exec({input, weight, output});
    bool ok = true;
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        const float expected = input[i] * weight[i];
        if (std::fabs(output[i] - expected) > 0.001f) {
            ok = false;
            break;
        }
    }
    std::free(input);
    std::free(weight);
    std::free(output);
    return ok;
}
/** --------------------------------------------------------------------------------------------------------- Executable Element Primitives
 * @brief Verifies elementwise subtract, scalar broadcast, exp, log, and pow execution.
 */
static bool test_executable_element_primitives() {
    auto* input = static_cast<float*>(alloc64(F32_VECTOR_ELEMENTS * sizeof(float)));
    auto* weight = static_cast<float*>(alloc64(F32_VECTOR_ELEMENTS * sizeof(float)));
    auto* signed_input = static_cast<float*>(alloc64(F32_VECTOR_ELEMENTS * sizeof(float)));
    auto* output = static_cast<float*>(alloc64(F32_VECTOR_ELEMENTS * sizeof(float)));
    auto* expected = static_cast<float*>(alloc64(F32_VECTOR_ELEMENTS * sizeof(float)));
    auto* scalar = static_cast<float*>(alloc64(sizeof(float)));
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        input[i] = 0.5f + 0.05f * static_cast<float>(i);
        weight[i] = 1.25f + 0.025f * static_cast<float>(i);
        signed_input[i] = 0.20f * static_cast<float>(i) - 1.5f;
        output[i] = 0.0f;
        expected[i] = 0.0f;
    }
    scalar[0] = 1.75f;
    executable sub_program(R"(
        input: f32[16] ro;
        weight: f32[16] ro;
        output: f32[16] wo;
        output = input #- weight;
    )");
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        expected[i] = input[i] - weight[i];
    }
    sub_program.exec({input, weight, output});
    bool ok = check_close("sub", output, expected, F32_VECTOR_ELEMENTS, 0.001f);
    executable scalar_add_program(R"(
        input: f32[16] ro;
        scalar: f32 ro;
        output: f32[16] wo;
        output = input #+ scalar;
    )");
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        output[i] = 0.0f;
        expected[i] = input[i] + scalar[0];
    }
    scalar_add_program.exec({input, scalar, output});
    ok = ok && check_close("scalar_add", output, expected, F32_VECTOR_ELEMENTS, 0.001f);
    executable scalar_mul_program(R"(
        input: f32[16] ro;
        scalar: f32 ro;
        output: f32[16] wo;
        output = input #* scalar;
    )");
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        output[i] = 0.0f;
        expected[i] = input[i] * scalar[0];
    }
    scalar_mul_program.exec({input, scalar, output});
    ok = ok && check_close("scalar_mul", output, expected, F32_VECTOR_ELEMENTS, 0.001f);
    executable scalar_lhs_sub_program(R"(
        scalar: f32 ro;
        input: f32[16] ro;
        output: f32[16] wo;
        output = scalar #- input;
    )");
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        output[i] = 0.0f;
        expected[i] = scalar[0] - input[i];
    }
    scalar_lhs_sub_program.exec({scalar, input, output});
    ok = ok && check_close("scalar_lhs_sub", output, expected, F32_VECTOR_ELEMENTS, 0.001f);
    executable div_program(R"(
        input: f32[16] ro;
        scalar: f32 ro;
        output: f32[16] wo;
        output = input #/ scalar;
    )");
    scalar[0] = 2.0f;
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        output[i] = 0.0f;
        expected[i] = input[i] / scalar[0];
    }
    div_program.exec({input, scalar, output});
    ok = ok && check_close("scalar_div", output, expected, F32_VECTOR_ELEMENTS, 0.001f);
    executable exp_program(R"(
        input: f32[16] ro;
        output: f32[16] wo;
        output = exp input;
    )");
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        output[i] = 0.0f;
        expected[i] = std::exp(input[i]);
    }
    exp_program.exec({input, output});
    ok = ok && check_close("exp", output, expected, F32_VECTOR_ELEMENTS, 0.02f);
    executable log_program(R"(
        input: f32[16] ro;
        output: f32[16] wo;
        output = log input;
    )");
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        output[i] = 0.0f;
        expected[i] = std::log(input[i]);
    }
    log_program.exec({input, output});
    ok = ok && check_close("log", output, expected, F32_VECTOR_ELEMENTS, 0.02f);
    scalar[0] = 1.5f;
    executable pow_program(R"(
        input: f32[16] ro;
        scalar: f32 ro;
        output: f32[16] wo;
        output = pow input scalar;
    )");
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        output[i] = 0.0f;
        expected[i] = std::pow(input[i], scalar[0]);
    }
    pow_program.exec({input, scalar, output});
    ok = ok && check_close("pow", output, expected, F32_VECTOR_ELEMENTS, 0.05f);
    executable abs_program(R"(
        input: f32[16] ro;
        output: f32[16] wo;
        output = abs input;
    )");
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        output[i] = 0.0f;
        expected[i] = std::fabs(signed_input[i]);
    }
    abs_program.exec({signed_input, output});
    ok = ok && check_close("abs", output, expected, F32_VECTOR_ELEMENTS, 0.001f);
    executable sqrt_program(R"(
        input: f32[16] ro;
        output: f32[16] wo;
        output = sqrt input;
    )");
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        output[i] = 0.0f;
        expected[i] = std::sqrt(input[i]);
    }
    sqrt_program.exec({input, output});
    ok = ok && check_close("sqrt", output, expected, F32_VECTOR_ELEMENTS, 0.001f);
    executable rsqrt_program(R"(
        input: f32[16] ro;
        output: f32[16] wo;
        output = rsqrt input;
    )");
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        output[i] = 0.0f;
        expected[i] = 1.0f / std::sqrt(input[i]);
    }
    rsqrt_program.exec({input, output});
    ok = ok && check_close("rsqrt", output, expected, F32_VECTOR_ELEMENTS, 0.001f);
    executable square_program(R"(
        input: f32[16] ro;
        output: f32[16] wo;
        output = square input;
    )");
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        output[i] = 0.0f;
        expected[i] = signed_input[i] * signed_input[i];
    }
    square_program.exec({signed_input, output});
    ok = ok && check_close("square", output, expected, F32_VECTOR_ELEMENTS, 0.001f);
    executable relu_program(R"(
        input: f32[16] ro;
        output: f32[16] wo;
        output = relu input;
    )");
    for (uint32_t i = 0; i < F32_VECTOR_ELEMENTS; i++) {
        output[i] = 0.0f;
        expected[i] = signed_input[i] > 0.0f ? signed_input[i] : 0.0f;
    }
    relu_program.exec({signed_input, output});
    ok = ok && check_close("relu", output, expected, F32_VECTOR_ELEMENTS, 0.001f);
    std::free(input);
    std::free(weight);
    std::free(signed_input);
    std::free(output);
    std::free(expected);
    std::free(scalar);
    return ok;
}
/** --------------------------------------------------------------------------------------------------------- Executable Stream Reductions
 * @brief Verifies scheduled sum and max reductions over f32 stream chunks and final scalar chunks.
 */
static bool test_executable_stream_reductions() {
    constexpr uint32_t ChunkCount = F32_VECTOR_ELEMENTS;
    constexpr uint32_t ElementCount = ChunkCount * F32_VECTOR_ELEMENTS;
    auto* input = static_cast<float*>(alloc64(ElementCount * sizeof(float)));
    auto* partial_sum = static_cast<float*>(alloc64(ChunkCount * sizeof(float)));
    auto* partial_max = static_cast<float*>(alloc64(ChunkCount * sizeof(float)));
    auto* expected_sum = static_cast<float*>(alloc64(ChunkCount * sizeof(float)));
    auto* expected_max = static_cast<float*>(alloc64(ChunkCount * sizeof(float)));
    auto* total_sum = static_cast<float*>(alloc64(sizeof(float)));
    auto* total_max = static_cast<float*>(alloc64(sizeof(float)));
    auto* log_total_sum = static_cast<float*>(alloc64(sizeof(float)));
    for (uint32_t chunk = 0; chunk < ChunkCount; chunk++) {
        expected_sum[chunk] = 0.0f;
        expected_max[chunk] = -1000.0f;
        partial_sum[chunk] = 0.0f;
        partial_max[chunk] = 0.0f;
        for (uint32_t lane = 0; lane < F32_VECTOR_ELEMENTS; lane++) {
            const uint32_t idx = chunk * F32_VECTOR_ELEMENTS + lane;
            input[idx] = 0.125f * static_cast<float>((idx % 29U) + 1U) - 1.5f;
            expected_sum[chunk] += input[idx];
            if (input[idx] > expected_max[chunk]) {
                expected_max[chunk] = input[idx];
            }
        }
    }
    executable sum_program(R"(
        input: f32[16v] ro;
        partial_sum: f32[16] wo;
        partial_sum <+ input;
    )");
    executable max_program(R"(
        input: f32[16v] ro;
        partial_max: f32[16] wo;
        partial_max <max input;
    )");
    executable total_sum_program(R"(
        partial_sum: f32[16] ro;
        total_sum: f32 wo;
        total_sum <+ partial_sum;
    )");
    executable total_max_program(R"(
        partial_max: f32[16] ro;
        total_max: f32 wo;
        total_max <max partial_max;
    )");
    executable log_total_program(R"(
        total_sum: f32 ro;
        log_total_sum: f32 wo;
        log_total_sum = log total_sum;
    )");
    sum_program.exec({input, partial_sum});
    max_program.exec({input, partial_max});
    float expected_total_sum = 0.0f;
    float expected_total_max = -1000.0f;
    for (uint32_t chunk = 0; chunk < ChunkCount; chunk++) {
        expected_total_sum += expected_sum[chunk];
        if (expected_max[chunk] > expected_total_max) {
            expected_total_max = expected_max[chunk];
        }
    }
    total_sum[0] = 0.0f;
    total_max[0] = 0.0f;
    total_sum_program.exec({partial_sum, total_sum});
    total_max_program.exec({partial_max, total_max});
    log_total_sum[0] = 0.0f;
    log_total_program.exec({total_sum, log_total_sum});
    bool ok = check_close("partial_sum", partial_sum, expected_sum, ChunkCount, 0.001f);
    ok = ok && check_close("partial_max", partial_max, expected_max, ChunkCount, 0.001f);
    ok = ok && (std::fabs(total_sum[0] - expected_total_sum) < 0.001f);
    ok = ok && (std::fabs(total_max[0] - expected_total_max) < 0.001f);
    ok = ok && (std::fabs(log_total_sum[0] - std::log(expected_total_sum)) < 0.02f);
    std::free(input);
    std::free(partial_sum);
    std::free(partial_max);
    std::free(expected_sum);
    std::free(expected_max);
    std::free(total_sum);
    std::free(total_max);
    std::free(log_total_sum);
    return ok;
}
/** --------------------------------------------------------------------------------------------------------- Matmul Accumulation Contract
 * @brief Verifies plain stream multiply into a tile is a DSL matmul accumulation op.
 */
static bool test_matmul_accumulation_contract() {
    compiler c;
    auto program = c.compile(R"(
        a_panel: f32[16v] ro;
        b_panel: f32[16v] ro;
        partial_tile: f32[1t] wo;
        partial_tile = a_panel * b_panel;
    )");
    if (program.operations().size() != 1) {
        return false;
    }
    const auto& op = program.operations()[0];
    if (op.kind != operation_kind::matmul_accumulate || op.output_element_count != 256) {
        return false;
    }
    auto f16_program = c.compile(R"(
        a_panel: f16[8v] ro;
        b_panel: f16[8v] ro;
        partial_tile: f32[1t] wo;
        partial_tile = a_panel * b_panel;
    )");
    if (f16_program.operations()[0].kind != operation_kind::matmul_accumulate) {
        return false;
    }
    auto bf16_program = c.compile(R"(
        a_panel: bf16[8v] ro;
        b_panel: bf16[8v] ro;
        partial_tile: f32[1t] wo;
        partial_tile = a_panel * b_panel;
    )");
    if (bf16_program.operations()[0].kind != operation_kind::matmul_accumulate) {
        return false;
    }
    auto quantized_program = c.compile(R"(
        q4: f32[16] ro;
        a_panel: i4[256] table q4 ro;
        b_panel: i4[256] table q4 ro;
        partial_tile: f32[1t] wo;
        partial_tile = a_panel * b_panel;
    )");
    if (quantized_program.operations().size() != 1) {
        return false;
    }
    if (quantized_program.operations()[0].kind != operation_kind::quantized_matmul_accumulate) {
        return false;
    }
    auto quantized_i2_program = c.compile(R"(
        q2: f32[4] ro;
        a_panel: i2[256] table q2 ro;
        b_panel: i2[256] table q2 ro;
        partial_tile: f32[1t] wo;
        partial_tile = a_panel * b_panel;
    )");
    if (quantized_i2_program.operations()[0].kind != operation_kind::quantized_matmul_accumulate) {
        return false;
    }
    return expect_error(R"(
        a_panel: f32[16v] ro;
        b_panel: f32[8v] ro;
        partial_tile: f32[1t] wo;
        partial_tile = a_panel * b_panel;
    )", "Matrix multiply requires vector-panel sources and tile target");
}
/** --------------------------------------------------------------------------------------------------------- Executable Partial Tile Matmul
 * @brief Verifies PSL matmul accumulation lowers to one partial-tile bytecode and executes correctly.
 */
static bool test_executable_partial_tile_matmul() {
    constexpr uint32_t PanelElements = F32_VECTOR_ELEMENTS * F32_VECTOR_ELEMENTS;
    auto* a_panel = static_cast<float*>(alloc64(PanelElements * sizeof(float)));
    auto* b_panel = static_cast<float*>(alloc64(PanelElements * sizeof(float)));
    auto* partial_tile = static_cast<float*>(alloc64(PanelElements * sizeof(float)));
    auto* reference = static_cast<float*>(alloc64(PanelElements * sizeof(float)));
    for (uint32_t k = 0; k < F32_VECTOR_ELEMENTS; k++) {
        for (uint32_t lane = 0; lane < F32_VECTOR_ELEMENTS; lane++) {
            a_panel[k * F32_VECTOR_ELEMENTS + lane] = 0.01f * static_cast<float>((k + 1U) * (lane + 1U));
            b_panel[k * F32_VECTOR_ELEMENTS + lane] = 0.02f * static_cast<float>((k + 3U) + lane);
        }
    }
    for (uint32_t row = 0; row < F32_VECTOR_ELEMENTS; row++) {
        for (uint32_t col = 0; col < F32_VECTOR_ELEMENTS; col++) {
            double sum = 0.0;
            for (uint32_t k = 0; k < F32_VECTOR_ELEMENTS; k++) {
                sum += static_cast<double>(a_panel[k * F32_VECTOR_ELEMENTS + row])
                    * static_cast<double>(b_panel[k * F32_VECTOR_ELEMENTS + col]);
            }
            partial_tile[row * F32_VECTOR_ELEMENTS + col] = 0.0f;
            reference[row * F32_VECTOR_ELEMENTS + col] = static_cast<float>(sum);
        }
    }
    executable program(R"(
        a_panel: f32[16v] ro;
        b_panel: f32[16v] ro;
        partial_tile: f32[1t] wo;
        partial_tile = a_panel * b_panel;
    )");
    program.exec({a_panel, b_panel, partial_tile});
    bool ok = true;
    for (uint32_t i = 0; i < PanelElements; i++) {
        if (std::fabs(partial_tile[i] - reference[i]) > 0.001f) {
            ok = false;
            break;
        }
    }
    std::free(a_panel);
    std::free(b_panel);
    std::free(partial_tile);
    std::free(reference);
    return ok;
}
/** --------------------------------------------------------------------------------------------------------- Executable Stream Conversions
 * @brief Verifies assignment lowers to f32/f16 and f32/bf16 stream conversion opcodes.
 */
static bool test_executable_stream_conversions() {
    constexpr uint32_t ElementCount = F16_VECTOR_ELEMENTS;
    auto* input = static_cast<float*>(alloc64(ElementCount * sizeof(float)));
    auto* output = static_cast<float*>(alloc64(ElementCount * sizeof(float)));
    auto* half = static_cast<__fp16*>(alloc64(ElementCount * sizeof(__fp16)));
    auto* bfloat = static_cast<uint16_t*>(alloc64(ElementCount * sizeof(uint16_t)));
    for (uint32_t i = 0; i < ElementCount; i++) {
        input[i] = (static_cast<float>(static_cast<int32_t>(i)) - 16.0f) * 0.25f;
        output[i] = 0.0f;
        half[i] = static_cast<__fp16>(0.0f);
        bfloat[i] = 0;
    }
    executable f16_program(R"(
        input: f32[32] ro;
        half: f16[32];
        output: f32[32] wo;
        half = input;
        output = half;
    )");
    if (f16_program.ir().operations().size() != 2 || f16_program.ir().operations()[0].kind != operation_kind::convert || f16_program.ir().operations()[1].kind != operation_kind::convert) {
        std::free(input);
        std::free(output);
        std::free(half);
        std::free(bfloat);
        return false;
    }
    f16_program.exec({input, half, output});
    if (!check_close("f32-f16-f32 conversion", output, input, ElementCount, 0.001f)) {
        std::free(input);
        std::free(output);
        std::free(half);
        std::free(bfloat);
        return false;
    }
    for (uint32_t i = 0; i < ElementCount; i++) {
        output[i] = 0.0f;
        bfloat[i] = 0;
    }
    executable bf16_program(R"(
        input: f32[32] ro;
        bfloat: bf16[32];
        output: f32[32] wo;
        bfloat = input;
        output = bfloat;
    )");
    bf16_program.exec({input, bfloat, output});
    const bool ok = check_close("f32-bf16-f32 conversion", output, input, ElementCount, 0.001f);
    std::free(input);
    std::free(output);
    std::free(half);
    std::free(bfloat);
    return ok;
}
/** --------------------------------------------------------------------------------------------------------- Executable I4 LUT Partial Tile Matmul
 * @brief Verifies i4 panels lower to one fused LUT-decode plus FMOPA partial tile opcode.
 */
static bool test_executable_i4_lut_partial_tile_matmul() {
    constexpr uint32_t TileElements = F32_VECTOR_ELEMENTS * F32_VECTOR_ELEMENTS;
    constexpr uint32_t PackedBytes = TileElements / 2U;
    auto* q4_a = static_cast<float*>(alloc64(16U * sizeof(float)));
    auto* q4_b = static_cast<float*>(alloc64(16U * sizeof(float)));
    auto* a_panel = static_cast<uint8_t*>(alloc64(PackedBytes));
    auto* b_panel = static_cast<uint8_t*>(alloc64(PackedBytes));
    auto* partial_tile = static_cast<float*>(alloc64(TileElements * sizeof(float)));
    auto* reference = static_cast<float*>(alloc64(TileElements * sizeof(float)));
    for (uint32_t i = 0; i < 16U; i++) {
        q4_a[i] = (static_cast<float>(i) - 7.0f) * 0.125f;
        q4_b[i] = (static_cast<float>(i) - 5.0f) * 0.0625f;
    }
    for (uint32_t i = 0; i < PackedBytes; i++) {
        a_panel[i] = 0;
        b_panel[i] = 0;
    }
    for (uint32_t k = 0; k < F32_VECTOR_ELEMENTS; k++) {
        for (uint32_t lane = 0; lane < F32_VECTOR_ELEMENTS; lane++) {
            set_i4_index(a_panel, k * F32_VECTOR_ELEMENTS + lane, static_cast<uint8_t>((k + lane * 3U) & 15U));
            set_i4_index(b_panel, k * F32_VECTOR_ELEMENTS + lane, static_cast<uint8_t>((k * 5U + lane) & 15U));
        }
    }
    for (uint32_t row = 0; row < F32_VECTOR_ELEMENTS; row++) {
        for (uint32_t col = 0; col < F32_VECTOR_ELEMENTS; col++) {
            double sum = 0.0;
            for (uint32_t k = 0; k < F32_VECTOR_ELEMENTS; k++) {
                const uint8_t a_index = get_i4_index(a_panel, k * F32_VECTOR_ELEMENTS + row);
                const uint8_t b_index = get_i4_index(b_panel, k * F32_VECTOR_ELEMENTS + col);
                sum += static_cast<double>(q4_a[a_index]) * static_cast<double>(q4_b[b_index]);
            }
            partial_tile[row * F32_VECTOR_ELEMENTS + col] = 0.0f;
            reference[row * F32_VECTOR_ELEMENTS + col] = static_cast<float>(sum);
        }
    }
    executable program(R"(
        q4_a: f32[16] ro;
        q4_b: f32[16] ro;
        a_panel: i4[256] table q4_a ro;
        b_panel: i4[256] table q4_b ro;
        partial_tile: f32[1t] wo;
        partial_tile = a_panel * b_panel;
    )");
    program.exec({q4_a, q4_b, a_panel, b_panel, partial_tile});
    const bool ok = check_close("i4 LUT partial tile", partial_tile, reference, TileElements, 0.001f);
    std::free(q4_a);
    std::free(q4_b);
    std::free(a_panel);
    std::free(b_panel);
    std::free(partial_tile);
    std::free(reference);
    return ok;
}
/** --------------------------------------------------------------------------------------------------------- Executable I2 LUT Partial Tile Matmul
 * @brief Verifies i2 panels lower to one fused LUT-decode plus FMOPA partial tile opcode.
 */
static bool test_executable_i2_lut_partial_tile_matmul() {
    constexpr uint32_t TileElements = F32_VECTOR_ELEMENTS * F32_VECTOR_ELEMENTS;
    constexpr uint32_t PackedBytes = TileElements / 4U;
    auto* q2_a = static_cast<float*>(alloc64(64U));
    auto* q2_b = static_cast<float*>(alloc64(64U));
    auto* a_panel = static_cast<uint8_t*>(alloc64(PackedBytes));
    auto* b_panel = static_cast<uint8_t*>(alloc64(PackedBytes));
    auto* partial_tile = static_cast<float*>(alloc64(TileElements * sizeof(float)));
    auto* reference = static_cast<float*>(alloc64(TileElements * sizeof(float)));
    for (uint32_t i = 0; i < 16U; i++) {
        q2_a[i] = 0.0f;
        q2_b[i] = 0.0f;
    }
    for (uint32_t i = 0; i < 4U; i++) {
        q2_a[i] = (static_cast<float>(i) - 1.0f) * 0.25f;
        q2_b[i] = (static_cast<float>(i) - 2.0f) * 0.5f;
    }
    for (uint32_t i = 0; i < PackedBytes; i++) {
        a_panel[i] = 0;
        b_panel[i] = 0;
    }
    for (uint32_t k = 0; k < F32_VECTOR_ELEMENTS; k++) {
        for (uint32_t lane = 0; lane < F32_VECTOR_ELEMENTS; lane++) {
            set_i2_index(a_panel, k * F32_VECTOR_ELEMENTS + lane, static_cast<uint8_t>((k + lane) & 3U));
            set_i2_index(b_panel, k * F32_VECTOR_ELEMENTS + lane, static_cast<uint8_t>((k * 3U + lane * 2U) & 3U));
        }
    }
    for (uint32_t row = 0; row < F32_VECTOR_ELEMENTS; row++) {
        for (uint32_t col = 0; col < F32_VECTOR_ELEMENTS; col++) {
            double sum = 0.0;
            for (uint32_t k = 0; k < F32_VECTOR_ELEMENTS; k++) {
                const uint8_t a_index = get_i2_index(a_panel, k * F32_VECTOR_ELEMENTS + row);
                const uint8_t b_index = get_i2_index(b_panel, k * F32_VECTOR_ELEMENTS + col);
                sum += static_cast<double>(q2_a[a_index]) * static_cast<double>(q2_b[b_index]);
            }
            partial_tile[row * F32_VECTOR_ELEMENTS + col] = 0.0f;
            reference[row * F32_VECTOR_ELEMENTS + col] = static_cast<float>(sum);
        }
    }
    executable program(R"(
        q2_a: f32[4] ro;
        q2_b: f32[4] ro;
        a_panel: i2[256] table q2_a ro;
        b_panel: i2[256] table q2_b ro;
        partial_tile: f32[1t] wo;
        partial_tile = a_panel * b_panel;
    )");
    program.exec({q2_a, q2_b, a_panel, b_panel, partial_tile});
    const bool ok = check_close("i2 LUT partial tile", partial_tile, reference, TileElements, 0.001f);
    std::free(q2_a);
    std::free(q2_b);
    std::free(a_panel);
    std::free(b_panel);
    std::free(partial_tile);
    std::free(reference);
    return ok;
}
/** --------------------------------------------------------------------------------------------------------- Executable F16 Element Primitives
 * @brief Verifies f16 stream add/sub/mul lower to f16 z-register bytecode.
 */
static bool test_executable_f16_element_primitives() {
    auto* left = static_cast<__fp16*>(alloc64(F16_VECTOR_ELEMENTS * sizeof(__fp16)));
    auto* right = static_cast<__fp16*>(alloc64(F16_VECTOR_ELEMENTS * sizeof(__fp16)));
    auto* output = static_cast<__fp16*>(alloc64(F16_VECTOR_ELEMENTS * sizeof(__fp16)));
    auto* scalar = static_cast<__fp16*>(alloc64(sizeof(__fp16)));
    for (uint32_t i = 0; i < F16_VECTOR_ELEMENTS; i++) {
        left[i] = static_cast<__fp16>(0.25f + 0.01f * static_cast<float>(i));
        right[i] = static_cast<__fp16>(0.5f - 0.005f * static_cast<float>(i));
        output[i] = static_cast<__fp16>(0.0f);
    }
    scalar[0] = static_cast<__fp16>(0.375f);
    executable add_program(R"(
        left: f16[1v] ro;
        right: f16[1v] ro;
        output: f16[1v] wo;
        output = left #+ right;
    )");
    add_program.exec({left, right, output});
    for (uint32_t i = 0; i < F16_VECTOR_ELEMENTS; i++) {
        const float expected = static_cast<float>(left[i]) + static_cast<float>(right[i]);
        if (std::fabs(static_cast<float>(output[i]) - expected) > 0.002f) {
            std::free(left);
            std::free(right);
            std::free(output);
            std::free(scalar);
            return false;
        }
    }
    executable sub_program(R"(
        left: f16[1v] ro;
        right: f16[1v] ro;
        output: f16[1v] wo;
        output = left #- right;
    )");
    sub_program.exec({left, right, output});
    for (uint32_t i = 0; i < F16_VECTOR_ELEMENTS; i++) {
        const float expected = static_cast<float>(left[i]) - static_cast<float>(right[i]);
        if (std::fabs(static_cast<float>(output[i]) - expected) > 0.002f) {
            std::free(left);
            std::free(right);
            std::free(output);
            std::free(scalar);
            return false;
        }
    }
    executable mul_program(R"(
        left: f16[1v] ro;
        right: f16[1v] ro;
        output: f16[1v] wo;
        output = left #* right;
    )");
    mul_program.exec({left, right, output});
    for (uint32_t i = 0; i < F16_VECTOR_ELEMENTS; i++) {
        const float expected = static_cast<float>(left[i]) * static_cast<float>(right[i]);
        if (std::fabs(static_cast<float>(output[i]) - expected) > 0.002f) {
            std::free(left);
            std::free(right);
            std::free(output);
            std::free(scalar);
            return false;
        }
    }
    executable scalar_add_program(R"(
        left: f16[1v] ro;
        scalar: f16 ro;
        output: f16[1v] wo;
        output = left #+ scalar;
    )");
    scalar_add_program.exec({left, scalar, output});
    for (uint32_t i = 0; i < F16_VECTOR_ELEMENTS; i++) {
        const float expected = static_cast<float>(left[i]) + static_cast<float>(scalar[0]);
        if (std::fabs(static_cast<float>(output[i]) - expected) > 0.002f) {
            std::free(left);
            std::free(right);
            std::free(output);
            std::free(scalar);
            return false;
        }
    }
    executable scalar_mul_program(R"(
        left: f16[1v] ro;
        scalar: f16 ro;
        output: f16[1v] wo;
        output = left #* scalar;
    )");
    scalar_mul_program.exec({left, scalar, output});
    for (uint32_t i = 0; i < F16_VECTOR_ELEMENTS; i++) {
        const float expected = static_cast<float>(left[i]) * static_cast<float>(scalar[0]);
        if (std::fabs(static_cast<float>(output[i]) - expected) > 0.002f) {
            std::free(left);
            std::free(right);
            std::free(output);
            std::free(scalar);
            return false;
        }
    }
    executable scalar_lhs_sub_program(R"(
        scalar: f16 ro;
        left: f16[1v] ro;
        output: f16[1v] wo;
        output = scalar #- left;
    )");
    scalar_lhs_sub_program.exec({scalar, left, output});
    for (uint32_t i = 0; i < F16_VECTOR_ELEMENTS; i++) {
        const float expected = static_cast<float>(scalar[0]) - static_cast<float>(left[i]);
        if (std::fabs(static_cast<float>(output[i]) - expected) > 0.002f) {
            std::free(left);
            std::free(right);
            std::free(output);
            std::free(scalar);
            return false;
        }
    }
    std::free(left);
    std::free(right);
    std::free(output);
    std::free(scalar);
    return true;
}
/** --------------------------------------------------------------------------------------------------------- Executable F16 Partial Tile Matmul
 * @brief Verifies f16 packed panels accumulate into a f32 tile through widening FMOPA.
 */
static bool test_executable_f16_partial_tile_matmul() {
    constexpr uint32_t TileElements = F32_VECTOR_ELEMENTS * F32_VECTOR_ELEMENTS;
    constexpr uint32_t PairCount = F32_VECTOR_ELEMENTS / 2U;
    auto* a_panel = static_cast<__fp16*>(alloc64(TileElements * sizeof(__fp16)));
    auto* b_panel = static_cast<__fp16*>(alloc64(TileElements * sizeof(__fp16)));
    auto* partial_tile = static_cast<float*>(alloc64(TileElements * sizeof(float)));
    auto* reference = static_cast<float*>(alloc64(TileElements * sizeof(float)));
    for (uint32_t pair = 0; pair < PairCount; pair++) {
        for (uint32_t lane = 0; lane < 2; lane++) {
            const uint32_t k = pair * 2U + lane;
            for (uint32_t row = 0; row < F32_VECTOR_ELEMENTS; row++) {
                a_panel[pair * F16_VECTOR_ELEMENTS + row * 2U + lane] = static_cast<__fp16>(0.01f * static_cast<float>((row + 1U) * (k + 1U)));
            }
            for (uint32_t col = 0; col < F32_VECTOR_ELEMENTS; col++) {
                b_panel[pair * F16_VECTOR_ELEMENTS + col * 2U + lane] = static_cast<__fp16>(0.02f * static_cast<float>((col + 2U) + k));
            }
        }
    }
    for (uint32_t row = 0; row < F32_VECTOR_ELEMENTS; row++) {
        for (uint32_t col = 0; col < F32_VECTOR_ELEMENTS; col++) {
            double sum = 0.0;
            for (uint32_t pair = 0; pair < PairCount; pair++) {
                for (uint32_t lane = 0; lane < 2; lane++) {
                    sum += static_cast<double>(static_cast<float>(a_panel[pair * F16_VECTOR_ELEMENTS + row * 2U + lane]))
                        * static_cast<double>(static_cast<float>(b_panel[pair * F16_VECTOR_ELEMENTS + col * 2U + lane]));
                }
            }
            partial_tile[row * F32_VECTOR_ELEMENTS + col] = 0.0f;
            reference[row * F32_VECTOR_ELEMENTS + col] = static_cast<float>(sum);
        }
    }
    executable program(R"(
        a_panel: f16[8v] ro;
        b_panel: f16[8v] ro;
        partial_tile: f32[1t] wo;
        partial_tile = a_panel * b_panel;
    )");
    program.exec({a_panel, b_panel, partial_tile});
    const bool ok = check_close("f16 partial tile", partial_tile, reference, TileElements, 0.02f);
    std::free(a_panel);
    std::free(b_panel);
    std::free(partial_tile);
    std::free(reference);
    return ok;
}
/** --------------------------------------------------------------------------------------------------------- Executable BF16 Partial Tile Matmul
 * @brief Verifies bf16 packed panels accumulate into a f32 tile through widening BFMOPA.
 */
static bool test_executable_bf16_partial_tile_matmul() {
    constexpr uint32_t TileElements = F32_VECTOR_ELEMENTS * F32_VECTOR_ELEMENTS;
    constexpr uint32_t PairCount = F32_VECTOR_ELEMENTS / 2U;
    auto* a_panel = static_cast<uint16_t*>(alloc64(TileElements * sizeof(uint16_t)));
    auto* b_panel = static_cast<uint16_t*>(alloc64(TileElements * sizeof(uint16_t)));
    auto* partial_tile = static_cast<float*>(alloc64(TileElements * sizeof(float)));
    auto* reference = static_cast<float*>(alloc64(TileElements * sizeof(float)));
    for (uint32_t pair = 0; pair < PairCount; pair++) {
        for (uint32_t lane = 0; lane < 2; lane++) {
            const uint32_t k = pair * 2U + lane;
            for (uint32_t row = 0; row < F32_VECTOR_ELEMENTS; row++) {
                a_panel[pair * F16_VECTOR_ELEMENTS + row * 2U + lane] = float_to_bf16(0.01f * static_cast<float>((row + 3U) * (k + 1U)));
            }
            for (uint32_t col = 0; col < F32_VECTOR_ELEMENTS; col++) {
                b_panel[pair * F16_VECTOR_ELEMENTS + col * 2U + lane] = float_to_bf16(0.015f * static_cast<float>((col + 1U) + k));
            }
        }
    }
    for (uint32_t row = 0; row < F32_VECTOR_ELEMENTS; row++) {
        for (uint32_t col = 0; col < F32_VECTOR_ELEMENTS; col++) {
            double sum = 0.0;
            for (uint32_t pair = 0; pair < PairCount; pair++) {
                for (uint32_t lane = 0; lane < 2; lane++) {
                    sum += static_cast<double>(bf16_to_float(a_panel[pair * F16_VECTOR_ELEMENTS + row * 2U + lane]))
                        * static_cast<double>(bf16_to_float(b_panel[pair * F16_VECTOR_ELEMENTS + col * 2U + lane]));
                }
            }
            partial_tile[row * F32_VECTOR_ELEMENTS + col] = 0.0f;
            reference[row * F32_VECTOR_ELEMENTS + col] = static_cast<float>(sum);
        }
    }
    executable program(R"(
        a_panel: bf16[8v] ro;
        b_panel: bf16[8v] ro;
        partial_tile: f32[1t] wo;
        partial_tile = a_panel * b_panel;
    )");
    program.exec({a_panel, b_panel, partial_tile});
    const bool ok = check_close("bf16 partial tile", partial_tile, reference, TileElements, 0.05f);
    std::free(a_panel);
    std::free(b_panel);
    std::free(partial_tile);
    std::free(reference);
    return ok;
}
/** --------------------------------------------------------------------------------------------------------- Matmul Tile Wrapper Plan
 * @brief Verifies the matmul wrapper emits two PSL programs instead of a monolithic GEMM call.
 */
static bool test_matmul_tile_wrapper_plan() {
    auto plan = make_matmul_tile_f32_plan(F32_VECTOR_ELEMENTS);
    if (plan.chunk_source.find("sgemm") != std::string::npos || plan.reduction_source.find("sgemm") != std::string::npos) {
        return false;
    }
    if (plan.tile_element_count != 256 || plan.partial_element_count != 4096) {
        return false;
    }
    if (plan.chunk_program.operations().size() != 1 || plan.reduction_program.operations().size() != 1) {
        return false;
    }
    if (plan.chunk_program.operations()[0].kind != operation_kind::matmul_accumulate) {
        return false;
    }
    if (plan.reduction_program.operations()[0].kind != operation_kind::reduce_add) {
        return false;
    }
    return plan.reduction_program.operations()[0].output_element_count == 256;
}
/** --------------------------------------------------------------------------------------------------------- Executable Larger Matmul
 * @brief Verifies two PSL programs can loop partial tiles into a larger f32 GEMM.
 */
static bool test_executable_larger_matmul_two_programs() {
    constexpr uint32_t TileRows = F32_VECTOR_ELEMENTS;
    constexpr uint32_t TileCols = F32_VECTOR_ELEMENTS;
    constexpr uint32_t M = TileRows * 2U;
    constexpr uint32_t N = TileCols * 2U;
    constexpr uint32_t K = F32_VECTOR_ELEMENTS * F32_VECTOR_ELEMENTS;
    constexpr uint32_t KChunks = K / F32_VECTOR_ELEMENTS;
    constexpr uint32_t TileElements = TileRows * TileCols;
    auto* a = static_cast<float*>(alloc64(M * K * sizeof(float)));
    auto* b = static_cast<float*>(alloc64(K * N * sizeof(float)));
    auto* c = static_cast<float*>(alloc64(M * N * sizeof(float)));
    auto* reference = static_cast<float*>(alloc64(M * N * sizeof(float)));
    auto* a_panel = static_cast<float*>(alloc64(TileElements * sizeof(float)));
    auto* b_panel = static_cast<float*>(alloc64(TileElements * sizeof(float)));
    auto* partial_tiles = static_cast<float*>(alloc64(KChunks * TileElements * sizeof(float)));
    auto* output_tile = static_cast<float*>(alloc64(TileElements * sizeof(float)));
    for (uint32_t row = 0; row < M; row++) {
        for (uint32_t k = 0; k < K; k++) {
            a[row * K + k] = 0.01f * static_cast<float>(((row % 7U) + 1U) * ((k % 5U) + 1U));
        }
    }
    for (uint32_t k = 0; k < K; k++) {
        for (uint32_t col = 0; col < N; col++) {
            b[k * N + col] = 0.005f * static_cast<float>(((k % 11U) + 1U) * ((col % 13U) + 1U));
        }
    }
    for (uint32_t row = 0; row < M; row++) {
        for (uint32_t col = 0; col < N; col++) {
            double sum = 0.0;
            for (uint32_t k = 0; k < K; k++) {
                sum += static_cast<double>(a[row * K + k]) * static_cast<double>(b[k * N + col]);
            }
            c[row * N + col] = 0.0f;
            reference[row * N + col] = static_cast<float>(sum);
        }
    }
    auto plan = make_matmul_tile_f32_plan(KChunks);
    executable chunk_program(plan.chunk_source);
    executable reduction_program(plan.reduction_source);
    for (uint32_t tile_row = 0; tile_row < M; tile_row += TileRows) {
        for (uint32_t tile_col = 0; tile_col < N; tile_col += TileCols) {
            for (uint32_t chunk = 0; chunk < KChunks; chunk++) {
                for (uint32_t kk = 0; kk < F32_VECTOR_ELEMENTS; kk++) {
                    const uint32_t k = chunk * F32_VECTOR_ELEMENTS + kk;
                    for (uint32_t row = 0; row < TileRows; row++) {
                        a_panel[kk * TileRows + row] = a[(tile_row + row) * K + k];
                    }
                    for (uint32_t col = 0; col < TileCols; col++) {
                        b_panel[kk * TileCols + col] = b[k * N + tile_col + col];
                    }
                }
                chunk_program.exec({a_panel, b_panel, partial_tiles + chunk * TileElements});
            }
            reduction_program.exec({partial_tiles, output_tile});
            for (uint32_t row = 0; row < TileRows; row++) {
                for (uint32_t col = 0; col < TileCols; col++) {
                    c[(tile_row + row) * N + tile_col + col] = output_tile[row * TileCols + col];
                }
            }
        }
    }
    bool ok = true;
    for (uint32_t i = 0; i < M * N; i++) {
        if (std::fabs(c[i] - reference[i]) > 0.05f) {
            ok = false;
            break;
        }
    }
    std::free(a);
    std::free(b);
    std::free(c);
    std::free(reference);
    std::free(a_panel);
    std::free(b_panel);
    std::free(partial_tiles);
    std::free(output_tile);
    return ok;
}
/** --------------------------------------------------------------------------------------------------------- Main
 * @brief Runs Psyne parser contract tests.
 */
int main() {
    int passed = 0;
    int total = 17;
    std::printf("\n=== Psyne parser contract tests ===\n\n");
    if (test_declarations()) {
        passed++;
        std::printf("  [PASS] declarations, units, 64-byte ABI padding\n");
    } else {
        std::printf("  [FAIL] declarations, units, 64-byte ABI padding\n");
    }
    if (test_element_operations()) {
        passed++;
        std::printf("  [PASS] element operations and one-op statements\n");
    } else {
        std::printf("  [FAIL] element operations and one-op statements\n");
    }
    if (test_reductions()) {
        passed++;
        std::printf("  [PASS] reduction target sizing\n");
    } else {
        std::printf("  [FAIL] reduction target sizing\n");
    }
    if (test_access_contract()) {
        passed++;
        std::printf("  [PASS] declaration access contract\n");
    } else {
        std::printf("  [FAIL] declaration access contract\n");
    }
    if (test_executable_stream_multiply()) {
        passed++;
        std::printf("  [PASS] executable f32 stream multiply\n");
    } else {
        std::printf("  [FAIL] executable f32 stream multiply\n");
    }
    if (test_executable_element_primitives()) {
        passed++;
        std::printf("  [PASS] executable element primitives\n");
    } else {
        std::printf("  [FAIL] executable element primitives\n");
    }
    if (test_executable_stream_reductions()) {
        passed++;
        std::printf("  [PASS] executable scheduled stream reductions\n");
    } else {
        std::printf("  [FAIL] executable scheduled stream reductions\n");
    }
    if (test_matmul_accumulation_contract()) {
        passed++;
        std::printf("  [PASS] plain stream multiply accumulates into tile\n");
    } else {
        std::printf("  [FAIL] plain stream multiply accumulates into tile\n");
    }
    if (test_executable_partial_tile_matmul()) {
        passed++;
        std::printf("  [PASS] executable partial-tile matmul bytecode\n");
    } else {
        std::printf("  [FAIL] executable partial-tile matmul bytecode\n");
    }
    if (test_executable_stream_conversions()) {
        passed++;
        std::printf("  [PASS] executable stream conversion bytecode\n");
    } else {
        std::printf("  [FAIL] executable stream conversion bytecode\n");
    }
    if (test_executable_i4_lut_partial_tile_matmul()) {
        passed++;
        std::printf("  [PASS] executable i4 LUT partial-tile matmul bytecode\n");
    } else {
        std::printf("  [FAIL] executable i4 LUT partial-tile matmul bytecode\n");
    }
    if (test_executable_i2_lut_partial_tile_matmul()) {
        passed++;
        std::printf("  [PASS] executable i2 LUT partial-tile matmul bytecode\n");
    } else {
        std::printf("  [FAIL] executable i2 LUT partial-tile matmul bytecode\n");
    }
    if (test_executable_f16_element_primitives()) {
        passed++;
        std::printf("  [PASS] executable f16 element primitives\n");
    } else {
        std::printf("  [FAIL] executable f16 element primitives\n");
    }
    if (test_executable_f16_partial_tile_matmul()) {
        passed++;
        std::printf("  [PASS] executable f16 partial-tile matmul bytecode\n");
    } else {
        std::printf("  [FAIL] executable f16 partial-tile matmul bytecode\n");
    }
    if (test_executable_bf16_partial_tile_matmul()) {
        passed++;
        std::printf("  [PASS] executable bf16 partial-tile matmul bytecode\n");
    } else {
        std::printf("  [FAIL] executable bf16 partial-tile matmul bytecode\n");
    }
    if (test_matmul_tile_wrapper_plan()) {
        passed++;
        std::printf("  [PASS] matmul tile wrapper emits chunk and reduction PSL\n");
    } else {
        std::printf("  [FAIL] matmul tile wrapper emits chunk and reduction PSL\n");
    }
    if (test_executable_larger_matmul_two_programs()) {
        passed++;
        std::printf("  [PASS] executable two-program larger matmul\n");
    } else {
        std::printf("  [FAIL] executable two-program larger matmul\n");
    }
    std::printf("\n  Results: %d/%d passed\n\n", passed, total);
    return (passed == total) ? 0 : 1;
}
