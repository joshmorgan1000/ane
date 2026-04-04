/** --------------------------------------------------------------------------------------------------------- Parser
 * @file parser.cpp
 * @brief Tokenizer and compiler for the ane scripting DSL. Parses a simple text language and
 * emits bytecodes into an ane::program. Variables map to z-registers automatically. Parameters
 * are passed as z_stream-compatible pointers and managed via the param table opcodes.
 *
 * @author Josh Morgan (@joshmorgan1000 on GitHub) with help from Claude
 * Released under the MIT License
 */
#include <ane/ane.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstdlib>
#include <cctype>
#include <cstring>

namespace ane {
/** --------------------------------------------------------------------------------------------------------- Token Types
 * @enum TokenType
 * @brief All token types recognized by the DSL tokenizer.
 */
enum class TokenType {
    IDENT,          ///< Variable name or keyword
    NUMBER,         ///< Integer literal
    FLOAT_NUMBER,   ///< Float literal (e.g., 1.0, 0.5f)
    COLON,          ///< ':'
    SEMICOLON,      ///< ';'
    DOT,            ///< '.'
    COMMA,          ///< ','
    LPAREN,         ///< '('
    RPAREN,         ///< ')'
    LBRACKET,       ///< '['
    RBRACKET,       ///< ']'
    PLUS,           ///< '+'
    MINUS,          ///< '-'
    STAR,           ///< '*'
    EQUALS,         ///< '='
    PLUSPLUS,        ///< '++'
    PLUSEQUALS,      ///< '+='
    END_OF_INPUT,   ///< End of source
};
/** --------------------------------------------------------------------------------------------------------- Token
 * @struct Token
 * @brief A single lexical token from the DSL source.
 */
struct Token {
    TokenType type;           ///< The type of this token
    std::string_view text;    ///< View into the original source for this token's text
    int line;                 ///< Line number (1-based) for error reporting
};
/** --------------------------------------------------------------------------------------------------------- Tokenizer
 * @class Tokenizer
 * @brief Splits DSL source into tokens. Handles single-char operators, ++, identifiers,
 * integer literals, float literals, and comment skipping (// and block comments).
 */
class Tokenizer {
private:
    std::string_view src_;    ///< Full source text
    size_t pos_ = 0;         ///< Current position in source
    int line_ = 1;           ///< Current line number
    void skipWhitespaceAndComments() {
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (c == '\n') { line_++; pos_++; }
            else if (std::isspace(static_cast<unsigned char>(c))) { pos_++; }
            else if (pos_ + 1 < src_.size() && c == '/' && src_[pos_ + 1] == '/') {
                while (pos_ < src_.size() && src_[pos_] != '\n') pos_++;
            } else if (pos_ + 1 < src_.size() && c == '/' && src_[pos_ + 1] == '*') {
                pos_ += 2;
                while (pos_ + 1 < src_.size() && !(src_[pos_] == '*' && src_[pos_ + 1] == '/')) {
                    if (src_[pos_] == '\n') line_++;
                    pos_++;
                }
                if (pos_ + 1 < src_.size()) pos_ += 2;
            } else break;
        }
    }
public:
    Tokenizer(std::string_view src) : src_(src) {}
    Token next() {
        skipWhitespaceAndComments();
        if (pos_ >= src_.size()) return {TokenType::END_OF_INPUT, {}, line_};
        int line = line_;
        char c = src_[pos_];
        if (c == '+' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '+') {
            pos_ += 2;
            return {TokenType::PLUSPLUS, src_.substr(pos_ - 2, 2), line};
        }
        if (c == '+' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '=') {
            pos_ += 2;
            return {TokenType::PLUSEQUALS, src_.substr(pos_ - 2, 2), line};
        }
        auto single = [&](TokenType t) -> Token {
            pos_++;
            return {t, src_.substr(pos_ - 1, 1), line};
        };
        switch (c) {
            case ':': return single(TokenType::COLON);
            case ';': return single(TokenType::SEMICOLON);
            case '.': return single(TokenType::DOT);
            case ',': return single(TokenType::COMMA);
            case '(': return single(TokenType::LPAREN);
            case ')': return single(TokenType::RPAREN);
            case '[': return single(TokenType::LBRACKET);
            case ']': return single(TokenType::RBRACKET);
            case '+': return single(TokenType::PLUS);
            case '-': return single(TokenType::MINUS);
            case '*': return single(TokenType::STAR);
            case '=': return single(TokenType::EQUALS);
            default: break;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            size_t start = pos_;
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) pos_++;
            if (pos_ < src_.size() && src_[pos_] == '.') {
                pos_++;
                while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) pos_++;
                if (pos_ < src_.size() && (src_[pos_] == 'f' || src_[pos_] == 'F')) pos_++;
                return {TokenType::FLOAT_NUMBER, src_.substr(start, pos_ - start), line};
            }
            return {TokenType::NUMBER, src_.substr(start, pos_ - start), line};
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t start = pos_;
            while (pos_ < src_.size() && (std::isalnum(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_')) pos_++;
            return {TokenType::IDENT, src_.substr(start, pos_ - start), line};
        }
        throw std::runtime_error(std::string("Unexpected character '") + c + "' at line " + std::to_string(line));
    }
};
/** --------------------------------------------------------------------------------------------------------- Intrinsic Argument Types
 * @enum ArgType
 * @brief Specifies how each argument of a DSL intrinsic function call is parsed and emitted.
 */
enum class ArgType : uint8_t {
    U8,   ///< Unsigned 8-bit integer from literal
    U32,  ///< Unsigned 32-bit integer from literal
    F32,  ///< 32-bit float from literal
    PTR,  ///< Pointer from params[N] — emits placeholder u64, records patch
};
/** --------------------------------------------------------------------------------------------------------- IntrinsicDef
 * @struct IntrinsicDef
 * @brief Defines an intrinsic function: its opcode and the expected argument types in order.
 */
struct IntrinsicDef {
    Op opcode;                    ///< Bytecode opcode to emit
    std::vector<ArgType> args;    ///< Argument types in order
};
/** --------------------------------------------------------------------------------------------------------- Compiler
 * @class Compiler
 * @brief Parses tokenized DSL source and emits bytecodes into an ane::program.
 *  - Variables are assigned z-registers starting from z2 (z0-z1 reserved as scratch).
 *  - Labels map to bytecode offsets for goto targets.
 *  - params[N] references map to the param table opcodes.
 *  - Intrinsic function calls emit high-level opcodes (GEMM, softmax, etc.) with
 *    pointer arguments deferred to exec time via PtrPatch fixups.
 */
/** --------------------------------------------------------------------------------------------------------- VarInfo
 * @struct VarInfo
 * @brief Tracks a DSL variable's register allocation — first slot and width (1, 2, or 4).
 */
struct VarInfo {
    uint8_t first_reg;  ///< First z-register slot (e.g., 2 for the first declared variable)
    uint8_t width;      ///< Number of consecutive z-register slots (1, 2, or 4)
};
class Compiler {
private:
    Tokenizer tok_;                                      ///< Token source
    Token cur_;                                          ///< Current lookahead token
    std::unordered_map<std::string, VarInfo> vars_;      ///< Variable name → register allocation
    std::unordered_map<std::string, size_t> labels_;     ///< Label name → bytecode offset
    std::unordered_map<std::string, IntrinsicDef> intrinsics_;  ///< Intrinsic name → definition
    std::vector<PtrPatch> patches_;                      ///< Pointer fixups for exec-time patching
    std::vector<U32Patch> u32_patches_;                  ///< Scalar u32 fixups for exec-time patching
    std::vector<U8Patch> u8_patches_;                    ///< Scalar u8 fixups for exec-time loop counts
    uint8_t next_reg_ = 2;                               ///< Next available z-register (0-1 reserved)
    program prog_;                                       ///< Bytecodes being built
    void advance() { cur_ = tok_.next(); }
    void expect(TokenType t, const char* msg) {
        if (cur_.type != t)
            throw std::runtime_error(std::string(msg) + " at line " + std::to_string(cur_.line));
        advance();
    }
    int parseNumber() {
        if (cur_.type != TokenType::NUMBER)
            throw std::runtime_error("Expected number at line " + std::to_string(cur_.line));
        int val = static_cast<int>(std::strtol(std::string(cur_.text).c_str(), nullptr, 10));
        advance();
        return val;
    }
    float parseFloat() {
        bool negate = false;
        if (cur_.type == TokenType::MINUS) { negate = true; advance(); }
        if (cur_.type == TokenType::FLOAT_NUMBER || cur_.type == TokenType::NUMBER) {
            float val = std::strtof(std::string(cur_.text).c_str(), nullptr);
            advance();
            return negate ? -val : val;
        }
        throw std::runtime_error("Expected float literal at line " + std::to_string(cur_.line));
    }
    /** --------------------------------------------------------------------------------------------- Parse Param Ref
     * @brief Parses `params[N]` and returns N.
     */
    uint8_t parseParamRef() {
        expect(TokenType::LBRACKET, "Expected '['");
        int idx = parseNumber();
        if (idx < 0 || idx >= 8)
            throw std::runtime_error("Param index out of range (0-7): " + std::to_string(idx));
        expect(TokenType::RBRACKET, "Expected ']'");
        return static_cast<uint8_t>(idx);
    }
    VarInfo varOf(std::string_view name) {
        auto it = vars_.find(std::string(name));
        if (it == vars_.end())
            throw std::runtime_error("Undeclared variable '" + std::string(name) + "'");
        return it->second;
    }
    /** --------------------------------------------------------------------------------------------- Emit Multi Load
     * @brief Emits load_param + mov_zreg + advance_param for each slot in a multi-vector variable.
     */
    void emitMultiLoad(uint8_t param_idx, VarInfo var) {
        if (var.width >= 2) {
            prog_.emit(Op::load_wide_param, var.width, param_idx, var.first_reg);
            // Auto-advance past all loaded vectors (user should not params[N]++ for wide)
            for (uint8_t i = 0; i < var.width; i++) prog_.emit(Op::advance_param, param_idx);
        } else {
            prog_.emit(Op::load_param, param_idx);
            prog_.emit(Op::mov_zreg, uint8_t(0), var.first_reg);
            // Width=1: no auto-advance — user controls with params[N]++
        }
    }
    /** --------------------------------------------------------------------------------------------- Emit Multi Save
     * @brief Emits mov_zreg + store_param + advance_param for each slot in a multi-vector variable.
     */
    void emitMultiSave(uint8_t param_idx, VarInfo var) {
        if (var.width >= 2) {
            prog_.emit(Op::store_wide_param, var.width, param_idx, var.first_reg);
            for (uint8_t i = 0; i < var.width; i++) prog_.emit(Op::advance_param, param_idx);
        } else {
            prog_.emit(Op::mov_zreg, var.first_reg, uint8_t(0));
            prog_.emit(Op::store_param, param_idx);
        }
    }
    /** --------------------------------------------------------------------------------------------- Emit Multi Arith
     * @brief Emits one arithmetic op per slot for multi-vector variables. All three must have the same width.
     */
    void emitMultiArith(Op op, VarInfo dst, VarInfo src1, VarInfo src2) {
        if (dst.width >= 2) {
            Op wide_op = (op == Op::fadd_zreg) ? Op::fadd_wide_zreg
                       : (op == Op::fsub_zreg) ? Op::fsub_wide_zreg
                       :                         Op::fmul_wide_zreg;
            prog_.emit(wide_op, dst.width, dst.first_reg, src1.first_reg, src2.first_reg);
        } else {
            prog_.emit(op, dst.first_reg, src1.first_reg, src2.first_reg);
        }
    }
    /** --------------------------------------------------------------------------------------------- Register Intrinsics
     * @brief Populates the intrinsic function table with all supported high-level ops.
     */
    void registerIntrinsics() {
        using A = ArgType;
        // ── Tile ZA operations ──────────────────────────────────────────────
        intrinsics_["zero_za"]       = {Op::zero_za, {}};
        intrinsics_["acc_smopa"]     = {Op::acc_smopa,   {A::U32, A::PTR, A::PTR}};
        intrinsics_["acc_umopa"]     = {Op::acc_umopa,   {A::U32, A::PTR, A::PTR}};
        intrinsics_["acc_usmopa"]    = {Op::acc_usmopa,  {A::U32, A::PTR, A::PTR}};
        intrinsics_["acc_sumopa"]    = {Op::acc_sumopa,   {A::U32, A::PTR, A::PTR}};
        intrinsics_["store_tiles"]   = {Op::store_tiles, {A::PTR}};
        intrinsics_["smopa_2x2"]     = {Op::smopa_2x2, {}};
        intrinsics_["umopa_2x2"]     = {Op::umopa_2x2, {}};
        intrinsics_["usmopa_2x2"]    = {Op::usmopa_2x2, {}};
        intrinsics_["load_bias"]     = {Op::load_bias,   {A::PTR}};
        intrinsics_["scale_store"]   = {Op::scale_store,  {A::F32, A::PTR}};
        intrinsics_["scatter_tile"]  = {Op::scatter_tile_fp32, {A::U32, A::U32, A::U32, A::PTR, A::PTR}};
        // ── Elementwise array ops ───────────────────────────────────────────
        intrinsics_["elementwise_add"]        = {Op::elementwise_add_fp32, {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["elementwise_scaled_add"] = {Op::elementwise_scaled_add_fp32, {A::U32, A::F32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["elementwise_mul"]        = {Op::elementwise_mul_fp32, {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["elementwise_sub"]        = {Op::elementwise_sub_fp32, {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["relu_backward"]          = {Op::relu_backward_fp32, {A::U32, A::PTR, A::PTR, A::PTR}};
        // ── Transpose / matmul / dense ──────────────────────────────────────
        intrinsics_["transpose"]     = {Op::transpose_fp32, {A::U32, A::U32, A::PTR, A::PTR}};
        intrinsics_["transpose_i8"]  = {Op::transpose_i8,   {A::U32, A::U32, A::PTR, A::PTR}};
        intrinsics_["softmax_argmax"] = {Op::softmax_argmax_fp32, {A::U32, A::U32, A::PTR, A::PTR, A::PTR, A::PTR, A::PTR}};
        intrinsics_["dense_fp32"]    = {Op::dense_fp32,   {A::U32, A::U32, A::U32, A::F32, A::U8, A::PTR, A::PTR, A::PTR, A::PTR}};
        intrinsics_["dense_i8"]      = {Op::dense_i8,     {A::U32, A::U32, A::U32, A::F32, A::F32, A::U8, A::PTR, A::PTR, A::PTR, A::PTR}};
        intrinsics_["dense_u8s8"]    = {Op::dense_u8s8,   {A::U32, A::U32, A::U32, A::F32, A::F32, A::U8, A::PTR, A::PTR, A::PTR, A::PTR}};
        intrinsics_["dense_strided"] = {Op::dense_strided_fp32, {A::U32, A::U32, A::U32, A::F32, A::U8, A::PTR, A::PTR, A::PTR, A::PTR}};
        intrinsics_["count_matches"] = {Op::count_matches, {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["reduce_sum"]    = {Op::reduce_sum_fp32, {A::U32, A::PTR, A::PTR}};
        intrinsics_["reduce_col_sum"] = {Op::reduce_col_sum_fp32, {A::U32, A::U32, A::U32, A::PTR, A::PTR}};
        // ── Quantization ────────────────────────────────────────────────────
        intrinsics_["quantize_fp32_i8"]           = {Op::quantize_fp32_i8, {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["dequantize_i8_fp32"]         = {Op::dequantize_i8_fp32, {A::U32, A::F32, A::PTR, A::PTR}};
        intrinsics_["pack_b_i8"]                  = {Op::pack_b_i8, {A::U32, A::U32, A::PTR, A::PTR}};
        intrinsics_["quantize_fp32_i8_channelwise"] = {Op::quantize_fp32_i8_channelwise, {A::U32, A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["quantize_pack_4bit"]         = {Op::quantize_pack_4bit_fp32, {A::U32, A::U32, A::PTR, A::PTR, A::PTR, A::PTR, A::PTR}};
        intrinsics_["quantize_accum_2bit"]        = {Op::quantize_accum_2bit, {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["accum_8bit"]                 = {Op::accum_8bit, {A::U32, A::PTR, A::PTR, A::PTR}};
        // ── Table lookup (fused kernel) ─────────────────────────────────────
        intrinsics_["luti2"]         = {Op::luti2_op, {A::U32, A::U8, A::PTR, A::PTR, A::PTR}};
        intrinsics_["luti4"]         = {Op::luti4_op, {A::U32, A::U8, A::PTR, A::PTR, A::PTR}};
        // ── Raw load/store (inline pointer, not via param table) ────────────
        intrinsics_["load_raw"]      = {Op::load, {A::PTR}};
        intrinsics_["store_raw"]     = {Op::store, {A::PTR}};
        // ── Distance / similarity metrics ───────────────────────────────────
        intrinsics_["l2_squared_fp32"]   = {Op::l2_squared_fp32,   {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["l2_squared_bf16"]   = {Op::l2_squared_bf16,   {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["l2_squared_f64"]    = {Op::l2_squared_f64,    {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["cosine_dist_fp32"]  = {Op::cosine_dist_fp32,  {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["cosine_dist_bf16"]  = {Op::cosine_dist_bf16,  {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["cosine_dist_f64"]   = {Op::cosine_dist_f64,   {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["normalize"]         = {Op::normalize_fp32,     {A::U32, A::PTR}};
        // ── DCT ─────────────────────────────────────────────────────────────
        intrinsics_["dct2_forward"]  = {Op::dct2_forward_fp32, {A::U32, A::PTR, A::PTR}};
        intrinsics_["dct2_inverse"]  = {Op::dct2_inverse_fp32, {A::U32, A::PTR, A::PTR}};
        // ── Bitmap / threshold / stats ──────────────────────────────────────
        intrinsics_["threshold_bitmap"] = {Op::threshold_bitmap_fp32, {A::U32, A::F32, A::PTR, A::PTR}};
        intrinsics_["welford_stats"]    = {Op::welford_stats_fp32, {A::U32, A::U32, A::PTR, A::PTR}};
        intrinsics_["threshold_8bit"]   = {Op::threshold_8bit, {A::U32, A::U8, A::PTR, A::PTR}};
        // ── SoA quantized accumulation ──────────────────────────────────────
        intrinsics_["soa_sub_scale_bf16"] = {Op::soa_sub_scale_bf16, {A::U32, A::PTR, A::F32, A::F32, A::PTR}};
        intrinsics_["soa_luti2_accum"]    = {Op::soa_luti2_accum, {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["soa_luti4_accum"]    = {Op::soa_luti4_accum, {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["bitmap_score"]       = {Op::bitmap_score_pipeline, {A::U32, A::U32, A::U32, A::U32, A::U32, A::PTR, A::PTR, A::PTR, A::PTR}};
        // ── Register-level z-vector ops ─────────────────────────────────────
        intrinsics_["mov"]           = {Op::mov_zreg, {A::U8, A::U8}};
        intrinsics_["fmla"]          = {Op::fmla_zreg, {A::U8, A::U8, A::U8}};
        intrinsics_["and_z"]         = {Op::and_zreg, {A::U8, A::U8, A::U8}};
        intrinsics_["orr_z"]         = {Op::orr_zreg, {A::U8, A::U8, A::U8}};
        intrinsics_["eor_z"]         = {Op::eor_zreg, {A::U8, A::U8, A::U8}};
        intrinsics_["not_z"]         = {Op::not_zreg, {A::U8, A::U8}};
        intrinsics_["lsl_z"]         = {Op::lsl_zreg, {A::U8, A::U8, A::U8}};
        intrinsics_["lsr_z"]         = {Op::lsr_zreg, {A::U8, A::U8, A::U8}};
        intrinsics_["asr_z"]         = {Op::asr_zreg, {A::U8, A::U8, A::U8}};
        intrinsics_["faddv"]         = {Op::faddv_zreg, {A::U8, A::U8}};
        intrinsics_["frsqrt"]        = {Op::frsqrt_zreg, {A::U8, A::U8}};
        intrinsics_["broadcast"]     = {Op::broadcast_scalar_zreg, {A::U8, A::F32}};
        intrinsics_["fscale"]        = {Op::fscale_zreg, {A::U8, A::U8, A::F32}};
        intrinsics_["fclamp"]        = {Op::fclamp_zreg, {A::U8, A::U8, A::U8, A::F32, A::F32}};
        intrinsics_["fdot"]          = {Op::fdot_zreg, {A::U8}};
        intrinsics_["fmla_wide"]     = {Op::fmla_wide_zreg, {A::U8}};
        // ── Table register / tile register ops ──────────────────────────────
        intrinsics_["load_zt0"]      = {Op::load_zt0, {A::PTR}};
        intrinsics_["luti2_zreg"]    = {Op::luti2_zreg, {A::U8, A::U8}};
        intrinsics_["luti4_zreg"]    = {Op::luti4_zreg, {A::U8, A::U8}};
        intrinsics_["smopa_zreg"]    = {Op::smopa_zreg,  {A::U8, A::U8, A::U8}};
        intrinsics_["umopa_zreg"]    = {Op::umopa_zreg,  {A::U8, A::U8, A::U8}};
        intrinsics_["usmopa_zreg"]   = {Op::usmopa_zreg, {A::U8, A::U8, A::U8}};
        intrinsics_["fmopa_zreg"]    = {Op::fmopa_zreg,  {A::U8, A::U8, A::U8}};
        // ── BLAS matrix multiply ────────────────────────────────────────────
        intrinsics_["sgemm"]   = {Op::cblas_sgemm,   {A::U8, A::U32, A::U32, A::U32, A::U32, A::U32, A::U32, A::F32, A::F32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["bfgemm"]  = {Op::cblas_bfgemm,   {A::U8, A::U32, A::U32, A::U32, A::U32, A::U32, A::U32, A::F32, A::F32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["igemm"]   = {Op::cblas_igemm,    {A::U8, A::U32, A::U32, A::U32, A::U32, A::U32, A::U32, A::F32, A::F32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["ugemm"]   = {Op::cblas_ugemm,    {A::U8, A::U32, A::U32, A::U32, A::U32, A::U32, A::U32, A::F32, A::F32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["usgemm"]  = {Op::cblas_usgemm,   {A::U8, A::U32, A::U32, A::U32, A::U32, A::U32, A::U32, A::F32, A::F32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["gemm_tile"] = {Op::gemm_tile_fp32, {A::U8, A::U32, A::U32, A::U32, A::U32, A::U32, A::U32, A::F32, A::F32, A::U32, A::U32, A::U32, A::U32, A::PTR, A::PTR, A::PTR}};
        // ── Transformer kernels ─────────────────────────────────────────────
        intrinsics_["softmax"]  = {Op::softmax_fp32,   {A::U32, A::PTR, A::PTR}};
        intrinsics_["rms_norm"] = {Op::rms_norm_fp32,  {A::U32, A::F32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["silu"]     = {Op::silu_fp32,      {A::U32, A::PTR, A::PTR}};
        intrinsics_["rope"]     = {Op::rope_fp32,      {A::U32, A::U32, A::F32, A::PTR, A::PTR}};
        intrinsics_["gelu"]     = {Op::gelu_fp32,      {A::U32, A::PTR, A::PTR}};
        intrinsics_["layer_norm"] = {Op::layer_norm_fp32, {A::U32, A::F32, A::PTR, A::PTR, A::PTR, A::PTR}};
        intrinsics_["causal_mask"] = {Op::causal_mask_fp32, {A::U32, A::U32, A::PTR}};
        // ── Decomposition building blocks ───────────────────────────────────
        intrinsics_["softmax_partial"] = {Op::softmax_partial_fp32, {A::U32, A::PTR, A::PTR, A::PTR, A::PTR}};
        intrinsics_["softmax_correct"] = {Op::softmax_correct_fp32, {A::U32, A::F32, A::F32, A::PTR, A::PTR}};
        intrinsics_["reduce_sum_sq"]   = {Op::reduce_sum_sq_fp32,   {A::U32, A::PTR, A::PTR}};
        // ── Backward passes ─────────────────────────────────────────────────
        intrinsics_["silu_backward"]       = {Op::silu_backward_fp32,       {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["softmax_backward"]    = {Op::softmax_backward_fp32,    {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["gelu_backward"]       = {Op::gelu_backward_fp32,       {A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["rms_norm_backward"]   = {Op::rms_norm_backward_fp32,   {A::U32, A::F32, A::PTR, A::PTR, A::PTR, A::PTR, A::PTR}};
        intrinsics_["layer_norm_backward"] = {Op::layer_norm_backward_fp32, {A::U32, A::F32, A::PTR, A::PTR, A::PTR, A::PTR, A::PTR, A::PTR}};
        intrinsics_["rope_backward"]       = {Op::rope_backward_fp32,       {A::U32, A::U32, A::F32, A::PTR, A::PTR}};
        intrinsics_["cross_entropy"]       = {Op::cross_entropy_fp32,       {A::U32, A::U32, A::PTR, A::PTR, A::PTR}};
        // ── Optimizer ───────────────────────────────────────────────────────
        intrinsics_["adam_step"] = {Op::adam_step_fp32, {A::U32, A::F32, A::F32, A::F32, A::F32, A::U32, A::PTR, A::PTR, A::PTR, A::PTR}};
        // ── Quantized GEMV ──────────────────────────────────────────────────
        intrinsics_["q8_0_gemv"] = {Op::q8_0_gemv, {A::U32, A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["q4_0_gemv"] = {Op::q4_0_gemv, {A::U32, A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["q4_k_gemv"] = {Op::q4_k_gemv, {A::U32, A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["q2_k_gemv"] = {Op::q2_k_gemv, {A::U32, A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["q3_k_gemv"] = {Op::q3_k_gemv, {A::U32, A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["q5_k_gemv"] = {Op::q5_k_gemv, {A::U32, A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["q6_k_gemv"] = {Op::q6_k_gemv, {A::U32, A::U32, A::PTR, A::PTR, A::PTR}};
        // ── Attention / embedding ───────────────────────────────────────────
        intrinsics_["flash_attention"] = {Op::flash_attention_fp32, {A::U32, A::U32, A::U8, A::PTR, A::PTR, A::PTR, A::PTR}};
        intrinsics_["get_rows"]      = {Op::get_rows_fp32, {A::U32, A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["get_rows_q8_0"] = {Op::get_rows_q8_0, {A::U32, A::U32, A::PTR, A::PTR, A::PTR}};
        intrinsics_["get_rows_q4_0"] = {Op::get_rows_q4_0, {A::U32, A::U32, A::PTR, A::PTR, A::PTR}};
        // ── Strided advance ─────────────────────────────────────────────────
        intrinsics_["advance_stride"] = {Op::advance_param_stride, {A::U8, A::U32}};
    }
    /** --------------------------------------------------------------------------------------------- Emit Intrinsic Arg
     * @brief Parses and emits one argument of an intrinsic function call according to its type.
     */
    void emitIntrinsicArg(ArgType type) {
        switch (type) {
            case ArgType::U8: {
                int val = parseNumber();
                uint8_t v = static_cast<uint8_t>(val);
                prog_.emit_raw(&v, 1);
                break;
            }
            case ArgType::U32: {
                if (cur_.type == TokenType::IDENT && cur_.text == "params") {
                    // Runtime U32 from scalar param table — deferred to exec time
                    advance();
                    uint8_t idx = parseParamRef();
                    size_t offset = prog_.mark();
                    uint32_t placeholder = 0;
                    prog_.emit_raw(&placeholder, 4);
                    u32_patches_.push_back({offset, idx});
                } else {
                    int val = parseNumber();
                    uint32_t v = static_cast<uint32_t>(val);
                    prog_.emit_raw(&v, 4);
                }
                break;
            }
            case ArgType::F32: {
                if (cur_.type == TokenType::IDENT && cur_.text == "params") {
                    // Runtime F32 from scalar param table — same 4-byte patch as U32
                    advance();
                    uint8_t idx = parseParamRef();
                    size_t offset = prog_.mark();
                    float placeholder = 0.0f;
                    prog_.emit_raw(&placeholder, 4);
                    u32_patches_.push_back({offset, idx});
                } else {
                    float val = parseFloat();
                    prog_.emit_raw(&val, 4);
                }
                break;
            }
            case ArgType::PTR: {
                if (cur_.type != TokenType::IDENT || cur_.text != "params")
                    throw std::runtime_error("Expected 'params[N]' for pointer argument at line "
                        + std::to_string(cur_.line));
                advance();
                uint8_t idx = parseParamRef();
                size_t offset = prog_.mark();
                uint64_t placeholder = 0;
                prog_.emit_raw(&placeholder, 8);
                patches_.push_back({offset, idx});
                break;
            }
        }
    }
    /** --------------------------------------------------------------------------------------------- Find Closest Intrinsic
     * @brief Finds the closest intrinsic name by prefix or edit distance for error suggestions.
     * @param name The unrecognized intrinsic name
     * @return A suggestion string, or empty if nothing is close
     */
    std::string findClosestIntrinsic(const std::string& name) const {
        std::string best;
        size_t best_score = 0;
        for (const auto& [iname, _] : intrinsics_) {
            // Prefix match: name is a prefix of an intrinsic or vice versa
            size_t common = 0;
            size_t min_len = std::min(name.size(), iname.size());
            for (size_t i = 0; i < min_len; i++) {
                if (name[i] == iname[i]) common++;
                else break;
            }
            if (common > best_score && common >= 3) {
                best_score = common;
                best = iname;
            }
        }
        return best;
    }
    /** --------------------------------------------------------------------------------------------- Parse Intrinsic Call
     * @brief Parses an intrinsic function call: name(arg1, arg2, ..., argN)
     * @return true if the identifier matched an intrinsic, false otherwise
     */
    bool tryParseIntrinsic(const std::string& name) {
        auto it = intrinsics_.find(name);
        if (it == intrinsics_.end()) return false;
        const IntrinsicDef& def = it->second;
        expect(TokenType::LPAREN, "Expected '(' after intrinsic name");
        prog_.emit_raw_u8(static_cast<uint8_t>(def.opcode));
        for (size_t i = 0; i < def.args.size(); i++) {
            if (i > 0) expect(TokenType::COMMA, "Expected ',' between arguments");
            emitIntrinsicArg(def.args[i]);
        }
        expect(TokenType::RPAREN, "Expected ')' after arguments");
        return true;
    }
    /** --------------------------------------------------------------------------------------------- Parse Statement
     * @brief Parses one statement. Dispatches based on the leading token pattern:
     *  - `name: TYPE` → declaration
     *  - `name.load(...)` / `name.save(...)` → param load/store
     *  - `name = expr` → assignment with arithmetic
     *  - `params[N]++` → advance pointer
     *  - `_LABEL_:` → label definition (identifier starting and ending with _)
     *  - `goto LABEL N` → counted loop
     *  - `intrinsic(args...)` → high-level opcode emission
     */
    void parseStatement() {
        if (cur_.type == TokenType::IDENT && cur_.text == "goto") {
            advance();
            std::string label(cur_.text);
            advance();
            auto it = labels_.find(label);
            if (it == labels_.end())
                throw std::runtime_error("Undefined label '" + label + "'");
            size_t label_offset = it->second;
            if (cur_.type == TokenType::IDENT && cur_.text == "params") {
                // Runtime loop count from scalar param table
                advance();
                uint8_t idx = parseParamRef();
                u8_patches_.push_back({label_offset + 1, idx});
            } else {
                int count = parseNumber();
                prog_.patch_u8(label_offset + 1, static_cast<uint8_t>(count));
            }
            size_t body_start = label_offset + 2;
            uint16_t offset = static_cast<uint16_t>(prog_.mark() - body_start + 3);
            prog_.emit(Op::loop_end, offset);
            return;
        }
        if (cur_.type == TokenType::IDENT && cur_.text == "params") {
            advance();
            uint8_t idx = parseParamRef();
            if (cur_.type == TokenType::PLUSEQUALS) {
                advance();
                if (cur_.type == TokenType::IDENT && cur_.text == "params") {
                    // Runtime stride from scalar param table
                    advance();
                    uint8_t stride_idx = parseParamRef();
                    prog_.emit_raw_u8(static_cast<uint8_t>(Op::advance_param_stride));
                    prog_.emit_raw_u8(idx);
                    size_t offset = prog_.mark();
                    uint32_t placeholder = 0;
                    prog_.emit_raw(&placeholder, 4);
                    u32_patches_.push_back({offset, stride_idx});
                } else {
                    int stride = parseNumber();
                    prog_.emit(Op::advance_param_stride, idx, uint32_t(stride));
                }
            } else {
                expect(TokenType::PLUSPLUS, "Expected '++' or '+='");
                prog_.emit(Op::advance_param, idx);
            }
            return;
        }
        if (cur_.type != TokenType::IDENT)
            throw std::runtime_error("Expected identifier at line " + std::to_string(cur_.line));
        std::string name(cur_.text);
        advance();
        // Check for intrinsic function call: name(...)
        if (cur_.type == TokenType::LPAREN) {
            if (!tryParseIntrinsic(name)) {
                std::string suggestion = findClosestIntrinsic(name);
                std::string msg = "Unknown intrinsic '" + name + "' at line "
                    + std::to_string(cur_.line);
                if (!suggestion.empty()) {
                    msg += ". Did you mean '" + suggestion + "'?";
                }
                throw std::runtime_error(msg);
            }
            return;
        }
        if (cur_.type == TokenType::COLON) {
            advance();
            if (cur_.type == TokenType::IDENT) {
                std::string type_name(cur_.text);
                advance();
                uint8_t width = 0;
                if (type_name == "ZVEC_F32" || type_name == "ZVEC") width = 1;
                else if (type_name == "ZVEC2_F32" || type_name == "ZVEC2") width = 2;
                else if (type_name == "ZVEC4_F32" || type_name == "ZVEC4") width = 4;
                if (width > 0) {
                    // Align register allocation: width=2 → even, width=4 → multiple of 4
                    if (width >= 2) next_reg_ = (next_reg_ + (width - 1)) & ~(width - 1);
                    if (next_reg_ + width > 32)
                        throw std::runtime_error("Register overflow: variable '" + name
                            + "' needs " + std::to_string(width) + " slots but only "
                            + std::to_string(32 - next_reg_) + " remain");
                    vars_[name] = {next_reg_, width};
                    next_reg_ += width;
                } else {
                    throw std::runtime_error("Unknown type '" + type_name + "'");
                }
            } else {
                // Label definition: `_LABEL_:` — no type follows the colon
                // Emit loop_begin with placeholder count (patched by goto)
                labels_[name] = prog_.mark();
                prog_.emit(Op::loop_begin, uint8_t(0));
            }
            return;
        }
        if (cur_.type == TokenType::DOT) {
            advance();
            std::string method(cur_.text);
            advance();
            expect(TokenType::LPAREN, "Expected '('");
            if (method == "load") {
                if (cur_.text != "params")
                    throw std::runtime_error("Expected 'params' in load()");
                advance();
                uint8_t idx = parseParamRef();
                expect(TokenType::RPAREN, "Expected ')'");
                VarInfo var = varOf(name);
                emitMultiLoad(idx, var);
            } else if (method == "save") {
                if (cur_.text != "params")
                    throw std::runtime_error("Expected 'params' in save()");
                advance();
                uint8_t idx = parseParamRef();
                expect(TokenType::RPAREN, "Expected ')'");
                VarInfo var = varOf(name);
                emitMultiSave(idx, var);
            } else {
                throw std::runtime_error("Unknown method '" + method + "'");
            }
            return;
        }
        if (cur_.type == TokenType::EQUALS) {
            advance();
            std::string lhs_name = name;
            VarInfo dst = varOf(lhs_name);
            std::string src1_name(cur_.text);
            VarInfo src1 = varOf(src1_name);
            advance();
            if (cur_.type == TokenType::PLUS || cur_.type == TokenType::MINUS
                || cur_.type == TokenType::STAR) {
                TokenType op_tok = cur_.type;
                advance();
                std::string src2_name(cur_.text);
                VarInfo src2 = varOf(src2_name);
                advance();
                Op arith_op = (op_tok == TokenType::PLUS)  ? Op::fadd_zreg
                            : (op_tok == TokenType::MINUS) ? Op::fsub_zreg
                            :                                Op::fmul_zreg;
                emitMultiArith(arith_op, dst, src1, src2);
            } else {
                // Simple assignment: dst = src1 (just moves)
                for (uint8_t i = 0; i < dst.width; i++) {
                    uint8_t d = static_cast<uint8_t>(dst.first_reg + i);
                    uint8_t s = static_cast<uint8_t>(src1.first_reg + i);
                    if (d != s) prog_.emit(Op::mov_zreg, s, d);
                }
            }
            return;
        }
        throw std::runtime_error("Unexpected token after '" + name + "' at line "
            + std::to_string(cur_.line));
    }
public:
    Compiler(std::string_view source) : tok_(source) {
        registerIntrinsics();
        advance();
    }
    /** --------------------------------------------------------------------------------------------- Compile
     * @brief Compiles the full source into a program. Does NOT emit set_param — those are
     * added at exec time when actual pointer values are known.
     * @return The compiled program (without set_param preamble)
     */
    program compile() {
        while (cur_.type != TokenType::END_OF_INPUT) {
            parseStatement();
            if (cur_.type == TokenType::SEMICOLON) advance();
        }
        return std::move(prog_);
    }
    uint8_t numVars() const { return next_reg_ - 2; }
    std::vector<PtrPatch> takePatches() { return std::move(patches_); }
    std::vector<U32Patch> takeU32Patches() { return std::move(u32_patches_); }
    std::vector<U8Patch> takeU8Patches() { return std::move(u8_patches_); }
};
/** --------------------------------------------------------------------------------------------------------- script Implementation
 */
script::script(const char* source) : source(source) {}
script::script(const std::string& source) : source(source) {}
void script::compile() {
    Compiler c(source);
    compiled_ = c.compile();
    patches_ = c.takePatches();
    u32_patches_ = c.takeU32Patches();
    u8_patches_ = c.takeU8Patches();
    compiled_valid_ = true;
}
void script::exec(std::initializer_list<const void*> params) {
    if (!compiled_valid_) compile();
    program p;
    std::vector<const void*> param_vec(params);
    uint8_t idx = 0;
    for (auto ptr : param_vec) {
        p.emit(Op::set_param, idx, reinterpret_cast<uintptr_t>(ptr));
        idx++;
    }
    size_t preamble_size = p.mark();
    p.append(compiled_);
    for (const auto& patch : patches_) {
        auto ptr_val = reinterpret_cast<uintptr_t>(param_vec[patch.param_idx]);
        p.patch_u64(preamble_size + patch.bytecode_offset, ptr_val);
    }
    p.exec();
}
void script::exec(std::initializer_list<const void*> ptrs,
                  std::initializer_list<uint32_t> scalars) {
    if (!compiled_valid_) compile();
    program p;
    std::vector<const void*> ptr_vec(ptrs);
    std::vector<uint32_t> scalar_vec(scalars);
    uint8_t idx = 0;
    for (auto ptr : ptr_vec) {
        p.emit(Op::set_param, idx, reinterpret_cast<uintptr_t>(ptr));
        idx++;
    }
    size_t preamble_size = p.mark();
    p.append(compiled_);
    for (const auto& patch : patches_) {
        auto ptr_val = reinterpret_cast<uintptr_t>(ptr_vec[patch.param_idx]);
        p.patch_u64(preamble_size + patch.bytecode_offset, ptr_val);
    }
    for (const auto& patch : u32_patches_) {
        p.patch_u32(preamble_size + patch.bytecode_offset, scalar_vec[patch.param_idx]);
    }
    for (const auto& patch : u8_patches_) {
        p.patch_u8(preamble_size + patch.bytecode_offset, static_cast<uint8_t>(scalar_vec[patch.param_idx]));
    }
    p.exec();
}
/** --------------------------------------------------------------------------------------------------------- prepared Implementation
 */
prepared prepared::from(const char* source, uint8_t num_ptrs, uint8_t num_scalars) {
    return from(std::string(source), num_ptrs, num_scalars);
}
prepared prepared::from(const std::string& source, uint8_t num_ptrs, uint8_t num_scalars) {
    Compiler c(source);
    program compiled = c.compile();
    auto ptr_patches  = c.takePatches();
    auto u32_patches  = c.takeU32Patches();
    auto u8_patches   = c.takeU8Patches();
    prepared p;
    p.num_ptrs_ = num_ptrs;
    p.ptr_patches_ = std::move(ptr_patches);
    p.u32_patches_ = std::move(u32_patches);
    p.u8_patches_ = std::move(u8_patches);
    // Build preamble: set_param opcodes with placeholder u64 values
    // Each set_param is [0x37][idx:u8][ptr:u64] = 10 bytes
    for (uint8_t i = 0; i < num_ptrs; i++) {
        p.ptr_preamble_.push_back(p.prog_.mark() + 2);  // offset of the u64 value
        p.prog_.emit(Op::set_param, i, uint64_t(0));
    }
    p.preamble_size_ = p.prog_.mark();
    p.prog_.append(compiled);
    return p;
}
void prepared::exec(std::initializer_list<const void*> ptrs) {
    // Patch set_param preamble pointer values in-place
    auto it = ptrs.begin();
    for (size_t i = 0; i < ptr_preamble_.size() && it != ptrs.end(); i++, ++it) {
        uint64_t val = reinterpret_cast<uintptr_t>(*it);
        prog_.patch_u64(ptr_preamble_[i], val);
    }
    // Patch inline PTR arguments in the body
    std::vector<const void*> ptr_vec(ptrs);
    for (const auto& patch : ptr_patches_) {
        auto ptr_val = reinterpret_cast<uintptr_t>(ptr_vec[patch.param_idx]);
        prog_.patch_u64(preamble_size_ + patch.bytecode_offset, ptr_val);
    }
    prog_.exec();
}
void prepared::exec(std::initializer_list<const void*> ptrs,
                    std::initializer_list<uint32_t> scalars) {
    // Patch set_param preamble pointer values in-place
    auto it = ptrs.begin();
    for (size_t i = 0; i < ptr_preamble_.size() && it != ptrs.end(); i++, ++it) {
        uint64_t val = reinterpret_cast<uintptr_t>(*it);
        prog_.patch_u64(ptr_preamble_[i], val);
    }
    // Patch inline PTR arguments in the body
    std::vector<const void*> ptr_vec(ptrs);
    for (const auto& patch : ptr_patches_) {
        auto ptr_val = reinterpret_cast<uintptr_t>(ptr_vec[patch.param_idx]);
        prog_.patch_u64(preamble_size_ + patch.bytecode_offset, ptr_val);
    }
    // Patch inline U32/F32 scalar arguments in the body
    std::vector<uint32_t> scalar_vec(scalars);
    for (const auto& patch : u32_patches_) {
        prog_.patch_u32(preamble_size_ + patch.bytecode_offset, scalar_vec[patch.param_idx]);
    }
    for (const auto& patch : u8_patches_) {
        prog_.patch_u8(preamble_size_ + patch.bytecode_offset, static_cast<uint8_t>(scalar_vec[patch.param_idx]));
    }
    prog_.exec();
}
} // namespace ane
