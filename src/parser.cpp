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

namespace ane {
/** --------------------------------------------------------------------------------------------------------- Token Types
 * @enum TokenType
 * @brief All token types recognized by the DSL tokenizer.
 */
enum class TokenType {
    IDENT,          ///< Variable name or keyword
    NUMBER,         ///< Integer literal
    COLON,          ///< ':'
    SEMICOLON,      ///< ';'
    DOT,            ///< '.'
    LPAREN,         ///< '('
    RPAREN,         ///< ')'
    LBRACKET,       ///< '['
    RBRACKET,       ///< ']'
    PLUS,           ///< '+'
    MINUS,          ///< '-'
    STAR,           ///< '*'
    EQUALS,         ///< '='
    PLUSPLUS,        ///< '++'
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
 * integer literals, and line-comment skipping (//).
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
        auto single = [&](TokenType t) -> Token {
            pos_++;
            return {t, src_.substr(pos_ - 1, 1), line};
        };
        switch (c) {
            case ':': return single(TokenType::COLON);
            case ';': return single(TokenType::SEMICOLON);
            case '.': return single(TokenType::DOT);
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
/** --------------------------------------------------------------------------------------------------------- Compiler
 * @class Compiler
 * @brief Parses tokenized DSL source and emits bytecodes into an ane::program.
 *  - Variables are assigned z-registers starting from z2 (z0-z1 reserved as scratch).
 *  - Labels map to bytecode offsets for goto targets.
 *  - params[N] references map to the param table opcodes.
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
        int val = std::atoi(std::string(cur_.text).c_str());
        advance();
        return val;
    }
    /** --------------------------------------------------------------------------------------------- Parse Param Ref
     * @brief Parses `params[N]` and returns N.
     */
    uint8_t parseParamRef() {
        expect(TokenType::LBRACKET, "Expected '['");
        int idx = parseNumber();
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
        for (uint8_t i = 0; i < var.width; i++) {
            prog_.emit(Op::load_param, param_idx);
            prog_.emit(Op::mov_zreg, uint8_t(0), static_cast<uint8_t>(var.first_reg + i));
            if (i + 1 < var.width) prog_.emit(Op::advance_param, param_idx);
        }
    }
    /** --------------------------------------------------------------------------------------------- Emit Multi Save
     * @brief Emits mov_zreg + store_param + advance_param for each slot in a multi-vector variable.
     */
    void emitMultiSave(uint8_t param_idx, VarInfo var) {
        for (uint8_t i = 0; i < var.width; i++) {
            prog_.emit(Op::mov_zreg, static_cast<uint8_t>(var.first_reg + i), uint8_t(0));
            prog_.emit(Op::store_param, param_idx);
            if (i + 1 < var.width) prog_.emit(Op::advance_param, param_idx);
        }
    }
    /** --------------------------------------------------------------------------------------------- Emit Multi Arith
     * @brief Emits one arithmetic op per slot for multi-vector variables. All three must have the same width.
     */
    void emitMultiArith(Op op, VarInfo dst, VarInfo src1, VarInfo src2) {
        for (uint8_t i = 0; i < dst.width; i++) {
            prog_.emit(op,
                static_cast<uint8_t>(dst.first_reg + i),
                static_cast<uint8_t>(src1.first_reg + i),
                static_cast<uint8_t>(src2.first_reg + i));
        }
    }
    /** --------------------------------------------------------------------------------------------- Parse Statement
     * @brief Parses one statement. Dispatches based on the leading token pattern:
     *  - `name: TYPE` → declaration
     *  - `name.load(...)` / `name.save(...)` → param load/store
     *  - `name = expr` → assignment with arithmetic
     *  - `params[N]++` → advance pointer
     *  - `_LABEL_:` → label definition (identifier starting and ending with _)
     *  - `goto LABEL N` → counted loop
     */
    void parseStatement() {
        if (cur_.type == TokenType::IDENT && cur_.text == "goto") {
            advance();
            std::string label(cur_.text);
            advance();
            int count = parseNumber();
            auto it = labels_.find(label);
            if (it == labels_.end())
                throw std::runtime_error("Undefined label '" + label + "'");
            size_t label_offset = it->second;
            prog_.patch_u8(label_offset + 1, static_cast<uint8_t>(count));
            size_t body_start = label_offset + 2;
            uint16_t offset = static_cast<uint16_t>(prog_.mark() - body_start + 3);
            prog_.emit(Op::loop_end, offset);
            return;
        }
        if (cur_.type == TokenType::IDENT && cur_.text == "params") {
            advance();
            uint8_t idx = parseParamRef();
            expect(TokenType::PLUSPLUS, "Expected '++'");
            prog_.emit(Op::advance_param, idx);
            return;
        }
        if (cur_.type != TokenType::IDENT)
            throw std::runtime_error("Expected identifier at line " + std::to_string(cur_.line));
        std::string name(cur_.text);
        advance();
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
    Compiler(std::string_view source) : tok_(source) { advance(); }
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
};
/** --------------------------------------------------------------------------------------------------------- script Implementation
 */
script::script(const char* source) : source(source) {}
script::script(const std::string& source) : source(source) {}
void script::compile() {
    Compiler c(source);
    compiled_ = c.compile();
    compiled_valid_ = true;
}
void script::exec(std::initializer_list<const void*> params) {
    if (!compiled_valid_) compile();
    program p;
    uint8_t idx = 0;
    for (auto ptr : params) {
        p.emit(Op::set_param, idx, reinterpret_cast<uintptr_t>(ptr));
        idx++;
    }
    // Append the compiled body
    p.append(compiled_);
    p.exec();
}
} // namespace ane
