#pragma once
// ============================================================================
// lexer.h — Verilog-A Lexical Analyzer
// ============================================================================
// First stage of the C7 Lowering Pipeline.
// Converts UTF-8 string into typed Tokens, immediately coercing literals
// into precision-stable internal representations. No Qt headers.
// ============================================================================

#include <string>
#include <vector>
#include <vector>

namespace acutesim {
namespace physics {
namespace veriloga {

enum class TokenType {
    // Keywords
    MODULE, ENDMODULE, PARAMETER, REAL, INTEGER, 
    ANALOG, BEGIN, END, ELECTRICAL, GROUND, BRANCH,

    // Identifiers & Values
    IDENTIFIER,
    NUMBER,     // Extracted as double
    STRING,     // Extracted without quotes ("name")

    // Punctuation
    LPAREN, RPAREN,   // ( )
    LBRACE, RBRACE,   // { }
    COMMA, SEMICOLON, // , ;
    COLON,            // : (for names blocks like `begin : block`)

    // Operators
    ASSIGN,           // =
    CONTRIBUTE,       // <+
    PLUS, MINUS, MULT, DIV, // + - * /
    LESS_THAN, GREATER_THAN, LESS_EQUAL, GREATER_EQUAL, // < > <= >=
    EQUAL, NOT_EQUAL, // == !=

    // Special Annotations
    PROBE,            // $strobe or .probe

    END_OF_FILE,
    UNKNOWN
};

struct Token {
    TokenType type;
    std::string text;
    double numberValue = 0.0;
    int line = 0;
    int column = 0;

    // Helper for debugging/errors
    std::string toString() const;
};

class Lexer {
public:
    explicit Lexer(const std::string& source);
    
    // Lex the entire source and return sequential tokens
    std::vector<Token> tokenize();

private:
    std::string source_;
    size_t pos_ = 0;
    int line_ = 1;
    int col_ = 1;

    char peek(size_t offset = 0) const;
    char advance();
    void skipWhitespaceAndComments();
    Token scanNumber();
    Token scanIdentifierOrKeyword();
    Token scanString();
    Token makeToken(TokenType type, std::string text);
};

} // namespace veriloga
} // namespace physics
} // namespace acutesim
