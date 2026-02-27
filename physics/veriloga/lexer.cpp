#include "lexer.h"
#include <cctype>
#include <stdexcept>
#include <unordered_map>
#include <sstream>

namespace acutesim {
namespace physics {
namespace veriloga {

std::string Token::toString() const {
    // Basic debug string
    return "[Line " + std::to_string(line) + ":" + std::to_string(column) + "] '" + text + "'";
}

// Keyword map
static const std::unordered_map<std::string, TokenType> KEYWORDS = {
    {"module",     TokenType::MODULE},
    {"endmodule",  TokenType::ENDMODULE},
    {"parameter",  TokenType::PARAMETER},
    {"real",       TokenType::REAL},
    {"integer",    TokenType::INTEGER},
    {"analog",     TokenType::ANALOG},
    {"begin",      TokenType::BEGIN},
    {"end",        TokenType::END},
    {"electrical", TokenType::ELECTRICAL},
    {"ground",     TokenType::GROUND},
    {"branch",     TokenType::BRANCH}
};

Lexer::Lexer(const std::string& source) : source_(source) {}

char Lexer::peek(size_t offset) const {
    if (pos_ + offset >= source_.length()) return '\0';
    return source_[pos_ + offset];
}

char Lexer::advance() {
    if (pos_ >= source_.length()) return '\0';
    char c = source_[pos_++];
    if (c == '\n') {
        line_++;
        col_ = 1;
    } else {
        col_++;
    }
    return c;
}

void Lexer::skipWhitespaceAndComments() {
    while (pos_ < source_.length()) {
        char c = peek();
        if (std::isspace(c)) {
            advance();
            continue;
        }

        // Single-line comment `//`
        if (c == '/' && peek(1) == '/') {
            // Check for explicit .probe magic comment (Phase C2/C5 artifact)
            // If it starts with "// .probe", we emit a PROBE token.
            std::string textAhead = source_.substr(pos_, 10);
            if (textAhead.find("// .probe") != std::string::npos ||
                textAhead.find("//  .probe") != std::string::npos) {
                // Return to token extraction — do not skip
                return;
            }

            // Normal comment, skip to end of line
            while (peek() != '\n' && peek() != '\0') {
                advance();
            }
            continue;
        }

        // Multi-line comment `/* ... */`
        if (c == '/' && peek(1) == '*') {
            advance(); advance(); // Skip /*
            while (peek() != '\0' && !(peek() == '*' && peek(1) == '/')) {
                advance();
            }
            if (peek() != '\0') {
                advance(); advance(); // Skip */
            }
            continue;
        }

        break; // Not whitespace or comment
    }
}

Token Lexer::makeToken(TokenType type, std::string text) {
    Token t;
    t.type = type;
    t.text = std::move(text);
    t.line = line_;
    // Column roughly marks the end of the token because we've already advanced;
    // accurate enough for parser error tracing.
    t.column = col_;
    return t;
}

Token Lexer::scanNumber() {
    std::string text;
    bool hasDot = false;
    bool hasExp = false;

    while (true) {
        char c = peek();
        if (std::isdigit(c)) {
            text += advance();
        } else if (c == '.' && !hasDot && !hasExp) {
            hasDot = true;
            text += advance();
        } else if ((c == 'e' || c == 'E') && !hasExp) {
            hasExp = true;
            text += advance();
            // Optional sign after e
            if (peek() == '+' || peek() == '-') {
                text += advance();
            }
        } else {
            break;
        }
    }

    // Handle engineering suffixes (e.g. 1k, 1u)
    char c = peek();
    double multiplier = 1.0;
    bool appliedSuffix = false;
    if (c == 'f') { multiplier = 1e-15; appliedSuffix = true; advance(); }
    else if (c == 'p') { multiplier = 1e-12; appliedSuffix = true; advance(); }
    else if (c == 'n') { multiplier = 1e-9;  appliedSuffix = true; advance(); }
    else if (c == 'u') { multiplier = 1e-6;  appliedSuffix = true; advance(); }
    else if (c == 'm') { multiplier = 1e-3;  appliedSuffix = true; advance(); }
    else if (c == 'k') { multiplier = 1e3;   appliedSuffix = true; advance(); }
    else if (c == 'M') {
        if (peek(1) == 'e' && peek(2) == 'g') {
            multiplier = 1e6; advance(); advance(); advance(); appliedSuffix = true;
        } else {
            // SPICE 'm' is milli, 'M' is milli or Meg depending on dialect.
            // Verilog-A generally uses 'M' for Meg.
            multiplier = 1e6; advance(); appliedSuffix = true;
        }
    }
    else if (c == 'G') { multiplier = 1e9; appliedSuffix = true; advance(); }
    else if (c == 'T') { multiplier = 1e12; appliedSuffix = true; advance(); }

    Token t = makeToken(TokenType::NUMBER, text);
    try {
        t.numberValue = std::stod(text) * multiplier;
    } catch (...) {
        t.numberValue = 0.0;
    }
    return t;
}

Token Lexer::scanIdentifierOrKeyword() {
    std::string text;
    char c = peek();
    
    // Verilog-A identifiers can start with ` a-z A-Z _
    // The ` indicates a macro like `M_PI; we'll treat it as part of the identifier.
    if (c == '`') {
        text += advance();
    }
    
    while (std::isalnum(peek()) || peek() == '_') {
        text += advance();
    }

    auto it = KEYWORDS.find(text);
    if (it != KEYWORDS.end()) {
        return makeToken(it->second, text);
    }
    return makeToken(TokenType::IDENTIFIER, text);
}

Token Lexer::scanString() {
    std::string text;
    advance(); // Skip opening quote
    while (peek() != '"' && peek() != '\0') {
        text += advance();
    }
    if (peek() == '"') advance(); // Skip closing quote
    return makeToken(TokenType::STRING, text);
}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;

    while (pos_ < source_.length()) {
        skipWhitespaceAndComments();
        if (pos_ >= source_.length()) break;

        char c = peek();

        // ── Special Probe Annotation ──
        if (c == '/' && peek(1) == '/') {
            // Must be the // .probe line because we skipped normal comments
            std::string line;
            while (peek() != '\n' && peek() != '\0') {
                line += advance();
            }
            tokens.push_back(makeToken(TokenType::PROBE, line));
            continue;
        }

        if (std::isalpha(c) || c == '_' || c == '`') {
            tokens.push_back(scanIdentifierOrKeyword());
        } 
        else if (std::isdigit(c) || (c == '.' && std::isdigit(peek(1)))) {
            tokens.push_back(scanNumber());
        } 
        else if (c == '"') {
            tokens.push_back(scanString());
        } 
        else {
            // Punctuation & Operators
            switch (c) {
                case '(': tokens.push_back(makeToken(TokenType::LPAREN, std::string(1, advance()))); break;
                case ')': tokens.push_back(makeToken(TokenType::RPAREN, std::string(1, advance()))); break;
                case '{': tokens.push_back(makeToken(TokenType::LBRACE, std::string(1, advance()))); break;
                case '}': tokens.push_back(makeToken(TokenType::RBRACE, std::string(1, advance()))); break;
                case ',': tokens.push_back(makeToken(TokenType::COMMA, std::string(1, advance()))); break;
                case ';': tokens.push_back(makeToken(TokenType::SEMICOLON, std::string(1, advance()))); break;
                case ':': tokens.push_back(makeToken(TokenType::COLON, std::string(1, advance()))); break;
                case '+': tokens.push_back(makeToken(TokenType::PLUS, std::string(1, advance()))); break;
                case '-': tokens.push_back(makeToken(TokenType::MINUS, std::string(1, advance()))); break;
                case '*': tokens.push_back(makeToken(TokenType::MULT, std::string(1, advance()))); break;
                case '/': tokens.push_back(makeToken(TokenType::DIV, std::string(1, advance()))); break;
                
                case '<': 
                    if (peek(1) == '+') {
                        advance(); advance();
                        tokens.push_back(makeToken(TokenType::CONTRIBUTE, "<+"));
                    } else if (peek(1) == '=') {
                        advance(); advance();
                        tokens.push_back(makeToken(TokenType::LESS_EQUAL, "<="));
                    } else {
                        tokens.push_back(makeToken(TokenType::LESS_THAN, std::string(1, advance())));
                    }
                    break;
                case '>':
                    if (peek(1) == '=') {
                        advance(); advance();
                        tokens.push_back(makeToken(TokenType::GREATER_EQUAL, ">="));
                    } else {
                        tokens.push_back(makeToken(TokenType::GREATER_THAN, std::string(1, advance())));
                    }
                    break;
                case '=':
                    if (peek(1) == '=') {
                        advance(); advance();
                        tokens.push_back(makeToken(TokenType::EQUAL, "=="));
                    } else {
                        tokens.push_back(makeToken(TokenType::ASSIGN, std::string(1, advance())));
                    }
                    break;
                case '!':
                    if (peek(1) == '=') {
                        advance(); advance();
                        tokens.push_back(makeToken(TokenType::NOT_EQUAL, "!="));
                    } else {
                        tokens.push_back(makeToken(TokenType::UNKNOWN, std::string(1, advance())));
                    }
                    break;
                default:
                    tokens.push_back(makeToken(TokenType::UNKNOWN, std::string(1, advance())));
                    break;
            }
        }
    }

    tokens.push_back(makeToken(TokenType::END_OF_FILE, "EOF"));
    return tokens;
}

} // namespace veriloga
} // namespace physics
} // namespace acutesim
