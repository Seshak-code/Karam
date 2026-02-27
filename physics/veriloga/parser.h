#pragma once
// ============================================================================
// parser.h — Verilog-A Recursive Descent Parser
// ============================================================================
// Second stage of the C7 Lowering Pipeline.
// Consumes Tokens from Lexer and builds the Abstract Syntax Tree (AST).
// ============================================================================

#include "ast.h"
#include <stdexcept>

namespace acutesim {
namespace physics {
namespace veriloga {

class ParseError : public std::runtime_error {
public:
    int line;
    int column;
    ParseError(const std::string& msg, int l, int c)
        : std::runtime_error(msg), line(l), column(c) {}
};

class Parser {
public:
    explicit Parser(std::vector<Token> tokens);

    std::unique_ptr<ModuleNode> parseModule();

private:
    std::vector<Token> tokens_;
    size_t pos_ = 0;

    const Token& peek(size_t offset = 0) const;
    const Token& advance();
    bool match(TokenType type);
    void consume(TokenType type, const std::string& errMsg);
    bool isAtEnd() const;

    // Declarations
    std::unique_ptr<NetDeclNode> parseNetDecl();
    std::unique_ptr<ParamDeclNode> parseParamDecl();

    // Analog Block & Statements
    std::unique_ptr<AnalogBlockNode> parseAnalogBlock();
    std::unique_ptr<StatementNode> parseStatement();
    std::unique_ptr<BlockStatementNode> parseBlockStatement();
    std::unique_ptr<IfElseNode> parseIfElse();

    // Expressions (Pratt Parsing / Precedence Climbing)
    std::unique_ptr<ExprNode> parseExpression();
    std::unique_ptr<ExprNode> parseTerm();
    std::unique_ptr<ExprNode> parseFactor();
    std::unique_ptr<ExprNode> parsePrimary();
};

} // namespace veriloga
} // namespace physics
} // namespace acutesim
