#include "parser.h"

namespace acutesim {
namespace physics {
namespace veriloga {

Parser::Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

const Token& Parser::peek(size_t offset) const {
    if (pos_ + offset >= tokens_.size()) return tokens_.back(); // EOF
    return tokens_[pos_ + offset];
}

const Token& Parser::advance() {
    if (!isAtEnd()) pos_++;
    return tokens_[pos_ - 1];
}

bool Parser::match(TokenType type) {
    if (peek().type == type) {
        advance();
        return true;
    }
    return false;
}

void Parser::consume(TokenType type, const std::string& errMsg) {
    if (peek().type == type) {
        advance();
        return;
    }
    throw ParseError(errMsg + " (found: " + peek().text + ")", peek().line, peek().column);
}

bool Parser::isAtEnd() const {
    return peek().type == TokenType::END_OF_FILE;
}

// ── Expressions ─────────────────────────────────────────────────────────────

std::unique_ptr<ExprNode> Parser::parseExpression() {
    return parseTerm();
}

std::unique_ptr<ExprNode> Parser::parseTerm() {
    auto expr = parseFactor();

    while (peek().type == TokenType::PLUS || peek().type == TokenType::MINUS) {
        auto opToken = advance();
        auto binOp = std::make_unique<BinaryOpExprNode>();
        binOp->op = opToken.type;
        binOp->left = std::move(expr);
        binOp->right = parseFactor();
        expr = std::move(binOp);
    }
    return expr;
}

std::unique_ptr<ExprNode> Parser::parseFactor() {
    auto expr = parsePrimary();

    while (peek().type == TokenType::MULT || peek().type == TokenType::DIV) {
        auto opToken = advance();
        auto binOp = std::make_unique<BinaryOpExprNode>();
        binOp->op = opToken.type;
        binOp->left = std::move(expr);
        binOp->right = parsePrimary();
        expr = std::move(binOp);
    }
    return expr;
}

std::unique_ptr<ExprNode> Parser::parsePrimary() {
    if (match(TokenType::NUMBER)) {
        auto node = std::make_unique<NumberExprNode>();
        node->value = tokens_[pos_ - 1].numberValue;
        return node;
    }

    if (match(TokenType::LPAREN)) {
        auto expr = parseExpression();
        consume(TokenType::RPAREN, "Expected ')' after expression.");
        return expr;
    }

    // Special nature access: V(a,b) or I(a,b)
    if ((peek().text == "V" || peek().text == "I") && peek(1).type == TokenType::LPAREN) {
        Token natTok = advance();
        consume(TokenType::LPAREN, "Expected '(' after nature accessor.");
        auto acc = std::make_unique<NatureAccessExprNode>();
        acc->nature = natTok.text;
        
        consume(TokenType::IDENTIFIER, "Expected node name.");
        acc->node1 = tokens_[pos_ - 1].text;
        
        if (match(TokenType::COMMA)) {
            consume(TokenType::IDENTIFIER, "Expected second node name.");
            acc->node2 = tokens_[pos_ - 1].text;
        }
        consume(TokenType::RPAREN, "Expected ')' after nature accessor arguments.");
        return acc;
    }

    if (match(TokenType::IDENTIFIER)) {
        Token idTok = tokens_[pos_ - 1];
        
        // Function call: exp(), sin(), ddt()
        if (match(TokenType::LPAREN)) {
            auto call = std::make_unique<FunctionCallExprNode>();
            call->functionName = idTok.text;
            if (peek().type != TokenType::RPAREN) {
                do {
                    call->arguments.push_back(parseExpression());
                } while (match(TokenType::COMMA));
            }
            consume(TokenType::RPAREN, "Expected ')' after arguments.");
            return call;
        }

        // Variable reference
        auto node = std::make_unique<IdentifierExprNode>();
        node->name = idTok.text;
        return node;
    }

    throw ParseError("Expected expression.", peek().line, peek().column);
}

// ── Statements & Blocks ──────────────────────────────────────────────────────

std::unique_ptr<StatementNode> Parser::parseStatement() {
    Token startToken = peek();

    // Block
    if (peek().type == TokenType::BEGIN) {
        return parseBlockStatement();
    }

    // If-Else
    if (peek().text == "if") {
        return parseIfElse();
    }

    // Special Annotation (Probe)
    if (match(TokenType::PROBE)) {
        auto node = std::make_unique<ProbeAnnotationNode>();
        node->probeName = "parsed_probe"; // Extract exact behavior later
        return node;
    }

    // Variable Declaration
    if (peek().type == TokenType::REAL || peek().type == TokenType::INTEGER) {
        auto node = std::make_unique<VariableDeclNode>();
        node->type = advance().text;
        consume(TokenType::IDENTIFIER, "Expected variable name.");
        node->name = tokens_[pos_ - 1].text;
        // Check for multiple declarations e.g. `real a, b, c;`
        while (match(TokenType::COMMA)) {
            // Simplify AST: only record first for MVP, or expand tree. We skip for now.
            consume(TokenType::IDENTIFIER, "Expected variable name.");
        }
        consume(TokenType::SEMICOLON, "Expected ';' after variable declaration.");
        return node;
    }

    // Two possibilities left: Assignment (Id = expr;) or Contribution (I(a,b) <+ expr;)
    
    // Assignment: ident = expr;
    if (peek().type == TokenType::IDENTIFIER && peek(1).type == TokenType::ASSIGN) {
        auto node = std::make_unique<AssignmentNode>();
        node->name = advance().text; // Eat identifier
        advance(); // Eat '='
        node->rhs = parseExpression();
        consume(TokenType::SEMICOLON, "Expected ';' after assignment.");
        return node;
    }

    // Contribution: V(...) <+ expr; or I(...) <+ expr;
    if ((peek().text == "V" || peek().text == "I") && peek(1).type == TokenType::LPAREN) {
        Token natTok = advance(); // 'V' or 'I'
        advance(); // '('
        
        auto node = std::make_unique<ContributionNode>();
        node->nature = natTok.text;
        
        consume(TokenType::IDENTIFIER, "Expected node identifier in contribution.");
        node->node1 = tokens_[pos_ - 1].text;
        
        if (match(TokenType::COMMA)) {
            consume(TokenType::IDENTIFIER, "Expected second node identifier.");
            node->node2 = tokens_[pos_ - 1].text;
        }
        consume(TokenType::RPAREN, "Expected ')' in contribution.");
        
        consume(TokenType::CONTRIBUTE, "Expected '<+' operator.");
        node->rhs = parseExpression();
        consume(TokenType::SEMICOLON, "Expected ';' after contribution.");
        
        return node;
    }

    throw ParseError("Unexpected statement.", peek().line, peek().column);
}

std::unique_ptr<BlockStatementNode> Parser::parseBlockStatement() {
    consume(TokenType::BEGIN, "Expected 'begin' to start block.");
    auto node = std::make_unique<BlockStatementNode>();
    
    if (match(TokenType::COLON)) {
        consume(TokenType::IDENTIFIER, "Expected block name after ':'.");
        node->blockName = tokens_[pos_ - 1].text;
    }
    
    while (!isAtEnd() && peek().type != TokenType::END) {
        node->statements.push_back(parseStatement());
    }
    
    consume(TokenType::END, "Expected 'end' to close block.");
    return node;
}

std::unique_ptr<IfElseNode> Parser::parseIfElse() {
    consume(TokenType::IDENTIFIER, "Expected 'if'."); // Assumes text is "if"
    auto node = std::make_unique<IfElseNode>();
    
    consume(TokenType::LPAREN, "Expected '(' after 'if'.");
    node->condition = parseExpression();
    consume(TokenType::RPAREN, "Expected ')' after condition.");
    
    node->trueBranch.push_back(parseStatement());
    
    if (peek().text == "else") {
        advance();
        node->falseBranch.push_back(parseStatement());
    }
    
    return node;
}

// ── Declarations ─────────────────────────────────────────────────────────────

std::unique_ptr<NetDeclNode> Parser::parseNetDecl() {
    auto node = std::make_unique<NetDeclNode>();
    node->discipline = advance().text; // "electrical" or "ground"
    
    do {
        consume(TokenType::IDENTIFIER, "Expected net name.");
        node->names.push_back(tokens_[pos_ - 1].text);
    } while (match(TokenType::COMMA));
    
    consume(TokenType::SEMICOLON, "Expected ';' after net declaration.");
    return node;
}

std::unique_ptr<ParamDeclNode> Parser::parseParamDecl() {
    auto node = std::make_unique<ParamDeclNode>();
    consume(TokenType::PARAMETER, "Internal error: parseParamDecl without 'parameter'.");
    
    if (match(TokenType::REAL) || match(TokenType::INTEGER)) {
        node->type = tokens_[pos_ - 1].text;
    }
    
    consume(TokenType::IDENTIFIER, "Expected parameter name.");
    node->name = tokens_[pos_ - 1].text;
    
    consume(TokenType::ASSIGN, "Expected '=' in parameter declaration.");
    node->value = parseExpression();
    
    consume(TokenType::SEMICOLON, "Expected ';' after parameter declaration.");
    return node;
}

// ── Top Level ───────────────────────────────────────────────────────────────

std::unique_ptr<AnalogBlockNode> Parser::parseAnalogBlock() {
    consume(TokenType::ANALOG, "Expected 'analog'.");
    consume(TokenType::BEGIN, "Expected 'begin' for analog block.");
    
    auto node = std::make_unique<AnalogBlockNode>();
    while (!isAtEnd() && peek().type != TokenType::END) {
        node->statements.push_back(parseStatement());
    }
    
    consume(TokenType::END, "Expected 'end' for analog block.");
    return node;
}

std::unique_ptr<ModuleNode> Parser::parseModule() {
    consume(TokenType::MODULE, "Expected 'module' at start of file.");
    
    auto node = std::make_unique<ModuleNode>();
    consume(TokenType::IDENTIFIER, "Expected module name.");
    node->name = tokens_[pos_ - 1].text;
    
    if (match(TokenType::LPAREN)) {
        if (peek().type != TokenType::RPAREN) {
            do {
                consume(TokenType::IDENTIFIER, "Expected port name.");
                node->ports.push_back(tokens_[pos_ - 1].text);
            } while (match(TokenType::COMMA));
        }
        consume(TokenType::RPAREN, "Expected ')' after ports.");
    }
    consume(TokenType::SEMICOLON, "Expected ';' after module definition.");
    
    // Parse bodies
    while (!isAtEnd() && peek().type != TokenType::ENDMODULE) {
        if (peek().type == TokenType::ELECTRICAL || peek().type == TokenType::GROUND) {
            node->declarations.push_back(parseNetDecl());
        } else if (peek().type == TokenType::PARAMETER) {
            node->declarations.push_back(parseParamDecl());
        } else if (peek().type == TokenType::ANALOG) {
            node->analogBlock = parseAnalogBlock();
        } else if (peek().type == TokenType::IDENTIFIER && peek().text == "`include") {
            // Ignore Verilog-AMS include directives
            advance();
            consume(TokenType::STRING, "Expected string after `include");
        } else {
            throw ParseError("Unexpected token in module body: " + peek().text, peek().line, peek().column);
        }
    }
    
    consume(TokenType::ENDMODULE, "Expected 'endmodule' at end of file.");
    return node;
}

} // namespace veriloga
} // namespace physics
} // namespace acutesim
