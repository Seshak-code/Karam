#pragma once
// ============================================================================
// ast.h — Verilog-A Abstract Syntax Tree (AST)
// ============================================================================
// Intermediate representation representing the exact syntax of the module.
// Designed specifically for the AcuteSim subset (electrical only, deterministic).
// ============================================================================

#include <string>
#include <vector>
#include <memory>
#include "lexer.h"

namespace acutesim {
namespace physics {
namespace veriloga {

// ── Base AST Node ───────────────────────────────────────────────────────────
class ASTVisitor;

struct ASTNode {
    virtual ~ASTNode() = default;
    virtual void accept(ASTVisitor& visitor) const = 0;
    int line = 0;
};

// ── Expressions ─────────────────────────────────────────────────────────────
struct ExprNode : public ASTNode {};

struct NumberExprNode : public ExprNode {
    double value = 0.0;
    void accept(ASTVisitor& visitor) const override;
};

struct IdentifierExprNode : public ExprNode {
    std::string name;
    void accept(ASTVisitor& visitor) const override;
};

struct BinaryOpExprNode : public ExprNode {
    TokenType op; // PLUS, MINUS, MULT, DIV
    std::unique_ptr<ExprNode> left;
    std::unique_ptr<ExprNode> right;
    void accept(ASTVisitor& visitor) const override;
};

struct FunctionCallExprNode : public ExprNode {
    std::string functionName; // "sin", "exp", "ddt"
    std::vector<std::unique_ptr<ExprNode>> arguments;
    void accept(ASTVisitor& visitor) const override;
};

// V(a, b) or I(a, b)
struct NatureAccessExprNode : public ExprNode {
    std::string nature; // "V" or "I"
    std::string node1;
    std::string node2;  // Empty if single-ended e.g. V(n1) implicit to gnd
    void accept(ASTVisitor& visitor) const override;
};

// ── Analog Block & Contributions ─────────────────────────────────────────────
struct StatementNode : public ASTNode {};

struct ContributionNode : public StatementNode {
    // V(a,b) <+ expr  OR  I(a,b) <+ expr
    std::string nature; // "V" or "I"
    std::string node1;
    std::string node2; 
    std::unique_ptr<ExprNode> rhs;
    void accept(ASTVisitor& visitor) const override;
};

struct IfElseNode : public StatementNode {
    std::unique_ptr<ExprNode> condition;
    std::vector<std::unique_ptr<StatementNode>> trueBranch;
    std::vector<std::unique_ptr<StatementNode>> falseBranch;
    void accept(ASTVisitor& visitor) const override;
};

struct BlockStatementNode : public StatementNode {
    std::string blockName; // e.g. "begin : block_name"
    std::vector<std::unique_ptr<StatementNode>> statements;
    void accept(ASTVisitor& visitor) const override;
};

// Local variables in analog block `real Id;`
struct VariableDeclNode : public StatementNode {
    std::string type; // "real", "integer"
    std::string name;
    void accept(ASTVisitor& visitor) const override;
};

// Local assignment `Id = 0.0;`
struct AssignmentNode : public StatementNode {
    std::string name;
    std::unique_ptr<ExprNode> rhs;
    void accept(ASTVisitor& visitor) const override;
};

struct ProbeAnnotationNode : public StatementNode {
    std::string probeName;
    std::string nature; // "V" or "I"
    std::string node1;
    std::string node2;
    void accept(ASTVisitor& visitor) const override;
};

// ── Module & Declarations ────────────────────────────────────────────────────
struct DeclNode : public ASTNode {};

struct NetDeclNode : public DeclNode {
    std::string discipline; // "electrical", "ground"
    std::vector<std::string> names;
    void accept(ASTVisitor& visitor) const override;
};

struct ParamDeclNode : public DeclNode {
    std::string type; // "real", "integer"
    std::string name;
    std::unique_ptr<ExprNode> value;
    void accept(ASTVisitor& visitor) const override;
};

struct AnalogBlockNode : public ASTNode {
    std::vector<std::unique_ptr<StatementNode>> statements;
    void accept(ASTVisitor& visitor) const override;
};

struct ModuleNode : public ASTNode {
    std::string name;
    std::vector<std::string> ports;
    std::vector<std::unique_ptr<DeclNode>> declarations;
    std::unique_ptr<AnalogBlockNode> analogBlock;
    void accept(ASTVisitor& visitor) const override;
};

// ── Visitor Interface ────────────────────────────────────────────────────────
class ASTVisitor {
public:
    virtual ~ASTVisitor() = default;
    
    virtual void visit(const ModuleNode& node) = 0;
    virtual void visit(const AnalogBlockNode& node) = 0;
    
    // Declarations
    virtual void visit(const NetDeclNode& node) = 0;
    virtual void visit(const ParamDeclNode& node) = 0;
    
    // Statements
    virtual void visit(const ContributionNode& node) = 0;
    virtual void visit(const IfElseNode& node) = 0;
    virtual void visit(const BlockStatementNode& node) = 0;
    virtual void visit(const VariableDeclNode& node) = 0;
    virtual void visit(const AssignmentNode& node) = 0;
    virtual void visit(const ProbeAnnotationNode& node) = 0;

    // Expressions
    virtual void visit(const NumberExprNode& node) = 0;
    virtual void visit(const IdentifierExprNode& node) = 0;
    virtual void visit(const BinaryOpExprNode& node) = 0;
    virtual void visit(const FunctionCallExprNode& node) = 0;
    virtual void visit(const NatureAccessExprNode& node) = 0;
};

} // namespace veriloga
} // namespace physics
} // namespace acutesim
