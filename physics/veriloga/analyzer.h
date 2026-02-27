#pragma once
// ============================================================================
// analyzer.h — AST Semantic Analyzer & Evaluator
// ============================================================================
// Third stage of the C7 Lowering Pipeline.
// Visits the AST to build the Symbol Table, evaluate mathematical constant
// expressions, and resolve net references to integer matrix indices.
// ============================================================================

#include "ast.h"
#include <unordered_map>
#include <stdexcept>

namespace acutesim {
namespace physics {
namespace veriloga {

class SemanticError : public std::runtime_error {
public:
    SemanticError(const std::string& msg) : std::runtime_error(msg) {}
};

class Analyzer : public ASTVisitor {
public:
    Analyzer();

    void analyze(ModuleNode& root);

    // After analysis, these tables are fully populated:
    std::unordered_map<std::string, double> parameters;
    std::unordered_map<std::string, int>    netIndices;
    int maxNodeIndex = 0;

    // Evaluate an isolated expression using the current parameter context
    double evaluate(const ExprNode& expr);

    // ── ASTVisitor Implementation ──
    void visit(const ModuleNode& node) override;
    void visit(const AnalogBlockNode& node) override;
    
    void visit(const NetDeclNode& node) override;
    void visit(const ParamDeclNode& node) override;
    
    void visit(const ContributionNode& node) override;
    void visit(const IfElseNode& node) override;
    void visit(const BlockStatementNode& node) override;
    void visit(const VariableDeclNode& node) override;
    void visit(const AssignmentNode& node) override;
    void visit(const ProbeAnnotationNode& node) override;

    void visit(const NumberExprNode& node) override;
    void visit(const IdentifierExprNode& node) override;
    void visit(const BinaryOpExprNode& node) override;
    void visit(const FunctionCallExprNode& node) override;
    void visit(const NatureAccessExprNode& node) override;

private:
    double lastEvalResult_ = 0.0;
    
    // Scoped local variables (inside `analog begin ... end`)
    std::unordered_map<std::string, double> localVariables_;

    int resolveNet(const std::string& name);
};

} // namespace veriloga
} // namespace physics
} // namespace acutesim
