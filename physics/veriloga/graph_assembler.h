#pragma once
// ============================================================================
// graph_assembler.h — AST to TensorBlock Lowering Engine
// ============================================================================
// Fourth and final stage of the C7 Lowering Pipeline.
// Consumes the semantically resolved AST and translates the deterministic 
// analog block statements into physical components inside a TensorBlock.
// ============================================================================

#include "ast.h"
#include "analyzer.h"
#include "components/circuit.h"
#include <string>

namespace acutesim {
namespace physics {
namespace veriloga {

class GraphAssembler : public ASTVisitor {
public:
    // Takes the fully resolved AST and the Semantic Analyzer (which holds net maps)
    GraphAssembler(const Analyzer& semanticAnalyzer);

    void assemble(const ModuleNode& root);

    // The final generated TensorBlock containing the compiled primitives
    TensorBlock getResult() const;

    // ── ASTVisitor Implementation ──
    void visit(const ModuleNode& node) override;
    void visit(const AnalogBlockNode& node) override;
    
    // We only care about statements during assembly; declarations were handled in semantics
    void visit(const NetDeclNode& node) override {}
    void visit(const ParamDeclNode& node) override {}
    void visit(const VariableDeclNode& node) override {}
    void visit(const AssignmentNode& node) override;
    
    void visit(const ContributionNode& node) override;
    void visit(const IfElseNode& node) override;
    void visit(const BlockStatementNode& node) override;
    void visit(const ProbeAnnotationNode& node) override;

    // Expressions are evaluated dynamically or matched as patterns
    void visit(const NumberExprNode& node) override {}
    void visit(const IdentifierExprNode& node) override {}
    void visit(const BinaryOpExprNode& node) override {}
    void visit(const FunctionCallExprNode& node) override {}
    void visit(const NatureAccessExprNode& node) override {}

private:
    const Analyzer& semanticAnalyzer_;
    TensorBlock block_;

    // Context tracking for complex block structures (BJT, MOSFET)
    enum class ContextState { NONE, BJT, MOSFET };
    
    struct PendingDevice {
        std::string instanceName;
        ContextState type = ContextState::NONE;
        bool isNPN = true;
        bool isPMOS = false;
        int nodeCollectorOrDrain = -1;
        int nodeBaseOrGate = -1;
        int nodeEmitterOrSource = -1;
    };
    
    PendingDevice pending_;

    // Pattern matching helpers
    bool isResistorPattern(const ExprNode& rhs, double& outResistance);
    bool isCapacitorPattern(const ExprNode& rhs, double& outCapacitance);
    bool isInductorPattern(const ExprNode& rhs, double& outInductance);
    bool isDiodePattern(const ExprNode& rhs, std::string& outInstanceName);
    bool isCurrentSourcePattern(const ExprNode& rhs, double& outCurrent);
    bool isVoltageSourcePattern(const ExprNode& rhs, double& outVoltage);

    double evaluateConstant(const ExprNode& expr);
};

} // namespace veriloga
} // namespace physics
} // namespace acutesim
