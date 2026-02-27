#include "analyzer.h"
#include <cmath>

namespace acutesim {
namespace physics {
namespace veriloga {

Analyzer::Analyzer() {
    // "ground" is always unconditionally node 0
    netIndices["ground"] = 0;
    netIndices["gnd"] = 0;
}

void Analyzer::analyze(ModuleNode& root) {
    root.accept(*this);
}

double Analyzer::evaluate(const ExprNode& expr) {
    expr.accept(*this);
    return lastEvalResult_;
}

int Analyzer::resolveNet(const std::string& name) {
    if (name.empty()) return 0; // single-ended references imply ground
    auto it = netIndices.find(name);
    if (it == netIndices.end()) {
        throw SemanticError("Undeclared net reference: " + name);
    }
    return it->second;
}

// ── Expressions ──

void Analyzer::visit(const NumberExprNode& node) {
    lastEvalResult_ = node.value;
}

void Analyzer::visit(const IdentifierExprNode& node) {
    // Check parameters first
    auto itParam = parameters.find(node.name);
    if (itParam != parameters.end()) {
        lastEvalResult_ = itParam->second;
        return;
    }
    
    // Check local variables
    auto itLocal = localVariables_.find(node.name);
    if (itLocal != localVariables_.end()) {
        lastEvalResult_ = itLocal->second;
        return;
    }

    throw SemanticError("Undefined variable or parameter: " + node.name);
}

void Analyzer::visit(const BinaryOpExprNode& node) {
    double leftVal = evaluate(*node.left);
    double rightVal = evaluate(*node.right);
    
    switch (node.op) {
        case TokenType::PLUS:  lastEvalResult_ = leftVal + rightVal; break;
        case TokenType::MINUS: lastEvalResult_ = leftVal - rightVal; break;
        case TokenType::MULT:  lastEvalResult_ = leftVal * rightVal; break;
        case TokenType::DIV:
            if (rightVal == 0.0) throw SemanticError("Division by zero in expression evaluation.");
            lastEvalResult_ = leftVal / rightVal;
            break;
        default:
            throw SemanticError("Unsupported binary operator in expression.");
    }
}

void Analyzer::visit(const FunctionCallExprNode& node) {
    if (node.functionName == "exp") {
        if (node.arguments.size() != 1) throw SemanticError("exp() requires 1 argument.");
        lastEvalResult_ = std::exp(evaluate(*node.arguments[0]));
    } 
    else if (node.functionName == "sin") {
        if (node.arguments.size() != 1) throw SemanticError("sin() requires 1 argument.");
        lastEvalResult_ = std::sin(evaluate(*node.arguments[0]));
    }
    else if (node.functionName == "ddt") {
        // Semantic pass doesn't collapse ddt(). 
        // We either raise an error if evaluated directly here, or we treat it 
        // symbolically during the Graph Lowering phase.
        // For the analyzer, if someone tries to fold it into a literal, it's 0.
        lastEvalResult_ = 0.0; 
    }
    else {
        throw SemanticError("Unknown arithmetic function: " + node.functionName);
    }
}

void Analyzer::visit(const NatureAccessExprNode& node) {
    // V(a,b) cannot evaluate to a literal scalar during semantic analysis.
    lastEvalResult_ = 0.0;
}

// ── Statements & Analog Block ──

void Analyzer::visit(const ContributionNode& node) {
    // Validate nets exist
    resolveNet(node.node1);
    resolveNet(node.node2);
    // RHS evaluation is deferred to the Graph Lowering / Jacobian builder.
    // But we check that symbols exist if it contains constants.
}

void Analyzer::visit(const IfElseNode& node) {
    // Just traverse to validate logic, not execute
    for (auto& stmt : node.trueBranch) stmt->accept(*this);
    for (auto& stmt : node.falseBranch) stmt->accept(*this);
}

void Analyzer::visit(const BlockStatementNode& node) {
    for (auto& stmt : node.statements) stmt->accept(*this);
}

void Analyzer::visit(const VariableDeclNode& node) {
    localVariables_[node.name] = 0.0;
}

void Analyzer::visit(const AssignmentNode& node) {
    if (localVariables_.find(node.name) == localVariables_.end()) {
        throw SemanticError("Assignment to undeclared local variable: " + node.name);
    }
    // Static evaluation if possible, else 0 (dynamic runtime)
    localVariables_[node.name] = evaluate(*node.rhs);
}

void Analyzer::visit(const ProbeAnnotationNode& node) {
    resolveNet(node.node1);
    resolveNet(node.node2);
}

void Analyzer::visit(const AnalogBlockNode& node) {
    for (auto& stmt : node.statements) stmt->accept(*this);
}

// ── Declarations ──

void Analyzer::visit(const NetDeclNode& node) {
    for (const auto& name : node.names) {
        if (netIndices.find(name) != netIndices.end()) continue; // Already exists
        if (node.discipline == "ground") {
            netIndices[name] = 0;
        } else {
            maxNodeIndex++;
            netIndices[name] = maxNodeIndex;
        }
    }
}

void Analyzer::visit(const ParamDeclNode& node) {
    double val = 0.0;
    if (node.value) {
        val = evaluate(*node.value);
    }
    parameters[node.name] = val;
}

// ── Module Walk ──

void Analyzer::visit(const ModuleNode& node) {
    // 1. Process all declarations to build tables
    for (const auto& decl : node.declarations) {
        decl->accept(*this);
    }
    
    // 2. Add any implicit ports to the net map
    for (const auto& port : node.ports) {
        if (netIndices.find(port) == netIndices.end()) {
            maxNodeIndex++;
            netIndices[port] = maxNodeIndex;
        }
    }
    
    // 3. Process analog block
    if (node.analogBlock) {
        node.analogBlock->accept(*this);
    }
}

} // namespace veriloga
} // namespace physics
} // namespace acutesim
