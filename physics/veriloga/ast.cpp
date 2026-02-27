#include "ast.h"

namespace acutesim {
namespace physics {
namespace veriloga {

// ── Expressions ─────────────────────────────────────────────────────────────

void NumberExprNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

void IdentifierExprNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

void BinaryOpExprNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

void FunctionCallExprNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

void NatureAccessExprNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

// ── Statements ──────────────────────────────────────────────────────────────

void ContributionNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

void IfElseNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

void BlockStatementNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

void VariableDeclNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

void AssignmentNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

void ProbeAnnotationNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

// ── Declarations ────────────────────────────────────────────────────────────

void NetDeclNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

void ParamDeclNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

// ── High-Level Blocks ───────────────────────────────────────────────────────

void AnalogBlockNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

void ModuleNode::accept(ASTVisitor& visitor) const {
    visitor.visit(*this);
}

} // namespace veriloga
} // namespace physics
} // namespace acutesim
