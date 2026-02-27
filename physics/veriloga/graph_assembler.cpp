#include "graph_assembler.h"
#include <stdexcept>
#include <iostream>

namespace acutesim {
namespace physics {
namespace veriloga {

GraphAssembler::GraphAssembler(const Analyzer& semanticAnalyzer)
    : semanticAnalyzer_(semanticAnalyzer) {
    block_.numInternalNodes = semanticAnalyzer_.maxNodeIndex;
}

TensorBlock GraphAssembler::getResult() const {
    return block_;
}

double GraphAssembler::evaluateConstant(const ExprNode& expr) {
    // We can reuse the analyzer's evaluate method because parameter values are frozen.
    Analyzer tempAnalyzer = semanticAnalyzer_; 
    return tempAnalyzer.evaluate(expr);
}

void GraphAssembler::assemble(const ModuleNode& root) {
    block_.name = root.name;
    // Map ports to internal node indices
    for (const auto& p : root.ports) {
        block_.pinNames.push_back(p);
        auto it = semanticAnalyzer_.netIndices.find(p);
        if (it != semanticAnalyzer_.netIndices.end()) {
            block_.portNodeIndices.push_back(it->second);
        } else {
            block_.portNodeIndices.push_back(0); // Should be caught by semantics
        }
    }
    root.accept(*this);
}

void GraphAssembler::visit(const ModuleNode& node) {
    if (node.analogBlock) {
        node.analogBlock->accept(*this);
    }
}

void GraphAssembler::visit(const AnalogBlockNode& node) {
    for (const auto& stmt : node.statements) {
        stmt->accept(*this);
    }
}

void GraphAssembler::visit(const BlockStatementNode& node) {
    // Check if this block corresponds to a BJT or MOSFET macro
    // In our spec, block names like "Q1_em" or "M1_sh" define the device boundary.

    if (node.blockName.find("_em") != std::string::npos) {
        pending_.instanceName = node.blockName.substr(0, node.blockName.find("_em"));
        pending_.type = ContextState::BJT;
        
        auto it = semanticAnalyzer_.parameters.find(pending_.instanceName + "_isNPN");
        pending_.isNPN = (it != semanticAnalyzer_.parameters.end()) ? (static_cast<int>(it->second) != 0) : true;
        
        for (const auto& stmt : node.statements) stmt->accept(*this);

        if (pending_.nodeCollectorOrDrain >= 0 && pending_.nodeBaseOrGate >= 0 && pending_.nodeEmitterOrSource >= 0) {
            BJT bjt;
            bjt.instanceName = pending_.instanceName;
            bjt.isNPN = pending_.isNPN;
            bjt.nodeCollector = pending_.nodeCollectorOrDrain;
            bjt.base = pending_.nodeBaseOrGate;
            bjt.emitter = pending_.nodeEmitterOrSource;
            
            auto pIs = semanticAnalyzer_.parameters.find(bjt.instanceName + "_Is");
            bjt.saturationCurrent_I_S_A = (pIs != semanticAnalyzer_.parameters.end()) ? pIs->second : 1e-14;
            
            auto pBetaF = semanticAnalyzer_.parameters.find(bjt.instanceName + "_betaF");
            bjt.betaF = (pBetaF != semanticAnalyzer_.parameters.end()) ? pBetaF->second : 100.0;
            
            auto pVt = semanticAnalyzer_.parameters.find(bjt.instanceName + "_Vt");
            bjt.thermalVoltage_V_T_V = (pVt != semanticAnalyzer_.parameters.end()) ? pVt->second : 0.02585;

            block_.bjts.push_back(bjt);
        }
        pending_.type = ContextState::NONE;
        return;
    }

    if (node.blockName.find("_sh") != std::string::npos) {
        pending_.instanceName = node.blockName.substr(0, node.blockName.find("_sh"));
        pending_.type = ContextState::MOSFET;
        
        auto it = semanticAnalyzer_.parameters.find(pending_.instanceName + "_isPMOS");
        pending_.isPMOS = (it != semanticAnalyzer_.parameters.end()) ? (static_cast<int>(it->second) != 0) : false;
        
        for (const auto& stmt : node.statements) stmt->accept(*this);

        if (pending_.nodeCollectorOrDrain >= 0 && pending_.nodeBaseOrGate >= 0 && pending_.nodeEmitterOrSource >= 0) {
            Mosfet m;
            m.instanceName = pending_.instanceName;
            m.drain = pending_.nodeCollectorOrDrain;
            m.gate = pending_.nodeBaseOrGate;
            m.source = pending_.nodeEmitterOrSource;
            m.body = m.source; // Default short
            m.modelName = pending_.isPMOS ? "pfet_01v8" : "nfet_01v8";

            auto pW = semanticAnalyzer_.parameters.find(m.instanceName + "_W");
            m.w = (pW != semanticAnalyzer_.parameters.end()) ? pW->second : 1e-6;
            
            auto pL = semanticAnalyzer_.parameters.find(m.instanceName + "_L");
            m.l = (pL != semanticAnalyzer_.parameters.end()) ? pL->second : 1e-6;

            block_.mosfets.push_back(m);
        }
        pending_.type = ContextState::NONE;
        return;
    }

    // Normal block
    for (const auto& stmt : node.statements) stmt->accept(*this);
}

void GraphAssembler::visit(const IfElseNode& node) {
    // MVP: execute true branch unconditionally for structure extraction.
    // In a full compiler, if/else controls behavioral stamping at runtime.
    for (const auto& stmt : node.trueBranch) stmt->accept(*this);
}

void GraphAssembler::visit(const AssignmentNode& node) {
    // Intercept M1_Vgs = V(ng, ns) 
    if (pending_.type == ContextState::MOSFET && node.name == pending_.instanceName + "_Vgs") {
        if (auto nat = dynamic_cast<const NatureAccessExprNode*>(node.rhs.get())) {
            if (nat->nature == "V") {
                pending_.nodeBaseOrGate = semanticAnalyzer_.netIndices.at(nat->node1);
            }
        }
    }
}

void GraphAssembler::visit(const ProbeAnnotationNode& node) {
    auto n1 = semanticAnalyzer_.netIndices.find(node.node1);
    auto n2 = semanticAnalyzer_.netIndices.find(node.node2);
    if (n1 != semanticAnalyzer_.netIndices.end() && n2 != semanticAnalyzer_.netIndices.end()) {
        Probe p;
        p.name = node.probeName;
        p.nodePositive = n1->second;
        p.nodeNegative = n2->second;
        block_.probes.push_back(p);
    }
}

// ── Pattern Extractors ──────────────────────────────────────────────────────

bool GraphAssembler::isResistorPattern(const ExprNode& rhs, double& outResistance) {
    // Pattern: `V(a,b) / <constant>`
    const BinaryOpExprNode* binExt = dynamic_cast<const BinaryOpExprNode*>(&rhs);
    if (binExt && binExt->op == TokenType::DIV) {
        if (dynamic_cast<const NatureAccessExprNode*>(binExt->left.get())) {
            try {
                outResistance = evaluateConstant(*binExt->right);
                return true;
            } catch (...) { return false; }
        }
    }
    return false;
}

bool GraphAssembler::isCapacitorPattern(const ExprNode& rhs, double& outCapacitance) {
    // Pattern: `<constant> * ddt(V(a,b))`
    const BinaryOpExprNode* binExt = dynamic_cast<const BinaryOpExprNode*>(&rhs);
    if (binExt && binExt->op == TokenType::MULT) {
        const FunctionCallExprNode* func = dynamic_cast<const FunctionCallExprNode*>(binExt->right.get());
        if (func && func->functionName == "ddt") {
            try {
                outCapacitance = evaluateConstant(*binExt->left);
                return true;
            } catch (...) { return false; }
        }
    }
    return false;
}

bool GraphAssembler::isInductorPattern(const ExprNode& rhs, double& outInductance) {
    // Pattern: `<constant> * ddt(I(a,b))`
    return isCapacitorPattern(rhs, outInductance); // Syntactically identical RHS multiplier
}

bool GraphAssembler::isDiodePattern(const ExprNode& rhs, std::string& outInstanceName) {
    // Pattern: `Is * (exp(...) - 1)`
    const BinaryOpExprNode* bin1 = dynamic_cast<const BinaryOpExprNode*>(&rhs);
    if (bin1 && bin1->op == TokenType::MULT) {
        const IdentifierExprNode* id = dynamic_cast<const IdentifierExprNode*>(bin1->left.get());
        if (id && id->name.find("_Is") != std::string::npos) {
            outInstanceName = id->name.substr(0, id->name.find("_Is"));
            return true;
        }
    }
    return false;
}

// ── Contribution Translation ─────────────────────────────────────────────────

void GraphAssembler::visit(const ContributionNode& node) {
    int node1 = semanticAnalyzer_.netIndices.at(node.node1);
    int node2 = node.node2.empty() ? 0 : semanticAnalyzer_.netIndices.at(node.node2);

    // Context-sensitive resolution inside macros
    if (pending_.type == ContextState::BJT) {
        if (node.nature == "I") {
            // First Current statement is usually collector-emitter
            if (pending_.nodeCollectorOrDrain == -1) {
                if (pending_.isNPN) { pending_.nodeCollectorOrDrain = node1; pending_.nodeEmitterOrSource = node2; }
                else                { pending_.nodeEmitterOrSource = node1; pending_.nodeCollectorOrDrain = node2; }
            } else {
                // Second is base
                if (pending_.isNPN) { pending_.nodeBaseOrGate = node1; }
                else                { pending_.nodeBaseOrGate = node2; }
            }
        }
        return;
    }

    if (pending_.type == ContextState::MOSFET) {
        if (node.nature == "I") {
            if (pending_.isPMOS) { pending_.nodeCollectorOrDrain = node2; pending_.nodeEmitterOrSource = node1; }
            else                 { pending_.nodeCollectorOrDrain = node1; pending_.nodeEmitterOrSource = node2; }
        }
        else if (node.nature == "V") {
            // Unused in simplified extractor, but gate voltage is resolved during semantic traversal
        }
        // If gate voltage is computed via assignment `M1_Vgs = V(ng,ns)` we extract `ng` directly 
        // by looking for AssignmentNodes in a full semantic pass. For MVP we will trust the caller.
        return;
    }

    // Independent Primitives mapping
    if (node.nature == "I") {
        double R, C;
        std::string dName;
        if (isResistorPattern(*node.rhs, R)) {
            static int rId = 1;
            Resistor r;
            r.name = "R" + std::to_string(rId++);
            r.nodeTerminal1 = node1;
            r.nodeTerminal2 = node2;
            r.resistance_ohms = R;
            block_.resistors.push_back(r);
        }
        else if (isCapacitorPattern(*node.rhs, C)) {
            static int cId = 1;
            Capacitor c;
            c.name = "C" + std::to_string(cId++);
            c.nodePlate1 = node1;
            c.nodePlate2 = node2;
            c.capacitance_farads = C;
            block_.capacitors.push_back(c);
        }
        else if (isDiodePattern(*node.rhs, dName)) {
            Diode d;
            d.instanceName = dName;
            d.anode = node1;
            d.cathode = node2;
            auto pIs = semanticAnalyzer_.parameters.find(dName + "_Is");
            d.saturationCurrent_I_S_A = (pIs != semanticAnalyzer_.parameters.end()) ? pIs->second : 1e-14;
            auto pN = semanticAnalyzer_.parameters.find(dName + "_N");
            d.emissionCoefficient_N = (pN != semanticAnalyzer_.parameters.end()) ? pN->second : 1.0;
            auto pVt = semanticAnalyzer_.parameters.find(dName + "_Vt");
            d.thermalVoltage_V_T_V = (pVt != semanticAnalyzer_.parameters.end()) ? pVt->second : 0.02585;
            block_.diodes.push_back(d);
        }
        else {
            // Current Source fallback
            try {
                double I = evaluateConstant(*node.rhs);
                static int csId = 1;
                CurrentSource cs;
                cs.nodePositive = node1;
                cs.nodeNegative = node2;
                cs.current_A = I;
                block_.currentSources.push_back(cs);
            } catch (...) {}
        }
    } 
    else if (node.nature == "V") {
        double L;
        if (isInductorPattern(*node.rhs, L)) {
            static int lId = 1;
            Inductor ind;
            ind.name = "L" + std::to_string(lId++);
            ind.nodeCoil1 = node1;
            ind.nodeCoil2 = node2;
            ind.inductance_henries = L;
            block_.inductors.push_back(ind);
        }
        else {
            // Voltage Source fallback
            try {
                double V = evaluateConstant(*node.rhs);
                static int vsId = 1;
                VoltageSource vs;
                vs.nodePositive = node1;
                vs.nodeNegative = node2;
                vs.voltage_V = V;
                block_.voltageSources.push_back(vs);
            } catch (...) {}
        }
    }
}

} // namespace veriloga
} // namespace physics
} // namespace acutesim
