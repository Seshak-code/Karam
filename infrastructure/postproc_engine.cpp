#include "postproc_engine.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace acutesim::compute::postproc {

// ── String Helpers ────────────────────────────────────────────────────────────

static std::string trimStr(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return {};
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static bool strToDouble(const std::string& s, double& out) {
    try {
        size_t pos = 0;
        out = std::stod(s, &pos);
        return pos == s.size(); // entire string consumed
    } catch (...) {
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────

PostProcEngine::PostProcEngine() {}

void PostProcEngine::setWaveforms(const std::unordered_map<std::string, Waveform>& waves) {
    waves_ = waves;
}

void PostProcEngine::setFamilies(const std::unordered_map<std::string, WaveFamily>& families) {
    families_ = families;
}

// ─── AST Parsing (Simple Recursive Descent) ──────────────────────────────────

static std::shared_ptr<AstNode> parseExpression(const std::string& expr) {
    auto node = std::make_shared<AstNode>();

    std::string e = trimStr(expr);
    size_t openPos = e.find('(');

    if (openPos != std::string::npos) {
        size_t closePos = e.rfind(')');
        node->type = AstNode::Function;
        node->value = trimStr(e.substr(0, openPos));

        // Single arg support
        std::string argsStr = (closePos != std::string::npos && closePos > openPos)
            ? e.substr(openPos + 1, closePos - openPos - 1)
            : "";
        auto child = parseExpression(argsStr);
        node->children.push_back(child);
    }
    else if (!e.empty() && e.front() == '"') {
        node->type = AstNode::Literal; // String literal for node names v("/out")
        node->value = e.substr(1, e.size() > 2 ? e.size() - 2 : 0);
    }
    else {
        double val;
        if (strToDouble(e, val)) {
            node->type = AstNode::Literal;
            node->value = e;
        } else {
            node->type = AstNode::Identifier;
            node->value = e;
        }
    }
    return node;
}

Operand PostProcEngine::evaluate(const std::string& expr) {
    auto ast = parseExpression(expr);
    return evaluate(ast);
}

Operand PostProcEngine::evaluate(const std::shared_ptr<AstNode>& node) {
    Operand res;
    res.type = OperandType::Scalar;

    if (node->type == AstNode::Literal) {
        double d;
        if (strToDouble(node->value, d)) res.scalarVal = d;
    }
    else if (node->type == AstNode::Identifier) {
        // Variable lookup — not used in current expression language
    }
    else if (node->type == AstNode::Function) {
        std::vector<Operand> args;
        for (auto& c : node->children) {
            // v() / i() act as waveform lookups
            if (node->value == "v" || node->value == "i") {
                auto it = waves_.find(c->value);
                if (it != waves_.end()) {
                    Operand w;
                    w.type = OperandType::Waveform;
                    w.wave = it->second;
                    args.push_back(w);
                }
            } else {
                args.push_back(evaluate(c));
            }
        }

        if (node->value != "v" && node->value != "i")
            return dispatchFunction(node->value, args);
        else if (!args.empty()) return args[0]; // v() returns its lookup result
    }

    return res;
}

// ─── Dispatch ────────────────────────────────────────────────────────────────

Operand PostProcEngine::dispatchFunction(const std::string& func, const std::vector<Operand>& args) {
    if (args.empty()) return {};
    const auto& arg = args[0];

    Operand result;

    // Vectorized Math Dispatcher
    auto applyVec = [&](std::function<double(double)> op) {
        if (arg.type == OperandType::Waveform) {
            result.type = OperandType::Waveform;
            result.wave = mathFunc(arg.wave, op);
            result.wave.name = func + "(" + arg.wave.name + ")";
        } else if (arg.type == OperandType::Scalar) {
            result.type = OperandType::Scalar;
            result.scalarVal = op(arg.scalarVal);
        }
    };

    if (func == "dB20") {
        applyVec([](double x) { return 20.0 * std::log10(std::abs(x) + 1e-12); });
        result.wave.unit = "dB";
    }
    else if (func == "mag") {
        applyVec([](double x) { return std::abs(x); });
    }
    else if (func == "phase") {
        // Needs complex support, shim for now
        applyVec([](double) { return 0.0; });
    }
    else if (func == "average") {
        if (arg.type == OperandType::Waveform) {
            result.type = OperandType::Scalar;
            double sum = 0;
            for (double v : arg.wave.values) sum += v;
            result.scalarVal = arg.wave.values.empty() ? 0.0 : sum / static_cast<double>(arg.wave.values.size());
        }
    }

    return result;
}

// ─── Vector Ops ──────────────────────────────────────────────────────────────

Waveform PostProcEngine::mathFunc(const Waveform& a, std::function<double(double)> op) {
    Waveform out;
    out.time = a.time; // Preserve time base
    out.unit = a.unit;
    out.values.resize(a.values.size());

    const double* src = a.values.data();
    double* dst = out.values.data();
    int N = static_cast<int>(a.values.size());

    for (int i = 0; i < N; ++i) {
        dst[i] = op(src[i]);
    }

    return out;
}

} // namespace acutesim::compute::postproc
