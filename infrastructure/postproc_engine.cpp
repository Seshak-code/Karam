#include "postproc_engine.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cstdlib>

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

// ─── AST Parsing (Recursive Descent) ─────────────────────────────────────────
//   expr    := term (('+'|'-') term)*
//   term    := unary (('*'|'/') unary)*
//   unary   := '-' unary | primary
//   primary := number | "string" | name '(' expr {',' expr} ')' | '(' expr ')'
// v(...) / i(...) become Identifier nodes holding the canonical call text,
// which is the key waveforms are registered under (e.g. v("/out"), v(node1)).

namespace {
struct Parser {
    const std::string& s;
    size_t pos = 0;

    void ws() { while (pos < s.size() && std::isspace((unsigned char)s[pos])) ++pos; }
    bool eat(char c) { ws(); if (pos < s.size() && s[pos] == c) { ++pos; return true; } return false; }

    static std::shared_ptr<AstNode> make(AstNode::Type t, std::string v,
                                         std::vector<std::shared_ptr<AstNode>> kids = {}) {
        auto n = std::make_shared<AstNode>();
        n->type = t; n->value = std::move(v); n->children = std::move(kids);
        return n;
    }

    std::shared_ptr<AstNode> expr() {
        auto lhs = term();
        for (;;) {
            if (eat('+'))      lhs = make(AstNode::Operator, "+", {lhs, term()});
            else if (eat('-')) lhs = make(AstNode::Operator, "-", {lhs, term()});
            else return lhs;
        }
    }

    std::shared_ptr<AstNode> term() {
        auto lhs = unary();
        for (;;) {
            if (eat('*'))      lhs = make(AstNode::Operator, "*", {lhs, unary()});
            else if (eat('/')) lhs = make(AstNode::Operator, "/", {lhs, unary()});
            else return lhs;
        }
    }

    std::shared_ptr<AstNode> unary() {
        if (eat('-')) return make(AstNode::Operator, "*", {make(AstNode::Literal, "-1"), unary()});
        return primary();
    }

    std::shared_ptr<AstNode> primary() {
        ws();
        if (pos >= s.size()) return make(AstNode::Literal, "");
        if (eat('(')) { auto e = expr(); eat(')'); return e; }
        if (s[pos] == '"') {
            size_t close = s.find('"', pos + 1);
            if (close == std::string::npos) close = s.size();
            std::string str = s.substr(pos + 1, close - pos - 1);
            pos = std::min(close + 1, s.size());
            return make(AstNode::Literal, str);
        }
        if (std::isdigit((unsigned char)s[pos]) || s[pos] == '.') {
            const char* begin = s.c_str() + pos;
            char* stop = nullptr;
            std::strtod(begin, &stop);
            size_t len = std::max<size_t>(stop - begin, 1);
            std::string num = s.substr(pos, len);
            pos += len;
            return make(AstNode::Literal, num);
        }
        size_t start = pos;
        while (pos < s.size() && (std::isalnum((unsigned char)s[pos]) || s[pos] == '_')) ++pos;
        std::string name = s.substr(start, pos - start);
        if (name.empty()) { ++pos; return make(AstNode::Literal, ""); } // skip junk char
        if (!eat('(')) return make(AstNode::Identifier, name);

        if (name == "v" || name == "i") {
            size_t close = s.find(')', pos);
            if (close == std::string::npos) close = s.size();
            std::string arg = trimStr(s.substr(pos, close - pos));
            pos = std::min(close + 1, s.size());
            return make(AstNode::Identifier, name + "(" + arg + ")");
        }
        auto fn = make(AstNode::Function, name);
        if (!eat(')')) {
            do { fn->children.push_back(expr()); } while (eat(','));
            eat(')');
        }
        return fn;
    }
};
} // namespace

static std::shared_ptr<AstNode> parseExpression(const std::string& expr) {
    Parser p{expr};
    return p.expr();
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
        auto it = waves_.find(node->value);
        if (it != waves_.end()) {
            res.type = OperandType::Waveform;
            res.wave = it->second;
        }
    }
    else if (node->type == AstNode::Function) {
        std::vector<Operand> args;
        for (auto& c : node->children) args.push_back(evaluate(c));
        return dispatchFunction(node->value, args);
    }
    else if (node->type == AstNode::Operator && node->children.size() == 2) {
        return dispatchOperator(node->value, evaluate(node->children[0]), evaluate(node->children[1]));
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

Operand PostProcEngine::dispatchOperator(const std::string& op, const Operand& a, const Operand& b) {
    std::function<double(double, double)> f;
    if      (op == "+") f = [](double x, double y) { return x + y; };
    else if (op == "-") f = [](double x, double y) { return x - y; };
    else if (op == "*") f = [](double x, double y) { return x * y; };
    else if (op == "/") f = [](double x, double y) { return x / y; };
    else return {};

    Operand r;
    const bool wa = a.type == OperandType::Waveform, wb = b.type == OperandType::Waveform;
    if (wa && wb) {
        r.type = OperandType::Waveform;
        r.wave = mathOp(a.wave, b.wave, f);
    } else if (wa || wb) {
        const Waveform& w = wa ? a.wave : b.wave;
        const double k = wa ? b.scalarVal : a.scalarVal;
        r.type = OperandType::Waveform;
        r.wave = wa ? mathFunc(w, [&](double x) { return f(x, k); })
                    : mathFunc(w, [&](double x) { return f(k, x); });
    } else {
        r.scalarVal = f(a.scalarVal, b.scalarVal);
    }
    if (r.type == OperandType::Waveform)
        r.wave.name = (wa ? a.wave.name : std::to_string(a.scalarVal)) + op +
                      (wb ? b.wave.name : std::to_string(b.scalarVal));
    return r;
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

// ponytail: pointwise by index, assumes a shared time base; truncates to the
// shorter wave. Upgrade path: interpolate b onto a.time when the bases differ.
Waveform PostProcEngine::mathOp(const Waveform& a, const Waveform& b, std::function<double(double,double)> op) {
    Waveform out;
    const size_t n = std::min(a.values.size(), b.values.size());
    out.time.assign(a.time.begin(), a.time.begin() + std::min(n, a.time.size()));
    out.unit = a.unit;
    out.values.resize(n);
    for (size_t i = 0; i < n; ++i) out.values[i] = op(a.values[i], b.values[i]);
    return out;
}

} // namespace acutesim::compute::postproc
