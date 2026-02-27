#pragma once
#include <vector>
#include <string>
#include <memory>
#include <complex>
#include <unordered_map>
#include <functional>

namespace acutesim::compute::postproc {

// ─── Data Types ──────────────────────────────────────────────────────────────

struct Waveform {
    std::vector<double> time;
    std::vector<double> values;
    // For AC:
    std::vector<std::complex<double>> complexValues;
    bool isComplex = false;

    std::string name;
    std::string unit;
};

struct WaveFamily {
    std::vector<Waveform> members;
    std::vector<double> sweepValues; // e.g. Rload=1k, 2k, 3k
    std::string sweepParam;          // "Rload"

    bool isEmpty() const { return members.empty(); }
};

enum class OperandType { Scalar, Waveform, Family, ComplexScalar };

struct Operand {
    OperandType type = OperandType::Scalar;
    double scalarVal = 0.0;
    std::complex<double> complexVal = 0.0;
    Waveform wave;
    WaveFamily family;
};

// ─── AST Node ────────────────────────────────────────────────────────────────

struct AstNode {
    enum Type {
        Literal,      // 3.14
        Identifier,   // v("/out")
        Function,     // dB20(...)
        Operator      // +, -, *, /
    } type = Literal;

    std::string value; // func name or literal string
    std::vector<std::shared_ptr<AstNode>> children;
};

// ─── Engine ──────────────────────────────────────────────────────────────────

class PostProcEngine {
public:
    PostProcEngine();

    // Register data context (available waveforms)
    void setWaveforms(const std::unordered_map<std::string, Waveform>& waves);
    void setFamilies(const std::unordered_map<std::string, WaveFamily>& families);

    // Evaluate expression string
    Operand evaluate(const std::string& expr);

    // Evaluate pre-built AST
    Operand evaluate(const std::shared_ptr<AstNode>& node);

private:
    std::unordered_map<std::string, Waveform> waves_;
    std::unordered_map<std::string, WaveFamily> families_;

    // Core math dispatch
    Operand dispatchFunction(const std::string& func, const std::vector<Operand>& args);
    Operand dispatchOperator(const std::string& op, const Operand& a, const Operand& b);

    // Vectorized Operations
    Waveform mathOp(const Waveform& a, const Waveform& b, std::function<double(double,double)> op);
    Waveform mathFunc(const Waveform& a, std::function<double(double)> op);
};

} // namespace acutesim::compute::postproc
