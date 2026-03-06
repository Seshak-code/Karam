#pragma once
/**
 * spice_parser.h — Full SPICE3f5 Netlist Parser
 *
 * Converts a SPICE netlist string into a fully populated TensorNetlist.
 *
 * Supported elements:
 *   R, C, L, K (mutual inductor), V, I, D, M (MOSFET), Q (BJT), J (JFET),
 *   E (VCVS), G (VCCS), H (CCVS), F (CCCS), T (transmission line)
 *
 * Supported directives:
 *   .model, .subckt/.ends, .param, .option, .op, .tran, .ac, .dc, .end,
 *   .include (no-op placeholder)
 *
 * Line continuation ('+' prefix) is supported.
 * Comment lines ('*' prefix) and inline comments (';') are stripped.
 * Node "0" (or "gnd"/"GND") maps to ground (index 0).
 */

#include "circuit.h"
#include <string>
#include <vector>
#include <map>
#include <functional>

namespace acutesim {

// Diagnostics from a parse run.
struct SPICEParseResult {
    TensorNetlist netlist;
    bool          success = true;
    std::string   errorMessage;
    int           errorLine = -1;

    // Informational
    int           linesProcessed = 0;
    int           elementsCreated = 0;
    int           modelsCreated = 0;
    int           subcircuitsCreated = 0;

    // Analysis directives discovered
    bool          hasOP = false;
    bool          hasTRAN = false;
    bool          hasAC = false;
    bool          hasDC = false;
    double        tranStop = 0.0;
    double        tranStep = 0.0;
    double        tranStart = 0.0;
};

class SPICEParser {
public:
    SPICEParser() = default;

    /// Parse a complete SPICE netlist string.
    SPICEParseResult parse(const std::string& netlistText);

private:
    // ── Value and Token Helpers ────────────────────────────────────────
    static double parseValue(const std::string& s);
    static std::string toUpper(const std::string& s);
    static std::vector<std::string> tokenize(const std::string& line);

    // ── Node Mapping ──────────────────────────────────────────────────
    int nodeIndex(const std::string& name);

    // ── Line Pre-Processing ───────────────────────────────────────────
    /// Merge continuation lines ('+' prefix), strip comments.
    static std::vector<std::pair<int, std::string>> preprocessLines(
        const std::string& text);

    // ── Element Parsers ───────────────────────────────────────────────
    void parseResistor   (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseCapacitor  (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseInductor   (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseMutualInd  (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseVoltSrc    (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseCurrSrc    (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseDiode      (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseMosfet     (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseBJT        (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseJFET       (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseVCVS       (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseVCCS       (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseCCVS       (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseCCCS       (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseTLine      (const std::vector<std::string>& tok, TensorBlock& blk);

    // ── Directive Parsers ─────────────────────────────────────────────
    void parseModelCard  (const std::vector<std::string>& tok);
    void parseSubcktBegin(const std::vector<std::string>& tok);
    void parseSubcktEnd  ();
    void parseParam      (const std::vector<std::string>& tok);

    // ── Source Waveform Helpers ────────────────────────────────────────
    static void parsePulse(const std::vector<std::string>& tok, size_t startIdx,
                           VoltageSource& vs);
    static void parseSine (const std::vector<std::string>& tok, size_t startIdx,
                           VoltageSource& vs);
    static void parsePulseI(const std::vector<std::string>& tok, size_t startIdx,
                            CurrentSource& cs);
    static void parseSineI (const std::vector<std::string>& tok, size_t startIdx,
                            CurrentSource& cs);

    // ── State ─────────────────────────────────────────────────────────
    std::map<std::string, int> nodeMap_;
    int nextNode_ = 1;

    // .subckt parsing state
    bool             inSubckt_ = false;
    std::string      subcktName_;
    std::vector<std::string> subcktPorts_;
    TensorBlock      subcktBlock_;

    // Model cards collected during parse, merged into netlist at end
    std::map<std::string, ModelCard> modelCards_;

    // .param expressions (name -> value)
    std::map<std::string, double> params_;

    // Inductor name -> index (for K statements)
    std::map<std::string, int> inductorIndex_;

    // Result accumulator
    SPICEParseResult result_;
};

} // namespace acutesim
