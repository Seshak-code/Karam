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
#include <set>
#include <functional>

namespace acutesim {

// Diagnostics from a parse run.
struct SPICEParseResult {
    TensorNetlist netlist;
    std::map<std::string, int> nodeIndex;  // top-level node name (upper case) -> index
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
    double        tranStop = 0.0;
    double        tranStep = 0.0;
    double        tranStart = 0.0;
    bool          tranUIC = false;   // .tran ... UIC: skip the DC operating point
    std::string   dcSource;          // .dc SRC start stop step ("" = no sweep)
    double        dcStart = 0.0, dcStop = 0.0, dcStep = 0.0;
    std::string   noiseSource;       // .noise v(out[,ref]) src sweep n fstart fstop
    std::string   noiseSweep;
    int           noiseOut = 0, noiseRef = 0, noisePoints = 0;
    double        noiseStart = 0.0, noiseStop = 0.0;
    std::string   acSweep;           // .ac dec|oct|lin n fstart fstop
    int           acPoints = 0;
    double        acStart = 0.0, acStop = 0.0;
};

class SPICEParser {
public:
    SPICEParser() = default;

    /// Parse a complete SPICE netlist string.
    // baseDir: directory .include / .lib paths are relative to (empty: working directory)
    SPICEParseResult parse(const std::string& netlistText, const std::string& baseDir = "");

private:
    // ── Value and Token Helpers ────────────────────────────────────────
    double parseValue(const std::string& s) const;  // throws on non-numbers; resolves .param
    bool isValue(const std::string& s) const;
    static void requireFields(const std::vector<std::string>& tok, size_t n);
    double takeMultiplier(std::vector<std::string>& tok) const;  // removes M <value> pairs; the last wins
    static void requireLinearForm(const std::vector<std::string>& tok, size_t n);
    double instanceArea(const std::vector<std::string>& tok, size_t from) const;
    static std::string toUpper(const std::string& s);
    static std::vector<std::string> tokenize(const std::string& line);

    // ── Node Mapping ──────────────────────────────────────────────────
    int nodeIndex(const std::string& name);

    // ── Line Pre-Processing ───────────────────────────────────────────
    /// Merge continuation lines ('+' prefix), strip comments.
    // Splice .include files and .lib sections into the lines (recursively).
    std::vector<std::pair<int, std::string>> expandIncludes(const std::vector<std::pair<int, std::string>>& lines,
                                                            const std::string& dir, int depth, int& errLine);
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
    void parseOsdiDevice (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseVCVS       (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseVCCS       (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseCCVS       (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseCCCS       (const std::vector<std::string>& tok, TensorBlock& blk);
    void parseTLine      (const std::vector<std::string>& tok, TensorBlock& blk);
    // F/H name their controlling V source; resolve once the block is complete
    void resolveControlledSources(TensorBlock& blk);

    // ── Directive Parsers ─────────────────────────────────────────────
    void parseModelCard  (const std::vector<std::string>& tok);
    void parseSubcktBegin(const std::vector<std::string>& tok);
    void parseSubcktEnd  ();
    void parseParam      (const std::vector<std::string>& tok);

    // ── Source Waveform Helpers ────────────────────────────────────────
    // Shared by V and I sources (identical waveform fields)
    template <class Src>
    void parseSourceSpec(const std::vector<std::string>& tok, Src& s, double& dc) const;
    template <class Src>
    void parsePulse(const std::vector<std::string>& tok, size_t startIdx, Src& s) const;
    template <class Src>
    void parseSine (const std::vector<std::string>& tok, size_t startIdx, Src& s) const;
    template <class Src>
    void parsePwl  (const std::vector<std::string>& tok, size_t startIdx, Src& s) const;

    // ── State ─────────────────────────────────────────────────────────
    std::map<std::string, int> nodeMap_;
    int nextNode_ = 1;
    double mult_ = 1.0;   // m= of the R, L, I, G, F or J card being parsed

    // .subckt parsing state. Bodies use LOCAL node numbers (ports 1..P, then
    // internal nodes); X cards are expanded after the whole file is read, so
    // a subcircuit may be defined after use and may contain X cards itself.
    bool             inSubckt_ = false;
    std::string      subcktName_;
    std::vector<std::string> subcktPorts_;
    TensorBlock      subcktBlock_;
    std::map<std::string, int> subcktNodeMap_;   // local node names of the body being parsed
    int              nextLocalNode_ = 1;
    struct PendingInstance { std::string subckt; std::vector<int> nodes; int line; std::string name; };
    std::vector<PendingInstance> topInstances_;   // nodes are global indices
    struct PendingIC { std::string node; double value; int line; bool nodeset; };
    std::vector<PendingIC> pendingICs_;           // resolved once every node is known
    std::map<std::string, std::vector<PendingIC>> subcktICs_;   // .ic / .nodeset in a body (local names), applied per instance
    std::map<std::string, std::vector<std::string>> subcktNames_;  // subckt -> name of local node l at [l-1]
    std::map<std::string, int> hierNodes_;         // "X1.N" (upper case) -> global node, for subcircuit internals
    std::string noiseOut_, noiseRef_;              // .noise node names, resolved likewise
    std::set<std::string> globals_;                // .global node names
    std::map<std::string, std::map<int, std::string>> subcktGlobals_;  // subckt -> local index -> name
    std::map<std::string, std::vector<PendingInstance>> subcktChildren_;  // nodes are body-local
    std::map<std::string, int> savedInductorIndex_; // top-level K lookup, saved inside a body
    void expandInstance(const PendingInstance& pi, const std::vector<int>& portNodes, int depth, const std::string& path);

    // Model cards collected during parse, merged into netlist at end
    std::map<std::string, ModelCard> modelCards_;
    std::set<std::string> modelNames_;
    std::map<std::string, int> cardLine_;   // "<subckt>/<CARD NAME>" and ".MODEL/<NAME>" -> line (deferred errors)
    std::string resolveAt_;                 // block being resolved
    double scale_ = 1.0;                    // .option scale: MOSFET W, L, PD, PS (and AD, AS squared)                              // every .model name (first pass)
    std::map<std::string, std::shared_ptr<osdi::Model>> osdiModels_; // per card, shared by its N devices

    // .param expressions (name -> value)
    std::map<std::string, double> params_;

    // Inductor name -> index (for K statements)
    std::map<std::string, int> inductorIndex_;

    // Result accumulator
    SPICEParseResult result_;
};

} // namespace acutesim
