/**
 * spice_parser.cpp — Full SPICE3f5 Netlist Parser Implementation
 *
 * Converts standard SPICE netlists into the TensorNetlist hierarchy.
 * This replaces the rudimentary deserialiseRequest() in engine_impl.cpp.
 */

#include "spice_parser.h"
#include <sstream>
#include <cctype>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace acutesim {

// ============================================================================
// Utility Helpers
// ============================================================================

std::string SPICEParser::toUpper(const std::string& s) {
    std::string r = s;
    for (auto& c : r) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return r;
}

std::vector<std::string> SPICEParser::tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string tok;

    // Handle parenthesized waveform args: PULSE(v1 v2 td ...) → flatten
    std::string flat = line;
    for (auto& c : flat) {
        if (c == '(' || c == ')' || c == ',' || c == '=') c = ' ';
    }

    std::istringstream flatStream(flat);
    while (flatStream >> tok) tokens.push_back(tok);
    return tokens;
}

double SPICEParser::parseValue(const std::string& s) {
    if (s.empty()) return 0.0;

    // Check if it's a known parameter reference
    // (handled externally; this is pure numeric parsing)

    size_t end = 0;
    double v = 0.0;
    try {
        v = std::stod(s, &end);
    } catch (...) {
        return 0.0;
    }

    if (end < s.size()) {
        std::string suffix = s.substr(end);
        std::string sfxUpper = toUpper(suffix);

        if (sfxUpper == "MEG")       v *= 1e6;
        else if (sfxUpper == "MIL")  v *= 25.4e-6; // mils to meters
        else {
            char c = static_cast<char>(std::tolower(static_cast<unsigned char>(suffix[0])));
            switch (c) {
                case 't': v *= 1e12;  break;
                case 'g': v *= 1e9;   break;
                case 'k': v *= 1e3;   break;
                case 'm': v *= 1e-3;  break;
                case 'u': v *= 1e-6;  break;
                case 'n': v *= 1e-9;  break;
                case 'p': v *= 1e-12; break;
                case 'f': v *= 1e-15; break;
                case 'a': v *= 1e-18; break;
                default: break;
            }
        }
    }
    return v;
}

int SPICEParser::nodeIndex(const std::string& name) {
    std::string upper = toUpper(name);
    if (upper == "0" || upper == "GND" || upper == "VSS" || upper == "GROUND")
        return 0;

    auto it = nodeMap_.find(upper);
    if (it != nodeMap_.end()) return it->second;

    int idx = nextNode_++;
    nodeMap_[upper] = idx;
    return idx;
}

// ============================================================================
// Line Pre-Processing (continuation, comments, blank)
// ============================================================================

std::vector<std::pair<int, std::string>> SPICEParser::preprocessLines(
    const std::string& text)
{
    // Split into raw lines, merge continuation lines ('+' prefix)
    std::vector<std::pair<int, std::string>> raw;
    {
        std::istringstream stream(text);
        std::string line;
        int lineNum = 0;
        while (std::getline(stream, line)) {
            ++lineNum;
            raw.push_back({lineNum, line});
        }
    }

    std::vector<std::pair<int, std::string>> merged;
    for (size_t i = 0; i < raw.size(); ++i) {
        auto& [num, line] = raw[i];

        // Strip inline comment (';')
        auto semiPos = line.find(';');
        if (semiPos != std::string::npos)
            line = line.substr(0, semiPos);

        // Strip trailing whitespace
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back())))
            line.pop_back();

        if (line.empty()) continue;

        // Comment line
        if (line[0] == '*') continue;

        // Continuation line
        if (line[0] == '+' && !merged.empty()) {
            merged.back().second += " " + line.substr(1);
            continue;
        }

        merged.push_back({num, line});
    }
    return merged;
}

// ============================================================================
// Main Parse Entry Point
// ============================================================================

SPICEParseResult SPICEParser::parse(const std::string& netlistText) {
    result_ = SPICEParseResult{};
    nodeMap_.clear();
    nextNode_ = 1;
    inSubckt_ = false;
    modelCards_.clear();
    params_.clear();
    inductorIndex_.clear();

    auto lines = preprocessLines(netlistText);
    if (lines.empty()) {
        result_.success = false;
        result_.errorMessage = "Empty netlist";
        return result_;
    }

    // First non-comment line is the title — skip it
    bool firstLine = true;

    for (const auto& [lineNum, line] : lines) {
        if (firstLine) { firstLine = false; continue; }

        auto tok = tokenize(line);
        if (tok.empty()) continue;

        result_.linesProcessed++;
        std::string keyword = toUpper(tok[0]);

        try {
            // Directives start with '.'
            if (keyword[0] == '.') {
                if (keyword == ".MODEL")  parseModelCard(tok);
                else if (keyword == ".SUBCKT") parseSubcktBegin(tok);
                else if (keyword == ".ENDS")   parseSubcktEnd();
                else if (keyword == ".PARAM")  parseParam(tok);
                else if (keyword == ".OP")     result_.hasOP = true;
                else if (keyword == ".TRAN") {
                    result_.hasTRAN = true;
                    if (tok.size() >= 3) {
                        result_.tranStep = parseValue(tok[1]);
                        result_.tranStop = parseValue(tok[2]);
                        if (tok.size() >= 4) result_.tranStart = parseValue(tok[3]);
                    }
                }
                else if (keyword == ".AC")  result_.hasAC = true;
                else if (keyword == ".DC")  result_.hasDC = true;
                else if (keyword == ".END") break;
                // .INCLUDE, .LIB, .OPTION, .GLOBAL — silently skip for now
                continue;
            }

            // Choose target block: subcircuit body or global
            TensorBlock& blk = inSubckt_ ? subcktBlock_ : result_.netlist.globalBlock;
            char type = static_cast<char>(std::toupper(static_cast<unsigned char>(keyword[0])));

            switch (type) {
                case 'R': parseResistor(tok, blk);   break;
                case 'C': parseCapacitor(tok, blk);   break;
                case 'L': parseInductor(tok, blk);    break;
                case 'K': parseMutualInd(tok, blk);   break;
                case 'V': parseVoltSrc(tok, blk);     break;
                case 'I': parseCurrSrc(tok, blk);     break;
                case 'D': parseDiode(tok, blk);       break;
                case 'M': parseMosfet(tok, blk);      break;
                case 'Q': parseBJT(tok, blk);         break;
                case 'J': parseJFET(tok, blk);        break;
                case 'E': parseVCVS(tok, blk);        break;
                case 'G': parseVCCS(tok, blk);        break;
                case 'H': parseCCVS(tok, blk);        break;
                case 'F': parseCCCS(tok, blk);        break;
                case 'T': parseTLine(tok, blk);       break;
                case 'X': {
                    // Sub-circuit instance: X<name> <nodes...> <subcktName>
                    if (tok.size() >= 3) {
                        std::string subcktRef = toUpper(tok.back());
                        std::vector<int> nodeMapping;
                        for (size_t i = 1; i < tok.size() - 1; ++i) {
                            nodeMapping.push_back(nodeIndex(tok[i]));
                        }
                        result_.netlist.addInstance(subcktRef, nodeMapping);
                    }
                    break;
                }
                default:
                    // Unknown element — skip with warning
                    std::cerr << "[SPICEParser] Warning: Unknown element '" 
                              << keyword << "' at line " << lineNum << "\n";
                    break;
            }
            result_.elementsCreated++;
        } catch (const std::exception& e) {
            result_.success = false;
            result_.errorMessage = std::string("Parse error at line ") +
                                   std::to_string(lineNum) + ": " + e.what();
            result_.errorLine = lineNum;
            return result_;
        }
    }

    // Finalize: compute numGlobalNodes from nodeMap (max index)
    for (const auto& [name, idx] : nodeMap_) {
        if (idx > result_.netlist.numGlobalNodes)
            result_.netlist.numGlobalNodes = idx;
    }

    // Merge model cards into netlist
    for (auto& [name, card] : modelCards_) {
        result_.netlist.addModelCard(card);
        result_.modelsCreated++;
    }

    // Initialize global state
    result_.netlist.globalState.initialize(result_.netlist.globalBlock);

    return result_;
}

// ============================================================================
// Element Parsers
// ============================================================================

// R<name> <n+> <n-> <value> [model] [W=<width>] [L=<length>]
void SPICEParser::parseResistor(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 4) return;
    Resistor r;
    r.name = tok[0];
    r.nodeTerminal1 = nodeIndex(tok[1]);
    r.nodeTerminal2 = nodeIndex(tok[2]);
    r.resistance_ohms = parseValue(tok[3]);

    // Optional parameters
    for (size_t i = 4; i < tok.size(); ++i) {
        std::string upper = toUpper(tok[i]);
        if (upper.substr(0, 2) == "W=") r.width_m = parseValue(tok[i].substr(2));
        else if (upper.substr(0, 2) == "L=") r.length_m = parseValue(tok[i].substr(2));
        else if (upper.substr(0, 4) == "TC1=") r.tc1 = parseValue(tok[i].substr(4));
        else if (upper.substr(0, 4) == "TC2=") r.tc2 = parseValue(tok[i].substr(4));
    }
    blk.resistors.push_back(r);
}

// C<name> <n+> <n-> <value> [model] [W=] [L=] [M=]
void SPICEParser::parseCapacitor(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 4) return;
    Capacitor c;
    c.name = tok[0];
    c.nodePlate1 = nodeIndex(tok[1]);
    c.nodePlate2 = nodeIndex(tok[2]);
    c.capacitance_farads = parseValue(tok[3]);

    for (size_t i = 4; i < tok.size(); ++i) {
        std::string upper = toUpper(tok[i]);
        if (upper.substr(0, 2) == "W=") c.width_m = parseValue(tok[i].substr(2));
        else if (upper.substr(0, 2) == "L=") c.length_m = parseValue(tok[i].substr(2));
        else if (upper.substr(0, 2) == "M=") c.m = parseValue(tok[i].substr(2));
    }
    blk.capacitors.push_back(c);
}

// L<name> <n+> <n-> <value>
void SPICEParser::parseInductor(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 4) return;
    Inductor ind;
    ind.name = tok[0];
    ind.nodeCoil1 = nodeIndex(tok[1]);
    ind.nodeCoil2 = nodeIndex(tok[2]);
    ind.inductance_henries = parseValue(tok[3]);

    // Track inductor by name for K-statement resolution
    inductorIndex_[toUpper(tok[0])] = static_cast<int>(blk.inductors.size());

    blk.inductors.push_back(ind);
}

// K<name> <L1name> <L2name> <coupling>
void SPICEParser::parseMutualInd(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 4) return;
    MutualInductor mi;
    std::string l1 = toUpper(tok[1]);
    std::string l2 = toUpper(tok[2]);

    auto it1 = inductorIndex_.find(l1);
    auto it2 = inductorIndex_.find(l2);
    if (it1 == inductorIndex_.end() || it2 == inductorIndex_.end()) {
        std::cerr << "[SPICEParser] Warning: K-statement references unknown inductor(s)\n";
        return;
    }

    mi.inductor1_index = it1->second;
    mi.inductor2_index = it2->second;
    mi.couplingCoefficient_k = parseValue(tok[3]);
    blk.mutualInductors.push_back(mi);
}

// V<name> <n+> <n-> [DC <val>] [AC <mag> [<phase>]] [PULSE(...)] [SIN(...)]
void SPICEParser::parseVoltSrc(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 3) return;
    VoltageSource vs;
    vs.nodePositive = nodeIndex(tok[1]);
    vs.nodeNegative = nodeIndex(tok[2]);

    size_t i = 3;
    while (i < tok.size()) {
        std::string kw = toUpper(tok[i]);

        if (kw == "DC" && i + 1 < tok.size()) {
            vs.voltage_V = parseValue(tok[i + 1]);
            vs.type = "DC";
            i += 2;
        } else if (kw == "AC" && i + 1 < tok.size()) {
            vs.ac_mag = parseValue(tok[i + 1]);
            if (i + 2 < tok.size()) {
                // Check if next token is a number (phase) or a keyword
                try {
                    vs.ac_phase = parseValue(tok[i + 2]);
                    i += 3;
                } catch (...) {
                    i += 2;
                }
            } else {
                i += 2;
            }
        } else if (kw == "PULSE") {
            vs.type = "PULSE";
            parsePulse(tok, i + 1, vs);
            break; // PULSE consumes rest of tokens
        } else if (kw == "SIN") {
            vs.type = "SIN";
            parseSine(tok, i + 1, vs);
            break;
        } else {
            // Bare value (e.g., "V1 1 0 5")
            vs.voltage_V = parseValue(tok[i]);
            vs.type = "DC";
            i++;
        }
    }
    blk.voltageSources.push_back(vs);
}

void SPICEParser::parsePulse(const std::vector<std::string>& tok, size_t idx,
                              VoltageSource& vs) {
    // PULSE V1 V2 TD TR TF PW PER
    if (idx < tok.size())     vs.pulse_v1  = parseValue(tok[idx++]);
    if (idx < tok.size())     vs.pulse_v2  = parseValue(tok[idx++]);
    if (idx < tok.size())     vs.pulse_td  = parseValue(tok[idx++]);
    if (idx < tok.size())     vs.pulse_tr  = parseValue(tok[idx++]);
    if (idx < tok.size())     vs.pulse_tf  = parseValue(tok[idx++]);
    if (idx < tok.size())     vs.pulse_pw  = parseValue(tok[idx++]);
    if (idx < tok.size())     vs.pulse_per = parseValue(tok[idx++]);
}

void SPICEParser::parseSine(const std::vector<std::string>& tok, size_t idx,
                             VoltageSource& vs) {
    // SIN VO VA FREQ TD THETA PHASE
    if (idx < tok.size())     vs.sine_vo    = parseValue(tok[idx++]);
    if (idx < tok.size())     vs.sine_va    = parseValue(tok[idx++]);
    if (idx < tok.size())     vs.sine_freq  = parseValue(tok[idx++]);
    if (idx < tok.size())     vs.sine_td    = parseValue(tok[idx++]);
    if (idx < tok.size())     vs.sine_theta = parseValue(tok[idx++]);
    if (idx < tok.size())     vs.sine_phase = parseValue(tok[idx++]);
}

// I<name> <n+> <n-> [DC <val>] [PULSE(...)] [SIN(...)]
void SPICEParser::parseCurrSrc(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 3) return;
    CurrentSource cs;
    cs.nodePositive = nodeIndex(tok[1]);
    cs.nodeNegative = nodeIndex(tok[2]);

    size_t i = 3;
    while (i < tok.size()) {
        std::string kw = toUpper(tok[i]);

        if (kw == "DC" && i + 1 < tok.size()) {
            cs.current_A = parseValue(tok[i + 1]);
            cs.type = "DC";
            i += 2;
        } else if (kw == "PULSE") {
            cs.type = "PULSE";
            parsePulseI(tok, i + 1, cs);
            break;
        } else if (kw == "SIN") {
            cs.type = "SIN";
            parseSineI(tok, i + 1, cs);
            break;
        } else {
            cs.current_A = parseValue(tok[i]);
            cs.type = "DC";
            i++;
        }
    }
    blk.currentSources.push_back(cs);
}

void SPICEParser::parsePulseI(const std::vector<std::string>& tok, size_t idx,
                               CurrentSource& cs) {
    if (idx < tok.size())     cs.pulse_v1  = parseValue(tok[idx++]);
    if (idx < tok.size())     cs.pulse_v2  = parseValue(tok[idx++]);
    if (idx < tok.size())     cs.pulse_td  = parseValue(tok[idx++]);
    if (idx < tok.size())     cs.pulse_tr  = parseValue(tok[idx++]);
    if (idx < tok.size())     cs.pulse_tf  = parseValue(tok[idx++]);
    if (idx < tok.size())     cs.pulse_pw  = parseValue(tok[idx++]);
    if (idx < tok.size())     cs.pulse_per = parseValue(tok[idx++]);
}

void SPICEParser::parseSineI(const std::vector<std::string>& tok, size_t idx,
                              CurrentSource& cs) {
    if (idx < tok.size())     cs.sine_vo    = parseValue(tok[idx++]);
    if (idx < tok.size())     cs.sine_va    = parseValue(tok[idx++]);
    if (idx < tok.size())     cs.sine_freq  = parseValue(tok[idx++]);
    if (idx < tok.size())     cs.sine_td    = parseValue(tok[idx++]);
    if (idx < tok.size())     cs.sine_theta = parseValue(tok[idx++]);
    if (idx < tok.size())     cs.sine_phase = parseValue(tok[idx++]);
}

// D<name> <n+> <n-> <modelName> [AREA=<val>]
void SPICEParser::parseDiode(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 4) return;
    Diode d;
    d.instanceName = tok[0];
    d.anode = nodeIndex(tok[1]);
    d.cathode = nodeIndex(tok[2]);
    d.modelName = tok[3];

    for (size_t i = 4; i < tok.size(); ++i) {
        std::string upper = toUpper(tok[i]);
        if (upper.substr(0, 5) == "AREA=")
            d.area_m2 = parseValue(tok[i].substr(5));
    }
    blk.diodes.push_back(d);
}

// M<name> <nd> <ng> <ns> <nb> <modelName> [W=] [L=] [AD=] [AS=] [PD=] [PS=] [NRD=] [NRS=] [NF=] [M=]
void SPICEParser::parseMosfet(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 6) return;
    Mosfet m;
    m.instanceName = tok[0];
    m.drain  = nodeIndex(tok[1]);
    m.gate   = nodeIndex(tok[2]);
    m.source = nodeIndex(tok[3]);
    m.body   = nodeIndex(tok[4]);
    m.modelName = tok[5];

    for (size_t i = 6; i < tok.size(); ++i) {
        std::string upper = toUpper(tok[i]);
        if (upper.substr(0, 2) == "W=")       m.w = parseValue(tok[i].substr(2));
        else if (upper.substr(0, 2) == "L=")   m.l = parseValue(tok[i].substr(2));
        else if (upper.substr(0, 3) == "AD=")  m.geo.ad = parseValue(tok[i].substr(3));
        else if (upper.substr(0, 3) == "AS=")  m.geo.as = parseValue(tok[i].substr(3));
        else if (upper.substr(0, 3) == "PD=")  m.geo.pd = parseValue(tok[i].substr(3));
        else if (upper.substr(0, 3) == "PS=")  m.geo.ps = parseValue(tok[i].substr(3));
        else if (upper.substr(0, 4) == "NRD=") m.geo.nrd = parseValue(tok[i].substr(4));
        else if (upper.substr(0, 4) == "NRS=") m.geo.nrs = parseValue(tok[i].substr(4));
        else if (upper.substr(0, 3) == "NF=")  m.extra.nf = static_cast<int>(parseValue(tok[i].substr(3)));
        else if (upper.substr(0, 2) == "M=")   m.extra.mult = static_cast<int>(parseValue(tok[i].substr(2)));
    }
    blk.mosfets.push_back(m);
}

// Q<name> <nc> <nb> <ne> [<ns>] <modelName> [AREA=] [M=]
void SPICEParser::parseBJT(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 5) return;
    BJT q;
    q.instanceName = tok[0];
    q.nodeCollector = nodeIndex(tok[1]);
    q.base = nodeIndex(tok[2]);
    q.emitter = nodeIndex(tok[3]);

    // The model name could be at index 4 or 5 (optional substrate node)
    size_t modelIdx = 4;
    // Heuristic: if tok[4] doesn't look like a model name (starts with letter, no '='),
    // check if it could be a substrate node
    if (tok.size() >= 6) {
        // Try to determine if tok[4] is a node (numeric or named node) or a model name
        // Convention: model names don't start with a digit
        bool tok4_is_node = std::isdigit(static_cast<unsigned char>(tok[4][0]));
        if (tok4_is_node) {
            // Q1 nc nb ne ns modelName
            modelIdx = 5;
        }
    }

    if (modelIdx < tok.size()) {
        q.modelName = tok[modelIdx];
        // Resolve NPN/PNP from model card if available
        auto it = modelCards_.find(toUpper(q.modelName));
        if (it != modelCards_.end()) {
            q.isNPN = (toUpper(it->second.type) == "NPN");
        }
    }
    blk.bjts.push_back(q);
}

// J<name> <nd> <ng> <ns> <modelName>
void SPICEParser::parseJFET(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 5) return;
    JFET j;
    j.instanceName = tok[0];
    j.drain  = nodeIndex(tok[1]);
    j.gate   = nodeIndex(tok[2]);
    j.source = nodeIndex(tok[3]);
    j.modelName = tok[4];

    auto it = modelCards_.find(toUpper(j.modelName));
    if (it != modelCards_.end()) {
        j.isNChannel = (toUpper(it->second.type) == "NJF" ||
                        toUpper(it->second.type) == "NJFET");
    }
    blk.jfets.push_back(j);
}

// E<name> <n+> <n-> <nc+> <nc-> <gain>
void SPICEParser::parseVCVS(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 6) return;
    VCVS e;
    e.outPos  = nodeIndex(tok[1]);
    e.outNeg  = nodeIndex(tok[2]);
    e.ctrlPos = nodeIndex(tok[3]);
    e.ctrlNeg = nodeIndex(tok[4]);
    e.gain    = parseValue(tok[5]);
    blk.voltageControlledVoltageSources.push_back(e);
}

// G<name> <n+> <n-> <nc+> <nc-> <gm>
void SPICEParser::parseVCCS(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 6) return;
    VCCS g;
    g.outPos  = nodeIndex(tok[1]);
    g.outNeg  = nodeIndex(tok[2]);
    g.ctrlPos = nodeIndex(tok[3]);
    g.ctrlNeg = nodeIndex(tok[4]);
    g.gm      = parseValue(tok[5]);
    blk.voltageControlledCurrentSources.push_back(g);
}

// H<name> <n+> <n-> <Vctrl> <rm>
void SPICEParser::parseCCVS(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 5) return;
    CCVS h;
    h.outPos  = nodeIndex(tok[1]);
    h.outNeg  = nodeIndex(tok[2]);
    // tok[3] = controlling voltage source name (linked externally)
    h.ctrlPos = 0; // Placeholder: resolved at stamp time via source name lookup
    h.ctrlNeg = 0;
    h.rm      = parseValue(tok[4]);
    blk.currentControlledVoltageSources.push_back(h);
}

// F<name> <n+> <n-> <Vctrl> <alpha>
void SPICEParser::parseCCCS(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 5) return;
    CCCS f;
    f.outPos  = nodeIndex(tok[1]);
    f.outNeg  = nodeIndex(tok[2]);
    f.ctrlPos = 0;
    f.ctrlNeg = 0;
    f.alpha   = parseValue(tok[4]);
    blk.currentControlledCurrentSources.push_back(f);
}

// T<name> <n1+> <n1-> <n2+> <n2-> Z0=<val> TD=<val>
void SPICEParser::parseTLine(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() < 5) return;
    TransmissionLine t;
    t.nodePort1Pos = nodeIndex(tok[1]);
    t.nodePort1Neg = nodeIndex(tok[2]);
    t.nodePort2Pos = nodeIndex(tok[3]);
    t.nodePort2Neg = nodeIndex(tok[4]);

    for (size_t i = 5; i < tok.size(); ++i) {
        std::string upper = toUpper(tok[i]);
        if (upper.substr(0, 3) == "Z0=")
            t.characteristicImpedance_Z0_ohms = parseValue(tok[i].substr(3));
        else if (upper.substr(0, 3) == "TD=")
            t.propagationDelay_Td_s = parseValue(tok[i].substr(3));
    }
    blk.transmissionLines.push_back(t);
}

// ============================================================================
// Directive Parsers
// ============================================================================

// .MODEL <name> <type> ([param=val]...)
void SPICEParser::parseModelCard(const std::vector<std::string>& tok) {
    if (tok.size() < 3) return;
    ModelCard card;
    card.name = tok[1];
    card.type = toUpper(tok[2]);

    // Parse optional parameters (key=value pairs flattened by tokenizer)
    for (size_t i = 3; i + 1 < tok.size(); i += 2) {
        std::string key = toUpper(tok[i]);
        double val = parseValue(tok[i + 1]);
        card.params[key] = val;

        // Extract well-known fields
        if (key == "LEVEL") card.level = static_cast<int>(val);
    }

    modelCards_[toUpper(card.name)] = card;
}

// .SUBCKT <name> <port1> <port2> ...
void SPICEParser::parseSubcktBegin(const std::vector<std::string>& tok) {
    if (tok.size() < 2) return;
    inSubckt_ = true;
    subcktName_ = toUpper(tok[1]);
    subcktBlock_ = TensorBlock{};
    subcktBlock_.name = subcktName_;

    subcktPorts_.clear();
    for (size_t i = 2; i < tok.size(); ++i) {
        subcktPorts_.push_back(tok[i]);
        subcktBlock_.pinNames.push_back(tok[i]);
        subcktBlock_.portNodeIndices.push_back(nodeIndex(tok[i]));
    }
}

void SPICEParser::parseSubcktEnd() {
    if (!inSubckt_) return;
    inSubckt_ = false;

    result_.netlist.defineBlock(subcktBlock_);
    result_.subcircuitsCreated++;
}

// .PARAM <name>=<value> [<name>=<value>]...
void SPICEParser::parseParam(const std::vector<std::string>& tok) {
    // After tokenizer: .PARAM key value key value ...
    for (size_t i = 1; i + 1 < tok.size(); i += 2) {
        params_[toUpper(tok[i])] = parseValue(tok[i + 1]);
    }
}

} // namespace acutesim
