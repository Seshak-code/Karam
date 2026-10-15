/**
 * spice_parser.cpp — Full SPICE3f5 Netlist Parser Implementation
 *
 * Converts standard SPICE netlists into the TensorNetlist hierarchy.
 * This replaces the rudimentary deserialiseRequest() in engine_impl.cpp.
 */

#include "spice_parser.h"
#include "numparam.h"
#include "../osdi/osdi_device.h"
#include "../physics/physics_constants.h"
#include <sstream>
#include <fstream>
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

double SPICEParser::parseValue(const std::string& raw) const {
    // .param reference: {NAME} or bare NAME (arithmetic expressions unsupported)
    std::string s = raw;
    if (s.size() >= 2 && s.front() == '{' && s.back() == '}')
        s = s.substr(1, s.size() - 2);
    auto p = params_.find(toUpper(s));
    if (p != params_.end()) return p->second;

    size_t end = 0;
    double v = 0.0;
    try {
        v = std::stod(s, &end);
    } catch (...) {
        // Was "return 0.0": a typo or {expr} silently became 0 (e.g. a 0-ohm R)
        throw std::runtime_error("'" + raw + "' is not a number or a known .param");
    }

    if (end < s.size()) {
        std::string suffix = s.substr(end);
        // a scale factor plus unit letters only: "0.5*R1" used to read as 0.5
        for (char ch : suffix)
            if (!std::isalpha(static_cast<unsigned char>(ch)))
                throw std::runtime_error("'" + raw + "' is not a number or a known .param");
        std::string sfxUpper = toUpper(suffix);

        if (sfxUpper.rfind("MEG", 0) == 0) v *= 1e6;
        else if (sfxUpper.rfind("MIL", 0) == 0) v *= 25.4e-6; // mils to meters
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
                default: break;  // unit letters (V, A, s, ohm) are ignored, as in SPICE
            }
        }
    }
    return v;
}

bool SPICEParser::isValue(const std::string& s) const {
    try { parseValue(s); return true; } catch (...) { return false; }
}

// A card with too few fields used to be silently dropped from the circuit.
void SPICEParser::requireFields(const std::vector<std::string>& tok, size_t n) {
    if (tok.size() < n)
        throw std::runtime_error(tok[0] + ": expected at least " +
                                 std::to_string(n - 1) + " fields after the name");
}

// Instance fields after a D/Q model name: [area] [AREA=a] [M=m]. AREA and M both
// scale the device (currents, capacitances; resistances divide). OFF and IC=
// steer which operating point SPICE finds (ngspice holds OFF junctions at 0 V
// through MODEINITFIX) and the UIC start: errors until they are modelled, not dropped.
double SPICEParser::instanceArea(const std::vector<std::string>& tok, size_t from) const {
    double area = 1.0;
    for (size_t i = from; i < tok.size(); ++i) {
        const std::string k = toUpper(tok[i]);
        if (i == from && isValue(tok[i])) area *= parseValue(tok[i]);
        else if ((k == "AREA" || k == "M") && i + 1 < tok.size() && isValue(tok[i + 1])) area *= parseValue(tok[++i]);
        else if (k == "OFF" || k == "IC") throw std::runtime_error(tok[0] + ": " + k + " is not supported yet");
        else throw std::runtime_error(tok[0] + ": parameter " + k + " is not supported yet");
    }
    if (!(area > 0.0)) throw std::runtime_error(tok[0] + ": AREA and M must be > 0");
    return area;
}

int SPICEParser::nodeIndex(const std::string& name) {
    std::string upper = toUpper(name);
    // Ground is "0" or "GND", as in ngspice, and global (also inside subcircuit
    // bodies). VSS/GROUND used to be ground too: a cell's VSS port became global
    // ground instead of a port, which shifted every later local node of the cell.
    if (upper == "0" || upper == "GND")
        return 0;

    // Inside a .subckt body node names are local to that definition
    auto& map  = inSubckt_ ? subcktNodeMap_ : nodeMap_;
    auto& next = inSubckt_ ? nextLocalNode_ : nextNode_;
    auto it = map.find(upper);
    if (it != map.end()) return it->second;
    if (inSubckt_ && globals_.count(upper)) subcktGlobals_[subcktName_][next] = upper;  // see expandInstance
    map[upper] = next;
    return next++;
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
        // and leading whitespace (PDK files indent their "+" continuation lines)
        line.erase(0, line.find_first_not_of(" \t"));

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

// .include <file> splices a file; .lib <file> <section> splices the lines between
// `.lib <section>` and `.endl` of <file> (PDK corners: .lib models.lib tt). Paths
// are relative to the including file. Spliced lines carry the including card's
// line number, which is also reported (errLine) when a file or section is missing.
std::vector<std::pair<int, std::string>> SPICEParser::expandIncludes(
    const std::vector<std::pair<int, std::string>>& lines, const std::string& dir, int depth, int& errLine)
{
    if (depth > 16) throw std::runtime_error(".include/.lib nested deeper than 16 files (a cycle?)");
    auto unquote = [](std::string s) {
        if (s.size() >= 2 && (s[0] == '"' || s[0] == '\'') && s.back() == s[0]) s = s.substr(1, s.size() - 2);
        return s;
    };
    auto fields = [&](const std::string& l) {  // first three fields, upper-case keyword
        std::istringstream t(l);
        std::vector<std::string> f(4);
        t >> f[0] >> f[1] >> f[2] >> f[3];
        f[0] = toUpper(f[0]);
        return f;
    };
    std::vector<std::pair<int, std::string>> out;
    for (const auto& [num, line] : lines) {
        const auto f = fields(line);
        const bool inc = f[0] == ".INCLUDE" || f[0] == ".INC", lib = f[0] == ".LIB" && !f[2].empty();
        if (!inc && !lib) { out.push_back({num, line}); continue; }
        errLine = num;
        if (f[1].empty() || (inc && !f[2].empty()) || !f[3].empty())
            throw std::runtime_error(inc ? ".include takes one file" : ".lib takes a file and a section");
        const std::string file = unquote(f[1]);
        const std::string path = file[0] == '/' || dir.empty() ? file : dir + "/" + file;
        std::ifstream in(path);
        if (!in) throw std::runtime_error(f[0] + ": cannot open '" + path + "'");
        std::stringstream buf;
        buf << in.rdbuf();
        auto body = preprocessLines(buf.str());
        if (lib) {  // keep .lib <section> ... .endl
            const std::string want = toUpper(unquote(f[2]));
            size_t from = 0, to;
            while (from < body.size() && !(fields(body[from].second)[0] == ".LIB" &&
                                           toUpper(unquote(fields(body[from].second)[1])) == want &&
                                           fields(body[from].second)[2].empty()))
                ++from;
            if (from == body.size()) throw std::runtime_error(".lib: no section " + f[2] + " in '" + path + "'");
            for (to = from + 1; to < body.size() && fields(body[to].second)[0] != ".ENDL"; ++to) {}
            body = {body.begin() + from + 1, body.begin() + to};
        }
        for (auto& b : body) b.first = num;
        const size_t slash = path.rfind('/');
        const auto sub = expandIncludes(body, slash == std::string::npos ? "" : path.substr(0, slash), depth + 1, errLine);
        out.insert(out.end(), sub.begin(), sub.end());
    }
    return out;
}

// ============================================================================
// Main Parse Entry Point
// ============================================================================

SPICEParseResult SPICEParser::parse(const std::string& netlistText, const std::string& baseDir) {
    result_ = SPICEParseResult{};
    nodeMap_.clear();
    nextNode_ = 1;
    inSubckt_ = false;
    topInstances_.clear();
    pendingICs_.clear();
    globals_.clear();
    subcktGlobals_.clear();
    noiseOut_.clear();
    noiseRef_.clear();
    subcktChildren_.clear();
    modelCards_.clear();
    params_.clear();
    inductorIndex_.clear();
    modelNames_.clear();
    osdiModels_.clear();

    auto lines = preprocessLines(netlistText);
    if (lines.empty()) {
        result_.success = false;
        result_.errorMessage = "Empty netlist";
        return result_;
    }

    auto fail = [&](int lineNum, const std::string& what) {
        result_.success = false;
        result_.errorMessage = "Parse error at line " + std::to_string(lineNum) + ": " + what;
        result_.errorLine = lineNum;
        return result_;
    };

    int incLine = 0;
    try {
        lines = expandIncludes(lines, baseDir, 0, incLine);
        lines = numparam::expand(lines, incLine);  // .param expressions, subckt parameters, .if
    } catch (const std::exception& e) { return fail(incLine, e.what()); }

    // .param values are global in SPICE: resolve them before any card uses one
    for (const auto& [lineNum, line] : lines) {
        if (lineNum == 1) continue;
        auto tok = tokenize(line);
        if (!tok.empty() && toUpper(tok[0]) == ".GLOBAL")  // global in every .subckt, wherever declared
            for (size_t i = 1; i < tok.size(); ++i) globals_.insert(toUpper(tok[i]));
        if (!tok.empty() && toUpper(tok[0]) == ".MODEL" && tok.size() > 1)  // N lines need every model name
            modelNames_.insert(toUpper(tok[1]));
        if (!tok.empty() && toUpper(tok[0]) == ".OSDI") {  // load compiled Verilog-A modules up front
            try {
                if (tok.size() != 2) throw std::runtime_error(".osdi needs exactly one library path");
                osdi::loadLibrary(tok[1]);
            } catch (const std::exception& e) { return fail(lineNum, e.what()); }
            continue;
        }
        if (tok.empty() || toUpper(tok[0]) != ".PARAM") continue;
        try { parseParam(tok); }
        catch (const std::exception& e) { return fail(lineNum, e.what()); }
    }

    for (const auto& [lineNum, line] : lines) {
        if (lineNum == 1) continue; // Line 1 is the SPICE title card

        auto tok = tokenize(line);
        if (tok.empty()) continue;

        result_.linesProcessed++;
        std::string keyword = toUpper(tok[0]);

        try {
            // Directives start with '.'
            if (keyword[0] == '.') {
                if (keyword == ".MODEL")  parseModelCard(tok);
                else if (keyword == ".OSDI") {}   // loaded in the first pass
                else if (keyword == ".SUBCKT") parseSubcktBegin(tok);
                else if (keyword == ".ENDS")   parseSubcktEnd();
                else if (keyword == ".PARAM")  {} // handled in the first pass
                else if (keyword == ".OP")     result_.hasOP = true;
                else if (keyword == ".OPTION" || keyword == ".OPTIONS") {
                    // gminsteps changes the DC homotopy; the other options are still ignored
                    for (size_t i = 1; i + 1 < tok.size(); ++i)
                        if (toUpper(tok[i]) == "GMINSTEPS")
                            result_.netlist.environment.gmin_stepping = parseValue(tok[i + 1]) != 0.0;
                }
                else if (keyword == ".TRAN") {
                    result_.hasTRAN = true;
                    if (tok.size() >= 3) {
                        result_.tranStep = parseValue(tok[1]);
                        result_.tranStop = parseValue(tok[2]);
                        // .tran tstep tstop [tstart [tmax]] [UIC]
                        if (tok.size() >= 4 && isValue(tok[3])) result_.tranStart = parseValue(tok[3]);
                    }
                    for (const auto& t : tok)
                        if (toUpper(t) == "UIC") result_.tranUIC = true;
                }
                else if (keyword == ".IC") {
                    // .ic V(node)=value ...  → tokens: V node value
                    if (inSubckt_) throw std::runtime_error(".ic inside .subckt is not supported");
                    if (tok.size() < 4 || (tok.size() - 1) % 3 != 0)
                        throw std::runtime_error(".ic expects V(node)=value entries");
                    for (size_t i = 1; i + 2 < tok.size(); i += 3) {
                        if (toUpper(tok[i]) != "V")
                            throw std::runtime_error(".ic supports V(node)=value only");
                        pendingICs_.push_back({toUpper(tok[i + 1]), parseValue(tok[i + 2]), lineNum});
                    }
                }
                else if (keyword == ".NOISE") {
                    // .noise v(out[,ref]) src dec|oct|lin n fstart fstop [pts_per_summary]
                    size_t sw = 4;
                    while (sw < tok.size() && toUpper(tok[sw]) != "DEC" && toUpper(tok[sw]) != "OCT" &&
                           toUpper(tok[sw]) != "LIN") ++sw;
                    if (toUpper(tok[1]) != "V" || (sw != 4 && sw != 5) || sw + 3 >= tok.size())
                        throw std::runtime_error(".noise expects v(out[,ref]) src dec|oct|lin n fstart fstop");
                    noiseOut_ = toUpper(tok[2]);
                    noiseRef_ = sw == 5 ? toUpper(tok[3]) : "0";
                    result_.noiseSource = toUpper(tok[sw - 1]);
                    result_.noiseSweep  = toUpper(tok[sw]) == "DEC" ? "dec" : toUpper(tok[sw]) == "OCT" ? "oct" : "lin";
                    result_.noisePoints = static_cast<int>(parseValue(tok[sw + 1]));
                    result_.noiseStart  = parseValue(tok[sw + 2]);
                    result_.noiseStop   = parseValue(tok[sw + 3]);
                }
                else if (keyword == ".AC") {
                    // .ac dec|oct|lin n fstart fstop
                    const std::string sw = tok.size() == 5 ? toUpper(tok[1]) : "";
                    if (sw != "DEC" && sw != "OCT" && sw != "LIN")
                        throw std::runtime_error(".ac expects dec|oct|lin n fstart fstop");
                    result_.acSweep = sw == "DEC" ? "dec" : sw == "OCT" ? "oct" : "lin";
                    result_.acPoints = static_cast<int>(parseValue(tok[2]));
                    result_.acStart = parseValue(tok[3]);
                    result_.acStop  = parseValue(tok[4]);
                }
                else if (keyword == ".DC") {
                    // .dc SRC start stop step (one V or I source)
                    if (tok.size() != 5)
                        throw std::runtime_error(".dc supports one source sweep (.dc SRC start stop step)");
                    const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(tok[1][0])));
                    if (c != 'V' && c != 'I')
                        throw std::runtime_error(".dc sweeps a V or I source only, not '" + tok[1] + "'");
                    result_.dcSource = toUpper(tok[1]);
                    result_.dcStart  = parseValue(tok[2]);
                    result_.dcStop   = parseValue(tok[3]);
                    result_.dcStep   = parseValue(tok[4]);
                }
                else if (keyword == ".END") break;
                // Output-only (.four only prints a Fourier table) and first-pass cards; anything else changes the
                // circuit or analysis (.step, .temp, .nodeset, ...) and used to be
                // ignored silently.
                else if (keyword != ".GLOBAL" && keyword != ".PRINT" && keyword != ".PLOT" &&
                         keyword != ".SAVE" && keyword != ".PROBE" && keyword != ".MEAS" &&
                         keyword != ".MEASURE" &&
                         keyword != ".WIDTH" && keyword != ".TITLE" && keyword != ".FOUR")
                    throw std::runtime_error(tok[0] + " is not supported yet");
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
                case 'N': parseOsdiDevice(tok, blk);  break;
                case 'E': parseVCVS(tok, blk);        break;
                case 'G': parseVCCS(tok, blk);        break;
                case 'H': parseCCVS(tok, blk);        break;
                case 'F': parseCCCS(tok, blk);        break;
                // T used to be two unconnected resistors (silently wrong)
                case 'T':
                    throw std::runtime_error(tok[0] + ": transmission lines are not supported yet");
                case 'X': {
                    // X<name> <nodes...> <subcktName>; expanded after parsing
                    requireFields(tok, 3);
                    PendingInstance pi{toUpper(tok.back()), {}, lineNum};
                    for (size_t i = 1; i + 1 < tok.size(); ++i)
                        pi.nodes.push_back(nodeIndex(tok[i]));
                    (inSubckt_ ? subcktChildren_[subcktName_] : topInstances_).push_back(pi);
                    break;
                }
                default:
                    // Was a warning: the element was dropped and the rest
                    // of the circuit "converged" without it.
                    throw std::runtime_error("unknown element '" + tok[0] + "'");
            }
            result_.elementsCreated++;
        } catch (const std::exception& e) {
            return fail(lineNum, e.what());
        }
    }

    if (inSubckt_)
        return fail(lines.back().first, ".subckt " + subcktName_ + " has no .ends");
    try { resolveControlledSources(result_.netlist.globalBlock); }
    catch (const std::exception& e) { return fail(lines.back().first, e.what()); }

    // Finalize: compute numGlobalNodes from nodeMap (max index)
    for (const auto& [name, idx] : nodeMap_) {
        if (idx > result_.netlist.numGlobalNodes)
            result_.netlist.numGlobalNodes = idx;
    }

    // .noise output nodes, resolved once every node is known
    if (!noiseOut_.empty()) {
        auto node = [&](const std::string& nm) {
            if (nm == "0" || nm == "GND") return 0;
            auto it = nodeMap_.find(nm);
            return it == nodeMap_.end() ? -1 : it->second;
        };
        result_.noiseOut = node(noiseOut_);
        result_.noiseRef = node(noiseRef_);
        if (result_.noiseOut <= 0 || result_.noiseRef < 0)
            return fail(lines.back().first, ".noise names an unknown output node");
    }

    // .ic may name a node defined later in the file
    for (const auto& ic : pendingICs_) {
        auto it = nodeMap_.find(ic.node);
        if (it == nodeMap_.end())
            return fail(ic.line, ".ic names unknown node '" + ic.node + "'");
        result_.netlist.nodeICs.push_back({it->second, ic.value});
    }

    // Expand X cards (recursively) now that every .subckt is defined
    for (const auto& pi : topInstances_) {
        try { expandInstance(pi, pi.nodes, 0); }
        catch (const std::exception& e) { return fail(pi.line, e.what()); }
    }

    // Resolve .model cards now that the whole file is read (SPICE allows .model
    // after the instance; resolving at the instance made a late PNP an NPN).
    // A card's missing parameters take SPICE defaults; a device with no card
    // keeps the engine defaults. PULSE TR/TF of 0 take TSTEP, as in SPICE.
    // An undefined model is an error (it used to fall back to engine defaults silently)
    auto card = [&](const std::string& model) -> const ModelCard* {
        auto it = modelCards_.find(toUpper(model));
        if (it == modelCards_.end()) throw std::runtime_error("model '" + model + "' is not defined");
        return &it->second;
    };
    // A parameter the engine does not model is an error, never silently dropped.
    // 0 is accepted for any key (SPICE: off / infinite), AF only without KF.
    // ponytail: a 0 that is not "off" (e.g. TNOM=0) is still accepted.
    auto supported = [&](const ModelCard* c, std::initializer_list<const char*> used) {
        for (const auto& [k, v] : c->params) {
            bool ok = v == 0.0 || (k == "LEVEL" && v == 1.0) || (k == "AF" && c->get("KF", 0.0) == 0.0);
            for (const char* u : used) ok = ok || k == u;
            if (!ok) throw std::runtime_error("model " + c->name + ": parameter " + k + " is not supported yet");
        }
    };
    // Magnitudes (>= 0) and scales (> 0): a negative value is a sign error, never a
    // model; it used to be simulated silently (VAF = -5 put 18 V on a 3 V supply).
    auto bounded = [&](const ModelCard* c, std::initializer_list<const char*> nonNeg,
                       std::initializer_list<const char*> pos) {
        for (const char* k : nonNeg)
            if (c->params.count(k) && !(c->params.at(k) >= 0.0))
                throw std::runtime_error("model " + c->name + ": " + k + " must be >= 0");
        for (const char* k : pos)
            if (c->params.count(k) && !(c->params.at(k) > 0.0))
                throw std::runtime_error("model " + c->name + ": " + k + " must be > 0");
    };
    // The card's type must fit the device (a Q on an NMOS card reported "VTO is not supported")
    auto kind = [&](const ModelCard* c, const std::string& inst, std::initializer_list<const char*> types) {
        for (const char* t : types) if (c->type == t) return c;
        throw std::runtime_error(inst + ": model " + c->name + " is " + c->type + ", not a " +
                                 std::string(*types.begin()) + "/" + *(types.begin() + 1) + " model");
    };
    auto resolveBlock = [&](TensorBlock& blk) {
        for (auto& m : blk.mosfets)
            if (const ModelCard* c = kind(card(m.modelName), m.instanceName, {"NMOS", "PMOS"})) {
                supported(c, {"KP", "VTO", "LAMBDA", "GAMMA", "PHI", "IS", "LD", "TOX", "UO", "U0",
                              "CGSO", "CGDO", "CGBO", "CBD", "CBS", "CJ", "MJ", "CJSW", "MJSW", "PB", "FC"});
                bounded(c, {"KP", "LAMBDA", "GAMMA", "IS", "LD", "TOX", "UO", "U0", "CGSO", "CGDO", "CGBO",
                            "CBD", "CBS", "CJ", "CJSW", "MJ", "MJSW"},
                        {"PHI"});
                auto& k = m.card;
                k.resolved = true;
                k.pmos = c->type == "PMOS";
                k.tox = c->get("TOX", 0.0);
                // No KP with TOX: KP = UO[cm^2/Vs] * COX (ngspice mos1temp)
                k.kp = c->get("KP", k.tox > 0.0 ? c->get("UO", c->get("U0", 600.0)) * 1e-4 * 3.9 * 8.854214871e-12 / k.tox
                                                : 2e-5);
                k.vto = c->get("VTO", 0.0); k.lambda = c->get("LAMBDA", 0.0); k.gamma = c->get("GAMMA", 0.0);
                k.phi = c->get("PHI", 0.6); k.is = c->get("IS", 1e-14); k.ld = c->get("LD", 0.0);
                k.cgso = c->get("CGSO", 0.0); k.cgdo = c->get("CGDO", 0.0); k.cgbo = c->get("CGBO", 0.0);
                k.cbd = c->get("CBD", 0.0); k.cbs = c->get("CBS", 0.0); k.cj = c->get("CJ", 0.0);
                k.cjsw = c->get("CJSW", 0.0); k.mj = c->get("MJ", 0.5); k.mjsw = c->get("MJSW", 0.5);
                k.pb = c->get("PB", 0.8); k.fc = c->get("FC", 0.5);
                if (!(m.l - 2.0 * k.ld > 0.0))
                    throw std::runtime_error(m.instanceName + ": L - 2*LD must be > 0");
                if (!(k.mj < 1.0 && k.mjsw < 1.0 && k.pb > 0.0 && k.fc < 1.0))
                    throw std::runtime_error("model " + c->name + ": needs MJ, MJSW < 1, PB > 0, FC < 1");
            }
        // first(c, {"A", "B"}, def): the first alias the card gives (SPICE spellings vary)
        auto first = [](const ModelCard* c, std::initializer_list<const char*> keys, double def) {
            for (const char* k : keys) if (c->params.count(k)) return c->params.at(k);
            return def;
        };
        // EG/XTI/XTB only act away from TNOM; the engine runs at TNOM.
        for (auto& d : blk.diodes) if (const ModelCard* c = kind(card(d.modelName), d.instanceName, {"D", "D"})) {
            supported(c, {"IS", "N", "RS", "CJO", "CJ0", "CJ", "VJ", "PB", "M", "MJ", "FC", "TT", "BV", "IBV",
                          "EG", "XTI"});
            bounded(c, {"IS", "RS", "CJO", "CJ0", "CJ", "M", "MJ", "TT"}, {"N", "BV"});  // ngspice: BV = 0 breaks down at 0 V
            d.saturationCurrent_I_S_A = c->get("IS", 1e-14);
            d.emissionCoefficient_N   = c->get("N", 1.0);
            auto& k = d.card;
            k.rs = c->get("RS", 0.0); k.cjo = first(c, {"CJO", "CJ0", "CJ"}, 0.0);
            k.vj = first(c, {"VJ", "PB"}, 1.0); k.m = first(c, {"M", "MJ"}, 0.5);
            k.fc = c->get("FC", 0.5); k.tt = c->get("TT", 0.0);
            k.bvGiven = c->params.count("BV") > 0;  // BV > 0 (checked above)
            k.bv = c->get("BV", 0.0); k.ibv = c->get("IBV", 1e-3);
            if (!(k.m < 1.0 && k.vj > 0.0 && k.fc < 1.0 && k.rs >= 0.0 && k.ibv > 0.0))
                throw std::runtime_error("model " + c->name + ": needs M < 1, VJ > 0, FC < 1, RS >= 0, IBV > 0");
        }
        for (auto& q : blk.bjts) if (const ModelCard* c = kind(card(q.modelName), q.instanceName, {"NPN", "PNP"})) {
            supported(c, {"IS", "BF", "BR", "NF", "NR", "VAF", "VA", "VAR", "VB", "IKF", "IK", "IKR",
                          "ISE", "NE", "ISC", "NC", "RB", "RC", "RE", "CJE", "VJE", "PE", "MJE", "ME",
                          "CJC", "VJC", "PC", "MJC", "MC", "XCJC", "CJS", "CCS", "VJS", "PS", "MJS", "MS",
                          "FC", "TF", "XTF", "VTF", "ITF", "TR", "EG", "XTI", "XTB", "RBM", "ISS", "NS", "SUBS"});
            bounded(c, {"IS", "VAF", "VA", "VAR", "VB", "IKF", "IK", "IKR", "ISE", "ISC", "CJE", "CJC", "CJS", "CCS",
                        "MJE", "ME", "MJC", "MC", "MJS", "MS", "TF", "XTF", "VTF", "ITF", "TR", "ISS"},
                    {"NE", "NC", "NS"});
            q.isNPN = (c->type != "PNP");
            q.saturationCurrent_I_S_A = c->get("IS", 1e-16);
            q.betaF = c->get("BF", 100.0);
            q.betaR = c->get("BR", 1.0);
            auto& k = q.card;
            k.nf = c->get("NF", 1.0); k.nr = c->get("NR", 1.0);
            k.vaf = first(c, {"VAF", "VA"}, 0.0); k.var = first(c, {"VAR", "VB"}, 0.0);
            k.ikf = first(c, {"IKF", "IK"}, 0.0); k.ikr = c->get("IKR", 0.0);
            k.ise = c->get("ISE", 0.0); k.ne = c->get("NE", 1.5);
            k.isc = c->get("ISC", 0.0); k.nc = c->get("NC", 2.0);
            k.rb = c->get("RB", 0.0); k.rc = c->get("RC", 0.0); k.re = c->get("RE", 0.0);
            k.cje = c->get("CJE", 0.0); k.vje = first(c, {"VJE", "PE"}, 0.75); k.mje = first(c, {"MJE", "ME"}, 0.33);
            k.cjc = c->get("CJC", 0.0); k.vjc = first(c, {"VJC", "PC"}, 0.75); k.mjc = first(c, {"MJC", "MC"}, 0.33);
            k.xcjc = c->get("XCJC", 1.0);
            k.cjs = first(c, {"CJS", "CCS"}, 0.0); k.vjs = first(c, {"VJS", "PS"}, 0.75); k.mjs = first(c, {"MJS", "MS"}, 0.0);
            k.fc = c->get("FC", 0.5);
            k.iss = c->get("ISS", 0.0); k.ns = c->get("NS", 1.0);
            // SUBS (ngspice): 1 vertical (substrate junction to C), -1 lateral (to B); default by polarity
            const double subs = c->get("SUBS", q.isNPN ? 1.0 : -1.0);
            if (subs != 1.0 && subs != -1.0) throw std::runtime_error("model " + c->name + ": SUBS must be 1 or -1");
            k.lateral = subs == -1.0;
            k.tf = c->get("TF", 0.0); k.xtf = c->get("XTF", 0.0); k.vtf = c->get("VTF", 0.0);
            k.itf = c->get("ITF", 0.0); k.tr = c->get("TR", 0.0);
            // RBM != RB makes the base resistance bias dependent (qb), which is not modelled
            if (c->params.count("RBM") && c->params.at("RBM") != k.rb)
                throw std::runtime_error("model " + c->name + ": RBM != RB (bias-dependent base resistance) is not supported yet");
            if (!(k.mje < 1.0 && k.mjc < 1.0 && k.mjs < 1.0 && k.vje > 0.0 && k.vjc > 0.0 && k.vjs > 0.0 &&
                  k.fc < 1.0 && k.xcjc >= 0.0 && k.xcjc <= 1.0 && k.nf > 0.0 && k.nr > 0.0 &&
                  k.rb >= 0.0 && k.rc >= 0.0 && k.re >= 0.0 && q.betaF > 0.0 && q.betaR > 0.0))
                throw std::runtime_error("model " + c->name + ": needs MJE/MJC/MJS < 1, VJE/VJC/VJS > 0, FC < 1, "
                                         "0 <= XCJC <= 1, NF/NR/BF/BR > 0, RB/RC/RE >= 0");
        }
        for (auto& o : blk.osdiDevices) {
            const ModelCard* c = card(o.modelName);
            const osdi::Descriptor* d = osdi::findModule(c->type);
            if (!d) throw std::runtime_error("model " + c->name + ": type " + c->type +
                                             " is not a loaded OSDI module (load it with .osdi <file>)");
            auto& m = osdiModels_[toUpper(c->name)];
            if (!m) m = std::make_shared<osdi::Model>(d, c->name, c->params);
            o.model = m;
        }
        for (auto& j : blk.jfets) if (const ModelCard* c = card(j.modelName)) {
            supported(c, {"BETA", "VTO", "LAMBDA"});
            j.isNChannel = (c->type == "NJF" || c->type == "NJFET");
            j.beta   = c->get("BETA", 1e-4);
            j.Vto    = c->get("VTO", -2.0);
            j.lambda = c->get("LAMBDA", 0.0);
        }
        if (result_.tranStep > 0.0) {
            auto edges = [&](auto& s) {
                if (s.type != "PULSE") return;
                if (s.pulse_tr <= 0.0) s.pulse_tr = result_.tranStep;
                if (s.pulse_tf <= 0.0) s.pulse_tf = result_.tranStep;
            };
            for (auto& v : blk.voltageSources) edges(v);
            for (auto& i : blk.currentSources) edges(i);
        }
    };
    try {
        resolveBlock(result_.netlist.globalBlock);
        for (auto& [name, blk] : result_.netlist.blockDefinitions) resolveBlock(blk);
    } catch (const std::exception& e) { return fail(lines.back().first, e.what()); }

    // Merge model cards into netlist
    for (auto& [name, card] : modelCards_) {
        result_.netlist.addModelCard(card);
        result_.modelsCreated++;
    }

    // Initialize global state
    result_.netlist.globalState.initialize(result_.netlist.globalBlock);
    try {  // OSDI instances: parameters, setup_instance, node collapsing
        result_.netlist.instantiateOsdi(result_.netlist.environment.ambient_temp_K - 300.15 +
                                        PhysicsConstants::T_DEVICE_NOMINAL_K);
    } catch (const std::exception& e) { return fail(lines.back().first, e.what()); }

    return result_;
}

// ============================================================================
// Element Parsers
// ============================================================================

void SPICEParser::parseResistor(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 4);
    Resistor r;
    r.name = tok[0];
    r.nodeTerminal1 = nodeIndex(tok[1]);
    r.nodeTerminal2 = nodeIndex(tok[2]);
    r.resistance_ohms = parseValue(tok[3]);
    if (r.resistance_ohms == 0.0) {  // an exact short: a 0 V branch (negative R is legal SPICE)
        blk.voltageSources.push_back({r.nodeTerminal1, r.nodeTerminal2, 0.0});
        return;
    }

    // tokenize() turns "KEY=val" into ["KEY", "val"]
    for (size_t i = 4; i < tok.size(); ++i) {
        std::string upper = toUpper(tok[i]);
        if (i + 1 >= tok.size()) throw std::runtime_error(tok[0] + ": parameter " + upper + " has no value");
        if (upper == "W")        r.width_m  = parseValue(tok[++i]);
        else if (upper == "L")   r.length_m = parseValue(tok[++i]);
        else if (upper == "TC1" || upper == "TC2") {
            if (parseValue(tok[++i]) != 0.0)  // no stamp applies them
                throw std::runtime_error(tok[0] + ": TC1/TC2 temperature coefficients are not supported yet");
        }
        else throw std::runtime_error(tok[0] + ": parameter " + upper + " is not supported yet");
    }
    blk.resistors.push_back(r);
}

void SPICEParser::parseCapacitor(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 4);
    Capacitor c;
    c.name = tok[0];
    c.nodePlate1 = nodeIndex(tok[1]);
    c.nodePlate2 = nodeIndex(tok[2]);
    c.capacitance_farads = parseValue(tok[3]);

    // tokenize() turns "KEY=val" into ["KEY", "val"]. IC= applies with UIC.
    for (size_t i = 4; i < tok.size(); ++i) {
        std::string upper = toUpper(tok[i]);
        if (i + 1 >= tok.size()) throw std::runtime_error(tok[0] + ": parameter " + upper + " has no value");
        if (upper == "W")      c.width_m  = parseValue(tok[++i]);
        else if (upper == "L") c.length_m = parseValue(tok[++i]);
        else if (upper == "M") c.m        = parseValue(tok[++i]);
        else if (upper == "IC") { c.ic_V = parseValue(tok[++i]); c.icGiven = true; }
        else throw std::runtime_error(tok[0] + ": parameter " + upper + " is not supported yet");
    }
    blk.capacitors.push_back(c);
}

void SPICEParser::parseInductor(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 4);
    Inductor ind;
    ind.name = tok[0];
    ind.nodeCoil1 = nodeIndex(tok[1]);
    ind.nodeCoil2 = nodeIndex(tok[2]);
    ind.inductance_henries = parseValue(tok[3]);
    // IC= (initial current) applies with UIC; anything else used to be ignored silently
    for (size_t i = 4; i < tok.size(); ++i) {
        if (toUpper(tok[i]) != "IC" || i + 1 >= tok.size())
            throw std::runtime_error(tok[0] + ": unsupported inductor parameter '" + tok[i] + "'");
        ind.ic_A = parseValue(tok[++i]);
        ind.icGiven = true;
    }

    // Track inductor by name for K-statement resolution
    inductorIndex_[toUpper(tok[0])] = static_cast<int>(blk.inductors.size());

    blk.inductors.push_back(ind);
}

void SPICEParser::parseMutualInd(const std::vector<std::string>& tok, TensorBlock& blk) {
    if (tok.size() != 4)  // a core model or a third inductor used to be dropped silently
        throw std::runtime_error(tok[0] + ": only K <L1> <L2> <k> (linear, two inductors) is supported");
    MutualInductor mi;
    auto it1 = inductorIndex_.find(toUpper(tok[1]));
    auto it2 = inductorIndex_.find(toUpper(tok[2]));
    if (it1 == inductorIndex_.end() || it2 == inductorIndex_.end())
        throw std::runtime_error(tok[0] + " references an undefined inductor");

    mi.inductor1_index = it1->second;
    mi.inductor2_index = it2->second;
    mi.couplingCoefficient_k = parseValue(tok[3]);
    blk.mutualInductors.push_back(mi);
}

// V<name> <n+> <n-> [DC <val>] [AC <mag> [<phase>]] [PULSE(...)] [SIN(...)]
void SPICEParser::parseVoltSrc(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 3);
    VoltageSource vs;
    vs.name = toUpper(tok[0]);
    vs.nodePositive = nodeIndex(tok[1]);
    vs.nodeNegative = nodeIndex(tok[2]);
    parseSourceSpec(tok, vs, vs.voltage_V);
    blk.voltageSources.push_back(vs);
}

// Shared V/I spec. If no DC value is given, the DC value is the waveform's
// t=0 value (SPICE rule), so every DC code path can keep using voltage_V.
template <class Src>
void SPICEParser::parseSourceSpec(const std::vector<std::string>& tok, Src& s, double& dc) const {
    bool hasDC = false;
    for (size_t i = 3; i < tok.size();) {
        std::string kw = toUpper(tok[i]);
        if (kw == "DC") {
            hasDC = true;  // bare "DC" means 0
            if (i + 1 < tok.size() && isValue(tok[i + 1])) { dc = parseValue(tok[i + 1]); i += 2; }
            else ++i;
        } else if (kw == "AC") {
            ++i;
            s.ac_mag = 1.0;  // SPICE: bare "AC" means unit magnitude
            if (i < tok.size() && isValue(tok[i])) s.ac_mag = parseValue(tok[i++]);
            if (i < tok.size() && isValue(tok[i])) s.ac_phase = parseValue(tok[i++]);
        } else if (kw == "PULSE") {
            s.type = "PULSE";
            parsePulse(tok, i + 1, s);
            break;  // waveform consumes the rest of the card
        } else if (kw == "SIN") {
            s.type = "SINE";  // canonical name used by the stampers
            parseSine(tok, i + 1, s);
            break;
        } else if (kw == "EXP" || kw == "PWL" || kw == "SFFM" || kw == "AM") {
            throw std::runtime_error(kw + " sources are not supported yet");
        } else {
            dc = parseValue(tok[i++]);  // bare value, e.g. "V1 1 0 5"
            hasDC = true;
        }
    }
    if (s.type.empty()) s.type = "DC";
    if (!hasDC) dc = waveformValue(s, 0.0, 0.0);
}

// PULSE V1 V2 TD TR TF PW PER
template <class Src>
void SPICEParser::parsePulse(const std::vector<std::string>& tok, size_t idx, Src& s) const {
    double* f[] = {&s.pulse_v1, &s.pulse_v2, &s.pulse_td, &s.pulse_tr,
                   &s.pulse_tf, &s.pulse_pw, &s.pulse_per};
    size_t n = 0;
    for (; n < 7 && idx + n < tok.size(); ++n) *f[n] = parseValue(tok[idx + n]);
    if (n < 6) s.pulse_pw = 1e99;  // SPICE default PW = TSTOP: never falls
    // TR/TF = 0 get SPICE's TSTEP default in parse() once .tran is known
}

// SIN VO VA FREQ TD THETA PHASE
template <class Src>
void SPICEParser::parseSine(const std::vector<std::string>& tok, size_t idx, Src& s) const {
    double* f[] = {&s.sine_vo, &s.sine_va, &s.sine_freq, &s.sine_td,
                   &s.sine_theta, &s.sine_phase};
    for (size_t n = 0; n < 6 && idx + n < tok.size(); ++n) *f[n] = parseValue(tok[idx + n]);
}

// I<name> <n+> <n-> [DC <val>] [AC <mag> [<phase>]] [PULSE(...)] [SIN(...)]
void SPICEParser::parseCurrSrc(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 3);
    CurrentSource cs;
    cs.name = toUpper(tok[0]);
    cs.nodePositive = nodeIndex(tok[1]);
    cs.nodeNegative = nodeIndex(tok[2]);
    parseSourceSpec(tok, cs, cs.current_A);
    blk.currentSources.push_back(cs);
}

// D<name> <n+> <n-> <modelName> [area] [AREA=] [M=] [OFF] [IC=]   (params from .model in parse())
void SPICEParser::parseDiode(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 4);
    Diode d;
    d.area = instanceArea(tok, 4);
    d.instanceName = tok[0];
    d.anode = nodeIndex(tok[1]);
    d.cathode = nodeIndex(tok[2]);
    d.modelName = tok[3];
    blk.diodes.push_back(d);
}

// M<name> <nd> <ng> <ns> <nb> <modelName> [W=] [L=] [AD=] [AS=] [PD=] [PS=] [NRD=] [NRS=] [NF=] [M=]
void SPICEParser::parseMosfet(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 6);
    Mosfet m;
    m.instanceName = tok[0];
    m.drain  = nodeIndex(tok[1]);
    m.gate   = nodeIndex(tok[2]);
    m.source = nodeIndex(tok[3]);
    m.body   = nodeIndex(tok[4]);
    m.modelName = tok[5];
    if (!modelNames_.count(toUpper(tok[5])))
        throw std::runtime_error(tok[0] + ": expected 4 nodes (d g s b), then a model name; '" + tok[5] +
                                 "' is not a .model card");
    m.w = m.l = 100e-6;  // SPICE DEFW / DEFL

    // tokenize() turns "W=4u" into ["W", "4u"]. AD/AS/PD/PS act through CJ/CJSW;
    // NRD/NRS only through RSH, which the model check rejects.
    for (size_t i = 6; i < tok.size(); ++i) {
        std::string upper = toUpper(tok[i]);
        if (upper == "OFF") throw std::runtime_error(tok[0] + ": OFF is not supported yet");
        if (i + 1 >= tok.size()) throw std::runtime_error(tok[0] + ": parameter " + upper + " has no value");
        if (upper == "W")        m.w = parseValue(tok[++i]);
        else if (upper == "L")   m.l = parseValue(tok[++i]);
        else if (upper == "AD")  m.geo.ad = parseValue(tok[++i]);
        else if (upper == "AS")  m.geo.as = parseValue(tok[++i]);
        else if (upper == "PD")  m.geo.pd = parseValue(tok[++i]);
        else if (upper == "PS")  m.geo.ps = parseValue(tok[++i]);
        else if (upper == "NRD") m.geo.nrd = parseValue(tok[++i]);
        else if (upper == "NRS") m.geo.nrs = parseValue(tok[++i]);
        else if (upper == "NF") {
            if (parseValue(tok[++i]) != 1.0) throw std::runtime_error(tok[0] + ": NF is not supported yet");
        }
        else if (upper == "M")   m.extra.mult = parseValue(tok[++i]);
        else throw std::runtime_error(tok[0] + ": parameter " + upper + " is not supported yet");
    }
    if (!(m.l > 0.0) || !(m.w > 0.0)) throw std::runtime_error(tok[0] + ": W and L must be > 0");
    blk.mosfets.push_back(m);
}

// Q<name> <nc> <nb> <ne> [<ns>] <modelName> [area] [OFF]   (params from .model in parse())
void SPICEParser::parseBJT(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 5);
    BJT q;
    q.instanceName = tok[0];
    q.nodeCollector = nodeIndex(tok[1]);
    q.base = nodeIndex(tok[2]);
    q.emitter = nodeIndex(tok[3]);

    // Optional substrate node: tok[4] is the substrate when tok[5] names a .model
    // card (all names are known after the first pass). No substrate node: ground.
    const size_t modelIdx = tok.size() >= 6 && modelNames_.count(toUpper(tok[5])) ? 5 : 4;
    if (modelIdx == 5) q.substrate = nodeIndex(tok[4]);
    q.modelName = tok[modelIdx];
    q.area = instanceArea(tok, modelIdx + 1);
    blk.bjts.push_back(q);
}

// J<name> <nd> <ng> <ns> <modelName>   (params from .model in parse())
void SPICEParser::parseJFET(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 5);
    if (instanceArea(tok, 5) != 1.0) throw std::runtime_error(tok[0] + ": AREA scaling is not supported yet");
    JFET j;
    j.instanceName = tok[0];
    j.drain  = nodeIndex(tok[1]);
    j.gate   = nodeIndex(tok[2]);
    j.source = nodeIndex(tok[3]);
    j.modelName = tok[4];
    blk.jfets.push_back(j);
}

// N<name> <terminal>... <modelName> [<param> <value>]...   (OSDI device, as ngspice)
// The model name is the first field naming a .model card (all are known after
// the first pass), so the terminals are numbered in order like any element.
void SPICEParser::parseOsdiDevice(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireFields(tok, 3);
    size_t m = 1;
    while (m < tok.size() && !modelNames_.count(toUpper(tok[m]))) ++m;
    if (m == tok.size()) throw std::runtime_error(tok[0] + ": no field names a .model card");
    if (m == 1) throw std::runtime_error(tok[0] + ": no terminals before the model name");
    if ((tok.size() - m - 1) % 2) throw std::runtime_error(tok[0] + ": instance parameters must be name=value pairs");
    OsdiDevice o;
    o.instanceName = tok[0];
    o.modelName = tok[m];
    for (size_t i = 1; i < m; ++i) o.nodes.push_back(nodeIndex(tok[i]));
    for (size_t i = m + 1; i + 1 < tok.size(); i += 2) o.params.push_back({tok[i], parseValue(tok[i + 1])});
    blk.osdiDevices.push_back(o);
}

// E<name> <n+> <n-> <nc+> <nc-> <gain>
// E/G/F/H: only the linear form. POLY / TABLE / VALUE / VOL / CUR / LAPLACE ...
// used to be read as node names (E2 2 0 VOL=2*V(1) -> nodes "VOL", "2*V").
void SPICEParser::requireLinearForm(const std::vector<std::string>& tok, size_t n) {
    for (size_t i = 3; i < tok.size(); ++i) {
        const std::string k = toUpper(tok[i]);
        if (k == "POLY" || k == "TABLE" || k == "VALUE" || k == "VOL" || k == "CUR" || k == "LAPLACE" ||
            k == "FREQ" || k == "CHEBYSHEV")
            throw std::runtime_error(tok[0] + ": " + k + " form is not supported yet");
    }
    if (tok.size() != n)
        throw std::runtime_error(tok[0] + ": expected " + std::to_string(n - 1) + " fields (linear form)");
}

void SPICEParser::parseVCVS(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireLinearForm(tok, 6);
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
    requireLinearForm(tok, 6);
    VCCS g;
    g.outPos  = nodeIndex(tok[1]);
    g.outNeg  = nodeIndex(tok[2]);
    g.ctrlPos = nodeIndex(tok[3]);
    g.ctrlNeg = nodeIndex(tok[4]);
    g.gm      = parseValue(tok[5]);
    blk.voltageControlledCurrentSources.push_back(g);
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
    requireFields(tok, 2);
    if (inSubckt_)
        throw std::runtime_error("nested .subckt definitions are not supported (missing .ends for " +
                                 subcktName_ + "?)");
    inSubckt_ = true;
    subcktName_ = toUpper(tok[1]);
    if (result_.netlist.blockDefinitions.count(subcktName_))
        throw std::runtime_error("subcircuit " + subcktName_ + " defined twice");
    subcktBlock_ = TensorBlock{};
    subcktBlock_.name = subcktName_;
    subcktNodeMap_.clear();
    nextLocalNode_ = 1;
    savedInductorIndex_.swap(inductorIndex_);  // K cards inside see only this body's L
    inductorIndex_.clear();

    subcktPorts_.clear();
    for (size_t i = 2; i < tok.size(); ++i) {
        subcktPorts_.push_back(tok[i]);
        subcktBlock_.pinNames.push_back(tok[i]);
        subcktBlock_.portNodeIndices.push_back(nodeIndex(tok[i]));  // local 1..P
    }
}

// H<name> <n+> <n-> <Vctrl> <transresistance>
void SPICEParser::parseCCVS(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireLinearForm(tok, 5);
    CCVS h{};
    h.outPos = nodeIndex(tok[1]);
    h.outNeg = nodeIndex(tok[2]);
    h.ctrlPos = h.ctrlNeg = 0;
    h.ctrlName = toUpper(tok[3]);
    h.rm = parseValue(tok[4]);
    blk.currentControlledVoltageSources.push_back(h);
}

// F<name> <n+> <n-> <Vctrl> <gain>
void SPICEParser::parseCCCS(const std::vector<std::string>& tok, TensorBlock& blk) {
    requireLinearForm(tok, 5);
    CCCS f{};
    f.outPos = nodeIndex(tok[1]);
    f.outNeg = nodeIndex(tok[2]);
    f.ctrlPos = f.ctrlNeg = 0;
    f.ctrlName = toUpper(tok[3]);
    f.alpha = parseValue(tok[4]);
    blk.currentControlledCurrentSources.push_back(f);
}

void SPICEParser::resolveControlledSources(TensorBlock& blk) {
    auto find = [&](const std::string& name) {
        for (size_t i = 0; i < blk.voltageSources.size(); ++i)
            if (blk.voltageSources[i].name == name) return (int)i;
        throw std::runtime_error("controlling source " + name + " not found" +
                                 (blk.name.empty() ? "" : " in subcircuit " + blk.name) +
                                 " (F/H must reference a V source in the same block)");
    };
    for (auto& h : blk.currentControlledVoltageSources) h.ctrlSource = find(h.ctrlName);
    for (auto& f : blk.currentControlledCurrentSources) f.ctrlSource = find(f.ctrlName);
}

void SPICEParser::parseSubcktEnd() {
    if (!inSubckt_) throw std::runtime_error(".ends without .subckt");
    resolveControlledSources(subcktBlock_);
    inSubckt_ = false;
    inductorIndex_.swap(savedInductorIndex_);
    subcktBlock_.numInternalNodes = nextLocalNode_ - 1;  // ports + internal nodes

    result_.netlist.defineBlock(subcktBlock_);
    result_.subcircuitsCreated++;
}

// One instance of `pi.subckt` with ports on global nodes `portNodes`: fresh
// global nodes for its internal nodes, then its own X cards, recursively.
void SPICEParser::expandInstance(const PendingInstance& pi, const std::vector<int>& portNodes,
                                 int depth) {
    if (depth > 64) throw std::runtime_error("subcircuit nesting deeper than 64 (recursive .subckt?)");
    auto it = result_.netlist.blockDefinitions.find(pi.subckt);
    if (it == result_.netlist.blockDefinitions.end())
        throw std::runtime_error("unknown subcircuit '" + pi.subckt + "'");
    const TensorBlock& blk = it->second;
    if (portNodes.size() != blk.pinNames.size())
        throw std::runtime_error(pi.subckt + " has " + std::to_string(blk.pinNames.size()) +
                                 " pins but the instance connects " + std::to_string(portNodes.size()));

    std::vector<int> map(portNodes);  // local node l -> global map[l-1]
    const auto& globals = subcktGlobals_[pi.subckt];  // internal locals named in .global
    for (int l = (int)map.size() + 1; l <= blk.numInternalNodes; ++l) {
        auto g = globals.find(l);
        if (g == globals.end()) { map.push_back(nextNode_++); continue; }
        auto [node, isNew] = nodeMap_.emplace(g->second, nextNode_);
        if (isNew) ++nextNode_;
        map.push_back(node->second);
    }
    result_.netlist.addInstance(pi.subckt, map);

    for (const auto& child : subcktChildren_[pi.subckt]) {
        std::vector<int> childPorts;
        for (int l : child.nodes) childPorts.push_back(l == 0 ? 0 : map[l - 1]);
        expandInstance(child, childPorts, depth + 1);
    }
}

// .PARAM <name>=<value> [<name>=<value>]...
void SPICEParser::parseParam(const std::vector<std::string>& tok) {
    // After tokenizer: .PARAM key value key value ...
    for (size_t i = 1; i + 1 < tok.size(); i += 2) {
        params_[toUpper(tok[i])] = parseValue(tok[i + 1]);
    }
}

} // namespace acutesim
