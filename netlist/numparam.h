#pragma once
// Parameter front end (ngspice "numparam"): .param expressions, parameterized
// subcircuits and .if/.elseif/.else/.endif, resolved before the netlist parser
// sees the lines. Every {expr}, 'expr' and name=<param> becomes a number; each
// distinct parameter set of a subcircuit becomes one specialized .subckt (equal
// sets share it, so the hierarchy stays reusable); .model cards inside a
// subcircuit are hoisted per specialization.
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace acutesim { namespace numparam {

// Evaluate an expression; lookup(NAME, value) resolves an upper-case identifier.
double eval(const std::string &expr, const std::function<bool(const std::string &, double &)> &lookup);

// Expand the (line number, text) lines of a netlist whose .include/.lib are already
// spliced. Line 1 (the title) passes through. errLine: the line an error refers to.
std::vector<std::pair<int, std::string>> expand(const std::vector<std::pair<int, std::string>> &lines,
                                                int &errLine);

}}  // namespace acutesim::numparam
