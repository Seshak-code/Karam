#pragma once
#include "circuitsim.h"
#include <vector>

/**
 * SuperpositionSolver
 * 
 * Implements the Superposition Principle for circuit analysis.
 * Decomposes a multi-source circuit into N sub-problems, where each sub-problem
 * considers only one active independent source while "killing" the others.
 * 
 * Rules for Killing Sources:
 * 1. Voltage Source -> Short Circuit (0V)
 * 2. Current Source -> Open Circuit (0A)
 * 
 * NOTE: Superposition is strictly valid only for LINEAR circuits.
 * For non-linear circuits (Diodes, BJTs, MOSFETs), strict superposition does not hold.
 * However, this tool can be used for:
 * - Linearized Small-Signal Analysis
 * - Educational demonstration of source contributions
 * - Initialization heuristics
 */
class SuperpositionSolver {
public:
    enum class Mode { LAB_BENCHMARK, SYSTEM_INTEGRATION };

    struct SubProblem {
        std::string sourceName;
        TensorNetlist netlist; // The modified netlist for this case
        std::vector<double> solution; // The solution vector (to be filled)
    };
    
    /**
     * decompose
     * Generates a set of sub-problems based on the selected mode:
     * - LAB_BENCHMARK: Decomposes Ideal Voltage/Current Sources.
     * - SYSTEM_INTEGRATION: Decomposes PowerRails and SignalSources (preserving passive parasitics).
     */
    static std::vector<SubProblem> decompose(const TensorNetlist& original, Mode mode = Mode::LAB_BENCHMARK);
    
    /**
     * combine
     * Sums the solution vectors of all sub-problems to produce the total response.
     * V_total = Sum(V_i)
     */
    static std::vector<double> combine(const std::vector<SubProblem>& results);
};
