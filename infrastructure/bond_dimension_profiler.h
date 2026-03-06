#pragma once

#include "factor_graph.h"
#include <cstdint>
#include <string>
#include <vector>

/**
 * bond_dimension_profiler.h — Per-Step Bond Dimension Profiling (Phase 5.3.0)
 *
 * Extends the Min-Fill elimination analysis from Phase A to record the clique
 * size (= bond dimension) at every elimination step. This produces a profile
 * that answers: "How does the bond dimension grow as we eliminate variables?"
 *
 * The bond dimension at step k equals the size of the clique formed around
 * the eliminated variable — the same quantity that determines fill-in in
 * sparse LU and intermediate tensor size in tensor network contraction.
 *
 * Purpose: Empirically validate the Electrical Area Law before committing to
 * the full TN engine (Phase 5.3.1+). If target circuit classes show bounded
 * χ, tensor network methods are viable. If χ grows as O(N), fall back to
 * conventional sparse solvers.
 */

struct BondDimensionProfile {
    std::vector<uint32_t> eliminationStep;  // step index (0 .. N-1)
    std::vector<uint32_t> bondDimension;    // clique size at each step
    uint32_t maxBondDimension = 0;          // = estimatedTreewidth + 1
    uint32_t circuitSize = 0;              // N (number of MNA variables)
    std::string circuitClass;              // "rc_ladder", "opamp", "pdn_mesh", etc.
};

class BondDimensionProfiler {
public:
    /**
     * Run Min-Fill with per-step profiling.
     *
     * Same algorithm as TreewidthAnalyzer::analyze() but records clique size
     * at every elimination step into bondDimension[]. This is purely a
     * diagnostic extension — no solver path changes.
     *
     * @param fg           Factor graph to profile
     * @param circuitClass Human-readable circuit family label
     * @return Per-step bond dimension profile
     */
    static BondDimensionProfile profile(const FactorGraph& fg,
                                         const std::string& circuitClass);
};
