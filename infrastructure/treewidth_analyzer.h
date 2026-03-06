#pragma once
#include "factor_graph.h"
#include <cstdint>
#include <vector>

/**
 * treewidth_analyzer.h — Min-Fill Treewidth Estimator (Phase A)
 *
 * Implements the Min-Fill elimination heuristic over the primal factor graph.
 * At each step the variable that introduces the fewest fill edges is selected
 * for elimination; ties are broken by ascending variable ID for cross-platform
 * determinism (required by the FP-determinism policy).
 *
 * Complexity: O(n² · k) where k is the average fill introduced per step.
 * Sufficient for Phase A validation on op-amp and PDN-mesh scale circuits
 * (n < 10,000, including branch-current MNA variables).
 *
 * The result is embedded in CompiledTensorBlock::tnAnalysis and is purely
 * analytical — no solver execution path is modified.
 */

struct TreewidthAnalysis {
    std::vector<uint32_t> eliminationOrdering;    // Min-Fill variable order
    uint32_t estimatedTreewidth      = 0;          // max clique size − 1 during elim
    std::vector<uint32_t> schurBoundaryCandidates; // top-10% fill nodes (Phase 5.3.2)
    bool     isAcyclic  = false;                   // primal graph is a tree/forest
    bool     isAnalyzed = false;                   // false for empty/degenerate graphs
};

class TreewidthAnalyzer {
public:
    // Run Min-Fill on the primal graph derived from fg.
    // Branch-current variables (voltage sources, inductors) are included as
    // first-class MNA unknowns for mathematically correct treewidth estimation.
    static TreewidthAnalysis analyze(const FactorGraph& fg);
};
