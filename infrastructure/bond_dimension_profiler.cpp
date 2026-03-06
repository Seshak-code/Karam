#include "bond_dimension_profiler.h"
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// BondDimensionProfiler::profile
// ============================================================================

BondDimensionProfile BondDimensionProfiler::profile(
    const FactorGraph& fg,
    const std::string& circuitClass)
{
    BondDimensionProfile result;
    result.circuitClass = circuitClass;

    if (fg.variableCount() == 0)
        return result;

    const size_t n = fg.variableCount();
    result.circuitSize = static_cast<uint32_t>(n);

    // ── Build mutable adjacency from the factor graph ─────────────────────
    // Mirrors TreewidthAnalyzer::analyze() exactly for consistency.
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> adj;
    for (const auto& v : fg.variables)
        adj[v.id];

    for (const auto& factor : fg.factors) {
        const auto& cvars = factor.connectedVariables;
        for (size_t i = 0; i < cvars.size(); ++i) {
            for (size_t j = i + 1; j < cvars.size(); ++j) {
                uint32_t u = cvars[i], v = cvars[j];
                if (u != v) {
                    adj[u].insert(v);
                    adj[v].insert(u);
                }
            }
        }
    }

    // ── Min-Fill elimination loop with per-step profiling ──────────────────
    std::vector<uint32_t> remaining;
    remaining.reserve(n);
    for (const auto& v : fg.variables)
        remaining.push_back(v.id);
    std::sort(remaining.begin(), remaining.end());

    result.eliminationStep.reserve(n);
    result.bondDimension.reserve(n);
    uint32_t maxClique = 0;

    for (size_t step = 0; step < n; ++step) {
        std::sort(remaining.begin(), remaining.end());

        uint32_t bestNode = remaining[0];
        uint32_t bestFill = std::numeric_limits<uint32_t>::max();

        for (uint32_t v : remaining) {
            const auto& nbrSet = adj[v];
            std::vector<uint32_t> nbrsVec(nbrSet.begin(), nbrSet.end());
            std::sort(nbrsVec.begin(), nbrsVec.end());

            uint32_t fillCount = 0;
            for (size_t i = 0; i < nbrsVec.size(); ++i) {
                for (size_t j = i + 1; j < nbrsVec.size(); ++j) {
                    if (!adj[nbrsVec[i]].count(nbrsVec[j]))
                        ++fillCount;
                }
            }

            if (fillCount < bestFill || (fillCount == bestFill && v < bestNode)) {
                bestFill = fillCount;
                bestNode = v;
            }
        }

        // Record bond dimension = clique size = |N(bestNode)| + 1
        const auto& nbrsSet = adj[bestNode];
        uint32_t cliqueSize = static_cast<uint32_t>(nbrsSet.size()) + 1;
        if (cliqueSize > maxClique) maxClique = cliqueSize;

        result.eliminationStep.push_back(static_cast<uint32_t>(step));
        result.bondDimension.push_back(cliqueSize);

        // Add fill edges: make N(bestNode) a clique
        std::vector<uint32_t> nbrsVec(nbrsSet.begin(), nbrsSet.end());
        for (size_t i = 0; i < nbrsVec.size(); ++i) {
            for (size_t j = i + 1; j < nbrsVec.size(); ++j) {
                adj[nbrsVec[i]].insert(nbrsVec[j]);
                adj[nbrsVec[j]].insert(nbrsVec[i]);
            }
        }

        // Remove bestNode from the graph
        for (uint32_t u : nbrsVec)
            adj[u].erase(bestNode);
        adj.erase(bestNode);

        remaining.erase(std::find(remaining.begin(), remaining.end(), bestNode));
    }

    result.maxBondDimension = maxClique;
    return result;
}
