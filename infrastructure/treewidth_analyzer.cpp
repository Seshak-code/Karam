#include "treewidth_analyzer.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>
#include <limits>
#include <utility>

// ============================================================================
// TreewidthAnalyzer::analyze
// ============================================================================

TreewidthAnalysis TreewidthAnalyzer::analyze(const FactorGraph& fg)
{
    TreewidthAnalysis result;

    // Trivial/degenerate cases — nothing to analyse.
    if (fg.variableCount() == 0)
        return result;  // isAnalyzed = false

    const size_t n = fg.variableCount();

    // ── Build mutable adjacency from the factor graph ─────────────────────────
    // Initialise adj from co-factor pairs.  Using unordered_set<uint32_t> per
    // node avoids duplicate edges and gives O(1) adjacency queries.
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> adj;
    for (const auto& v : fg.variables)
        adj[v.id]; // ensure all nodes exist, even isolated ones

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

    // ── Acyclicity check: |E| ≤ |V| − 1  ↔  forest (tree/forest) ────────────
    // Count unique edges.  A forest satisfies |E| = |V| − C (C ≥ 1 components).
    {
        size_t totalDegree = 0;
        for (const auto& [v, nbrs] : adj)
            totalDegree += nbrs.size();
        // totalDegree = 2 * |E|; check |E| ≤ n-1
        result.isAcyclic = (totalDegree / 2 <= n - 1);
    }

    // ── Min-Fill elimination loop ─────────────────────────────────────────────
    // Maintain a sorted candidate list for deterministic tie-breaking.
    // Iteration order of unordered containers is never used for ordering logic.
    std::vector<uint32_t> remaining;
    remaining.reserve(n);
    for (const auto& v : fg.variables)
        remaining.push_back(v.id);
    std::sort(remaining.begin(), remaining.end()); // deterministic initial order

    uint32_t maxClique = 0;

    // Track (fill_count, node_id) at elimination time for schurBoundaryCandidates.
    std::vector<std::pair<uint32_t, uint32_t>> fillRecord;
    fillRecord.reserve(n);

    result.eliminationOrdering.reserve(n);

    for (size_t step = 0; step < n; ++step) {
        // Evaluate fill cost for every remaining variable.
        // Sort candidates first to ensure a deterministic scan order, so that
        // the argmin tie-break (by node ID) is independent of hash-map layout.
        std::sort(remaining.begin(), remaining.end());

        uint32_t bestNode = remaining[0];
        uint32_t bestFill = std::numeric_limits<uint32_t>::max();

        for (uint32_t v : remaining) {
            // Collect neighbours as a sorted vector (from adj, which is current).
            const auto& nbrSet = adj[v];
            std::vector<uint32_t> nbrsVec(nbrSet.begin(), nbrSet.end());
            std::sort(nbrsVec.begin(), nbrsVec.end()); // deterministic iteration

            // Count fill edges: pairs (u,w) in nbrsVec not yet adjacent.
            uint32_t fillCount = 0;
            for (size_t i = 0; i < nbrsVec.size(); ++i) {
                for (size_t j = i + 1; j < nbrsVec.size(); ++j) {
                    uint32_t u = nbrsVec[i], w = nbrsVec[j];
                    if (!adj[u].count(w))
                        ++fillCount;
                }
            }

            // argmin fill; tie-break by smallest variable ID (ascending).
            if (fillCount < bestFill || (fillCount == bestFill && v < bestNode)) {
                bestFill = fillCount;
                bestNode = v;
            }
        }

        result.eliminationOrdering.push_back(bestNode);
        fillRecord.push_back({bestFill, bestNode});

        // Clique size = |N(bestNode)| + 1 (bestNode itself).
        const auto& nbrsSet = adj[bestNode];
        uint32_t cliqueSize = static_cast<uint32_t>(nbrsSet.size()) + 1;
        if (cliqueSize > maxClique) maxClique = cliqueSize;

        // Add fill edges: make N(bestNode) a clique.
        std::vector<uint32_t> nbrsVec(nbrsSet.begin(), nbrsSet.end());
        for (size_t i = 0; i < nbrsVec.size(); ++i) {
            for (size_t j = i + 1; j < nbrsVec.size(); ++j) {
                uint32_t u = nbrsVec[i], w = nbrsVec[j];
                adj[u].insert(w);
                adj[w].insert(u);
            }
        }

        // Remove bestNode from the graph.
        for (uint32_t u : nbrsVec)
            adj[u].erase(bestNode);
        adj.erase(bestNode);

        // Remove from remaining (swap-and-pop for O(n) not O(n²)).
        remaining.erase(std::find(remaining.begin(), remaining.end(), bestNode));
    }

    result.estimatedTreewidth = (maxClique > 0) ? maxClique - 1 : 0;

    // ── Schur boundary candidates: top-10% by fill count at elimination ───────
    std::sort(fillRecord.begin(), fillRecord.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    size_t topK = std::max(size_t{1}, n / 10);
    for (size_t i = 0; i < topK && i < fillRecord.size(); ++i)
        result.schurBoundaryCandidates.push_back(fillRecord[i].second);

    result.isAnalyzed = true;
    return result;
}
