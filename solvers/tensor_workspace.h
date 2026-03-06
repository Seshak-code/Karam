#pragma once

#include "../infrastructure/contraction_tree.h"
#include <cstdint>
#include <vector>

/**
 * tensor_workspace.h — Pre-Allocated Workspace for Contraction Execution
 *                      (Phase 5.3.3)
 *
 * Provides storage for intermediate contraction results and elimination
 * records needed for back-substitution. Allocated once per topology,
 * reused across NR iterations via reset().
 *
 * Each IntermediateTensor holds a dense matrix + RHS over a set of MNA
 * variable IDs (the "index set"). During forward elimination, variables
 * are removed from the index set and EliminationRecords are saved so
 * that back-substitution can recover their values.
 */

struct EliminationRecord {
    uint32_t eliminatedVar;                // MNA variable ID that was eliminated
    double pivotValue;                     // A[v][v] at time of elimination
    std::vector<uint32_t> neighborVars;    // remaining variables at elimination time
    std::vector<double> eliminationRow;    // A[v, neighbors] saved before elimination
    double eliminationRHS;                 // rhs[v] saved before elimination
};

struct IntermediateTensor {
    uint32_t nodeId = UINT32_MAX;
    std::vector<uint32_t> indexSet;   // sorted MNA variable IDs
    std::vector<double> matrix;       // row-major dense |indexSet|×|indexSet|
    std::vector<double> rhs;          // |indexSet| RHS vector
    uint32_t dim = 0;
    std::vector<EliminationRecord> eliminations;  // for back-substitution recovery
};

class TensorWorkspace {
public:
    /**
     * Allocate workspace storage for every node in the contraction tree.
     * One IntermediateTensor per tree node (leaves + internals).
     */
    void allocate(const ContractionTree& tree) {
        if (tree.nodes.empty()) return;
        uint32_t maxId = 0;
        for (const auto& node : tree.nodes) {
            if (node.id > maxId) maxId = node.id;
        }
        nodeToIndex_.resize(maxId + 1, UINT32_MAX);
        intermediates_.resize(tree.nodes.size());
        for (size_t i = 0; i < tree.nodes.size(); ++i) {
            intermediates_[i].nodeId = tree.nodes[i].id;
            nodeToIndex_[tree.nodes[i].id] = static_cast<uint32_t>(i);
        }
        peakBytes_ = 0;
    }

    /**
     * Reset all intermediates for reuse (new NR iteration).
     * Clears data but keeps allocations.
     */
    void reset() {
        for (auto& it : intermediates_) {
            it.indexSet.clear();
            it.matrix.clear();
            it.rhs.clear();
            it.dim = 0;
            it.eliminations.clear();
        }
        peakBytes_ = 0;
    }

    IntermediateTensor& at(uint32_t nodeId) {
        return intermediates_[nodeToIndex_[nodeId]];
    }

    const IntermediateTensor& at(uint32_t nodeId) const {
        return intermediates_[nodeToIndex_[nodeId]];
    }

    bool isAllocated() const { return !intermediates_.empty(); }

    uint64_t peakMemoryBytes() const { return peakBytes_; }
    void updatePeak(uint64_t bytes) {
        if (bytes > peakBytes_) peakBytes_ = bytes;
    }

private:
    std::vector<IntermediateTensor> intermediates_;
    std::vector<uint32_t> nodeToIndex_;
    uint64_t peakBytes_ = 0;
};
