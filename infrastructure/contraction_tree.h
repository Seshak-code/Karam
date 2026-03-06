#pragma once

#include "factor_graph.h"
#include "treewidth_analyzer.h"
#include "mpo_builder.h"
#include <cstdint>
#include <vector>

/**
 * contraction_tree.h — Tensor Network Contraction DAG (Phase 5.3.1)
 *
 * Defines the order in which device MPOs are contracted (= eliminated from
 * the factor graph). This is the tensor-network analogue of the elimination
 * tree in sparse LU.
 *
 * Leaf nodes correspond to individual DeviceMPOs.
 * Internal nodes represent the result of contracting two children over shared
 * indices (circuit nodes being eliminated).
 *
 * The contraction ordering comes directly from the Min-Fill elimination
 * ordering computed by TreewidthAnalyzer (Phase A). Walking this ordering:
 *   1. For each eliminated variable v, find all MPO leaf nodes touching v
 *   2. Build a local contraction sub-tree over those MPOs
 *   3. The result becomes a new intermediate node
 */

struct ContractionNode {
    uint32_t id;
    uint32_t leftChild  = UINT32_MAX;  // leaf if == UINT32_MAX
    uint32_t rightChild = UINT32_MAX;
    std::vector<uint32_t> openIndices;       // uncontracted tensor indices (circuit nodes)
    std::vector<uint32_t> contractedIndices; // indices summed over in this contraction
    uint64_t estimatedFlops = 0;             // cost of this contraction step
    uint64_t estimatedMemory = 0;            // size of intermediate tensor
};

struct ContractionTree {
    std::vector<ContractionNode> nodes;
    uint32_t rootId = UINT32_MAX;
    uint64_t totalFlops = 0;
    uint64_t peakMemory = 0;

    // Leaf node IDs (one per MPO, in MPO order)
    std::vector<uint32_t> leafIds;
};

class ContractionTreeBuilder {
public:
    /**
     * Build a contraction tree from the FactorGraph using the Min-Fill
     * elimination ordering (already computed by TreewidthAnalyzer).
     *
     * @param fg   Factor graph providing device→variable adjacency
     * @param tw   Treewidth analysis with elimination ordering
     * @param mpos Device MPOs (leaf data)
     * @return Contraction tree DAG with cost estimates
     */
    static ContractionTree build(
        const FactorGraph& fg,
        const TreewidthAnalysis& tw,
        const std::vector<DeviceMPO>& mpos);
};
