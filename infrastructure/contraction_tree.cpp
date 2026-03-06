#include "contraction_tree.h"
#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>

// ============================================================================
// ContractionTreeBuilder::build
// ============================================================================

ContractionTree ContractionTreeBuilder::build(
    const FactorGraph& fg,
    const TreewidthAnalysis& tw,
    const std::vector<DeviceMPO>& mpos)
{
    ContractionTree tree;

    if (mpos.empty() || !tw.isAnalyzed || tw.eliminationOrdering.empty())
        return tree;

    // ── Step 1: Create leaf nodes for each MPO ────────────────────────────
    // Each leaf represents a single device's local Jacobian tensor.
    // Its openIndices are the device's non-ground terminal nodes.
    for (size_t i = 0; i < mpos.size(); ++i) {
        ContractionNode leaf;
        leaf.id = static_cast<uint32_t>(tree.nodes.size());
        leaf.openIndices = mpos[i].terminalNodes;
        leaf.estimatedMemory = static_cast<uint64_t>(mpos[i].rank) * mpos[i].rank;
        tree.nodes.push_back(std::move(leaf));
        tree.leafIds.push_back(leaf.id);
    }

    // ── Step 2: Build variable → active node mapping ─────────────────────
    // For each circuit-node variable, track which tree nodes currently
    // have that variable in their openIndices.
    std::unordered_map<uint32_t, std::set<uint32_t>> varToNodes;
    for (size_t i = 0; i < tree.nodes.size(); ++i) {
        for (uint32_t idx : tree.nodes[i].openIndices) {
            varToNodes[idx].insert(static_cast<uint32_t>(i));
        }
    }

    // ── Step 3: Walk elimination ordering ─────────────────────────────────
    // For each eliminated variable v:
    //   - Find all active tree nodes that reference v
    //   - Contract them pairwise into a new node
    //   - The new node's openIndices = union of parents' indices minus v
    uint64_t peakMem = 0;

    for (uint32_t v : tw.eliminationOrdering) {
        auto it = varToNodes.find(v);
        if (it == varToNodes.end() || it->second.empty())
            continue;

        // Collect all active nodes touching this variable
        std::vector<uint32_t> touching(it->second.begin(), it->second.end());
        std::sort(touching.begin(), touching.end()); // deterministic order

        if (touching.size() < 2) {
            // Only one node references v — just remove v from its indices
            // (it becomes a free index that gets contracted implicitly)
            uint32_t nodeId = touching[0];
            auto& openIdx = tree.nodes[nodeId].openIndices;
            openIdx.erase(std::remove(openIdx.begin(), openIdx.end(), v),
                          openIdx.end());
            varToNodes.erase(v);
            continue;
        }

        // Pairwise contraction: fold all touching nodes into one
        uint32_t currentId = touching[0];

        for (size_t j = 1; j < touching.size(); ++j) {
            uint32_t otherId = touching[j];

            // Create new internal node
            ContractionNode internal;
            internal.id = static_cast<uint32_t>(tree.nodes.size());
            internal.leftChild = currentId;
            internal.rightChild = otherId;
            internal.contractedIndices.push_back(v);

            // Compute open indices = union of children's indices minus contracted
            std::set<uint32_t> merged;
            for (uint32_t idx : tree.nodes[currentId].openIndices)
                merged.insert(idx);
            for (uint32_t idx : tree.nodes[otherId].openIndices)
                merged.insert(idx);
            merged.erase(v);

            internal.openIndices.assign(merged.begin(), merged.end());

            // Cost estimate: product of dimensions of all open indices
            // Each index has dimension 1 in our Jacobian representation,
            // but the FLOPs scale with the number of indices involved
            uint64_t flops = static_cast<uint64_t>(merged.size() + 1) *
                             static_cast<uint64_t>(merged.size() + 1);
            internal.estimatedFlops = flops;
            tree.totalFlops += flops;

            uint64_t mem = static_cast<uint64_t>(merged.size()) *
                           static_cast<uint64_t>(merged.size());
            internal.estimatedMemory = mem;
            if (mem > peakMem) peakMem = mem;

            tree.nodes.push_back(std::move(internal));
            uint32_t newId = static_cast<uint32_t>(tree.nodes.size() - 1);

            // Update varToNodes: remove old nodes, add new one
            for (uint32_t idx : tree.nodes[newId].openIndices) {
                varToNodes[idx].erase(currentId);
                varToNodes[idx].erase(otherId);
                varToNodes[idx].insert(newId);
            }

            // Also clean up the eliminated variable's set
            // (already removing currentId and otherId from all vars)
            for (uint32_t idx : tree.nodes[currentId].openIndices) {
                if (idx != v) varToNodes[idx].erase(currentId);
            }
            for (uint32_t idx : tree.nodes[otherId].openIndices) {
                if (idx != v) varToNodes[idx].erase(otherId);
            }

            currentId = newId;
        }

        // Remove v from the mapping entirely
        varToNodes.erase(v);
    }

    // ── Step 4: Find root (last created internal node, or merge remaining) ─
    // After all eliminations, there may be multiple disconnected subtrees.
    // Merge them into a single root by contracting pairwise.
    std::set<uint32_t> activeRoots;
    for (const auto& [var, nodeSet] : varToNodes) {
        for (uint32_t nid : nodeSet)
            activeRoots.insert(nid);
    }
    // Also add any nodes that have openIndices but aren't in varToNodes
    // (nodes with no remaining open indices)
    if (activeRoots.empty()) {
        // All variables eliminated — root is the last created node
        if (!tree.nodes.empty())
            tree.rootId = static_cast<uint32_t>(tree.nodes.size() - 1);
    } else {
        std::vector<uint32_t> roots(activeRoots.begin(), activeRoots.end());
        uint32_t currentRoot = roots[0];
        for (size_t j = 1; j < roots.size(); ++j) {
            ContractionNode merge;
            merge.id = static_cast<uint32_t>(tree.nodes.size());
            merge.leftChild = currentRoot;
            merge.rightChild = roots[j];

            std::set<uint32_t> merged;
            for (uint32_t idx : tree.nodes[currentRoot].openIndices)
                merged.insert(idx);
            for (uint32_t idx : tree.nodes[roots[j]].openIndices)
                merged.insert(idx);
            merge.openIndices.assign(merged.begin(), merged.end());

            tree.nodes.push_back(std::move(merge));
            currentRoot = static_cast<uint32_t>(tree.nodes.size() - 1);
        }
        tree.rootId = currentRoot;
    }

    tree.peakMemory = peakMem;
    return tree;
}
