#pragma once

#include "contraction_tree.h"
#include <cstdint>
#include <vector>

/**
 * contraction_render_graph.h — GPU Render Graph for the Contraction Tree
 *                               (Phase 5.7.4)
 *
 * The contraction tree is a natural "render graph" — a DAG of compute passes
 * with well-defined read/write dependencies on workspace buffer slots.
 *
 * This module performs:
 *   1. Resource Lifetime Analysis — first_write / last_read for each slot
 *   2. Slot Aliasing — non-overlapping lifetimes share physical memory
 *   3. Wavefront Grouping — independent passes batched for parallel dispatch
 *   4. Barrier Computation — which physical slots need fencing before each pass
 *
 * Expected VRAM reduction for a 1000-device circuit (tw ~ 12, maxRank = 8):
 *   Without aliasing: 1000 slots x 64 floats x 2 (hi/lo) ~ 512 KB
 *   With aliasing:      ~12 slots x 64 floats x 2         ~   6 KB
 */

// ── Per-tree-node lifetime record ────────────────────────────────────────────

struct SlotLifetime {
    uint32_t treeNodeId;        // ContractionNode this slot belongs to
    uint32_t firstWrite;        // Walk index where slot is first written
    uint32_t lastRead;          // Walk index where slot is last consumed
    uint32_t physicalSlot;      // Aliased physical slot index
};

// ── Wavefront: set of passes that can execute in parallel ────────────────────

struct WavefrontGroup {
    uint32_t depth;                         // Tree depth (0 = leaves, max = root)
    std::vector<uint32_t> walkIndices;      // Walk indices of passes in this group
};

// ── Complete render graph schedule ───────────────────────────────────────────

struct RenderGraphSchedule {
    // nodeId -> physical slot.  Indexed by ContractionNode::id.
    // Size = maxNodeId + 1.  Unmapped entries = UINT32_MAX.
    std::vector<uint32_t> nodeIdToPhysical;

    // Number of physical slots required (after aliasing)
    uint32_t numPhysicalSlots = 0;

    // Per-tree-node lifetime metadata
    std::vector<SlotLifetime> lifetimes;

    // Wavefront groups (depth-sorted).  Each group's passes can be batched
    // into a single GPU command buffer submission.
    std::vector<WavefrontGroup> wavefronts;

    // Per-walk-index: physical slots that must be fenced before the pass
    // begins (a prior pass wrote to the same physical slot via aliasing).
    std::vector<std::vector<uint32_t>> barriers;

    // Memory statistics (populated when slotBytes > 0 in analyze())
    uint64_t unaliasedBytes = 0;
    uint64_t aliasedBytes   = 0;

    // True if the schedule was successfully computed
    bool valid = false;
};

// ── Analyzer ─────────────────────────────────────────────────────────────────

class ContractionRenderGraph {
public:
    /**
     * Analyze the contraction tree and produce an optimized execution schedule.
     *
     * @param tree       The contraction tree DAG
     * @param slotBytes  Size of one workspace slot in bytes (for statistics).
     *                   Pass (maxRank^2 + maxRank) * 2 * sizeof(float) for
     *                   EmulatedF64 hi+lo matrix+RHS.
     * @return Optimized schedule (schedule.valid == true on success)
     */
    static RenderGraphSchedule analyze(const ContractionTree& tree,
                                       uint64_t slotBytes = 0);
};
