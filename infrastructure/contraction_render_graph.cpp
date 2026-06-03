#include "contraction_render_graph.h"
#include <algorithm>
#include <unordered_set>

// ============================================================================
// ContractionRenderGraph::analyze
// ============================================================================

RenderGraphSchedule ContractionRenderGraph::analyze(
    const ContractionTree& tree,
    uint64_t slotBytes)
{
    RenderGraphSchedule sched;
    if (tree.nodes.empty()) return sched;

    const auto& nodes = tree.nodes;
    const size_t N = nodes.size();

    // ── Build nodeId -> walk index map ───────────────────────────────────
    uint32_t maxNodeId = 0;
    for (const auto& n : nodes)
        if (n.id > maxNodeId) maxNodeId = n.id;

    std::vector<uint32_t> idToWalk(maxNodeId + 1, UINT32_MAX);
    for (uint32_t i = 0; i < N; ++i)
        idToWalk[nodes[i].id] = i;

    // ── Step 1: Compute per-node lifetimes ──────────────────────────────
    sched.lifetimes.resize(N);
    for (uint32_t i = 0; i < N; ++i) {
        auto& lt = sched.lifetimes[i];
        lt.treeNodeId = nodes[i].id;
        lt.physicalSlot = UINT32_MAX;

        bool isLeaf = (nodes[i].leftChild == UINT32_MAX &&
                       nodes[i].rightChild == UINT32_MAX);

        // Leaves are bulk-uploaded before the walk (firstWrite = 0).
        // Internal nodes are written when they are processed.
        lt.firstWrite = isLeaf ? 0 : i;

        // Default: never consumed (root or disconnected)
        lt.lastRead = i;
    }

    // Set lastRead = walk index of the parent that reads this node
    for (uint32_t i = 0; i < N; ++i) {
        const auto& node = nodes[i];
        if (node.leftChild == UINT32_MAX) continue;   // leaf — no children

        auto setLastRead = [&](uint32_t childId) {
            if (childId > maxNodeId) return;
            uint32_t cw = idToWalk[childId];
            if (cw == UINT32_MAX) return;
            auto& lt = sched.lifetimes[cw];
            lt.lastRead = std::max(lt.lastRead, i);
        };
        setLastRead(node.leftChild);
        setLastRead(node.rightChild);
    }

    // Root node survives the entire walk (read by CPU after)
    if (tree.rootId != UINT32_MAX && tree.rootId <= maxNodeId) {
        uint32_t rw = idToWalk[tree.rootId];
        if (rw != UINT32_MAX)
            sched.lifetimes[rw].lastRead = static_cast<uint32_t>(N);
    }

    // ── Step 2: Greedy first-fit slot aliasing ──────────────────────────
    // Track when each physical slot becomes available for reuse.
    struct PhysSlot { uint32_t freeAfter = 0; };
    std::vector<PhysSlot> pool;

    sched.nodeIdToPhysical.assign(maxNodeId + 1, UINT32_MAX);

    // Process nodes in walk order (leaves first, root last)
    for (uint32_t i = 0; i < N; ++i) {
        const auto& lt = sched.lifetimes[i];

        // Try to reuse a physical slot whose last consumer has already run
        bool reused = false;
        for (uint32_t s = 0; s < pool.size(); ++s) {
            if (pool[s].freeAfter <= lt.firstWrite) {
                sched.nodeIdToPhysical[lt.treeNodeId] = s;
                sched.lifetimes[i].physicalSlot = s;
                pool[s].freeAfter = lt.lastRead;
                reused = true;
                break;
            }
        }

        if (!reused) {
            uint32_t newSlot = static_cast<uint32_t>(pool.size());
            sched.nodeIdToPhysical[lt.treeNodeId] = newSlot;
            sched.lifetimes[i].physicalSlot = newSlot;
            pool.push_back({lt.lastRead});
        }
    }

    sched.numPhysicalSlots = static_cast<uint32_t>(pool.size());

    // ── Step 3: Wavefront grouping ──────────────────────────────────────
    // Compute tree depth: leaves = 0, internals = max(child depth) + 1.
    std::vector<uint32_t> depth(N, 0);
    uint32_t maxDepth = 0;

    for (uint32_t i = 0; i < N; ++i) {
        const auto& node = nodes[i];
        if (node.leftChild == UINT32_MAX) {
            depth[i] = 0;
            continue;
        }
        uint32_t dl = 0, dr = 0;
        if (node.leftChild <= maxNodeId) {
            uint32_t lw = idToWalk[node.leftChild];
            if (lw != UINT32_MAX) dl = depth[lw];
        }
        if (node.rightChild <= maxNodeId) {
            uint32_t rw = idToWalk[node.rightChild];
            if (rw != UINT32_MAX) dr = depth[rw];
        }
        depth[i] = std::max(dl, dr) + 1;
        if (depth[i] > maxDepth) maxDepth = depth[i];
    }

    // Group internal nodes by depth
    std::vector<WavefrontGroup> wfTemp(maxDepth + 1);
    for (uint32_t d = 0; d <= maxDepth; ++d)
        wfTemp[d].depth = d;

    for (uint32_t i = 0; i < N; ++i) {
        if (nodes[i].leftChild != UINT32_MAX)      // internal only
            wfTemp[depth[i]].walkIndices.push_back(i);
    }

    // Keep only non-empty wavefronts
    for (auto& wf : wfTemp) {
        if (!wf.walkIndices.empty())
            sched.wavefronts.push_back(std::move(wf));
    }

    // ── Step 4: Compute per-pass barriers ───────────────────────────────
    // Before each internal node's merge dispatch, insert a buffer barrier
    // on the physical slots consumed (children) and produced (output).
    sched.barriers.resize(N);

    for (uint32_t i = 0; i < N; ++i) {
        const auto& node = nodes[i];
        if (node.leftChild == UINT32_MAX) continue;

        std::unordered_set<uint32_t> bset;

        auto addBarrier = [&](uint32_t childId) {
            if (childId > maxNodeId) return;
            uint32_t cw = idToWalk[childId];
            if (cw == UINT32_MAX) return;
            uint32_t ps = sched.nodeIdToPhysical[childId];
            if (ps != UINT32_MAX) bset.insert(ps);
        };

        addBarrier(node.leftChild);
        addBarrier(node.rightChild);

        // Output slot
        uint32_t outSlot = sched.nodeIdToPhysical[node.id];
        if (outSlot != UINT32_MAX) bset.insert(outSlot);

        sched.barriers[i].assign(bset.begin(), bset.end());
        std::sort(sched.barriers[i].begin(), sched.barriers[i].end());
    }

    // ── Memory statistics ───────────────────────────────────────────────
    if (slotBytes > 0) {
        sched.unaliasedBytes = static_cast<uint64_t>(N) * slotBytes;
        sched.aliasedBytes   = static_cast<uint64_t>(sched.numPhysicalSlots) * slotBytes;
    }

    sched.valid = true;
    return sched;
}
