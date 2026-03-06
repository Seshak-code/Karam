#include "graph_partitioner.h"

#include <algorithm>
#include <cassert>
#include <queue>
#include <unordered_map>
#include <unordered_set>

// ============================================================================
// Internal helpers
// ============================================================================

namespace {

/**
 * Collect the circuit-node indices touched by a single device.
 * Returns a small, stack-local vector (at most 4 nodes per device).
 */
static std::vector<int> nodesForDevice(const TensorizedBlock& t, uint32_t flatIdx,
                                       size_t numDiodes, size_t numMosfets)
{
    if (flatIdx < static_cast<uint32_t>(numDiodes)) {
        size_t i = flatIdx;
        return { t.diodes.node_a[i], t.diodes.node_c[i] };
    }
    flatIdx -= static_cast<uint32_t>(numDiodes);
    if (flatIdx < static_cast<uint32_t>(numMosfets)) {
        size_t i = flatIdx;
        return { t.mosfets.drains[i], t.mosfets.gates[i],
                 t.mosfets.sources[i], t.mosfets.bodies[i] };
    }
    flatIdx -= static_cast<uint32_t>(numMosfets);
    // BJT
    size_t i = flatIdx;
    return { t.bjts.collectors[i], t.bjts.bases[i], t.bjts.emitters[i] };
}

} // anonymous namespace

// ============================================================================
// GraphPartitioner::partition
// ============================================================================

std::vector<TensorPartition> GraphPartitioner::partition(
    const TensorizedBlock& tensors,
    size_t devicesPerWorkgroup) const
{
    const size_t numDiodes  = tensors.diodes.size();
    const size_t numMosfets = tensors.mosfets.size();
    const size_t numBJTs    = tensors.bjts.size();
    const size_t totalDevs  = numDiodes + numMosfets + numBJTs;

    if (totalDevs == 0) return {};

    // ── Step 1: Build node → device adjacency ────────────────────────────
    // For each circuit node (1-based; node 0 = ground, skip), record the flat
    // device indices that touch it.  Use unordered_map for sparse node spaces.
    std::unordered_map<int, std::vector<uint32_t>> nodeToDevices;
    nodeToDevices.reserve(totalDevs * 3); // rough upper bound

    for (uint32_t d = 0; d < static_cast<uint32_t>(totalDevs); ++d) {
        for (int node : nodesForDevice(tensors, d, numDiodes, numMosfets)) {
            if (node > 0) { // skip ground
                nodeToDevices[node].push_back(d);
            }
        }
    }

    // ── Step 2: Build device → neighbour adjacency ───────────────────────
    // Two devices are neighbours if they share ≥1 circuit node.
    // Use a vector<unordered_set> indexed by flat device id.
    std::vector<std::unordered_set<uint32_t>> devNeighbours(totalDevs);

    for (uint32_t d = 0; d < static_cast<uint32_t>(totalDevs); ++d) {
        for (int node : nodesForDevice(tensors, d, numDiodes, numMosfets)) {
            if (node <= 0) continue;
            auto it = nodeToDevices.find(node);
            if (it == nodeToDevices.end()) continue;
            for (uint32_t nbr : it->second) {
                if (nbr != d) devNeighbours[d].insert(nbr);
            }
        }
    }

    // ── Step 3: Compute degree for each device ───────────────────────────
    std::vector<uint32_t> degree(totalDevs);
    for (uint32_t d = 0; d < static_cast<uint32_t>(totalDevs); ++d) {
        degree[d] = static_cast<uint32_t>(devNeighbours[d].size());
    }

    // ── Step 4: Build a sorted seed order (highest degree first) ─────────
    std::vector<uint32_t> seedOrder(totalDevs);
    for (uint32_t i = 0; i < static_cast<uint32_t>(totalDevs); ++i) seedOrder[i] = i;
    std::sort(seedOrder.begin(), seedOrder.end(),
              [&](uint32_t a, uint32_t b) { return degree[a] > degree[b]; });

    // ── Step 5: BFS partitioning ─────────────────────────────────────────
    std::vector<bool> visited(totalDevs, false);
    std::vector<TensorPartition> partitions;
    uint32_t wgId = 0;

    for (uint32_t seed : seedOrder) {
        if (visited[seed]) continue;

        TensorPartition part;
        part.workgroupId = wgId++;

        // BFS queue — use a FIFO queue for true BFS ordering
        std::queue<uint32_t> bfsQ;
        bfsQ.push(seed);
        visited[seed] = true;

        while (!bfsQ.empty() &&
               part.deviceIndices.size() < devicesPerWorkgroup) {
            uint32_t cur = bfsQ.front(); bfsQ.pop();
            part.deviceIndices.push_back(cur);

            // Enqueue unvisited neighbours, respecting the partition capacity.
            for (uint32_t nbr : devNeighbours[cur]) {
                if (!visited[nbr] &&
                    part.deviceIndices.size() + bfsQ.size() < devicesPerWorkgroup) {
                    visited[nbr] = true;
                    bfsQ.push(nbr);
                }
            }
        }

        // Devices that were queued but did not fit are already marked visited;
        // un-mark them so they become seeds for a future partition.
        // (They were never popped and added to deviceIndices.)
        while (!bfsQ.empty()) {
            uint32_t spill = bfsQ.front(); bfsQ.pop();
            visited[spill] = false; // will be re-seeded
        }

        // ── Build node set for this partition ────────────────────────────
        std::unordered_set<uint32_t> nodeSetTmp;
        for (uint32_t d : part.deviceIndices) {
            for (int node : nodesForDevice(tensors, d, numDiodes, numMosfets)) {
                if (node > 0) nodeSetTmp.insert(static_cast<uint32_t>(node));
            }
        }
        part.nodeSet.assign(nodeSetTmp.begin(), nodeSetTmp.end());
        std::sort(part.nodeSet.begin(), part.nodeSet.end());

        partitions.push_back(std::move(part));
    }

    // ── Sanity: every device must appear in exactly one partition ─────────
    assert(([&]() {
        size_t total = 0;
        for (const auto& p : partitions) total += p.deviceIndices.size();
        return total == totalDevs;
    })() && "GraphPartitioner: device coverage invariant violated");

    return partitions;
}
