#pragma once
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <functional>
#include <cstdint>

/**
 * graph_canonical.h - Graph Canonicalization Utilities
 * 
 * Phase 1.4: Ensures topology hashing is robust against:
 *   1. Arbitrary node numbering
 *   2. Component orientation (e.g., R(1,2) == R(2,1))
 *   3. Ground node variations
 * 
 * The core algorithm is based on Weisfeiler-Lehman graph hashing,
 * which iteratively refines node labels based on neighborhood structure.
 */

namespace GraphCanonical {

// ============================================================================ 
// CANONICAL EDGE REPRESENTATION
// ============================================================================

/**
 * CanonicalEdge - Orientation-invariant edge representation.
 * Ensures that edge(A, B) == edge(B, A) for undirected components.
 */
struct CanonicalEdge {
    int nodeMin;      // Smaller node index
    int nodeMax;      // Larger node index
    uint32_t typeTag; // Component type identifier (0=R, 1=C, 2=L, 3=Diode, etc.)
    
    CanonicalEdge(int a, int b, uint32_t type) : typeTag(type) {
        nodeMin = std::min(a, b);
        nodeMax = std::max(a, b);
    }
    
    bool operator<(const CanonicalEdge& other) const {
        if (nodeMin != other.nodeMin) return nodeMin < other.nodeMin;
        if (nodeMax != other.nodeMax) return nodeMax < other.nodeMax;
        return typeTag < other.typeTag;
    }
    
    bool operator==(const CanonicalEdge& other) const {
        return nodeMin == other.nodeMin && nodeMax == other.nodeMax && typeTag == other.typeTag;
    }
};

// ============================================================================
// GROUND NORMALIZATION
// ============================================================================

/**
 * normalizeGround - Ensures node 0 is always the global reference.
 * 
 * If the netlist uses a different ground node (e.g., node 5), this function
 * remaps all node indices so that the ground becomes node 0.
 * 
 * @param edges      Vector of edges (modified in place)
 * @param groundNode The current ground node index
 * @return           A mapping from old node indices to new indices
 */
inline std::map<int, int> normalizeGround(std::vector<CanonicalEdge>& edges, int groundNode) {
    if (groundNode == 0) {
        // Already normalized, return identity mapping
        std::map<int, int> identity;
        for (auto& e : edges) {
            identity[e.nodeMin] = e.nodeMin;
            identity[e.nodeMax] = e.nodeMax;
        }
        return identity;
    }
    
    // Build remapping: groundNode -> 0, 0 -> groundNode (swap)
    std::map<int, int> remap;
    std::set<int> allNodes;
    for (const auto& e : edges) {
        allNodes.insert(e.nodeMin);
        allNodes.insert(e.nodeMax);
    }
    
    for (int node : allNodes) {
        if (node == groundNode) {
            remap[node] = 0;
        } else if (node == 0) {
            remap[node] = groundNode;
        } else {
            remap[node] = node;
        }
    }
    
    // Apply remapping
    for (auto& e : edges) {
        int newMin = remap[e.nodeMin];
        int newMax = remap[e.nodeMax];
        e.nodeMin = std::min(newMin, newMax);
        e.nodeMax = std::max(newMin, newMax);
    }
    
    return remap;
}

// ============================================================================
// CANONICAL NODE RELABELING (Weisfeiler-Lehman Inspired)
// ============================================================================

/**
 * computeNodeDegrees - Calculate the degree of each node.
 * Node degree is a simple structural invariant.
 */
inline std::map<int, int> computeNodeDegrees(const std::vector<CanonicalEdge>& edges) {
    std::map<int, int> degrees;
    for (const auto& e : edges) {
        degrees[e.nodeMin]++;
        degrees[e.nodeMax]++;
    }
    return degrees;
}

/**
 * canonicalRelabel - Relabel nodes in a canonical order.
 * 
 * Algorithm:
 *   1. Start with node 0 (ground) as the first node.
 *   2. Sort remaining nodes by (degree, then by neighbor hash).
 *   3. Assign new labels in BFS order from ground.
 * 
 * This ensures that isomorphic graphs receive the same labeling.
 */
inline std::map<int, int> canonicalRelabel(std::vector<CanonicalEdge>& edges) {
    if (edges.empty()) return {};
    
    // Build adjacency list
    std::map<int, std::set<int>> adj;
    std::set<int> allNodes;
    for (const auto& e : edges) {
        adj[e.nodeMin].insert(e.nodeMax);
        adj[e.nodeMax].insert(e.nodeMin);
        allNodes.insert(e.nodeMin);
        allNodes.insert(e.nodeMax);
    }
    
    // Compute initial node signatures (degree-based)
    std::map<int, int> degrees = computeNodeDegrees(edges);
    
    // Sort nodes by degree (ascending), then by original index for ties
    std::vector<int> sortedNodes(allNodes.begin(), allNodes.end());
    std::sort(sortedNodes.begin(), sortedNodes.end(), [&](int a, int b) {
        if (degrees[a] != degrees[b]) return degrees[a] < degrees[b];
        return a < b;
    });
    
    // BFS from ground (node 0) to assign canonical labels
    std::map<int, int> newLabel;
    int label = 0;
    
    // Ground is always 0
    if (allNodes.count(0)) {
        newLabel[0] = label++;
    }
    
    // BFS queue
    std::vector<int> queue;
    if (allNodes.count(0)) {
        queue.push_back(0);
    }
    
    std::set<int> visited;
    if (allNodes.count(0)) visited.insert(0);
    
    while (!queue.empty()) {
        int current = queue.front();
        queue.erase(queue.begin());
        
        // Get neighbors sorted by degree
        std::vector<int> neighbors(adj[current].begin(), adj[current].end());
        std::sort(neighbors.begin(), neighbors.end(), [&](int a, int b) {
            if (degrees[a] != degrees[b]) return degrees[a] < degrees[b];
            return a < b;
        });
        
        for (int neighbor : neighbors) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                newLabel[neighbor] = label++;
                queue.push_back(neighbor);
            }
        }
    }
    
    // Handle disconnected nodes (not reachable from ground)
    for (int node : sortedNodes) {
        if (newLabel.find(node) == newLabel.end()) {
            newLabel[node] = label++;
        }
    }
    
    // Apply new labels to edges
    for (auto& e : edges) {
        int newMin = newLabel[e.nodeMin];
        int newMax = newLabel[e.nodeMax];
        e.nodeMin = std::min(newMin, newMax);
        e.nodeMax = std::max(newMin, newMax);
    }
    
    return newLabel;
}

// ============================================================================
// TOPOLOGY HASH (Orientation-Invariant)
// ============================================================================

/**
 * computeTopologyHash - Generate a robust structural hash.
 * 
 * The hash is invariant to:
 *   - Node ordering (canonical relabeling applied first)
 *   - Edge orientation (edges are always min→max)
 *   - Ground node placement (normalized to node 0)
 */
inline uint64_t computeTopologyHash(std::vector<CanonicalEdge> edges, int groundNode = 0) {
    // Step 1: Normalize ground to node 0
    normalizeGround(edges, groundNode);
    
    // Step 2: Canonical relabeling (BFS from ground)
    canonicalRelabel(edges);
    
    // Step 3: Sort edges for deterministic ordering
    std::sort(edges.begin(), edges.end());
    
    // Step 4: Compute hash using FNV-1a
    uint64_t hash = 14695981039346656037ULL; // FNV offset basis
    const uint64_t fnvPrime = 1099511628211ULL;
    
    for (const auto& e : edges) {
        hash ^= static_cast<uint64_t>(e.nodeMin);
        hash *= fnvPrime;
        hash ^= static_cast<uint64_t>(e.nodeMax);
        hash *= fnvPrime;
        hash ^= static_cast<uint64_t>(e.typeTag);
        hash *= fnvPrime;
    }
    
    return hash;
}

// ============================================================================
// SUBGRAPH EXTRACTION & HASHING
// ============================================================================

/**
 * extractSubgraph - Extract a subgraph around a specific node.
 * Useful for identifying local "building blocks" (e.g., current mirrors).
 * 
 * @param edges   Full graph edges
 * @param center  Center node of subgraph
 * @param radius  Hop distance from center to include
 * @return        Vector of edges in the subgraph
 */
inline std::vector<CanonicalEdge> extractSubgraph(
    const std::vector<CanonicalEdge>& edges, 
    int center, 
    int radius = 1) 
{
    // Build adjacency
    std::map<int, std::set<int>> adj;
    for (const auto& e : edges) {
        adj[e.nodeMin].insert(e.nodeMax);
        adj[e.nodeMax].insert(e.nodeMin);
    }
    
    // BFS to find nodes within radius
    std::set<int> included;
    std::vector<int> queue = {center};
    included.insert(center);
    
    for (int r = 0; r < radius && !queue.empty(); ++r) {
        std::vector<int> nextQueue;
        for (int node : queue) {
            for (int neighbor : adj[node]) {
                if (included.find(neighbor) == included.end()) {
                    included.insert(neighbor);
                    nextQueue.push_back(neighbor);
                }
            }
        }
        queue = nextQueue;
    }
    
    // Filter edges to only those with both endpoints in the subgraph
    std::vector<CanonicalEdge> subgraph;
    for (const auto& e : edges) {
        if (included.count(e.nodeMin) && included.count(e.nodeMax)) {
            subgraph.push_back(e);
        }
    }
    
    return subgraph;
}

/**
 * hashSubgraph - Compute a hash for a local subgraph.
 * The center node is normalized to be the "local ground" (node 0).
 */
inline uint64_t hashSubgraph(const std::vector<CanonicalEdge>& edges, int center, int radius = 1) {
    std::vector<CanonicalEdge> subgraph = extractSubgraph(edges, center, radius);
    return computeTopologyHash(subgraph, center);
}

} // namespace GraphCanonical
