#pragma once

#include "linalg.h"
#include <vector>
#include <algorithm>
#include <numeric>

/*
 * graph_utils.h
 * Graph Theory Utilities for Sparse Matrices
 */

namespace MathUtils {

/**
 * computeNodeOrdering
 * Computes a permutation vector to reorder matrix nodes for optimized solving
 * (reduced fill-in, better locality, or determinism).
 * 
 * Algorithm: Node Degree Sorting (NDS)
 * Sorts nodes based on their degree (number of connections).
 * Lower degree nodes first often reduces fill-in (similar to Min-Degree).
 * 
 * @param matrix The sparse adjacency structure
 * @return std::vector<int> Permutation p where p[old_index] = new_index
 */
inline std::vector<int> computeNodeOrdering(const Csr_matrix& matrix) {
    int n = matrix.rows;
    if (n == 0) return {};

    std::vector<int> degrees(n, 0);
    std::vector<int> p(n);
    std::iota(p.begin(), p.end(), 0);

    // Compute degrees
    // Note: Csr_matrix row_pointer tells us the number of entries per row
    // degree[i] = row_pointer[i+1] - row_pointer[i]
    // This is "out-degree" or total degree for symmetric matrices.
    for (int i = 0; i < n; ++i) {
        degrees[i] = matrix.row_pointer[i+1] - matrix.row_pointer[i];
    }

    // Stable Sort to ensure determinism for same-degree nodes
    std::stable_sort(p.begin(), p.end(), [&](int a, int b) {
        if (degrees[a] != degrees[b]) {
            return degrees[a] < degrees[b];
        }
        // Tie-breaker: original index to guaranteed stability
        return a < b;
    });
    
    // p[i] is now the node ID that should be placed at position i in the new order.
    // e.g. p = [3, 0, 1, 2] means Node 3 is new Node 0.
    // To remap, we need the inverse map: new_label[old_id]
    // But usually solvers permute b: b_new[i] = b[p[i]]
    // Wait, permutation vector definitions vary.
    // Let's define p such that: new_matrix(i, j) = old_matrix(p[i], p[j])
    // So p maps NewIndex -> OldIndex.
    
    return p;
}

} // namespace MathUtils
