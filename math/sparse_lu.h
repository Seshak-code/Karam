#pragma once

#include "linalg.h"
#include <cstdint>
#include <vector>

/**
 * sparse_lu.h — Treewidth-Guided Sparse LU Solver (Phase B)
 *
 * Two-phase sparse direct solver that consumes the Min-Fill elimination
 * ordering from Phase A's TreewidthAnalyzer:
 *
 *   1. Symbolic phase (once per topology change):
 *      Compute the permuted fill-in pattern from the elimination ordering.
 *      Produces a SymbolicFactorization (frozen CSR pattern of L+U
 *      including fill positions).
 *
 *   2. Numeric phase (every NR iteration):
 *      Permute the input matrix, run sparse Crout LU in-place using
 *      the pre-computed structure, then forward/back-substitute.
 *
 * The symbolic factorization IS the tree decomposition; the numeric
 * factorization IS the index contraction. Complexity: O(N * tw^2).
 *
 * No external dependencies (no METIS, no SuiteSparse). Pure C++17.
 */

// Threshold for treewidth-based solver routing.
// Circuits with estimatedTreewidth <= this use sparse LU.
// Above this threshold, fall back to PCG or dense LU.
constexpr uint32_t SPARSE_LU_TW_LIMIT = 50;

struct SymbolicFactorization {
    std::vector<int> perm;          // Min-Fill permutation: perm[new] = old
    std::vector<int> iperm;         // Inverse: iperm[old] = new
    CachedCsrPattern filledPattern; // CSR pattern of L+U including fill positions
    int estimatedFillIn = 0;        // Number of fill entries added beyond original
};

/**
 * Symbolic phase: compute fill-in pattern from elimination ordering.
 *
 * Walks the elimination ordering, simulates fill by tracking neighbour
 * sets (same algorithm as Min-Fill in treewidth_analyzer.cpp, but emits
 * the CSR pattern instead of just counting).
 *
 * Called once per topology change (when topologyHash differs).
 *
 * @param originalPattern  CSR pattern of the original Jacobian
 * @param eliminationOrdering  Min-Fill variable ordering from Phase A
 *                             (indices are 0-based solver node IDs)
 * @return SymbolicFactorization with permutation vectors and filled pattern
 */
SymbolicFactorization symbolicFactorize(
    const CachedCsrPattern& originalPattern,
    const std::vector<uint32_t>& eliminationOrdering);

/**
 * Numeric phase: solve Ax = b using pre-computed symbolic structure.
 *
 * Permutes the input matrix using perm, runs Crout LU with partial
 * pivoting (constrained to the fill pattern), then forward/back-substitutes.
 *
 * @param matrix   CSR Jacobian (original, unpermuted)
 * @param rhs      Right-hand side vector
 * @param symbolic Pre-computed symbolic factorization
 * @return SolverResult with solution, residual, convergence flag
 */
SolverResult solveSparse_LU(
    const Csr_matrix& matrix,
    const std::vector<double>& rhs,
    const SymbolicFactorization& symbolic);
