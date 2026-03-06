#include "sparse_lu.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// symbolicFactorize — Compute fill-in pattern from Min-Fill ordering
// ============================================================================

SymbolicFactorization symbolicFactorize(
    const CachedCsrPattern& originalPattern,
    const std::vector<uint32_t>& eliminationOrdering)
{
    SymbolicFactorization result;
    const int n = originalPattern.rows;

    if (n == 0 || eliminationOrdering.empty()) {
        return result;
    }

    assert(static_cast<int>(eliminationOrdering.size()) == n &&
           "Elimination ordering size must match matrix dimension");

    // Build permutation vectors.
    // eliminationOrdering[i] = the original variable eliminated at step i.
    // perm[new_index] = old_index   (new → old mapping)
    // iperm[old_index] = new_index  (old → new mapping)
    result.perm.resize(n);
    result.iperm.resize(n);
    for (int i = 0; i < n; ++i) {
        int oldIdx = static_cast<int>(eliminationOrdering[i]);
        result.perm[i] = oldIdx;
        result.iperm[oldIdx] = i;
    }

    // Build mutable adjacency from the original CSR pattern (undirected).
    std::vector<std::unordered_set<int>> adj(n);
    for (int row = 0; row < n; ++row) {
        for (int k = originalPattern.row_pointer[row];
             k < originalPattern.row_pointer[row + 1]; ++k) {
            int col = originalPattern.col_indices[k];
            if (col != row) {
                adj[row].insert(col);
                adj[col].insert(row);
            }
        }
    }

    // Simulate elimination in the original-index graph, collecting the
    // full L+U non-zero pattern in the permuted ordering.
    //
    // Key insight: at step `step`, variable v's remaining neighbors map to
    // permuted columns > step (U entries for row step). But those same
    // neighbors also need L entries: column step in THEIR rows. We build
    // the pattern symmetrically: when row `step` gets column `permCol`,
    // row `permCol` also gets column `step`.

    int fillCount = 0;

    // Use a set per row to collect non-zero columns (avoids duplicates).
    std::vector<std::unordered_set<int>> rowSets(n);
    // Diagonal is always present.
    for (int i = 0; i < n; ++i) rowSets[i].insert(i);

    for (int step = 0; step < n; ++step) {
        int v = static_cast<int>(eliminationOrdering[step]);

        // Collect current neighbors (may include fill from earlier steps).
        std::vector<int> nbrs(adj[v].begin(), adj[v].end());

        // Record U entries: row `step`, columns = iperm[each remaining neighbor]
        // Record L entries: for each neighbor u, row iperm[u], column `step`
        for (int u : nbrs) {
            int permCol = result.iperm[u];
            rowSets[step].insert(permCol);      // U entry in row step
            rowSets[permCol].insert(step);       // L entry in neighbor's row
        }

        // Add fill edges: make neighbors of v a clique.
        for (size_t i = 0; i < nbrs.size(); ++i) {
            for (size_t j = i + 1; j < nbrs.size(); ++j) {
                int u = nbrs[i], w = nbrs[j];
                if (!adj[u].count(w)) {
                    adj[u].insert(w);
                    adj[w].insert(u);
                    ++fillCount;
                }
            }
        }

        // Remove v from the graph.
        for (int u : nbrs) {
            adj[u].erase(v);
        }
        adj[v].clear();
    }

    result.estimatedFillIn = fillCount;

    // Build the filled CSR pattern from rowSets (sorted columns per row).
    result.filledPattern.rows = n;
    result.filledPattern.cols = n;
    result.filledPattern.row_pointer.resize(n + 1, 0);

    int totalNnz = 0;
    for (int i = 0; i < n; ++i) {
        totalNnz += static_cast<int>(rowSets[i].size());
    }
    result.filledPattern.nnz = totalNnz;
    result.filledPattern.col_indices.reserve(totalNnz);

    for (int i = 0; i < n; ++i) {
        result.filledPattern.row_pointer[i] = static_cast<int>(result.filledPattern.col_indices.size());
        // Sort columns for deterministic CSR construction.
        std::vector<int> sortedCols(rowSets[i].begin(), rowSets[i].end());
        std::sort(sortedCols.begin(), sortedCols.end());
        for (int c : sortedCols) {
            result.filledPattern.col_indices.push_back(c);
        }
    }
    result.filledPattern.row_pointer[n] = static_cast<int>(result.filledPattern.col_indices.size());

    return result;
}

// ============================================================================
// solveSparse_LU — Numeric phase using pre-computed symbolic structure
// ============================================================================

SolverResult solveSparse_LU(
    const Csr_matrix& matrix,
    const std::vector<double>& rhs,
    const SymbolicFactorization& symbolic)
{
    const int n = matrix.rows;
    assert(matrix.rows == matrix.cols && "Sparse LU requires a square matrix");
    assert(matrix.rows == static_cast<int>(rhs.size()) && "Dimension mismatch");
    assert(static_cast<int>(symbolic.perm.size()) == n && "Symbolic factorization dimension mismatch");

    // Build a fast lookup: for original matrix, map (row,col) → value.
    // Use the CSR structure for O(1)-amortized lookup per row.
    // We'll scatter into a dense-in-permuted-order workspace, but only
    // for the positions in the filled pattern (sparse).

    // Step 1: Allocate workspace arrays sized to filledPattern.nnz.
    // Store L and U factors interleaved in a single values array matching
    // the filled CSR pattern (L below diagonal, U on+above diagonal).
    const auto& fp = symbolic.filledPattern;
    std::vector<double> luValues(fp.nnz, 0.0);

    // Step 2: Scatter the original matrix values into the permuted pattern.
    // For each original row r, column c with value v:
    //   permuted row = iperm[r], permuted col = iperm[c]
    //   Find position in filledPattern and store.

    // Build col→position maps per row for fast scatter.
    // For each permuted row, map permuted col → index in luValues.
    std::vector<std::unordered_map<int, int>> colToPos(n);
    for (int i = 0; i < n; ++i) {
        for (int k = fp.row_pointer[i]; k < fp.row_pointer[i + 1]; ++k) {
            colToPos[i][fp.col_indices[k]] = k;
        }
    }

    // Scatter original matrix entries.
    for (int origRow = 0; origRow < n; ++origRow) {
        int permRow = symbolic.iperm[origRow];
        for (int k = matrix.row_pointer[origRow]; k < matrix.row_pointer[origRow + 1]; ++k) {
            int origCol = matrix.col_indices[k];
            int permCol = symbolic.iperm[origCol];
            auto it = colToPos[permRow].find(permCol);
            if (it != colToPos[permRow].end()) {
                luValues[it->second] = matrix.values[k];
            }
            // If not found in pattern, this is a zero that the pattern
            // doesn't track (shouldn't happen for correctly built pattern).
        }
    }

    // Step 3: Permute the RHS vector.
    std::vector<double> b(n);
    for (int i = 0; i < n; ++i) {
        b[i] = rhs[symbolic.perm[i]];
    }

    // Step 4: Crout LU factorization in-place on the sparse structure.
    // For each column k (pivot column):
    //   - For rows i > k: L(i,k) = (A(i,k) - sum_{j<k} L(i,j)*U(j,k)) / U(k,k)
    //   - For columns j >= k in row k: U(k,j) = A(k,j) - sum_{m<k} L(k,m)*U(m,j)
    //
    // With partial pivoting within the filled pattern.

    // For efficiency, use a dense column workspace for each pivot step.
    std::vector<double> colWork(n, 0.0);
    std::vector<int> pivotPerm(n);
    for (int i = 0; i < n; ++i) pivotPerm[i] = i;

    // Phase 2.9: Rank-revealing pivot tracking
    constexpr double kPivotThreshold = 1e-18;
    double smallestPivot = 1e300;
    int smallestPivotRow = -1;

    // Dense workspace approach: convert filled pattern to dense for
    // factorization (the filled pattern is typically much smaller than N^2
    // for low-treewidth circuits, but for correctness and robustness we
    // use a dense workspace with the permuted ordering).
    //
    // For circuits where the filled pattern is truly sparse, a more
    // sophisticated supernodal approach would be faster, but that's
    // reserved for Phase 5.3.4.

    std::vector<double> A(static_cast<size_t>(n) * n, 0.0);

    // Scatter luValues into dense A.
    for (int i = 0; i < n; ++i) {
        for (int k = fp.row_pointer[i]; k < fp.row_pointer[i + 1]; ++k) {
            int j = fp.col_indices[k];
            A[i * n + j] = luValues[k];
        }
    }

    // LU decomposition with partial pivoting (same as solveLU_Pivoted
    // but operating on the permuted system).
    for (int i = 0; i < n; ++i) {
        // Find pivot for column i
        double maxVal = 0.0;
        int maxRow = i;
        for (int k = i; k < n; ++k) {
            double val = std::abs(A[k * n + i]);
            if (val > maxVal) {
                maxVal = val;
                maxRow = k;
            } else if (val == maxVal && k < maxRow) {
                maxRow = k;
            }
        }

        // Track smallest pivot
        if (maxVal < smallestPivot) {
            smallestPivot = maxVal;
            smallestPivotRow = i;
        }

        // Check for singularity
        if (maxVal < 1e-25) {
            std::cerr << "Sparse LU: Singular matrix at permuted row " << i
                      << " (maxVal=" << maxVal << ")\n";
            SolverResult failResult;
            failResult.solution.assign(n, 0.0);
            failResult.finalResidual = 1.0;
            failResult.iterations = 0;
            failResult.converged = false;
            failResult.rankDeficient = true;
            failResult.rankDeficientRow = smallestPivotRow;
            failResult.smallestPivot = smallestPivot;
            return failResult;
        }

        // Swap rows if necessary
        if (maxRow != i) {
            std::swap(pivotPerm[i], pivotPerm[maxRow]);
            for (int k = 0; k < n; ++k) {
                std::swap(A[i * n + k], A[maxRow * n + k]);
            }
            std::swap(b[i], b[maxRow]);
        }

        // Eliminate: only process positions that are in the filled pattern
        // (or their fill-in). Since we're using a dense workspace, we
        // eliminate fully — the sparsity saving comes from the permutation
        // minimizing the number of non-zeros that appear.
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(A[j * n + i]) < 1e-30) continue; // skip true zeros
            A[j * n + i] /= A[i * n + i];
            for (int k = i + 1; k < n; ++k) {
                A[j * n + k] -= A[j * n + i] * A[i * n + k];
            }
        }
    }

    // Step 5: Forward substitution (Ly = b)
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        y[i] = b[i];
        for (int j = 0; j < i; ++j) {
            y[i] -= A[i * n + j] * y[j];
        }
    }

    // Step 6: Backward substitution (Ux = y)
    std::vector<double> xPerm(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        xPerm[i] = y[i];
        for (int j = i + 1; j < n; ++j) {
            xPerm[i] -= A[i * n + j] * xPerm[j];
        }
        xPerm[i] /= A[i * n + i];
    }

    // Step 7: Unpermute the solution back to original ordering.
    // xPerm is in permuted order; x[perm[i]] = xPerm[i]
    std::vector<double> x(n, 0.0);
    for (int i = 0; i < n; ++i) {
        x[symbolic.perm[i]] = xPerm[i];
    }

    // Step 8: Verify residual (||Ax - b|| in original space)
    std::vector<double> check(n, 0.0);
    matrixMultiplication(matrix, x.data(), check.data());
    double residual = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = check[i] - rhs[i];
        residual += err * err;
    }
    residual = std::sqrt(residual);

    SolverResult result;
    result.solution = std::move(x);
    result.finalResidual = residual;
    result.iterations = 1;
    result.converged = true;
    result.smallestPivot = smallestPivot;
    result.rankDeficientRow = smallestPivotRow;
    result.rankDeficient = (smallestPivot < kPivotThreshold);
    return result;
}
