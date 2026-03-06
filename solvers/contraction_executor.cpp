#include "contraction_executor.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <unordered_map>

// ============================================================================
// Helper: find position of a variable ID in a sorted index set
// ============================================================================
static int findIndex(uint32_t var, const std::vector<uint32_t>& indexSet) {
    auto it = std::lower_bound(indexSet.begin(), indexSet.end(), var);
    if (it != indexSet.end() && *it == var)
        return static_cast<int>(it - indexSet.begin());
    return -1;
}

// ============================================================================
// Helper: sorted union of two index sets
// ============================================================================
static std::vector<uint32_t> sortedUnion(const std::vector<uint32_t>& a,
                                          const std::vector<uint32_t>& b) {
    std::vector<uint32_t> result;
    result.reserve(a.size() + b.size());
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(result));
    return result;
}

// ============================================================================
// ContractionExecutor::seedLeaves
// ============================================================================

void ContractionExecutor::seedLeaves(
    const ContractionTree& tree,
    const std::vector<DeviceMPO>& mpos,
    TensorWorkspace& workspace)
{
    for (size_t i = 0; i < tree.leafIds.size() && i < mpos.size(); ++i) {
        uint32_t leafId = tree.leafIds[i];
        auto& it = workspace.at(leafId);
        const auto& mpo = mpos[i];

        it.indexSet = mpo.terminalNodes;  // already sorted
        it.dim = mpo.rank;
        it.matrix = mpo.localMatrix;
        it.rhs = mpo.localRHS;

        // Ensure RHS has correct size even if MPO has empty localRHS
        if (it.rhs.size() != it.dim) {
            it.rhs.assign(it.dim, 0.0);
        }

        workspace.updatePeak(
            static_cast<uint64_t>(it.dim) * it.dim * sizeof(double) +
            static_cast<uint64_t>(it.dim) * sizeof(double));
    }
}

// ============================================================================
// ContractionExecutor::mergeAndEliminate
// ============================================================================

void ContractionExecutor::mergeAndEliminate(
    const IntermediateTensor& left,
    const IntermediateTensor& right,
    const std::vector<uint32_t>& contractedIndices,
    IntermediateTensor& result,
    double gmin)
{
    // Step 1: Compute merged index set = sorted_union(left, right)
    std::vector<uint32_t> merged = sortedUnion(left.indexSet, right.indexSet);
    uint32_t mergedDim = static_cast<uint32_t>(merged.size());

    // Step 2: Allocate combined dense block
    std::vector<double> mat(mergedDim * mergedDim, 0.0);
    std::vector<double> rhs(mergedDim, 0.0);

    // Step 3: Scatter left's entries into merged
    for (uint32_t li = 0; li < left.dim; ++li) {
        int mi = findIndex(left.indexSet[li], merged);
        if (mi < 0) continue;
        for (uint32_t lj = 0; lj < left.dim; ++lj) {
            int mj = findIndex(left.indexSet[lj], merged);
            if (mj < 0) continue;
            mat[mi * mergedDim + mj] += left.matrix[li * left.dim + lj];
        }
        rhs[mi] += left.rhs[li];
    }

    // Step 4: Scatter right's entries (accumulate)
    for (uint32_t ri = 0; ri < right.dim; ++ri) {
        int mi = findIndex(right.indexSet[ri], merged);
        if (mi < 0) continue;
        for (uint32_t rj = 0; rj < right.dim; ++rj) {
            int mj = findIndex(right.indexSet[rj], merged);
            if (mj < 0) continue;
            mat[mi * mergedDim + mj] += right.matrix[ri * right.dim + rj];
        }
        rhs[mi] += right.rhs[ri];
    }

    // Step 5: Eliminate contracted variables (Gaussian elimination)
    result.eliminations.clear();
    std::set<uint32_t> toRemove;

    for (uint32_t v : contractedIndices) {
        int p = findIndex(v, merged);
        if (p < 0) continue;  // variable not in merged set

        // GMIN conditioning: add gmin to pivot if near-zero
        if (std::abs(mat[p * mergedDim + p]) < 1e-20) {
            mat[p * mergedDim + p] += gmin;
        }

        double pivot = mat[p * mergedDim + p];
        if (std::abs(pivot) < 1e-30) {
            // Pivot too small even with GMIN — skip this variable
            continue;
        }

        // Save EliminationRecord for back-substitution
        EliminationRecord rec;
        rec.eliminatedVar = v;
        rec.pivotValue = pivot;
        rec.eliminationRHS = rhs[p];

        // Save the elimination row (A[v, neighbors]) for all non-eliminated vars
        for (uint32_t j = 0; j < mergedDim; ++j) {
            if (static_cast<int>(j) == p) continue;
            if (toRemove.count(merged[j])) continue;
            rec.neighborVars.push_back(merged[j]);
            rec.eliminationRow.push_back(mat[p * mergedDim + j]);
        }

        // Perform elimination: A[i][j] -= A[i][p] * (1/pivot) * A[p][j]
        double invPivot = 1.0 / pivot;
        for (uint32_t i = 0; i < mergedDim; ++i) {
            if (static_cast<int>(i) == p) continue;
            double factor = mat[i * mergedDim + p] * invPivot;
            if (std::abs(factor) < 1e-30) continue;
            for (uint32_t j = 0; j < mergedDim; ++j) {
                if (static_cast<int>(j) == p) continue;
                mat[i * mergedDim + j] -= factor * mat[p * mergedDim + j];
            }
            rhs[i] -= factor * rhs[p];
        }

        result.eliminations.push_back(std::move(rec));
        toRemove.insert(v);
    }

    // Step 6: Compact — remove eliminated variables from merged set
    std::vector<uint32_t> remaining;
    for (uint32_t var : merged) {
        if (toRemove.count(var) == 0)
            remaining.push_back(var);
    }

    uint32_t newDim = static_cast<uint32_t>(remaining.size());
    result.indexSet = remaining;
    result.dim = newDim;
    result.matrix.assign(newDim * newDim, 0.0);
    result.rhs.assign(newDim, 0.0);

    // Copy surviving entries from the full merged matrix
    for (uint32_t i = 0; i < newDim; ++i) {
        int mi = findIndex(remaining[i], merged);
        for (uint32_t j = 0; j < newDim; ++j) {
            int mj = findIndex(remaining[j], merged);
            result.matrix[i * newDim + j] = mat[mi * mergedDim + mj];
        }
        result.rhs[i] = rhs[mi];
    }
}

// ============================================================================
// ContractionExecutor::forwardElimination
// ============================================================================

void ContractionExecutor::forwardElimination(
    const ContractionTree& tree,
    TensorWorkspace& workspace,
    double gmin)
{
    // Process nodes in order — by construction, children appear before parents
    // (leaves have lower IDs, internal nodes have higher IDs).
    for (const auto& node : tree.nodes) {
        // Skip leaf nodes (already seeded)
        if (node.leftChild == UINT32_MAX && node.rightChild == UINT32_MAX)
            continue;

        const auto& left = workspace.at(node.leftChild);
        const auto& right = workspace.at(node.rightChild);
        auto& result = workspace.at(node.id);

        mergeAndEliminate(left, right, node.contractedIndices, result, gmin);

        workspace.updatePeak(
            static_cast<uint64_t>(result.dim) * result.dim * sizeof(double) +
            static_cast<uint64_t>(result.dim) * sizeof(double));
    }
}

// ============================================================================
// ContractionExecutor::backSubstitute
// ============================================================================

bool ContractionExecutor::backSubstitute(
    const ContractionTree& tree,
    TensorWorkspace& workspace,
    std::vector<double>& solution)
{
    if (tree.rootId == UINT32_MAX) return false;

    auto& root = workspace.at(tree.rootId);

    // Solve root's small dense system via dense LU with partial pivoting
    if (root.dim > 0 && !root.matrix.empty()) {
        // Build a small CSR matrix for solveLU_Pivoted
        uint32_t n = root.dim;
        Csr_matrix csr;
        csr.rows = static_cast<int>(n);
        csr.cols = static_cast<int>(n);
        csr.row_pointer.push_back(0);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                double val = root.matrix[i * n + j];
                if (std::abs(val) > 1e-30) {
                    csr.values.push_back(val);
                    csr.col_indices.push_back(static_cast<int>(j));
                }
            }
            csr.row_pointer.push_back(static_cast<int>(csr.values.size()));
        }
        csr.nnz = static_cast<int>(csr.values.size());

        SolverResult rootResult = solveLU_Pivoted(csr, root.rhs);
        if (!rootResult.converged) return false;

        // Write root solution into the full solution vector
        for (uint32_t i = 0; i < root.dim; ++i) {
            uint32_t var = root.indexSet[i];
            if (var > 0 && (var - 1) < solution.size()) {
                solution[var - 1] = rootResult.solution[i];
            }
        }
    }

    // Walk tree top-down (reverse order): recover eliminated variables
    // Process nodes from highest ID to lowest (parents before children in ID space,
    // but we need to process in reverse of creation order for top-down recovery).
    for (int idx = static_cast<int>(tree.nodes.size()) - 1; idx >= 0; --idx) {
        const auto& node = tree.nodes[idx];
        const auto& intermed = workspace.at(node.id);

        // Recover eliminated variables in reverse order
        for (int e = static_cast<int>(intermed.eliminations.size()) - 1; e >= 0; --e) {
            const auto& rec = intermed.eliminations[e];

            // x_v = (rhs_v − Σ_j A[v][j]·x_j) / pivot_v
            double sum = 0.0;
            for (size_t j = 0; j < rec.neighborVars.size(); ++j) {
                uint32_t nvar = rec.neighborVars[j];
                double x_j = 0.0;
                if (nvar > 0 && (nvar - 1) < solution.size()) {
                    x_j = solution[nvar - 1];
                }
                sum += rec.eliminationRow[j] * x_j;
            }

            double x_v = (rec.eliminationRHS - sum) / rec.pivotValue;
            uint32_t var = rec.eliminatedVar;
            if (var > 0 && (var - 1) < solution.size()) {
                solution[var - 1] = x_v;
            }
        }
    }

    return true;
}

// ============================================================================
// ContractionExecutor::solve
// ============================================================================

SolverResult ContractionExecutor::solve(
    const TNCompiledProgram& program,
    TensorWorkspace& workspace,
    uint32_t nodeCount,
    double gmin)
{
    SolverResult result;
    result.converged = false;
    result.iterations = 1;
    result.finalResidual = 1e9;

    if (program.tree.rootId == UINT32_MAX ||
        program.tree.nodes.empty() ||
        program.mpos.empty()) {
        return result;
    }

    // Allocate workspace if needed, otherwise just reset
    if (!workspace.isAllocated()) {
        workspace.allocate(program.tree);
    }
    workspace.reset();

    // Phase 1: Seed leaf nodes with MPO data
    seedLeaves(program.tree, program.mpos, workspace);

    // Phase 2: Forward elimination (bottom-up)
    forwardElimination(program.tree, workspace, gmin);

    // Phase 3: Back-substitution (top-down)
    result.solution.assign(nodeCount, 0.0);
    bool ok = backSubstitute(program.tree, workspace, result.solution);
    if (!ok) return result;

    result.converged = true;
    result.finalResidual = 0.0;
    return result;
}
