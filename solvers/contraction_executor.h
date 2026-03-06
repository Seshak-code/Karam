#pragma once

#include "../infrastructure/tn_compiler.h"
#include "../math/linalg.h"
#include "tensor_workspace.h"
#include <cstdint>
#include <vector>

/**
 * contraction_executor.h — Tensor Contraction Execution Engine (Phase 5.3.3)
 *
 * Walks the ContractionTree bottom-up, performing dense Schur elimination
 * at each internal node. This is the CPU reference implementation for
 * tensor-native MNA solving.
 *
 * Algorithm:
 *   1. seedLeaves()        — Copy MPO localMatrix + localRHS into workspace
 *   2. forwardElimination() — Bottom-up: merge children, eliminate contracted
 *                             variables via Gaussian elimination
 *   3. backSubstitute()    — Top-down: solve root's small system, then
 *                             recover eliminated variables from saved
 *                             EliminationRecords
 *
 * Mathematical operation per contraction step:
 *   Given children L and R sharing variable v:
 *     merged = L ⊕ R  (accumulate into union of index sets)
 *     S = A_NN − A_Nv · (1/A_vv) · A_vN     (Schur elimination)
 *     rhs_N −= A_Nv · (1/A_vv) · rhs_v
 *
 * Complexity: O(N · tw²) where tw = treewidth.
 *
 * Routing threshold: circuits with tw ≤ CONTRACTION_TW_LIMIT and a viable
 * TN program are eligible. Falls back to sparse/dense LU on failure.
 */

constexpr uint32_t CONTRACTION_TW_LIMIT = 80;

class ContractionExecutor {
public:
    /**
     * Full contraction solve: seed → forward elimination → back-substitution.
     *
     * @param program    Compiled TN program (tree + MPOs with localRHS)
     * @param workspace  Pre-allocated workspace (reset before use)
     * @param nodeCount  Total MNA nodes in the circuit
     * @param gmin       Minimum conductance for conditioning (default 1e-12)
     * @return SolverResult with complete solution vector
     */
    static SolverResult solve(
        const TNCompiledProgram& program,
        TensorWorkspace& workspace,
        uint32_t nodeCount,
        double gmin = 1e-12);

private:
    /**
     * Copy each MPO's localMatrix + localRHS into the workspace leaf nodes.
     */
    static void seedLeaves(const ContractionTree& tree,
                           const std::vector<DeviceMPO>& mpos,
                           TensorWorkspace& workspace);

    /**
     * Bottom-up traversal: for each internal node, merge children's
     * intermediates and eliminate contracted variables.
     */
    static void forwardElimination(const ContractionTree& tree,
                                   TensorWorkspace& workspace,
                                   double gmin);

    /**
     * Merge two IntermediateTensors and eliminate contractedIndices.
     */
    static void mergeAndEliminate(const IntermediateTensor& left,
                                  const IntermediateTensor& right,
                                  const std::vector<uint32_t>& contractedIndices,
                                  IntermediateTensor& result,
                                  double gmin);

    /**
     * Top-down traversal: solve root's small dense system, then recover
     * each eliminated variable from saved EliminationRecords.
     *
     * @return true if back-substitution succeeded
     */
    static bool backSubstitute(const ContractionTree& tree,
                               TensorWorkspace& workspace,
                               std::vector<double>& solution);
};
