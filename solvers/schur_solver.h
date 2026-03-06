#pragma once

#include "../infrastructure/tn_compiler.h"
#include "../infrastructure/graph_partitioner.h"
#include "../math/linalg.h"
#include <cstdint>
#include <vector>

/**
 * schur_solver.h — Hybrid Schur Complement Domain Decomposition (Phase 5.3.2)
 *
 * Decomposes a partitioned circuit into:
 *   - Interior variables (per-partition, eliminated locally)
 *   - Boundary variables (shared between partitions, solved globally)
 *
 * The Schur complement reduces the global system to a smaller boundary
 * system:  S · x_Γ = b̃_Γ  where  S = A_ΓΓ − A_ΓI · A_II⁻¹ · A_IΓ
 *
 * After solving the boundary system, interior variables are recovered via
 * back-substitution:  x_I = A_II⁻¹ · (b_I − A_IΓ · x_Γ)
 *
 * This solver guarantees parity with dense LU within 1e-13.
 * If Schur residual exceeds 1e-6, falls back to sparse/dense LU.
 */

// Threshold for Schur complement solver routing.
// Circuits with estimatedTreewidth <= this AND partitions.size() > 1
// use the Schur decomposition path.
constexpr uint32_t SCHUR_TW_LIMIT = 100;

struct SchurContribution {
    uint32_t partitionId;
    std::vector<uint32_t> boundaryNodes;    // global node IDs at partition boundary
    std::vector<uint32_t> internalNodes;    // global node IDs internal to partition
    std::vector<double> schurMatrix;        // dense |∂Ω|×|∂Ω| boundary matrix
    std::vector<double> schurRHS;           // dense |∂Ω| boundary RHS
    std::vector<double> internalSolution;   // x_I (filled after back-substitution)

    // Cached factorization data for back-substitution
    std::vector<double> A_II_inv_A_IG;      // A_II⁻¹ · A_IΓ  (nI × nΓ)
    std::vector<double> A_II_inv_b_I;       // A_II⁻¹ · b_I   (nI)
};

class SchurSolver {
public:
    /**
     * Phase 1: For each partition, extract interior/boundary decomposition
     * and compute the Schur complement on boundary nodes.
     *
     * S = A_ΓΓ − A_ΓI · A_II⁻¹ · A_IΓ
     *
     * @param partitions     Graph partitions from Phase C
     * @param globalMatrix   Full CSR Jacobian
     * @param globalRHS      Full RHS vector
     * @return Per-partition Schur contributions
     */
    static std::vector<SchurContribution> extractSchurBoundaries(
        const std::vector<TensorPartition>& partitions,
        const Csr_matrix& globalMatrix,
        const std::vector<double>& globalRHS);

    /**
     * Phase 2: Assemble the global Schur system from partition contributions
     * and solve via dense LU.
     *
     * @param contributions  Per-partition Schur data
     * @param totalNodes     Total nodes in the full system
     * @return SolverResult with boundary + full solution
     */
    static SolverResult solveGlobalSchur(
        const std::vector<SchurContribution>& contributions,
        int totalNodes);

    /**
     * Phase 3: Back-substitute to recover internal variables.
     * x_I = A_II⁻¹ · (b_I − A_IΓ · x_Γ)
     *
     * @param contributions   Schur data (internalSolution filled on return)
     * @param boundarySolution Solved boundary voltages
     */
    static void recoverInternalVariables(
        std::vector<SchurContribution>& contributions,
        const std::vector<double>& boundarySolution);

    /**
     * Full Schur solve: extract → solve boundary → recover internals.
     * Returns a SolverResult with the complete solution vector.
     *
     * @param partitions    Graph partitions
     * @param globalMatrix  Full CSR Jacobian
     * @param globalRHS     Full RHS vector
     * @return SolverResult with full solution
     */
    static SolverResult solve(
        const std::vector<TensorPartition>& partitions,
        const Csr_matrix& globalMatrix,
        const std::vector<double>& globalRHS);
};
