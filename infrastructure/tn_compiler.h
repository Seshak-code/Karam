#pragma once

#include "bond_dimension_profiler.h"
#include "contraction_tree.h"
#include "factor_graph.h"
#include "mpo_builder.h"
#include "treewidth_analyzer.h"
#include "../tensors/physics_tensors.h"
#include <memory>
#include <vector>

/**
 * tn_compiler.h — Top-Level TN Compilation (Phase 5.3.1)
 *
 * Orchestrates the full tensor network compilation pipeline:
 *   FactorGraph → MPOs → ContractionTree → TNCompiledProgram
 *
 * The compiled program is stored in CompiledTensorBlock::tnProgram and
 * consumed by the Schur solver (Phase 5.3.2) for domain decomposition.
 *
 * Viability gate: if the BondDimensionProfile shows exponential χ growth,
 * the program is marked as non-viable and the solver falls back to
 * conventional sparse/dense methods.
 */

struct TNCompiledProgram {
    ContractionTree tree;
    std::vector<DeviceMPO> mpos;    // leaf data
    BondDimensionProfile profile;   // diagnostic
    bool viable = false;            // false if χ growth is exponential
    uint64_t estimatedWorkspaceBytes = 0;  // pre-computed workspace size
    uint32_t maxIntermediateDim = 0;       // largest intermediate tensor dimension

    // ─── TBR: Tile Memory Budget ─────────────────────────────────────────
    // Set by TNCompiler from the GraphPartitioner's TileBudget config.
    // Used by ContractionExecutor to verify no intermediate tensor exceeds
    // the SM shared memory limit at runtime.
    uint64_t tileMemoryBudget = 163840;    // default: 163KB (GPU SM limit)
};

class TNCompiler {
public:
    /**
     * Full TN compilation: FactorGraph → MPOs → ContractionTree.
     *
     * @param fg       Factor graph (from FactorGraph::fromTensorizedBlock)
     * @param tw       Treewidth analysis (from TreewidthAnalyzer::analyze)
     * @param tensors  SoA tensor block
     * @param voltages Current node voltages (0-based)
     * @param temp_K   Temperature in Kelvin
     * @return Compiled TN program with viability assessment
     */
    static TNCompiledProgram compile(
        const FactorGraph& fg,
        const TreewidthAnalysis& tw,
        const TensorizedBlock& tensors,
        const std::vector<double>& voltages,
        double temp_K);
};
