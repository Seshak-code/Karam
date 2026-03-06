#include "netlist_compiler.h"
#include "graph_partitioner.h"
#include "factor_graph.h"
#include "treewidth_analyzer.h"
#include "tn_compiler.h"
#include "../netlist/circuit.h"
#include "../tensors/physics_tensors.h"

/**
 * netlist_compiler.cpp
 *
 * Moves the tensorization logic out of the solver hot-path.
 * Delegates to the existing tensorizeNetlist() for the heavy lifting,
 * then wraps the result with topology metadata.
 */

std::shared_ptr<const CompiledTensorBlock>
NetlistCompiler::compile(const TensorNetlist& netlist)
{
    auto block = std::make_shared<CompiledTensorBlock>();

    // ── 1. Structural → SoA conversion (re-uses existing logic) ─────────
    block->tensors = tensorizeNetlist(netlist);

    // ── 2. Populate metadata ────────────────────────────────────────────
    block->topologyHash = netlist.globalBlock.topologyHash;
    block->nodeCount    = static_cast<size_t>(netlist.numGlobalNodes);

    // ── 3. Snapshot environment (solver must not reach back into netlist)
    block->ambientTempK        = netlist.environment.ambient_temp_K;
    block->globalVoltageScale  = netlist.environment.global_voltage_scale;
    block->monteCarloSeed      = netlist.environment.monte_carlo_seed;
    block->thermalNoiseEnabled = netlist.environment.thermal_noise_enabled;
    block->flickerNoiseEnabled = netlist.environment.flicker_noise_enabled;

    // ── 4. Embed structural netlist (migration shim) ─────────────────────
    //   Internal solver paths (stampAllElements, calculatePhysicalResiduals)
    //   still iterate the structural netlist.  This copy ensures the solver
    //   never holds a pointer to external GUI state.
    block->structural_ = netlist;

    // ── 5. Run graph partitioner for Phase C GPU workgroup assignment ─────
    //   Groups topology-adjacent devices into workgroup-sized batches so that
    //   the Phase B shared-memory voltage cache achieves maximum node reuse.
    {
        GraphPartitioner partitioner;
        block->partitions = partitioner.partition(block->tensors);
    }

    // ── 6. Analytical topology pass (Phase A) ─────────────────────────────
    //   Builds a FactorGraph IR from the SoA tensors — including branch-current
    //   MNA variables for voltage sources and inductors — then runs Min-Fill to
    //   produce an elimination ordering and treewidth estimate.  The result is
    //   stored in CompiledTensorBlock for use by the Phase 5.3 TN compiler.
    //   This pass is purely analytical: it never modifies tensors or partitions.
    FactorGraph fg = FactorGraph::fromTensorizedBlock(
        block->tensors, static_cast<uint32_t>(block->nodeCount));
    block->tnAnalysis = TreewidthAnalyzer::analyze(fg);

    // ── 6.5. TN Compilation (Phase 5.3.1) ─────────────────────────────────
    //   Build MPOs and contraction tree from the factor graph + treewidth data.
    //   Only if treewidth analysis succeeded and circuit is large enough.
    if (block->tnAnalysis.isAnalyzed && block->nodeCount >= 20) {
        block->tnProgram = std::make_shared<TNCompiledProgram>(
            TNCompiler::compile(fg, block->tnAnalysis, block->tensors,
                               std::vector<double>(block->nodeCount, 0.0),
                               block->ambientTempK));
    }

    // ── 7. Propagate interface pin metadata from TensorBlock ──────────────
    //   TensorBlock::InterfacePin (string-based UUID) → CompiledInterfacePin
    //   (typed NetUUID) for use by the hierarchy resolver and future
    //   HierarchyResolver class.
    for (const auto& src : netlist.globalBlock.interfacePins) {
        CompiledInterfacePin dst;
        dst.name      = src.name;
        dst.nodeIndex = src.nodeIndex;
        dst.isPower   = src.isPower;
        dst.netId     = acutesim::components::NetUUID::fromString(src.netUUIDStr);

        if      (src.direction == "Output") dst.direction = acutesim::components::PinDirection::Output;
        else if (src.direction == "BiDir")  dst.direction = acutesim::components::PinDirection::BiDir;
        else if (src.direction == "Power")  dst.direction = acutesim::components::PinDirection::Power;
        else                                dst.direction = acutesim::components::PinDirection::Input;

        block->interfacePins.push_back(dst);
    }

    return block;
}
