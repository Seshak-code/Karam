#include "tn_compiler.h"

// ============================================================================
// TNCompiler::compile
// ============================================================================

TNCompiledProgram TNCompiler::compile(
    const FactorGraph& fg,
    const TreewidthAnalysis& tw,
    const TensorizedBlock& tensors,
    const std::vector<double>& voltages,
    double temp_K)
{
    TNCompiledProgram program;

    if (!tw.isAnalyzed || fg.empty())
        return program;

    // ── Step 1: Bond dimension profiling (diagnostic) ─────────────────────
    program.profile = BondDimensionProfiler::profile(fg, "compiled");

    // ── Step 2: Viability gate ────────────────────────────────────────────
    // If max bond dimension is bounded relative to circuit size, TN is viable.
    // Heuristic: viable if maxBondDimension <= sqrt(circuitSize) * 2
    // This catches pathological star topologies where χ ~ N.
    uint32_t N = program.profile.circuitSize;
    uint32_t maxChi = program.profile.maxBondDimension;
    if (N > 0) {
        double sqrtN = std::sqrt(static_cast<double>(N));
        program.viable = (maxChi <= static_cast<uint32_t>(sqrtN * 2.0 + 5));
    }

    // ── Step 3: Build MPOs ────────────────────────────────────────────────
    program.mpos = MPOBuilder::buildMPOs(tensors, voltages, temp_K);

    // ── Step 4: Build contraction tree ────────────────────────────────────
    program.tree = ContractionTreeBuilder::build(fg, tw, program.mpos);

    // ── Step 5: Estimate workspace sizing (Phase 5.3.3) ──────────────────
    // Walk the contraction tree to compute the maximum intermediate tensor
    // dimension and total workspace memory requirement.
    if (!program.tree.nodes.empty()) {
        uint32_t maxDim = 0;
        uint64_t totalBytes = 0;

        // Leaf contributions
        for (const auto& mpo : program.mpos) {
            uint32_t k = mpo.rank;
            if (k > maxDim) maxDim = k;
            totalBytes += static_cast<uint64_t>(k) * k * sizeof(double)
                        + static_cast<uint64_t>(k) * sizeof(double);
        }

        // Internal node contributions: estimate merged dimension from openIndices
        for (const auto& node : program.tree.nodes) {
            uint32_t dim = static_cast<uint32_t>(node.openIndices.size());
            if (dim > maxDim) maxDim = dim;
            totalBytes += static_cast<uint64_t>(dim) * dim * sizeof(double)
                        + static_cast<uint64_t>(dim) * sizeof(double);
        }

        program.maxIntermediateDim = maxDim;
        program.estimatedWorkspaceBytes = totalBytes;
    }

    return program;
}
