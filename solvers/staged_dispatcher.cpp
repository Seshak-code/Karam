#include "acutesim_engine/solvers/staged_dispatcher.h"
#include <algorithm>

namespace acutesim {
namespace compute {

StagedDispatcher::StagedDispatcher(const HardwareCapabilities& hw) : hw_(hw) {}

ExecutionTier StagedDispatcher::selectTier(const CircuitMetrics& m) const {
    // Phase 4 Reconcile: batch-aware tiering.
    // When batchSize > 10, the workload is throughput-bound (Monte Carlo / sweep).
    // Dispatch to GPU or MULTICORE regardless of individual circuit size.
    if (m.batchSize > 10) {
        if (hw_.hasGPU) return ExecutionTier::GPU;
        return ExecutionTier::MULTICORE_CPU;
    }

    // Small linear circuits: scalar is fastest (no SIMD setup overhead)
    if (m.nodeCount < 30 && m.isLinear)
        return ExecutionTier::SCALAR;

    // Large linear systems: GPU excels at dense/sparse linear algebra
    if (m.isLinear && m.nodeCount > 500 && hw_.hasGPU)
        return ExecutionTier::GPU;

    // Nonlinear with many active devices: MULTICORE (NR iterations dominate)
    if (m.activeDeviceCount > 50)
        return ExecutionTier::MULTICORE_CPU;

    // Medium circuits with AVX2 or better
    if (m.nodeCount < 200 &&
        static_cast<int>(hw_.simdTier) >=
            static_cast<int>(HardwareCapabilities::SIMDTier::AVX2))
        return ExecutionTier::SIMD_CPU;

    // Large without GPU — multicore for scalability
    if (m.nodeCount >= 200 && !hw_.hasGPU)
        return ExecutionTier::MULTICORE_CPU;

    // Large with GPU
    if (m.nodeCount >= 200 && hw_.hasGPU)
        return ExecutionTier::GPU;

    // Default fallback
    return ExecutionTier::SIMD_CPU;
}

CircuitMetrics StagedDispatcher::measureCircuit(
    const orchestration::SimulationRequestDTO& req)
{
    CircuitMetrics m;

    // nodeNames field doesn't exist in SimulationRequestDTO. We derive node
    // count from the source payload metadata when available.
    // For DTO-level estimation we rely on what is present in the DTO.
    // The Verilog-A parser fills this during actual simulation; here we
    // provide a best-effort estimate for tier selection purposes only.

    // Count nonlinear devices from the legacy JSON path is not viable in
    // the DTO; the source field carries raw text. We therefore use a
    // conservative scalar default that will be overridden at solver time.
    m.nodeCount        = 0;
    m.activeDeviceCount = 0;
    m.isLinear         = true;
    m.matrixNNZ        = 0;

    (void)req; // req is reserved for future structured DTO fields
    return m;
}

} // namespace compute
} // namespace acutesim
