#pragma once
// ============================================================================
// compute/staged_dispatcher.h — Circuit-size aware execution tier selector
// ============================================================================
// Inspects a SimulationRequestDTO and HardwareCapabilities to select the
// most appropriate execution tier for a given circuit. Stateless — the
// returned ExecutionTier is a hint; drivers may override it.
//
// Phase 4.2
// ============================================================================

#include "acutesim_engine/infrastructure/hardware_capabilities.h"
#include "engine_api/simulation_dto.h"

namespace acutesim {
namespace compute {

enum class ExecutionTier {
    SCALAR,         // Generic C++, no SIMD — best for tiny linear circuits
    SIMD_CPU,       // AVX2/AVX-512/NEON kernel dispatch
    MULTICORE_CPU,  // Thread-pool NR — nonlinear heavy circuits
    GPU             // Dawn/WebGPU backend — large matrices
};

struct CircuitMetrics {
    int  nodeCount        = 0;
    int  activeDeviceCount = 0;  // BJTs + MOSFETs + diodes + JFETs
    int  matrixNNZ        = 0;   // Non-zero entries (conservative estimate)
    bool isLinear         = true;
    // Phase 4 Reconcile: batch-aware tiering — number of independent runs
    // When > 10, execution becomes throughput-bound, not latency-bound.
    int  batchSize        = 1;
};

class StagedDispatcher {
public:
    explicit StagedDispatcher(const HardwareCapabilities& hw);

    // Select the best execution tier for the given circuit metrics.
    ExecutionTier selectTier(const CircuitMetrics& m) const;

    // Extract circuit metrics from the request DTO (node list + component types).
    static CircuitMetrics measureCircuit(
        const orchestration::SimulationRequestDTO& req);

private:
    HardwareCapabilities hw_;
};

} // namespace compute
} // namespace acutesim
