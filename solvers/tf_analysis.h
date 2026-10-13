#pragma once
// ============================================================================
// compute/solvers/tf_analysis.h — Transfer Function Analysis API
// ============================================================================
// Phase 4.5
// ============================================================================

#include "engine_api/simulation_dto.h"
#include "netlist/circuit.h"
#include "engine_api/isimulation_engine.h"
#include <vector>

class CircuitSim;

namespace acutesim {
namespace compute {
namespace solvers {

/**
 * Run transfer-function analysis H(jω) = V_out/V_in. `sim` must hold the
 * converged DC operating point of `netlist` (call solveDC first).
 */
ENGINE_API orchestration::SimulationResponseDTO runTFAnalysis(
    CircuitSim& sim, TensorNetlist& netlist, const orchestration::SimulationRequestDTO& req);

} // namespace solvers
} // namespace compute
} // namespace acutesim
