#pragma once
// ============================================================================
// compute/solvers/tf_analysis.h — Transfer Function Analysis API
// ============================================================================
// Phase 4.5
// ============================================================================

#include "engine_api/simulation_dto.h"
#include "acutesim_engine/netlist/circuit.h"
#include <vector>

namespace acutesim {
namespace compute {
namespace solvers {

/**
 * Run transfer-function analysis H(jω) = V_out/V_in.
 *
 * @param netlist     Fully-constructed TensorNetlist.
 * @param dcSolution  DC operating-point voltages (from a prior solveDC call).
 * @param req         Request carrying parameters (fStart, fStop, points,
 *                    outputNode, inputNode) and topologyHash.
 * @return            SimulationResponseDTO with extended.tf populated.
 */
orchestration::SimulationResponseDTO runTFAnalysis(
    const TensorNetlist& netlist,
    const std::vector<double>& dcSolution,
    const orchestration::SimulationRequestDTO& req);

} // namespace solvers
} // namespace compute
} // namespace acutesim
