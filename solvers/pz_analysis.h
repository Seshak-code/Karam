#pragma once
// ============================================================================
// compute/solvers/pz_analysis.h — Pole-Zero Analysis API
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
 * Run pole-zero analysis.
 *
 * Builds the linearised Y(0) Jacobian at the DC operating point,
 * then computes eigenvalues via power iteration to find poles.
 *
 * @param netlist     Fully-constructed TensorNetlist.
 * @param dcSolution  DC operating-point voltages.
 * @param req         Request carrying parameters and topologyHash.
 * @return            SimulationResponseDTO with extended.pz populated.
 */
orchestration::SimulationResponseDTO runPZAnalysis(
    const TensorNetlist& netlist,
    const std::vector<double>& dcSolution,
    const orchestration::SimulationRequestDTO& req);

} // namespace solvers
} // namespace compute
} // namespace acutesim
