#pragma once
// ============================================================================
// compute/solvers/pz_analysis.h — Pole-Zero Analysis API
// ============================================================================
// Phase 4.5
// ============================================================================

#include "engine_api/simulation_dto.h"
#include "acutesim_engine/netlist/circuit.h"
#include "engine_api/isimulation_engine.h"
#include <vector>

namespace acutesim {
namespace compute {
namespace solvers {

/**
 * Run pole-zero analysis.
 */
ENGINE_API orchestration::SimulationResponseDTO runPZAnalysis(
    const TensorNetlist& netlist,
    const std::vector<double>& dcSolution,
    const orchestration::SimulationRequestDTO& req);

} // namespace solvers
} // namespace compute
} // namespace acutesim
