#pragma once
// ============================================================================
// compute/solvers/fourier_analysis.h — Fourier / THD Analysis API
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
 * Run Fourier analysis (DFT + THD) on a transient waveform.
 */
ENGINE_API orchestration::SimulationResponseDTO runFourierAnalysis(
    const TensorNetlist& netlist,
    const std::vector<double>& timeVector,
    const std::vector<double>& waveform,
    const orchestration::SimulationRequestDTO& req);

} // namespace solvers
} // namespace compute
} // namespace acutesim
