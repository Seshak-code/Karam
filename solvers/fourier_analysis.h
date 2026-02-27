#pragma once
// ============================================================================
// compute/solvers/fourier_analysis.h — Fourier / THD Analysis API
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
 * Run Fourier analysis (DFT + THD) on a transient waveform.
 *
 * The caller runs a TRANSIENT simulation first, then passes the resulting
 * probe waveform via req.parameters["probeData"] or through the
 * timeVector + nodeVoltages fields.
 *
 * @param netlist     Fully-constructed TensorNetlist.
 * @param timeVector  Time samples from transient analysis [s].
 * @param waveform    Voltage waveform samples at probe node.
 * @param req         Request carrying fundamental frequency
 *                    (parameters["fundamental"] in Hz) and harmonicCount.
 * @return            SimulationResponseDTO with extended.fourier populated.
 */
orchestration::SimulationResponseDTO runFourierAnalysis(
    const TensorNetlist& netlist,
    const std::vector<double>& timeVector,
    const std::vector<double>& waveform,
    const orchestration::SimulationRequestDTO& req);

} // namespace solvers
} // namespace compute
} // namespace acutesim
