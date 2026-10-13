#pragma once
// ============================================================================
// compute/analysis_registry.h — Self-registration map for analysis handlers
// ============================================================================
// Maps AnalysisTypeDTO values to solver functions, registered explicitly by
// registerAllAnalyses() and LocalWorkerDriver (static-init registration was
// dropped: the linker strips it from static libraries).
//
// ARCHITECTURAL RULE: Belongs in compute/ — NOT in engine_api/ — because
// it references implementation types (TensorNetlist, CircuitSim).
// ============================================================================

// Must be included before any namespace declarations to avoid namespace
// pollution from STL headers (do NOT move these inside a namespace block).
#include "netlist/circuit.h"     // TensorNetlist (class definition)
#include "physics/circuitsim.h"     // CircuitSim   (class definition)

#include "engine_api/simulation_dto.h"
#include <functional>
#include <unordered_map>
#include <vector>

namespace acutesim {
namespace compute {

using AnalysisHandler = std::function<
    orchestration::SimulationResponseDTO(
        const orchestration::SimulationRequestDTO&,
        const TensorNetlist&,
        CircuitSim&)>;

class AnalysisRegistry {
public:
    static AnalysisRegistry& instance();

    void registerHandler(orchestration::AnalysisTypeDTO type, AnalysisHandler handler);

    orchestration::SimulationResponseDTO dispatch(
        orchestration::AnalysisTypeDTO type,
        const orchestration::SimulationRequestDTO& req,
        const TensorNetlist& netlist,
        CircuitSim& sim) const;

    bool supports(orchestration::AnalysisTypeDTO type) const;

    std::vector<orchestration::AnalysisTypeDTO> registeredTypes() const;

private:
    AnalysisRegistry() = default;
    std::unordered_map<int, AnalysisHandler> handlers_;
};

} // namespace compute
} // namespace acutesim
