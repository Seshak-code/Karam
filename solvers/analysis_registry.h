#pragma once
// ============================================================================
// compute/analysis_registry.h — Self-registration map for analysis handlers
// ============================================================================
// Maps AnalysisTypeDTO values to solver functions. Handlers register
// themselves at static-init time via the RegisterAnalysis<T,Fn> helper.
//
// ARCHITECTURAL RULE: Belongs in compute/ — NOT in engine_api/ — because
// it references implementation types (TensorNetlist, CircuitSim).
// ============================================================================

// Must be included before any namespace declarations to avoid namespace
// pollution from STL headers (do NOT move these inside a namespace block).
#include "acutesim_engine/netlist/circuit.h"     // TensorNetlist (class definition)
#include "acutesim_engine/physics/circuitsim.h"     // CircuitSim   (class definition)

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

// ── Static-init self-registration helper ──────────────────────────────────
// Usage (in a .cpp file):
//   static RegisterAnalysis<AnalysisTypeDTO::TF, &myTFHandler> s_reg;
//
// IMPORTANT: Instantiate only in .cpp files, never in headers.
// ─────────────────────────────────────────────────────────────────────────
template<orchestration::AnalysisTypeDTO T, auto Fn>
struct RegisterAnalysis {
    RegisterAnalysis() {
        AnalysisRegistry::instance().registerHandler(T, Fn);
    }
};

} // namespace compute
} // namespace acutesim
