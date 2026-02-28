#include "analysis_registry.h"

namespace acutesim {
namespace compute {

AnalysisRegistry& AnalysisRegistry::instance() {
    static AnalysisRegistry reg;
    return reg;
}

void AnalysisRegistry::registerHandler(orchestration::AnalysisTypeDTO type,
                                        AnalysisHandler handler) {
    handlers_[static_cast<int>(type)] = std::move(handler);
}

orchestration::SimulationResponseDTO AnalysisRegistry::dispatch(
    orchestration::AnalysisTypeDTO type,
    const orchestration::SimulationRequestDTO& req,
    const TensorNetlist& netlist,
    CircuitSim& sim) const
{
    auto it = handlers_.find(static_cast<int>(type));
    if (it == handlers_.end()) {
        orchestration::SimulationResponseDTO resp;
        resp.success = false;
        resp.errorDetail = "No handler registered for analysis type " +
                            std::to_string(static_cast<int>(type));
        return resp;
    }
    return it->second(req, netlist, sim);
}

bool AnalysisRegistry::supports(orchestration::AnalysisTypeDTO type) const {
    return handlers_.count(static_cast<int>(type)) > 0;
}

std::vector<orchestration::AnalysisTypeDTO> AnalysisRegistry::registeredTypes() const {
    std::vector<orchestration::AnalysisTypeDTO> types;
    types.reserve(handlers_.size());
    for (const auto& kv : handlers_)
        types.push_back(static_cast<orchestration::AnalysisTypeDTO>(kv.first));
    return types;
}

} // namespace compute
} // namespace acutesim
