// ============================================================================
// compute/solvers/register_analyses.cpp — Explicit registration entrypoint
// ============================================================================
// Call registerAllAnalyses() to populate AnalysisRegistry with fully
// functional TF / PZ / FOURIER handlers that go through the solver.
//
// Each handler:
//   1. Applies corner parameters to the netlist.
//   2. Runs a DC operating point for linearisation.
//   3. Delegates to the dedicated solver function.
//
// Phase 4.5 / Reconcile — stubs replaced with real implementations.
// ============================================================================

#include "acutesim_engine/solvers/register_analyses.h"
#include "acutesim_engine/solvers/analysis_registry.h"
#include "acutesim_engine/solvers/tf_analysis.h"
#include "acutesim_engine/solvers/pz_analysis.h"
#include "acutesim_engine/solvers/fourier_analysis.h"

// Full physics headers for handler implementations
#include "acutesim_engine/physics/circuitsim.h"
#include "acutesim_engine/physics/corner_config.h"
#include "acutesim_engine/netlist/circuit.h"

namespace acutesim {
namespace compute {
namespace solvers {

using namespace orchestration;

// ── Helper: apply corner and run DC ─────────────────────────────────────────
static bool applyCornerAndDC(
    CircuitSim& sim,
    TensorNetlist& runNetlist,
    const SimulationRequestDTO& req,
    SolverStep& dcOut,
    SimulationResponseDTO& errOut)
{
    acutesim::physics::CornerConfig corner;
    corner.name           = req.corner.processCorner;
    corner.temperatureC   = req.corner.temperature;
    corner.voltageScale   = req.corner.supplyVoltage / 1.8;
    sim.applyCorner(runNetlist, corner);

    dcOut = sim.solveDC(runNetlist);
    if (!dcOut.stats.converged) {
        errOut.success      = false;
        errOut.errorDetail = "DC operating point failed: " + dcOut.stats.error_detail;
        return false;
    }
    return true;
}

// ── Transfer Function ────────────────────────────────────────────────────────
static SimulationResponseDTO tfHandler(
    const SimulationRequestDTO& req,
    const TensorNetlist& netlist,
    CircuitSim& sim)
{
    TensorNetlist runNetlist = netlist;  // deep copy — handler owns this
    SimulationResponseDTO errOut;
    SolverStep dc;
    if (!applyCornerAndDC(sim, runNetlist, req, dc, errOut)) return errOut;
    errOut.topologyHash = req.topologyHash;
    return runTFAnalysis(runNetlist, dc.nodeVoltages, req);
}

// ── Pole-Zero ────────────────────────────────────────────────────────────────
static SimulationResponseDTO pzHandler(
    const SimulationRequestDTO& req,
    const TensorNetlist& netlist,
    CircuitSim& sim)
{
    TensorNetlist runNetlist = netlist;
    SimulationResponseDTO errOut;
    SolverStep dc;
    if (!applyCornerAndDC(sim, runNetlist, req, dc, errOut)) return errOut;
    errOut.topologyHash = req.topologyHash;
    return runPZAnalysis(runNetlist, dc.nodeVoltages, req);
}

// ── Fourier ──────────────────────────────────────────────────────────────────
static SimulationResponseDTO fourierHandler(
    const SimulationRequestDTO& req,
    const TensorNetlist& netlist,
    CircuitSim& sim)
{
    // Fourier requires a transient waveform — run transient first.
    TensorNetlist runNetlist = netlist;
    acutesim::physics::CornerConfig corner;
    corner.name           = req.corner.processCorner;
    corner.temperatureC   = req.corner.temperature;
    corner.voltageScale   = req.corner.supplyVoltage / 1.8;
    sim.applyCorner(runNetlist, corner);

    // Extract stopTime from parameters (default 1ms)
    double stopTime = 1e-3;
    {
        auto it = req.parameters.find("stopTime");
        if (it != req.parameters.end()) {
            try { stopTime = std::stod(it->second); } catch (...) {}
        }
    }
    int probeNode = 1;
    {
        auto it = req.parameters.find("probeNode");
        if (it != req.parameters.end()) {
            try { probeNode = std::stoi(it->second); } catch (...) {}
        }
        probeNode = std::max(1, std::min(probeNode, runNetlist.numGlobalNodes));
    }

    std::vector<double> timeVec;
    std::vector<double> waveform;
    double dt = 1e-5;
    double t  = 0.0;
    while (t < stopTime) {
        SolverStep step = sim.stepTransient(runNetlist, dt, t);
        t = step.time;
        timeVec.push_back(t);
        double v = (probeNode - 1 < (int)step.nodeVoltages.size())
                   ? step.nodeVoltages[probeNode - 1] : 0.0;
        waveform.push_back(v);
        if (!step.stats.converged) break;
    }

    return runFourierAnalysis(runNetlist, timeVec, waveform, req);
}

// ── Public entry point ───────────────────────────────────────────────────────
void registerAllAnalyses() {
    auto& reg = AnalysisRegistry::instance();
    // Idempotent — overwrites any previous stub; safe to call multiple times.
    reg.registerHandler(AnalysisTypeDTO::TRANSFER_FUNCTION, tfHandler);
    reg.registerHandler(AnalysisTypeDTO::POLE_ZERO,         pzHandler);
    reg.registerHandler(AnalysisTypeDTO::FOURIER,           fourierHandler);
}

} // namespace solvers
} // namespace compute
} // namespace acutesim
