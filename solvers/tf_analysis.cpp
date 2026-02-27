// ============================================================================
// compute/solvers/tf_analysis.cpp — Transfer Function Analysis
// ============================================================================
// H(jω) = V_out/V_in — computed via the AC admittance matrix.
// Extracts gain/phase margins from the Bode plot.
//
// Exposes: acutesim::compute::solvers::runTFAnalysis(netlist, dcSolution, req)
// Self-registers into AnalysisRegistry at static-init time.
//
// Phase 4.5
// ============================================================================

#include "compute/solvers/tf_analysis.h"
#include "compute/analysis_registry.h"
#include "engine_api/simulation_dto.h"
#include "compute/solvers/ac_solver.h"

#include <cmath>
#include <algorithm>

namespace acutesim {
namespace compute {
namespace solvers {

using namespace orchestration;

// Bode plot → gain/phase margins
static void computeMargins(
    const std::vector<double>& freqs,
    const std::vector<double>& gainDb,
    const std::vector<double>& phaseDeg,
    TFResultDTO& out)
{
    const int n = static_cast<int>(freqs.size());

    // Gain crossover: 0 dB crossing
    for (int i = 1; i < n; ++i) {
        if ((gainDb[i - 1] >= 0.0) != (gainDb[i] >= 0.0)) {
            double t  = gainDb[i - 1] / (gainDb[i - 1] - gainDb[i]);
            out.gainCrossoverHz  = freqs[i - 1] + t * (freqs[i] - freqs[i - 1]);
            double ph = phaseDeg[i - 1] + t * (phaseDeg[i] - phaseDeg[i - 1]);
            out.phaseMarginDeg   = 180.0 + ph;
            break;
        }
    }

    // Phase crossover: ±180° crossing
    for (int i = 1; i < n; ++i) {
        double p0 = phaseDeg[i - 1];
        double p1 = phaseDeg[i];
        // Detect wrap through ±180
        if ((p0 > -181.0 && p1 <= -180.0) ||
            (p0 < -179.0 && p1 >= -180.0)) {
            double t  = (-180.0 - p0) / (p1 - p0);
            t = std::max(0.0, std::min(1.0, t));
            out.phaseCrossoverHz = freqs[i - 1] + t * (freqs[i] - freqs[i - 1]);
            double g  = gainDb[i - 1] + t * (gainDb[i] - gainDb[i - 1]);
            out.gainMarginDb     = -g;
            break;
        }
    }
}

SimulationResponseDTO runTFAnalysis(
    const TensorNetlist& netlist,
    const std::vector<double>& dcSolution,
    const SimulationRequestDTO& req)
{
    SimulationResponseDTO resp;
    resp.topologyHash = req.topologyHash;

    if (netlist.numGlobalNodes == 0) {
        resp.success      = false;
        resp.errorMessage = "TF analysis: empty netlist";
        return resp;
    }

    // Extract parameters
    auto getParam = [&](const std::string& key, double def) -> double {
        auto it = req.parameters.find(key);
        if (it == req.parameters.end()) return def;
        try { return std::stod(it->second); } catch (...) { return def; }
    };
    auto getParamInt = [&](const std::string& key, int def) -> int {
        auto it = req.parameters.find(key);
        if (it == req.parameters.end()) return def;
        try { return std::stoi(it->second); } catch (...) { return def; }
    };

    double fStart   = getParam("fStart",   1.0);
    double fStop    = getParam("fStop",    1e9);
    int    points   = getParamInt("points", 200);
    int    outNode  = getParamInt("outputNode", netlist.numGlobalNodes);
    int    inNode   = getParamInt("inputNode",  0);

    outNode = std::max(1, std::min(outNode, netlist.numGlobalNodes));

    // AC sweep with excitation at input node
    ACSolver::ACResult ac = ACSolver::solveAC(
        netlist, dcSolution, fStart, fStop, points, inNode, 1.0);

    TFResultDTO tf;
    tf.frequencies  = ac.frequencies;
    tf.gainDb.resize(points,       0.0);
    tf.phaseDegrees.resize(points, 0.0);

    const double toDB  = 20.0 / std::log(10.0);
    const double toDeg = 180.0 / M_PI;

    for (int fi = 0; fi < points; ++fi) {
        if (fi >= static_cast<int>(ac.nodeVoltages.size())) break;
        const auto& vSlice = ac.nodeVoltages[fi];
        if (outNode - 1 < static_cast<int>(vSlice.size())) {
            ACSolver::Complex v = vSlice[outNode - 1];
            double mag = std::abs(v);
            tf.gainDb[fi]       = toDB * std::log(std::max(mag, 1e-300));
            tf.phaseDegrees[fi] = std::arg(v) * toDeg;
        }
    }

    computeMargins(tf.frequencies, tf.gainDb, tf.phaseDegrees, tf);

    // Pack into response
    resp.success             = true;
    resp.timeVector          = ac.frequencies;  // x-axis = frequency
    resp.extended.hasTF      = true;
    resp.extended.tf         = std::move(tf);

    // Also populate nodeVoltages for backward-compat renderers
    for (int node = 0; node < netlist.numGlobalNodes; ++node) {
        std::vector<double> mag;
        mag.reserve(points);
        for (int fi = 0; fi < points; ++fi) {
            if (fi < static_cast<int>(ac.nodeVoltages.size()) &&
                node < static_cast<int>(ac.nodeVoltages[fi].size())) {
                mag.push_back(std::abs(ac.nodeVoltages[fi][node]));
            }
        }
        resp.nodeVoltages[node] = std::move(mag);
    }

    return resp;
}

} // namespace solvers

namespace {

// ── Registry stub ────────────────────────────────────────────────────────────
// The registry expects a handler with signature:
//   SimulationResponseDTO(const SimulationRequestDTO&, CircuitSim&)
// TF analysis requires a TensorNetlist that the driver constructs from the
// request source. We register a stub that reports "use driver path".
// Real TF execution goes through LocalWorkerDriver::Impl::solverLambda.

orchestration::SimulationResponseDTO tfStub(
    const orchestration::SimulationRequestDTO&,
    const TensorNetlist&,
    CircuitSim&)
{
    orchestration::SimulationResponseDTO r;
    r.success = false;
    r.errorMessage = "TF: use ISimulationDriver::submitJob with AnalysisTypeDTO::TRANSFER_FUNCTION";
    return r;
}

static RegisterAnalysis<orchestration::AnalysisTypeDTO::TRANSFER_FUNCTION, &tfStub> s_reg;

} // anonymous namespace
} // namespace compute
} // namespace acutesim
