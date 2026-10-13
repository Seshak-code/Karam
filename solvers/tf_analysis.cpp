// ============================================================================
// compute/solvers/tf_analysis.cpp — Transfer Function Analysis
// ============================================================================
// H(jω) = V_out/V_in from CircuitSim::solveAC (the circuit's own AC sources).
// Extracts gain/phase margins from the Bode plot.
//
// Phase 4.5
// ============================================================================

#include "solvers/tf_analysis.h"
#include "engine_api/simulation_dto.h"
#include "physics/circuitsim.h"
#include <complex>

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

SimulationResponseDTO runTFAnalysis(CircuitSim& sim, TensorNetlist& netlist,
                                    const SimulationRequestDTO& req)
{
    SimulationResponseDTO resp;
    resp.topologyHash = req.topologyHash;
    auto param = [&](const std::string& key, double def) {
        auto it = req.parameters.find(key);
        if (it == req.parameters.end()) return def;
        try { return std::stod(it->second); } catch (...) { return def; }
    };
    const int n       = netlist.numGlobalNodes;
    const int outNode = std::max(1, std::min(static_cast<int>(param("outputNode", n)), n));
    const int inNode  = static_cast<int>(param("inputNode", 0));  // 0 = absolute V(out)
    const auto freqs  = CircuitSim::acFrequencies("log", static_cast<int>(param("points", 200)),
                                                  param("fStart", 1.0), param("fStop", 1e9));
    const auto X = sim.solveAC(netlist, freqs);
    bool excited = false;
    for (const auto& x : X) for (const auto& v : x) excited |= v != 0.0;
    if (n == 0 || freqs.empty() || X.size() != freqs.size() || !excited) {
        resp.errorDetail = "TF needs a converged operating point, a frequency range and an AC source "
                           "(AC magnitude on a V or I source).";
        return resp;
    }

    TFResultDTO tf;
    tf.frequencies = freqs;
    for (const auto& x : X) {
        const std::complex<double> h = x[outNode - 1] / (inNode > 0 && inNode <= n ? x[inNode - 1] : 1.0);
        tf.gainDb.push_back(20.0 * std::log10(std::max(std::abs(h), 1e-300)));
        tf.phaseDegrees.push_back(std::arg(h) * 180.0 / M_PI);
    }
    computeMargins(tf.frequencies, tf.gainDb, tf.phaseDegrees, tf);

    resp.success = resp.converged = true;
    resp.timeVector = freqs;  // x-axis = frequency
    for (int node = 0; node < n; ++node)
        for (const auto& x : X) resp.nodeVoltages[node].push_back(std::abs(x[node]));
    resp.extended.hasTF = true;
    resp.extended.tf    = std::move(tf);
    return resp;
}

} // namespace solvers
} // namespace compute
} // namespace acutesim
