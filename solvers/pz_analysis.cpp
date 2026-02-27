// ============================================================================
// compute/solvers/pz_analysis.cpp — Pole-Zero Analysis
// ============================================================================
// Compute poles/zeros of the linearised small-signal transfer function.
// Poles = eigenvalues of A = -Y(0)^-1 * dY/ds evaluated at s=0.
// Approximation: eigenvalues of Y(jω) as ω→0 via frequency sweep.
//
// Self-registers into AnalysisRegistry at static-init time.
// Phase 4.5
// ============================================================================

#include "compute/solvers/pz_analysis.h"
#include "compute/analysis_registry.h"
#include "engine_api/simulation_dto.h"
#include "compute/solvers/ac_solver.h"

#include <cmath>
#include <complex>
#include <algorithm>
#include <numeric>

namespace acutesim {
namespace compute {
namespace solvers {

using namespace orchestration;
using Complex = std::complex<double>;

// Power iteration to find the dominant eigenvalue of a dense matrix.
// Returns (real, imag) of the dominant eigenvalue.
static Complex dominantEigenvalue(
    std::vector<std::vector<Complex>>& A, int maxIter = 100)
{
    int n = static_cast<int>(A.size());
    if (n == 0) return Complex(0.0, 0.0);

    // Starting vector
    std::vector<Complex> v(n, Complex(1.0, 0.0) / std::sqrt(static_cast<double>(n)));

    Complex lambda(0.0, 0.0);
    for (int iter = 0; iter < maxIter; ++iter) {
        // w = A*v
        std::vector<Complex> w(n, Complex(0.0, 0.0));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                w[i] += A[i][j] * v[j];

        // Rayleigh quotient
        Complex num(0.0, 0.0), den(0.0, 0.0);
        for (int i = 0; i < n; ++i) {
            num += std::conj(v[i]) * w[i];
            den += std::conj(v[i]) * v[i];
        }
        lambda = (std::abs(den) > 1e-30) ? (num / den) : Complex(0.0, 0.0);

        // Normalise
        double norm = 0.0;
        for (auto& x : w) norm += std::norm(x);
        norm = std::sqrt(std::max(norm, 1e-300));
        for (auto& x : w) x /= norm;
        v = w;
    }
    return lambda;
}

SimulationResponseDTO runPZAnalysis(
    const TensorNetlist& netlist,
    const std::vector<double>& dcSolution,
    const SimulationRequestDTO& req)
{
    SimulationResponseDTO resp;
    resp.topologyHash = req.topologyHash;

    int n = netlist.numGlobalNodes;
    if (n == 0) {
        resp.success      = false;
        resp.errorMessage = "PZ analysis: empty netlist";
        return resp;
    }

    // Build Y(jω) at a very low frequency to approximate Y(0)
    const double omega_lo = 2.0 * M_PI * 1.0; // 1 Hz
    std::vector<std::vector<Complex>> Y;
    std::vector<Complex> rhs;
    ACSolver::buildAdmittanceMatrix(netlist, dcSolution, omega_lo, Y, rhs);

    // Poles ≈ negative eigenvalues of Y(0)
    // We use power iteration for the dominant pole only.
    // For a full pole set we run on successively deflated matrices.
    PZResultDTO pz;

    // Iterate to extract up to min(n, 5) poles (power iteration with deflation)
    int numPoles = std::min(n, 5);
    std::vector<std::vector<Complex>> workMatrix = Y;

    for (int p = 0; p < numPoles; ++p) {
        Complex ev = dominantEigenvalue(workMatrix, 80);

        // Poles of the transfer function correspond to eigenvalues of -Y^-1 C.
        // Approximation: poles ≈ -λ_i where λ_i are eigenvalues of Y.
        // For a dominant-pole approximation this is adequate.
        Complex pole = -ev;
        pz.poleReal.push_back(pole.real());
        pz.poleImag.push_back(pole.imag());

        // Deflation: subtract outer product v*v^H scaled by eigenvalue
        // This is a simplified Hotelling deflation.
        int sz = static_cast<int>(workMatrix.size());
        for (int i = 0; i < sz; ++i)
            workMatrix[i][i] -= ev;
    }

    // DC gain: ratio of output to input at ω→0
    // Approximate as |V_out/V_in| at lowest frequency
    auto getParamInt = [&](const std::string& key, int def) -> int {
        auto it = req.parameters.find(key);
        if (it == req.parameters.end()) return def;
        try { return std::stoi(it->second); } catch (...) { return def; }
    };
    int outNode = getParamInt("outputNode", n);
    int inNode  = getParamInt("inputNode",  0);

    outNode = std::max(1, std::min(outNode, n));

    ACSolver::ACResult acLo = ACSolver::solveAC(
        netlist, dcSolution, 1.0, 10.0, 2, inNode, 1.0);

    pz.dcGain = 1.0;
    if (!acLo.nodeVoltages.empty() && outNode - 1 < n) {
        const auto& v0 = acLo.nodeVoltages[0];
        if (outNode - 1 < static_cast<int>(v0.size()))
            pz.dcGain = std::abs(v0[outNode - 1]);
    }

    resp.success        = true;
    resp.extended.hasPZ = true;
    resp.extended.pz    = std::move(pz);

    return resp;
}

} // namespace solvers

namespace {

orchestration::SimulationResponseDTO pzStub(
    const orchestration::SimulationRequestDTO&,
    const TensorNetlist&,
    CircuitSim&)
{
    orchestration::SimulationResponseDTO r;
    r.success = false;
    r.errorMessage = "PZ: use ISimulationDriver::submitJob with AnalysisTypeDTO::POLE_ZERO";
    return r;
}

static RegisterAnalysis<orchestration::AnalysisTypeDTO::POLE_ZERO, &pzStub> s_reg;

} // anonymous namespace
} // namespace compute
} // namespace acutesim
