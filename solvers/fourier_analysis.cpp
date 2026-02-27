// ============================================================================
// compute/solvers/fourier_analysis.cpp — Fourier / THD Analysis
// ============================================================================
// Post-processes a transient waveform to extract harmonic content via DFT.
// Computes Total Harmonic Distortion (THD%) as defined by IEC 61000-3-2.
//
// Self-registers into AnalysisRegistry at static-init time.
// Phase 4.5
// ============================================================================

#include "acutesim_engine/solvers/fourier_analysis.h"
#include "compute/analysis_registry.h"
#include "engine_api/simulation_dto.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <complex>

namespace acutesim {
namespace compute {
namespace solvers {

using namespace orchestration;
using Complex = std::complex<double>;

// DFT coefficient at a given frequency using Goertzel algorithm.
// Efficient for computing a small subset of frequencies.
static Complex goertzel(const std::vector<double>& x,
                         double freq_hz,
                         double sample_rate_hz)
{
    int N = static_cast<int>(x.size());
    if (N == 0) return Complex(0.0, 0.0);

    double k     = freq_hz / sample_rate_hz * static_cast<double>(N);
    double omega = 2.0 * M_PI * k / static_cast<double>(N);
    double coeff = 2.0 * std::cos(omega);

    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (int n = 0; n < N; ++n) {
        s0 = x[n] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }

    // Final output
    Complex result(s1 - s2 * std::cos(omega),
                   s2 * std::sin(omega));
    return result / static_cast<double>(N);
}

SimulationResponseDTO runFourierAnalysis(
    const TensorNetlist& /*netlist*/,
    const std::vector<double>& timeVector,
    const std::vector<double>& waveform,
    const SimulationRequestDTO& req)
{
    SimulationResponseDTO resp;
    resp.topologyHash = req.topologyHash;

    const int N = static_cast<int>(waveform.size());
    if (N < 4 || timeVector.size() < 2) {
        resp.success      = false;
        resp.errorMessage = "Fourier analysis: insufficient transient data (need >= 4 samples)";
        return resp;
    }

    // Sample rate from time vector
    double dt          = timeVector[1] - timeVector[0];
    if (dt <= 0.0) dt  = 1e-6;
    double sampleRateHz = 1.0 / dt;

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

    double fundamentalHz = getParam("fundamental", 1e3);
    int    harmonicCount = getParamInt("harmonics", 9);
    harmonicCount = std::max(1, std::min(harmonicCount, 20));

    FourierResultDTO fourier;
    fourier.fundamentalHz = fundamentalHz;

    const double toDB  = 20.0 / std::log(10.0);
    const double toDeg = 180.0 / M_PI;

    // DC component removal (window centering)
    double mean = std::accumulate(waveform.begin(), waveform.end(), 0.0) / N;
    std::vector<double> sig(N);
    for (int i = 0; i < N; ++i) sig[i] = waveform[i] - mean;

    // Apply Hann window to reduce spectral leakage
    for (int i = 0; i < N; ++i) {
        double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (N - 1)));
        sig[i] *= w;
    }

    // Compute harmonic magnitudes via Goertzel
    double mag_fund = 0.0;
    double harm_power_sum = 0.0;

    for (int h = 1; h <= harmonicCount; ++h) {
        double freq = h * fundamentalHz;
        Complex c   = goertzel(sig, freq, sampleRateHz);
        double mag  = std::abs(c) * 2.0; // one-sided spectrum

        fourier.harmonicFreqHz.push_back(freq);
        fourier.harmonicMagDb.push_back(
            toDB * std::log(std::max(mag, 1e-300)));
        fourier.harmonicPhaseDeg.push_back(std::arg(c) * toDeg);

        if (h == 1) {
            mag_fund = mag;
        } else {
            harm_power_sum += mag * mag;
        }
    }

    // THD% = sqrt(sum of harmonic^2) / fundamental * 100
    fourier.thdPercent = (mag_fund > 1e-30)
        ? (std::sqrt(harm_power_sum) / mag_fund * 100.0)
        : 0.0;

    resp.success              = true;
    resp.extended.hasFourier  = true;
    resp.extended.fourier     = std::move(fourier);

    return resp;
}

} // namespace solvers

namespace {

orchestration::SimulationResponseDTO fourierStub(
    const orchestration::SimulationRequestDTO&,
    const TensorNetlist&,
    CircuitSim&)
{
    orchestration::SimulationResponseDTO r;
    r.success = false;
    r.errorMessage = "FOURIER: use ISimulationDriver::submitJob with AnalysisTypeDTO::FOURIER";
    return r;
}

static RegisterAnalysis<orchestration::AnalysisTypeDTO::FOURIER, &fourierStub> s_reg;

} // anonymous namespace
} // namespace compute
} // namespace acutesim
