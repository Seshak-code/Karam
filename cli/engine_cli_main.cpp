// ============================================================================
// engine_cli_main.cpp — Headless CLI simulation runner
// ============================================================================
// Usage:
//   echo "V1 1 0 DC 5\nR1 1 2 1000\nR2 2 0 2000\n.dc" | ./acutesim_engine_cli
//   ./acutesim_engine_cli inverter.spice
//
// Exits 0 on convergence, 1 on failure.
// Primarily used for:
//   - Golden file regression tests
//   - Phase F ABI stability verification
//   - CI headless validation
// ============================================================================

#include "engine_api/isimulation_engine.h"
#include "engine_api/simulation_session.h"
#include "engine_api/simulation_dto.h"
#include "engine_api/engine_callbacks.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>

// ENGINE_API visibility macro (provided by engine, consumed by clients)
#ifndef ENGINE_API
  #define ENGINE_API
#endif

namespace {
using namespace acutesim;
using namespace acutesim::compute::orchestration;

// ---------------------------------------------------------------------------
// Minimal SPICE line parser → SimulationRequestDTO
// Understands: Resistor (R), Voltage source (V), .dc, .tran
// ---------------------------------------------------------------------------
static SimulationRequestDTO parseSpice(std::istream& in) {
    SimulationRequestDTO req;
    std::string line;
    std::vector<std::string> lines;

    while (std::getline(in, line)) {
        lines.push_back(line);
    }

    // Store raw netlist text in DTO for the engine's internal parser
    std::ostringstream oss;
    for (auto& l : lines) oss << l << "\n";
    req.source.payload = oss.str();
    req.source.format  = NetlistFormatDTO::VERILOG_A;

    // Detect analysis type from .control cards
    for (auto& l : lines) {
        std::string lower = l;
        for (char& c : lower) c = static_cast<char>(std::tolower(c));
        if (lower.rfind(".tran", 0) == 0) {
            req.analysis = AnalysisTypeDTO::TRANSIENT;
            // .tran <step> <stop>
            double step = 1e-9, stop = 1e-6;
            std::istringstream ss(l.substr(5));
            ss >> step >> stop;
            req.transient.stepSizeS = step;
            req.transient.stopTimeS  = stop;
        } else if (lower.rfind(".ac", 0) == 0) {
            req.analysis = AnalysisTypeDTO::AC;
        } else if (lower.rfind(".dc", 0) == 0 || lower.rfind(".op", 0) == 0) {
            req.analysis = AnalysisTypeDTO::DC;
        }
    }
    return req;
}
} // anonymous namespace

int main(int argc, char** argv) {
    // ── Create engine ──────────────────────────────────────────────────────
    using namespace acutesim;
    using namespace acutesim::compute::orchestration;

    auto engine = ISimulationEngine::create();
    auto caps   = engine->capabilities();

    std::cout << "AcuteSim Engine CLI\n"
              << "  Version : " << caps.engineVersionString << "\n"
              << "  GPU     : " << (caps.gpuAvailable ? "YES (" + caps.gpuAdapterName + ")" : "NO") << "\n"
              << "  AVX-512 : " << (caps.avx512Supported ? "YES" : "NO") << "\n"
              << "  NEON    : " << (caps.neonSupported   ? "YES" : "NO") << "\n\n";

    // ── Parse input ────────────────────────────────────────────────────────
    SimulationRequestDTO req;

    if (argc >= 2) {
        std::ifstream f(argv[1]);
        if (!f.is_open()) {
            std::cerr << "Error: Cannot open file: " << argv[1] << "\n";
            return 1;
        }
        req = parseSpice(f);
        std::cout << "Netlist: " << argv[1] << "\n";
    } else {
        std::cout << "Reading netlist from stdin (end with EOF)...\n";
        req = parseSpice(std::cin);
    }

    // Analysis type is chosen per req.analysis

    // ── Run simulation ─────────────────────────────────────────────────────
    auto session = engine->createSession();

    EngineCallbacks cb;
    cb.onConvergenceStep = [](int iter, double res) {
        std::cout << "  NR iter " << std::setw(3) << iter
                  << " | residual = " << std::scientific << std::setprecision(3) << res << "\n";
    };
    cb.onTimeStep = [](double t, double stop) {
        std::cout << "\r  t = " << std::scientific << std::setprecision(3) << t
                  << " / " << stop << std::flush;
    };

    auto t0 = std::chrono::steady_clock::now();
    SimulationResponseDTO result;

    if (req.analysis == AnalysisTypeDTO::TRANSIENT) {
        result = session->runTransient(req, cb, nullptr);
    } else if (req.analysis == AnalysisTypeDTO::AC) {
        result = session->runAC(req, cb);
    } else {
        result = session->runDC(req, cb);
    }
    auto elapsed = std::chrono::steady_clock::now() - t0;

    std::cout << "\n\n── Simulation Result ──────────────────────────────\n";
    std::cout << "  Converged : " << (result.converged ? "YES" : "NO") << "\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  Residual  : " << std::scientific << result.residual << "\n";
    std::cout << "  Wall time : " << std::fixed << std::setprecision(2)
              << std::chrono::duration<double, std::milli>(elapsed).count() << " ms\n";

    if (!result.waveformPoints.empty()) {
        const auto& pt = result.waveformPoints.back();
        std::cout << "\n── Final Node Voltages (t=" << pt.timeS << " s) ──\n";
        for (size_t i = 0; i < pt.nodeVoltages.size(); ++i) {
            std::cout << "  V(" << (i+1) << ") = "
                      << std::fixed << std::setprecision(6) << pt.nodeVoltages[i] << " V\n";
        }
    }

    if (!result.errorDetail.empty()) {
        std::cerr << "\n[WARN] " << result.errorDetail << "\n";
    }

    std::cout << "\nProvenance: " << result.solverProvenance << "\n";

    return result.converged ? 0 : 1;
}
