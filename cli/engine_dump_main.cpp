// acutesim_engine_dump <netlist> <cpu|gpu> <dc|tran|ac> [step stop]
// Machine-readable waveform dump for differential testing (tests/ngspice_diff).
// Prints: "CONVERGED <0|1>", "ERROR <text>", then one line per waveform point:
//   <time> <v1> <v2> ...   (engine-internal node order)
#include "engine_api/isimulation_engine.h"
#include "engine_api/simulation_session.h"
#include "engine_api/simulation_dto.h"
#include "engine_api/engine_callbacks.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace acutesim;
using namespace acutesim::compute::orchestration;

int main(int argc, char** argv) {
    if (argc < 4) return 2;
    std::ifstream f(argv[1]);
    std::stringstream ss; ss << f.rdbuf();
    auto engine = ISimulationEngine::create();
    engine->setPrecisionMode(std::string(argv[2]) == "cpu" ? PrecisionMode::Native_F64
                                                         : PrecisionMode::Mixed);
    auto session = engine->createSession();
    SimulationRequestDTO req;
    req.source.payload = ss.str();
    EngineCallbacks cb;
    std::string mode = argv[3];
    SimulationResponseDTO r;
    if (mode == "tran") {
        req.transient.stepSizeS = std::atof(argv[4]);
        req.transient.stopTimeS = std::atof(argv[5]);
        r = session->runTransient(req, cb, nullptr);
    } else if (mode == "ac") {
        r = session->runAC(req, cb);
    } else {
        r = session->runDC(req, cb);
    }
    std::printf("CONVERGED %d\nERROR %s\n", r.converged ? 1 : 0, r.errorDetail.c_str());
    for (const auto& p : r.waveformPoints) {
        std::printf("%.9g", p.timeS);
        for (double v : p.nodeVoltages) std::printf(" %.9g", v);
        std::printf("\n");
    }
    return 0;
}
