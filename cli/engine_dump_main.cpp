// acutesim_engine_dump <netlist> <cpu|gpu> <dc|tran|ac|noise> [step stop]
// Machine-readable waveform dump for differential testing (tests/ngspice_diff).
// Prints: "CONVERGED <0|1>", "ERROR <text>", then one line per waveform point:
//   <time> <v1> <v2> ...   (engine-internal node order)
//   ac: <freq> <re1> <im1> <re2> <im2> ...
//   (env TREESPICE_BASEDIR overrides the directory .include/.lib paths resolve against)
//   noise: <freq> <onoise> <inoise> ... then "TOTAL <onoise_total> <inoise_total>"
// A nested .dc / .temp list adds "SWEEP <name>=<value> ..." before the first row of each outer value
//   (the row's first column is the innermost sweep value).
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
    const std::string path = argv[1];
    req.source.baseDir = path.find('/') == std::string::npos ? "" : path.substr(0, path.rfind('/'));
    // The harness runs a copy of the netlist; TREESPICE_BASEDIR points .include/.lib at the original directory.
    if (const char* d = std::getenv("TREESPICE_BASEDIR")) req.source.baseDir = d;
    EngineCallbacks cb;
    std::string mode = argv[3];
    SimulationResponseDTO r;
    if (mode == "tran") {
        req.transient.stepSizeS = std::atof(argv[4]);
        req.transient.stopTimeS = std::atof(argv[5]);
        r = session->runTransient(req, cb, nullptr);
    } else if (mode == "ac") {
        r = session->runAC(req, cb);
    } else if (mode == "noise") {
        r = session->runNoise(req, cb);
    } else {
        r = session->runDC(req, cb);
    }
    std::printf("CONVERGED %d\nERROR %s\n", r.converged ? 1 : 0, r.errorDetail.c_str());
    const auto& nz = r.extended.noise;
    for (size_t k = 0; k < nz.frequencies.size(); ++k)
        std::printf("%.9g %.9g %.9g\n", nz.frequencies[k], nz.onoise[k], nz.inoise[k]);
    if (r.extended.hasNoise) std::printf("TOTAL %.9g %.9g\n", nz.onoiseTotal, nz.inoiseTotal);
    const std::vector<double>* shown = nullptr;
    for (const auto& p : r.waveformPoints) {
        if (!p.outer.empty() && (!shown || *shown != p.outer)) {
            std::printf("SWEEP");
            for (size_t i = 0; i < p.outer.size() && i < r.outerNames.size(); ++i)
                std::printf(" %s=%.9g", r.outerNames[i].c_str(), p.outer[i]);
            std::printf("\n");
            shown = &p.outer;
        }
        std::printf("%.9g", p.timeS);
        for (size_t i = 0; i < p.nodeVoltages.size(); ++i) {
            std::printf(" %.9g", p.nodeVoltages[i]);
            if (i < p.nodeVoltagesImag.size()) std::printf(" %.9g", p.nodeVoltagesImag[i]);
        }
        std::printf("\n");
    }
    return 0;
}
