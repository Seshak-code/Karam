// test_transient_session.cpp — lockstep transient: CircuitSim::beginTransient/advanceTo/setSource
// and the public TransientHandle.
//   (a) runTransient == begin + advanceTo in chunks (PULSE-driven RC)
//   (b) setSource steps on an RC match 1-exp(-t/RC)
//   (c) a closed digital loop (relaxation oscillator) has the analytic period
// Run: ./build/test_transient_session

#include "engine_api/isimulation_engine.h"
#include "engine_api/simulation_session.h"
#include "physics/circuitsim.h"
#include "netlist/spice_parser.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace acutesim;
using namespace acutesim::compute::orchestration;

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

static std::unique_ptr<TransientHandle> open(const std::string& netlist, double step) {
    static auto engine = ISimulationEngine::create();
    static auto session = engine->createSession();
    SimulationRequestDTO req;
    req.source.payload = netlist;
    req.transient.stepSizeS = step;
    return session->beginTransient(req);
}

static const char* kRC = "rc\nV1 in 0 %s\nR1 in out 1k\nC1 out 0 1n\n.end\n";  // RC = 1 us

static std::string rcNetlist(const std::string& src) {
    char buf[256];
    std::snprintf(buf, sizeof buf, kRC, src.c_str());
    return buf;
}

int main() {
    // (a)
    {
        const std::string net = rcNetlist("PULSE(0 1 1u 1u 1u 5u 20u)");
        const double step = 0.1e-6, stop = 60e-6;
        auto vout = [](CircuitSim& s) { return s.nodeVoltage(2); };  // in = 1, out = 2 (first appearance)
        SPICEParseResult a = SPICEParser().parse(net), b = SPICEParser().parse(net);
        CircuitSim sa, sb;
        check(sa.runTransient(a.netlist, step, stop, false, [](const SolverStep&) { return true; }).empty(), "runTransient ok");
        check(sb.beginTransient(b.netlist, step, false, stop).empty(), "beginTransient ok");
        double maxDiff = 0, endDiff = 0;
        SPICEParseResult c = SPICEParser().parse(net);
        CircuitSim sc;
        sc.beginTransient(c.netlist, step, false, stop);
        // one advanceTo(stop) is the whole loop: bit-identical
        sb.advanceTo(b.netlist, stop);
        endDiff = std::abs(vout(sa) - vout(sb));
        // 100 chunks: ticks clip steps, so the step sequence differs; the answer must not
        for (int k = 1; k <= 100; ++k) {
            sc.advanceTo(c.netlist, stop * k / 100);
        }
        maxDiff = std::abs(vout(sa) - vout(sc));
        std::printf("  one chunk diff %.3g, 100 chunks diff %.3g, t=%.9g\n", endDiff, maxDiff, sc.tranTime());
        check(endDiff <= 1e-12, "advanceTo(stop) equals runTransient to 1e-12");
        check(maxDiff <= 1e-12, "100 chunks equal runTransient to 1e-12");
        check(std::abs(sc.tranTime() - stop) < 1e-15, "time() lands on stop");
    }
    // (b)
    {
        auto h = open(rcNetlist("DC 0"), 0.05e-6);
        check(h->error().empty(), "begin: " + h->error());
        const double t0 = 2e-6, RC = 1e-6;
        bool ok = h->advanceTo(t0) && h->setSource("v1", 1.0, 0.0);
        double worst = 0;
        for (int k = 1; k <= 50 && ok; ++k) {
            const double t = t0 + k * 0.1e-6;
            ok = h->advanceTo(t);
            double v = 0;
            ok = ok && h->voltage("OUT", v);
            const double want = 1.0 - std::exp(-(t - t0) / RC);
            worst = std::max(worst, std::abs(v - want) / want);
        }
        std::printf("  worst relative error %.3g\n", worst);
        check(ok && worst < 1e-3, "setSource step matches 1-exp(-t/RC) within 1e-3");
        double v;
        check(!h->setSource("nope", 1, 0), "unknown source returns false");
        check(!h->voltage("nowhere", v), "unknown node returns false");
    }
    // (c)
    {
        const double RC = 1e-6;
        auto h = open("osc\nV1 in 0 DC 0\nR1 in out 1k\nC1 out 0 1n\n.end\n", RC / 100);
        double drive = 1, v = 0;  // starts charging; the loop below is the "digital side"
        h->setSource("V1", 1.0, 0.0);
        std::vector<double> rises{0.0};
        for (int k = 1; k <= 12000 && h->error().empty(); ++k) {
            if (!h->advanceTo(k * RC / 1000) || !h->voltage("out", v)) break;
            if (drive == 1.0 && v > 0.6) { drive = 0.0; h->setSource("V1", 0.0, 0.0); }
            else if (drive == 0.0 && v < 0.4) { drive = 1.0; h->setSource("V1", 1.0, 0.0); rises.push_back(h->time()); }
        }
        check(h->error().empty(), "oscillator ran: " + h->error());
        const double want = 2 * RC * std::log(0.6 / 0.4);
        const size_t n = rises.size();
        const double period = n >= 4 ? (rises[n - 1] - rises[1]) / (n - 2) : 0;  // skips the 0 -> 0.6 start-up
        std::printf("  period %.4g s vs analytic %.4g s (%zu cycles)\n", period, want, n);
        check(n >= 4 && std::abs(period - want) < 0.02 * want, "oscillation period within 2% of 2RC ln(0.6/0.4)");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "OK");
    return g_fail != 0;
}
