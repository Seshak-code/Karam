// ============================================================================
// engine_impl.cpp — Concrete Engine + Session Implementation
// ============================================================================
// Wires the public API (ISimulationEngine + SimulationSession) to the
// internal solver stack:
//
//   SessionImpl::runDC()  →  CircuitSim::solveDC()  →  SolverStep  →  DTO
//   SessionImpl::runAC()  →  CircuitSim::solveAC()  →  SolverStep  →  DTO
//   SessionImpl::runTransient()  →  CircuitSim::stepTransient() x N  →  DTO
//
// The GPU path is selected transparently via CircuitSim::ExecutionMode::GPU
// when GPUContextManager::instance().isAvailable() is true.
// ============================================================================

#include "internal/engine_pch.h"
#include "engine_impl.h"
#include "gpu_context_manager.h"

// Engine internal headers (invisible to consumers via -fvisibility=hidden)
#include "physics/circuitsim.h"
#include "infrastructure/simtrace.h"
#include "netlist/circuit.h"

// engine_api public DTO types
#include "engine_api/simulation_dto.h"
#include "engine_api/isimulation_engine.h"

#include <chrono>
#include <functional>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <cctype>
#include <map>

// Full SPICE3f5 parser (must be included OUTSIDE any namespace to avoid
// double-qualification: the parser itself already declares namespace acutesim)
#include "netlist/spice_parser.h"

namespace acutesim {

using namespace acutesim::compute::orchestration;

// ── Helper: SolverStep → SimulationResponseDTO ─────────────────────────────
static SimulationResponseDTO stepToResponse(const SolverStep& step,
                                              JobID jobId,
                                              PrecisionMode prec) {
    SimulationResponseDTO dto;
    dto.jobId      = jobId;
    dto.converged  = step.stats.converged;
    dto.iterations = step.stats.iterations;
    dto.residual   = step.stats.residual;
    dto.errorDetail = step.stats.error_detail;

    // Pack node voltages into waveform data point for single-step analyses
    WaveformDataPoint pt;
    pt.timeS = step.time;
    pt.nodeVoltages = step.nodeVoltages;
    dto.waveformPoints.push_back(std::move(pt));

    // Trust provenance
    dto.solverProvenance = step.stats.provenance();
    dto.precisionMode    = prec;
    return dto;
}

// ── Helper: select CircuitSim ExecutionMode ────────────────────────────────
static CircuitSim::ExecutionMode selectExecMode(PrecisionMode prec) {
    auto& gpu = GPUContextManager::instance();
    if (gpu.isAvailable() && prec != PrecisionMode::Native_F64) {
        return CircuitSim::ExecutionMode::GPU;
    }
    return CircuitSim::ExecutionMode::TENSOR_SOA_SIMD;
}

// ── Helper: generate JobID ────────────────────────────────────────────────
static JobID generateJobID() {
    static std::atomic<JobID> nextId{1};
    return nextId.fetch_add(1);
}

// ── Internal Netlist Deserializer ─────────────────────────────────────────
// Delegates to the full SPICE3f5 parser (netlist/spice_parser.h).
// Handles R, C, L, K, V, I, D, M, Q, J, E, G, X; rejects unsupported cards.
// Supports .model, .subckt/.ends, .param, waveform specs (PULSE, SIN), and SI suffixes.
// Returns false (filling `failure`) on a parse error. It used to be logged and the
// partial (or empty) netlist solved anyway, reporting "converged".
static bool deserialiseRequest(const SimulationRequestDTO& req, SPICEParseResult& out,
                               SimulationResponseDTO& failure) {
    out = SPICEParser().parse(req.source.payload, req.source.baseDir);
    if (out.success) return true;
    failure.errorDetail = "SPICE parse error: " + out.errorMessage;
    return false;
}

// ============================================================================
// SessionImpl
// ============================================================================

SessionImpl::SessionImpl(GPUContextManager& gpu, PrecisionMode prec)
    : gpu_(gpu)
    , precision_(prec)
    , jobId_(generateJobID())
{
    status_.store(JobStatus::Queued);
}

SessionImpl::~SessionImpl() = default;

void SessionImpl::cancel() {
    status_.store(JobStatus::Cancelled);
}

void SessionImpl::setIncrementalHints(uint64_t topologyHash, uint64_t parameterHash) {
    topologyHash_ = topologyHash;
    paramHash_    = parameterHash;
}

// ── DC Analysis ───────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runDC(const SimulationRequestDTO& req,
                                          const EngineCallbacks& cb) {
    if (status_.load() == JobStatus::Cancelled) {
        SimulationResponseDTO r;
        r.jobId     = jobId_;
        r.converged = false;
        r.errorDetail = "Session cancelled before start.";
        return r;
    }
    status_.store(JobStatus::Running);
    auto start = std::chrono::steady_clock::now();

    SimulationResponseDTO result = dispatch_dc_internal(req, cb);

    auto elapsed = std::chrono::steady_clock::now() - start;
    result.solveTimeMs = std::chrono::duration<double, std::milli>(elapsed).count();
    result.jobId = jobId_;

    result.success = result.converged;  // driver-path consumers read success
    status_.store(result.converged ? JobStatus::Completed : JobStatus::Failed);
    return result;
}

// .dc: the swept source's DC value at start + i*step (no drift), one solve per
// point on the same CircuitSim so each point is seeded by the last (continuation).
// Each point is a waveform point whose timeS holds the swept value.
static SimulationResponseDTO dcSweep(CircuitSim& sim, TensorNetlist& netlist, const DCConfigDTO& sw,
                                     const EngineCallbacks& cb, const std::function<bool()>& cancelled) {
    SimulationResponseDTO r;
    r.analysisType = AnalysisTypeDTO::DC;
    std::string name = sw.sweepVariable;
    for (auto& ch : name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    double* value = nullptr;
    for (auto& v : netlist.globalBlock.voltageSources)
        if (v.name == name) { value = &v.voltage_V; v.type.clear(); }  // DC value only
    for (auto& i : netlist.globalBlock.currentSources)
        if (i.name == name) { value = &i.current_A; i.type.clear(); }
    const double span = sw.step != 0.0 ? (sw.stop - sw.start) / sw.step : -1.0;
    if (!value || !(span >= -1e-9)) {
        r.errorDetail = !value ? ".dc source '" + sw.sweepVariable + "' not found at top level"
                               : ".dc step must be non-zero and point from start to stop";
        return r;
    }
    const long n = static_cast<long>(std::floor(span + 1e-9));
    r.converged = true;
    // A point that does not converge from the previous one is approached in
    // halves from the last converged value (continuation), down to 1/256 step.
    std::function<SolverStep(double, double, int)> reach = [&](double from, double to, int depth) {
        *value = to;
        SolverStep s = sim.solveDC(netlist);
        if (s.stats.converged || depth == 0) return s;
        const double mid = 0.5 * (from + to);
        SolverStep m = reach(from, mid, depth - 1);
        return m.stats.converged ? reach(mid, to, depth - 1) : m;
    };
    for (long k = 0; k <= n && !cancelled(); ++k) {
        const double target = sw.start + k * sw.step;
        SolverStep s = reach(target - sw.step, target, k == 0 ? 0 : 8);
        *value = target;
        if (!s.stats.converged) {
            r.converged   = false;
            r.errorDetail = "DC sweep failed at " + name + "=" + std::to_string(*value) + ": " +
                            s.stats.error_detail;
            break;
        }
        r.waveformPoints.push_back({*value, s.nodeVoltages});
        cb.onProgress(static_cast<int>(k + 1), static_cast<int>(n + 1));
    }
    return r;
}

SimulationResponseDTO SessionImpl::dispatch_dc_internal(const SimulationRequestDTO& req,
                                                          const EngineCallbacks& cb) {
    SPICEParseResult parsed;
    SimulationResponseDTO failure;
    if (!deserialiseRequest(req, parsed, failure)) return failure;
    TensorNetlist& netlist = parsed.netlist;
    DCConfigDTO sweep = req.dc;  // an explicit request wins over the netlist's .dc
    if (sweep.sweepVariable.empty() && !parsed.dcSource.empty())
        sweep = {parsed.dcSource, parsed.dcStart, parsed.dcStop, parsed.dcStep};

    CircuitSim sim;
    sim.execMode = selectExecMode(precision_);
    
    // Wire callbacks
    sim.onConvergenceStep = [&](int it, double res) {
        cb.onConvergenceStep(it, res);
    };

    if (sweep.sweepVariable.empty()) return stepToResponse(sim.solveDC(netlist), jobId_, precision_);
    return dcSweep(sim, netlist, sweep, cb, [this] { return status_.load() == JobStatus::Cancelled; });
}


// ── AC Analysis ───────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runAC(const SimulationRequestDTO& req,
                                          const EngineCallbacks& cb) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO result = dispatch_ac_internal(req, cb);
    result.jobId = jobId_;
    result.success = result.converged;  // driver-path consumers read success
    status_.store(result.converged ? JobStatus::Completed : JobStatus::Failed);
    return result;
}

SimulationResponseDTO SessionImpl::dispatch_ac_internal(const SimulationRequestDTO& req,
                                                          const EngineCallbacks& cb) {
    SPICEParseResult parsed;
    SimulationResponseDTO r;
    if (!deserialiseRequest(req, parsed, r)) return r;
    TensorNetlist& netlist = parsed.netlist;
    ACConfigDTO ac = req.ac;  // an explicit request wins over the netlist's .ac
    if (ac.points == 0) ac = {parsed.acSweep, parsed.acPoints, parsed.acStart, parsed.acStop};
    const auto freqs = CircuitSim::acFrequencies(ac.sweep, ac.points, ac.fStart, ac.fStop);
    if (freqs.empty()) {
        r.errorDetail = "AC needs a sweep: .ac dec|oct|lin n fstart fstop (0 < fstart <= fstop).";
        return r;
    }
    CircuitSim sim;
    sim.execMode = selectExecMode(precision_);
    SolverStep dc = sim.solveDC(netlist);
    if (!dc.stats.converged) {
        r.errorDetail = "DC operating point failed before AC analysis: " + dc.stats.error_detail;
        return r;
    }
    const auto X = sim.solveAC(netlist, freqs);
    r.converged    = X.size() == freqs.size();
    r.analysisType = AnalysisTypeDTO::AC;
    for (size_t k = 0; k < X.size(); ++k) {
        WaveformDataPoint pt{freqs[k], {}, {}, {}};
        for (int i = 0; i < netlist.numGlobalNodes; ++i) {
            pt.nodeVoltages.push_back(X[k][i].real());
            pt.nodeVoltagesImag.push_back(X[k][i].imag());
        }
        r.waveformPoints.push_back(std::move(pt));
        cb.onProgress(static_cast<int>(k + 1), static_cast<int>(freqs.size()));
    }
    return r;
}

// ── Transient Analysis ────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runTransient(const SimulationRequestDTO& req,
                                                  const EngineCallbacks& cb,
                                                  ResultChunkCallback streamCallback) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO result = dispatch_tran_internal(req, cb, std::move(streamCallback));
    result.jobId = jobId_;
    result.success = result.converged;  // driver-path consumers read success
    status_.store(result.converged ? JobStatus::Completed : JobStatus::Failed);
    return result;
}

SimulationResponseDTO SessionImpl::dispatch_tran_internal(const SimulationRequestDTO& req,
                                                            const EngineCallbacks& cb,
                                                            ResultChunkCallback streamCb) {
    SPICEParseResult parsed;
    SimulationResponseDTO failure;
    if (!deserialiseRequest(req, parsed, failure)) return failure;
    TensorNetlist& netlist = parsed.netlist;
    const bool uic = parsed.tranUIC || req.transient.uic;
    const double  stopTime = req.transient.stopTimeS;
    const double  dt       = req.transient.stepSizeS;

    CircuitSim sim;
    sim.execMode = selectExecMode(precision_);

    SimulationResponseDTO result;
    uint64_t stepIdx = 0;
    const std::string err = sim.runTransient(
        netlist, dt, stopTime, uic, [&](const SolverStep& step) {
            if (status_.load() == JobStatus::Cancelled) return false;
            WaveformDataPoint pt;
            pt.timeS        = step.time;
            pt.nodeVoltages = step.nodeVoltages;
            result.waveformPoints.push_back(pt);
            if (streamCb) streamCb(pt, stepIdx);
            cb.onTimeStep(step.time, stopTime);
            ++stepIdx;
            return true;
        });
    result.converged   = err.empty();
    result.errorDetail = err;

    result.iterations = static_cast<int>(stepIdx);
    result.analysisType = AnalysisTypeDTO::TRANSIENT;
    return result;
}

// ── Lockstep transient ────────────────────────────────────────────────────
namespace {
// Owns the parsed netlist and the CircuitSim whose state it advances (the sim keeps
// references into the netlist, so neither moves after construction).
class TransientImpl final : public TransientHandle {
public:
    TransientImpl(const SimulationRequestDTO& req, PrecisionMode prec) {
        SimulationResponseDTO failure;
        if (!deserialiseRequest(req, parsed_, failure)) { err_ = failure.errorDetail; return; }
        sim_.execMode = selectExecMode(prec);
        const double step = req.transient.stepSizeS > 0 ? req.transient.stepSizeS : parsed_.tranStep;
        err_ = sim_.beginTransient(parsed_.netlist, step, parsed_.tranUIC || req.transient.uic);
    }
    bool setSource(const std::string& name, double volts, double rampS) override {
        if (!err_.empty()) return false;
        if (sim_.setSource(parsed_.netlist, name, volts, rampS)) return true;
        err_ = "unknown source '" + name + "'";
        return false;
    }
    bool advanceTo(double t) override {
        if (!err_.empty()) return false;
        err_ = sim_.advanceTo(parsed_.netlist, t);
        return err_.empty();
    }
    double time() const override { return sim_.tranTime(); }
    bool voltage(const std::string& node, double& v) const override {
        std::string key = node;
        for (auto& c : key) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (key == "0" || key == "GND") { v = 0.0; return err_.empty(); }
        const auto it = parsed_.nodeIndex.find(key);
        if (it == parsed_.nodeIndex.end() || !err_.empty()) return false;
        v = sim_.nodeVoltage(it->second);
        return true;
    }
    std::string error() const override { return err_; }
private:
    SPICEParseResult parsed_;
    CircuitSim sim_;
    std::string err_;
};
}  // namespace

std::unique_ptr<TransientHandle> SessionImpl::beginTransient(const SimulationRequestDTO& req) {
    return std::make_unique<TransientImpl>(req, precision_);
}

// ── Monte Carlo ───────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runMonteCarlo(const SimulationRequestDTO& req,
                                                   const MonteCarloConfigDTO& mc,
                                                   const EngineCallbacks& cb) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO agg;
    agg.analysisType = AnalysisTypeDTO::MONTE_CARLO;
    agg.converged    = true;
    agg.jobId        = jobId_;

    for (int i = 0; i < mc.sampleCount && status_.load() != JobStatus::Cancelled; ++i) {
        SimulationRequestDTO run_req = req;
        run_req.mcSeed = mc.seed + i;
        auto r = dispatch_dc_internal(run_req, cb);
        if (!r.converged) agg.converged = false;
        if (!r.waveformPoints.empty()) {
            agg.waveformPoints.push_back(r.waveformPoints.front());
        }
        cb.onProgress(i + 1, mc.sampleCount);
    }

    agg.success = agg.converged;  // driver-path consumers read success
    status_.store(agg.converged ? JobStatus::Completed : JobStatus::Failed);
    return agg;
}

// ── Noise ─────────────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runNoise(const SimulationRequestDTO& req,
                                             const EngineCallbacks& /*cb*/) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO r;
    r.jobId        = jobId_;
    r.analysisType = AnalysisTypeDTO::NOISE;
    SPICEParseResult parsed;
    if (deserialiseRequest(req, parsed, r)) {
        TensorNetlist& netlist = parsed.netlist;
        const auto freqs = CircuitSim::acFrequencies(parsed.noiseSweep, parsed.noisePoints,
                                                     parsed.noiseStart, parsed.noiseStop);
        CircuitSim sim;
        sim.execMode = selectExecMode(precision_);
        if (parsed.noiseSource.empty() || freqs.empty())
            r.errorDetail = "Noise needs .noise v(out[,ref]) src dec|oct|lin n fstart fstop.";
        else if (SolverStep dc = sim.solveDC(netlist); !dc.stats.converged)
            r.errorDetail = "DC operating point failed before noise analysis: " + dc.stats.error_detail;
        else {
            auto nz = sim.solveNoise(netlist, freqs, parsed.noiseOut, parsed.noiseRef, parsed.noiseSource);
            r.errorDetail = nz.error;
            r.converged   = nz.error.empty();
            r.extended.hasNoise = r.converged;
            r.extended.noise = {freqs, std::move(nz.onoise), std::move(nz.inoise), nz.onoiseTotal, nz.inoiseTotal};
        }
    }
    r.success = r.converged;  // driver-path consumers read success
    status_.store(r.converged ? JobStatus::Completed : JobStatus::Failed);
    return r;
}

// ============================================================================
// EngineImpl
// ============================================================================

EngineImpl::EngineImpl()
    : gpu_(GPUContextManager::instance())
{
    detectCapabilities();
}

EngineImpl::~EngineImpl() = default;

void EngineImpl::detectCapabilities() {
    capabilities_.gpuAvailable         = gpu_.isAvailable();
    capabilities_.emulatedF64Supported = true;
    capabilities_.batchedSolveSupported = true;
    capabilities_.maxParallelSessions  = 8;
    capabilities_.maxBatchSize         = 64;

    if (capabilities_.gpuAvailable) {
        capabilities_.gpuAdapterName = gpu_.adapterInfo().name;
    }

#if defined(__AVX512F__)
    capabilities_.avx512Supported = true;
#endif
#if defined(__ARM_NEON)
    capabilities_.neonSupported = true;
#endif

    capabilities_.engineVersionString = "AcuteSim Engine 1.0-alpha";
}

std::unique_ptr<SimulationSession> EngineImpl::createSession() {
    return std::make_unique<SessionImpl>(gpu_, precision_);
}

EngineCapabilities EngineImpl::capabilities() const {
    return capabilities_;
}

void EngineImpl::setPrecisionMode(PrecisionMode mode) {
    precision_ = mode;
}

PrecisionMode EngineImpl::precisionMode() const {
    return precision_;
}

// ── Factory (defined here, declared ENGINE_API in isimulation_engine.h) ────
std::unique_ptr<ISimulationEngine> ISimulationEngine::create() {
    return std::make_unique<EngineImpl>();
}

} // namespace acutesim
