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

#include "acutesim_engine/internal/engine_pch.h"
#include "acutesim_engine/engine_impl.h"
#include "acutesim_engine/gpu_context_manager.h"

// Engine internal headers (invisible to consumers via -fvisibility=hidden)
#include "acutesim_engine/physics/circuitsim.h"
#include "acutesim_engine/infrastructure/simtrace.h"
#include "acutesim_engine/netlist/circuit.h"

// engine_api public DTO types
#include "engine_api/simulation_dto.h"
#include "engine_api/isimulation_engine.h"

#include <chrono>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <cctype>
#include <map>

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
// Handles all standard elements: R, C, L, K, V, I, D, M, Q, J, E, G, H, F, T, X.
// Supports .model, .subckt/.ends, .param, waveform specs (PULSE, SIN), and SI suffixes.
#include "acutesim_engine/netlist/spice_parser.h"

static TensorNetlist deserialiseRequest(const SimulationRequestDTO& req) {
    if (req.source.payload.empty()) return TensorNetlist{};

    acutesim::SPICEParser parser;
    auto result = parser.parse(req.source.payload);

    if (!result.success) {
        std::cerr << "[AcuteSim] SPICE parse error: " << result.errorMessage << "\n";
    }

    return std::move(result.netlist);
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

    status_.store(result.converged ? JobStatus::Completed : JobStatus::Failed);
    return result;
}

SimulationResponseDTO SessionImpl::dispatch_dc_internal(const SimulationRequestDTO& req,
                                                          const EngineCallbacks& cb) {
    TensorNetlist netlist = deserialiseRequest(req);
    if (netlist.numGlobalNodes == 0 && req.source.payload.empty()) {
        SimulationResponseDTO r;
        r.converged = false;
        r.errorDetail = "Empty or invalid netlist.";
        return r;
    }

    CircuitSim sim;
    sim.execMode = selectExecMode(precision_);
    
    // Wire callbacks
    sim.onConvergenceStep = [&](int it, double res) {
        cb.onConvergenceStep(it, res);
    };

    SolverStep step = sim.solveDC(netlist);
    return stepToResponse(step, jobId_, precision_);
}

// ── AC Analysis ───────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runAC(const SimulationRequestDTO& req,
                                          const EngineCallbacks& cb) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO result = dispatch_ac_internal(req, cb);
    result.jobId = jobId_;
    status_.store(result.converged ? JobStatus::Completed : JobStatus::Failed);
    return result;
}

SimulationResponseDTO SessionImpl::dispatch_ac_internal(const SimulationRequestDTO& req,
                                                          const EngineCallbacks& cb) {
    TensorNetlist netlist = deserialiseRequest(req);
    CircuitSim sim;
    sim.execMode = selectExecMode(precision_);

    SolverStep dc = sim.solveDC(netlist);
    if (!dc.stats.converged) {
        SimulationResponseDTO r;
        r.converged   = false;
        r.errorDetail = "DC operating point failed before AC analysis.";
        return r;
    }

    SimulationResponseDTO result = stepToResponse(dc, jobId_, precision_);
    result.analysisType = AnalysisTypeDTO::AC;
    return result;
}

// ── Transient Analysis ────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runTransient(const SimulationRequestDTO& req,
                                                  const EngineCallbacks& cb,
                                                  ResultChunkCallback streamCallback) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO result = dispatch_tran_internal(req, cb, std::move(streamCallback));
    result.jobId = jobId_;
    status_.store(result.converged ? JobStatus::Completed : JobStatus::Failed);
    return result;
}

SimulationResponseDTO SessionImpl::dispatch_tran_internal(const SimulationRequestDTO& req,
                                                            const EngineCallbacks& cb,
                                                            ResultChunkCallback streamCb) {
    TensorNetlist netlist = deserialiseRequest(req);
    const double  stopTime = req.transient.stopTimeS;
    const double  dt       = req.transient.stepSizeS;

    CircuitSim sim;
    sim.execMode = selectExecMode(precision_);

    // DC bias
    SolverStep dcStep = sim.solveDC(netlist);
    if (!dcStep.stats.converged) {
        SimulationResponseDTO r;
        r.converged   = false;
        r.errorDetail = "DC bias failed.";
        return r;
    }

    SimulationResponseDTO result;
    result.converged = true;
    double currentTime = 0.0;
    uint64_t stepIdx   = 0;

    while (currentTime < stopTime) {
        if (status_.load() == JobStatus::Cancelled) {
            result.converged  = false;
            result.errorDetail = "Cancelled.";
            break;
        }

        SolverStep step = sim.stepTransient(netlist, dt, currentTime);
        if (!step.nodeVoltages.empty()) {
            currentTime = step.time;
            WaveformDataPoint pt;
            pt.timeS        = currentTime;
            pt.nodeVoltages = step.nodeVoltages;
            result.waveformPoints.push_back(pt);

            if (streamCb) streamCb(pt, stepIdx);
            cb.onTimeStep(currentTime, stopTime);
        } else {
            result.converged  = false;
            break;
        }
        ++stepIdx;
    }

    result.iterations = static_cast<int>(stepIdx);
    result.analysisType = AnalysisTypeDTO::TRANSIENT;
    return result;
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

    for (int i = 0; i < mc.numRuns && status_.load() != JobStatus::Cancelled; ++i) {
        SimulationRequestDTO run_req = req;
        run_req.mcSeed = mc.seed + i;
        auto r = dispatch_dc_internal(run_req, cb);
        if (!r.converged) agg.converged = false;
        if (!r.waveformPoints.empty()) {
            agg.waveformPoints.push_back(r.waveformPoints.front());
        }
        cb.onProgress(i + 1, mc.numRuns);
    }

    status_.store(agg.converged ? JobStatus::Completed : JobStatus::Failed);
    return agg;
}

// ── Noise ─────────────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runNoise(const SimulationRequestDTO& req,
                                             const EngineCallbacks& cb) {
    return runAC(req, cb);
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
