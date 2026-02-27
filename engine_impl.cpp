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
//
// ENGINEERING RULES:
//   - virtual dispatch happens ONCE (run* call) then is compile-time from here down
//   - No memory allocation inside the NR loop (pre-allocated in CircuitSim)
//   - ACUTESIM_PRECISE_FP_BEGIN/END wraps pnjlim callers
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
    dto.precisionMode    = (prec == PrecisionMode::Emulated_F64 ||
                             prec == PrecisionMode::Mixed) ? "EmulatedF64" : "Native_F64";
    return dto;
}

// ── Helper: select CircuitSim ExecutionMode ────────────────────────────────
static CircuitSim::ExecutionMode selectExecMode(PrecisionMode prec) {
    auto& gpu = GPUContextManager::instance();
    if (gpu.isAvailable() && prec != PrecisionMode::Native_F64) {
        return CircuitSim::ExecutionMode::GPU;
    }
    // Fall through to SIMD on CPU
    return CircuitSim::ExecutionMode::TENSOR_SOA_SIMD;
}

// ============================================================================
// SessionImpl
// ============================================================================

SessionImpl::SessionImpl(GPUContextManager& gpu, PrecisionMode prec)
    : gpu_(gpu)
    , precision_(prec)
    , jobId_(JobID::generate())
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

    status_.store(result.converged ? JobStatus::Succeeded : JobStatus::Failed);
    return result;
}

// ── Non-virtual hot path: DC ───────────────────────────────────────────────
SimulationResponseDTO SessionImpl::dispatch_dc_internal(const SimulationRequestDTO& req,
                                                          const EngineCallbacks& cb) {
    // Deserialise request → TensorNetlist
    // NOTE: req.serialisedNetlist is the DTO representation; CircuitSim operates
    // on TensorNetlist. The NetlistContract converts between the two.
    TensorNetlist netlist = req.deserialiseNetlist();
    if (netlist.numGlobalNodes == 0) {
        SimulationResponseDTO r;
        r.converged = false;
        r.errorDetail = "Empty or invalid netlist (0 nodes).";
        return r;
    }

    CircuitSim sim;
    sim.execMode = selectExecMode(precision_);

    // Incremental reuse: if topology hasn't changed, reuse cached CSR pattern
    if (topologyHash_ != 0) {
        netlist.globalBlock.topologyHash = topologyHash_;
    }

    SolverStep step = sim.solveDC(netlist);

    if (cb.onConvergenceStep) {
        cb.onConvergenceStep(step.stats.iterations, step.stats.residual);
    }
    return stepToResponse(step, jobId_, precision_);
}

// ── AC Analysis ───────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runAC(const SimulationRequestDTO& req,
                                          const EngineCallbacks& cb) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO result = dispatch_ac_internal(req, cb);
    result.jobId = jobId_;
    status_.store(result.converged ? JobStatus::Succeeded : JobStatus::Failed);
    return result;
}

SimulationResponseDTO SessionImpl::dispatch_ac_internal(const SimulationRequestDTO& req,
                                                          const EngineCallbacks& cb) {
    TensorNetlist netlist = req.deserialiseNetlist();
    if (netlist.numGlobalNodes == 0) {
        SimulationResponseDTO r;
        r.converged   = false;
        r.errorDetail = "Empty or invalid netlist.";
        return r;
    }

    // AC solver path: DC operating point followed by small-signal linearisation
    CircuitSim sim;
    sim.execMode = selectExecMode(precision_);

    SolverStep dc = sim.solveDC(netlist);
    if (!dc.stats.converged) {
        SimulationResponseDTO r;
        r.converged   = false;
        r.errorDetail = "DC operating point failed before AC analysis.";
        r.solverProvenance = dc.stats.provenance();
        return r;
    }

    // Full AC analysis is implemented in acutesim_engine/solvers/ac_solver.h
    // Delegating to CircuitSim::solveAC once it migrates its signature to TensorNetlist.
    // For now, return the DC point with an AC flag.
    SimulationResponseDTO result = stepToResponse(dc, jobId_, precision_);
    result.analysisType = "AC";
    return result;
}

// ── Transient Analysis ────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runTransient(const SimulationRequestDTO& req,
                                                  const EngineCallbacks& cb,
                                                  ResultChunkCallback streamCallback) {
    if (status_.load() == JobStatus::Cancelled) {
        SimulationResponseDTO r;
        r.jobId     = jobId_;
        r.converged = false;
        r.errorDetail = "Cancelled.";
        return r;
    }
    status_.store(JobStatus::Running);

    SimulationResponseDTO result = dispatch_tran_internal(req, cb, std::move(streamCallback));
    result.jobId = jobId_;
    status_.store(result.converged ? JobStatus::Succeeded : JobStatus::Failed);
    return result;
}

SimulationResponseDTO SessionImpl::dispatch_tran_internal(const SimulationRequestDTO& req,
                                                            const EngineCallbacks& cb,
                                                            ResultChunkCallback streamCb) {
    TensorNetlist netlist = req.deserialiseNetlist();
    const double  stopTime = req.transient.stopTimeS;
    double        dt       = req.transient.initialTimeStepS > 0
                             ? req.transient.initialTimeStepS : 1e-9;

    if (netlist.numGlobalNodes == 0 || stopTime <= 0) {
        SimulationResponseDTO r;
        r.converged   = false;
        r.errorDetail = "Invalid transient parameters.";
        return r;
    }

    CircuitSim sim;
    sim.execMode = selectExecMode(precision_);
    sim.arbiterInitialized = false; // Reset for fresh transient

    // DC bias first
    SolverStep dcStep = sim.solveDC(netlist);
    if (!dcStep.stats.converged) {
        SimulationResponseDTO r;
        r.converged   = false;
        r.errorDetail = "DC bias failed: " + dcStep.stats.error_detail;
        return r;
    }

    SimulationResponseDTO result;
    result.converged = true;

    double currentTime = 0.0;
    uint64_t stepIdx   = 0;

    // Apply initial conditions from DC
    netlist.initFromVoltages(dcStep.nodeVoltages);

    while (currentTime < stopTime) {
        if (status_.load() == JobStatus::Cancelled) {
            result.converged  = false;
            result.errorDetail = "Cancelled during transient.";
            break;
        }

        SolverStep step = sim.stepTransient(netlist, dt, currentTime);

        if (!step.nodeVoltages.empty()) {
            currentTime = step.time;

            WaveformDataPoint pt;
            pt.timeS        = currentTime;
            pt.nodeVoltages = step.nodeVoltages;
            result.waveformPoints.push_back(pt);

            if (streamCb) {
                streamCb(pt, stepIdx);
            }

            if (cb.onTimeStep) {
                cb.onTimeStep(currentTime, stopTime);
            }
        } else {
            result.converged  = false;
            result.errorDetail = "Transient step failed at t=" + std::to_string(currentTime);
            break;
        }
        ++stepIdx;
    }

    result.iterations = static_cast<int>(stepIdx);
    result.analysisType = "TRAN";
    return result;
}

// ── Monte Carlo ───────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runMonteCarlo(const SimulationRequestDTO& req,
                                                   const MonteCarloConfigDTO& mc,
                                                   const EngineCallbacks& cb) {
    // Run N DC sweeps with statistical parameter variation
    // Delegates to MonteCarloEngine in infrastructure/
    status_.store(JobStatus::Running);
    SimulationResponseDTO agg;
    agg.analysisType = "MC";
    agg.converged    = true;
    agg.jobId        = jobId_;

    for (int i = 0; i < mc.numRuns && status_.load() != JobStatus::Cancelled; ++i) {
        SimulationRequestDTO run_req = req;
        run_req.mcSeed = mc.seed + i;
        auto r = dispatch_dc_internal(run_req, cb);
        if (!r.converged) { agg.converged = false; }
        if (!r.waveformPoints.empty()) {
            agg.waveformPoints.push_back(r.waveformPoints.front());
        }
        if (cb.onProgress) cb.onProgress(i + 1, mc.numRuns);
    }

    status_.store(agg.converged ? JobStatus::Succeeded : JobStatus::Failed);
    return agg;
}

// ── Noise ─────────────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runNoise(const SimulationRequestDTO& req,
                                             const EngineCallbacks& cb) {
    // Noise analysis requires AC operating point + noise density calculation
    // Placeholder: runs DC + returns noise floor
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
    capabilities_.emulatedF64Supported = true;  // Always: software implementation
    capabilities_.batchedSolveSupported = true;
    capabilities_.maxParallelSessions  = 8;     // tune per platform
    capabilities_.maxBatchSize         = 64;

    if (capabilities_.gpuAvailable) {
        capabilities_.gpuAdapterName = gpu_.adapterInfo().name;
    }

    // Runtime SIMD detection
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
