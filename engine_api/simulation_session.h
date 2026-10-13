#pragma once
// ============================================================================
// simulation_session.h — Isolated Simulation Execution Object
// ============================================================================
// Each session represents one independent simulation context.
//
// Design rationale:
//   - Avoids global engine state (which blocks concurrency)
//   - AI speculative runs get their own session → cancel without side effects
//   - GPU contexts are acquired from GPUContextManager pool per session
//   - Multiple sessions can run concurrently (limited by maxParallelSessions)
//
// RULE: Do NOT add virtual methods to runDC_step, stepNewton, etc.
// Virtual dispatch in the NR inner loop destroys SIMD vectorization.
// Only the top-level run* methods are virtual (called once per simulation).
// ============================================================================

#include "simulation_dto.h"
#include "engine_callbacks.h"
#include "job_status.h"

#include <memory>
#include <functional>
#include <cstdint>

namespace acutesim {

// ── Streaming result callback (for live waveform updates) ─────────────────
using ResultChunkCallback = std::function<void(const compute::orchestration::WaveformDataPoint&, uint64_t timeStepIdx)>;

// ── SimulationSession ──────────────────────────────────────────────────────
class ENGINE_API SimulationSession {
public:
    virtual ~SimulationSession() = default;

    // === Analysis methods ===
    virtual compute::orchestration::SimulationResponseDTO runDC(
        const compute::orchestration::SimulationRequestDTO&,
        const EngineCallbacks& = {}) = 0;

    virtual compute::orchestration::SimulationResponseDTO runAC(
        const compute::orchestration::SimulationRequestDTO&,
        const EngineCallbacks& = {}) = 0;

    virtual compute::orchestration::SimulationResponseDTO runTransient(
        const compute::orchestration::SimulationRequestDTO&,
        const EngineCallbacks& = {},
        ResultChunkCallback   streamCallback = nullptr) = 0;

    virtual compute::orchestration::SimulationResponseDTO runMonteCarlo(
        const compute::orchestration::SimulationRequestDTO&,
        const compute::orchestration::MonteCarloConfigDTO&,
        const EngineCallbacks& = {}) = 0;

    virtual compute::orchestration::SimulationResponseDTO runNoise(
        const compute::orchestration::SimulationRequestDTO&,
        const EngineCallbacks& = {}) = 0;

    // === Lifecycle ===
    virtual JobID       jobId()   const = 0;
    virtual JobStatus   status()  const = 0;
    virtual void        cancel()        = 0;

    // === Incremental reuse ===
    // Provide topology/param hashes from SchematicSimRequest for cache reuse.
    virtual void setIncrementalHints(uint64_t topologyHash,
                                      uint64_t parameterHash) = 0;
};

} // namespace acutesim
