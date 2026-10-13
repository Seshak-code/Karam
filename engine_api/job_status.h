#pragma once
// ============================================================================
// job_status.h — Explicit Monotonic Job State Machine
// ============================================================================
// All transitions must be forward-only. No backward moves.
//
// Valid transitions:
//   Queued → Compiling → Running → Streaming → Completed
//                      ↘ Failed  ↗ Cancelled
// ============================================================================

#include <cstdint>

namespace acutesim {

using JobID    = uint64_t;
using ClientID = uint64_t;

enum class JobStatus : uint8_t {
    Queued     = 0,  // Submitted, awaiting executor
    Compiling  = 1,  // TensorNetlist being compiled from DTO
    Running    = 2,  // Solver actively iterating
    Streaming  = 3,  // Partial results flowing to ResultsDB
    Completed  = 4,  // Terminal: success
    Failed     = 5,  // Terminal: error
    Cancelled  = 6   // Terminal: user-requested abort
};

/// Returns true if the transition from 'from' → 'to' is legal.
inline bool isValidTransition(JobStatus from, JobStatus to) {
    // Terminal states have no outgoing transitions
    if (from == JobStatus::Completed ||
        from == JobStatus::Failed    ||
        from == JobStatus::Cancelled) return false;

    // Monotonic forward or lateral divergence to Cancelled/Failed
    if (to == JobStatus::Cancelled) return true;
    if (to == JobStatus::Failed)    return true;

    return static_cast<uint8_t>(to) == static_cast<uint8_t>(from) + 1;
}

/// Returns true if the status is a terminal state (no further transitions).
inline bool isTerminal(JobStatus s) {
    return s == JobStatus::Completed ||
           s == JobStatus::Failed    ||
           s == JobStatus::Cancelled;
}

} // namespace acutesim
