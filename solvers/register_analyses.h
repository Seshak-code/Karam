#pragma once
// ============================================================================
// compute/solvers/register_analyses.h — Force self-registration of all solvers
// ============================================================================
// Call registerAllAnalyses() once at program startup to ensure static-init
// self-registration objects are not dead-stripped by the linker.
// ============================================================================

#include "engine_api/isimulation_engine.h"

namespace acutesim {
namespace compute {
namespace solvers {

/**
 * Explicitly trigger registration of all Phase 4 analysis handlers.
 */
ENGINE_API void registerAllAnalyses();

} // namespace solvers
} // namespace compute
} // namespace acutesim
