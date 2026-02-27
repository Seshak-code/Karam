#pragma once
// ============================================================================
// compute/solvers/register_analyses.h — Force self-registration of all solvers
// ============================================================================
// Call registerAllAnalyses() once at program startup to ensure static-init
// self-registration objects are not dead-stripped by the linker.
//
// This header exists because C++ does not guarantee static-init objects in
// unreferenced translation units are initialised when linking against a
// static library. Calling registerAllAnalyses() creates a direct reference
// that forces the linker to include all solver TUs.
//
// Phase 4.5
// ============================================================================

namespace acutesim {
namespace compute {
namespace solvers {

/**
 * Explicitly trigger registration of all Phase 4 analysis handlers:
 *   - TRANSFER_FUNCTION
 *   - POLE_ZERO
 *   - FOURIER
 *
 * Call once from main() or from any guaranteed early-init site.
 * Safe to call multiple times (idempotent via handler map overwrite).
 */
void registerAllAnalyses();

} // namespace solvers
} // namespace compute
} // namespace acutesim
