#pragma once
// ============================================================================
// engine_pch.h — Internal Precompiled Header
// ============================================================================
// All internal engine .cpp files should include this first.
// Benefits:
//   - Consistent SIMD flags across all translation units
//   - Compile-time stability (changes here only rebuild engine files)
//   - Migration sanity during extraction (one place to fix stale includes)
//   - Ensures ACUTESIM_PRECISE_FP macros are always available
//
// DO NOT include this from engine_api/ headers.
// DO NOT include this from gui/, ai_api/, or connectivity/ code.
// ============================================================================

#include "acutesim_engine/engine_export.h"
#include "acutesim_engine/internal/precision.h"

// Standard library fundamentals used pervasively in the engine
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cassert>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <optional>
#include <span>

// These will resolve correctly once migration is complete.
// During migration transitional includes are allowed via CMake's
// ${CMAKE_SOURCE_DIR} fallback include path.
#if __has_include("acutesim_engine/math/emulated_f64.h")
    #include "acutesim_engine/math/emulated_f64.h"
#elif __has_include("compute/math/emulated_f64.h")
    #include "acutesim_engine/math/emulated_f64.h"   // transitional
#endif

#if __has_include("acutesim_engine/autodiff/dual.h")
    #include "acutesim_engine/autodiff/dual.h"
#elif __has_include("autodiff/dual.h")
    #include "acutesim_engine/autodiff/dual.h"               // transitional
#endif
