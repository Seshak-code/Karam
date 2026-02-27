#pragma once
// ============================================================================
// precision.h — IEEE-754 Precision Control Macros
// ============================================================================
// The engine builds with -ffast-math for performance. This is SAFE because
// critical precision paths use EmulatedF64 (hi-lo pair splitting).
//
// However, a few functions (pnjlim, LTE, physical residuals) require IEEE
// strict behavior. Use ACUTESIM_PRECISE_FP_BEGIN/END around those sections.
//
// NEVER use raw #pragma float_control in source files. Always use these
// macros — they handle compiler differences and serve as documentation.
// ============================================================================

#if defined(__clang__)
    #define ACUTESIM_PRECISE_FP_BEGIN \
        _Pragma("clang fp contract(off)") \
        _Pragma("clang optimize off")
    #define ACUTESIM_PRECISE_FP_END \
        _Pragma("clang optimize on")
#elif defined(__GNUC__)
    #define ACUTESIM_PRECISE_FP_BEGIN \
        _Pragma("GCC push_options") \
        _Pragma("GCC optimize(\"no-fast-math\")")
    #define ACUTESIM_PRECISE_FP_END \
        _Pragma("GCC pop_options")
#elif defined(_MSC_VER)
    #define ACUTESIM_PRECISE_FP_BEGIN \
        __pragma(float_control(precise, on, push))
    #define ACUTESIM_PRECISE_FP_END \
        __pragma(float_control(pop))
#else
    // Unknown compiler: no-op (add support if needed)
    #define ACUTESIM_PRECISE_FP_BEGIN
    #define ACUTESIM_PRECISE_FP_END
#endif

// ============================================================================
// PrecisionMode — exposed through engine_api, defined here for internal use
// ============================================================================
// Note: The PrecisionMode enum is declared in engine_api/isimulation_engine.h
// and used here only for documentation clarity.
//
// PrecisionMode::Native_F64    — standard IEEE 754 double (CPU scalar path)
// PrecisionMode::Emulated_F64  — hi-lo pair splitting (GPU determinism path)
// PrecisionMode::Mixed         — F64 for NR outer loop, EmF64 for GPU kernels
// ============================================================================
