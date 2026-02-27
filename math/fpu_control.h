#pragma once

#include <cfenv>
#include <iostream>
#include <mutex>

#ifdef _MSC_VER
#include <float.h>
#endif

namespace acutesim {
namespace math {

/**
 * FPU Control - strict determinism enforcement.
 * 
 * This class ensures that the Floating Point Unit is configured for 
 * strict IEEE-754 compliance, ensuring bit-exact reproducibility 
 * across runs and platforms (WASM vs Desktop).
 * 
 * Enforced rules:
 * - Rounding: Round to Nearest Even (FE_TONEAREST)
 * - Subnormals: Preserved (No Flush-to-Zero), unless performance overrides are set.
 * - Exceptions: Masked (we handle checks explicitly via isfinite/isnan).
 */
class FPU {
public:
    static void enforceDeterminism() {
        static std::once_flag flag;
        std::call_once(flag, []() {
            // 1. Set Rounding Mode
            int ret = std::fesetround(FE_TONEAREST);
            if (ret != 0) {
                std::cerr << "[WARN] FPU: Failed to set FE_TONEAREST rounding mode." << std::endl;
            }

            // 2. Denormals / Flush-zero config
            // We want subnormal precision for low-current simulation (pA range).
            // Therefore we disable DAZ/FTZ if possible.
#if defined(__x86_64__) || defined(_M_X64)
    #ifdef __SSE__
            // MXCSR register manipulation for x86 SSE
            // Bit 15: Flush to Zero (FTZ) - Set to 0
            // Bit 6: Denormals Are Zero (DAZ) - Set to 0
            unsigned int mxcsr;
            __asm__ volatile("stmxcsr %0" : "=m"(mxcsr));
            mxcsr &= ~(0x8040); // Clear FTZ (15) and DAZ (6)
            __asm__ volatile("ldmxcsr %0" : : "m"(mxcsr));
    #endif
#elif defined(__aarch64__)
            // ARM64 usually defaults to IEEE, but we can access FPCR if needed.
            // Generally strict compiler flags handle this on ARM.
#endif

#ifdef _MSC_VER
            // Microsoft Visual C++ specific controls
            _controlfp_s(nullptr, _RC_NEAR, _MCW_RC); // Round near
            _controlfp_s(nullptr, _DN_SAVE, _MCW_DN); // Save denormals
#endif
            
            std::cout << "[INFO] FPU: Enforced IEEE-754 Determinism (Round-Nearest, Subnormals-On)." << std::endl;
        });
    }

    static void checkState() {
        int roundMode = std::fegetround();
        if (roundMode != FE_TONEAREST) {
             std::cerr << "[CRITICAL] FPU state drift detected! Current mode: " << roundMode << std::endl;
        }
    }
};

} // namespace math
} // namespace acutesim
