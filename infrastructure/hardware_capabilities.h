#pragma once

/**
 * compute/hardware_capabilities.h
 *
 * Queryable struct for CPU SIMD tier and GPU availability.
 * Used by: kernel dispatch, GUI WorkerStatusPanel, batch runner.
 * Qt-free, no gRPC — safe to include anywhere in the stack.
 *
 * Call detectCapabilities() once at startup (it is thread-safe).
 *
 * Phase 3.4
 */

#include <string>

namespace acutesim::compute {

struct HardwareCapabilities {
    enum class SIMDTier {
        SCALAR,   // No SIMD (generic C++)
        AVX2,     // x86 AVX2 + FMA
        AVX512,   // x86 AVX-512F
        NEON      // ARM Neon (Apple Silicon / AArch64)
    };

    SIMDTier    simdTier        = SIMDTier::SCALAR;
    bool        hasGPU          = false;
    std::string gpuAdapterName;       // e.g. "Apple M3 Pro" or ""
    int         physicalCpuCores = 1;
};

/**
 * Detect CPU SIMD tier via CPUID / compiler feature macros.
 * Detect GPU via Dawn adapter enumeration (if Dawn is available).
 * Thread-safe: uses a static-local result; safe to call from any thread.
 */
HardwareCapabilities detectCapabilities();

/**
 * Returns a human-readable tier string.
 * "AVX-512" | "AVX2" | "NEON" | "Scalar"
 */
const char* simdTierName(HardwareCapabilities::SIMDTier tier);

} // namespace acutesim::compute
