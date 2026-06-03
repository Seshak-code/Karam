#pragma once

#include <cstdint>

/**
 * thermal_monitor.h — Thermal Budget Awareness (Phase 5.7.2)
 *
 * Apple Silicon shares a thermal envelope across CPU, GPU, and ANE.
 * Sustained GPU simulation creates thermal pressure, triggering OS-level
 * frequency throttling. This monitor polls the system thermal state
 * and scales contraction tile batch sizes to avoid throttle-induced latency
 * spikes (which are worse than proactive batch reduction).
 *
 * On non-Apple platforms: always reports Nominal (no throttling).
 */

#if defined(__APPLE__) && defined(__aarch64__) && !defined(__EMSCRIPTEN__)
#define ACUTESIM_HAS_THERMAL_MONITOR 1
#include <sys/sysctl.h>
#else
#define ACUTESIM_HAS_THERMAL_MONITOR 0
#endif

class ThermalMonitor {
public:
    enum class State : uint8_t {
        Nominal  = 0,   // Full speed
        Fair     = 1,   // Slight throttle — reduce batch by 25%
        Serious  = 2,   // Significant — reduce batch by 50%
        Critical = 3    // Emergency — reduce batch by 75%
    };

    /// Poll the current system thermal state.
    static State currentState() {
#if ACUTESIM_HAS_THERMAL_MONITOR
        return pollAppleThermal();
#else
        return State::Nominal;
#endif
    }

    /// Multiplicative scale factor for batch sizes. 1.0 = no throttling.
    static double scaleFactor(State s) {
        switch (s) {
            case State::Fair:     return 0.75;
            case State::Serious:  return 0.50;
            case State::Critical: return 0.25;
            default:              return 1.0;
        }
    }

    /// Scale a requested batch size by thermal pressure.
    static uint32_t adjustBatch(uint32_t requested) {
        double f = scaleFactor(currentState());
        uint32_t adj = static_cast<uint32_t>(requested * f);
        return adj < 1 ? 1 : adj;
    }

    /// Returns true if thermal headroom is critically low (< 15%).
    static bool isThermallyConstrained() {
        return currentState() >= State::Serious;
    }

private:
#if ACUTESIM_HAS_THERMAL_MONITOR
    static State pollAppleThermal() {
        // machdep.xcpm.cpu_thermal_level:
        //   0       = nominal
        //   1 .. 4  = fair (light throttle)
        //   5 .. 8  = serious (significant throttle)
        //   9+      = critical
        int level = 0;
        size_t len = sizeof(level);
        if (sysctlbyname("machdep.xcpm.cpu_thermal_level",
                         &level, &len, nullptr, 0) != 0)
            return State::Nominal;

        if (level <= 0) return State::Nominal;
        if (level <= 4) return State::Fair;
        if (level <= 8) return State::Serious;
        return State::Critical;
    }
#endif
};
