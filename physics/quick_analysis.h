#pragma once

#include <unordered_map>
#include "acutesim_engine/netlist/circuit.h"
#include "../physics/circuitsim.h"

/**
 * QuickAnalysis - Fast approximate DC analysis for hover previews
 *
 * Part of Phase 1.7: Simulation Suite
 *
 * Features:
 * - Single Newton iteration (linearized)
 * - Hard 100ms timeout
 * - Returns approximate node voltages
 * - Cached until topology changes
 */
class QuickAnalysis {
public:
    // Run quick DC estimate (single iteration, linearized)
    // Returns map of nodeId -> estimated voltage
    static std::unordered_map<int, double> estimateDC(const TensorNetlist& net);

    // Check if estimate is valid (no singularity, reasonable values)
    static bool isValid(const std::unordered_map<int, double>& estimate);

    // Get cached result (returns empty if stale)
    static std::unordered_map<int, double> getCached(size_t currentHash);

    // Clear cache
    static void invalidate();

private:
    static std::unordered_map<int, double> cachedEstimate_;
    static size_t cachedHash_;
    static bool isStale_;
};
