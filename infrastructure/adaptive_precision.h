#pragma once

#include "../infrastructure/graph_partitioner.h"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>

/**
 * adaptive_precision.h — Variable Rate Shading (VRS) for Circuit Simulation
 *
 * Graphics Analogy: VRS renders complex, high-motion regions at full resolution
 * while simple/static regions use lower resolution. Here we do the same for
 * circuit partitions:
 *
 *   FP64_FULL:     Partitions with active switching (high dV/dt) — full precision
 *   FP32_RELAXED:  Moderate activity — single precision (2× throughput on GPU)
 *   CACHED_REUSE:  Near-static partitions (dV/dt ≈ 0) — reuse previous solution
 *
 * Mathematical Basis:
 *   Mixed-precision iterative refinement: do most work in FP16/FP32, correct in
 *   FP64. The outer NR loop already provides this correction automatically —
 *   the Jacobian evaluation precision affects convergence rate but not final
 *   accuracy (verified by the KCL residual check in FP64).
 *
 * Usage:
 *   AdaptivePrecisionRouter router;
 *   auto assignments = router.assignTiers(partitions, voltages, prev_voltages, dt);
 *   // ... evaluate only FP64_FULL and FP32_RELAXED partitions ...
 *   // ... skip CACHED_REUSE partitions (reuse previous NR solution) ...
 */

// ─── Precision Tiers ─────────────────────────────────────────────────────────

enum class PrecisionTier {
    FP64_FULL,       // Full double-precision evaluation
    FP32_RELAXED,    // Single-precision Jacobian + FP64 outer correction
    CACHED_REUSE     // Skip evaluation, reuse previous converged result
};

// ─── Per-Partition Activity Metadata ──────────────────────────────────────────

struct SubgraphActivity {
    uint32_t partitionId;
    double maxDvDt;            // Maximum |dV/dt| across all nodes in partition
    double maxDiDt;            // Maximum |dI/dt| (estimated from dV/dt × G_max)
    double rmsActivity;        // RMS of |dV/dt| across partition nodes
    PrecisionTier assignedTier;
};

// ─── Adaptive Precision Router ────────────────────────────────────────────────

class AdaptivePrecisionRouter {
public:
    // Thresholds for tier assignment (configurable)
    struct Config {
        double switchingThreshold = 1e3;   // |dV/dt| > this → FP64_FULL (V/s)
        double staticThreshold    = 1.0;   // |dV/dt| < this → CACHED_REUSE (V/s)
        bool   enableCachedReuse  = true;  // Set false to disable caching
        bool   enableFP32         = false; // Set true when GPU FP32 path exists
    };

    Config config;

    /**
     * Assign precision tiers to all partitions based on voltage activity.
     *
     * @param partitions      Graph partitions from Phase C
     * @param currentVoltages Current node voltages (0-based, size = nodeCount)
     * @param prevVoltages    Previous step's voltages
     * @param dt              Current timestep
     * @return Per-partition activity assessments with tier assignments
     */
    std::vector<SubgraphActivity> assignTiers(
        const std::vector<TensorPartition>& partitions,
        const std::vector<double>& currentVoltages,
        const std::vector<double>& prevVoltages,
        double dt) const
    {
        std::vector<SubgraphActivity> result;
        result.reserve(partitions.size());

        if (dt <= 0 || currentVoltages.empty() || prevVoltages.empty() ||
            currentVoltages.size() != prevVoltages.size()) {
            // Cannot compute rates — default everything to FP64
            for (size_t i = 0; i < partitions.size(); ++i) {
                result.push_back({partitions[i].workgroupId, 0.0, 0.0, 0.0,
                                  PrecisionTier::FP64_FULL});
            }
            return result;
        }

        for (const auto& partition : partitions) {
            SubgraphActivity activity;
            activity.partitionId = partition.workgroupId;
            activity.maxDvDt = 0.0;
            activity.maxDiDt = 0.0;
            activity.rmsActivity = 0.0;

            double sumSq = 0.0;
            uint32_t count = 0;

            for (uint32_t nodeIdx : partition.nodeSet) {
                if (nodeIdx == 0) continue;  // skip ground
                size_t idx = static_cast<size_t>(nodeIdx - 1);  // 1-based to 0-based
                if (idx >= currentVoltages.size() || idx >= prevVoltages.size()) continue;

                double dv = currentVoltages[idx] - prevVoltages[idx];
                double dv_dt = std::abs(dv / dt);

                activity.maxDvDt = std::max(activity.maxDvDt, dv_dt);
                sumSq += dv_dt * dv_dt;
                ++count;
            }

            if (count > 0) {
                activity.rmsActivity = std::sqrt(sumSq / static_cast<double>(count));
            }

            // Tier assignment
            if (activity.maxDvDt > config.switchingThreshold) {
                activity.assignedTier = PrecisionTier::FP64_FULL;
            } else if (config.enableCachedReuse &&
                       activity.maxDvDt < config.staticThreshold) {
                activity.assignedTier = PrecisionTier::CACHED_REUSE;
            } else if (config.enableFP32) {
                activity.assignedTier = PrecisionTier::FP32_RELAXED;
            } else {
                activity.assignedTier = PrecisionTier::FP64_FULL;
            }

            result.push_back(activity);
        }

        return result;
    }

    /**
     * Count how many partitions are in each tier.
     * Useful for diagnostics and performance monitoring.
     */
    static void countTiers(const std::vector<SubgraphActivity>& activities,
                           uint32_t& fp64Count, uint32_t& fp32Count,
                           uint32_t& cachedCount) {
        fp64Count = fp32Count = cachedCount = 0;
        for (const auto& a : activities) {
            switch (a.assignedTier) {
                case PrecisionTier::FP64_FULL:     ++fp64Count; break;
                case PrecisionTier::FP32_RELAXED:  ++fp32Count; break;
                case PrecisionTier::CACHED_REUSE:  ++cachedCount; break;
            }
        }
    }

    /**
     * Compute the theoretical speedup from skipping cached partitions.
     * Returns ratio of total partitions to actively-evaluated partitions.
     */
    static double estimatedSpeedup(const std::vector<SubgraphActivity>& activities) {
        if (activities.empty()) return 1.0;
        uint32_t active = 0;
        for (const auto& a : activities) {
            if (a.assignedTier != PrecisionTier::CACHED_REUSE) ++active;
        }
        if (active == 0) return 1.0;  // all cached — don't divide by zero
        return static_cast<double>(activities.size()) / static_cast<double>(active);
    }
};
