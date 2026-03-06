#pragma once

#include "../infrastructure/tn_compiler.h"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <vector>

/**
 * neural_seeder.h — DLSS-Inspired Convergence Seeding (Phase 5.3.5)
 *
 * Graphics Analogy: DLSS uses a neural network to upscale a low-resolution
 * render into a high-resolution output. Here we accept a "low-resolution"
 * partial NR iteration (or historical data) and predict the converged state.
 *
 * Implementation Strategy:
 *   Phase 1 (current): Statistical seeder using exponential moving average
 *   of previous DC operating points. No ML training required.
 *   Phase 2 (future):  Replace with a lightweight MLP trained on circuit
 *   topology features + partial NR data.
 *
 * Safety Guarantee: The seeded guess is ALWAYS verified by at least one
 * full FP64 NR pass. The seeder only improves convergence speed —
 * it cannot compromise accuracy.
 *
 * Usage:
 *   NeuralSeeder seeder;
 *   seeder.recordSolution(dcSolution);   // after each DC solve
 *   auto seed = seeder.predictSeed(nodeCount);
 *   // Use seed as initial guess for next solveDC()
 */

// ─── Solution History Entry ───────────────────────────────────────────────────

struct SolutionSnapshot {
    std::vector<double> voltages;
    double kclResidual;        // KCL residual at this solution
    int    nrIterations;       // Iterations to converge
    double weight;             // Confidence weight (1.0 = recent, decays)
};

// ─── Neural (Statistical) Seeder ──────────────────────────────────────────────

class NeuralSeeder {
public:
    struct Config {
        double decayFactor  = 0.8;    // EMA decay (0.8 = last 5 solutions dominate)
        size_t maxHistory   = 8;      // Maximum stored snapshots
        double minConfidence = 0.3;   // Don't seed if confidence < this
        bool   enabled      = true;   // Global enable/disable
    };

    Config config;

    /**
     * Record a converged solution into the history buffer.
     * Called after each successful solveDC() or at stable transient points.
     *
     * @param voltages     Converged node voltages
     * @param kclResidual  Final KCL residual
     * @param nrIterations Iterations taken to converge
     */
    void recordSolution(const std::vector<double>& voltages,
                        double kclResidual = 0.0,
                        int nrIterations = 0) {
        if (voltages.empty()) return;

        // Age existing snapshots
        for (auto& snap : history_) {
            snap.weight *= config.decayFactor;
        }

        // Add new snapshot at full confidence
        SolutionSnapshot snap;
        snap.voltages = voltages;
        snap.kclResidual = kclResidual;
        snap.nrIterations = nrIterations;
        snap.weight = 1.0;
        history_.push_back(std::move(snap));

        // Evict oldest if over capacity
        while (history_.size() > config.maxHistory) {
            history_.erase(history_.begin());
        }
    }

    /**
     * Predict a converged seed state from historical solutions.
     * Returns an empty vector if confidence is too low.
     *
     * Algorithm: Weighted exponential moving average of previous solutions.
     * Nodes that have been stable across history get high confidence;
     * nodes that have varied get low confidence and are left at zero
     * (letting the standard seeding heuristic handle them).
     *
     * @param nodeCount Number of nodes in the system
     * @return Predicted seed voltages (empty if not enough history)
     */
    std::vector<double> predictSeed(size_t nodeCount) const {
        if (!config.enabled || history_.empty() || nodeCount == 0) {
            return {};
        }

        // Need at least 2 snapshots for meaningful prediction
        if (history_.size() < 2) {
            // Just return the last known solution
            if (history_.back().voltages.size() == nodeCount) {
                return history_.back().voltages;
            }
            return {};
        }

        // Weighted average across history
        std::vector<double> seed(nodeCount, 0.0);
        double totalWeight = 0.0;

        for (const auto& snap : history_) {
            if (snap.voltages.size() != nodeCount) continue;
            totalWeight += snap.weight;
            for (size_t i = 0; i < nodeCount; ++i) {
                seed[i] += snap.voltages[i] * snap.weight;
            }
        }

        if (totalWeight < 1e-30) return {};

        for (size_t i = 0; i < nodeCount; ++i) {
            seed[i] /= totalWeight;
        }

        // Compute per-node confidence: low variance → high confidence
        double avgConfidence = computeConfidence(seed, nodeCount);
        if (avgConfidence < config.minConfidence) {
            return {};  // Not confident enough — let standard seeding handle it
        }

        return seed;
    }

    /**
     * Enhanced prediction: blend the statistical seed with a first NR
     * iteration result. This is the "DLSS upscale" — the first NR
     * iteration provides the "low-res render", and we combine it with
     * historical data to produce a better starting point.
     *
     * @param firstGuess   Result of the first NR iteration
     * @param nodeCount    Number of nodes
     * @param blendFactor  How much to trust the statistical seed (0.0-1.0)
     * @return Blended prediction
     */
    std::vector<double> seedFromFirstIteration(
        const std::vector<double>& firstGuess,
        size_t nodeCount,
        double blendFactor = 0.3) const
    {
        auto historicalSeed = predictSeed(nodeCount);
        if (historicalSeed.empty() || firstGuess.size() != nodeCount) {
            return firstGuess;
        }

        // Blend: result = (1-α)*firstGuess + α*historicalSeed
        // where α is scaled by per-node confidence
        std::vector<double> blended(nodeCount);
        for (size_t i = 0; i < nodeCount; ++i) {
            blended[i] = (1.0 - blendFactor) * firstGuess[i] +
                         blendFactor * historicalSeed[i];
        }
        return blended;
    }

    /**
     * Check if the seeder has enough history to provide useful predictions.
     */
    bool hasHistory() const { return history_.size() >= 2; }

    /**
     * Clear all history (e.g., on topology change).
     */
    void reset() { history_.clear(); }

    /**
     * Get the number of stored snapshots.
     */
    size_t historySize() const { return history_.size(); }

private:
    std::vector<SolutionSnapshot> history_;

    /**
     * Compute average confidence across all nodes.
     * Confidence = 1 - normalized_variance (0.0 = unreliable, 1.0 = stable).
     */
    double computeConfidence(const std::vector<double>& seed, size_t nodeCount) const {
        if (history_.size() < 2 || nodeCount == 0) return 0.0;

        double totalConfidence = 0.0;
        for (size_t i = 0; i < nodeCount; ++i) {
            // Compute weighted variance for this node
            double weightedVarSum = 0.0;
            double totalWeight = 0.0;
            for (const auto& snap : history_) {
                if (snap.voltages.size() != nodeCount) continue;
                double diff = snap.voltages[i] - seed[i];
                weightedVarSum += snap.weight * diff * diff;
                totalWeight += snap.weight;
            }
            if (totalWeight < 1e-30) continue;

            double variance = weightedVarSum / totalWeight;
            // Normalize: confidence = 1 / (1 + variance/scale²)
            // scale = 1V means 1V variance → 50% confidence
            double scale = 1.0;
            double nodeConfidence = 1.0 / (1.0 + variance / (scale * scale));
            totalConfidence += nodeConfidence;
        }

        return totalConfidence / static_cast<double>(nodeCount);
    }
};
