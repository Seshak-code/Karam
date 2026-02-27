#include "quick_analysis.h"
#include <chrono>
#include <cmath>

// Static member initialization
std::unordered_map<int, double> QuickAnalysis::cachedEstimate_;
size_t QuickAnalysis::cachedHash_ = 0;
bool QuickAnalysis::isStale_ = true;

std::unordered_map<int, double> QuickAnalysis::estimateDC(const TensorNetlist& net) {
    std::unordered_map<int, double> result;

    // Start timeout timer
    auto t0 = std::chrono::steady_clock::now();
    const int TIMEOUT_MS = 100;
    const int numNodes = net.numGlobalNodes;

    if (numNodes <= 0) {
        return result;
    }

    // Simple linearized estimate using source superposition
    // This is a quick approximation, not a full Newton solve

    // Initialize all nodes to 0V
    std::vector<double> voltages(numNodes, 0.0);

    // Access components through globalBlock
    const TensorBlock& block = net.globalBlock;

    // Apply voltage sources directly (they set node voltages)
    for (const auto& vs : block.voltageSources) {
        if (vs.nodePositive > 0 && static_cast<size_t>(vs.nodePositive) <= voltages.size()) {
            voltages[vs.nodePositive - 1] = vs.voltage_V;
        }
    }

    // For each current source, estimate based on load resistance
    // (Very rough: assume 1k default load)
    for (const auto& cs : block.currentSources) {
        if (cs.nodePositive > 0 && static_cast<size_t>(cs.nodePositive) <= voltages.size()) {
            // I * R estimate (assume 1k load)
            voltages[cs.nodePositive - 1] += cs.current_A * 1000.0;
        }
    }

    // Check timeout
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    if (elapsed > TIMEOUT_MS) {
        return result; // Return empty on timeout
    }

    // Build result map
    for (int i = 0; i < numNodes; ++i) {
        result[i + 1] = voltages[i]; // 1-indexed nodes
    }

    // Cache result using block's topology hash
    cachedEstimate_ = result;
    cachedHash_ = block.topologyHash;
    isStale_ = false;

    return result;
}

bool QuickAnalysis::isValid(const std::unordered_map<int, double>& estimate) {
    if (estimate.empty()) {
        return false;
    }

    // Check for unreasonable values (>1000V or NaN)
    for (const auto& [key, v] : estimate) {
        if (std::isnan(v) || std::isinf(v) || std::abs(v) > 1000.0) {
            return false;
        }
    }

    return true;
}

std::unordered_map<int, double> QuickAnalysis::getCached(size_t currentHash) {
    if (!isStale_ && cachedHash_ == currentHash) {
        return cachedEstimate_;
    }
    return {}; // Empty if stale or hash mismatch
}

void QuickAnalysis::invalidate() {
    isStale_ = true;
    cachedEstimate_.clear();
}
