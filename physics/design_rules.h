#pragma once

#include "../components/circuit.h"
#include "../components/component_library.h"
#include <vector>
#include <string>
#include <set>
#include <cmath>

/**
 * design_rules.h - Physics-Backed Design Rule Checking Engine
 * 
 * Architecture Notes:
 * - All checks are stateless predicates for GPU portability
 * - Uses uint8_t bitmasks for parallel rule checking (GPU-friendly)
 * - Raw State: Always operates in emulated f64 mode for maximum accuracy
 * - SoA-compatible for batch validation on tensors
 */

namespace DesignRules {

// ============================================================================
// PRECISION (RAW STATE - ALWAYS HIGH ACCURACY)
// ============================================================================

/**
 * All compute paths now operate in "Raw State" using emulated F64 
 * (Double-Float) for maximum physical accuracy.
 */
inline double getTolerance(double base_tol) {
    return base_tol; 
}

// ============================================================================
// VIOLATION CODES (BITMASK FOR GPU PARALLEL CHECK)
// ============================================================================

namespace Violation {
    constexpr uint8_t NONE                 = 0;
    constexpr uint8_t FLOATING_GATE        = 1 << 0;  // Gate with no DC path
    constexpr uint8_t OPEN_DRAIN           = 1 << 1;  // Drain with no load
    constexpr uint8_t VDD_GND_SHORT        = 1 << 2;  // Rail short circuit
    constexpr uint8_t INDEPENDENT_SOURCE   = 1 << 3;  // V/I source at interface
    constexpr uint8_t THERMAL_OVERLOAD     = 1 << 4;  // P > P_max
    constexpr uint8_t CROSSTALK_RISK       = 1 << 5;  // High-Z parallel paths
    constexpr uint8_t EMC_VIOLATION        = 1 << 6;  // Fast edge / loop area
    constexpr uint8_t MULTIPLE_DRIVERS     = 1 << 7;  // Multiple sources on net
}

// ============================================================================
// VIOLATION REPORT (SERIALIZABLE FOR AI API)
// ============================================================================

struct ViolationEntry {
    uint8_t code;              // Bitmask of violations
    int device_index = -1;     // Index in tensor (or -1 for net-level)
    int node_id = -1;          // Related node (if applicable)
    std::string message;       // Human-readable description
    
    // Severity levels per mode
    enum class Severity { INFO, WARN, ERR };
    Severity getSeverity() const;
};

struct ViolationReport {
    std::vector<ViolationEntry> entries;
    bool hasErrors = false;
    bool hasWarnings = false;
    
    void add(uint8_t code, int device_idx, int node, const std::string& msg) {
        entries.push_back({code, device_idx, node, msg});
        // Note: hasErrors/hasWarnings set during mode-aware postprocess
    }
    
    bool isClean() const { return entries.empty(); }
    
    // Serialize for AI API consumption
    std::string toJSON() const;
};

// ============================================================================
// UNIFIED CONSTRAINTS (RAW STATE)
// ============================================================================

struct RawConstraints {
    static constexpr bool allowIdealSources = false;
    static constexpr bool requireRailDrivers = true;
    static constexpr bool enforceSingleDriverPerRail = true;
    static constexpr double maxVoltageSwing_V = 20.0;
    static constexpr double maxPowerPerDevice_W = 5.0;
    static constexpr double maxCurrentDensity_A_um2 = 1e-6;  // Electromigration
    static constexpr bool checkEMC = true;
    
    static ViolationEntry::Severity severity(uint8_t code) {
        using S = ViolationEntry::Severity;
        // In Raw State, most physical violations are errors
        if (code & (Violation::VDD_GND_SHORT | Violation::FLOATING_GATE | Violation::MULTIPLE_DRIVERS)) 
            return S::ERR;
        if (code & (Violation::INDEPENDENT_SOURCE | Violation::OPEN_DRAIN))
            return S::ERR;
        if (code & Violation::THERMAL_OVERLOAD) 
            return S::WARN;
        return S::INFO;
    }
};

// ============================================================================
// PHYSICS-BACKED VALIDATION CHECKS
// ============================================================================

namespace PhysicsChecks {

/**
 * Check for floating gates (no DC path to reference).
 */
inline bool isFloatingGate(int gateNode, const TensorBlock& block,
                           const std::set<int>& groundedNodes) {
    return groundedNodes.find(gateNode) == groundedNodes.end();
}

/**
 * Check for open drain (no load path).
 */
inline bool isOpenDrain(int drainNode, const TensorBlock& block) {
    int connections = 0;
    for (const auto& r : block.resistors) {
        if (r.nodeTerminal1 == drainNode || r.nodeTerminal2 == drainNode)
            connections++;
    }
    for (const auto& m : block.mosfets) {
        if (m.drain == drainNode) connections++;
    }
    for (const auto& q : block.bjts) {
        if (q.nodeCollector == drainNode) connections++;
    }
    return connections < 2;
}

/**
 * Check for VDD/GND short.
 */
inline bool isRailShort(int node1, int node2, const TensorBlock& block, 
                         double tolerance = 1.0) {
    for (const auto& r : block.resistors) {
        if ((r.nodeTerminal1 == node1 && r.nodeTerminal2 == node2) ||
            (r.nodeTerminal1 == node2 && r.nodeTerminal2 == node1)) {
            if (r.resistance_ohms < tolerance) return true;
        }
    }
    return false;
}

/**
 * Check thermal overload.
 */
inline bool isThermalOverload(double voltage_drop, double current, 
                               double maxPower_W) {
    double power = std::abs(voltage_drop * current);
    return power > maxPower_W;
}

} // namespace PhysicsChecks

// ============================================================================
// MAIN VALIDATION FUNCTION
// ============================================================================

/**
 * Validate a TensorBlock against "Raw State" physical design rules.
 */
inline ViolationReport validate(const TensorBlock& block) {
    ViolationReport report;
    
    // Build set of grounded nodes
    std::set<int> groundedNodes;
    groundedNodes.insert(0);
    
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& r : block.resistors) {
            bool n1Grounded = groundedNodes.count(r.nodeTerminal1) > 0;
            bool n2Grounded = groundedNodes.count(r.nodeTerminal2) > 0;
            if (n1Grounded && !n2Grounded) {
                groundedNodes.insert(r.nodeTerminal2);
                changed = true;
            }
            if (n2Grounded && !n1Grounded) {
                groundedNodes.insert(r.nodeTerminal1);
                changed = true;
            }
        }
    }
    
    // Check MOSFETs
    for (size_t i = 0; i < block.mosfets.size(); ++i) {
        const auto& m = block.mosfets[i];
        if (PhysicsChecks::isFloatingGate(m.gate, block, groundedNodes)) {
            report.add(Violation::FLOATING_GATE, i, m.gate,
                      "MOSFET " + m.instanceName + " has floating gate");
        }
        if (PhysicsChecks::isOpenDrain(m.drain, block)) {
            report.add(Violation::OPEN_DRAIN, i, m.drain,
                      "MOSFET " + m.instanceName + " drain has no load");
        }
    }
    
    // Check BJTs
    for (size_t i = 0; i < block.bjts.size(); ++i) {
        const auto& q = block.bjts[i];
        if (groundedNodes.find(q.base) == groundedNodes.end()) {
            report.add(Violation::FLOATING_GATE, i, q.base,
                      "BJT " + q.instanceName + " has floating base");
        }
    }
    
    // Check for independent sources (Raw state only allows RailDrivers)
    for (const auto& vs : block.voltageSources) {
        if (vs.type != "RailDriver") {
            report.add(Violation::INDEPENDENT_SOURCE, -1, vs.nodePositive, 
                      "Generic Voltage Source not allowed (Use RailDriver for power)");
        }
    }

    if (!block.currentSources.empty()) {
        report.add(Violation::INDEPENDENT_SOURCE, -1, -1,
                  "Independent current source not allowed in raw state");
    }
    
    // Check Multiple Drivers
    std::map<int, int> posDriverCount;
    for (const auto& vs : block.voltageSources) {
        if (vs.nodePositive > 0) posDriverCount[vs.nodePositive]++;
    }
    for (const auto& kv : posDriverCount) {
        if (kv.second > 1) {
            report.add(Violation::MULTIPLE_DRIVERS, -1, kv.first,
                      "Node " + std::to_string(kv.first) + " driven by multiple sources");
        }
    }
    
    // Check VDD/GND shorts
    for (const auto& r : block.resistors) {
        if (r.resistance_ohms < getTolerance(1.0)) {
            report.add(Violation::VDD_GND_SHORT, -1, r.nodeTerminal1,
                      "Potential short circuit detected (R < 1Ω)");
        }
    }
    
    // Apply severity and set flags
    for (auto& entry : report.entries) {
        auto sev = RawConstraints::severity(entry.code);
        if (sev == ViolationEntry::Severity::ERR) report.hasErrors = true;
        if (sev == ViolationEntry::Severity::WARN) report.hasWarnings = true;
    }
    
    return report;
}

// ============================================================================
// JSON SERIALIZATION (AI API)
// ============================================================================

inline std::string ViolationReport::toJSON() const {
    std::string json = "{\"violations\":[";
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        json += "{\"code\":" + std::to_string(e.code) + ",";
        json += "\"device\":" + std::to_string(e.device_index) + ",";
        json += "\"node\":" + std::to_string(e.node_id) + ",";
        json += "\"message\":\"" + e.message + "\"}";
        if (i < entries.size() - 1) json += ",";
    }
    json += "],\"hasErrors\":" + std::string(hasErrors ? "true" : "false");
    json += ",\"hasWarnings\":" + std::string(hasWarnings ? "true" : "false") + "}";
    return json;
}

inline ViolationEntry::Severity ViolationEntry::getSeverity() const {
    return RawConstraints::severity(code);
}

// ============================================================================
// GPU BATCH VALIDATION (TENSOR-COMPATIBLE)
// ============================================================================

namespace GPUBatch {

/**
 * Batch validate diode tensor.
 * 
 * For GPU: Each diode is validated independently (embarrassingly parallel).
 * Uses precision mode for tolerance checks.
 * 
 * @param diodes       Input tensor of diodes
 * @param voltages     Current node voltages (for thermal check)
 * @param validity     Output: bitmask per device (0 = valid)
 * @param maxPower     Maximum allowed power per device
 */
inline void validateDiodes(
    const std::vector<int>& anodes,
    const std::vector<int>& cathodes,
    const std::vector<double>& currents,  // Pre-computed from physics
    const std::vector<double>& nodeVoltages,
    std::vector<uint8_t>& validity,
    double maxPower = 5.0)
{
    size_t n = anodes.size();
    validity.resize(n, Violation::NONE);
    
    // GPU: This loop would be parallelized across threads
    for (size_t i = 0; i < n; ++i) {
        double va = (anodes[i] > 0 && anodes[i] < (int)nodeVoltages.size()) 
                   ? nodeVoltages[anodes[i]] : 0.0;
        double vc = (cathodes[i] > 0 && cathodes[i] < (int)nodeVoltages.size()) 
                   ? nodeVoltages[cathodes[i]] : 0.0;
        double vdrop = std::abs(va - vc);
        double power = vdrop * std::abs(currents[i]);
        
        if (power > maxPower) {
            validity[i] |= Violation::THERMAL_OVERLOAD;
        }
    }
}

/**
 * Batch validate MOSFET tensor for floating gates.
 * 
 * @param gates           Gate node indices
 * @param groundedMask    Pre-computed: 1 if node has DC path to GND
 * @param validity        Output bitmask per device
 */
inline void validateMosfetGates(
    const std::vector<int>& gates,
    const std::vector<uint8_t>& groundedMask,  // groundedMask[node] = 1 if grounded
    std::vector<uint8_t>& validity)
{
    size_t n = gates.size();
    validity.resize(n, Violation::NONE);
    
    // GPU: Parallel across devices
    for (size_t i = 0; i < n; ++i) {
        int g = gates[i];
        if (g >= 0 && g < (int)groundedMask.size()) {
            if (groundedMask[g] == 0) {
                validity[i] |= Violation::FLOATING_GATE;
            }
        }
    }
}

} // namespace GPUBatch

} // namespace DesignRules
