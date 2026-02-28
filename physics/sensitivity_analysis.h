#pragma once
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include "acutesim_engine/netlist/circuit.h"
#include "statistical_engine.h"

// Note: Requires a solve() callback or function pointer to run simulations
#include <functional>

namespace Analysis {

struct SensitivityResult {
    std::string componentName;
    std::string paramName;
    double sensitivity;       // Absolute: dV/dP
    double normSensitivity;   // Normalized: (dV/dP) * (P/V)
    double originalParamValue;
    
    // For sorting
    bool operator>(const SensitivityResult& other) const {
        return std::abs(normSensitivity) > std::abs(other.normSensitivity);
    }
};

class SensitivityEngine {
public:
    using SimulatorFunc = std::function<double(TensorBlock&)>;

    // Calculate sensitivities for a target output metric (e.g., node voltage)
    static std::vector<SensitivityResult> analyze(
        const TensorBlock& nominalBlock,
        SimulatorFunc runSimulation,
        double perturbationPercent = 0.1 // 0.1% perturbation
    ) {
        std::vector<SensitivityResult> results;
        
        // 1. Run Nominal Simulation
        TensorBlock tempBlock = nominalBlock; // Deep copy
        double nominalOutput = runSimulation(tempBlock);
        
        if (std::abs(nominalOutput) < 1e-12) nominalOutput = 1e-12; // Avoid div by zero

        double perturbationFactor = 1.0 + (perturbationPercent / 100.0);

        // 2. Iterate Resistors
        for (const auto& r : nominalBlock.resistors) {
            TensorBlock perturbed = nominalBlock;
            // Find and perturb
            for (auto& pr : perturbed.resistors) {
                if (pr.name == r.name) {
                    pr.resistance_ohms *= perturbationFactor;
                    break;
                }
            }
            
            double perturbedOutput = runSimulation(perturbed);
            double dOutput = perturbedOutput - nominalOutput;
            double dParam = r.resistance_ohms * (perturbationPercent / 100.0);
            
            double sensitivity = dOutput / dParam;
            double normSensitivity = sensitivity * (r.resistance_ohms / nominalOutput);
            
            results.push_back({r.name, "resistance", sensitivity, normSensitivity, r.resistance_ohms});
        }
        
        // 3. Iterate Capacitors (Similar logic - focusing on AC or Transient metrics usually)
        // For DC sensitivity, Caps don't matter, but for general sensitivity they do.
        // We assume runSimulation handles the metric extraction (e.g. Bandwidth).
        for (const auto& c : nominalBlock.capacitors) {
            TensorBlock perturbed = nominalBlock;
            for (auto& pc : perturbed.capacitors) {
                if (pc.name == c.name) {
                    pc.capacitance_farads *= perturbationFactor;
                    break;
                }
            }
            
            double perturbedOutput = runSimulation(perturbed);
            double dOutput = perturbedOutput - nominalOutput;
            double dParam = c.capacitance_farads * (perturbationPercent / 100.0);
            
            double sensitivity = dOutput / dParam;
            double normSensitivity = sensitivity * (c.capacitance_farads / nominalOutput);
            
            results.push_back({c.name, "capacitance", sensitivity, normSensitivity, c.capacitance_farads});
        }

        // Sort by impact (Normalized Sensitivity)
        std::sort(results.begin(), results.end(), std::greater<SensitivityResult>());
        
        return results;
    }
};

} // namespace Analysis
