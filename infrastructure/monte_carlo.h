#pragma once
#include "circuitsim.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

struct MCSimStats {
    double mean;
    double stddev;
    double minVal;
    double maxVal;
    int samples;
};

class MonteCarloRunner
{
public:
    /**
     * copyWithVariation
     * Creates a deep copy of the netlist but perturbs component values
     * by a Gaussian distribution (e.g., tolerance = 0.05 for 5%).
     */
    static TensorNetlist copyWithVariation(const TensorNetlist& base, double tolerance, uint32_t seed)
    {
        TensorNetlist varied = base; // Copy structure
        std::mt19937 gen(seed);
        
        // 3-Sigma Rule: tolerance is usually 3 standard deviations
        double sigmaScale = tolerance / 3.0; 
        std::normal_distribution<double> dist(0.0, sigmaScale);

        auto vary = [&](double val) {
            double percentChange = dist(gen);
            return val * (1.0 + percentChange);
        };

        // Perturb Global Components
        for (auto& r : varied.globalBlock.resistors) r.resistance_ohms = vary(r.resistance_ohms);
        for (auto& c : varied.globalBlock.capacitors) c.capacitance_farads = vary(c.capacitance_farads);
        
        // Perturb Power Rails (Vdd Variation)
        for (auto& p : varied.globalBlock.powerRails) {
            if (p.tolerance_percent > 0.0) {
                 // Use specific sigma for this rail
                 double railSigma = p.tolerance_percent * 0.01 / 3.0;
                 std::normal_distribution<double> railDist(0.0, railSigma);
                 double change = railDist(gen);
                 p.nominal_V *= (1.0 + change);
            }
        }

        return varied;
    }

    static void run(int numTrials = 1000)
    {
        std::cout << "==================================================\n";
        std::cout << " MONTE-CARLO SENSITIVITY ANALYSIS\n";
        std::cout << " Circuit: Voltage Divider (Target: 2.500V)\n";
        std::cout << " Tolerance: 5% Resistors (Gaussian distribution)\n";
        std::cout << "==================================================\n";

        // 1. Definition: Simple Voltage Divider (5V source, Two 1k resistors)
        TensorNetlist baseNetlist;
        baseNetlist.addVoltageSource(1, 0, 5.0);
        baseNetlist.addResistor(1, 2, 1000.0); // R1
        baseNetlist.addResistor(2, 0, 1000.0); // R2
        // Output is at node 2. Ideal = 2.5V.

        CircuitSim sim;
        std::vector<double> outputs;
        outputs.reserve(numTrials);

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < numTrials; ++i)
        {
            // Vary parameters by 5%
            TensorNetlist instance = copyWithVariation(baseNetlist, 0.05, 1234 + i);
            
            SolverStep step = sim.solveDC(instance);
            if (step.stats.converged) {
                // Node 2 is index 1
                double v_out = (step.nodeVoltages.size() > 1) ? step.nodeVoltages[1] : 0.0;
                outputs.push_back(v_out);
            } else {
                 std::cout << " [MC Fail] Res: " << step.stats.residual << "\n";
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        
        // Statistical Analysis
        if (outputs.empty()) {
            std::cout << " [ERROR] No trials converged.\n";
            return;
        }

        double sum = 0.0;
        double minVal = 1e9, maxVal = -1e9;
        for (double v : outputs) {
            sum += v;
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
        double mean = sum / outputs.size();

        double sqSum = 0.0;
        for (double v : outputs) {
            sqSum += (v - mean) * (v - mean);
        }
        double stddev = std::sqrt(sqSum / outputs.size());

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "RESULTS:\n";
        std::cout << "  - Samples:      " << outputs.size() << "\n";
        std::cout << "  - Mean Vout:    " << mean << " V\n";
        std::cout << "  - StdDev (sigma): " << stddev << " V\n";
        std::cout << "  - 3-Sigma Range: [" << (mean - 3*stddev) << ", " << (mean + 3*stddev) << "] V\n";
        std::cout << "  - Min/Max:      [" << minVal << ", " << maxVal << "] V\n";
        std::cout << "  - Time:         " << std::chrono::duration<double>(end - start).count() << "s\n";
        std::cout << "==================================================\n";
    }
};
