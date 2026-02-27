#pragma once
#include <vector>
#include <random>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include "../components/circuit.h"

/**
 * statistical_engine.h
 * 
 * Provides robust statistical sampling for Monte Carlo analysis.
 * Features:
 *  - Distributions: Gaussian, Uniform, Lognormal
 *  - Strategy: Random, Latin Hypercube Sampling (LHS)
 *  - Batch Execution: Helper structures for parallel runs
 */

namespace Statistics {

enum class DistributionType {
    GAUSSIAN,
    UNIFORM,
    LOGNORMAL
};

struct Distribution {
    DistributionType type;
    double param1; // Mean (Gaussian), Min (Uniform), LogMean (LogNormal)
    double param2; // Sigma (Gaussian), Max (Uniform), LogSigma (LogNormal)

    static Distribution Gaussian(double mean, double sigma) {
        return {DistributionType::GAUSSIAN, mean, sigma};
    }
    
    static Distribution Uniform(double min, double max) {
        return {DistributionType::UNIFORM, min, max};
    }

    // Typical for resistor/process variation where values must be > 0
    static Distribution Lognormal(double mean, double sigma) {
         // Convert normal mean/sigma to log-space params
         double var = sigma * sigma;
         double mu = std::log(mean * mean / std::sqrt(var + mean * mean));
         double s = std::sqrt(std::log(1 + var / (mean * mean)));
         return {DistributionType::LOGNORMAL, mu, s};
    }
};

enum class SamplingStrategy {
    RANDOM,              // Naive Monte Carlo
    LATIN_HYPERCUBE      // Stratified Sampling (Better coverage)
};

class RandomEngine {
public:
    RandomEngine(unsigned int seed = std::random_device{}()) : gen(seed) {}

    double sample(const Distribution& dist) {
        switch (dist.type) {
            case DistributionType::GAUSSIAN: {
                std::normal_distribution<double> d(dist.param1, dist.param2);
                return d(gen);
            }
            case DistributionType::UNIFORM: {
                std::uniform_real_distribution<double> d(dist.param1, dist.param2);
                return d(gen);
            }
            case DistributionType::LOGNORMAL: {
                std::lognormal_distribution<double> d(dist.param1, dist.param2);
                return d(gen);
            }
        }
        return 0.0;
    }

    // Latin Hypercube Sampling for N samples
    // Returns a vector of N samples that are stratified
    std::vector<double> sampleLHS(const Distribution& dist, int nSamples) {
        std::vector<double> results(nSamples);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        // 1. Generate stratified CDF probabilities
        std::vector<double> probs(nSamples);
        for (int i = 0; i < nSamples; ++i) {
            double binStart = static_cast<double>(i) / nSamples;
            double binWidth = 1.0 / nSamples;
            double r = uniform(gen); // Random point within bin
            probs[i] = binStart + r * binWidth;
        }

        // 2. Shuffle probabilities to decorrelate dimensions (if used in multi-dim)
        std::shuffle(probs.begin(), probs.end(), gen);

        // 3. Inverse CDF to get values
        for (int i = 0; i < nSamples; ++i) {
            results[i] = inverseCDF(dist, probs[i]);
        }
        
        return results;
    }

private:
    std::mt19937 gen;

    double inverseCDF(const Distribution& dist, double p) {
        // Clamp probability to avoid infinity
        p = std::max(1e-9, std::min(p, 1.0 - 1e-9));

        switch (dist.type) {
            case DistributionType::UNIFORM:
                return dist.param1 + p * (dist.param2 - dist.param1);
            
            case DistributionType::GAUSSIAN:
                return dist.param1 + dist.param2 * normalInverseCDF(p);
                
            case DistributionType::LOGNORMAL:
                // exp(mu + sigma * Z)
                return std::exp(dist.param1 + dist.param2 * normalInverseCDF(p));
        }
        return 0.0;
    }

    // Acklam's algorithm or simple approximation for Probit function
    double normalInverseCDF(double p) {
        // Standard approximation for error function inverse region
        // Using rational approximation for speed
        if (p < 0.5) return -rationalApproximation( std::sqrt(-2.0*std::log(p)) );
        else return rationalApproximation( std::sqrt(-2.0*std::log(1.0-p)) );
    }

    double rationalApproximation(double t) {
        // Constants for rational approximation
        const double c0 = 2.515517;
        const double c1 = 0.802853;
        const double c2 = 0.010328;
        const double d1 = 1.432788;
        const double d2 = 0.189269;
        const double d3 = 0.001308;
        return t - ((c2*t + c1)*t + c0) / (((d3*t + d2)*t + d1)*t + 1.0);
    }
};

struct ParameterOverride {
    std::string componentName;
    std::string paramName;
    double value;
};

class MonteCarloRunner {
public:
    // Generate N sets of parameter overrides using LHS
    static std::vector<std::vector<ParameterOverride>> generateScenarios(
        const std::map<std::string, Distribution>& variations, // Key: "R1.resistance"
        int nSamples,
        SamplingStrategy strategy = SamplingStrategy::LATIN_HYPERCUBE
    ) {
        RandomEngine rng;
        std::vector<std::vector<ParameterOverride>> scenarios(nSamples);

        for (const auto& [key, dist] : variations) {
            // Split "R1.resistance"
            size_t dotPos = key.find('.');
            std::string compName = key.substr(0, dotPos);
            std::string paramName = key.substr(dotPos + 1);

            // Get N samples for this parameter based on strategy
            std::vector<double> samples;
            if (strategy == SamplingStrategy::LATIN_HYPERCUBE) {
                samples = rng.sampleLHS(dist, nSamples);
            } else {
                samples.resize(nSamples);
                for (int i = 0; i < nSamples; ++i) {
                    samples[i] = rng.sample(dist);
                }
            }

            // Distribute into scenarios
            for (int i = 0; i < nSamples; ++i) {
                scenarios[i].push_back({compName, paramName, samples[i]});
            }
        }
        
        return scenarios;
    }
    
    // Helper to apply overrides to a TensorBlock (Deep copy strategy recommended)
    static void applyOverrides(TensorBlock& block, const std::vector<ParameterOverride>& overrides) {
        // This requires TensorBlock to have a look-up mechanism
        // For Phase 1.1 MVP, we will iterate lists nicely.
        
        for (const auto& ov : overrides) {
            // 1. Check Resistors
            for (auto& r : block.resistors) {
                if (r.name == ov.componentName && ov.paramName == "resistance") {
                    r.resistance_ohms = ov.value;
                }
            }
            // 2. Check Capacitors
            for (auto& c : block.capacitors) {
                if (c.name == ov.componentName && ov.paramName == "capacitance") {
                    c.capacitance_farads = ov.value;
                }
            }
            // 3. Check Inductors
             for (auto& l : block.inductors) {
                if (l.name == ov.componentName && ov.paramName == "inductance") {
                     l.inductance_henries = ov.value;
                }
            }
            // Add other components as needed...
        }
    }
};

} // namespace Statistics
