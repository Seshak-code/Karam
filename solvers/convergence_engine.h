#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "../infrastructure/simtrace.h"

/**
 * convergence_engine.h
 *
 * Classification engine for Newton-Raphson convergence failure analysis.
 * Analyzes convergence history to determine failure mode and produce
 * human-readable justification reports.
 *
 * Stateless: all methods are static, no internal state.
 */

namespace ConvergenceEngine {

/**
 * classifyFailure
 * Analyzes NR convergence history to determine the failure mode.
 * Priority: SINGULAR_MATRIX > EXPONENTIAL_OVERFLOW > DIVERGENCE > LIMIT_CYCLE > SLOW_CONVERGENCE
 */
inline ConvergenceFailureType classifyFailure(
    const std::vector<ConvergenceStep>& history,
    bool singularMatrixDetected = false)
{
    if (singularMatrixDetected) {
        return ConvergenceFailureType::SINGULAR_MATRIX;
    }

    if (history.empty()) {
        return ConvergenceFailureType::NONE;
    }

    // Check for EXPONENTIAL_OVERFLOW: any voltage delta > 100V in a single step
    for (const auto& step : history) {
        if (step.maxVoltageDelta > 100.0) {
            return ConvergenceFailureType::EXPONENTIAL_OVERFLOW;
        }
    }

    // Analyze the last 5 iterations for DIVERGENCE
    // Residual monotonically increasing over last 5 iterations
    if (history.size() >= 5) {
        size_t start = history.size() - 5;
        bool monotonic_increase = true;
        for (size_t i = start + 1; i < history.size(); ++i) {
            if (history[i].residualNorm <= history[i - 1].residualNorm) {
                monotonic_increase = false;
                break;
            }
        }
        if (monotonic_increase) {
            return ConvergenceFailureType::DIVERGENCE;
        }
    }

    // Check for LIMIT_CYCLE: residual direction changes > 60% of iterations
    if (history.size() >= 4) {
        int direction_changes = 0;
        for (size_t i = 2; i < history.size(); ++i) {
            double prev_delta = history[i - 1].residualNorm - history[i - 2].residualNorm;
            double curr_delta = history[i].residualNorm - history[i - 1].residualNorm;
            if ((prev_delta > 0 && curr_delta < 0) || (prev_delta < 0 && curr_delta > 0)) {
                direction_changes++;
            }
        }
        double change_ratio = static_cast<double>(direction_changes) / static_cast<double>(history.size() - 2);
        if (change_ratio > 0.6) {
            return ConvergenceFailureType::LIMIT_CYCLE;
        }
    }

    // Check for SLOW_CONVERGENCE: < 50% reduction in 10 iterations
    if (history.size() >= 10) {
        double initial = history[history.size() - 10].residualNorm;
        double final_val = history.back().residualNorm;
        if (initial > 0.0 && final_val > 0.5 * initial) {
            return ConvergenceFailureType::SLOW_CONVERGENCE;
        }
    }

    return ConvergenceFailureType::NONE;
}

/**
 * generateJustification
 * Produces a human-readable report describing the convergence failure.
 */
inline std::string generateJustification(
    ConvergenceFailureType failureType,
    const std::vector<ConvergenceStep>& history,
    const std::vector<ResidualContribution>& worstDevices)
{
    std::string report;

    if (failureType == ConvergenceFailureType::NONE) {
        if (!history.empty()) {
            report = "Solver converged in " + std::to_string(history.size()) +
                     " iterations. Final residual: " +
                     std::to_string(history.back().residualNorm) + " A.";
        } else {
            report = "Solver converged successfully.";
        }
        return report;
    }

    report = "CONVERGENCE FAILURE: " + convergenceFailureToString(failureType) + "\n";

    // Add history summary
    if (!history.empty()) {
        report += "  Iterations attempted: " + std::to_string(history.size()) + "\n";
        report += "  Final residual: " + std::to_string(history.back().residualNorm) + " A\n";

        if (history.back().worstNode > 0) {
            report += "  Worst node: " + std::to_string(history.back().worstNode) + "\n";
        }
    }

    // Add worst device attribution
    if (!worstDevices.empty()) {
        report += "  Dominant error source:\n";
        size_t limit = std::min(worstDevices.size(), size_t(3));
        for (size_t i = 0; i < limit; ++i) {
            const auto& d = worstDevices[i];
            report += "    " + d.deviceType + " " + d.deviceName +
                      " at node " + std::to_string(d.nodeIndex) +
                      " contributing " + std::to_string(d.percentOfTotal) +
                      "% of KCL error (" + std::to_string(d.currentContribution) + " A)\n";
        }
    }

    // Add failure-specific guidance
    switch (failureType) {
        case ConvergenceFailureType::SINGULAR_MATRIX:
            report += "  Cause: The circuit Jacobian is rank-deficient. Check for floating nodes, "
                      "voltage source loops, or inductor/capacitor topological issues.\n";
            break;
        case ConvergenceFailureType::DIVERGENCE:
            report += "  Cause: The Newton-Raphson residual is growing monotonically. "
                      "This often indicates missing DC paths, extreme bias conditions, "
                      "or incorrect device parameters.\n";
            break;
        case ConvergenceFailureType::LIMIT_CYCLE:
            report += "  Cause: The solver is oscillating between two or more states. "
                      "This can occur with strong feedback loops or latch-up conditions. "
                      "Try adjusting initial conditions or adding damping.\n";
            break;
        case ConvergenceFailureType::SLOW_CONVERGENCE:
            report += "  Cause: The solver is converging but extremely slowly. "
                      "This may indicate a poorly-conditioned Jacobian or device "
                      "operating near a bifurcation point.\n";
            break;
        case ConvergenceFailureType::EXPONENTIAL_OVERFLOW:
            report += "  Cause: A voltage step exceeded 100V in a single iteration, "
                      "suggesting exponential device model overflow. Check diode/BJT "
                      "bias conditions and consider adding limiting resistors.\n";
            break;
        default:
            break;
    }

    return report;
}

/**
 * findWorstNode
 * Returns the 1-based index of the node with the largest residual.
 */
inline int findWorstNode(const std::vector<double>& residuals) {
    if (residuals.empty()) return 0;
    int worst = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < residuals.size(); ++i) {
        double abs_val = std::abs(residuals[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
            worst = static_cast<int>(i);
        }
    }
    return worst + 1; // 1-based
}

} // namespace ConvergenceEngine
