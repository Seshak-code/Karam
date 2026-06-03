#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include "../netlist/circuit.h"
#include "../math/linalg.h"

/**
 * pss_solver.h — Periodic Steady-State (PSS) Solver via Shooting Method (Gap 4)
 *
 * Finds the periodic steady-state of autonomous or driven oscillator circuits
 * by iteratively solving for the initial state x₀ such that:
 *   φ(x₀) = x₀   (the state after one period equals the initial state)
 *
 * Algorithm (Newton Shooting):
 *   1. Guess x₀ (typically from a transient warm-up)
 *   2. Simulate one period: x_T = φ(x₀) via stepTransient()
 *   3. Compute mismatch: F(x₀) = φ(x₀) − x₀
 *   4. Compute Jacobian: J = ∂φ/∂x₀ − I (via finite-difference perturbation)
 *   5. Newton update: x₀_new = x₀ − J⁻¹ · F(x₀)
 *   6. Repeat from (2) until ‖F‖ < tolerance
 *
 * The Jacobian is computed column-by-column: perturb x₀[k] by δ, simulate
 * one period, and compute (φ(x₀+δeₖ) − φ(x₀)) / δ.
 *
 * Complexity: O(N²) transient simulations per shooting iteration, where
 * N = number of state variables. For N < 50, this is tractable; for larger
 * circuits, use Harmonic Balance instead.
 *
 * Reference: Kundert, "Steady-State Methods for Simulating Analog and
 *            Microwave Circuits" (Kluwer, 1990), Ch. 3.
 */

// Forward declaration: CircuitSim provides the transient engine
class CircuitSim;

namespace PSSSolver {

struct PSSResult {
    std::vector<double> steadyStateVoltages;  // periodic steady-state at t=0
    double period;                             // confirmed period
    int shootingIterations;                    // outer Newton iterations
    double residual;                           // final ‖φ(x₀) − x₀‖
    bool converged;
    std::string detail;
};

/**
 * Simulate one period of a circuit starting from initial voltages x0.
 * Returns the final state after T seconds.
 *
 * @param netlist   Circuit (mutable for transient state)
 * @param sim       CircuitSim instance
 * @param x0        Initial node voltages
 * @param period    Period T to simulate
 * @param dt        Transient timestep (subdivides the period)
 * @return Final node voltages after one period
 */
inline std::vector<double> simulateOnePeriod(
    TensorNetlist& netlist,
    CircuitSim& sim,
    const std::vector<double>& x0,
    double period,
    double dt)
{
    // Set initial voltages
    // This is done by resetting the sim state and forcing x0
    int N = static_cast<int>(x0.size());
    if (N == 0 || period <= 0.0) return x0;

    // Use a copy of the netlist to avoid corrupting caller's state
    TensorNetlist nl = netlist;

    // Initialize capacitor voltages from x0
    for (size_t i = 0; i < nl.globalBlock.capacitors.size(); ++i) {
        const auto& cap = nl.globalBlock.capacitors[i];
        if (i < nl.globalState.capacitorState.size()) {
            auto& hist = nl.globalState.capacitorState[i];
            if (hist.v.empty()) hist.resize(3);
            double v_cap = 0.0;
            if (cap.nodePlate1 > 0 && cap.nodePlate1 <= N)
                v_cap += x0[cap.nodePlate1 - 1];
            if (cap.nodePlate2 > 0 && cap.nodePlate2 <= N)
                v_cap -= x0[cap.nodePlate2 - 1];
            hist.v[0] = v_cap;
        }
    }

    // Step through one period
    double t = 0.0;
    double step = dt;
    std::vector<double> v_current = x0;

    int maxSteps = static_cast<int>(period / dt) + 10;
    int stepCount = 0;

    while (t < period && stepCount < maxSteps) {
        double remaining = period - t;
        if (remaining < step * 1.01) step = remaining;  // land exactly on T

        SolverStep result = sim.stepTransient(nl, step, t);
        if (!result.stats.converged) {
            // NR failed — halve timestep and retry
            step *= 0.5;
            if (step < 1e-18) break;
            continue;
        }

        t = result.time;
        if (!result.nodeVoltages.empty()) {
            v_current = result.nodeVoltages;
        }
        stepCount++;
    }

    return v_current;
}

/**
 * Solve for the periodic steady-state using the Newton shooting method.
 *
 * @param netlist   Circuit netlist
 * @param sim       CircuitSim instance (must have been initialized with DC solve)
 * @param period    Expected oscillation period T [seconds]
 * @param maxIter   Maximum shooting iterations (default 20)
 * @param tol       Convergence tolerance on ‖φ(x₀)−x₀‖ (default 1e-6)
 * @param warmUpPeriods  Number of warm-up periods before shooting (default 3)
 * @return PSSResult with steady-state voltages and convergence info
 */
inline PSSResult solvePSS(
    TensorNetlist& netlist,
    CircuitSim& sim,
    double period,
    int maxIter = 20,
    double tol = 1e-6,
    int warmUpPeriods = 3)
{
    PSSResult result;
    result.period = period;
    result.converged = false;
    result.shootingIterations = 0;
    result.residual = 1e9;

    int N = netlist.numGlobalNodes;
    if (N == 0 || period <= 0.0) {
        result.detail = "Invalid netlist or period";
        return result;
    }

    // Transient timestep: 100 points per period (matching SPICE defaults)
    double dt = period / 100.0;

    // Phase 1: Warm-up — run several periods to approach the periodic orbit
    std::vector<double> x0(N, 0.0);

    // Get DC operating point as initial guess
    SolverStep dcResult = sim.solveDC(netlist);
    if (dcResult.stats.converged && !dcResult.nodeVoltages.empty()) {
        x0 = dcResult.nodeVoltages;
    }

    std::cout << "[PSS] Warm-up: " << warmUpPeriods << " periods at T="
              << period << "s, dt=" << dt << "s\n";

    for (int w = 0; w < warmUpPeriods; ++w) {
        x0 = simulateOnePeriod(netlist, sim, x0, period, dt);
    }

    // Phase 2: Newton Shooting iterations
    double perturbation = 1e-6;  // finite-difference step for Jacobian

    for (int iter = 0; iter < maxIter; ++iter) {
        result.shootingIterations = iter + 1;

        // Simulate one period from x0
        std::vector<double> phi_x0 = simulateOnePeriod(netlist, sim, x0, period, dt);

        // Compute mismatch F = φ(x₀) − x₀
        std::vector<double> F(N, 0.0);
        double normF = 0.0;
        for (int i = 0; i < N; ++i) {
            F[i] = phi_x0[i] - x0[i];
            normF += F[i] * F[i];
        }
        normF = std::sqrt(normF);
        result.residual = normF;

        std::cout << "[PSS] Iteration " << iter + 1
                  << ": ||F|| = " << normF << "\n";

        if (normF < tol) {
            result.converged = true;
            result.steadyStateVoltages = x0;
            result.detail = "Converged in " + std::to_string(iter + 1) + " shooting iterations";
            std::cout << "[PSS] Converged!\n";
            return result;
        }

        // Compute Jacobian J = ∂φ/∂x₀ − I via finite differences
        // J[i][j] = (φ(x₀ + δ·eⱼ)[i] − φ(x₀)[i]) / δ  − δᵢⱼ
        //
        // For efficiency, only perturb nodes that have significant state
        // (skip ground-tied or constant nodes)
        std::vector<std::vector<double>> J(N, std::vector<double>(N, 0.0));

        for (int j = 0; j < N; ++j) {
            // Skip nodes with near-zero voltage (likely ground-tied)
            double delta = std::max(perturbation, std::abs(x0[j]) * perturbation);

            std::vector<double> x_pert = x0;
            x_pert[j] += delta;

            std::vector<double> phi_pert = simulateOnePeriod(netlist, sim, x_pert, period, dt);

            for (int i = 0; i < N; ++i) {
                J[i][j] = (phi_pert[i] - phi_x0[i]) / delta;
                if (i == j) J[i][j] -= 1.0;  // subtract identity
            }
        }

        // Solve J · dx = -F via LU
        // Negate F for the Newton step
        std::vector<double> negF(N);
        for (int i = 0; i < N; ++i) negF[i] = -F[i];

        // Build dense CSR for solveLU_Pivoted
        Csr_matrix csr;
        csr.rows = N;
        csr.cols = N;
        csr.row_pointer.push_back(0);
        for (int i = 0; i < N; ++i) {
            for (int j2 = 0; j2 < N; ++j2) {
                if (std::abs(J[i][j2]) > 1e-30) {
                    csr.values.push_back(J[i][j2]);
                    csr.col_indices.push_back(j2);
                }
            }
            csr.row_pointer.push_back(static_cast<int>(csr.values.size()));
        }
        csr.nnz = static_cast<int>(csr.values.size());

        SolverResult luResult = solveLU_Pivoted(csr, negF);
        if (!luResult.converged) {
            result.detail = "Jacobian LU solve failed at iteration " +
                           std::to_string(iter + 1);
            return result;
        }

        // Apply Newton update with damping
        double alpha = 1.0;  // full Newton step (could add line search)
        for (int i = 0; i < N; ++i) {
            x0[i] += alpha * luResult.solution[i];
        }
    }

    result.detail = "Did not converge in " + std::to_string(maxIter) + " iterations";
    result.steadyStateVoltages = x0;
    return result;
}

} // namespace PSSSolver
