/**
 * test_convergence_engine.cpp
 *
 * Unit tests for the NR convergence classification engine.
 * Tests synthetic convergence histories against expected classifications.
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include "solvers/convergence_engine.h"

static int tests_passed = 0;
static int tests_failed = 0;

static void check(const std::string& label, bool condition) {
    if (condition) {
        tests_passed++;
        std::cout << "  [PASS] " << label << "\n";
    } else {
        tests_failed++;
        std::cout << "  [FAIL] " << label << "\n";
    }
}

// ===========================================================================
// Test: Successful convergence => NONE
// ===========================================================================
void test_converged() {
    std::cout << "\n[Test: Converged]\n";

    std::vector<ConvergenceStep> history;
    // Residual decreasing monotonically
    for (int i = 0; i < 5; ++i) {
        ConvergenceStep cs;
        cs.iteration = i;
        cs.residualNorm = 1.0 / std::pow(10.0, i + 1);
        cs.maxVoltageDelta = 0.5 / std::pow(2.0, i);
        cs.dampingFactor = 1.0;
        cs.worstNode = 1;
        history.push_back(cs);
    }

    auto result = ConvergenceEngine::classifyFailure(history, false);
    check("Converged circuit classifies as NONE",
          result == ConvergenceFailureType::NONE);
}

// ===========================================================================
// Test: Singular matrix => SINGULAR_MATRIX (priority override)
// ===========================================================================
void test_singular_matrix() {
    std::cout << "\n[Test: Singular Matrix]\n";

    std::vector<ConvergenceStep> history;
    // Even with a "normal" history, singular flag overrides
    for (int i = 0; i < 3; ++i) {
        ConvergenceStep cs;
        cs.iteration = i;
        cs.residualNorm = 1.0;
        cs.maxVoltageDelta = 0.1;
        history.push_back(cs);
    }

    auto result = ConvergenceEngine::classifyFailure(history, true);
    check("Singular matrix flag overrides everything",
          result == ConvergenceFailureType::SINGULAR_MATRIX);
}

// ===========================================================================
// Test: Divergence => monotonically increasing residual over last 5 iters
// ===========================================================================
void test_divergence() {
    std::cout << "\n[Test: Divergence]\n";

    std::vector<ConvergenceStep> history;
    for (int i = 0; i < 8; ++i) {
        ConvergenceStep cs;
        cs.iteration = i;
        // Residual increasing monotonically for last 5
        cs.residualNorm = (i < 3) ? 1.0 : (1.0 + (i - 2) * 0.5);
        cs.maxVoltageDelta = 5.0;
        history.push_back(cs);
    }

    auto result = ConvergenceEngine::classifyFailure(history, false);
    check("Monotonically increasing residual classifies as DIVERGENCE",
          result == ConvergenceFailureType::DIVERGENCE);
}

// ===========================================================================
// Test: Limit cycle => oscillating residual direction
// ===========================================================================
void test_limit_cycle() {
    std::cout << "\n[Test: Limit Cycle]\n";

    std::vector<ConvergenceStep> history;
    // Oscillating: up, down, up, down, up, down, up, down
    for (int i = 0; i < 10; ++i) {
        ConvergenceStep cs;
        cs.iteration = i;
        cs.residualNorm = (i % 2 == 0) ? 1.0 : 0.5;
        cs.maxVoltageDelta = 2.0;
        history.push_back(cs);
    }

    auto result = ConvergenceEngine::classifyFailure(history, false);
    check("Oscillating residual classifies as LIMIT_CYCLE",
          result == ConvergenceFailureType::LIMIT_CYCLE);
}

// ===========================================================================
// Test: Slow convergence => less than 50% reduction in 10 iters
// ===========================================================================
void test_slow_convergence() {
    std::cout << "\n[Test: Slow Convergence]\n";

    std::vector<ConvergenceStep> history;
    for (int i = 0; i < 12; ++i) {
        ConvergenceStep cs;
        cs.iteration = i;
        // Very slow reduction: 1.0 -> 0.95 -> 0.90 -> ... (only 5% per iter but not monotonically increasing)
        cs.residualNorm = 1.0 - i * 0.02; // Goes 1.0 -> 0.78 over 12 iters (22% reduction)
        cs.maxVoltageDelta = 0.1;
        history.push_back(cs);
    }

    auto result = ConvergenceEngine::classifyFailure(history, false);
    check("Slow convergence classifies as SLOW_CONVERGENCE",
          result == ConvergenceFailureType::SLOW_CONVERGENCE);
}

// ===========================================================================
// Test: Exponential overflow => voltage delta > 100V
// ===========================================================================
void test_exponential_overflow() {
    std::cout << "\n[Test: Exponential Overflow]\n";

    std::vector<ConvergenceStep> history;
    for (int i = 0; i < 3; ++i) {
        ConvergenceStep cs;
        cs.iteration = i;
        cs.residualNorm = 1e6;
        cs.maxVoltageDelta = (i == 1) ? 250.0 : 5.0; // Overflow on iter 1
        history.push_back(cs);
    }

    auto result = ConvergenceEngine::classifyFailure(history, false);
    check("Voltage delta > 100V classifies as EXPONENTIAL_OVERFLOW",
          result == ConvergenceFailureType::EXPONENTIAL_OVERFLOW);
}

// ===========================================================================
// Test: Justification generation
// ===========================================================================
void test_justification() {
    std::cout << "\n[Test: Justification Generation]\n";

    std::vector<ConvergenceStep> history;
    for (int i = 0; i < 6; ++i) {
        ConvergenceStep cs;
        cs.iteration = i;
        cs.residualNorm = 1.0 + i * 0.5; // Diverging
        cs.maxVoltageDelta = 10.0;
        cs.worstNode = 3;
        history.push_back(cs);
    }

    std::vector<ResidualContribution> worstDevices;
    ResidualContribution rc;
    rc.deviceName = "Q1";
    rc.deviceType = "BJT";
    rc.nodeIndex = 3;
    rc.currentContribution = 0.5;
    rc.percentOfTotal = 85.0;
    worstDevices.push_back(rc);

    auto type = ConvergenceEngine::classifyFailure(history, false);
    std::string report = ConvergenceEngine::generateJustification(type, history, worstDevices);

    check("Justification contains failure type", report.find("Divergence") != std::string::npos);
    check("Justification contains device name", report.find("Q1") != std::string::npos);
    check("Justification contains device type", report.find("BJT") != std::string::npos);
    check("Justification is non-empty", !report.empty());
}

// ===========================================================================
// Test: findWorstNode
// ===========================================================================
void test_find_worst_node() {
    std::cout << "\n[Test: findWorstNode]\n";

    std::vector<double> residuals = {0.001, -0.05, 0.002, 0.1, -0.003};
    int worst = ConvergenceEngine::findWorstNode(residuals);
    check("Worst node is 4 (1-based) for max abs residual", worst == 4);

    std::vector<double> empty;
    int worst_empty = ConvergenceEngine::findWorstNode(empty);
    check("Empty residuals returns 0", worst_empty == 0);
}

// ===========================================================================
int main() {
    std::cout << "=== Convergence Engine Test Suite ===\n";

    test_converged();
    test_singular_matrix();
    test_divergence();
    test_limit_cycle();
    test_slow_convergence();
    test_exponential_overflow();
    test_justification();
    test_find_worst_node();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    if (tests_failed > 0) {
        std::cerr << "CONVERGENCE ENGINE TEST FAILED\n";
        return 1;
    }

    std::cout << "ALL CONVERGENCE ENGINE TESTS PASSED\n";
    return 0;
}
