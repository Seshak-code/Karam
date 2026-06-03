#pragma once

#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <deque>
#include "../netlist/circuit.h"

/**
 * lte_controller.h — Local Truncation Error Controller & Method Arbiter (Gap 5)
 *
 * Implements SPICE3-compatible adaptive timestep control based on Local
 * Truncation Error (LTE) estimation for reactive elements (capacitors,
 * inductors).
 *
 * LTE Estimation (3rd-order finite-difference):
 *   For Trapezoidal:  LTE ≈ (h³/12) · d³v/dt³ ≈ (1/12)(v_n - 2·v_{n-1} + v_{n-2})
 *   For Gear2:        LTE ≈ (2h³/9) · d³v/dt³
 *
 * Timestep Control:
 *   dt_new = dt · (tol / maxLTE)^(1/(p+1))
 *   where p = method order (2 for Trap, 2 for Gear2)
 *
 * Method Arbiter:
 *   Detects trapezoidal ringing (sign oscillation in dv/dt for 3+ consecutive
 *   steps) and switches to Gear2 (BDF2) which is A-stable and ring-free.
 *   Returns to Trapezoidal after ringing subsides (4 consecutive non-ringing
 *   steps) for better accuracy on oscillatory circuits.
 *
 * Reference: SPICE3 tran.c: TRTOL, CHGTOL timestep control
 */

// Forward declare IntegrationType from the integrator header
enum class IntegrationType;

struct LTEResult {
    double maxLTE;               // worst-case LTE across all reactive elements
    double suggestedDt;          // LTE-limited timestep suggestion
    int worstElementIndex;       // index of element with largest LTE
    bool worstIsCapacitor;       // true = capacitor, false = inductor
    bool requiresRejection;      // true if LTE > tolerance (timestep too large)
    double safetyFactor;         // actual ratio tol/maxLTE (< 1 means reject)
};

class LTEController {
public:
    explicit LTEController(double tolerance = 1e-4, double trtol = 7.0)
        : tolerance_(tolerance), trtol_(trtol) {}

    /**
     * Estimate the Local Truncation Error for all reactive elements.
     *
     * Uses the 3-step difference formula to approximate the 3rd derivative:
     *   d³v/dt³ ≈ (v_n - 3·v_{n-1} + 3·v_{n-2} - v_{n-3}) / dt³
     * Simplified for 2-step history (available when history depth ≥ 3):
     *   LTE_est ≈ |v_n - 2·v_{n-1} + v_{n-2}| / coeff
     * where coeff depends on the integration method.
     *
     * @param netlist    Circuit netlist with capacitor/inductor state history
     * @param v_curr     Current converged node voltages
     * @param dt         Current timestep
     * @param method     Current integration method
     * @return LTEResult with worst-case LTE and timestep suggestion
     */
    LTEResult estimateLTE(
        const TensorNetlist& netlist,
        const std::vector<double>& v_curr,
        double dt,
        IntegrationType method) const
    {
        LTEResult result{};
        result.maxLTE = 0.0;
        result.suggestedDt = dt;
        result.worstElementIndex = -1;
        result.worstIsCapacitor = true;
        result.requiresRejection = false;
        result.safetyFactor = 1.0;

        if (dt <= 0.0) return result;

        // Method-dependent LTE coefficient:
        //   Trapezoidal: LTE ≈ h²/12 · |d²v/dt²|  → coeff = 12
        //   Gear2:       LTE ≈ 2h²/9 · |d²v/dt²|  → coeff = 4.5
        double coeff = 12.0;
        if (method == IntegrationType::GEAR_2) {
            coeff = 4.5;
        }

        auto getV = [&](int nodeIdx) -> double {
            if (nodeIdx <= 0 || nodeIdx > (int)v_curr.size()) return 0.0;
            return v_curr[nodeIdx - 1];
        };

        // Capacitor LTE estimation
        for (size_t i = 0; i < netlist.globalBlock.capacitors.size(); ++i) {
            const auto& cap = netlist.globalBlock.capacitors[i];
            if (i >= netlist.globalState.capacitorState.size()) continue;
            const auto& hist = netlist.globalState.capacitorState[i];

            if (hist.v.size() < 2) continue;  // need at least 2 history points

            // Current capacitor voltage
            double v_n = getV(cap.nodePlate1) - getV(cap.nodePlate2);
            double v_n1 = hist.v[0];     // previous step
            double v_n2 = (hist.v.size() > 1) ? hist.v[1] : v_n1;

            // 2nd-order difference: approximates dt² · d²v/dt²
            double d2v = v_n - 2.0 * v_n1 + v_n2;

            // LTE = |d2v| / coeff, scaled by charge: Q_lte = C · V_lte
            // SPICE uses charge-based LTE (CHGTOL) for capacitors:
            //   LTE_charge = C · |d2v| / coeff
            // We normalize to voltage for consistent comparison:
            double lte_v = std::abs(d2v) / coeff;

            // TRTOL scaling: SPICE3 uses TRTOL * CHGTOL as the tolerance
            // We use trtol * tolerance as the effective threshold
            double effective_tol = trtol_ * tolerance_;

            if (lte_v > result.maxLTE) {
                result.maxLTE = lte_v;
                result.worstElementIndex = static_cast<int>(i);
                result.worstIsCapacitor = true;
            }
        }

        // Inductor LTE estimation
        for (size_t i = 0; i < netlist.globalBlock.inductors.size(); ++i) {
            const auto& ind = netlist.globalBlock.inductors[i];
            if (i >= netlist.globalState.inductorState.size()) continue;
            const auto& hist = netlist.globalState.inductorState[i];

            if (hist.i.size() < 2) continue;

            // Current inductor current (from voltage across it)
            double v_L = getV(ind.nodeCoil1) - getV(ind.nodeCoil2);
            double i_n = hist.i.empty() ? 0.0 : hist.i[0];
            // Approximate current from companion model:
            // For BE: i = i_prev + (v_L / L) * dt
            double i_curr = i_n + (v_L / std::max(ind.inductance_henries, 1e-15)) * dt;

            double i_n1 = hist.i[0];
            double i_n2 = (hist.i.size() > 1) ? hist.i[1] : i_n1;

            double d2i = i_curr - 2.0 * i_n1 + i_n2;
            double lte_i = std::abs(d2i) / coeff;

            if (lte_i > result.maxLTE) {
                result.maxLTE = lte_i;
                result.worstElementIndex = static_cast<int>(i);
                result.worstIsCapacitor = false;
            }
        }

        // Compute safety factor and timestep suggestion
        double effective_tol = trtol_ * tolerance_;
        if (result.maxLTE > 0.0) {
            result.safetyFactor = effective_tol / result.maxLTE;
            result.requiresRejection = (result.safetyFactor < 1.0);

            // Optimal timestep: dt_new = dt * (tol/LTE)^(1/(p+1))
            // p = order = 2 for both Trap and Gear2, so exponent = 1/3
            // Apply safety margin of 0.8 to avoid re-rejection
            double ratio = std::pow(result.safetyFactor, 1.0 / 3.0);
            ratio = std::min(ratio, 2.0);   // cap growth at 2× per step
            ratio = std::max(ratio, 0.1);   // prevent collapse to zero
            result.suggestedDt = dt * ratio * 0.8;
        }

        return result;
    }

    /**
     * Suggest a new timestep based on LTE estimate.
     * Applies SPICE3-style safety margins and growth limits.
     */
    double suggestTimestep(double currentDt, double maxLTE,
                           double tolerance, IntegrationType method) const
    {
        if (maxLTE <= 0.0) return currentDt * 1.5;  // no LTE info → grow cautiously

        double effective_tol = trtol_ * tolerance;
        double ratio = effective_tol / maxLTE;

        // p+1 = 3 for both Trapezoidal and Gear2 (both are 2nd-order)
        double scale = std::pow(ratio, 1.0 / 3.0);

        // SPICE3-style limits:
        //   - Never grow more than 2× per step
        //   - Never shrink below 0.125× per step
        //   - Apply 0.8 safety factor
        scale = std::clamp(scale * 0.8, 0.125, 2.0);

        return currentDt * scale;
    }

    /**
     * Method Arbiter: detect trapezoidal ringing and select integration method.
     *
     * Trapezoidal integration is 2nd-order accurate but can produce artificial
     * ringing (2h-oscillations) on stiff circuits. Gear2 (BDF2) is also 2nd-order
     * but is L-stable (damps all parasitic oscillations).
     *
     * Detection: If any node exhibits 3+ consecutive sign reversals in dv/dt,
     * switch to Gear2. Return to Trapezoidal after 4 consecutive stable steps.
     *
     * @param v_curr   Current voltages
     * @param current  Current integration method
     * @return Recommended integration method
     */
    IntegrationType selectMethod(
        const std::vector<double>& v_curr,
        IntegrationType current)
    {
        // Update history
        voltageHistory_.push_back(v_curr);
        if (voltageHistory_.size() > 4) {
            voltageHistory_.pop_front();
        }

        if (voltageHistory_.size() < 3) return current;

        size_t N = v_curr.size();
        bool ringingDetected = false;

        // Check each node for sign reversals in dv/dt
        for (size_t n = 0; n < N; ++n) {
            int signChanges = 0;
            for (size_t s = 2; s < voltageHistory_.size(); ++s) {
                if (n >= voltageHistory_[s].size() ||
                    n >= voltageHistory_[s-1].size() ||
                    n >= voltageHistory_[s-2].size()) continue;

                double dv_curr = voltageHistory_[s][n] - voltageHistory_[s-1][n];
                double dv_prev = voltageHistory_[s-1][n] - voltageHistory_[s-2][n];

                // Sign reversal with significant magnitude
                if (dv_curr * dv_prev < 0.0 &&
                    std::abs(dv_curr) > 1e-10 &&
                    std::abs(dv_prev) > 1e-10) {
                    signChanges++;
                }
            }
            if (signChanges >= 2) {
                ringingDetected = true;
                break;
            }
        }

        if (ringingDetected) {
            stableStepCount_ = 0;
            return IntegrationType::GEAR_2;
        }

        // If currently on Gear2, count stable steps before returning to Trap
        if (current == IntegrationType::GEAR_2) {
            stableStepCount_++;
            if (stableStepCount_ >= 4) {
                stableStepCount_ = 0;
                return IntegrationType::TRAPEZOIDAL;
            }
            return IntegrationType::GEAR_2;
        }

        return current;
    }

    /**
     * Reset controller state (call on topology change).
     */
    void reset() {
        voltageHistory_.clear();
        stableStepCount_ = 0;
    }

    // Configuration
    void setTolerance(double tol) { tolerance_ = tol; }
    void setTRTOL(double trtol) { trtol_ = trtol; }
    double getTolerance() const { return tolerance_; }

private:
    double tolerance_;     // absolute LTE tolerance (default 1e-4)
    double trtol_;         // SPICE TRTOL multiplier (default 7.0)

    // Ringing detection state
    std::deque<std::vector<double>> voltageHistory_;  // last 4 voltage snapshots
    int stableStepCount_ = 0;  // consecutive non-ringing steps while on Gear2
};
