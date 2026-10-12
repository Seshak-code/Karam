#pragma once
#include "gui_boundary_guard.h"
#include "../netlist/circuit.h"
#include "../math/linalg.h"
#include "../math/contraction_solver.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <functional>
#include <vector>

/*
 * circuitsim.h
 * Circuit Simulator - Physics and Solver Engine.
 */

// ============================================================================
// STAMP FUNCTIONS (Component -> Matrix Entry)
// ============================================================================

#include "../autodiff/dual.h"
#include "device_physics.h"
#include "physics_constants.h"
#include "../tensors/physics_tensors.h"
#include "stamps.h"
#include "../infrastructure/model_registry.h"

// ============================================================================
// CIRCUITSIM CLASS
// ============================================================================

#include "../infrastructure/simtrace.h"
#include "../infrastructure/tensorscheduler.h"
#include "corner_config.h"
#include "../solvers/neural_seeder.h"
#include "../solvers/lte_controller.h"
#include "../solvers/digital_event_engine.h"

// ============================================================================
// CIRCUITSIM CLASS
// ============================================================================

#include "../math/integration_method.h"

/**
 * CircuitSim is the primary engine of the simulator.
 */

// ... (SimTrace/TensorScheduler includes)

// ... (Previous includes) ...

/*
 * Heuristic Arbiter for Adaptive Integration (Hardened).
 * Monitors energy stability, weighted ringing, and LTE.
 */

class IntegratorArbiter {
private:
  std::vector<double> prev_voltages;
  std::vector<double> prev_prev_voltages;
  std::vector<double> node_weights; 
  double prev_energy = 0.0;
  double prev_dt = 1e-9; 
  int instability_counter = 0;
  static constexpr int INSTABILITY_THRESHOLD = 3;

  // ─── TAA (Temporal Anti-Aliasing) History Rejection ──────────────────────
  // Per-node "motion vectors": tracks voltage acceleration (ddV/dt²) to
  // detect violent switching events where polynomial predictors diverge.
  // Analogy: TAA rejects history frames when motion vectors indicate rapid
  // scene changes. Here we reject polynomial extrapolations when ddV/dt²
  // indicates rapid switching, falling back to first-order Euler.
  std::vector<double> prev_dv_dt_;   // First derivative at each node (t-1)
  bool taaInitialized_ = false;
  // Counts accepted steps; LTE predictor needs ≥2 real data points to be valid.
  int stepsAccepted_ = 0;

  // Threshold for acceleration-based rejection. If |ddV/dt²| * dt² exceeds
  // this value (in volts), the polynomial predictor is untrustworthy.
  // Calibrated to reject at ~10mV of quadratic prediction error per step.
  static constexpr double TAA_ACCEL_THRESHOLD = 0.01;  // 10mV

public:
  // ─── TAA: History Rejection API ──────────────────────────────────────────
  // Computes per-node "motion vectors" (voltage acceleration) and rejects
  // polynomial predictor guesses for nodes undergoing violent switching.
  //
  // @param predicted     In/out: polynomial-predicted voltage state
  // @param current_v     Most recent converged voltage state (= prev_voltages)
  // @param dt            Current timestep
  // @return Number of nodes where history was rejected
  int rejectHistory(std::vector<double>& predicted, double dt) const {
      if (!taaInitialized_ || prev_voltages.empty() || prev_prev_voltages.empty() ||
          dt <= 0 || prev_dt <= 0) {
          return 0;
      }

      int rejectionCount = 0;
      const size_t N = std::min(predicted.size(), prev_voltages.size());
      const double dt_sq = dt * dt;

      for (size_t i = 0; i < N; ++i) {
          // Current first derivative estimate: dV/dt_{n-0.5}
          double dv_dt_current = (prev_voltages[i] - prev_prev_voltages[i]) / prev_dt;

          // Acceleration (second derivative): ddV/dt²
          double ddv_dt2 = 0.0;
          if (i < prev_dv_dt_.size()) {
              ddv_dt2 = (dv_dt_current - prev_dv_dt_[i]) / (0.5 * (dt + prev_dt));
          }

          // Quadratic prediction error estimate: |ddV/dt²| * dt²
          double accel_error = std::abs(ddv_dt2) * dt_sq;

          if (accel_error > TAA_ACCEL_THRESHOLD) {
              // Reject polynomial predictor → fall back to first-order Euler
              // predicted[i] = V[n-1] + dV/dt * dt  (linear extrapolation only)
              predicted[i] = prev_voltages[i] + dv_dt_current * dt;
              ++rejectionCount;
          }
      }
      return rejectionCount;
  }


  void initialize(const TensorNetlist &netlist) {
    int n = netlist.numGlobalNodes;
    prev_voltages.assign(n, 0.0);
    prev_prev_voltages.assign(n, 0.0);
    node_weights.assign(n, 1e-12);
    prev_dv_dt_.assign(n, 0.0);
    taaInitialized_ = false;
    stepsAccepted_ = 0;
    prev_energy = 0.0;
    prev_dt = 1e-9;
    instability_counter = 0;

    auto addWeight = [&](int node, double w) {
      if (node > 0 && node <= n)
        node_weights[node - 1] += std::abs(w);
    };

    for (const auto &r : netlist.globalBlock.resistors) {
      double g = 1.0 / r.resistance_ohms;
      addWeight(r.nodeTerminal1, g);
      addWeight(r.nodeTerminal2, g);
    }
    for (const auto &c : netlist.globalBlock.capacitors) {
      addWeight(c.nodePlate1, c.capacitance_farads);
      addWeight(c.nodePlate2, c.capacitance_farads);
    }
    
    // Add hierarchical weights
    for (const auto &inst : netlist.instances) {
        auto it = netlist.blockDefinitions.find(inst.blockName);
        if (it != netlist.blockDefinitions.end()) {
            const auto &block = it->second;
            auto mapNode = [&](int local) { return local == 0 ? 0 : inst.nodeMapping[local-1]; };
            for (const auto &c : block.capacitors) {
                addWeight(mapNode(c.nodePlate1), c.capacitance_farads);
                addWeight(mapNode(c.nodePlate2), c.capacitance_farads);
            }
        }
    }
  }

  const std::vector<double> &getPrevVoltages() const { return prev_voltages; }

  // Research-Grade LTE: Combined Node & Device Error
  double calculateLTE(const TensorNetlist &netlist, const std::vector<double>& current_voltages, double dt) {
      if (prev_voltages.empty() || prev_prev_voltages.empty() || dt <= 0 || prev_dt <= 0) return 0.0;
      // The second-order predictor 2*v(t) - v(t-h) requires at least 2 accepted
      // steps so that both prev_voltages and prev_prev_voltages contain real
      // simulation data.  Before that, the predictor is 0 (initialization) which
      // makes the LTE appear enormous for any non-trivial circuit — causing
      // unnecessary step-halving and premature GEAR_2 switches on cold starts.
      if (stepsAccepted_ < 2) return 0.0;
      
      double c_factor = dt / (dt + prev_dt);
      double sum_sq_error = 0.0;
      double max_err = 0.0;
      int count = 0;

      auto getNorm = [](double v) { return 1e-6 + std::abs(v) * 1e-3; };

      // 1. Nodal Voltages (State Variables)
      for (size_t i = 0; i < current_voltages.size(); ++i) {
          double v_pred = prev_voltages[i] + (prev_voltages[i] - prev_prev_voltages[i]) * (dt / prev_dt);
          double err = std::abs(current_voltages[i] - v_pred) * c_factor;
          double w_err = err / getNorm(current_voltages[i]);
          
          sum_sq_error += w_err * w_err;
          max_err = std::max(max_err, w_err);
          count++;
      }

      // 2. Capacitor Branch Voltages (Internal State), global + instances
      auto getVc = [&](int n) { return (n > 0 && n <= (int)current_voltages.size()) ? current_voltages[n-1] : 0.0; };
      auto capErr = [&](const TensorBlock &blk, const BlockState &st, auto mapNode) {
        for (size_t i = 0; i < blk.capacitors.size() && i < st.capacitorState.size(); ++i) {
          const auto &c = blk.capacitors[i];
          const auto &hist = st.capacitorState[i];
          if (hist.v.size() < 2) continue;
          double v_branch = getVc(mapNode(c.nodePlate1)) - getVc(mapNode(c.nodePlate2));
          double v_pred = hist.v[0] + (hist.v[0] - hist.v[1]) * (dt / prev_dt);
          double w_err = std::abs(v_branch - v_pred) * c_factor / getNorm(v_branch);
          sum_sq_error += w_err * w_err;
          max_err = std::max(max_err, w_err);
          count++;
        }
      };
      capErr(netlist.globalBlock, netlist.globalState, [](int n) { return n; });
      for (size_t k = 0; k < netlist.instances.size() && k < netlist.instanceStates.size(); ++k) {
        const auto &inst = netlist.instances[k];
        auto it = netlist.blockDefinitions.find(inst.blockName);
        if (it == netlist.blockDefinitions.end()) continue;
        capErr(it->second, netlist.instanceStates[k], [&](int n) {
          return (n <= 0 || n > (int)inst.nodeMapping.size()) ? 0 : inst.nodeMapping[n - 1];
        });
      }


      // 3. MOSFET Gate Drive (Vgs) - Critical for Switching
      for (const auto &m : netlist.globalBlock.mosfets) {
           auto getV = [&](int n, const std::vector<double>& src) { return (n>0 && n<=(int)src.size()) ? src[n-1] : 0.0; };
           
           double vgs = getV(m.gate, current_voltages) - getV(m.source, current_voltages);
           double vgs_p = getV(m.gate, prev_voltages) - getV(m.source, prev_voltages);
           double vgs_pp = getV(m.gate, prev_prev_voltages) - getV(m.source, prev_prev_voltages);
           
           double vgs_pred = vgs_p + (vgs_p - vgs_pp) * (dt / prev_dt);
           double err = std::abs(vgs - vgs_pred) * c_factor;
           double w_err = err / getNorm(vgs);
           
           sum_sq_error += w_err * w_err;
           max_err = std::max(max_err, w_err);
           count++;
      }

      if (count == 0) return 0.0;

      // Return Max-Norm (Conservative) effectively, but could switch to RMS
      // Standard SPICE uses RMS, but Max-Norm guarantees no single node deviates.
      // We return Max-Norm as it safely bounds RMS.
      return max_err;
  }

  // M4: Formal Passivity Checks (Static)
  bool checkPassivity(const TensorNetlist &netlist) {
      // 1. Check Global Block
      if (!checkBlockPassivity(netlist.globalBlock)) return false;
      // 2. Check Instances
      for (const auto &inst : netlist.instances) {
          auto it = netlist.blockDefinitions.find(inst.blockName);
          if (it != netlist.blockDefinitions.end()) {
              if (!checkBlockPassivity(it->second)) return false;
          }
      }
      return true;
  }

  bool checkBlockPassivity(const TensorBlock &block) {
      for (const auto &r : block.resistors) {
          if (r.resistance_ohms <= 0.0) return false;
      }
      for (const auto &c : block.capacitors) {
          if (c.capacitance_farads < 0.0) return false;
      }
      for (const auto &m : block.mosfets) {
          if (m.w <= 0.0 || m.l <= 0.0) return false;
      }
      return true;
  }

  // Returns: 0=Stable, 1=Reduce dt, 2=Switch Method
  int analyzeStability(const TensorNetlist &netlist,
                       const std::vector<double> &current_voltages,
                       double dt,
                       double &next_dt_suggestion) {
    if (current_voltages.empty()) return 0;
    if (current_voltages.size() != prev_voltages.size()) {
       initialize(netlist);
       next_dt_suggestion = dt; 
       return 0;
    }

    // 1. Check LTE
    double lte = calculateLTE(netlist, current_voltages, dt);
    // The predictor-corrector formula used in calculateLTE is the FIRST-ORDER
    // linear-extrapolation mismatch (≈ h²·v''/2), which overestimates the
    // actual TRAPEZOIDAL truncation error (≈ h³·v'''/12) by a factor of
    // ~12·τ/h.  For an RC circuit with dt/τ=0.22 this ratio is ~27×.
    // Setting the threshold at 30.0 means: the step is accepted when the
    // predictor mismatch is < 30×getNorm(v), which corresponds to roughly
    // 0.6% relative voltage error — consistent with standard SPICE LTE targets.
    // Real instabilities (NR oscillations, diverging active circuits) produce
    // LTE values well above 100 and are still caught by this guard.
    bool high_lte = (lte > 35.0); // Threshold

    // 2. Check Energy
    double current_energy = 0.0;
    for (const auto &c : netlist.globalBlock.capacitors) {
        double v = 0.0;
        if (c.nodePlate1 > 0 && c.nodePlate1 <= (int)current_voltages.size()) v += current_voltages[c.nodePlate1 - 1];
        if (c.nodePlate2 > 0 && c.nodePlate2 <= (int)current_voltages.size()) v -= current_voltages[c.nodePlate2 - 1];
        current_energy += 0.5 * c.capacitance_farads * v * v;
    }
    
    // Add instance energy
    for (const auto &inst : netlist.instances) {
        auto it = netlist.blockDefinitions.find(inst.blockName);
        if (it != netlist.blockDefinitions.end()) {
            const auto &block = it->second;
            auto mapNode = [&](int local) { return local == 0 ? 0 : inst.nodeMapping[local-1]; };
            auto getV = [&](int n) { return (n > 0 && n <= (int)current_voltages.size()) ? current_voltages[n-1] : 0.0; };
            for (const auto &c : block.capacitors) {
                 double v = getV(mapNode(c.nodePlate1)) - getV(mapNode(c.nodePlate2));
                 current_energy += 0.5 * c.capacitance_farads * v * v;
            }
        }
    }



    // M4: Lyapunov Energy Monitor
    // Calculate Work Done by Sources (Approximate Pin)
    double power_in = 0.0;
    // Current Sources
    for (const auto &i : netlist.globalBlock.currentSources) {
         double v_diff = 0.0;
         if (i.nodePositive > 0 && i.nodePositive <= (int)current_voltages.size()) v_diff += current_voltages[i.nodePositive-1];
         if (i.nodeNegative > 0 && i.nodeNegative <= (int)current_voltages.size()) v_diff -= current_voltages[i.nodeNegative-1];
         power_in += std::abs(i.current_A * v_diff);
    }
    // Voltage Sources: estimate power as |V| * sum of cap-branch currents at the
    // VS node (using the previous step's stored hist.i).  This prevents the
    // Lyapunov energy check from flagging normal RC charging as instability —
    // the cap current from hist reflects the actual energy the source delivers.
    for (const auto &vs : netlist.globalBlock.voltageSources) {
        int nPos = vs.nodePositive;
        double V = std::abs(vs.voltage_V);
        for (size_t ci = 0; ci < netlist.globalBlock.capacitors.size(); ++ci) {
            const auto &c = netlist.globalBlock.capacitors[ci];
            if (c.nodePlate1 != nPos && c.nodePlate2 != nPos) continue;
            if (ci < netlist.globalState.capacitorState.size() &&
                !netlist.globalState.capacitorState[ci].i.empty()) {
                power_in += V * std::abs(netlist.globalState.capacitorState[ci].i[0]);
            }
        }
    }
    // Safety Margin for Voltage Sources/Norton Equivalents (Heuristic)
    // We allow energy to grow if there is significant voltage activity, scaling with system size
    double energy_tolerance = 1e-9 + power_in * dt * 1.5; 
    
    // Strict Lyapunov Bound: E(t) <= E(t-1) + Pin*dt + tolerance
    // We use a relaxed bound here to account for numeric noise and voltage source estimation gaps
    // Guard against cold-start false positive: when prev_energy == 0 (arbiter
    // just initialized), the "* 1.5" condition is trivially true and causes a
    // spurious instability on every first step.
    bool energy_exploding = (prev_energy > 1e-12) &&
                            (current_energy > prev_energy + energy_tolerance + 1e-6) &&
                            (current_energy > prev_energy * 1.5);
    
    if (high_lte || energy_exploding) {
        instability_counter++;
        next_dt_suggestion = dt * 0.5; // Cut step
        if (instability_counter >= INSTABILITY_THRESHOLD) return 2; // Switch Method
        return 1; // Retry
    } else {
        instability_counter = std::max(0, instability_counter - 1);

        // ─── TAA: Update motion vectors (dV/dt) for next step ──────────
        // These are used by rejectHistory() to compute acceleration and
        // detect violent switching events where polynomial predictors fail.
        if (!prev_voltages.empty() && prev_dt > 0) {
            const size_t N = current_voltages.size();
            prev_dv_dt_.resize(N, 0.0);
            for (size_t i = 0; i < N; ++i) {
                prev_dv_dt_[i] = (current_voltages[i] - prev_voltages[i]) / dt;
            }
            taaInitialized_ = true;
        }

        prev_prev_voltages = prev_voltages;
        prev_voltages = current_voltages;
        prev_energy = current_energy;
        prev_dt = dt;
        stepsAccepted_++;
        
        if (lte < 0.1) next_dt_suggestion = dt * 1.5; 
        else if (lte < 0.5) next_dt_suggestion = dt * 1.1; 
        else next_dt_suggestion = dt; 
        
        if (next_dt_suggestion > 1e-3) next_dt_suggestion = 1e-3; 
        return 0;
    }
  }
};

class CircuitSim {
public:
  enum class ExecutionMode {
      LEGACY_AOS = 0,
      TENSOR_SOA_SCALAR = 1,
      TENSOR_SOA_SIMD = 2,
      GPU = 3,
      GPU_DEBUG = 4,
      HYBRID = 5    // CPU+GPU parallel: GPU handles TN contraction while CPU stamps next iteration
  };

  IntegratorArbiter arbiter;
  bool arbiterInitialized = false;
  std::unique_ptr<IIntegrationMethod> integrator;
  ExecutionMode execMode = ExecutionMode::LEGACY_AOS;

  // CSR pattern cache (Sparse Merge — Phase D5): rebuilt when a stamp falls
  // outside it (the topology hash ignores sources and node numbering)
  std::unique_ptr<CachedCsrPattern> cachedPattern_;

  // Default linear solver: tensor-contraction plan, rebuilt only when the
  // Jacobian's sparsity pattern (circuit topology) changes.
  std::unique_ptr<ContractionPlan> contractionPlan_;
  SolverResult solveLinear(const Csr_matrix& A, const std::vector<double>& b) {
    if (!contractionPlan_ || !contractionPlan_->matches(A))
      contractionPlan_ = std::make_unique<ContractionPlan>(planContraction(A));
    return solveContraction(A, b, *contractionPlan_);
  }

  // Converged solution of the prior solve (seeder history, dV/dt)
  std::vector<double> prevVoltages_;

  // Gap 4: Neural convergence seeder — physics-informed dV/dt extrapolation.
  // Records each converged solution and uses history for next initial guess.
  NeuralSeeder neuralSeeder_;

  // Gap 5: LTE-based adaptive timestep controller (truncation error, step size).
  LTEController lteController_;
  double lteTolerance_ = 1e-4;  // absolute LTE tolerance

  // Gap 2: Mixed-signal digital event engine.
  // Handles A2D/D2A boundary conditions and zero-delay digital propagation.
  DigitalEventEngine digitalEngine_;

  // Callbacks for orchestration
  std::function<void(int, double)> onConvergenceStep;
  std::function<void(float)>       onProgress;

  // Phase 2.10: Expose cached pattern for AC solver fast-path

  // Model card library pointer (set before stamping, for per-model parameter lookup)
  const std::map<std::string, ModelCard>* currentModelCards = nullptr;

  CircuitSim() {
    integrator = std::make_unique<TrapezoidalIntegration>();
  }

  /* Implicitly deleted move ops due to atomic members - safe to omit or implement deep copy if needed */
  /* CircuitSim(CircuitSim &&) = delete; */
  /* CircuitSim &operator=(CircuitSim &&) = delete; */

  void setIntegrationMethod(IntegrationType type) {
    if(!integrator || integrator->getType() != type) {
        switch (type) {
        case IntegrationType::TRAPEZOIDAL:
          integrator = std::make_unique<TrapezoidalIntegration>();
          break;
        case IntegrationType::GEAR_1_EULER:
          integrator = std::make_unique<Gear1Integration>();
          break;
        case IntegrationType::GEAR_2:
          integrator = std::make_unique<Gear2Integration>();
          break;
        }
    }
  }
  

  /*
   * Stamping Helpers
   */
  // ===========================================================================
  // COMPONENT STAMPERS (Reusability Layer)
  // ===========================================================================
  // These helpers encapsulate the physics-to-matrix mapping.
  // They are used by:
  // 1. stampBlockNonLinear (AoS / Teaching Path)
  // 2. stampSoABlock (SoA / Fast Path) [Partial coverage]
  // 3. calculateSensitivity (Adjoint Method) [Future]

  void stampResistor(int n1, int n2, double R, MatrixConstructor& mb) {
      if (std::abs(R) < 1e-15) R = 1e-15; // Safety
      double g = 1.0 / R;
      if (n1 != 0 && n2 != 0) {
          mb.add(n1-1, n1-1, g);
          mb.add(n2-1, n2-1, g);
          mb.add(n1-1, n2-1, -g);
          mb.add(n2-1, n1-1, -g);
      } else if (n1 != 0) {
          mb.add(n1-1, n1-1, g);
      } else if (n2 != 0) {
          mb.add(n2-1, n2-1, g);
      }
  }

  // Helper for Zener/Diode generic stamping
  void stampTwoTerminalNonLinear(int n1, int n2, double g_eq, double i_eq, 
                                 MatrixConstructor& mb, std::vector<double>& rhs) {
      // Safety floor for M4 numerical stability
      if (g_eq < 1e-15) g_eq = 1e-15;

      // Stamp G (J = dI/dV)
      if (n1 != 0 && n2 != 0) {
          mb.add(n1-1, n1-1, g_eq);
          mb.add(n2-1, n2-1, g_eq);
          mb.add(n1-1, n2-1, -g_eq);
          mb.add(n2-1, n1-1, -g_eq);
      } else if (n1 != 0) {
          mb.add(n1-1, n1-1, g_eq);
      } else if (n2 != 0) {
          mb.add(n2-1, n2-1, g_eq);
      }

      // Stamp RHS (Current Residual)
      if (n1 != 0) rhs[n1-1] -= i_eq;
      if (n2 != 0) rhs[n2-1] += i_eq;
  }

  void stampDiode(const Diode& d, int nA, int nC, double v_diff, double temp_K,
                  MatrixConstructor& mb, std::vector<double>& rhs) {
      // 1. Setup Parameters
      DiodeParams<double> params = {d.saturationCurrent_I_S_A, d.emissionCoefficient_N,
                                    d.thermalVoltage_V_T_V};
      
      // Thermal Scaling
      double temp_scale = temp_K / 300.15;
      params.Vt *= temp_scale;

      // 2. Evaluate Physics (AD)
      Dual<double> v_diff_dual(v_diff, 1.0);
      Dual<double> i_d = diode_current_clamped(v_diff_dual, params);

      // 3. Extract Linear Model
      double g_eq = i_d.grad;
      double i_src = i_d.val - g_eq * v_diff;

      // 4. Stamp
      stampTwoTerminalNonLinear(nA, nC, g_eq, i_src, mb, rhs);
  }

  void stampSchottky(const SchottkyDiode& sd, int nA, int nC, double v_diff, double temp_K,
                     MatrixConstructor& mb, std::vector<double>& rhs) {
      // Identical structure to Diode but different params/handling potentially
      DiodeParams<double> params = {sd.saturationCurrent_I_S_A, sd.emissionCoefficient_N,
                                    sd.thermalVoltage_V_T_V};
      params.Vt *= (temp_K / 300.15);

      Dual<double> v_diff_dual(v_diff, 1.0);
      Dual<double> i_d = diode_current_clamped(v_diff_dual, params);

      double g_eq = i_d.grad;
      double i_src = i_d.val - g_eq * v_diff;

      stampTwoTerminalNonLinear(nA, nC, g_eq, i_src, mb, rhs);
  }

  void stampZener(const ZenerDiode& z, int nA, int nC, double v_diff, double temp_K,
                  MatrixConstructor& mb, std::vector<double>& rhs) {
      ZenerParams<double> zp;
      zp.BV = z.breakdownVoltage_V;
      zp.IBV = z.currentAtBreakdown_A;
      zp.Rs = z.seriesResistance_Rs_ohms;
      zp.N = z.emissionCoefficient_N;
      zp.Is = z.saturationCurrent_I_S_A;
      zp.Vt = z.thermalVoltage_V_T_V * (temp_K / 300.15);

      Dual<double> v_d(v_diff, 1.0);
      Dual<double> result = zener_current(v_d, zp);

      double g_eq = result.grad;
      double i_src = result.val - g_eq * v_diff;

      stampTwoTerminalNonLinear(nA, nC, g_eq, i_src, mb, rhs);
  }

  void stampMosfet(const Mosfet& m, int nD, int nG, int nS, int nB,
                   double v_d, double v_g, double v_s, double v_b,
                   MatrixConstructor& mb, std::vector<double>& rhs) {
      double vgs = v_g - v_s;
      double vds = v_d - v_s;

      MosfetParams<double> p;

      // Look up model card by name (falls back to defaults)
      const ModelCard* card = currentModelCards ?
          currentModelCards->count(m.modelName) ? &currentModelCards->at(m.modelName) : nullptr
          : nullptr;

      bool isPMOS = false;
      if (card) {
          isPMOS = (card->type == "PMOS");
          p.Kp     = card->get("KP", isPMOS ? 100e-6 : 200e-6);
          p.Vth    = card->get("VTO", isPMOS ? -0.7 : 0.7);
          p.lambda = card->get("LAMBDA", 0.02);
      } else {
          isPMOS = (m.modelName.find("P") != std::string::npos ||
                    m.modelName.find("p") != std::string::npos);
          p.Kp = isPMOS ? 100e-6 : 200e-6;
          p.Vth = isPMOS ? -0.7 : 0.7;
          p.lambda = 0.02;
      }
      p.W = m.w;
      p.L = m.l;
      
      // Calculate GM (dId/dVgs)
      Dual<double> ids_vgs = mosfet_ids(Dual<double>(vgs, 1.0), Dual<double>(vds, 0.0), p);
      double gm = ids_vgs.grad;
      
      // Calculate GDS (dId/dVds)
      Dual<double> ids_vds = mosfet_ids(Dual<double>(vgs, 0.0), Dual<double>(vds, 1.0), p);
      double gds = ids_vds.grad;
      
      double ids = ids_vgs.val; 
      double gmb = 0.0; // Body effect placeholder
      double i_eq = ids - (gm * vgs + gmb * 0.0 + gds * vds);

      stampMosfetMatrix(nD, nG, nS, nB, gm, gmb, gds, i_eq, mb, rhs);
  }

  void stampMosfetMatrix(int d, int g, int s, int b,
                         double gm, double gmb, double gds, double i_eq,
                         MatrixConstructor& mb, std::vector<double>& rhs) {
      // Linearized Model:
      // I_ds = gm*Vgs + gmb*Vbs + gds*Vds + I_eq
      
      // Gds (Drain-Source Conductance)
      if (d!=0 && s!=0) {
           mb.add(d-1, d-1,  gds);
           mb.add(s-1, s-1,  gds);
           mb.add(d-1, s-1, -gds);
           mb.add(s-1, d-1, -gds);
      } else if (d!=0) mb.add(d-1, d-1, gds);
      else if (s!=0)   mb.add(s-1, s-1, gds);

      // Gm (Transconductance)
      if (d!=0) {
           if (g!=0) mb.add(d-1, g-1,  gm);
           if (s!=0) mb.add(d-1, s-1, -gm);
      }
      if (s!=0) {
           if (g!=0) mb.add(s-1, g-1, -gm);
           if (s!=0) mb.add(s-1, s-1,  gm);
      }

      // RHS
      if (d!=0) rhs[d-1] -= i_eq;
      if (s!=0) rhs[s-1] += i_eq;
  }

  void stampBJT(const BJT& bjt, int nC, int nB, int nE, int nS,
                double v_c, double v_b, double v_e, double v_s,
                double temp_K,
                MatrixConstructor& mb, std::vector<double>& rhs) {
      // Use efficient analytical model (No dual overhead if not needed)
      BJTParams<double> p;
      p.Is = bjt.saturationCurrent_I_S_A;
      p.BetaF = bjt.betaF;
      p.BetaR = bjt.betaR;
      p.Vt = bjt.thermalVoltage_V_T_V;

      // Thermal scaling (consistent with stampDiode/stampZener)
      double Tnom = 300.15; // 27°C nominal
      double temp_scale = temp_K / Tnom;
      p.Vt *= temp_scale;

      // Is temperature scaling: Is(T) = Is(Tnom) * (T/Tnom)^Xti * exp(Eg/Vt_nom - Eg/Vt_T)
      double Xti = 3.0;  // Silicon default
      double Eg = 1.11;  // Silicon bandgap [eV]
      double Vt_nom = bjt.thermalVoltage_V_T_V;
      double Vt_T = p.Vt;
      p.Is *= std::pow(temp_scale, Xti) * exp(Eg / Vt_nom - Eg / Vt_T);

      bool isNPN = (bjt.modelName.find("NPN") != std::string::npos || bjt.modelName.find("npn") != std::string::npos);
      
      BJTCurrents<double> res = bjt_ebers_moll(v_c, v_b, v_e, isNPN, p);

      // Stamp G (7x7 block for 3 terminals usually, simplified here)
      // Helper to add conductance
      auto addG = [&](int r, int c, double val) {
          if (r!=0 && c!=0) mb.add(r-1, c-1, val);
      };

      addG(nC, nC, res.g_cc); addG(nC, nB, res.g_cb); addG(nC, nE, res.g_ce);
      addG(nB, nC, res.g_bc); addG(nB, nB, res.g_bb); addG(nB, nE, res.g_be);
      addG(nE, nC, res.g_ec); addG(nE, nB, res.g_eb); addG(nE, nE, res.g_ee);

      // Stamp I_rhs
      // I_node = -Sum(G*V) + I_eq
      // I_eq = I_actual(V) - G*V
      // RHS needs -I_eq (to move I_eq to LHS? No wait).
      // KCL: Sum(G*V) = Sum(I_sources).
      // Linearized: I(V) ~ I(Vo) + G*(V-Vo) = I(Vo) - G*Vo + G*V.
      // So I_source_equiv = I(Vo) - G*Vo.
      // KCL: ... + I(V) = 0 -> ... + G*V + I_source_equiv = 0 -> ... + G*V = -I_source_equiv.
      // RHS = -I_source_equiv.
      // RHS = -(I(Vo) - G*Vo) = G*Vo - I(Vo).
      
      // Let's compute I_source_equiv for each terminal
      double i_c_eq = res.Ic - (res.g_cc*v_c + res.g_cb*v_b + res.g_ce*v_e);
      double i_b_eq = res.Ib - (res.g_bc*v_c + res.g_bb*v_b + res.g_be*v_e);
      double i_e_eq = res.Ie - (res.g_ec*v_c + res.g_eb*v_b + res.g_ee*v_e);

      if (nC != 0) rhs[nC-1] -= i_c_eq;
      if (nB != 0) rhs[nB-1] -= i_b_eq;
      if (nE != 0) rhs[nE-1] -= i_e_eq;
  }

  void stampJFET(const JFET& j, int nD, int nG, int nS,
                 double v_d, double v_g, double v_s,
                 MatrixConstructor& mb, std::vector<double>& rhs) {
      double vgs = v_g - v_s;
      double vds = v_d - v_s;

      JFETParams<double> p;
      p.Beta = j.beta;
      p.Vto = j.Vto;
      p.Lambda = j.lambda;

      // P-channel: I = s*f(s*vgs, s*vds) with s = -1 (same form as the KCL
      // audit). Chain rule: dI/dvgs = s*f'*s = f', so gm/gds take NO sign; only
      // the current flips. (Previously ids was unsigned and gm/gds were signed,
      // so P-channel JFETs were linearised around the wrong current.)
      double sign = j.isNChannel ? 1.0 : -1.0;

      Dual<double> gm_ad = jfet_ids(Dual<double>(vgs * sign, 1.0), Dual<double>(vds * sign, 0.0), p);
      Dual<double> gds_ad = jfet_ids(Dual<double>(vgs * sign, 0.0), Dual<double>(vds * sign, 1.0), p);
      double ids = sign * gm_ad.val;
      double gm  = gm_ad.grad;
      double gds = gds_ad.grad;

      // Linearized current source
      double i_eq = ids - (gm * vgs + gds * vds);

      // Stamp JFET
      // Ids = gm*Vgs + gds*Vds + I_eq

      // Drain node (nD)
      if (nD != 0) {
          if (nG != 0) mb.add(nD-1, nG-1, gm);
          if (nS != 0) mb.add(nD-1, nS-1, -gm - gds);
          if (nD != 0) mb.add(nD-1, nD-1, gds);
          rhs[nD-1] -= i_eq;
      }

      // Source node (nS)
      // KCL at S: -Ids = Is
      // Is = -Ids = -(gm*Vgs + gds*Vds + I_eq)
      if (nS != 0) {
          if (nG != 0) mb.add(nS-1, nG-1, -gm);
          if (nS != 0) mb.add(nS-1, nS-1, gm + gds);
          if (nD != 0) mb.add(nS-1, nD-1, -gds);
          rhs[nS-1] += i_eq;
      }
  }

  SolverStep solveDC(TensorNetlist &netlist);
  SolverStep stepTransient(TensorNetlist &netlist, double timeStep, double currentTime);

  // The one transient loop (session, GUI driver, Fourier). t = 0 state: the DC
  // operating point with .ic nodes forced, or with `uic` the .ic / IC= values.
  // Steps to `stop` at `step`; a failed step retries at h/4 (down to step*1e-9)
  // and regrows. `onPoint` gets each accepted step and returns false to cancel.
  // Returns an empty string on success, else the error.
  std::string runTransient(TensorNetlist &netlist, double step, double stop, bool uic,
                           const std::function<bool(const SolverStep &)> &onPoint);

  // AC small-signal sweep around the operating point of the last solveDC().
  // Y(jw) = G + jwC: G is the Newton Jacobian at the OP (the DC stamps, so every
  // device, model card, subcircuit and MNA branch row), C the capacitors plus the
  // inductor/mutual L terms of the branch rows; excitation = AC mag/phase of the
  // V and I sources. Solved as the real 2N system [[G, -wC], [wC, G]] by the
  // contraction solver (one plan for the sweep). Returns x(f): nodes, then branches;
  // empty if solveDC() has not converged on this netlist.
  std::vector<std::vector<std::complex<double>>> solveAC(TensorNetlist &netlist,
                                                         const std::vector<double> &freqs);

  // Noise around the last solveDC() operating point (ngspice .noise semantics).
  // Generators: resistor thermal 4kT/R, diode shot 2q|Id|, BJT shot 2q|Ic| (C-E)
  // and 2q|Ib| (B-E), MOSFET/JFET channel thermal 4kT*(2/3)*gm (MOS1). Per
  // frequency one adjoint solve gives every generator's transfer to
  // v(out)-v(ref) and one forward solve the gain from `inputSource` at unit AC;
  // totals integrate each generator with ngspice's power law (Nintegrate).
  // Densities in V/sqrt(Hz), totals in V rms; `error` is set on failure.
  struct NoiseResult {
    std::vector<double> onoise, inoise;
    double onoiseTotal = 0.0, inoiseTotal = 0.0;
    std::string error;
  };
  NoiseResult solveNoise(TensorNetlist &netlist, const std::vector<double> &freqs,
                         int outNode, int refNode, const std::string &inputSource);

  // .ac frequencies as ngspice steps them: dec/oct multiply by 10^(1/n) / 2^(1/n)
  // up to fstop (+1e-3 of a step, ngspice's reltol slack); lin gives n points;
  // log (the GUI's sweep) gives n log-spaced points from f1 to f2.
  static std::vector<double> acFrequencies(const std::string &sweep, int n, double f1, double f2) {
    std::vector<double> f;
    if (n < 1 || !(f1 > 0.0) || f2 < f1) return f;
    if (sweep == "lin" || sweep == "log") {
      for (int k = 0; k < n; ++k) {
        const double t = n == 1 ? 0.0 : double(k) / (n - 1);
        f.push_back(sweep == "lin" ? f1 + t * (f2 - f1) : f1 * std::pow(f2 / f1, t));
      }
      return f;
    }
    const double delta = std::pow(sweep == "oct" ? 2.0 : 10.0, 1.0 / n);
    for (int k = 0; f1 * std::pow(delta, k) <= f2 * (1.0 + 1e-3 * delta); ++k)
      f.push_back(f1 * std::pow(delta, k));
    return f;
  }
  

  // Phase 1.8: PVT Application
  void applyCorner(TensorNetlist &netlist, const acutesim::physics::CornerConfig& corner) {
      netlist.environment.process = (SimulationEnvironment::Corner)corner.process;
      netlist.environment.ambient_temp_K = corner.temperatureC + 273.15;
      
      // If voltage scale is 0 or negative, ignore (safety)
      if (corner.voltageScale > 0.1) {
          netlist.environment.global_voltage_scale = corner.voltageScale;
      }
      
      // Monte Carlo Seed (Phase 1.8.2)
      netlist.environment.monte_carlo_seed = corner.monteCarloSeed;
  }
  
  // Deterministic RNG (FNV-1a Hash + Box-Muller)
  static uint64_t fnv1a(const std::string& s, uint64_t seed) {
      uint64_t hash = 14695981039346656037ULL;
      hash ^= seed;
      for (char c : s) {
          hash ^= static_cast<uint64_t>(c);
          hash *= 1099511628211ULL;
      }
      return hash;
  }
  
  // Returns standard normal sample N(0, 1)
  double getGaussian(uint64_t hash) {
      // Split 64-bit hash into two 32-bit uniform variates
      uint32_t h1 = (uint32_t)(hash & 0xFFFFFFFF);
      uint32_t h2 = (uint32_t)(hash >> 32);
      
      // Avoid 0 (log(0) is bad)
      if (h1 == 0) h1 = 1; 
      
      double u1 = h1 / 4294967296.0;
      double u2 = h2 / 4294967296.0;
      
      // Box-Muller Transform
      return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265359 * u2);
  }
  
  // Get MC-perturbed value
  double getPerturbedValue(double nominal, double tolerance_pct, const std::string& name, int seed) {
      if (tolerance_pct <= 1e-6 || seed == 0) return nominal;
      
      // Hash combining seed + name
      uint64_t h = fnv1a(name, static_cast<uint64_t>(seed));
      double z = getGaussian(h);
      
      // 3-sigma rule: tolerance is usually 3-sigma (or we treat it as 1-sigma?)
      // Standard industry practice: Tolerance is usually 3-sigma limit.
      // So sigma = tolerance / 3.
      double sigma_pct = tolerance_pct / 3.0;
      
      double factor = 1.0 + (z * sigma_pct * 0.01);
      return nominal * factor;
  }

private:
  MatrixConstructor matrixBuilder;
  std::vector<double> rhsVector;
  TensorScheduler scheduler;
  
  // SoA Tensor Storage (Batch Physics)
  TensorizedBlock soaBlock;

  // =========================================================================
  // PHASE 1 HARDENING: Helper Methods (Decomposition of solveDC)
  // =========================================================================

  /**
   * seedInitialGuess - Initialize voltage estimates for Newton-Raphson.
   * Encapsulates heuristic seeding for Diodes, BJTs, and supply nodes.
   */
  void seedInitialGuess(const TensorNetlist& netlist, std::vector<double>& v);

  /**
   * applyPnJunctionLimits - Apply SPICE-style pnjlim to prevent NR overshoot.
   * Modifies v_new in-place based on v_old.
   */
  void applyPnJunctionLimits(const TensorizedBlock& block,
                              std::vector<double>& v_new,
                              const std::vector<double>& v_old,
                              double temp_K = 300.15);

  /**
   * coldStartSweep - SPICE3-accurate cold-start using dt_init = tstep/100 micro-steps.
   * Replaces the single Backward Euler first step with nSubsteps internal BE steps,
   * each of duration totalTime/nSubsteps, accumulating history so that the final
   * capacitor state matches what SPICE3 produces at t = tstep.
   *
   * @param netlist    Circuit netlist (AoS history is updated in-place per sub-step)
   * @param totalTime  Full first timestep duration (the tstep value)
   * @param nSubsteps  Number of internal micro-steps (typically COLD_START_SUBSTEPS=100)
   * @param finalVoltages  Output: converged voltages after the last micro-step
   * @return true if all micro-steps converged; false if any micro-step NR failed
   */
  bool coldStartSweep(TensorNetlist& netlist, double totalTime, int nSubsteps,
                      std::vector<double>& finalVoltages);

  /** Tensorize `netlist` into soaBlock and sync reactive-element history
   *  (global + instances) from the AoS state. */
  void loadSoAFromNetlist(const TensorNetlist& netlist);

  /** Capacitor + inductor history after an accepted step (global + instances),
   *  using the same companion formulas as the stamps and the KCL audit. */
  void updateReactiveHistory(TensorNetlist& netlist, const std::vector<double>& v, double h);
  void seedReactiveHistory(TensorNetlist& netlist, const std::vector<double>& x, bool uic);

  /** Last full solution x (node voltages + branch currents). */
  std::vector<double> lastSolution_;
  /** Newton starting point for a transient step: the last full solution when
   *  the topology matches, else the arbiter's node voltages, zero-padded. */
  std::vector<double> transientSeed(const TensorNetlist& netlist, int nTot) {
    if ((int)lastSolution_.size() == nTot) return lastSolution_;
    std::vector<double> x = arbiter.getPrevVoltages();
    x.resize(nTot, 0.0);
    return x;
  }

  // NR Parameters (Using centralized PhysicsConstants)
  // Note: Kept as local constexpr for backward compatibility; prefer PhysicsConstants::
  static constexpr double RELTOL = 1e-3;
  static constexpr double VNTOL = 1e-6;
  static constexpr double ABSTOL = PhysicsConstants::ABSTOL; 
  static constexpr int MAX_NR_ITER = PhysicsConstants::MAX_NR_ITER;
  static constexpr double PCG_TOL = 1e-10;
  static constexpr int PCG_MAX_ITER = 1000;

  /**
   * checkConvergence
   * Verifies BOTH voltage stability (approximate) AND physical charge
   * conservation (exact).
   */
  bool checkConvergence(const std::vector<double> &v_new,
                        const std::vector<double> &v_old,
                        double max_kcl_error,
                        double kcl_tol = 1e-6) {
    // Every test is written as "pass only if x <= tol" so that NaN (for which
    // all comparisons are false) fails instead of slipping through.
    if (v_new.size() != v_old.size()) return false;
    for (size_t i = 0; i < v_new.size(); ++i)
      if (!std::isfinite(v_new[i])) return false;

    // 1. Check Physical KCL Conservation (Primary Truth)
    if (max_kcl_error < 1e-9) return true;

    // 2. Check Voltage Updates (Standard RELTOL/VNTOL)
    for (size_t i = 0; i < v_new.size(); ++i) {
      double diff = std::abs(v_new[i] - v_old[i]);
      double limit = RELTOL * std::max(std::abs(v_new[i]), std::abs(v_old[i])) + VNTOL;
      if (!(diff <= limit)) return false;
    }

    // 3. Check Physical KCL Conservation (Research-Grade)
    // Relax ABSTOL slightly for Norton-stamped voltage sources (which are
    // stiff)
    return max_kcl_error <= kcl_tol;
  }

public:
  // ── MNA branch currents ────────────────────────────────────────────────
  // Unknowns x = [node voltages (N) | branch currents (B)]. V sources,
  // inductors, E (VCVS) and H (CCVS) each carry a branch current flowing from
  // their + node through the element to their - node; F (CCCS) and H read the
  // branch current of their controlling V source. Branch order: all V
  // sources, then inductors, then E, then H, each over the global block then
  // every instance (tensorizeNetlist order), so indices are fixed per topology.
  // These replace the 1 mOhm / 100 uOhm Norton approximations: sources and
  // shorts are now exact (an SoC power grid IS milliohms of metal).
  struct BranchSlot {
    const TensorBlock *blk;
    const BlockState *st;
    std::function<int(int)> map;   // local node -> global node
    int vs, ind, e, h;             // first matrix row of each branch kind
  };
  std::vector<BranchSlot> branchSlots(const TensorNetlist &nl, int &total) const {
    std::vector<BranchSlot> branchList;
    branchList.push_back({&nl.globalBlock, &nl.globalState, [](int x) { return x; }, 0, 0, 0, 0});
    for (size_t k = 0; k < nl.instances.size() && k < nl.instanceStates.size(); ++k) {
      auto it = nl.blockDefinitions.find(nl.instances[k].blockName);
      if (it == nl.blockDefinitions.end()) continue;
      const auto *inst = &nl.instances[k];
      branchList.push_back({&it->second, &nl.instanceStates[k], [inst](int x) {
        return (x <= 0 || x > (int)inst->nodeMapping.size()) ? 0 : inst->nodeMapping[x - 1];
      }, 0, 0, 0, 0});
    }
    int next = nl.numGlobalNodes;
    for (auto &s : branchList) { s.vs = next; next += (int)s.blk->voltageSources.size(); }
    for (auto &s : branchList) { s.ind = next; next += (int)s.blk->inductors.size(); }
    for (auto &s : branchList) { s.e = next; next += (int)s.blk->voltageControlledVoltageSources.size(); }
    for (auto &s : branchList) { s.h = next; next += (int)s.blk->currentControlledVoltageSources.size(); }
    total = next;
    return branchList;
  }
  int systemSize(const TensorNetlist &nl) const { int t = 0; branchSlots(nl, t); return t; }

  // Kernel-form device parameters shared by the KCL audit and noise analysis.
  // MOSFET: sign = -1 for PMOS; kernels take the NMOS-equivalent threshold
  // sign*VTO (PMOS -0.45 -> +0.45). BJT: thermal scaling as stampBJT.
  static MosfetParams<double> mosfetKernelParams(const TensorNetlist &nl, const Mosfet &m, double &sign) {
    const MosfetModel mm = mosfetModel(m, &nl.modelCards);
    sign = mm.pmos ? -1.0 : 1.0;
    MosfetParams<double> p;
    p.Kp = mm.kp; p.Vth = sign * mm.vth; p.lambda = mm.lambda; p.W = m.w; p.L = m.l;
    return p;
  }
  static BJTParams<double> bjtKernelParams(const TensorNetlist &nl, const BJT &q) {
    BJTParams<double> p = {q.saturationCurrent_I_S_A, q.betaF, q.betaR, q.thermalVoltage_V_T_V};
    const double ts = nl.environment.ambient_temp_K / 300.15;
    p.Vt *= ts;
    p.Is *= std::pow(ts, 3.0) * std::exp(1.11 / q.thermalVoltage_V_T_V - 1.11 / p.Vt);
    return p;
  }

  // Small-signal system at the last DC operating point: G (Newton Jacobian
  // without node gmin), reactive triplets C. Shared by solveAC and solveNoise.
  struct ACSystem { Csr_matrix G; std::vector<int> cr, cc; std::vector<double> cv; int n = 0; };
  bool acSystem(TensorNetlist &netlist, ACSystem &sys);
  // R(Y) = [[G, -wC], [wC, G]] for Y = G + jwC (transpose: R(Y)^T = R(Y^H)).
  static Csr_matrix acMatrix(const ACSystem &s, double w, bool transpose);
  // Excitation: the sources' AC mag/phase, or unit AC at source `unit` only
  // (empty vector if no source has that name).
  std::vector<std::complex<double>> acExcitation(const TensorNetlist &nl, const std::string &unit = "") const;

  // flux[a] = {(b, L_ab)}: self inductance plus K couplings M = k*sqrt(La*Lb).
  // Shared by the transient branch rows and the AC reactive matrix.
  static std::vector<std::vector<std::pair<int, double>>> fluxLinks(const TensorBlock &blk) {
    const auto &L = blk.inductors;
    std::vector<std::vector<std::pair<int, double>>> flux(L.size());
    for (size_t a = 0; a < L.size(); ++a) flux[a].push_back({(int)a, L[a].inductance_henries});
    for (const auto &k : blk.mutualInductors) {
      int a = k.inductor1_index, c = k.inductor2_index;
      if (a < 0 || c < 0 || a >= (int)L.size() || c >= (int)L.size() || a == c) continue;
      double M = k.couplingCoefficient_k * std::sqrt(L[a].inductance_henries * L[c].inductance_henries);
      flux[a].push_back({c, M});
      flux[c].push_back({a, M});
    }
    return flux;
  }

  // All branch-element matrix entries + rhs. Stamping adds them to the matrix;
  // the KCL audit evaluates rhs - A x from the SAME triplets, so the two can
  // never disagree.
  struct BranchSystem { std::vector<int> r, c; std::vector<double> v, rhs; };
  void buildBranchSystem(const TensorNetlist &nl, double h, double time, BranchSystem &S) const {
    int total = 0;
    auto branchList = branchSlots(nl, total);
    S.r.clear(); S.c.clear(); S.v.clear(); S.rhs.assign(total, 0.0);
    auto add = [&](int r, int c, double v) {
      if (r >= 0 && c >= 0) { S.r.push_back(r); S.c.push_back(c); S.v.push_back(v); }
    };
    const bool tran = h > 1e-15 && integrator;
    const IntegrationType type = tran ? integrator->getType() : IntegrationType::GEAR_1_EULER;
    // Inductor companion: v = coef * sum_j L_aj * i_j + history (per method)
    const double coef = !tran ? 0.0 : type == IntegrationType::TRAPEZOIDAL ? 2.0 / h
                       : type == IntegrationType::GEAR_2 ? 1.5 / h : 1.0 / h;
    const double scale = nl.environment.global_voltage_scale;
    for (const auto &sl : branchList) {
      auto node = [&](int local) { return sl.map(local) - 1; };  // ground -> -1 (skipped)
      auto branch = [&](int b, int np, int nn) {  // KCL coupling + v(+) - v(-) in row b
        add(np, b, 1.0); add(nn, b, -1.0); add(b, np, 1.0); add(b, nn, -1.0);
      };
      const auto &blk = *sl.blk;
      for (size_t i = 0; i < blk.voltageSources.size(); ++i) {
        const auto &v = blk.voltageSources[i];
        const int b = sl.vs + (int)i;
        branch(b, node(v.nodePositive), node(v.nodeNegative));
        S.rhs[b] += scale * (tran ? waveformValue(v, v.voltage_V, time) : v.voltage_V);
      }
      // Inductors (+ mutual coupling K: off-diagonal L_ab terms between branch rows)
      const auto &L = blk.inductors;
      const auto flux = fluxLinks(blk);
      for (size_t a = 0; a < L.size(); ++a) {
        const int b = sl.ind + (int)a;
        branch(b, node(L[a].nodeCoil1), node(L[a].nodeCoil2));  // DC: v1 - v2 = 0 (short)
        if (!tran) continue;
        for (auto [j, Laj] : flux[a]) {
          add(b, sl.ind + j, -coef * Laj);
          const auto &hj = sl.st->inductorState[j];
          double i1 = hj.i.empty() ? 0.0 : hj.i[0];
          double i2 = hj.i.size() > 1 ? hj.i[1] : i1;
          S.rhs[b] += (type == IntegrationType::GEAR_2) ? (Laj / h) * (-2.0 * i1 + 0.5 * i2)
                                                        : -coef * Laj * i1;
        }
        if (type == IntegrationType::TRAPEZOIDAL) {
          const auto &ha = sl.st->inductorState[a];
          S.rhs[b] -= ha.v.empty() ? 0.0 : ha.v[0];
        }
      }
      for (size_t k = 0; k < blk.voltageControlledVoltageSources.size(); ++k) {
        const auto &e = blk.voltageControlledVoltageSources[k];
        const int b = sl.e + (int)k;
        branch(b, node(e.outPos), node(e.outNeg));
        add(b, node(e.ctrlPos), -e.gain);
        add(b, node(e.ctrlNeg), e.gain);
      }
      for (size_t k = 0; k < blk.currentControlledVoltageSources.size(); ++k) {
        const auto &hs = blk.currentControlledVoltageSources[k];
        const int b = sl.h + (int)k;
        branch(b, node(hs.outPos), node(hs.outNeg));
        if (hs.ctrlSource >= 0 && hs.ctrlSource < (int)blk.voltageSources.size())
          add(b, sl.vs + hs.ctrlSource, -hs.rm);
      }
      for (const auto &f : blk.currentControlledCurrentSources) {
        if (f.ctrlSource < 0 || f.ctrlSource >= (int)blk.voltageSources.size()) continue;
        const int bc = sl.vs + f.ctrlSource;  // alpha * I(ctrl) leaves out+ through the source
        add(node(f.outPos), bc, f.alpha);
        add(node(f.outNeg), bc, -f.alpha);
      }
    }
  }
  void stampBranchSystem(const TensorNetlist &nl, double h, double time) {
    BranchSystem S;
    buildBranchSystem(nl, h, time, S);
    for (size_t t = 0; t < S.r.size(); ++t) matrixBuilder.add(S.r[t], S.c[t], S.v[t]);
    for (size_t i = 0; i < S.rhs.size() && i < rhsVector.size(); ++i) rhsVector[i] += S.rhs[i];
  }

  /**
   * calculatePhysicalResiduals
   * Audits the solution by checking Kirchhoff's Current Law using EXACT
   * non-linear equations. Returns the maximum KCL violation (in Amperes) across
   * all nodes.
   */
  double calculatePhysicalResiduals(const TensorNetlist &netlist,
                                    const std::vector<double> &v_candidate,
                                    double h = 0.0, double time = 0.0,
                                    std::vector<double> *perNodeOut = nullptr) {
    int n = netlist.numGlobalNodes;
    if (n == 0)
      return 0.0;

    // Rows: node KCL (amps) then branch equations (volts)
    std::vector<double> node_residuals(systemSize(netlist), 0.0);

    // helper to get node voltage (Safe Access)
    auto getV = [&](int nodeIdx) {
      if (nodeIdx <= 0 || nodeIdx > (int)v_candidate.size())
        return 0.0;
      return v_candidate[nodeIdx - 1];
    };

    // Helper to accumulate residual (Safe Access)
    auto addRes = [&](int nodeIdx, double val) {
      if (nodeIdx > 0 && nodeIdx <= n)
        node_residuals[nodeIdx - 1] += val;
    };

    // 1. Resistors (Ohm's Law: I = V/R)
    for (const auto &r : netlist.globalBlock.resistors) {
      double v1 = getV(r.nodeTerminal1);
      double v2 = getV(r.nodeTerminal2);
      double i_r = (v1 - v2) / r.resistance_ohms;
      addRes(r.nodeTerminal1, -i_r); // Current LEAVING node
      addRes(r.nodeTerminal2, i_r);  // Current ENTERING node
    }

    // 2. V sources, inductors, E/F/H: branch-current elements, audited from
    //    the same triplets they are stamped with (see buildBranchSystem below)

    // 2.5. Current Sources (Ideal Current Injection)
    // Current sources inject fixed current regardless of voltage
    for (const auto &cs : netlist.globalBlock.currentSources) {
      double is = (h > 1e-15) ? waveformValue(cs, cs.current_A, time) : cs.current_A;
      // Current flows from + to - inside source, leaves + externally
      addRes(cs.nodePositive, -is); // Current leaving + node
      addRes(cs.nodeNegative, is);  // Current entering - node
    }

    // 2.7 Dependent Sources (Global): G only; E/F/H are branch elements
    for (const auto &vccs : netlist.globalBlock.voltageControlledCurrentSources) {
        double v_ctrl = getV(vccs.ctrlPos) - getV(vccs.ctrlNeg);
        double i_vccs = vccs.gm * v_ctrl;
        addRes(vccs.outPos, -i_vccs);
        addRes(vccs.outNeg, i_vccs);
    }

    // 3. Diodes (Shockley Equation: I = Is * (exp(V/Vt) - 1))
    for (const auto &d : netlist.globalBlock.diodes) {
      double v_anode = getV(d.anode);
      double v_cathode = getV(d.cathode);
      double v_d = v_anode - v_cathode;

      // Use CLAMPED physics kernel (matching solver) to avoid false divergence
      // on valid linear continuation steps.
      DiodeParams<double> params = {d.saturationCurrent_I_S_A, d.emissionCoefficient_N,
                            d.thermalVoltage_V_T_V};
      double i_d = diode_current_clamped(v_d, params); // device_physics.h default cap

      addRes(d.anode, -i_d);
      addRes(d.cathode, i_d);
    }

    // 3.25 Schottky Diode Audit
    for (const auto &sd : netlist.globalBlock.schottkyDiodes) {
      double v_a = getV(sd.anode);
      double v_c = getV(sd.cathode);
      double v_d = v_a - v_c;
      
      DiodeParams<double> params = {sd.saturationCurrent_I_S_A, sd.emissionCoefficient_N,
                            sd.thermalVoltage_V_T_V};
      double i_d = diode_current_clamped(v_d, params);
      
      addRes(sd.anode, -i_d);
      addRes(sd.cathode, i_d);
    }

    // 3.5 BJT Audit (temperature-aware)
    for (const auto &q : netlist.globalBlock.bjts) {
        double v_c = getV(q.nodeCollector);
        double v_b = getV(q.base);
        double v_e = getV(q.emitter);

        BJTCurrents currents = bjt_ebers_moll(v_c, v_b, v_e, q.isNPN, bjtKernelParams(netlist, q));
        
        addRes(q.nodeCollector, -currents.Ic);
        addRes(q.base, -currents.Ib);
        addRes(q.emitter, -currents.Ie);
    }

    // 3.55 MOSFET Audit (uses .model card parameters when available — must
    // match the solver's stamping exactly, otherwise false KCL residuals)
    for (const auto &m : netlist.globalBlock.mosfets) {
        double v_d = getV(m.drain);
        double v_g = getV(m.gate);
        double v_s = getV(m.source);
        
        double sign;
        const MosfetParams<double> p = mosfetKernelParams(netlist, m, sign);
        double ids = sign * mosfet_ids(sign * (v_g - v_s), sign * (v_d - v_s), p);
        
        addRes(m.drain, -ids);
        addRes(m.source, ids);
    }
    
    // 3.7 Hierarchical Instances — audited in Section 5 (parallel scheduler path)
    
    // 3.8 JFET Audit (Shichman-Hodges)
    for (const auto &jfet_inst : netlist.globalBlock.jfets) {
        double v_d = getV(jfet_inst.drain);
        double v_g = getV(jfet_inst.gate);
        double v_s = getV(jfet_inst.source);
        double vgs = v_g - v_s;
        double vds = v_d - v_s;
        
        JFETParams<double> params;
        params.Beta = jfet_inst.beta;
        params.Vto = jfet_inst.Vto;
        params.Lambda = jfet_inst.lambda;
        
        // For P-channel, negate voltages and result
        double sign = jfet_inst.isNChannel ? 1.0 : -1.0;
        double ids = sign * jfet_ids(sign * vgs, sign * vds, params);
        
        // KCL: Current leaves drain, enters source
        addRes(jfet_inst.drain, -ids);
        addRes(jfet_inst.source, ids);
    }
    
    // 3.7 Zener Audit
    for (const auto &z : netlist.globalBlock.zenerDiodes) {
        double v_a = getV(z.anode);
        double v_c = getV(z.cathode);
        double vd = v_a - v_c;
        
        ZenerParams<double> params;
        params.BV = z.breakdownVoltage_V;
        params.IBV = z.currentAtBreakdown_A;
        params.Rs = z.seriesResistance_Rs_ohms;
        params.N = z.emissionCoefficient_N;
        params.Is = z.saturationCurrent_I_S_A;
        params.Vt = z.thermalVoltage_V_T_V;
        
        double iz = zener_current(vd, params);
        
        // KCL: current leaves anode, enters cathode
        addRes(z.anode, -iz);
        addRes(z.cathode, iz);
    }

    // 4. Capacitors (Transient Only)
    // If h > 0, we are in transient and must audit the displacement current
    if (h > 1e-15 && integrator) {
      for (size_t i = 0; i < netlist.globalBlock.capacitors.size(); ++i) {
        const auto &c = netlist.globalBlock.capacitors[i];
        const auto &hist = netlist.globalState.capacitorState[i];
          
        double v_curr = getV(c.nodePlate1) - getV(c.nodePlate2);
        double i_c = 0.0;
        IntegrationType type = integrator->getType();

        if (hist.v.empty())
          continue;

        if (type == IntegrationType::TRAPEZOIDAL) {
          double g_eq = (2.0 * c.capacitance_farads) / h;
          i_c = g_eq * (v_curr - hist.v[0]) - hist.i[0];
        } else if (type == IntegrationType::GEAR_1_EULER) {
          i_c = (c.capacitance_farads / h) * (v_curr - hist.v[0]);
        } else if (type == IntegrationType::GEAR_2) {
          if (hist.v.size() >= 2) {
            i_c = (c.capacitance_farads / h) *
                  (1.5 * v_curr - 2.0 * hist.v[0] +
                   0.5 * hist.v[1]);
          } else {
            i_c = (c.capacitance_farads / h) * (v_curr - hist.v[0]);
          }
        }
        addRes(c.nodePlate1, -i_c);
        addRes(c.nodePlate2, i_c);
      }
      
    }

    // 5. Hierarchical Instances
    // Sequential accumulation required for bit-identical determinism.
    // Parallel atomic updates are non-deterministic due to non-associative float addition.
    for (int i = 0; i < (int)scheduler.jobs.size(); ++i) {
      const auto &job = scheduler.jobs[i];
      const TensorBlock &block = *job.blockTmpl;
      for (int instanceIdx : job.instanceIds) {
        const auto &instance = netlist.instances[instanceIdx];
        auto mapNode = [&](int localNode) {
          return (localNode == 0) ? 0 : instance.nodeMapping[localNode - 1];
        };

        // Resistors
        // Resistors
        for (const auto &r : block.resistors) {
          double i_r = (getV(mapNode(r.nodeTerminal1)) -
                        getV(mapNode(r.nodeTerminal2))) /
                       r.resistance_ohms;
          addRes(mapNode(r.nodeTerminal1), -i_r);
          addRes(mapNode(r.nodeTerminal2), i_r);
        }

        // Current sources inside the subcircuit (V sources are branch elements)
        for (const auto &cs : block.currentSources) {
          double is = (h > 1e-15) ? waveformValue(cs, cs.current_A, time) : cs.current_A;
          addRes(mapNode(cs.nodePositive), -is);
          addRes(mapNode(cs.nodeNegative), is);
        }

        // Diodes
        for (const auto &d : block.diodes) {
          double v_d = getV(mapNode(d.anode)) - getV(mapNode(d.cathode));
          DiodeParams<double> params = {d.saturationCurrent_I_S_A,
                                 d.emissionCoefficient_N,
                                 d.thermalVoltage_V_T_V};
          double i_d = diode_current_clamped(v_d, params);  // as stamped
          addRes(mapNode(d.anode), -i_d);
          addRes(mapNode(d.cathode), i_d);
        }

        // Dependent Sources: G only (E/F/H are branch elements)
        for (const auto &vccs : block.voltageControlledCurrentSources) {
            double v_ctrl = getV(mapNode(vccs.ctrlPos)) - getV(mapNode(vccs.ctrlNeg));
            double i_vccs = vccs.gm * v_ctrl;
            addRes(mapNode(vccs.outPos), -i_vccs);
            addRes(mapNode(vccs.outNeg), i_vccs);
        }

        // MOSFETs (Instance)
        for (const auto &m : block.mosfets) {
            double v_d = getV(mapNode(m.drain));
            double v_g = getV(mapNode(m.gate));
            double v_s = getV(mapNode(m.source));
            
            double sign;
            const MosfetParams<double> p = mosfetKernelParams(netlist, m, sign);
            double ids = sign * mosfet_ids(sign * (v_g - v_s), sign * (v_d - v_s), p);
            
            addRes(mapNode(m.drain), -ids);
            addRes(mapNode(m.source), ids);
        }

        // BJTs (Instance)
        for (const auto &q : block.bjts) {
            double v_c = getV(mapNode(q.nodeCollector));
            double v_b = getV(mapNode(q.base));
            double v_e = getV(mapNode(q.emitter));

            BJTCurrents currents = bjt_ebers_moll(v_c, v_b, v_e, q.isNPN, bjtKernelParams(netlist, q));

            addRes(mapNode(q.nodeCollector), -currents.Ic);
            addRes(mapNode(q.base), -currents.Ib);
            addRes(mapNode(q.emitter), -currents.Ie);
        }

        // JFETs (Instance) — same form as the global audit
        for (const auto &j : block.jfets) {
            JFETParams<double> params;
            params.Beta = j.beta; params.Vto = j.Vto; params.Lambda = j.lambda;
            double sign = j.isNChannel ? 1.0 : -1.0;
            double vgs = getV(mapNode(j.gate)) - getV(mapNode(j.source));
            double vds = getV(mapNode(j.drain)) - getV(mapNode(j.source));
            double ids = sign * jfet_ids(sign * vgs, sign * vds, params);
            addRes(mapNode(j.drain), -ids);
            addRes(mapNode(j.source), ids);
        }

        // Capacitors (Transient)
        if (h > 1e-15 && integrator) {
          const auto &instState = netlist.instanceStates[instanceIdx];
          for (size_t c_idx = 0; c_idx < block.capacitors.size(); ++c_idx) {
            const auto &c = block.capacitors[c_idx];
            const auto &hist = instState.capacitorState[c_idx];

            double v_curr =
                getV(mapNode(c.nodePlate1)) - getV(mapNode(c.nodePlate2));
            double i_c = 0.0;
            IntegrationType type = integrator->getType();
            if (hist.v.empty())
              continue;

            if (type == IntegrationType::TRAPEZOIDAL) {
              double g_eq = (2.0 * c.capacitance_farads) / h;
              i_c = g_eq * (v_curr - hist.v[0]) - hist.i[0];
            } else if (type == IntegrationType::GEAR_1_EULER) {
              i_c = (c.capacitance_farads / h) * (v_curr - hist.v[0]);
            } else if (type == IntegrationType::GEAR_2) {
              if (hist.v.size() >= 2) {
                i_c = (c.capacitance_farads / h) *
                      (1.5 * v_curr - 2.0 * hist.v[0] +
                       0.5 * hist.v[1]);
              } else {
                i_c = (c.capacitance_farads / h) * (v_curr - hist.v[0]);
              }
            }
            addRes(mapNode(c.nodePlate1), -i_c);
            addRes(mapNode(c.nodePlate2), i_c);
          }
        }
      }
    }

    // Branch elements: residual = rhs - A x over exactly the stamped triplets
    {
      BranchSystem S;
      buildBranchSystem(netlist, h, time, S);
      auto xAt = [&](int c) { return c < (int)v_candidate.size() ? v_candidate[c] : 0.0; };
      for (size_t t = 0; t < S.r.size(); ++t) node_residuals[S.r[t]] -= S.v[t] * xAt(S.c[t]);
      for (size_t i = 0; i < S.rhs.size(); ++i) node_residuals[i] += S.rhs[i];
    }

    // 6. Find Max Residual
    double max_err = 0.0;
    for (double val : node_residuals) {
      max_err = std::max(max_err, std::abs(val));
    }
    if (perNodeOut) *perNodeOut = std::move(node_residuals);
    return max_err;
  }

  /**
   * kclExcess — the NR convergence metric: KCL error beyond each node's
   * floating-point noise floor, max_i max(0, |r_i| - KCL_RELTOL*(|A||v|)_i),
   * where A is the Jacobian just solved. Raw max|r_i| made correct solutions
   * with stiff stamps (ideal op-amps) report "not converged".
   */
  double kclExcess(const TensorNetlist &netlist, const std::vector<double> &v,
                   const Csr_matrix &A, double h = 0.0, double time = 0.0) {
    std::vector<double> r;
    calculatePhysicalResiduals(netlist, v, h, time, &r);
    double worst = 0.0;
    for (int i = 0; i < (int)r.size(); ++i) {
      if (!std::isfinite(r[i])) return INFINITY;  // NaN must fail, not vanish in max()
      double scale = 0.0;
      if (i < A.rows)
        for (int k = A.row_pointer[i]; k < A.row_pointer[i + 1]; ++k)
          scale += std::abs(A.values[k]) * std::abs(v[A.col_indices[k]]);
      worst = std::max(worst, std::abs(r[i]) - PhysicsConstants::KCL_RELTOL * scale);
    }
    return worst;
  }

  /**
   * calculatePhysicalResiduals (per-node overload) — Phase 2.9
   * Like the scalar overload but also populates perNodeResiduals with the
   * raw KCL violation at each node. Used for RMS/worst-node computation.
   * Calling this on every NR iteration is expensive — invoke only at final step.
   */
  double calculatePhysicalResiduals(const TensorNetlist &netlist,
                                    const std::vector<double> &v_candidate,
                                    std::vector<double> &perNodeResiduals,
                                    double h = 0.0) {
    // Same audit as the scalar overload (it used to be a partial re-implementation)
    return calculatePhysicalResiduals(netlist, v_candidate, h, 0.0, &perNodeResiduals);
  }

  /**
   * calculatePhysicalResidualsDetailed
   * Like calculatePhysicalResiduals but additionally records per-device
   * attribution for the worst KCL violations. Returns max KCL error.
   * Populates worstDevices with sorted contributions to the worst node.
   */
  double calculatePhysicalResidualsDetailed(
      const TensorNetlist &netlist,
      const std::vector<double> &v_candidate,
      std::vector<ResidualContribution> &worstDevices,
      int &worstNodeOut) {

    int n = netlist.numGlobalNodes;
    worstDevices.clear();
    worstNodeOut = 0;
    if (n == 0) return 0.0;

    std::vector<double> node_residuals(n, 0.0);

    auto getV = [&](int nodeIdx) {
      if (nodeIdx <= 0 || nodeIdx > (int)v_candidate.size()) return 0.0;
      return v_candidate[nodeIdx - 1];
    };
    auto addRes = [&](int nodeIdx, double val) {
      if (nodeIdx > 0 && nodeIdx <= n)
        node_residuals[nodeIdx - 1] += val;
    };

    // Per-device attribution: (deviceName, deviceType, affectedNodes[], current)
    struct DeviceAttribution {
        std::string name;
        std::string type;
        std::vector<std::pair<int, double>> nodeCurrents; // (1-based node, current contributed)
    };
    std::vector<DeviceAttribution> attributions;

    // 1. Resistors
    for (const auto &r : netlist.globalBlock.resistors) {
      double v1 = getV(r.nodeTerminal1);
      double v2 = getV(r.nodeTerminal2);
      double i_r = (v1 - v2) / r.resistance_ohms;
      addRes(r.nodeTerminal1, -i_r);
      addRes(r.nodeTerminal2, i_r);
      attributions.push_back({r.name, "Resistor",
          {{r.nodeTerminal1, -i_r}, {r.nodeTerminal2, i_r}}});
    }

    // 2. Branch elements (V, L, E, F, H): node-row terms of the shared branch
    //    system; their currents are unknowns in v_candidate (DC: h = 0)
    {
      BranchSystem S;
      buildBranchSystem(netlist, 0.0, 0.0, S);
      for (size_t t = 0; t < S.r.size(); ++t)
        if (S.r[t] < n && S.c[t] < (int)v_candidate.size())
          node_residuals[S.r[t]] -= S.v[t] * v_candidate[S.c[t]];
    }

    // Current Sources
    for (const auto &cs : netlist.globalBlock.currentSources) {
      addRes(cs.nodePositive, -cs.current_A);
      addRes(cs.nodeNegative, cs.current_A);
      attributions.push_back({cs.type.empty() ? "I?" : cs.type, "CurrentSource",
          {{cs.nodePositive, -cs.current_A}, {cs.nodeNegative, cs.current_A}}});
    }


    // 3. Diodes
    for (const auto &d : netlist.globalBlock.diodes) {
      double v_d = getV(d.anode) - getV(d.cathode);
      DiodeParams<double> params = {d.saturationCurrent_I_S_A, d.emissionCoefficient_N,
                            d.thermalVoltage_V_T_V};
      double i_d = diode_current_clamped(v_d, params);
      addRes(d.anode, -i_d);
      addRes(d.cathode, i_d);
      attributions.push_back({d.modelName, "Diode",
          {{d.anode, -i_d}, {d.cathode, i_d}}});
    }

    // Schottky Diodes
    for (const auto &sd : netlist.globalBlock.schottkyDiodes) {
      double v_d = getV(sd.anode) - getV(sd.cathode);
      DiodeParams<double> params = {sd.saturationCurrent_I_S_A, sd.emissionCoefficient_N,
                            sd.thermalVoltage_V_T_V};
      double i_d = diode_current_clamped(v_d, params);
      addRes(sd.anode, -i_d);
      addRes(sd.cathode, i_d);
      attributions.push_back({sd.instanceName, "SchottkyDiode",
          {{sd.anode, -i_d}, {sd.cathode, i_d}}});
    }

    // BJTs (temperature-aware)
    for (const auto &q : netlist.globalBlock.bjts) {
      double v_c = getV(q.nodeCollector);
      double v_b = getV(q.base);
      double v_e = getV(q.emitter);
      BJTParams<double> params;
      params.Is = q.saturationCurrent_I_S_A;
      params.BetaF = q.betaF;
      params.BetaR = q.betaR;
      params.Vt = q.thermalVoltage_V_T_V;
      double ts_det = netlist.environment.ambient_temp_K / 300.15;
      params.Vt *= ts_det;
      params.Is *= std::pow(ts_det, 3.0) * exp(1.11 / q.thermalVoltage_V_T_V - 1.11 / params.Vt);
      BJTCurrents<double> currents = bjt_ebers_moll(v_c, v_b, v_e, q.isNPN, params);
      addRes(q.nodeCollector, -currents.Ic);
      addRes(q.base, -currents.Ib);
      addRes(q.emitter, -currents.Ie);
      attributions.push_back({q.instanceName, "BJT",
          {{q.nodeCollector, -currents.Ic}, {q.base, -currents.Ib}, {q.emitter, -currents.Ie}}});
    }

    // MOSFETs
    for (const auto &m : netlist.globalBlock.mosfets) {
      double vd = getV(m.drain), vg = getV(m.gate), vs = getV(m.source);
      MosfetParams<double> p;
      p.Kp = 200e-6; p.Vth = 0.7; p.lambda = 0.02; p.W = m.w; p.L = m.l;
      double ids = mosfet_ids(vg - vs, vd - vs, p);
      addRes(m.drain, -ids);
      addRes(m.source, ids);
      attributions.push_back({m.instanceName, "MOSFET",
          {{m.drain, -ids}, {m.source, ids}}});
    }

    // JFETs
    for (const auto &j : netlist.globalBlock.jfets) {
      double vgs = getV(j.gate) - getV(j.source);
      double vds = getV(j.drain) - getV(j.source);
      JFETParams<double> p;
      p.Beta = j.beta; p.Vto = j.Vto; p.Lambda = j.lambda;
      double sign = j.isNChannel ? 1.0 : -1.0;
      double ids = sign * jfet_ids(sign * vgs, sign * vds, p);
      addRes(j.drain, -ids);
      addRes(j.source, ids);
      attributions.push_back({j.instanceName, "JFET",
          {{j.drain, -ids}, {j.source, ids}}});
    }

    // Zener Diodes
    for (const auto &z : netlist.globalBlock.zenerDiodes) {
      double vd = getV(z.anode) - getV(z.cathode);
      ZenerParams<double> zp;
      zp.BV = z.breakdownVoltage_V; zp.IBV = z.currentAtBreakdown_A;
      zp.Rs = z.seriesResistance_Rs_ohms; zp.N = z.emissionCoefficient_N;
      zp.Is = z.saturationCurrent_I_S_A; zp.Vt = z.thermalVoltage_V_T_V;
      double iz = zener_current(vd, zp);
      addRes(z.anode, -iz);
      addRes(z.cathode, iz);
      attributions.push_back({z.instanceName, "ZenerDiode",
          {{z.anode, -iz}, {z.cathode, iz}}});
    }

    // DYNAMIC COMPONENTS (Generic Physics Models)
    // Re-use stampJacobian but only capture RHS (which sums currents/residuals)
    for (auto& [modelType, dynEntry] : soaBlock.dynamicTensors) {
        if (!dynEntry.tensor) continue;
        auto& registry = ModelRegistry::instance();
        const ModelInfo* info = registry.getModel(modelType, modelType);

        if (info && info->batchPhysics && info->stampJacobian) {
             info->batchPhysics(dynEntry.tensor, v_candidate);

             struct ResidualTarget : public SparseStampTarget {
                std::function<void(int, double)> addResFn;
                ResidualTarget(std::function<void(int, double)> fn) : addResFn(fn) {}
                void add(int r, int c, double v) override { (void)r; (void)c; (void)v; }
                void addRhs(int r, double v) override {
                    addResFn(r + 1, -v);
                }
            };
            ResidualTarget target(addRes);
            info->stampJacobian(dynEntry.tensor, dynEntry.count, &target);
        }
    }

    // Find worst node
    double max_err = 0.0;
    int worst_idx = 0;
    for (int i = 0; i < n; ++i) {
      double abs_val = std::abs(node_residuals[i]);
      if (abs_val > max_err) {
        max_err = abs_val;
        worst_idx = i;
      }
    }
    worstNodeOut = worst_idx + 1; // 1-based

    // Build per-device attribution for worst node
    for (const auto &attr : attributions) {
      for (const auto &[node, current] : attr.nodeCurrents) {
        if (node == worstNodeOut && std::abs(current) > 1e-15) {
          ResidualContribution rc;
          rc.deviceName = attr.name;
          rc.deviceType = attr.type;
          rc.nodeIndex = node;
          rc.currentContribution = current;
          rc.percentOfTotal = (max_err > 0.0) ? (std::abs(current) / max_err * 100.0) : 0.0;
          worstDevices.push_back(rc);
        }
      }
    }

    // Sort by contribution magnitude (descending)
    std::sort(worstDevices.begin(), worstDevices.end(),
              [](const ResidualContribution &a, const ResidualContribution &b) {
                return std::abs(a.currentContribution) > std::abs(b.currentContribution);
              });

    return max_err;
  }

  // Phase 2.5: Differentiable Physics
  enum class ComponentType { RESISTOR, CAPACITOR, INDUCTOR, DIODE, MOSFET, BJT, JFET };

  /**
   * calculateSensitivity
   * Computes the gradient of node voltages with respect to a specific component parameter (dV/dp).
   * Solves: J * (dV/dp) = - (dF/dp)
   */
  std::vector<double> calculateSensitivity(TensorNetlist &netlist, 
                                           ComponentType type, 
                                           int index, 
                                           const std::string& paramName, 
                                           const std::vector<double>& v_converged) {
      if (v_converged.empty()) return {};

      // 1. Re-build Jacobian (J) at convergence point
      // We assume standard stamping (double params) gives us the correct J = dF/dV
      scheduler.schedule(netlist);
      matrixBuilder.clear();
      const int nTot = systemSize(netlist);  // nodes + branch currents
      matrixBuilder.setDimensions(nTot, nTot);
      rhsVector.assign(nTot, 0.0);
      std::vector<double> x_converged = v_converged;
      if ((int)lastSolution_.size() == nTot) x_converged = lastSolution_;  // with branch currents
      x_converged.resize(nTot, 0.0);
      stampAllElements(netlist, 0.0, x_converged, 0.0); // h=0, time=0 for DC sensitivity
      addDiagonalConditioning(netlist.numGlobalNodes);
      Csr_matrix J = matrixBuilder.createCsr(); /* Explicitly J */
      
      // 2. Build RHS vector b = - dF/dp
      // This vector is zero everywhere except at the nodes of the target component
      std::vector<double> b_sens(nTot, 0.0);
      
      auto getV = [&](int n) { return (n>0 && n<=(int)v_converged.size()) ? v_converged[n-1] : 0.0; };

      try {
          if (type == ComponentType::RESISTOR) {
              if (index < 0 || index >= (int)netlist.globalBlock.resistors.size()) return {};
              const auto& r = netlist.globalBlock.resistors[index];
              
              // Only 'resistance' is supported for now
              if (paramName != "resistance" && paramName != "R") return {};
              
              // I_leaving_n1 = (V1 - V2) / R
              // dI/dR = -(V1 - V2) / R^2
              double v_diff = getV(r.nodeTerminal1) - getV(r.nodeTerminal2);
              double R = r.resistance_ohms;
              double didp = -v_diff / (R * R);
              
              // F_n1 += I_leaving. dF_n1/dp = dI/dp
              // b[n1] = -dF/dp = -dI/dp
              if (r.nodeTerminal1 > 0) b_sens[r.nodeTerminal1-1] -= didp;
              if (r.nodeTerminal2 > 0) b_sens[r.nodeTerminal2-1] += didp; // I_leaving_n2 is -I -> d(-I)/dp = -dI/dp -> b = +dI/dp
          } 
          else if (type == ComponentType::DIODE) {
              if (index < 0 || index >= (int)netlist.globalBlock.diodes.size()) return {};
              const auto& d = netlist.globalBlock.diodes[index];
              
              double v_diff = getV(d.anode) - getV(d.cathode);
              
              // Create DUAL parameters
              // Example: paramName="Is" -> seed Is with grad=1.0
              Dual<double> Is(d.saturationCurrent_I_S_A, (paramName == "Is" ? 1.0 : 0.0));
              Dual<double> N(d.emissionCoefficient_N, (paramName == "N" ? 1.0 : 0.0));
              Dual<double> Vt(d.thermalVoltage_V_T_V, (paramName == "Vt" ? 1.0 : 0.0));
              
              DiodeParams<Dual<double>> p_dual = {Is, N, Vt};
              
              // Evaluate physics with Dual params and Constant voltage
              // v_diff is treated as constant (grad=0) for partial dF/dp
              Dual<double> i_d = diode_current_clamped(Dual<double>(v_diff), p_dual);
              
              double didp = i_d.grad; // This is dI_anode / dp
              
              if (d.anode > 0) b_sens[d.anode-1] -= didp;
              if (d.cathode > 0) b_sens[d.cathode-1] += didp;
          }
          else if (type == ComponentType::MOSFET) {
              if (index < 0 || index >= (int)netlist.globalBlock.mosfets.size()) return {};
              const auto& m = netlist.globalBlock.mosfets[index];

              double vgs = getV(m.gate) - getV(m.source);
              double vds = getV(m.drain) - getV(m.source);
              
              MosfetParams<Dual<double>> p;
              // Hardcoded model params for now, ideally fetch from model library
              // But 'W' and 'L' are instance params
              p.Kp = Dual<double>(200e-6); 
              p.Vth = Dual<double>(0.7, (paramName == "Vth" ? 1.0 : 0.0));
              p.lambda = Dual<double>(0.02);
              p.W = Dual<double>(m.w, (paramName == "W" ? 1.0 : 0.0));
              p.L = Dual<double>(m.l, (paramName == "L" ? 1.0 : 0.0));
              
              Dual<double> ids = mosfet_ids(Dual<double>(vgs), Dual<double>(vds), p);
              double didp = ids.grad; // dIds / dp
              
              // I_drain leaves (positive in F_drain)
              if (m.drain > 0) b_sens[m.drain-1] -= didp;
               // I_source enters (negative in F_source)
               if (m.source > 0) b_sens[m.source-1] += didp;
          }
          // CAPACITOR SENSITIVITY (DC: dI/dC = 0, AC needs omega)
          else if (type == ComponentType::CAPACITOR) {
              if (index < 0 || index >= (int)netlist.globalBlock.capacitors.size()) return {};
              // DC: Capacitors are open circuit, no current, so dI/dC = 0.
              // For AC sensitivity, user would need to provide omega. Placeholder for now.
          }
          // INDUCTOR SENSITIVITY (DC: dI/dL = 0, AC needs omega)
          else if (type == ComponentType::INDUCTOR) {
              if (index < 0 || index >= (int)netlist.globalBlock.inductors.size()) return {};
              // DC: Inductors are short, no L term in DC model. dI/dL = 0.
          }
          // BJT SENSITIVITY
          else if (type == ComponentType::BJT) {

               if (index < 0 || index >= (int)netlist.globalBlock.bjts.size()) return {};
               const auto& q = netlist.globalBlock.bjts[index];
               
               double vc = getV(q.nodeCollector);
               double vb = getV(q.base);
               double ve = getV(q.emitter);
               
               // Seed Dual parameters based on what we are differentiating against
               Dual<double> Is(q.saturationCurrent_I_S_A, (paramName == "Is" ? 1.0 : 0.0));
               Dual<double> Bf(q.betaF, ((paramName == "BetaF" || paramName == "Beta") ? 1.0 : 0.0));
               Dual<double> Br(q.betaR, (paramName == "BetaR" ? 1.0 : 0.0));
               Dual<double> Vt(q.thermalVoltage_V_T_V, (paramName == "Vt" ? 1.0 : 0.0));
               
               // Reuse the existing templated physics kernel? 
               // bjt_ebers_moll is currently hardcoded for double (inline double-only implementation).
               // We need to look at device_physics.h. It seems I didn't verify if it's templated!
               // Checking device_physics.h... it returns BJTCurrents struct with doubles.
               // Ah, the implementation in device_physics.h lines 164 is:
               // inline BJTCurrents bjt_ebers_moll(double v_c, ... const BJTParams<double>& p)
               // It accepts double params. It is NOT templated for Dual<double> yet.
               // Ideally, I should template it. But for now, I can replicate the simple Ebers-Moll equation logic here using Duals.
               // Ebers-Moll is relatively simple.
               
               // voltages are constant for partial dF/dp
               Dual<double> v_be_d = (q.isNPN ? 1.0 : -1.0) * (vb - ve);
               Dual<double> v_bc_d = (q.isNPN ? 1.0 : -1.0) * (vb - vc);
               
               auto safe_exp = [&](Dual<double> v, Dual<double> vt) { 
                    Dual<double> arg = v / vt;
                    if (arg.val > 40.0) arg = Dual<double>(40.0, 0.0);
                    using std::exp;
                    return exp(arg); // Use ADL for Dual
               };

               Dual<double> i_f = Is * (safe_exp(v_be_d, Vt) - 1.0);
               Dual<double> i_r = Is * (safe_exp(v_bc_d, Vt) - 1.0);
               
               Dual<double> i_ct = i_f - i_r;
               Dual<double> i_b_f = i_f / Bf;
               Dual<double> i_b_r = i_r / Br;
               
               Dual<double> ic = i_ct - i_b_r;
               Dual<double> ib = i_b_f + i_b_r;
               // Dual<double> ie = -ic - ib;
               
               // We need dIc/dp, dIb/dp, dIe/dp
               // Note: These are currents INTO the terminal. 
               // In RHS b_sens, we add +dI/dp if it enters?
               // F_node = Sum(I_entering) = 0.
               // dF/dp = Sum(dI_entering/dp)
               // b = -dF/dp.
               // So b -= dI_entering/dp.
               
               // Correct sign fix:
               // If I_c enters Collector, residual R_c += I_c
               // dR_c / dp = dI_c / dp
               // b_c = - dR_c / dp = - dI_c / dp.
               
               double sign = (q.isNPN ? 1.0 : -1.0);
               
               if (q.nodeCollector > 0) b_sens[q.nodeCollector-1] -= (sign * ic.grad);
               if (q.base > 0)          b_sens[q.base-1]          -= (sign * ib.grad);
               if (q.emitter > 0)       b_sens[q.emitter-1]       -= (sign * (-ic.grad - ib.grad));
            }
      } catch(...) {
          return {};
      }

      // 3. Solve J * x = b
      // J is nonsymmetric (PCG assumed symmetry); solve exactly, return node rows
      SolverResult res = solveLinear(J, b_sens);
      res.solution.resize(netlist.numGlobalNodes);
      return res.solution;
  }

  // Phase 2: Numerical Integration Strategy


  // --- SOLID/NASA Helper Functions ---

  // --- SOLID/NASA Helper Functions ---

  /**
   * stampBlockLinear
   * Stamps linear elements (Resistors, Sources, Capacitors) for a specific
   * block instance.
   */
  void stampBlockLinear(const TensorBlock &block, const BlockState &state,
                        const std::function<int(int)> &mapNode, double h, double time,
                        const std::vector<double> &v_guess,
                        double voltageScale = 1.0, int mcSeed = 0) {

    // 1. Resistors
    for (const auto &r : block.resistors) {
      // MC Perturbation
      double R = getPerturbedValue(r.resistance_ohms, r.tolerance_percent, r.name, mcSeed);
      
      stampResistor(mapNode(r.nodeTerminal1), mapNode(r.nodeTerminal2),
                    R, matrixBuilder);
    }
    // 2. V sources, inductors (+K), E/F/H: branch-current elements, stamped
    //    for every block by stampBranchSystem() from stampAllElements()
    // 2.5 Power Rails (System Source)
    for (size_t i = 0; i < block.powerRails.size(); ++i) {
        const auto &rail = block.powerRails[i];
        const auto &hist = state.powerRailState[i];
        // Retrieve history (dc/transient)
        double v_prev = 0.0;
        double i_prev_cap = 0.0;
        if (!hist.v.empty()) {
            v_prev = hist.v[0];
            i_prev_cap = hist.i[0]; // Stores capacitor current
        }
        
        stampPowerRail(mapNode(rail.nodeRail), rail.nominal_V * voltageScale, rail.ripple_Vpp, rail.frequency_Hz,
                       rail.ESR_ohms, rail.capacitance_F, h, time, 
                       v_prev, i_prev_cap,
                       matrixBuilder, rhsVector);
    }

    // 3. Current Sources
    // 3. Current Sources
    for (const auto &i : block.currentSources) {
      double val = (h > 1e-15) ? waveformValue(i, i.current_A, time) : i.current_A;

      int nP = mapNode(i.nodePositive);
      int nN = mapNode(i.nodeNegative);
      if (nP != 0)
        rhsVector[nP - 1] -= val;
      if (nN != 0)
        rhsVector[nN - 1] += val;
    }
    // 4. Capacitors (Transient)
    if (h > 1e-15 && integrator) {
      for (size_t i = 0; i < block.capacitors.size(); ++i) {
        const auto &c = block.capacitors[i];
        const auto &hist = state.capacitorState[i];
        double C = getPerturbedValue(c.capacitance_farads, c.tolerance_percent, c.name, mcSeed);
        
        integrator->stampCapacitor(mapNode(c.nodePlate1), mapNode(c.nodePlate2),
                                   C, h, hist.v,
                                   hist.i, matrixBuilder, rhsVector);
      }
    }

    // 6. Dependent Sources: G here; E/F/H are branch elements
    for (const auto &vccs : block.voltageControlledCurrentSources) {
        stampVCCS(mapNode(vccs.ctrlPos), mapNode(vccs.ctrlNeg),
                  mapNode(vccs.outPos), mapNode(vccs.outNeg),
                  vccs.gm, matrixBuilder, rhsVector);
    }

    // 7. Transmission Lines (Phase 2 placeholder)
    for (const auto &t : block.transmissionLines) {
      // Simple resistive model for Phase 1
      double z0 = t.characteristicImpedance_Z0_ohms;
      stampResistor(mapNode(t.nodePort1Pos), mapNode(t.nodePort1Neg), z0,
                    matrixBuilder);
      stampResistor(mapNode(t.nodePort2Pos), mapNode(t.nodePort2Neg), z0,
                    matrixBuilder);
    }
  }

  /**
   * stampBlockNonLinear
   * Linearizes and stamps non-linear elements (Diodes, MOSFETs, BJTs) for a specific
   * block instance.
   * Uses Forward-Mode Automatic Differentiation (AD) via Dual<double>.
   */
  /**
   * stampSoABlock
   * Uses Structure-of-Arrays (SoA) tensors for batch processing.
   * Calls specialized SIMD-friendly physics kernels and then stamps the results.
   */
  void stampSoABlock(TensorizedBlock& block, const std::vector<double>& v_guess, double h = 0.0,
                     IntegrationType intType = IntegrationType::TRAPEZOIDAL) {
    // 0. Linear Components
    
    // Resistors
    size_t nR = block.resistors.size();
    if (nR > 0) {
        std::vector<int> rows, cols;
        std::vector<double> vals;
        rows.reserve(nR * 4); cols.reserve(nR * 4); vals.reserve(nR * 4);
        for (size_t i = 0; i < nR; ++i) {
            int n1 = block.resistors.nodes1[i];
            int n2 = block.resistors.nodes2[i];
            double g = 1.0 / block.resistors.R[i];
            if (n1 != 0 && n2 != 0) {
                rows.push_back(n1-1); cols.push_back(n1-1); vals.push_back(g);
                rows.push_back(n2-1); cols.push_back(n2-1); vals.push_back(g);
                rows.push_back(n1-1); cols.push_back(n2-1); vals.push_back(-g);
                rows.push_back(n2-1); cols.push_back(n1-1); vals.push_back(-g);
            } else if (n1 != 0) {
                rows.push_back(n1-1); cols.push_back(n1-1); vals.push_back(g);
            } else if (n2 != 0) {
                rows.push_back(n2-1); cols.push_back(n2-1); vals.push_back(g);
            }
        }
        matrixBuilder.addBatch(rows, cols, vals);
    }

    // V sources / inductors: branch-current elements (stampBranchSystem)

    // Current Sources
    size_t nIS = block.currentSources.size();
    for (size_t i = 0; i < nIS; ++i) {
        int nP = block.currentSources.nodesPos[i];
        int nN = block.currentSources.nodesNeg[i];
        double I = block.currentSources.currents[i];
        if (nP != 0) rhsVector[nP-1] -= I;
        if (nN != 0) rhsVector[nN-1] += I;
    }

    // Capacitors (Transient)
    size_t nC = block.capacitors.size();
    if (nC > 0 && h > 0.0) {
        std::vector<int> rows, cols;
        std::vector<double> vals;
        rows.reserve(nC * 4); cols.reserve(nC * 4); vals.reserve(nC * 4);
        for (size_t i = 0; i < nC; ++i) {
            int n1 = block.capacitors.plates1[i];
            int n2 = block.capacitors.plates2[i];
            double g_eq, i_hist;
            // GEAR_1_EULER (Backward Euler): g=C/h, no i_prev term.
            // GEAR_2 (BDF2): g=1.5C/h, I=C(2v1-0.5v2)/h — must match the KCL
            //   audit and history update; the old BE fallback here made every
            //   transient that switched to Gear2 fail NR at all step sizes.
            // TRAPEZOIDAL (default): g=2C/h, companion current uses i_prev.
            if (intType == IntegrationType::GEAR_1_EULER) {
                g_eq   = block.capacitors.C[i] / h;
                i_hist = g_eq * block.capacitors.v_prev[i];
            } else if (intType == IntegrationType::GEAR_2) {
                g_eq   = 1.5 * block.capacitors.C[i] / h;
                i_hist = block.capacitors.C[i] / h *
                         (2.0 * block.capacitors.v_prev[i] - 0.5 * block.capacitors.v_prev2[i]);
            } else {
                g_eq   = (2.0 * block.capacitors.C[i]) / h;
                i_hist = g_eq * block.capacitors.v_prev[i] + block.capacitors.i_prev[i];
            }

            if (n1 != 0 && n2 != 0) {
                rows.push_back(n1-1); cols.push_back(n1-1); vals.push_back(g_eq);
                rows.push_back(n2-1); cols.push_back(n2-1); vals.push_back(g_eq);
                rows.push_back(n1-1); cols.push_back(n2-1); vals.push_back(-g_eq);
                rows.push_back(n2-1); cols.push_back(n1-1); vals.push_back(-g_eq);
            } else if (n1 != 0) {
                rows.push_back(n1-1); cols.push_back(n1-1); vals.push_back(g_eq);
            } else if (n2 != 0) {
                rows.push_back(n2-1); cols.push_back(n2-1); vals.push_back(g_eq);
            }
            if (n1 != 0) rhsVector[n1-1] += i_hist;
            if (n2 != 0) rhsVector[n2-1] -= i_hist;
        }
        matrixBuilder.addBatch(rows, cols, vals);
    }
    // 1. Batch Compute Physics (Updates internal state: v_d, i_d, g_d, etc.)
    batchDiodePhysics(block.diodes, v_guess);
    batchMosfetPhysics(block.mosfets, v_guess);
    batchBJTPhysics(block.bjts, v_guess);

    // 2. Batch Stamp (Iterate over contiguous arrays)
    
    // Diodes
    // Diodes
    size_t nD = block.diodes.size();
    if (nD > 0) {
        std::vector<int> rows, cols;
        std::vector<double> vals;
        rows.reserve(nD * 4);
        cols.reserve(nD * 4);
        vals.reserve(nD * 4);
        
        for (size_t i = 0; i < nD; ++i) {
            int nA = block.diodes.node_a[i];
            int nC = block.diodes.node_c[i];
            double g = block.diodes.g_d[i];
            double i_eq = block.diodes.i_d[i] - g * block.diodes.v_d[i];
            
            // Matrix Stamping (G)
            if (nA != 0 && nC != 0) {
                // Diagonal
                rows.push_back(nA-1); cols.push_back(nA-1); vals.push_back(g);
                rows.push_back(nC-1); cols.push_back(nC-1); vals.push_back(g);
                // Off-Diagonal
                rows.push_back(nA-1); cols.push_back(nC-1); vals.push_back(-g);
                rows.push_back(nC-1); cols.push_back(nA-1); vals.push_back(-g);
            } else if (nA != 0) {
                rows.push_back(nA-1); cols.push_back(nA-1); vals.push_back(g);
            } else if (nC != 0) {
                rows.push_back(nC-1); cols.push_back(nC-1); vals.push_back(g);
            }
            
            // RHS Stamping
            if (nA != 0) rhsVector[nA-1] -= i_eq;
            if (nC != 0) rhsVector[nC-1] += i_eq;
        }
        matrixBuilder.addBatch(rows, cols, vals);
        
        // GPU Hook (Future Integration)
        /*
        if (activeBackend == Backend::WEBGPU || activeBackend == Backend::CUDA) {
            // Upload Tensors to GPU Storage Buffer
            gpuAssembler.upload(block.diodes.g_d_buffer, block.diodes.g_d); 
            // Dispatch Assembly Shader (Scatter-Add)
            gpuAssembler.dispatchStamping("diode_stamp_kernel");
        }
        */
    }


    
    // MOSFETs
    // MOSFETs
    size_t nM = block.mosfets.size();
    if (nM > 0) {
        std::vector<int> rows, cols;
        std::vector<double> vals;
        // Estimate size: 4 nodes * 3 connections approx? 
        // gm: ID vs VGS -> (D,G), (S,G)
        // gds: ID vs VDS -> (D,D), (D,S), (S,D), (S,S)
        rows.reserve(nM * 12);
        cols.reserve(nM * 12);
        vals.reserve(nM * 12);
        
        for (size_t i=0; i<nM; ++i) {
             int d = block.mosfets.drains[i];
             int g = block.mosfets.gates[i];
             int s = block.mosfets.sources[i];
             int b = block.mosfets.bodies[i];
             
             double gm = block.mosfets.gm[i];
             double gmb = block.mosfets.gmb[i];
             double gds = block.mosfets.gds[i];
             double ids = block.mosfets.ids[i];
             double vgs = block.mosfets.vgs[i];
             double vds = block.mosfets.vds[i];
             double vbs = (b != 0) ? (v_guess[b-1] - (s != 0 ? v_guess[s-1] : 0.0)) : 0.0;
             double i_eq = ids - (gm * vgs + gmb * vbs + gds * vds);
             
             // Stamping Gds (Conductance between Drain and Source)
             if (d!=0 && s!=0) {
                 rows.push_back(d-1); cols.push_back(d-1); vals.push_back(gds);
                 rows.push_back(s-1); cols.push_back(s-1); vals.push_back(gds);
                 rows.push_back(d-1); cols.push_back(s-1); vals.push_back(-gds);
                 rows.push_back(s-1); cols.push_back(d-1); vals.push_back(-gds);
             } else if (d!=0) {
                 rows.push_back(d-1); cols.push_back(d-1); vals.push_back(gds);
             } else if (s!=0) {
                 rows.push_back(s-1); cols.push_back(s-1); vals.push_back(gds);
             }
             
             // Stamping Gm (VGS control)
             if (d!=0) {
                 if (g!=0) { rows.push_back(d-1); cols.push_back(g-1); vals.push_back(gm); }
                 if (s!=0) { rows.push_back(d-1); cols.push_back(s-1); vals.push_back(-gm); }
             }
             if (s!=0) {
                 if (g!=0) { rows.push_back(s-1); cols.push_back(g-1); vals.push_back(-gm); }
                 if (s!=0) { rows.push_back(s-1); cols.push_back(s-1); vals.push_back(gm); }
             }

             // Stamping Gmb (VBS control)
             if (d!=0) {
                 if (b!=0) { rows.push_back(d-1); cols.push_back(b-1); vals.push_back(gmb); }
                 if (s!=0) { rows.push_back(d-1); cols.push_back(s-1); vals.push_back(-gmb); }
             }
             if (s!=0) {
                 if (b!=0) { rows.push_back(s-1); cols.push_back(b-1); vals.push_back(-gmb); }
                 if (s!=0) { rows.push_back(s-1); cols.push_back(s-1); vals.push_back(gmb); }
             }
             
             // RHS
             // I_d_total = I_eq + ...
             // KCL DrainNode: Isum = 0 -> ... + I_d_leaving = 0 -> ... + I_eq = 0 -> RHS = -I_eq
             // I_d leaving drain
             if (d!=0) rhsVector[d-1] -= i_eq;
             if (s!=0) rhsVector[s-1] += i_eq;
        }
        matrixBuilder.addBatch(rows, cols, vals);

        // GPU Hook (Future Integration)
        /*
        if (activeBackend == Backend::WEBGPU) {
             gpuAssembler.upload(block.mosfets.gds_buffer, block.mosfets.gds);
             gpuAssembler.dispatchStamping("mosfet_stamp_kernel");
        }
        */
    }


    
    // BJTs
    // MOSFETs (Omitted for brevity in previous view, following same pattern...)
    
    // BJTs
    size_t nB = block.bjts.size();
    if (nB > 0) {
        std::vector<int> rows, cols;
        std::vector<double> vals;
        rows.reserve(nB * 9); cols.reserve(nB * 9); vals.reserve(nB * 9);
        
        for (size_t i = 0; i < nB; ++i) {
            int nC = block.bjts.collectors[i];
            int nB = block.bjts.bases[i];
            int nE = block.bjts.emitters[i];
            
            double g_cc = block.bjts.g_cc[i]; double g_cb = block.bjts.g_cb[i]; double g_ce = block.bjts.g_ce[i];
            double g_bc = block.bjts.g_bc[i]; double g_bb = block.bjts.g_bb[i]; double g_be = block.bjts.g_be[i];
            double g_ec = block.bjts.g_ec[i]; double g_eb = block.bjts.g_eb[i]; double g_ee = block.bjts.g_ee[i];
            
            // Helper to push triplet
            auto addG = [&](int r, int c, double v) { 
                if(r!=0 && c!=0) { rows.push_back(r-1); cols.push_back(c-1); vals.push_back(v); }
            };
            
            addG(nC, nC, g_cc); addG(nC, nB, g_cb); addG(nC, nE, g_ce);
            addG(nB, nC, g_bc); addG(nB, nB, g_bb); addG(nB, nE, g_be);
            addG(nE, nC, g_ec); addG(nE, nB, g_eb); addG(nE, nE, g_ee);
            
            // RHS
             double v_c = (nC>0 ? v_guess[nC-1] : 0.0);
             double v_b = (nB>0 ? v_guess[nB-1] : 0.0);
             double v_e = (nE>0 ? v_guess[nE-1] : 0.0);
             
             double Iceq = block.bjts.Ic[i] - (g_cc*v_c + g_cb*v_b + g_ce*v_e);
             double Ibeq = block.bjts.Ib[i] - (g_bc*v_c + g_bb*v_b + g_be*v_e);
             double Ieeq = block.bjts.Ie[i] - (g_ec*v_c + g_eb*v_b + g_ee*v_e);
             
             if (nC != 0) rhsVector[nC-1] -= Iceq;
             if (nB != 0) rhsVector[nB-1] -= Ibeq;
             if (nE != 0) rhsVector[nE-1] -= Ieeq;
        }
        matrixBuilder.addBatch(rows, cols, vals);

        // GPU Hook (Future Integration)
        /*
        if (activeBackend == Backend::WEBGPU) {
             gpuAssembler.dispatchStamping("bjt_stamp_kernel");
        }
        */
    }

    // 3. Dynamic Models (ModelRegistry dispatch)
    auto& registry = ModelRegistry::instance();
    for (auto& [deviceType, entry] : block.dynamicTensors) {
        if (!entry.tensor || entry.count == 0) continue;

        auto models = registry.listModels(deviceType);
        if (models.empty()) continue;
        const ModelInfo* info = registry.getModel(deviceType, models[0]);
        if (!info) continue;

        // Batch physics evaluation
        if (info->batchPhysics) {
            info->batchPhysics(entry.tensor, v_guess);
        }

        // Jacobian + RHS stamping via SparseStampTarget adapter
        if (info->stampJacobian) {
            struct StampAdapter : SparseStampTarget {
                MatrixConstructor& mb;
                std::vector<double>& rhs;
                StampAdapter(MatrixConstructor& mb_, std::vector<double>& rhs_)
                    : mb(mb_), rhs(rhs_) {}
                void add(int row, int col, double val) override { mb.add(row, col, val); }
                void addRhs(int row, double val) override { rhs[row] += val; }
            };
            StampAdapter adapter(matrixBuilder, rhsVector);
            info->stampJacobian(entry.tensor, entry.count, &adapter);
        }
    }

  }

  void stampBlockNonLinear(const TensorBlock &block,
                           const std::function<int(int)> &mapNode,
                           const std::vector<double> &v_guess,
                           double temp_K = 300.15) {
    auto getV = [&](int globalNode) {
      if (globalNode > 0 && globalNode <= (int)v_guess.size())
        return v_guess[globalNode - 1];
      return 0.0;
    };

    // 1. Diodes
    for (const auto &d : block.diodes) {
      int nA = mapNode(d.anode);
      int nC = mapNode(d.cathode);
      stampDiode(d, nA, nC, getV(nA) - getV(nC), temp_K, matrixBuilder, rhsVector);
    }

    // 1.5 Schottky Diodes (Reuse Unified Diode Logic)
    for (const auto &sd : block.schottkyDiodes) {
      int nA = mapNode(sd.anode);
      int nC = mapNode(sd.cathode);
      stampSchottky(sd, nA, nC, getV(nA) - getV(nC), temp_K, matrixBuilder, rhsVector);
    }

    // 2. MOSFETs
    for (const auto &m : block.mosfets) {
      int nD = mapNode(m.drain);
      int nG = mapNode(m.gate);
      int nS = mapNode(m.source);
      int nB = mapNode(m.body);

      stampMosfet(m, nD, nG, nS, nB,
                  getV(nD), getV(nG), getV(nS), getV(nB),
                  matrixBuilder, rhsVector);
    }

    // 3. BJTs
    for (const auto &q : block.bjts) {
        int nC = mapNode(q.nodeCollector);
        int nB = mapNode(q.base);
        int nE = mapNode(q.emitter);
        int nS = 0; // Substrate not yet modeled

        stampBJT(q, nC, nB, nE, nS,
                 getV(nC), getV(nB), getV(nE), 0.0,
                 temp_K,
                 matrixBuilder, rhsVector);
    }

    // 4. JFETs
    for (const auto &j : block.jfets) {
        int nD = mapNode(j.drain);
        int nG = mapNode(j.gate);
        int nS = mapNode(j.source);

        stampJFET(j, nD, nG, nS,
                  getV(nD), getV(nG), getV(nS),
                  matrixBuilder, rhsVector);
    }

    // 5. Zener Diodes
    for (const auto &z : block.zenerDiodes) {
        int nA = mapNode(z.anode);
        int nC = mapNode(z.cathode);
        
        stampZener(z, nA, nC, getV(nA) - getV(nC), temp_K, matrixBuilder, rhsVector);
    }
  }

  /**
   * stampAllElements
   * High-level entry point for matrix assembly.
   */
  void stampAllElements(const TensorNetlist &netlist, double h,
                        const std::vector<double> &v_guess, double time) {
    // Set model card pointer for per-model parameter lookup
    currentModelCards = &netlist.modelCards;

    if (execMode == ExecutionMode::LEGACY_AOS) {
        // A. Global Block
        auto identityMap = [](int n) { return n; };
        stampBlockLinear(netlist.globalBlock, netlist.globalState, identityMap, h, time, v_guess, netlist.environment.global_voltage_scale, netlist.environment.monte_carlo_seed);
        stampBlockNonLinear(netlist.globalBlock, identityMap, v_guess, netlist.environment.ambient_temp_K);

        // B. Hierarchical Batches
        for (const auto &job : scheduler.jobs) {
          const TensorBlock &block = *job.blockTmpl;
          for (int instanceIdx : job.instanceIds) {
            const auto &instance = netlist.instances[instanceIdx];
            const auto &instState = netlist.instanceStates[instanceIdx];
            auto mapNode = [&](int localNode) {
              return (localNode <= 0) ? 0 : instance.nodeMapping[localNode - 1]; // NodeMapping check: 0->0, 1->mapping[0]
            };
            stampBlockLinear(block, instState, mapNode, h, time, v_guess, netlist.environment.global_voltage_scale, netlist.environment.monte_carlo_seed);
            stampBlockNonLinear(block, mapNode, v_guess, netlist.environment.ambient_temp_K);
          }
        }
    } else {
        // SoA Pathway (Flattened). tensorizeNetlist() stores DC values only, so
        // in transient re-evaluate every source at `time`, walking blocks in the
        // same order tensorizeNetlist() flattens them (global, then instances).
        if (h > 1e-15) {
            size_t ki = 0;
            auto refresh = [&](const TensorBlock &blk) {
                for (const auto &i : blk.currentSources)
                    if (ki < soaBlock.currentSources.currents.size())
                        soaBlock.currentSources.currents[ki++] = waveformValue(i, i.current_A, time);
            };
            refresh(netlist.globalBlock);
            for (const auto &inst : netlist.instances) {
                auto it = netlist.blockDefinitions.find(inst.blockName);
                if (it != netlist.blockDefinitions.end()) refresh(it->second);
            }
        }
        stampSoABlock(soaBlock, v_guess, h, integrator ? integrator->getType() : IntegrationType::TRAPEZOIDAL);

        // Devices tensorizeNetlist() has no SoA tensors for (G, J) were
        // silently dropped here. Stamp them with the same helpers the AoS path
        // uses, so both paths share one implementation.
        auto getV = [&](int n) { return (n > 0 && n <= (int)v_guess.size()) ? v_guess[n - 1] : 0.0; };
        auto stampUncovered = [&](const TensorBlock &blk, const std::function<int(int)> &mapNode) {
            for (const auto &g : blk.voltageControlledCurrentSources)
                stampVCCS(mapNode(g.ctrlPos), mapNode(g.ctrlNeg), mapNode(g.outPos), mapNode(g.outNeg),
                          g.gm, matrixBuilder, rhsVector);
            for (const auto &j : blk.jfets) {
                int nD = mapNode(j.drain), nG = mapNode(j.gate), nS = mapNode(j.source);
                stampJFET(j, nD, nG, nS, getV(nD), getV(nG), getV(nS), matrixBuilder, rhsVector);
            }
        };
        stampUncovered(netlist.globalBlock, [](int n) { return n; });
        for (const auto &inst : netlist.instances) {
            auto it = netlist.blockDefinitions.find(inst.blockName);
            if (it == netlist.blockDefinitions.end()) continue;
            stampUncovered(it->second, [&](int n) {  // same remap as tensorizeNetlist()
                return (n <= 0 || n > (int)inst.nodeMapping.size()) ? 0 : inst.nodeMapping[n - 1];
            });
        }
    }

    // Branch-current elements (V, L+K, E, F, H): one implementation for both paths
    stampBranchSystem(netlist, h, time);

    // ─── Gap 2: Digital Event Engine — D2A Stamping ──────────────────────
    if (digitalEngine_.hasOutputPorts()) {
        digitalEngine_.stampD2A(matrixBuilder, rhsVector, netlist.numGlobalNodes, h);
    }
  }


  void addDiagonalConditioning(int n, double gmin = 1e-12) {
    for (int i = 0; i < n; ++i)
      matrixBuilder.add(i, i, gmin);
  }


  // Placeholder for audit logic (implemented in audit file)
#include "circuitsim_audit.inc"
};

// =============================================================================
// SOLVER IMPLEMENTATIONS
// =============================================================================
// solveDC and stepTransient are implemented in circuitsim_solver.cpp
// to reduce header size and improve compile times.
