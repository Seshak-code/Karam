#pragma once
#include "gui_boundary_guard.h"
#include "../netlist/circuit.h"
#include "../math/linalg.h"
#include "../math/contraction_solver.h"
#include "../math/worker_pool.h"
#include <mutex>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <functional>
#include <type_traits>
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
#include "../osdi/osdi_device.h"
#include "physics_constants.h"
#include "../tensors/physics_tensors.h"
#include "stamps.h"

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
  ContractionCache contractionCache_;  // unchanged subtrees keep their factors between solves
  SolverResult solveLinear(const Csr_matrix& A, const std::vector<double>& b) {
    if (!contractionPlan_ || !contractionPlan_->matches(A)) {
      contractionPlan_ = std::make_unique<ContractionPlan>(planContraction(A));
      contractionCache_ = {};  // factors belong to the old plan
    }
    return solveContraction(A, b, *contractionPlan_, 2, &contractionCache_);
  }

  // Converged solution of the prior solve (seeder history, dV/dt)
  std::vector<double> prevVoltages_;

  // Gap 4: Neural convergence seeder — physics-informed dV/dt extrapolation.
  // Records each converged solution and uses history for next initial guess.
  NeuralSeeder neuralSeeder_;

  // Gap 5: LTE-based adaptive timestep controller (truncation error, step size).
  LTEController lteController_;
  // Accepted steps after a waveform corner whose LTE is not checked: divided differences across
  // the kink measure the kink (an inductor's voltage jumps with dI/dt), not truncation error, and
  // rejected step after step down to femtoseconds. Three refill a trapezoidal history past it.
  int lteHold_ = 0;
  double lteTolerance_ = 1e-4;  // absolute LTE tolerance

  // Gap 2: Mixed-signal digital event engine.
  // Handles A2D/D2A boundary conditions and zero-delay digital propagation.
  DigitalEventEngine digitalEngine_;

  // Callbacks for orchestration
  std::function<void(int, double)> onConvergenceStep;
  std::function<void(float)>       onProgress;

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

  // MOSFET charge branches gs, gd, gb, bd, bs (mos1_charges) as terminal
  // pairs (d=0, g=1, s=2, b=3): q flows into `from` and out of `to`.
  static constexpr int mosFrom[5] = {1, 1, 1, 3, 3}, mosTo[5] = {2, 0, 3, 0, 2};

  // Terminal currents (d, g, s, b) into a MOSFET at x: DC (mos1_currents) plus,
  // in transient, the currents of its charges integrated from the last accepted
  // step `st` (TRAP, else BE: ngspice NIintegrate). One definition for the Newton
  // stamp (T = Dual), the KCL audit and the history update (next != null).
  // ponytail: GEAR_2 (never chosen by runTransient) integrates the charges with BE.
  template <typename T>
  std::array<T, 4> mosCurrents(const std::array<T, 4> &x, const Mos1Params<double> &p,
                               const BlockState::DeviceCharge *st, double h,
                               BlockState::DeviceCharge *next = nullptr) const {
      const auto dc = mos1_currents(x[0], x[1], x[2], x[3], p);
      std::array<T, 4> I{dc.d, T(0.0), dc.s, dc.b};
      if (!p.caps || !st || !(h > 1e-15) || !integrator) return I;
      const auto c = mos1_charges(x[0], x[1], x[2], x[3], p, st->v[0], st->q, st->half);
      if constexpr (std::is_same_v<T, double>)
          if (next) {  // shift the voltage history (next may alias st)
              std::copy(st->v[1], st->v[1] + 5, next->v[2]);
              std::copy(st->v[0], st->v[0] + 5, next->v[1]);
          }
      const bool trap = integrator->getType() == IntegrationType::TRAPEZOIDAL;
      for (int k = 0; k < 5; ++k) {
          const T i = trap ? (2.0 / h) * (c.q[k] - st->q[k]) - st->i[k] : (c.q[k] - st->q[k]) / h;
          I[mosFrom[k]] += p.polarity * i;
          I[mosTo[k]] -= p.polarity * i;
          if constexpr (std::is_same_v<T, double>)
              if (next) { next->v[0][k] = c.v[k]; next->q[k] = c.q[k]; next->i[k] = i; }
      }
      if constexpr (std::is_same_v<T, double>)
          if (next) std::copy(c.half, c.half + 3, next->half);
      return I;
  }

  // Every MOSFET of the flattened netlist with its global nodes (d, g, s, b) and
  // charge history (null when the block state is not sized, e.g. GUI-built).
  template <class F> void forEachMosfet(const TensorNetlist &nl, F &&f) const {
      int total = 0;
      for (const auto &sl : branchSlots(nl, total))
          for (size_t i = 0; i < sl.blk->mosfets.size(); ++i) {
              const Mosfet &m = sl.blk->mosfets[i];
              f(m, std::array<int, 4>{sl.map(m.drain), sl.map(m.gate), sl.map(m.source), sl.map(m.body)},
                i < sl.st->mosfetState.size() ? &sl.st->mosfetState[i] : nullptr);
          }
  }

  // Newton stamp of mosCurrents() at v. The Jacobian comes from Dual (one pass
  // per terminal), so it is the exact derivative of the currents the KCL audit
  // checks: gm, gds, gmbs, the junctions and charges, either orientation.
  void stampMos1(const std::array<int, 4>& n, const Mos1Params<double>& p,
                 const BlockState::DeviceCharge* st, double h, const std::vector<double>& v) {
      stampByDual(n, 4, v, [&](const std::array<Dual<double>, 4> &x) { return mosCurrents(x, p, st, h); });
  }

  // Diodes and BJTs ("junction devices"): one evaluation of currents and charges
  // (diode_iq / bjt_gp) for the Newton stamp, both KCL audits, the history
  // update, AC and noise. Series resistances (diode RS; BJT RB, RC, RE) add an
  // internal node each, an extra unknown after the branch currents (BranchSlot::dev);
  // without the resistance the internal terminal is the external one.
  static int junctionInternals(const TensorBlock &blk) {
    int k = 0;
    for (const auto &d : blk.diodes) k += d.card.rs > 0.0;
    for (const auto &q : blk.bjts) k += (q.card.rb > 0.0) + (q.card.rc > 0.0) + (q.card.re > 0.0);
    return k;
  }

  // A junction device of the flattened netlist. n: its terminal unknowns, 1-based
  // (0 = ground; internal nodes > numGlobalNodes). Diode: A, C, A'. BJT: C, B, E,
  // S, C', B', E'. st: charge history (null when the block state is not sized).
  struct Junction {
    const Diode *d = nullptr;
    const BJT *q = nullptr;
    std::array<int, 7> n{};
    int terms = 0;
    int sub = 4;           // BJT: substrate junction's inner terminal, C' (vertical) or B' (lateral)
    double subSign = 1.0;  // BJT SUBS: substrate voltage type*SUBS*(S - that terminal)
    BlockState::DeviceCharge *st = nullptr;
  };
  template <class F> void forEachJunction(const TensorNetlist &nl, F &&f) const {
    int total = 0;
    for (const auto &sl : branchSlots(nl, total)) {
      int next = sl.dev + 1;  // 1-based unknown of the next internal node
      auto *st = const_cast<BlockState *>(sl.st);
      for (size_t i = 0; i < sl.blk->diodes.size(); ++i) {
        const Diode &d = sl.blk->diodes[i];
        Junction j;
        j.d = &d; j.terms = 3;
        j.n[0] = sl.map(d.anode); j.n[1] = sl.map(d.cathode);
        j.n[2] = d.card.rs > 0.0 ? next++ : j.n[0];
        j.st = i < st->diodeState.size() ? &st->diodeState[i] : nullptr;
        f(j);
      }
      for (size_t i = 0; i < sl.blk->bjts.size(); ++i) {
        const BJT &q = sl.blk->bjts[i];
        Junction j;
        j.q = &q; j.terms = 7;
        j.n[0] = sl.map(q.nodeCollector); j.n[1] = sl.map(q.base);
        j.n[2] = sl.map(q.emitter); j.n[3] = sl.map(q.substrate);
        j.n[4] = q.card.rc > 0.0 ? next++ : j.n[0];
        j.n[5] = q.card.rb > 0.0 ? next++ : j.n[1];
        j.n[6] = q.card.re > 0.0 ? next++ : j.n[2];
        if (q.card.lateral) { j.sub = 5; j.subSign = -1.0; }
        j.st = i < st->bjtState.size() ? &st->bjtState[i] : nullptr;
        f(j);
      }
    }
  }

  // Independent source value: its waveform in transient, else DC, times the
  // source-stepping factor (1 except while solveDC ramps the sources).
  double srcFact_ = 1.0;
  // A fatal OSDI model error during the current analysis (the model's own text):
  // it fails the analysis at once instead of retrying ever smaller steps.
  mutable std::string osdiFatal_;
  mutable std::string osdiNonFinite_;
  double ptranGmin_ = 0.0;  // node-to-ground gmin while solveDC runs its transient OP  // the last non-finite OSDI evaluation: reported if the analysis fails
  template <class S> double sourceValue(const S &s, double dc, double h, double time) const {
    return srcFact_ * (h > 1e-15 ? waveformValue(s, dc, time) : dc);
  }

  // OSDI devices (compiled Verilog-A: BSIM4, BSIM-CMG, ...). An instance's local
  // nodes are its terminals, then the internal nodes left after collapsing; the
  // internal nodes are unknowns after the junction internals (BranchSlot::dev).
  static int osdiInternals(const BlockState &st) {
    int k = 0;
    for (const auto &in : st.osdiState) k += in ? (int)in->numInternal() : 0;
    return k;
  }
  static int deviceInternals(const TensorBlock &blk, const BlockState &st) {
    return junctionInternals(blk) + osdiInternals(st);
  }
  struct OsdiRef {
    const OsdiDevice *dev;
    osdi::Instance *in;
    std::vector<int> n;  // 1-based unknown of each local node (0 = ground)
  };
  // osdiCurrents() of every OSDI device at v, evaluated in parallel (an instance
  // writes only its own data; model evaluation was ~half of a BSIM4 transient), in
  // forEachOsdi order: osdiRefs_[i], osdiI_[i] and, with jac, osdiJ_[i]. Callers
  // accumulate serially, so stamps and residuals keep their order.
  // The KCL audit at a Newton iterate and the next iteration's stamp evaluate at the
  // same point (OSDI devices have no voltage limiting), so the last evaluation is
  // reused when v (bitwise), h, the method and the devices match; an accepted step
  // moves the charge history and clears it. Jacobians are always kept for that.
  mutable std::vector<OsdiRef> osdiRefs_;
  mutable std::vector<std::vector<double>> osdiI_, osdiJ_;
  mutable std::vector<double> osdiAtV_;
  mutable double osdiAtH_ = -1.0;
  mutable int osdiAtMethod_ = -1;
  mutable bool osdiAtValid_ = false;
  void osdiAll(const TensorNetlist &nl, const std::vector<double> &v, double h, bool /*jac*/, bool accept) const {
    // The device list (with its node vectors) is rebuilt only when the devices change:
    // walking the instances and comparing pointers allocates nothing (the rebuild did,
    // per device per call: ~25 % of a BSIM4 chain transient).
    size_t k = 0;
    bool listed = true;
    int total = 0;
    for (const auto &sl : branchSlots(nl, total))
      for (size_t i = 0; i < sl.blk->osdiDevices.size() && i < sl.st->osdiState.size(); ++i) {
        const osdi::Instance *in = sl.st->osdiState[i].get();
        if (!in) continue;
        listed = listed && k < osdiRefs_.size() && osdiRefs_[k].dev == &sl.blk->osdiDevices[i] && osdiRefs_[k].in == in;
        ++k;
      }
    if (!listed || k != osdiRefs_.size()) {
      osdiRefs_.clear();
      forEachOsdi(nl, [&](const OsdiRef &r) { osdiRefs_.push_back(r); });
      osdiAtValid_ = false;
    }
    const int method = integrator ? (int)integrator->getType() : -1;
    if (osdiAtValid_ && !accept && h == osdiAtH_ && method == osdiAtMethod_ && v == osdiAtV_) return;
    const int n = (int)osdiRefs_.size();
    osdiI_.resize(n);
    osdiJ_.resize(n);
    parallelFor(n, 2e4 * n,  // ~5 us an evaluation: a few devices are worth the pool
                [&](int i) { osdiI_[i] = osdiCurrents(osdiRefs_[i], v, h, &osdiJ_[i], accept); });
    osdiAtValid_ = !accept;
    osdiAtV_ = v;
    osdiAtH_ = h;
    osdiAtMethod_ = method;
  }
  template <class F> void forEachOsdi(const TensorNetlist &nl, F &&f) const {
    int total = 0;
    for (const auto &sl : branchSlots(nl, total)) {
      int next = sl.dev + junctionInternals(*sl.blk) + 1;
      for (size_t i = 0; i < sl.blk->osdiDevices.size() && i < sl.st->osdiState.size(); ++i) {
        osdi::Instance *in = sl.st->osdiState[i].get();
        if (!in) continue;
        OsdiRef r{&sl.blk->osdiDevices[i], in, {}};
        for (int t : r.dev->nodes) r.n.push_back(sl.map(t));
        for (uint32_t k = 0; k < in->numInternal(); ++k) r.n.push_back(next++);
        f(r);
      }
    }
  }

  // Currents into an OSDI device's local nodes at the unknowns v: the resistive
  // residual plus, in transient, its charges' current integrated from the last
  // accepted step (TRAP, else BE, as mosCurrents). One definition for the stamp,
  // the KCL audit and the history update. J: dI/dx = G + coef*C (row-major).
  // accept: store this point as the accepted step. A fatal model error gives NaN.
  static std::mutex &osdiErrorMutex() { static std::mutex m; return m; }
  std::vector<double> osdiCurrents(const OsdiRef &r, const std::vector<double> &v, double h,
                                   std::vector<double> *J = nullptr, bool accept = false) const {
    const size_t L = r.n.size();
    std::vector<double> x(L), I(L);
    for (size_t k = 0; k < L; ++k) x[k] = r.n[k] > 0 && r.n[k] <= (int)v.size() ? v[r.n[k] - 1] : 0.0;
    const bool tran = h > 1e-15 && integrator;
    osdi::Instance &in = *r.in;
    const uint32_t flags = tran ? osdi::ANALYSIS_TRAN : osdi::ANALYSIS_DC | osdi::ANALYSIS_STATIC;
    // Device bypass (ngspice's, voltage criterion): a device whose local voltages moved
    // less than RELTOL*|v| + VNTOL (1e-3, 1 uV) since its last evaluation, of the same
    // analysis, is not evaluated; its stored f, q extrapolate linearly with G, C
    // (error O(dv^2)). 98 % of the evaluations of a 1000-stage BSIM4 chain transient
    // were such. Never for the Jacobian at an operating point itself (AC, noise,
    // sensitivity run with limitJunctions_ off). TREESPICE_BYPASS=0 evaluates always.
    static const bool bypassOn = [] { const char *s = std::getenv("TREESPICE_BYPASS"); return !s || std::atoi(s) != 0; }();
    bool bypass = bypassOn && limitJunctions_ && in.evalFlags == flags && in.evalAt.size() == L;
    for (size_t k = 0; bypass && k < L; ++k)
      bypass = std::abs(x[k] - in.evalAt[k]) <= 1e-3 * std::max(std::abs(x[k]), std::abs(in.evalAt[k])) + 1e-6;
    if (!bypass && !in.evaluate(x.data(), flags, 0.0)) {
      std::lock_guard<std::mutex> lock(osdiErrorMutex());  // devices evaluate in parallel
      osdiFatal_ = r.dev->instanceName + ": " + in.error();  // reported instead of a singular matrix
      std::fill(I.begin(), I.end(), NAN);
      if (J) J->assign(L * L, 0.0);
      return I;
    }
    std::vector<double> f(in.f()), q(in.q());  // at x: as evaluated, or extrapolated when bypassed
    if (bypass)
      for (size_t a = 0; a < L; ++a)
        for (size_t b = 0; b < L; ++b) {
          const double d = x[b] - in.evalAt[b];
          f[a] += in.G(a, b) * d;
          q[a] += in.C(a, b) * d;
        }
    for (size_t k = 0; k < L; ++k)
      if (!std::isfinite(f[k]) || !std::isfinite(q[k])) {
        // An extreme Newton iterate may do this and the homotopy recovers, so it
        // only explains a failed analysis (osdiNonFinite_), it does not end one.
        std::lock_guard<std::mutex> lock(osdiErrorMutex());
        osdiNonFinite_ = r.dev->instanceName + ": the model returned a non-finite current or charge "
                                               "(parameters outside its valid range?)";
        std::fill(I.begin(), I.end(), NAN);
        if (J) J->assign(L * L, 0.0);
        return I;
      }
    const bool trap = tran && integrator->getType() == IntegrationType::TRAPEZOIDAL;
    const double coef = !tran ? 0.0 : trap ? 2.0 / h : 1.0 / h;
    for (size_t k = 0; k < L; ++k) {
      const double iq = tran ? coef * (q[k] - in.qPrev[k]) - (trap ? in.iPrev[k] : 0.0) : 0.0;
      I[k] = f[k] + iq;
      if (accept) { in.qPrev[k] = q[k]; in.iPrev[k] = iq; }
    }
    if (accept) { in.vHist[2] = in.vHist[1]; in.vHist[1] = in.vHist[0]; in.vHist[0] = x; }
    if (J) {
      J->assign(L * L, 0.0);
      for (size_t a = 0; a < L; ++a)
        for (size_t b = 0; b < L; ++b) (*J)[a * L + b] = in.G(a, b) + coef * in.C(a, b);
    }
    // A branch-current unknown (V(a,b) <+ 0 branches, e.g. BSIM4's body network with
    // rbodymod off, which closes a loop of zero-volt branches) gets 1 uOhm in series,
    // so roundoff cannot drive a circulating current (1e-12 there gave 1e-4 A).
    // Internal nodes get nothing, as in ngspice: a floating one (an unconnected bulk)
    // is set by the model's own junction gmin, which a leak to ground would override.
    for (size_t k = in.terminals(); k < L; ++k)
      if (in.isFlow(k)) {
        I[k] += 1e-6 * x[k];
        if (J) (*J)[k * L + k] += 1e-6;
      }
    return I;
  }

  // Newton stamp of the currents I and Jacobian J of osdiCurrents() at v (a node
  // shared by two terminals sums).
  void stampOsdi(const OsdiRef &r, const std::vector<double> &I, const std::vector<double> &J,
                 const std::vector<double> &v) {
    const size_t L = r.n.size();
    auto V = [&](int g) { return g > 0 && g <= (int)v.size() ? v[g - 1] : 0.0; };
    for (size_t t = 0; t < L; ++t) {
      if (r.n[t] <= 0) continue;
      double ieq = I[t];
      for (size_t u = 0; u < L; ++u) {
        const double g = J[t * L + u];
        if (g == 0.0) continue;
        ieq -= g * V(r.n[u]);
        if (r.n[u] > 0) matrixBuilder.add(r.n[t] - 1, r.n[u] - 1, g);
      }
      rhsVector[r.n[t] - 1] -= ieq;
    }
  }

  // Charge branches (terminal pairs, q flows from -> to): diode A'->C; BJT
  // be (B'->E'), bc (B'->C'), bx (B->C'), substrate (S->C'; lateral: S->B', Junction::sub).
  static constexpr int diodeFrom[1] = {2}, diodeTo[1] = {1};
  static constexpr int bjtFrom[4] = {5, 5, 1, 3}, bjtTo[4] = {6, 4, 4, 4};

  // Branch voltages and charges (type domain) of j at terminal values x.
  // Junction-limiting shifts (junctionLimit): added to the junction voltages
  // (diode vd; BJT vbe, vbc) so the device is evaluated at its limited point.
  // seed: the shift carries derivative 1 (the G*(vl - v) correction pass).
  struct JShift { double d[2] = {0.0, 0.0}; bool seed = false; };
  template <typename T> static T shifted(const T &v, const JShift *sh, int k) {
    if (!sh || sh->d[k] == 0.0) return v;
    if constexpr (std::is_same_v<T, double>) return v + sh->d[k];
    else return v + T(sh->d[k], sh->seed ? sh->d[k] : 0.0);
  }

  template <typename T>
  int junctionCharges(const Junction &j, const std::array<T, 7> &x, double tempK, T *v, T *q,
                      const JShift *sh = nullptr) const {
    if (j.d) {
      v[0] = shifted(T(x[2] - x[1]), sh, 0);
      q[0] = diode_iq(v[0], diodeModel(*j.d, tempK)).second;
      return 1;
    }
    const auto p = bjtModel(*j.q, tempK);
    const double t = p.type;
    for (int k = 0; k < 3; ++k) v[k] = t * (x[bjtFrom[k]] - x[bjtTo[k]]);
    v[3] = t * j.subSign * (x[3] - x[j.sub]);
    v[0] = shifted(v[0], sh, 0); v[1] = shifted(v[1], sh, 1);
    const auto e = bjt_gp(v[0], v[1], v[2], v[3], p);
    for (int k = 0; k < 4; ++k) q[k] = e.q[k];
    return 4;
  }
  static bool junctionHasCharge(const Junction &j) {
    if (j.d) return j.d->card.cjo > 0.0 || j.d->card.tt > 0.0;
    const auto &c = j.q->card;
    return c.cje > 0.0 || c.cjc > 0.0 || c.cjs > 0.0 || c.tf > 0.0 || c.tr > 0.0;
  }

  // Terminal currents into j at x: DC plus, in transient, the charge currents
  // integrated from the last accepted step (TRAP, else BE), as mosCurrents().
  // next != null: write the accepted charge state (history update).
  template <typename T>
  std::array<T, 7> junctionCurrents(const Junction &j, const std::array<T, 7> &x, double tempK, double h,
                                    BlockState::DeviceCharge *next = nullptr, const JShift *sh = nullptr) const {
    std::array<T, 7> I{};
    for (auto &i : I) i = T(0.0);
    double type = 1.0;
    const int *from, *to;
    if (j.d) {
      const auto p = diodeModel(*j.d, tempK);
      if (p.rs > 0.0) { const T ir = (x[0] - x[2]) / p.rs; I[0] += ir; I[2] -= ir; }
      const T id = diode_iq(shifted(T(x[2] - x[1]), sh, 0), p).first;
      I[2] += id; I[1] -= id;
      from = diodeFrom; to = diodeTo;
    } else {
      const auto p = bjtModel(*j.q, tempK);
      type = p.type;
      auto res = [&](int a, int b, double r) {
        if (r > 0.0) { const T i = (x[a] - x[b]) / r; I[a] += i; I[b] -= i; }
      };
      res(0, 4, p.rc); res(1, 5, p.rb); res(2, 6, p.re);
      const auto e = bjt_gp(shifted(T(type * (x[5] - x[6])), sh, 0), shifted(T(type * (x[5] - x[4])), sh, 1),
                            T(type * (x[1] - x[4])), T(type * j.subSign * (x[3] - x[j.sub])), p);
      I[4] += type * e.cc; I[5] += type * e.cb; I[6] -= type * (e.cc + e.cb);
      I[3] += type * j.subSign * e.cs; I[j.sub] -= type * j.subSign * e.cs;  // substrate junction
      from = bjtFrom; to = bjtTo;
    }
    const BlockState::DeviceCharge *st = j.st;
    if (!st || !(h > 1e-15) || !integrator || !junctionHasCharge(j)) return I;
    T v[5], q[5];
    const int K = junctionCharges(j, x, tempK, v, q, sh);
    if constexpr (std::is_same_v<T, double>)
      if (next) {  // shift the voltage history (next may alias st)
        std::copy(st->v[1], st->v[1] + 5, next->v[2]);
        std::copy(st->v[0], st->v[0] + 5, next->v[1]);
      }
    const bool trap = integrator->getType() == IntegrationType::TRAPEZOIDAL;
    for (int k = 0; k < K; ++k) {
      const T i = trap ? (2.0 / h) * (q[k] - st->q[k]) - st->i[k] : (q[k] - st->q[k]) / h;
      const bool sub = j.q && k == 3;  // substrate branch: to Junction::sub, SUBS sign
      I[from[k]] += (sub ? j.subSign : 1.0) * type * i;
      I[sub ? j.sub : to[k]] -= (sub ? j.subSign : 1.0) * type * i;
      if constexpr (std::is_same_v<T, double>)
        if (next) { next->v[0][k] = v[k]; next->q[k] = q[k]; next->i[k] = i; }
    }
    return I;
  }

  // Newton stamp of f(x) (terminal currents) over the terminal unknowns n.
  // Terminals may share an unknown (an internal node without its resistance),
  // so derivatives are seeded per unknown, not per terminal.
  template <size_t NT, class F>
  void stampByDual(const std::array<int, NT> &n, int terms, const std::vector<double> &v, F &&f) {
    auto V = [&](int k) { return n[k] > 0 && n[k] <= (int)v.size() ? v[n[k] - 1] : 0.0; };
    std::array<double, NT> I{};
    bool any = false;
    for (int u = 0; u < terms; ++u) {
      if (n[u] <= 0) continue;
      bool seen = false;
      for (int w = 0; w < u; ++w) seen = seen || n[w] == n[u];
      if (seen) continue;
      any = true;
      std::array<Dual<double>, NT> x;
      for (int k = 0; k < (int)NT; ++k) x[k] = Dual<double>(V(k), k < terms && n[k] == n[u] ? 1.0 : 0.0);
      const auto cur = f(x);
      for (int t = 0; t < terms; ++t) {
        I[t] = cur[t].val;
        if (n[t] <= 0 || cur[t].grad == 0.0) continue;
        matrixBuilder.add(n[t] - 1, n[u] - 1, cur[t].grad);
        rhsVector[n[t] - 1] += cur[t].grad * V(u);  // -(ieq) = -(I - J V)
      }
    }
    if (!any) return;  // every terminal grounded
    for (int t = 0; t < terms; ++t)
      if (n[t] > 0) rhsVector[n[t] - 1] -= I[t];
  }

  // ngspice junction limiting (DEVpnjlim in dioload/bjtload), device-local: the
  // raw junction voltages at x are limited against the last iteration's
  // (DeviceCharge::jv) and the device is linearised there; node voltages are
  // left alone. First iteration of a solve: vcrit (BJT vbc = 0), as MODEINITJCT.
  JShift junctionLimit(const Junction &j, const std::vector<double> &x, double tempK) const {
    JShift sh;
    if (!j.st) return sh;
    auto V = [&](int k) { return j.n[k] > 0 && j.n[k] <= (int)x.size() ? x[j.n[k] - 1] : 0.0; };
    auto *st = j.st;
    if (j.d) {
      const auto m = diodeModel(*j.d, tempK);
      const double vcrit = compute_vcrit(m.vte, m.is), raw = V(2) - V(1);
      double vl = vcrit;
      if (st->jvState) {
        vl = (m.breakdown && raw < std::min(0.0, -m.xbv + 10.0 * m.vte))
                 ? -(pnjlim(-(raw + m.xbv), -(st->jv[0] + m.xbv), m.vte, vcrit) + m.xbv)
                 : pnjlim(raw, st->jv[0], m.vte, vcrit);
      }
      st->jv[0] = vl; st->jvState = 1;
      sh.d[0] = vl - raw;
      return sh;
    }
    const auto m = bjtModel(*j.q, tempK);
    const double vt = m.vtf / j.q->card.nf, vcrit = compute_vcrit(vt, m.is);
    const double raw[2] = {m.type * (V(5) - V(6)), m.type * (V(5) - V(4))};
    for (int k = 0; k < 2; ++k) {
      const double vl = st->jvState ? pnjlim(raw[k], st->jv[k], vt, vcrit) : (k == 0 ? vcrit : 0.0);
      st->jv[k] = vl;
      sh.d[k] = vl - raw[k];
    }
    st->jvState = 1;
    return sh;
  }
  // Limiting state from x (a Newton restart point) or back to MODEINITJCT.
  void syncJunctionLimits(const TensorNetlist &nl, const std::vector<double> &x) const {
    forEachJunction(nl, [&](const Junction &j) {
      if (!j.st) return;
      auto V = [&](int k) { return j.n[k] > 0 && j.n[k] <= (int)x.size() ? x[j.n[k] - 1] : 0.0; };
      const double t = j.q ? (j.q->isNPN ? 1.0 : -1.0) : 1.0;
      j.st->jv[0] = j.d ? V(2) - V(1) : t * (V(5) - V(6));
      j.st->jv[1] = j.d ? 0.0 : t * (V(5) - V(4));
      j.st->jvState = 1;
    });
  }
  void resetJunctionLimits(const TensorNetlist &nl) const {
    forEachJunction(nl, [&](const Junction &j) { if (j.st) j.st->jvState = 0; });
  }
  bool limitJunctions_ = true;  // off for the AC / sensitivity Jacobian at a converged point

  // Newton stamp of a junction device: I(x) ~ I(vl) + G (v(x) - vl) at the
  // limited junction voltages vl (ngspice's companion), i.e. stampByDual at the
  // shifted point plus the G*(vl - v) correction from one seeded pass.
  void stampJunction(const Junction &j, const std::vector<double> &v, double tempK, double h) {
    const JShift sh = limitJunctions_ ? junctionLimit(j, v, tempK) : JShift{};
    stampByDual(j.n, j.terms, v, [&](const std::array<Dual<double>, 7> &x) { return junctionCurrents(j, x, tempK, h, nullptr, &sh); });
    if (sh.d[0] == 0.0 && sh.d[1] == 0.0) return;
    JShift seeded = sh;
    seeded.seed = true;
    std::array<Dual<double>, 7> x;
    for (int k = 0; k < 7; ++k) x[k] = Dual<double>(j.n[k] > 0 && j.n[k] <= (int)v.size() ? v[j.n[k] - 1] : 0.0, 0.0);
    const auto cur = junctionCurrents(j, x, tempK, h, nullptr, &seeded);
    for (int t = 0; t < j.terms; ++t)
      if (j.n[t] > 0) rhsVector[j.n[t] - 1] += cur[t].grad;
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
  // Generators: resistor thermal 4kT/R (incl. diode RS, BJT RB/RC/RE), diode shot 2q|Id|,
  // BJT shot 2q|Ic| (C'-E') and 2q|Ib| (B'-E'), MOSFET/JFET channel thermal 4kT*(2/3)*gm (MOS1). Per
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

    // Both the update (RELTOL/VNTOL, as ngspice) and KCL must pass. A former
    // "KCL < 1 nA => converged" shortcut accepted stale voltages at
    // high-impedance nodes (a 555 output stage settled 0.5 V off, above VCC).

    // 1. Check Voltage Updates (Standard RELTOL/VNTOL)
    for (size_t i = 0; i < v_new.size(); ++i) {
      double diff = std::abs(v_new[i] - v_old[i]);
      double limit = RELTOL * std::max(std::abs(v_new[i]), std::abs(v_old[i])) + VNTOL;
      if (!(diff <= limit)) return false;
    }

    // 2. Check Physical KCL Conservation (Research-Grade)
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
    int vs, ind, e, h, dev;        // first matrix row of each branch kind; dev: device internal nodes
  };
  std::vector<BranchSlot> branchSlots(const TensorNetlist &nl, int &total) const {
    std::vector<BranchSlot> branchList;
    branchList.push_back({&nl.globalBlock, &nl.globalState, [](int x) { return x; }, 0, 0, 0, 0, 0});
    for (size_t k = 0; k < nl.instances.size() && k < nl.instanceStates.size(); ++k) {
      auto it = nl.blockDefinitions.find(nl.instances[k].blockName);
      if (it == nl.blockDefinitions.end()) continue;
      const auto *inst = &nl.instances[k];
      branchList.push_back({&it->second, &nl.instanceStates[k], [inst](int x) {
        return (x <= 0 || x > (int)inst->nodeMapping.size()) ? 0 : inst->nodeMapping[x - 1];
      }, 0, 0, 0, 0, 0});
    }
    int next = nl.numGlobalNodes;
    for (auto &s : branchList) { s.vs = next; next += (int)s.blk->voltageSources.size(); }
    for (auto &s : branchList) { s.ind = next; next += (int)s.blk->inductors.size(); }
    for (auto &s : branchList) { s.e = next; next += (int)s.blk->voltageControlledVoltageSources.size(); }
    for (auto &s : branchList) { s.h = next; next += (int)s.blk->currentControlledVoltageSources.size(); }
    for (auto &s : branchList) { s.dev = next; next += deviceInternals(*s.blk, *s.st); }
    total = next;
    return branchList;
  }
  int systemSize(const TensorNetlist &nl) const { int t = 0; branchSlots(nl, t); return t; }

  // Small-signal system at the last DC operating point: G (Newton Jacobian
  // without node gmin), reactive triplets C. Shared by solveAC and solveNoise.
  struct ACSystem { Csr_matrix G; std::vector<int> cr, cc; std::vector<double> cv; int n = 0; };
  bool acSystem(TensorNetlist &netlist, ACSystem &sys);
  // R(Y) = [[G, -wC], [wC, G]] for Y = G + jwC (transpose: R(Y)^T = R(Y^H)).
  static Csr_matrix acMatrix(const ACSystem &s, double w, bool transpose);
  // Excitation: the sources' AC mag/phase, or unit AC at source `unit` only
  // (empty vector if no source has that name).
  std::vector<std::complex<double>> acExcitation(const TensorNetlist &nl, const std::string &unit = "") const;

  // used[n]: some element (any block, mapped to global nodes) touches node n.
  // Recomputed per call, O(elements): a cache keyed on counts went stale when a
  // CircuitSim was reused on a differently wired netlist.
  std::vector<char> usedNodes(const TensorNetlist &nl) const {
    int total = 0;
    std::vector<char> used(nl.numGlobalNodes + 1, 0);
    for (const auto &sl : branchSlots(nl, total)) {
      auto mark = [&](std::initializer_list<int> ns) {
        for (int local : ns) { int g = sl.map(local); if (g > 0 && g <= nl.numGlobalNodes) used[g] = 1; }
      };
      const auto &b = *sl.blk;
      for (const auto &e : b.resistors) mark({e.nodeTerminal1, e.nodeTerminal2});
      for (const auto &e : b.capacitors) mark({e.nodePlate1, e.nodePlate2});
      for (const auto &e : b.inductors) mark({e.nodeCoil1, e.nodeCoil2});
      for (const auto &e : b.voltageSources) mark({e.nodePositive, e.nodeNegative});
      for (const auto &e : b.currentSources) mark({e.nodePositive, e.nodeNegative});
      for (const auto &e : b.diodes) mark({e.anode, e.cathode});
      for (const auto &e : b.schottkyDiodes) mark({e.anode, e.cathode});
      for (const auto &e : b.zenerDiodes) mark({e.anode, e.cathode});
      for (const auto &e : b.mosfets) mark({e.drain, e.gate, e.source, e.body});
      for (const auto &e : b.bjts) mark({e.nodeCollector, e.base, e.emitter, e.substrate});
      for (const auto &e : b.osdiDevices)
        for (int t : e.nodes) mark({t});
      for (const auto &e : b.jfets) mark({e.drain, e.gate, e.source});
      for (const auto &e : b.voltageControlledVoltageSources) mark({e.outPos, e.outNeg, e.ctrlPos, e.ctrlNeg});
      for (const auto &e : b.voltageControlledCurrentSources) mark({e.outPos, e.outNeg, e.ctrlPos, e.ctrlNeg});
      for (const auto &e : b.currentControlledVoltageSources) mark({e.outPos, e.outNeg});
      for (const auto &e : b.currentControlledCurrentSources) mark({e.outPos, e.outNeg});
      for (const auto &e : b.powerRails) mark({e.nodeRail});
      for (const auto &e : b.parasiticResistors) mark({e.node1, e.node2});
      for (const auto &e : b.parasiticCapacitors) mark({e.node1, e.node2});
    }
    return used;
  }

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
        S.rhs[b] += scale * sourceValue(v, v.voltage_V, tran ? h : 0.0, time);
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
      double is = sourceValue(cs, cs.current_A, h, time);
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

    // 3.55 MOSFETs, global block and every instance: the stamped mosCurrents()
    forEachMosfet(netlist, [&](const Mosfet &m, const std::array<int, 4> &t,
                               const BlockState::DeviceCharge *st) {
        const auto I = mosCurrents<double>({getV(t[0]), getV(t[1]), getV(t[2]), getV(t[3])},
                                           mos1Params(m), st, h);
        for (int k = 0; k < 4; ++k) addRes(t[k], -I[k]);
    });
    // 3.6 Diodes and BJTs, global block and every instance: the stamped
    //     junctionCurrents(), including their internal-node rows
    forEachJunction(netlist, [&](const Junction &j) {
        std::array<double, 7> x{};
        for (int k = 0; k < j.terms; ++k) x[k] = getV(j.n[k]);
        const auto I = junctionCurrents(j, x, netlist.environment.ambient_temp_K, h);
        for (int k = 0; k < j.terms; ++k)
          if (j.n[k] > 0 && j.n[k] <= (int)node_residuals.size()) node_residuals[j.n[k] - 1] -= I[k];
    });
    // 3.65 OSDI devices: the stamped osdiCurrents(), internal-node rows included
    osdiAll(netlist, v_candidate, h, false, false);
    for (size_t i = 0; i < osdiRefs_.size(); ++i) {
        const auto &r = osdiRefs_[i];
        for (size_t k = 0; k < r.n.size(); ++k)
          if (r.n[k] > 0 && r.n[k] <= (int)node_residuals.size()) node_residuals[r.n[k] - 1] -= osdiI_[i][k];
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
          double is = sourceValue(cs, cs.current_A, h, time);
          addRes(mapNode(cs.nodePositive), -is);
          addRes(mapNode(cs.nodeNegative), is);
        }

        // Dependent Sources: G only (E/F/H are branch elements)
        for (const auto &vccs : block.voltageControlledCurrentSources) {
            double v_ctrl = getV(mapNode(vccs.ctrlPos)) - getV(mapNode(vccs.ctrlNeg));
            double i_vccs = vccs.gm * v_ctrl;
            addRes(mapNode(vccs.outPos), -i_vccs);
            addRes(mapNode(vccs.outNeg), i_vccs);
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
      const double is = sourceValue(cs, cs.current_A, 0.0, 0.0);
      addRes(cs.nodePositive, -is);
      addRes(cs.nodeNegative, is);
      attributions.push_back({cs.type.empty() ? "I?" : cs.type, "CurrentSource",
          {{cs.nodePositive, -is}, {cs.nodeNegative, is}}});
    }


    // 3. Diodes and BJTs (every block): DC junctionCurrents() at the external
    //    nodes (internal-node rows are not node KCL rows here)
    forEachJunction(netlist, [&](const Junction &j) {
      std::array<double, 7> x{};
      for (int k = 0; k < j.terms; ++k) x[k] = getV(j.n[k]);
      const auto I = junctionCurrents(j, x, netlist.environment.ambient_temp_K, 0.0);
      DeviceAttribution a{j.d ? j.d->instanceName : j.q->instanceName, j.d ? "Diode" : "BJT", {}};
      for (int k = 0; k < j.terms; ++k) {
        addRes(j.n[k], -I[k]);
        if (j.n[k] > 0 && j.n[k] <= n) a.nodeCurrents.push_back({j.n[k], -I[k]});
      }
      attributions.push_back(std::move(a));
    });
    // OSDI devices, likewise (DC, external nodes)
    osdiAll(netlist, v_candidate, 0.0, false, false);
    for (size_t i = 0; i < osdiRefs_.size(); ++i) {
      const auto &r = osdiRefs_[i];
      const auto &I = osdiI_[i];
      DeviceAttribution a{r.dev->instanceName, "OSDI", {}};
      for (size_t k = 0; k < r.dev->nodes.size(); ++k) {
        addRes(r.n[k], -I[k]);
        if (r.n[k] > 0 && r.n[k] <= n) a.nodeCurrents.push_back({r.n[k], -I[k]});
      }
      attributions.push_back(std::move(a));
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

    // MOSFETs
    for (const auto &m : netlist.globalBlock.mosfets) {
      const auto c = mos1_currents(getV(m.drain), getV(m.gate), getV(m.source), getV(m.body),
                                   mos1Params(m));
      addRes(m.drain, -c.d);
      addRes(m.source, -c.s);
      addRes(m.body, -c.b);
      attributions.push_back({m.instanceName, "MOSFET",
          {{m.drain, -c.d}, {m.source, -c.s}, {m.body, -c.b}}});
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
      limitJunctions_ = false;  // linearise at the converged point itself
      stampAllElements(netlist, 0.0, x_converged, 0.0); // h=0, time=0 for DC sensitivity
      limitJunctions_ = true;
      // (no node gmin: the Jacobian of the converged OP has only junction GMIN)
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
          else if (type == ComponentType::DIODE || type == ComponentType::BJT) {
              // dI/dp of the same junctionCurrents() the solve used, by central
              // difference on the instance's parameter (global-block devices).
              // ponytail: finite difference (relative step 1e-6); Dual parameters if precision matters.
              const bool isD = type == ComponentType::DIODE;
              const size_t count = isD ? netlist.globalBlock.diodes.size() : netlist.globalBlock.bjts.size();
              if (index < 0 || index >= (int)count) return {};
              Diode d; BJT q;
              double *par = nullptr;
              const void *orig = nullptr;
              if (isD) {
                  d = netlist.globalBlock.diodes[index]; orig = &netlist.globalBlock.diodes[index];
                  par = paramName == "Is" ? &d.saturationCurrent_I_S_A : paramName == "N" ? &d.emissionCoefficient_N
                      : paramName == "Vt" ? &d.thermalVoltage_V_T_V : nullptr;
              } else {
                  q = netlist.globalBlock.bjts[index]; orig = &netlist.globalBlock.bjts[index];
                  par = paramName == "Is" ? &q.saturationCurrent_I_S_A
                      : (paramName == "BetaF" || paramName == "Beta") ? &q.betaF
                      : paramName == "BetaR" ? &q.betaR : paramName == "Vt" ? &q.thermalVoltage_V_T_V : nullptr;
              }
              if (!par) return {};
              Junction jj;
              forEachJunction(netlist, [&](const Junction &j) { if ((const void *)(isD ? (const void *)j.d : (const void *)j.q) == orig) jj = j; });
              jj.d = isD ? &d : nullptr; jj.q = isD ? nullptr : &q; jj.st = nullptr;
              std::array<double, 7> x{};
              for (int k = 0; k < jj.terms; ++k) x[k] = getV(jj.n[k]);
              const double p0 = *par, dp = 1e-6 * std::max(std::abs(p0), 1e-30);
              *par = p0 + dp; const auto Ip = junctionCurrents(jj, x, netlist.environment.ambient_temp_K, 0.0);
              *par = p0 - dp; const auto Im = junctionCurrents(jj, x, netlist.environment.ambient_temp_K, 0.0);
              for (int k = 0; k < jj.terms; ++k)
                  if (jj.n[k] > 0 && jj.n[k] <= (int)b_sens.size()) b_sens[jj.n[k] - 1] -= (Ip[k] - Im[k]) / (2.0 * dp);
          }
          else if (type == ComponentType::MOSFET) {
              if (index < 0 || index >= (int)netlist.globalBlock.mosfets.size()) return {};
              const auto& m = netlist.globalBlock.mosfets[index];

              // dI/dp of the same mos1_currents() the solve used: Vth (VTO), W, L
              const Mos1Params<double> p = mos1Params(m);
              Mos1Params<Dual<double>> pd{p.polarity, p.beta, p.vto, p.lambda, p.gamma, p.phi, p.isat, p.vt};
              const double leff = m.l - 2.0 * m.card.ld;
              if (paramName == "Vth") pd.vto.grad = 1.0;
              if (paramName == "W") pd.beta.grad = p.beta / m.w;
              if (paramName == "L") pd.beta.grad = -p.beta / leff;
              const auto c = mos1_currents(Dual<double>(getV(m.drain)), Dual<double>(getV(m.gate)),
                                           Dual<double>(getV(m.source)), Dual<double>(getV(m.body)), pd);
              if (m.drain > 0) b_sens[m.drain-1] -= c.d.grad;
              if (m.source > 0) b_sens[m.source-1] -= c.s.grad;
              if (m.body > 0) b_sens[m.body-1] -= c.b.grad;
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
      double val = sourceValue(i, i.current_A, h, time);

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
        double I = srcFact_ * block.currentSources.currents[i];  // (refreshed to the waveform in transient)
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
    // Diodes, BJTs and MOSFETs: stampAllElements() (stampByDual, both paths)


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

    // 1.5 Schottky Diodes (Reuse Unified Diode Logic)
    for (const auto &sd : block.schottkyDiodes) {
      int nA = mapNode(sd.anode);
      int nC = mapNode(sd.cathode);
      stampSchottky(sd, nA, nC, getV(nA) - getV(nC), temp_K, matrixBuilder, rhsVector);
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

    // Branch-current elements (V, L+K, E, F, H), MOSFETs, diodes and BJTs: one implementation for both paths
    stampBranchSystem(netlist, h, time);
    forEachMosfet(netlist, [&](const Mosfet &m, const std::array<int, 4> &n,
                               const BlockState::DeviceCharge *st) { stampMos1(n, mos1Params(m), st, h, v_guess); });
    const double tempK = netlist.environment.ambient_temp_K;
    forEachJunction(netlist, [&](const Junction &j) { stampJunction(j, v_guess, tempK, h); });
    osdiAll(netlist, v_guess, h, true, false);  // stamped in device order: the same sequence every iteration
    for (size_t i = 0; i < osdiRefs_.size(); ++i) stampOsdi(osdiRefs_[i], osdiI_[i], osdiJ_[i], v_guess);

    // A node no element touches is not a circuit node (ngspice drops it after
    // flattening, e.g. an X pin wired to a port the .subckt never uses): pin it
    // to 0 V. A real floating node (series caps, an undriven gate) still fails.
    // ponytail: skipped when D2A ports exist (they stamp nodes of their own).
    if (!digitalEngine_.hasOutputPorts()) {
      const std::vector<char> used = usedNodes(netlist);
      for (int n = 1; n <= netlist.numGlobalNodes; ++n)
        if (!used[n]) matrixBuilder.add(n - 1, n - 1, 1.0);
    }

    // ─── Gap 2: Digital Event Engine — D2A Stamping ──────────────────────
    if (digitalEngine_.hasOutputPorts()) {
        digitalEngine_.stampD2A(matrixBuilder, rhsVector, netlist.numGlobalNodes, h);
    }
  }


  void addDiagonalConditioning(int n, double gmin) {
    // gmin <= 0: no node-to-ground conditioning. Like ngspice, once converged
    // the only gmin is the 1e-12 S across each junction (device models); a
    // node-to-ground leak settles floating junction nodes differently
    // (aac_10_bridge). Mid-ladder DC steps still pass gmin > 0 for conditioning.
    if (gmin <= 0.0) return;
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
