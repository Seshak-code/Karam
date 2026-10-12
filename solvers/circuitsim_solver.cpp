/**
 * circuitsim_solver.cpp
 *
 * Implementation of CircuitSim solver methods:
 * - solveDC: DC Operating Point Analysis (Newton-Raphson)
 * - stepTransient: Transient Analysis (Adaptive Time-Stepping)
 *
 * Extracted from circuitsim.h for modularity.
 *
 * Phase 1 Hardening: Refactored for NASA-style safety.
 */

#include <sstream>

#include "../infrastructure/adaptive_precision.h"
#include "../infrastructure/compiled_block.h"
#include "../infrastructure/mpo_builder.h"
#include "../infrastructure/safe_utils.h"
#include "../physics/circuitsim.h"
#include "../physics/device_physics.h"
#include "../physics/physics_constants.h"
#include "../solvers/contraction_executor.h"
#include "../solvers/convergence_engine.h"
#include "../solvers/schur_solver.h"
#include "acutesim_engine/netlist/netlist_export.h"
#if !defined(__EMSCRIPTEN__) && defined(ACUTESIM_HAS_DAWN_NATIVE)
#include "../solvers/webgpu_solver.h"
#include "acutesim_engine/gpu_context_manager.h"
#endif

// Matches SPICE3's dt_init = tstep/100: divides the first transient step into
// 100 internal Backward Euler micro-steps for cold-start accuracy.
static constexpr int COLD_START_SUBSTEPS = 100;

// =============================================================================
// PHASE 1 HARDENING: Helper Method Implementations
// =============================================================================

void CircuitSim::seedInitialGuess(const TensorNetlist &netlist,
                                  std::vector<double> &voltageEstimate) {
  const size_t N = voltageEstimate.size();

  // Temperature-corrected Vbe seed: at cold temps, Vbe increases (~+2mV/K below
  // nominal)
  double T = netlist.environment.ambient_temp_K;
  double vbe_seed = PhysicsConstants::V_BE_SEED - 0.002 * (T - 300.15);
  // Clamp to reasonable range [0.5V, 1.2V]
  vbe_seed = std::max(0.5, std::min(1.2, vbe_seed));

  // 1. Seed from Voltage Sources (direct)
  for (const auto &source : netlist.globalBlock.voltageSources) {
    int idx = SafeUtils::safe_idx(source.nodePositive, N, "VSource seed");
    if (idx >= 0) {
      voltageEstimate[idx] = source.voltage_V;
    }
  }

  // 2. Seed Diode anodes (0.6V initial for silicon)
  for (const auto &diode : netlist.globalBlock.diodes) {
    int idx = SafeUtils::safe_idx(diode.anode, N, "Diode seed");
    if (idx >= 0) {
      voltageEstimate[idx] =
          std::max(voltageEstimate[idx], PhysicsConstants::V_DIODE_SEED);
    }
  }

  // 3. Find supply voltage for mid-rail seeding
  double vSupply = PhysicsConstants::V_SUPPLY_DEFAULT;
  for (const auto &vs : netlist.globalBlock.voltageSources) {
    if (vs.voltage_V > vSupply)
      vSupply = vs.voltage_V;
  }

  // 4. Current Source Node Seeding (BJT base detection)
  for (const auto &cs : netlist.globalBlock.currentSources) {
    int nPos = SafeUtils::safe_idx(cs.nodePositive, N, "ISource seed");
    if (nPos < 0)
      continue;

    // Check if this node is a BJT base
    bool isBase = false;
    for (const auto &bjt : netlist.globalBlock.bjts) {
      if (SafeUtils::safe_idx(bjt.base, N, "BJT base check") == nPos) {
        int nE = SafeUtils::safe_idx(bjt.emitter, N, "BJT emitter");
        double v_e = (nE >= 0) ? voltageEstimate[nE] : 0.0;
        double vbe = bjt.isNPN ? vbe_seed : -vbe_seed;
        voltageEstimate[nPos] = std::max(voltageEstimate[nPos], v_e + vbe);
        isBase = true;
        break;
      }
    }
    // Default: mid-rail for unconnected current source nodes
    if (!isBase && voltageEstimate[nPos] == 0.0) {
      voltageEstimate[nPos] = vSupply * 0.5;
    }
  }

  // 5. BJT Multi-Pass Seeding (Darlington propagation) - Flattened
  for (int pass = 0; pass < 3; ++pass) {
    for (size_t i = 0; i < soaBlock.bjts.size(); ++i) {
      int nB = soaBlock.bjts.bases[i] - 1;
      int nE = soaBlock.bjts.emitters[i] - 1;
      int nC = soaBlock.bjts.collectors[i] - 1;

      if (soaBlock.bjts.isNPN[i]) {
        if (nE >= 0 && nB >= 0) {
          voltageEstimate[nB] =
              std::max(voltageEstimate[nB], voltageEstimate[nE] + vbe_seed);
        }
        if (nC >= 0 && nB >= 0) {
          voltageEstimate[nC] =
              std::max(voltageEstimate[nC], voltageEstimate[nB] + 0.5);
        }
      } else { // PNP
        if (nE >= 0 && nB >= 0) {
          voltageEstimate[nB] =
              std::min(voltageEstimate[nB], voltageEstimate[nE] - vbe_seed);
        }
        if (nC >= 0 && nB >= 0) {
          voltageEstimate[nC] =
              std::min(voltageEstimate[nC], voltageEstimate[nB] - 0.5);
        }
      }
    }
  }
}

void CircuitSim::applyPnJunctionLimits(const TensorizedBlock &block,
                                       std::vector<double> &v_new,
                                       const std::vector<double> &v_old,
                                       double temp_K) {
  if (v_new.empty())
    return;
  const size_t N = v_new.size();
  const double temp_scale = temp_K / 300.15;

  // Helper lambda for limiting a single junction
  auto limitJunction = [&](int p_node, int n_node, double Is, double Vt) {
    // Scale Vt and Is for temperature
    Vt *= temp_scale;
    Is *= std::pow(temp_scale, 3.0) * exp(1.11 / (Vt / temp_scale) - 1.11 / Vt);
    int p = p_node - 1; // 1-based to 0-based
    int n = n_node - 1; // 1-based to 0-based

    if (p < 0 && n < 0)
      return;
    if (p >= (int)N || n >= (int)N)
      return;

    double vcrit = compute_vcrit(Vt, Is);
    double v_junction_old =
        (p >= 0 ? v_old[p] : 0.0) - (n >= 0 ? v_old[n] : 0.0);
    double v_junction_new =
        (p >= 0 ? v_new[p] : 0.0) - (n >= 0 ? v_new[n] : 0.0);

    double v_limited = pnjlim(v_junction_new, v_junction_old, Vt, vcrit);
    double corr = v_junction_new - v_limited;

    if (std::abs(corr) > 1e-12) {
      if (p >= 0 && n >= 0) {
        v_new[p] -= corr * 0.5;
        v_new[n] += corr * 0.5;
      } else if (p >= 0) {
        v_new[p] -= corr;
      } else if (n >= 0) {
        v_new[n] += corr;
      }
    }
  };

  // 1. Diode Limiting (Flattened)
  for (size_t i = 0; i < block.diodes.size(); ++i) {
    limitJunction(block.diodes.node_a[i], block.diodes.node_c[i],
                  block.diodes.Is[i], block.diodes.Vt[i]);
  }

  // 2. BJT Junction Limiting (Flattened)
  for (size_t i = 0; i < block.bjts.size(); ++i) {
    limitJunction(block.bjts.bases[i], block.bjts.emitters[i], block.bjts.Is[i],
                  block.bjts.Vt[i]);
    limitJunction(block.bjts.bases[i], block.bjts.collectors[i],
                  block.bjts.Is[i], block.bjts.Vt[i]);
  }
}

// =============================================================================
// MAIN SOLVER METHODS
// =============================================================================

SolverStep CircuitSim::solveDC(TensorNetlist &netlist) {
  if (netlist.numGlobalNodes == 0)
    return {0.0, {}, {0, 0.0, true}, 0};

  // GPU mode currently solves on the CPU path: the former WebGPU "hybrid" loop
  // stamped and solved on the CPU anyway with its own copy of the device
  // physics (voltage-only, no branch currents). Phase 4 replaces it with GPU
  // device evaluation + batched front contraction.

  // Ensure topology hash is fresh
  netlist.globalBlock.computeTopologyHash();

  // Initialize SoA Block for this solve
  soaBlock = tensorizeNetlist(netlist);

  // Unknowns: node voltages then MNA branch currents (V, L, E, H)
  const int nTot = systemSize(netlist);
  std::vector<double> voltageEstimate(nTot, 0.0);

  // Gap 4: Neural seeder — use dV/dt extrapolation or EMA history if available.
  // The heuristic seedInitialGuess() always runs on top for supply rails / PN
  // junctions.
  if (!prevVoltages_.empty() &&
      prevVoltages_.size() == voltageEstimate.size()) {
    // dV/dt extrapolation from last two converged solutions (DC: dt=1.0)
    auto seed = neuralSeeder_.predict(prevVoltages_, prevVoltages_, 1.0);
    if (!seed.empty()) {
      voltageEstimate = std::move(seed);
    }
  } else if (neuralSeeder_.hasHistory()) {
    auto seed = neuralSeeder_.predictSeed(voltageEstimate.size());
    if (!seed.empty()) {
      voltageEstimate = std::move(seed);
    }
  }

  // Phase 1 Hardening: Use extracted helper for initial guess seeding
  // Runs on top of neural seed to enforce supply rail / PN junction constraints
  seedInitialGuess(netlist, voltageEstimate);

  bool converged = false;
  int total_iterations = 0;
  double final_kcl_error = 0.0;
  bool singularMatrixDetected = false;
  std::vector<ConvergenceStep> convergenceHistory;
  // Phase 2.9: Numerical Intelligence state
  bool stagnationDetected = false;
  bool divergenceDetected = false;
  bool rankDeficientDetected = false;
  int rankDeficientRow = -1;
  double smallestPivotOverall = 1e300;
  RegularizationInfo regulInfo;

  // GMIN Stepping: Start with large parallel conductances, reduce each pass
  // This prevents singular matrices for cold-start BJT circuits
  // Use half-decade steps for better convergence at extreme temperatures
  double gmin_factor = 1.0;
  const int GMIN_STEPS = 19; // 1e-3 to 1e-12 in half-decade steps

  // SPICE order: plain Newton at nominal GMIN first; the GMIN ladder runs only
  // if that fails. (The ladder used to run unconditionally: ~19 full solves
  // even for a linear circuit, which dominated large-circuit DC time.)
  const std::vector<double> seededGuess = voltageEstimate;
  std::vector<int> gminSchedule{GMIN_STEPS - 1};
  for (size_t si = 0; si < gminSchedule.size(); ++si) {
    const int gmin_step = gminSchedule[si];
    // Scale GMIN: start at 1e-3, reduce by sqrt(10) each step to 1e-12
    gmin_factor = std::pow(10.0, -(3.0 + gmin_step * 0.5));
    if (gmin_factor < 1e-12)
      gmin_factor = 1e-12;
    bool gmin_converged = false;

    // Phase 2.9: Track regularization
    if (gmin_factor > 1e-12) {
      regulInfo.applied = true;
      regulInfo.gminValue = gmin_factor;
      regulInfo.injections = static_cast<uint32_t>(netlist.numGlobalNodes);
    }

    int iteration = 0; // Declare outside loop for scope access
    for (; iteration < MAX_NR_ITER; ++iteration) {
      // Unified Stamping (Phase 1: Structural Abstraction)
      scheduler.schedule(netlist);
      matrixBuilder.clear();
      matrixBuilder.setDimensions(nTot, nTot);
      rhsVector.assign(nTot, 0.0);

      stampAllElements(netlist, 0.0, voltageEstimate, 0.0);
      addDiagonalConditioning(netlist.numGlobalNodes, gmin_factor);

      // Sparse Merge (D5): Use cached pattern for O(N) assembly on iterations >
      // 0
      Csr_matrix matrix;
      bool patternValid = false;
      if (cachedPattern_ && cachedPattern_->rows == nTot)
        matrix = matrixBuilder.assembleFast(*cachedPattern_, &patternValid);
      if (!patternValid) {
        cachedPattern_ =
            std::make_unique<CachedCsrPattern>(matrixBuilder.buildPattern());
        matrix = matrixBuilder.createCsr();

        // Topology changed: drop history tied to the old unknown vector
        prevVoltages_.clear();
        // Gap 4: Reset seeder history on topology change
        neuralSeeder_.reset();
      }

      // Tensor-contraction direct solver for every circuit (the legacy
      // treewidth / MPO / Schur routes solved a voltage-only system)
      SolverResult result = solveLinear(matrix, rhsVector);
      if (iteration == 0 && si == 0) {
        std::cerr << "[TN-ROUTE] CONTRACTION (unknowns=" << nTot
                  << ", branches=" << nTot - netlist.numGlobalNodes
                  << ", fronts=" << contractionPlan_->fronts.size()
                  << ", levels=" << contractionPlan_->levels.size() << ")\n";
      }
      // Phase 2.9: Capture pivot diagnostics from LU
      if (result.smallestPivot < smallestPivotOverall) {
        smallestPivotOverall = result.smallestPivot;
        rankDeficientRow = result.rankDeficientRow;
      }
      if (result.rankDeficient && !rankDeficientDetected) {
        rankDeficientDetected = true;
        std::cerr << "DC Solver: Near-rank-deficient matrix (pivot="
                  << result.smallestPivot
                  << " at row=" << result.rankDeficientRow
                  << ") — proceeding with GMIN conditioning.\n";
      }

      if (!result.converged) {
        // Matrix is singular - circuit topology error or floating node
        std::cerr
            << "DC Solver: Linear solver failed (Singular Matrix) at iter "
            << iteration << "\n";
        final_kcl_error = 1e9; // Flag as bad residual
        singularMatrixDetected = true;
        // If singular, retrying with same Jacobian won't help. Abort this GMIN
        // step.
        total_iterations += iteration + 1;
        break;
      }

      std::vector<double> v_new = result.solution;

      // NaN Protection using SafeUtils
      SafeUtils::sanitize_vector(v_new, 0.0);

      // Phase 1 Hardening: Use extracted helper for PN junction limiting
      applyPnJunctionLimits(soaBlock, v_new, voltageEstimate,
                            netlist.environment.ambient_temp_K);

      // Check 1 + 2: Voltage Convergence AND Physical KCL Accuracy
      double kcl_error = kclExcess(netlist, v_new, matrix);

      // Limit Cycle Breaker: Damping
      // If residual is not improving, dampen the step to force a different path
      if (iteration > 0 &&
          kcl_error >= final_kcl_error) { // final_kcl_error holds prev_error
        // Apply damping: v_new = v_old + 0.5 * (v_new - v_old)
        for (size_t i = 0; i < v_new.size(); ++i) {
          v_new[i] = voltageEstimate[i] + 0.5 * (v_new[i] - voltageEstimate[i]);
        }
        // Re-evaluate residual after damping
        kcl_error = kclExcess(netlist, v_new, matrix);

        // If still bad, dampen again (aggressive backoff)
        if (kcl_error >= final_kcl_error) {
          for (size_t i = 0; i < v_new.size(); ++i) {
            v_new[i] =
                voltageEstimate[i] + 0.5 * (v_new[i] - voltageEstimate[i]);
          }
          kcl_error = kclExcess(netlist, v_new, matrix);
        }
      }
      // Record convergence step for diagnostics
      {
        ConvergenceStep cs;
        cs.iteration = iteration;
        cs.residualNorm = kcl_error;
        cs.dampingFactor =
            (iteration > 0 && kcl_error >= final_kcl_error) ? 0.5 : 1.0;
        cs.linearSolverResidual = result.finalResidual; // Phase 2.9

        // Compute max voltage delta
        double maxDelta = 0.0;
        for (size_t i = 0; i < v_new.size(); ++i) {
          double d = std::abs(v_new[i] - voltageEstimate[i]);
          if (d > maxDelta)
            maxDelta = d;
        }
        cs.maxVoltageDelta = maxDelta;
        cs.worstNode = ConvergenceEngine::findWorstNode(std::vector<double>(
            v_new.size(), kcl_error)); // Simplified; worst node from residual
        convergenceHistory.push_back(cs);
      }

      // Phase 2.9: Stagnation and divergence detection
      {
        // Stagnation: residual plateaued but not converged
        if (iteration > 0 && std::abs(final_kcl_error - kcl_error) < 1e-15 &&
            kcl_error > PhysicsConstants::KCL_TOL) {
          stagnationDetected = true;
        }
        // Divergence: last 3 iterations show monotonic increase
        size_t sz = convergenceHistory.size();
        if (sz >= 3) {
          if (convergenceHistory[sz - 1].residualNorm >
                  convergenceHistory[sz - 2].residualNorm &&
              convergenceHistory[sz - 2].residualNorm >
                  convergenceHistory[sz - 3].residualNorm) {
            divergenceDetected = true;
          }
        }
      }

      final_kcl_error = kcl_error; // Store for next iteration comparison

      // During GMIN stepping, we relax the KCL check because GMIN itself adds
      // residual. We only enforce strict KCL at the final (nominal GMIN) step.
      double effective_kcl_tol =
          (gmin_step == GMIN_STEPS - 1) ? PhysicsConstants::KCL_TOL : 1.0;

      if (checkConvergence(v_new, voltageEstimate, kcl_error,
                           effective_kcl_tol)) {
        voltageEstimate = v_new;
        gmin_converged = true;
        total_iterations += iteration + 1;
        break;
      }

      voltageEstimate = v_new;
    }

    // If we finished the loop without converging, accumulate iterations
    if (!gmin_converged) {
      total_iterations += iteration + 1;
    }

    // Only the final GMIN step should mark the overall simulation as converged
    if (gmin_step == GMIN_STEPS - 1 && gmin_converged) {
      converged = true;
      break;
    }

    // Direct attempt failed: restart from the seed and walk the full ladder
    if (si == 0) {
      voltageEstimate = seededGuess;
      stagnationDetected = divergenceDetected = false;
      for (int g = 0; g < GMIN_STEPS; ++g) gminSchedule.push_back(g);
      continue;
    }

    // If we failed to converge a single GMIN step, check if we're making
    // progress. If residual is within a reasonable range, continue stepping as
    // the current solution may be a good starting point for the next (tighter)
    // GMIN step.
    if (!gmin_converged) {
      if (final_kcl_error < 1.0) {
        // Making progress — continue GMIN stepping
        std::cerr << "  -> GMIN step did not fully converge (res="
                  << final_kcl_error << "), continuing...\n";
      } else {
        break;
      }
    }

  } // End GMIN stepping loop

  // Phase 2.9: Final per-node residual for RMS + worst-node tracking
  std::vector<double> perNodeResiduals;
  final_kcl_error =
      calculatePhysicalResiduals(netlist, voltageEstimate, perNodeResiduals);
  perNodeResiduals.resize(netlist.numGlobalNodes);  // worst NODE (not branch row)

  // Compute RMS and worst node
  double rmsResidual = 0.0;
  uint32_t worstNodeIndex = 0;
  double worstNodeVoltage = 0.0;
  if (!perNodeResiduals.empty()) {
    double sumSq = 0.0;
    double maxAbs = 0.0;
    for (size_t i = 0; i < perNodeResiduals.size(); ++i) {
      double v = std::abs(perNodeResiduals[i]);
      sumSq += v * v;
      if (v > maxAbs) {
        maxAbs = v;
        worstNodeIndex = static_cast<uint32_t>(i + 1); // 1-based
        worstNodeVoltage =
            (i < voltageEstimate.size()) ? voltageEstimate[i] : 0.0;
      }
    }
    rmsResidual =
        std::sqrt(sumSq / static_cast<double>(perNodeResiduals.size()));
  }

  std::string details;
  if (final_kcl_error > 1.0 || !converged) {
    details = auditPhysicalResiduals(netlist, voltageEstimate);
  }

  // NR Convergence Diagnostics
  ConvergenceFailureType failureType = ConvergenceEngine::classifyFailure(
      convergenceHistory, singularMatrixDetected);

  std::vector<ResidualContribution> worstDevices;
  int worstNode = 0;
  if (!converged) {
    calculatePhysicalResidualsDetailed(netlist, voltageEstimate, worstDevices,
                                       worstNode);
  }

  std::string justification = ConvergenceEngine::generateJustification(
      failureType, convergenceHistory, worstDevices);

  // Phase 1.7.9 + Phase 2.9: Build trust-enriched SolverStats
  SolverStats stats;
  stats.iterations = total_iterations;
  stats.residual = final_kcl_error;
  stats.converged = converged;
  stats.error_detail = details;
  stats.method = "Newton-Raphson";
  stats.integrationMethod = ""; // DC has no integration
  stats.gminSteppingUsed =
      (gmin_factor < 1e-3);         // If we progressed past initial steps
  stats.sourceSteppingUsed = false; // Not implemented yet
  stats.failureType = failureType;
  stats.convergenceHistory = std::move(convergenceHistory);
  stats.worstDevices = std::move(worstDevices);
  stats.convergenceJustification = justification;
  stats.convergenceTolerance = 1e-6;
  // Phase 2.9: Numerical Intelligence fields
  stats.rmsResidual = rmsResidual;
  stats.worstNodeIndex = worstNodeIndex;
  stats.worstNodeVoltage = worstNodeVoltage;
  stats.stagnationDetected = stagnationDetected;
  stats.divergenceDetected = divergenceDetected;
  stats.rankDeficient = rankDeficientDetected || singularMatrixDetected;
  stats.rankDeficientRow = rankDeficientRow;
  stats.smallestPivot = smallestPivotOverall;
  stats.regularization = regulInfo;

  // Gap 3: Save converged voltages for next solve's adaptive precision routing
  // Gap 4: Record solution in neural seeder history
  if (converged) {
    prevVoltages_ = voltageEstimate;
    neuralSeeder_.recordSolution(voltageEstimate, final_kcl_error,
                                 total_iterations);
  }

  lastSolution_ = voltageEstimate;  // full x (incl. branch currents) seeds transient
  voltageEstimate.resize(netlist.numGlobalNodes);  // API returns node voltages
  return {0.0, voltageEstimate, stats, netlist.globalBlock.topologyHash};
}

// =============================================================================
// COLD-START MICRO-STEPPING (SPICE3 dt_init = tstep/100)
// =============================================================================

// Re-tensorize and carry the AoS companion-model history (global block AND
// subcircuit instances, in tensorizeNetlist()'s flattening order) into the SoA
// arrays. Previously only global devices were synced, so capacitors inside
// subcircuits restarted from zero history every step.
void CircuitSim::loadSoAFromNetlist(const TensorNetlist &netlist) {
  soaBlock = tensorizeNetlist(netlist);
  size_t kc = 0;
  auto sync = [&](const TensorBlock &blk, const BlockState &st) {
    for (size_t i = 0; i < blk.capacitors.size(); ++i, ++kc) {
      if (i >= st.capacitorState.size() || kc >= soaBlock.capacitors.size()) continue;
      const auto &hist = st.capacitorState[i];
      if (!hist.v.empty()) soaBlock.capacitors.v_prev[kc] = hist.v[0];
      if (hist.v.size() > 1) soaBlock.capacitors.v_prev2[kc] = hist.v[1];
      if (!hist.i.empty()) soaBlock.capacitors.i_prev[kc] = hist.i[0];
    }
  };
  sync(netlist.globalBlock, netlist.globalState);
  for (size_t k = 0; k < netlist.instances.size(); ++k) {
    auto it = netlist.blockDefinitions.find(netlist.instances[k].blockName);
    if (it != netlist.blockDefinitions.end() && k < netlist.instanceStates.size())
      sync(it->second, netlist.instanceStates[k]);
  }
}

// Reactive-element history after an accepted step, for the global block AND
// every subcircuit instance. Capacitors use exactly the formulas of
// stampSoABlock() and the KCL audit; inductors record their SOLVED branch
// current (previously inductor history was never written back at all, so any
// transient with an inductor failed to converge).
void CircuitSim::updateReactiveHistory(TensorNetlist &netlist, const std::vector<double> &v,
                                        double h) {
  const IntegrationType type = integrator->getType();
  auto getV = [&](int n) { return (n > 0 && n <= (int)v.size()) ? v[n - 1] : 0.0; };
  int total = 0;
  const auto branchList = branchSlots(netlist, total);  // same block order as below
  size_t slot = 0;
  auto update = [&](const TensorBlock &blk, BlockState &st, const std::function<int(int)> &mapNode) {
    const int indBase = slot < branchList.size() ? branchList[slot++].ind : -1;
    for (size_t a = 0; a < blk.inductors.size() && a < st.inductorState.size() && indBase >= 0; ++a) {
      auto &hist = st.inductorState[a];
      if (hist.v.empty()) hist.resize(3);
      const auto &L = blk.inductors[a];
      const int b = indBase + (int)a;
      integrator->updateHistory(hist.v, hist.i,
                                getV(mapNode(L.nodeCoil1)) - getV(mapNode(L.nodeCoil2)),
                                b < (int)v.size() ? v[b] : 0.0);
    }
    for (size_t i = 0; i < blk.capacitors.size() && i < st.capacitorState.size(); ++i) {
      const auto &c = blk.capacitors[i];
      auto &hist = st.capacitorState[i];
      if (hist.v.empty()) hist.resize(3);
      double v_curr = getV(mapNode(c.nodePlate1)) - getV(mapNode(c.nodePlate2));
      double C = c.capacitance_farads, i_curr;
      if (type == IntegrationType::TRAPEZOIDAL)
        i_curr = (2.0 * C / h) * (v_curr - hist.v[0]) - hist.i[0];
      else if (type == IntegrationType::GEAR_2 && hist.v.size() >= 2)
        i_curr = (C / h) * (1.5 * v_curr - 2.0 * hist.v[0] + 0.5 * hist.v[1]);
      else
        i_curr = (C / h) * (v_curr - hist.v[0]);
      integrator->updateHistory(hist.v, hist.i, v_curr, i_curr);
    }
  };
  update(netlist.globalBlock, netlist.globalState, [](int n) { return n; });
  for (size_t k = 0; k < netlist.instances.size() && k < netlist.instanceStates.size(); ++k) {
    const auto &inst = netlist.instances[k];
    auto it = netlist.blockDefinitions.find(inst.blockName);
    if (it == netlist.blockDefinitions.end()) continue;
    update(it->second, netlist.instanceStates[k], [&](int n) {
      return (n <= 0 || n > (int)inst.nodeMapping.size()) ? 0 : inst.nodeMapping[n - 1];
    });
  }
}

// t = 0 reactive state from x (DC operating point, or the UIC node voltages):
// capacitors hold their voltage at zero current, inductors their current at
// zero voltage, so the first step starts in equilibrium. With UIC an element's
// own IC= wins (SPICE3: unspecified nodes are 0 V, inductor currents 0 A).
void CircuitSim::seedReactiveHistory(TensorNetlist &netlist, const std::vector<double> &x,
                                     bool uic) {
  auto getV = [&](int n) { return (n > 0 && n <= (int)x.size()) ? x[n - 1] : 0.0; };
  int total = 0;
  const auto branchList = branchSlots(netlist, total);  // same block order as below
  size_t slot = 0;
  auto seed = [&](const TensorBlock &blk, BlockState &st, const std::function<int(int)> &mapNode) {
    const int indBase = slot < branchList.size() ? branchList[slot++].ind : -1;
    for (size_t a = 0; a < blk.inductors.size() && a < st.inductorState.size(); ++a) {
      const auto &L = blk.inductors[a];
      const int b = indBase + (int)a;
      const double i0 = uic ? (L.icGiven ? L.ic_A : 0.0)
                            : (indBase >= 0 && b < (int)x.size() ? x[b] : 0.0);
      st.inductorState[a].v.assign(3, 0.0);
      st.inductorState[a].i.assign(3, i0);
    }
    for (size_t i = 0; i < blk.capacitors.size() && i < st.capacitorState.size(); ++i) {
      const auto &c = blk.capacitors[i];
      const double v0 = (uic && c.icGiven) ? c.ic_V
                                           : getV(mapNode(c.nodePlate1)) - getV(mapNode(c.nodePlate2));
      st.capacitorState[i].v.assign(3, v0);
      st.capacitorState[i].i.assign(3, 0.0);
    }
  };
  seed(netlist.globalBlock, netlist.globalState, [](int n) { return n; });
  for (size_t k = 0; k < netlist.instances.size() && k < netlist.instanceStates.size(); ++k) {
    const auto &inst = netlist.instances[k];
    auto it = netlist.blockDefinitions.find(inst.blockName);
    if (it == netlist.blockDefinitions.end()) continue;
    seed(it->second, netlist.instanceStates[k], [&](int n) {
      return (n <= 0 || n > (int)inst.nodeMapping.size()) ? 0 : inst.nodeMapping[n - 1];
    });
  }
}

std::string CircuitSim::runTransient(TensorNetlist &netlist, double step, double stop, bool uic,
                                     const std::function<bool(const SolverStep &)> &onPoint) {
  if (!(step > 0.0) || !(stop > 0.0)) return "Transient needs step > 0 and stop time > 0.";

  const int nTot = systemSize(netlist);
  std::vector<double> x0(nTot, 0.0);
  if (uic) {
    for (const auto &[n, v] : netlist.nodeICs)
      if (n > 0 && n <= netlist.numGlobalNodes) x0[n - 1] = v;
  } else {
    // DC bias. SPICE3 forces .ic nodes during it through 1e10 S to the value
    // (Norton: 1e-10 ohm to ground plus v*1e10 A in). UIC skips it, which an
    // op-amp integrator without DC feedback needs (its OP is the open-loop rail).
    TensorNetlist forced;
    if (!netlist.nodeICs.empty()) {
      forced = netlist;
      for (const auto &[n, v] : netlist.nodeICs) {
        forced.addResistor(n, 0, 1e-10);
        forced.addCurrentSource(0, n, v * 1e10);
      }
    }
    SolverStep dc = solveDC(netlist.nodeICs.empty() ? netlist : forced);
    if (!dc.stats.converged)
      return "DC bias failed." + (dc.stats.error_detail.empty() ? "" : " " + dc.stats.error_detail);
    x0 = lastSolution_;
    x0.resize(nTot, 0.0);
  }
  seedReactiveHistory(netlist, x0, uic);
  lastSolution_ = x0;  // seeds the first Newton iteration

  // Waveform sources whose corners the steps must land on (SPICE breakpoints).
  // ponytail: scans every waveform source per step; a merged corner queue is
  // the upgrade if designs carry thousands of PULSE sources.
  std::vector<std::function<double(double)>> corners;
  auto addCorners = [&](const TensorBlock &b) {
    for (const auto &v : b.voltageSources)
      if (v.type == "PULSE" || v.type == "SINE") corners.push_back([&v](double x) { return nextBreakpoint(v, x); });
    for (const auto &i : b.currentSources)
      if (i.type == "PULSE" || i.type == "SINE") corners.push_back([&i](double x) { return nextBreakpoint(i, x); });
  };
  addCorners(netlist.globalBlock);
  for (const auto &[name, blk] : netlist.blockDefinitions) addCorners(blk);

  // A failed step returns converged=false at the SAME time: shrink h and
  // retry, give up below dtMin, regrow toward the user step.
  const double dtMin = step * 1e-9;
  const double endTol = step * 1e-9;  // absorb FP drift at the final step
  double h = step, t = 0.0;
  while (stop - t > endTol) {
    double hTry = std::min(h, stop - t);
    for (const auto &next : corners) hTry = std::min(hTry, next(t + endTol) - t);
    SolverStep s = stepTransient(netlist, hTry, t);
    if (!s.stats.converged || !(s.time > t)) {
      h = hTry * 0.25;
      if (h < dtMin) {
        std::ostringstream msg;  // default float format: 1e-14, not 0.000000
        msg << "Transient step failed to converge at t=" << t << " s (timestep below " << dtMin
            << " s).";
        return msg.str();
      }
      continue;
    }
    const double taken = s.time - t;
    t = s.time;
    if (!onPoint(s)) return "Cancelled.";
    h = std::min(step, taken * 2.0);
  }
  return "";
}

bool CircuitSim::coldStartSweep(TensorNetlist &netlist, double totalTime,
                                int nSubsteps,
                                std::vector<double> &finalVoltages) {
  if (nSubsteps <= 0 || totalTime <= 0.0 || netlist.numGlobalNodes == 0)
    return false;

  // All micro-steps use Backward Euler (GEAR_1_EULER).
  // The caller already set this before invoking coldStartSweep.
  const double dt_sub = totalTime / static_cast<double>(nSubsteps);

  // Seed from the last full solution (node voltages + branch currents)
  const int nTot = systemSize(netlist);
  std::vector<double> nrEstimate = transientSeed(netlist, nTot);

  for (int k = 0; k < nSubsteps; ++k) {
    // Sync soaBlock capacitor/inductor history from AoS netlist state.
    // This must happen at the start of each sub-step so stampAllElements
    // uses the correct companion-model source term for this micro-step.
    loadSoAFromNetlist(netlist);

    integrator->prepareStep(dt_sub);

    // Inner NR loop — identical structure to the one in stepTransient.
    bool nrConverged = false;
    SolverResult subResult{};
    for (int iteration = 0; iteration < MAX_NR_ITER; ++iteration) {
      scheduler.schedule(netlist);
      matrixBuilder.clear();
      matrixBuilder.setDimensions(nTot, nTot);
      rhsVector.assign(nTot, 0.0);

      stampAllElements(netlist, dt_sub, nrEstimate, dt_sub * (k + 1));
      addDiagonalConditioning(netlist.numGlobalNodes);

      Csr_matrix A = matrixBuilder.createCsr();

      subResult = solveLinear(A, rhsVector);

      std::vector<double> v_new = subResult.solution;

      double kcl_error = kclExcess(netlist, v_new, A, dt_sub, dt_sub * (k + 1));
      if (checkConvergence(v_new, nrEstimate, kcl_error,
                           PhysicsConstants::KCL_TOL)) {
        nrEstimate = v_new;
        nrConverged = true;
        break;
      }
      nrEstimate = v_new;
    }

    if (!nrConverged) {
      // NR failed for this micro-step — abort the sweep.
      return false;
    }

    updateReactiveHistory(netlist, nrEstimate, dt_sub);
    lteController_.recordStep(dt_sub);
    // Note: Inductor history write-back is not performed here, consistent with
    // stepTransient which also does not write back inductorState in its main
    // history-update block.  The soaBlock re-sync at the top of each sub-step
    // reads the current hist.i[0] and hist.v[0] values correctly.
  }

  finalVoltages = nrEstimate;
  return true;
}

SolverStep CircuitSim::stepTransient(TensorNetlist &netlist, double timeStep,
                                     double currentTime) {
  if (netlist.numGlobalNodes == 0)
    return {currentTime, {}, {0, 0.0, true}, 0};

  bool isFirstTransientStep = !arbiterInitialized;
  // When coldStartSweep() succeeds it updates all capacitor history internally,
  // so we must skip the main NR while-loop and the post-convergence
  // history-update block below.  This flag is set to true only when the sweep
  // converges.
  bool isFirstTransientStepHandled = false;

  std::vector<double> finalVoltages;
  SolverResult lastResult{}; // zero-initialize: iterations=0,
                             // finalResidual=0.0, converged=false
  bool simulationConverged = false;
  double next_dt_suggestion = timeStep;

  if (!arbiterInitialized) {
    arbiter.initialize(netlist);
    arbiterInitialized = true;
    lteController_.reset();

    // M4: Enforce Formal Passivity
    if (!arbiter.checkPassivity(netlist)) {
      // Halt simulation if physically impossible device parameters found
      return {currentTime, {}, {0, 0.0, false}};
    }

    // Use Backward Euler for all cold-start micro-steps.  TRAPEZOIDAL's
    // companion current I_eq = -G_eq*v_prev - i_prev is zero when both
    // hist.v and hist.i are zero (cold start / UIC ic=0), which gives a
    // systematically wrong first-step solution.  BE's companion model
    // I_eq = 0 (purely a conductance C/h) is correct for a zero-IC start
    // and doesn't depend on i_prev at all.
    setIntegrationMethod(IntegrationType::GEAR_1_EULER);

    // SPICE3-accurate cold-start: divide the first transient step into
    // COLD_START_SUBSTEPS internal BE micro-steps (dt_init = tstep/100).
    // Each micro-step updates the AoS capacitor history so that by the end
    // the companion-model state matches what SPICE3 produces at t = tstep.
    std::vector<double> sweepVoltages;
    bool sweepOk =
        coldStartSweep(netlist, timeStep, COLD_START_SUBSTEPS, sweepVoltages);
    if (sweepOk) {
      finalVoltages = sweepVoltages;
      simulationConverged = true;
      // Restore TRAPEZOIDAL now — hist.i has been recorded with BE formula
      // for all 100 micro-steps, so the next step can use the high-accuracy
      // second-order method with a valid hist.i.
      setIntegrationMethod(IntegrationType::TRAPEZOIDAL);
      isFirstTransientStepHandled = true;
    }
    // Fallback: if sweep failed (e.g. NR divergence on a stiff circuit),
    // isFirstTransientStepHandled remains false and the main NR loop below
    // executes a single BE step at timeStep — the original cold-start behavior.
  }

  // Initialize SoA Block for transient steps.
  // tensorizeNetlist() zero-inits v_prev/i_prev; we must carry the AoS history
  // (updated by the previous step's history-update loop) into the SoA arrays so
  // that stampSoABlock() uses the correct companion-model source term.
  loadSoAFromNetlist(netlist);

  double actualTimeStep = timeStep;
  int attempt = 0;
  const int MAX_ATTEMPTS = 4;

  // finalVoltages, lastResult, simulationConverged, next_dt_suggestion are
  // declared above (before the !arbiterInitialized block) so coldStartSweep can
  // populate them.

  if (!isFirstTransientStepHandled) {
    while (attempt < MAX_ATTEMPTS) {
      integrator->prepareStep(actualTimeStep);

      const int nTot = systemSize(netlist);
      std::vector<double> nrEstimate = transientSeed(netlist, nTot);

      // Gap 4: Neural seeder dV/dt extrapolation for transient steps.
      // Uses the two most recent converged solutions to predict the next state.
      if (!prevVoltages_.empty() && prevVoltages_.size() == nrEstimate.size()) {
        auto seed =
            neuralSeeder_.predict(nrEstimate, prevVoltages_, actualTimeStep);
        if (!seed.empty()) {
          nrEstimate = std::move(seed);
        }
      }

      // ─── TAA: History Rejection ──────────────────────────────────────
      // Reject polynomial predictors for nodes undergoing violent switching.
      // Replaces their guess with first-order Euler to prevent NR divergence
      // without globally cutting the timestep.
      arbiter.rejectHistory(nrEstimate, actualTimeStep);

      bool nrConverged = false;
      for (int iteration = 0; iteration < MAX_NR_ITER; ++iteration) {
        scheduler.schedule(netlist);
        matrixBuilder.clear();
        matrixBuilder.setDimensions(nTot, nTot);
        rhsVector.assign(nTot, 0.0);

        stampAllElements(netlist, actualTimeStep, nrEstimate,
                         currentTime + actualTimeStep);
        addDiagonalConditioning(netlist.numGlobalNodes);

        Csr_matrix A = matrixBuilder.createCsr();

        // --- Solver Strategy Optimization ---
        // LU is O(N^3) but robust for non-symmetric active circuits.
        // PCG is O(N*sqrt(K)) but requires SPD (Resistor/Capacitor meshes).
        lastResult = solveLinear(A, rhsVector);

        std::vector<double> v_new = lastResult.solution;

        double kcl_error =
            kclExcess(netlist, v_new, A, actualTimeStep, currentTime + actualTimeStep);
        if (checkConvergence(v_new, nrEstimate, kcl_error,
                             PhysicsConstants::KCL_TOL)) {
          nrEstimate = v_new;
          nrConverged = true;
          break;
        }
        nrEstimate = v_new;
      }

      if (!nrConverged) {
        actualTimeStep *= 0.5;
        attempt++;
        continue;
      }

      // ─── Gap 5: LTE-Based Timestep Control ─────────────────────────────
      // Estimate local truncation error BEFORE the arbiter's stability check.
      // If LTE exceeds tolerance, reject the step and retry with smaller dt.
      auto lteResult = lteController_.estimateLTE(
          netlist, nrEstimate, actualTimeStep, integrator->getType());

      if (lteResult.requiresRejection) {
        // LTE too large — timestep was too aggressive
        actualTimeStep = lteResult.suggestedDt;
        attempt++;
        continue;
      }

      // ponytail: no automatic TRAP -> GEAR_2 switch on "ringing" (here or for
      // arbiter suggestion 2 below). Every GEAR_2 formula (stamps, audit, history,
      // inductor rows) has constant-step BDF2 coefficients, and the switch came
      // when h was changing: each switch injected a jump (~40 mV in
      // hier_rc_ladder_pulse). Variable-step BDF2 coefficients are the upgrade.

      // Original arbiter stability analysis (energy monitor, Lyapunov check)
      // The arbiter tracks node voltages only (it re-initialises on a size change)
      int suggestion = arbiter.analyzeStability(
          netlist,
          std::vector<double>(nrEstimate.begin(), nrEstimate.begin() + netlist.numGlobalNodes),
          actualTimeStep, next_dt_suggestion);

      if (suggestion == 1) {
        actualTimeStep *= 0.5;
        attempt++;
        continue;
      }

      // Use LTE-based timestep suggestion if it's more conservative
      double lte_dt =
          lteController_.suggestTimestep(actualTimeStep, lteResult.maxLTE,
                                         lteTolerance_, integrator->getType());
      if (lte_dt < next_dt_suggestion) {
        next_dt_suggestion = lte_dt;
      }

      // ─── Gap 2: Digital Event Engine — A2D Sampling ──────────────────
      // After analog convergence, sample voltages at A2D boundary ports
      // and propagate digital events. This prepares D2A stamps for the
      // next transient step.
      if (digitalEngine_.portCount() > 0) {
        digitalEngine_.sampleAnalog(nrEstimate, currentTime + actualTimeStep);
        digitalEngine_.propagateEvents();
      }

      finalVoltages = nrEstimate;
      simulationConverged = true;
      timeStep = next_dt_suggestion;
      break;
    }
  } // end if (!isFirstTransientStepHandled)

  // Hang Protection: Detect if timeStep has underflowed
  if (timeStep < 1e-18) {
    // Prevent infinite loop by returning failing step
    return {currentTime,
            std::vector<double>(netlist.numGlobalNodes, 0.0),
            {lastResult.iterations, lastResult.finalResidual, false}};
  }

  // When coldStartSweep() handled the first step, all capacitor history was
  // updated inside the sweep loop.  Skip the history-update block to avoid
  // double-updating the companion-model state.
  if (simulationConverged && !isFirstTransientStepHandled) {
    updateReactiveHistory(netlist, finalVoltages, actualTimeStep);

    lteController_.recordStep(actualTimeStep);

    // Restore TRAPEZOIDAL after the first step (which used Backward Euler
    // for cold-start accuracy).  The restore must happen AFTER the cap
    // history update above so that hist.i is recorded with the BE formula,
    // not the TRAPEZOIDAL formula — otherwise i_prev would be doubled.
    if (isFirstTransientStep &&
        integrator->getType() == IntegrationType::GEAR_1_EULER) {
      setIntegrationMethod(IntegrationType::TRAPEZOIDAL);
    }

    // Update Power Rail History (Local Decap)
    for (size_t i = 0; i < netlist.globalBlock.powerRails.size(); ++i) {
      const auto &rail = netlist.globalBlock.powerRails[i];
      auto &hist = netlist.globalState.powerRailState[i];

      double v_curr = 0.0;
      if (rail.nodeRail > 0 && rail.nodeRail <= (int)finalVoltages.size())
        v_curr = finalVoltages[rail.nodeRail - 1];

      double i_curr = 0.0;
      double C = rail.capacitance_F;
      IntegrationType type = integrator->getType();

      if (hist.v.empty())
        hist.resize(3);

      // I_cap = C * dv/dt (Trapezoidal integration)
      if (type == IntegrationType::TRAPEZOIDAL) {
        if (!hist.v.empty())
          i_curr =
              (2.0 * C / actualTimeStep) * (v_curr - hist.v[0]) - hist.i[0];
      } else if (type == IntegrationType::GEAR_1_EULER) {
        if (!hist.v.empty())
          i_curr = (C / actualTimeStep) * (v_curr - hist.v[0]);
      }

      integrator->updateHistory(hist.v, hist.i, v_curr, i_curr);
    }
  }

  if (!simulationConverged)
    return {currentTime,
            std::vector<double>(netlist.numGlobalNodes, 0.0),
            {lastResult.iterations, lastResult.finalResidual, false},
            netlist.globalBlock.topologyHash};

  // Gap 4: Record converged transient solution for neural seeder
  prevVoltages_ = finalVoltages;
  neuralSeeder_.recordSolution(finalVoltages, lastResult.finalResidual,
                               lastResult.iterations);
  lastSolution_ = finalVoltages;                    // full x seeds the next step
  finalVoltages.resize(netlist.numGlobalNodes);     // API returns node voltages

  return {
      currentTime + actualTimeStep,
      finalVoltages,
      {lastResult.iterations, lastResult.finalResidual, simulationConverged},
      netlist.globalBlock.topologyHash};
}

// =============================================================================
// COMPILED BLOCK API — Public entry points
// =============================================================================
// These overloads accept the pre-compiled, immutable CompiledTensorBlock.
// They pre-load the SoA tensors and then delegate to the internal
// TensorNetlist-based methods via the embedded structural_ copy.

SolverStep CircuitSim::solveDC(const CompiledTensorBlock &block) {
  // Pre-load the SoA tensors so tensorizeNetlist() inside is a no-op
  soaBlock = block.tensors;

  // Phase B: Make the compiled block available for treewidth-guided routing.
  // Use a non-owning shared_ptr (aliasing constructor) since the block
  // outlives this call — the caller holds ownership.
  compiledBlock_ = std::shared_ptr<const CompiledTensorBlock>(
      std::shared_ptr<const CompiledTensorBlock>{}, &block);

  // Delegate to internal TensorNetlist-based solver
  // A mutable copy is needed because the solver writes to instance states
  TensorNetlist nl = block.structural_;
  auto result = solveDC(nl);

  compiledBlock_.reset(); // Release non-owning reference
  return result;
}

SolverStep CircuitSim::stepTransient(const CompiledTensorBlock &block,
                                     double timeStep, double currentTime) {
  // Pre-load the SoA tensors
  soaBlock = block.tensors;

  // Phase B: Make the compiled block available for treewidth-guided routing.
  compiledBlock_ = std::shared_ptr<const CompiledTensorBlock>(
      std::shared_ptr<const CompiledTensorBlock>{}, &block);

  // Delegate to internal TensorNetlist-based solver
  TensorNetlist nl = block.structural_;
  auto result = stepTransient(nl, timeStep, currentTime);

  compiledBlock_.reset();
  return result;
}
