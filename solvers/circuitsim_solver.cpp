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

#include "../infrastructure/safe_utils.h"
#include "../physics/circuitsim.h"
#include "../physics/device_physics.h"
#include "../physics/physics_constants.h"
#include "../solvers/convergence_engine.h"
#if !defined(__EMSCRIPTEN__) && defined(ACUTESIM_HAS_DAWN_NATIVE)
#include "gpu_context_manager.h"
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

  // 5. BJT Multi-Pass Seeding (Darlington propagation), every block
  auto at = [&](int n) -> double * { return n > 0 && n <= (int)N ? &voltageEstimate[n - 1] : nullptr; };
  for (int pass = 0; pass < 3; ++pass) {
    forEachJunction(netlist, [&](const Junction &j) {
      if (!j.q) return;
      double *c = at(j.n[0]), *b = at(j.n[1]), *e = at(j.n[2]);
      const double s = j.q->isNPN ? 1.0 : -1.0;  // PNP: mirrored
      if (e && b) *b = s * std::max(s * *b, s * (*e + s * vbe_seed));
      if (c && b) *c = s * std::max(s * *c, s * (*b + s * 0.5));
    });
  }

  // 6. Internal nodes (series resistances) start at their external terminal
  forEachJunction(netlist, [&](const Junction &j) {
    const int ext[7] = {0, 1, 2, 3, 0, 1, 2}, pairs = j.d ? 1 : 3, first = j.d ? 2 : 4;
    for (int k = first; k < first + pairs; ++k) {
      const int x = j.d ? 0 : ext[k];
      if (j.n[k] != j.n[x]) if (double *a = at(j.n[k])) *a = at(j.n[x]) ? *at(j.n[x]) : 0.0;
    }
  });
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

  struct RestoreSources { double &f; ~RestoreSources() { f = 1.0; } } restoreSources{srcFact_};

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
  resetJunctionLimits(netlist);  // first iteration: junctions at vcrit (ngspice MODEINITJCT)

  bool converged = false;
  int total_iterations = 0;
  double final_kcl_error = 0.0;
  bool singularMatrixDetected = false;
  int singularRow = -1;  // unknown with no pivot (ngspice: "singular matrix: check node N")
  std::vector<ConvergenceStep> convergenceHistory;
  // Phase 2.9: Numerical Intelligence state
  bool stagnationDetected = false;
  bool divergenceDetected = false;
  bool rankDeficientDetected = false;
  int rankDeficientRow = -1;
  double smallestPivotOverall = 1e300;
  RegularizationInfo regulInfo;

  // SPICE order: plain Newton with NO node gmin first (junction GMIN in the
  // device models keeps floating junction nodes referenced, like ngspice — a
  // node-to-ground leak settles them differently: aac_10_bridge). Only if that
  // fails, ngspice's dynamic gmin stepping (cktop.c dynamic_gmin): node gmin from
  // 1e-3 down to 1e-12, the step factor grown after easy steps and shrunk after
  // hard ones, and a failed step restarts from the last good solution with a
  // smaller decrement; then a final plain Newton at gmin = 0. The fixed ladder it
  // replaces gave up at the first failed step, so long inverter chains (>= 50
  // stages), which ngspice solves this way, never converged.
  constexpr int ITL1 = 100;               // ngspice: plain/final DC Newton limit
  constexpr int ITL2 = MAX_NR_ITER;       // ngspice: per gmin step (50)
  constexpr double GMIN_FACTOR = 10.0, GMIN_TARGET = 1e-12;
  double gmin_factor = 0.0;               // current node gmin (0 = plain Newton)
  double stepFactor = GMIN_FACTOR, oldGmin = 1e-2;
  bool finalStep = true, gminSteppingUsed = false;
  // Source stepping (ngspice gillespie_src), when gmin stepping fails too: all
  // independent sources scaled by srcFact_ from 0 up to 1, the increment grown
  // after easy steps, cut tenfold after a failed one (from the last good point).
  bool sourceSteppingUsed = false;
  double srcConv = 0.0, srcRaise = 1e-3;
  const std::vector<double> seededGuess = voltageEstimate;
  std::vector<double> lastGood;
  for (int si = 0; si < 1000; ++si) {  // ponytail: hard cap; ngspice has none (factor -> 1 ends it)
    bool gmin_converged = false;

    // Phase 2.9: Track regularization
    if (gmin_factor > 0.0) {
      regulInfo.applied = true;
      regulInfo.gminValue = gmin_factor;
      regulInfo.injections = static_cast<uint32_t>(netlist.numGlobalNodes);
    }

    int iteration = 0; // Declare outside loop for scope access
    for (; iteration < (finalStep ? ITL1 : ITL2); ++iteration) {
      // Unified Stamping (Phase 1: Structural Abstraction)
      scheduler.schedule(netlist);
      matrixBuilder.clear();
      matrixBuilder.setDimensions(nTot, nTot);
      rhsVector.assign(nTot, 0.0);

      stampAllElements(netlist, 0.0, voltageEstimate, 0.0);
      addDiagonalConditioning(netlist.numGlobalNodes, gmin_factor);
      if (gmin_factor > 0.0) {  // device-internal nodes are nodes too (ngspice CKTdiagGmin)
        int totalRows = 0;
        for (const auto &sl : branchSlots(netlist, totalRows))
          for (int r = sl.dev, e = sl.dev + deviceInternals(*sl.blk, *sl.st); r < e; ++r)
            matrixBuilder.add(r, r, gmin_factor);
      }

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
        singularRow = result.rankDeficientRow;
        // If singular, retrying with same Jacobian won't help. Abort this GMIN
        // step.
        total_iterations += iteration + 1;
        break;
      }

      std::vector<double> v_new = result.solution;

      // NaN Protection using SafeUtils
      SafeUtils::sanitize_vector(v_new, 0.0);


      // Check 1 + 2: Voltage Convergence AND Physical KCL Accuracy
      double kcl_error = kclExcess(netlist, v_new, matrix);

      // Limit Cycle Breaker: Damping
      // If the residual got worse (and is above tolerance), dampen the step to
      // force a different path. ">=" also fired on 0 >= 0 (both below the
      // scale-aware floor), halving every step until the update looked
      // converged: a 555 output node stopped 0.4 V short of its solution.
      const double dampTol = finalStep ? PhysicsConstants::KCL_TOL : 1.0;
      if (iteration > 0 && kcl_error > final_kcl_error && kcl_error > dampTol) { // final_kcl_error holds prev_error
        // Apply damping: v_new = v_old + 0.5 * (v_new - v_old)
        for (size_t i = 0; i < v_new.size(); ++i) {
          v_new[i] = voltageEstimate[i] + 0.5 * (v_new[i] - voltageEstimate[i]);
        }
        // Re-evaluate residual after damping
        kcl_error = kclExcess(netlist, v_new, matrix);

        // If still bad, dampen again (aggressive backoff)
        if (kcl_error > final_kcl_error) {
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
      double effective_kcl_tol = finalStep ? PhysicsConstants::KCL_TOL : 1.0;

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

    if (sourceSteppingUsed) {
      if (gmin_converged) {
        lastGood = voltageEstimate;
        srcConv = srcFact_;
        if (srcConv >= 1.0) { converged = true; break; }
        if (iteration + 1 <= ITL2 / 4) srcRaise *= 1.5;
        if (iteration + 1 > 3 * ITL2 / 4) srcRaise *= 0.5;
      } else {
        if (srcFact_ == 0.0 || srcRaise < 1e-7) break;  // zero sources fail, or the ramp stalled
        srcRaise /= 10.0;
        voltageEstimate = lastGood;
        syncJunctionLimits(netlist, voltageEstimate);
      }
      srcFact_ = std::min(1.0, srcConv + srcRaise);
      continue;
    }
    // Only a gmin = 0 solve marks the simulation as converged
    if (finalStep && gmin_converged) {
      converged = true;
      break;
    }
    if (finalStep && (gminSteppingUsed || !netlist.environment.gmin_stepping)) {  // gmin stepping failed (or off): sources from zero
      sourceSteppingUsed = true;
      std::cerr << "Note: starting source stepping\n";  // as ngspice
      srcFact_ = 0.0;
      voltageEstimate = lastGood = std::vector<double>(nTot, 0.0);
      resetJunctionLimits(netlist);
      stagnationDetected = divergenceDetected = false;
      continue;
    }

    if (!gminSteppingUsed) {  // direct attempt failed: start from the seed
      gminSteppingUsed = true;
      voltageEstimate = lastGood = seededGuess;
      resetJunctionLimits(netlist);  // ngspice dynamic_gmin starts in MODEINITJCT
      stagnationDetected = divergenceDetected = false;
      gmin_factor = oldGmin / stepFactor;
    } else if (gmin_converged) {
      const int iters = iteration + 1;
      lastGood = voltageEstimate;
      if (gmin_factor <= GMIN_TARGET) {
        gmin_factor = 0.0;
      } else {
        if (iters <= ITL2 / 4) stepFactor = std::min(stepFactor * std::sqrt(stepFactor), GMIN_FACTOR);
        if (iters > 3 * ITL2 / 4) stepFactor = std::sqrt(stepFactor);
        oldGmin = gmin_factor;
        if (gmin_factor < stepFactor * GMIN_TARGET) {
          stepFactor = gmin_factor / GMIN_TARGET;
          gmin_factor = GMIN_TARGET;
        } else {
          gmin_factor /= stepFactor;
        }
      }
    } else if (stepFactor < 1.00005) {
      gmin_factor = 0.0;  // ladder failed: ngspice still tries the final solve
    } else {              // back off: smaller decrement from the last good point
      stepFactor = std::sqrt(std::sqrt(stepFactor));
      gmin_factor = oldGmin / stepFactor;
      voltageEstimate = lastGood;
      syncJunctionLimits(netlist, voltageEstimate);
    }
    finalStep = gmin_factor == 0.0;
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
  if (!converged && singularRow >= 0)
    details = (singularRow < netlist.numGlobalNodes
                   ? "Singular matrix: check node " + std::to_string(singularRow + 1) +
                         " (no DC path to ground?)"
                   : "Singular matrix: check branch " + std::to_string(singularRow - netlist.numGlobalNodes + 1) +
                         " (a loop of voltage sources or inductors?)") +
              (details.empty() ? "" : "\n" + details);
  stats.error_detail = details;
  stats.method = "Newton-Raphson";
  stats.integrationMethod = ""; // DC has no integration
  stats.gminSteppingUsed = gminSteppingUsed;  // the direct attempt failed: ladder walked
  stats.sourceSteppingUsed = sourceSteppingUsed;
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
  // MOSFET charges: the state mosCurrents() integrated at v (netlist is non-const here)
  forEachMosfet(netlist, [&](const Mosfet &m, const std::array<int, 4> &n, const BlockState::DeviceCharge *st) {
    if (st) mosCurrents<double>({getV(n[0]), getV(n[1]), getV(n[2]), getV(n[3])}, mos1Params(m), st, h,
                                const_cast<BlockState::DeviceCharge *>(st));
  });
  // Diode / BJT charges, likewise (junctionCurrents with next = st)
  forEachJunction(netlist, [&](const Junction &j) {
    if (!j.st) return;
    std::array<double, 7> x{};
    for (int k = 0; k < j.terms; ++k) x[k] = getV(j.n[k]);
    junctionCurrents(j, x, netlist.environment.ambient_temp_K, h, j.st);
  });
  // OSDI charges, likewise (osdiCurrents with accept)
  forEachOsdi(netlist, [&](const OsdiRef &r) { osdiCurrents(r, v, h, nullptr, true); });
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
  // MOSFET charges at x, zero current (Meyer charges count from here: only their change matters)
  forEachMosfet(netlist, [&](const Mosfet &m, const std::array<int, 4> &n, const BlockState::DeviceCharge *st) {
    if (!st) return;
    const double zero[5] = {};
    const auto c = mos1_charges(getV(n[0]), getV(n[1]), getV(n[2]), getV(n[3]), mos1Params(m), zero, zero, zero);
    auto *s = const_cast<BlockState::DeviceCharge *>(st);  // netlist is non-const here
    *s = {};
    for (auto &vh : s->v) std::copy(c.v, c.v + 5, vh);
    std::copy(c.q, c.q + 5, s->q);
    std::copy(c.half, c.half + 3, s->half);
  });
  // Diode / BJT charges at x, zero current
  forEachJunction(netlist, [&](const Junction &j) {
    if (!j.st) return;
    std::array<double, 7> xv{};
    for (int k = 0; k < j.terms; ++k) xv[k] = getV(j.n[k]);
    double v[5], q[5];
    const int K = junctionCharges(j, xv, netlist.environment.ambient_temp_K, v, q);
    *j.st = {};
    for (auto &vh : j.st->v) std::copy(v, v + K, vh);
    std::copy(q, q + K, j.st->q);
  });
  // OSDI charges at x, zero current
  forEachOsdi(netlist, [&](const OsdiRef &r) {
    osdiCurrents(r, x, 0.0, nullptr, true);
    r.in->vHist[1] = r.in->vHist[2] = r.in->vHist[0];
  });
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
  if (step == 0.0) step = stop / 50.0;  // .tran 0 tstop: ngspice's default TMAX
  if (!(step > 0.0) || !(stop > 0.0)) return "Transient needs step >= 0 and stop time > 0.";

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
  // The step after a corner uses backward Euler (SPICE: order 1 after a
  // breakpoint): trapezoidal rings +-dv across a derivative jump (an L driven
  // by a ramp that stops) and LTE rejection then shrinks h until it gives up.
  double h = step, t = 0.0;
  bool landedOnCorner = false;
  while (stop - t > endTol) {
    double hTry = std::min(h, stop - t);
    bool toCorner = false;
    for (const auto &next : corners)
      if (next(t + endTol) - t <= hTry) { hTry = next(t + endTol) - t; toCorner = true; }
    if (landedOnCorner) setIntegrationMethod(IntegrationType::GEAR_1_EULER);
    SolverStep s = stepTransient(netlist, hTry, t);
    if (landedOnCorner) setIntegrationMethod(IntegrationType::TRAPEZOIDAL);
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
    landedOnCorner = toCorner && taken >= hTry * (1.0 - 1e-9);
    t = s.time;
    if (!onPoint(s)) return "Cancelled.";
    h = std::min(step, taken * 2.0);
  }
  return "";
}

bool CircuitSim::acSystem(TensorNetlist &netlist, ACSystem &sys) {
  sys = {};
  sys.n = systemSize(netlist);
  if ((int)lastSolution_.size() != sys.n) return false;
  // G: the DC Newton Jacobian at the operating point (same stamps as solveDC),
  // without the node-to-ground gmin: like ngspice, a 1e-12 S leak on every node
  // would load 1e10-ohm references by 1 % (aac_09 transformer secondaries).
  matrixBuilder.clear();
  matrixBuilder.setDimensions(sys.n, sys.n);
  rhsVector.assign(sys.n, 0.0);
  limitJunctions_ = false;  // the Jacobian at the operating point itself
  stampAllElements(netlist, 0.0, lastSolution_, 0.0);
  limitJunctions_ = true;
  sys.G = matrixBuilder.createCsr();
  auto addC = [&](int r, int c, double v) {
    if (r >= 0 && c >= 0) { sys.cr.push_back(r); sys.cc.push_back(c); sys.cv.push_back(v); }
  };
  int total = 0;
  for (const auto &sl : branchSlots(netlist, total)) {
    auto node = [&](int local) { return sl.map(local) - 1; };  // ground -> -1
    for (const auto &c : sl.blk->capacitors) {
      const int a = node(c.nodePlate1), b = node(c.nodePlate2);
      const double C = c.capacitance_farads;
      addC(a, a, C); addC(b, b, C); addC(a, b, -C); addC(b, a, -C);
    }
    const auto flux = fluxLinks(*sl.blk);  // inductor row: v1 - v2 - jw*sum L_aj i_j = 0
    for (size_t a = 0; a < flux.size(); ++a)
      for (auto [j, Laj] : flux[a]) addC(sl.ind + (int)a, sl.ind + j, -Laj);
  }
  // MOSFETs: C_tu = dQ_t/dV_u of the terminal charges at the operating point
  // (Meyer 2*half + overlap, junction dq/dv: ngspice mos1acld), from Dual.
  forEachMosfet(netlist, [&](const Mosfet &m, const std::array<int, 4> &n, const BlockState::DeviceCharge *) {
    const auto p = mos1Params(m);
    if (!p.caps) return;
    auto V = [&](int k) { return n[k] > 0 ? lastSolution_[n[k] - 1] : 0.0; };
    const double zero[5] = {};
    const auto op = mos1_charges(V(0), V(1), V(2), V(3), p, zero, zero, zero);
    for (int u = 0; u < 4; ++u) {
      std::array<Dual<double>, 4> x;
      for (int k = 0; k < 4; ++k) x[k] = Dual<double>(V(k), k == u ? 1.0 : 0.0);
      const auto c = mos1_charges(x[0], x[1], x[2], x[3], p, op.v, zero, op.half);
      double Q[4] = {};
      for (int k = 0; k < 5; ++k) {
        Q[mosFrom[k]] += p.polarity * c.q[k].grad;
        Q[mosTo[k]] -= p.polarity * c.q[k].grad;
      }
      for (int t = 0; t < 4; ++t) if (Q[t] != 0.0) addC(n[t] - 1, n[u] - 1, Q[t]);
    }
  });
  // Diodes / BJTs: C_tu = dQ_t/dV_u of the junction charges (depletion + transit
  // time, XTF/VTF cross terms included) at the operating point, from Dual.
  const double tempK = netlist.environment.ambient_temp_K;
  forEachJunction(netlist, [&](const Junction &j) {
    if (!junctionHasCharge(j)) return;
    auto V = [&](int k) { return j.n[k] > 0 ? lastSolution_[j.n[k] - 1] : 0.0; };
    const int *from = j.d ? diodeFrom : bjtFrom, *to = j.d ? diodeTo : bjtTo;
    const double type = j.q ? (j.q->isNPN ? 1.0 : -1.0) : 1.0;
    for (int u = 0; u < j.terms; ++u) {
      if (j.n[u] <= 0) continue;
      bool seen = false;
      for (int w = 0; w < u; ++w) seen = seen || j.n[w] == j.n[u];
      if (seen) continue;
      std::array<Dual<double>, 7> x;
      for (int k = 0; k < 7; ++k) x[k] = Dual<double>(V(k), k < j.terms && j.n[k] == j.n[u] ? 1.0 : 0.0);
      Dual<double> v[5], q[5];
      const int K = junctionCharges(j, x, tempK, v, q);
      double Q[7] = {};
      for (int k = 0; k < K; ++k) { Q[from[k]] += type * q[k].grad; Q[to[k]] -= type * q[k].grad; }
      for (int t = 0; t < j.terms; ++t) if (Q[t] != 0.0) addC(j.n[t] - 1, j.n[u] - 1, Q[t]);
    }
  });
  // OSDI devices: the reactive Jacobian dq/dx at the operating point
  forEachOsdi(netlist, [&](const OsdiRef &r) {
    osdiCurrents(r, lastSolution_, 0.0);
    for (size_t a = 0; a < r.n.size(); ++a)
      for (size_t b = 0; b < r.n.size(); ++b)
        if (const double c = r.in->C(a, b); c != 0.0) addC(r.n[a] - 1, r.n[b] - 1, c);
  });
  return true;
}

Csr_matrix CircuitSim::acMatrix(const ACSystem &s, double w, bool transpose) {
  const int n = s.n;
  MatrixConstructor M;
  M.setDimensions(2 * n, 2 * n);
  auto put = [&](int r, int c, double v) { transpose ? M.add(c, r, v) : M.add(r, c, v); };
  for (int r = 0; r < n; ++r)
    for (int p = s.G.row_pointer[r]; p < s.G.row_pointer[r + 1]; ++p) {
      put(r, s.G.col_indices[p], s.G.values[p]);
      put(r + n, s.G.col_indices[p] + n, s.G.values[p]);
    }
  for (size_t t = 0; t < s.cv.size(); ++t) {
    put(s.cr[t], s.cc[t] + n, -w * s.cv[t]);
    put(s.cr[t] + n, s.cc[t], w * s.cv[t]);
  }
  return M.createCsr();
}

std::vector<std::complex<double>> CircuitSim::acExcitation(const TensorNetlist &nl,
                                                           const std::string &unit) const {
  int total = 0;
  const auto blocks = branchSlots(nl, total);
  std::vector<std::complex<double>> e(total);
  bool found = unit.empty();
  auto phasor = [&](const std::string &name, double mag, double phaseDeg) -> std::complex<double> {
    if (!unit.empty()) return name == unit ? (found = true, 1.0) : 0.0;
    return std::polar(mag, phaseDeg * M_PI / 180.0);
  };
  for (const auto &sl : blocks) {
    for (size_t i = 0; i < sl.blk->voltageSources.size(); ++i) {
      const auto &v = sl.blk->voltageSources[i];
      e[sl.vs + i] += phasor(v.name, v.ac_mag, v.ac_phase);
    }
    for (const auto &cs : sl.blk->currentSources) {  // leaves n+ externally (as in DC)
      const auto p = phasor(cs.name, cs.ac_mag, cs.ac_phase);
      const int a = sl.map(cs.nodePositive) - 1, b = sl.map(cs.nodeNegative) - 1;
      if (a >= 0) e[a] -= p;
      if (b >= 0) e[b] += p;
    }
  }
  return found ? e : std::vector<std::complex<double>>{};
}

std::vector<std::vector<std::complex<double>>>
CircuitSim::solveAC(TensorNetlist &netlist, const std::vector<double> &freqs) {
  ACSystem sys;
  if (!acSystem(netlist, sys)) return {};
  const int n = sys.n;
  const auto e = acExcitation(netlist);
  std::vector<double> b(2 * n);
  for (int i = 0; i < n; ++i) { b[i] = e[i].real(); b[i + n] = e[i].imag(); }
  std::vector<std::vector<std::complex<double>>> out;
  for (double f : freqs) {
    const SolverResult s = solveLinear(acMatrix(sys, 2.0 * M_PI * f, false), b);
    std::vector<std::complex<double>> x(n);
    for (int i = 0; i < n && i + n < (int)s.solution.size(); ++i) x[i] = {s.solution[i], s.solution[i + n]};
    out.push_back(std::move(x));
  }
  return out;
}

CircuitSim::NoiseResult CircuitSim::solveNoise(TensorNetlist &netlist, const std::vector<double> &freqs,
                                               int outNode, int refNode, const std::string &inputSource) {
  NoiseResult r;
  ACSystem sys;
  if (!acSystem(netlist, sys)) { r.error = "noise needs a converged DC operating point"; return r; }
  const int n = sys.n;
  const auto ein = acExcitation(netlist, inputSource);
  if (ein.empty()) { r.error = "noise input source '" + inputSource + "' not found"; return r; }
  if (outNode < 1 || outNode > netlist.numGlobalNodes || refNode < 0 || refNode > netlist.numGlobalNodes) {
    r.error = "noise output node out of range";
    return r;
  }

  // Generators: current noise of `psd` [A^2/Hz] between global nodes a, b (0 = ground)
  struct Gen { int a, b; double psd; std::function<double(double)> psdAt = nullptr; };  // psdAt(f): coloured
  std::vector<Gen> gens;
  const double kT4 = 4.0 * PhysicsConstants::K_BOLTZMANN * netlist.environment.ambient_temp_K;
  const double q2 = 2.0 * PhysicsConstants::Q_ELECTRON;
  auto V = [&](int g) { return g > 0 ? lastSolution_[g - 1] : 0.0; };
  int total = 0;
  for (const auto &sl : branchSlots(netlist, total)) {
    const auto &blk = *sl.blk;
    for (const auto &rr : blk.resistors)
      gens.push_back({sl.map(rr.nodeTerminal1), sl.map(rr.nodeTerminal2), kT4 / rr.resistance_ohms});
    for (const auto &m : blk.mosfets) {
      const int d = sl.map(m.drain), g = sl.map(m.gate), s = sl.map(m.source), b = sl.map(m.body);
      // channel gm = dI_drain/dV_gate (the bulk junctions do not see the gate)
      const double gm = mos1_currents(Dual<double>(V(d)), Dual<double>(V(g), 1.0), Dual<double>(V(s)),
                                      Dual<double>(V(b)), mos1Params(m)).d.grad;
      gens.push_back({d, s, kT4 * (2.0 / 3.0) * std::abs(gm)});
    }
    for (const auto &j : blk.jfets) {
      const int d = sl.map(j.drain), g = sl.map(j.gate), s = sl.map(j.source);
      const double sign = j.isNChannel ? 1.0 : -1.0;
      JFETParams<double> p;
      p.Beta = j.beta; p.Vto = j.Vto; p.Lambda = j.lambda;
      const double gm = jfet_ids(Dual<double>(sign * (V(g) - V(s)), 1.0),
                                 Dual<double>(sign * (V(d) - V(s)), 0.0), p).grad;
      gens.push_back({d, s, kT4 * (2.0 / 3.0) * std::abs(gm)});
    }
  }
  // Diodes / BJTs (ngspice dionoise, bjtnoise): series-resistance thermal noise,
  // shot noise of the junction currents between the internal nodes
  const double tempK = netlist.environment.ambient_temp_K;
  forEachJunction(netlist, [&](const Junction &j) {
    if (j.d) {
      const auto m = diodeModel(*j.d, tempK);
      if (m.rs > 0.0) gens.push_back({j.n[0], j.n[2], kT4 / m.rs});
      gens.push_back({j.n[2], j.n[1], q2 * std::abs(diode_iq(V(j.n[2]) - V(j.n[1]), m).first)});
      return;
    }
    const auto m = bjtModel(*j.q, tempK);
    if (m.rc > 0.0) gens.push_back({j.n[0], j.n[4], kT4 / m.rc});
    if (m.rb > 0.0) gens.push_back({j.n[1], j.n[5], kT4 / m.rb});
    if (m.re > 0.0) gens.push_back({j.n[2], j.n[6], kT4 / m.re});
    const double t = m.type;
    const auto e = bjt_gp(t * (V(j.n[5]) - V(j.n[6])), t * (V(j.n[5]) - V(j.n[4])), t * (V(j.n[1]) - V(j.n[4])),
                          t * (V(j.n[3]) - V(j.n[4])), m);
    gens.push_back({j.n[4], j.n[6], q2 * std::abs(e.cc)});
    gens.push_back({j.n[5], j.n[6], q2 * std::abs(e.cb)});
  });
  // OSDI devices: the model's own noise sources (white and flicker) at the OP
  forEachOsdi(netlist, [&](const OsdiRef &r) {
    std::vector<double> x(r.n.size());
    for (size_t k = 0; k < x.size(); ++k) x[k] = V(r.n[k]);
    r.in->evaluate(x.data(), osdi::ANALYSIS_NOISE | osdi::CALC_NOISE, 0.0);
    const auto srcs = r.in->noise(1.0);
    for (size_t s = 0; s < srcs.size(); ++s) {
      auto node = [&](int l) { return l < 0 ? 0 : r.n[l]; };
      gens.push_back({node(srcs[s].a), node(srcs[s].b), 0.0,
                      [in = r.in, s](double f) { return in->noise(f)[s].psd; }});
    }
  });

  // ngspice Nintegrate: integral of a density varying as a power of f between points
  constexpr double kMinLog = 1e-38;
  auto nint = [&](double dens, double lnD, double lnLast, double lnF, double lnF0, double dF) {
    double ex = (lnD - lnLast) / (lnF - lnF0);
    if (std::abs(ex) < kMinLog) return dens * dF;
    const double a = std::exp(lnD - ex * lnF);
    ex += 1.0;
    if (std::abs(ex) < kMinLog) return a * (lnF - lnF0);
    return a * (std::exp(ex * lnF) - std::exp(ex * lnF0)) / ex;
  };

  std::vector<double> bin(2 * n, 0.0), bout(2 * n, 0.0), lnLast(gens.size());
  for (int i = 0; i < n; ++i) { bin[i] = ein[i].real(); bin[i + n] = ein[i].imag(); }
  bout[outNode - 1] = 1.0;
  if (refNode > 0) bout[refNode - 1] = -1.0;
  std::unique_ptr<ContractionPlan> adjointPlan;  // R^T has its own pattern
  double onTot = 0.0, inTot = 0.0, lnF0 = 0.0, f0 = 0.0;
  for (size_t k = 0; k < freqs.size(); ++k) {
    const double f = freqs[k], w = 2.0 * M_PI * f;
    const SolverResult fw = solveLinear(acMatrix(sys, w, false), bin);
    const Csr_matrix RT = acMatrix(sys, w, true);
    if (!adjointPlan) adjointPlan = std::make_unique<ContractionPlan>(planContraction(RT));
    const SolverResult z = solveContraction(RT, bout, *adjointPlan);
    if ((int)fw.solution.size() != 2 * n || (int)z.solution.size() != 2 * n) {
      r.error = "noise: linear solve failed at f=" + std::to_string(f);
      return r;
    }
    auto cx = [&](const std::vector<double> &x, int g) {
      return g > 0 ? std::complex<double>(x[g - 1], x[g - 1 + n]) : 0.0;
    };
    const std::complex<double> H = cx(fw.solution, outNode) - cx(fw.solution, refNode);
    const double gainInv = 1.0 / std::max(std::norm(H), kMinLog), lnGainInv = std::log(gainInv);
    const double lnF = std::log(std::max(f, kMinLog));
    double S = 0.0;
    for (size_t g = 0; g < gens.size(); ++g) {
      const double dens = std::norm(cx(z.solution, gens[g].a) - cx(z.solution, gens[g].b)) *
                          (gens[g].psdAt ? gens[g].psdAt(f) : gens[g].psd);
      const double lnD = std::log(std::max(dens, kMinLog));
      if (k > 0) {
        onTot += nint(dens, lnD, lnLast[g], lnF, lnF0, f - f0);
        inTot += nint(dens * gainInv, lnD + lnGainInv, lnLast[g] + lnGainInv, lnF, lnF0, f - f0);
      }
      lnLast[g] = lnD;
      S += dens;
    }
    r.onoise.push_back(std::sqrt(S));
    r.inoise.push_back(std::sqrt(S * gainInv));
    lnF0 = lnF;
    f0 = f;
  }
  r.onoiseTotal = std::sqrt(onTot);
  r.inoiseTotal = std::sqrt(inTot);
  return r;
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
    syncJunctionLimits(netlist, nrEstimate);
    bool nrConverged = false;
    SolverResult subResult{};
    for (int iteration = 0; iteration < MAX_NR_ITER; ++iteration) {
      scheduler.schedule(netlist);
      matrixBuilder.clear();
      matrixBuilder.setDimensions(nTot, nTot);
      rhsVector.assign(nTot, 0.0);

      stampAllElements(netlist, dt_sub, nrEstimate, dt_sub * (k + 1));
      // (no node-to-ground gmin in transient: junction GMIN in the device
      // models keeps floating junction nodes referenced, like ngspice)

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
      syncJunctionLimits(netlist, nrEstimate);  // limit relative to this step's start
      for (int iteration = 0; iteration < MAX_NR_ITER; ++iteration) {
        scheduler.schedule(netlist);
        matrixBuilder.clear();
        matrixBuilder.setDimensions(nTot, nTot);
        rhsVector.assign(nTot, 0.0);

        stampAllElements(netlist, actualTimeStep, nrEstimate,
                         currentTime + actualTimeStep);
        // (no node-to-ground gmin in transient: junction GMIN in the device
        // models keeps floating junction nodes referenced, like ngspice)

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
          netlist, nrEstimate, actualTimeStep, integrator->getType(),
          [&](const std::function<void(double, const std::array<double, 3> &)> &visit) {
            // diode / BJT charge branches, internal nodes included (as the MOS branches)
            auto getV = [&](int n) { return n > 0 && n <= (int)nrEstimate.size() ? nrEstimate[n - 1] : 0.0; };
            forEachJunction(netlist, [&](const Junction &j) {
              if (!j.st || !junctionHasCharge(j)) return;
              std::array<double, 7> x{};
              for (int k = 0; k < j.terms; ++k) x[k] = getV(j.n[k]);
              double v[5], q[5];
              const int K = junctionCharges(j, x, netlist.environment.ambient_temp_K, v, q);
              for (int k = 0; k < K; ++k) visit(v[k], {j.st->v[0][k], j.st->v[1][k], j.st->v[2][k]});
            });
            // OSDI devices: their local node voltages
            forEachOsdi(netlist, [&](const OsdiRef &r) {
              const auto &h = r.in->vHist;
              for (size_t k = 0; k < r.n.size(); ++k) visit(getV(r.n[k]), {h[0][k], h[1][k], h[2][k]});
            });
          });

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

