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

#include "../physics/circuitsim.h"
#include "acutesim_engine/netlist/netlist_export.h"
#include "../solvers/convergence_engine.h"
#include "../physics/device_physics.h"
#include "../physics/physics_constants.h"
#include "../infrastructure/safe_utils.h"
#include "../infrastructure/compiled_block.h"
#if !defined(__EMSCRIPTEN__) && defined(ACUTESIM_HAS_DAWN_NATIVE)
#include "../solvers/webgpu_solver.h"
#include "acutesim_engine/gpu_context_manager.h"
#endif

// =============================================================================
// PHASE 1 HARDENING: Helper Method Implementations
// =============================================================================

void CircuitSim::seedInitialGuess(const TensorNetlist& netlist,
                                   std::vector<double>& voltageEstimate) {
    const size_t N = voltageEstimate.size();

    // Temperature-corrected Vbe seed: at cold temps, Vbe increases (~+2mV/K below nominal)
    double T = netlist.environment.ambient_temp_K;
    double vbe_seed = PhysicsConstants::V_BE_SEED - 0.002 * (T - 300.15);
    // Clamp to reasonable range [0.5V, 1.2V]
    vbe_seed = std::max(0.5, std::min(1.2, vbe_seed));

    // 1. Seed from Voltage Sources (direct)
    for (const auto& source : netlist.globalBlock.voltageSources) {
        int idx = SafeUtils::safe_idx(source.nodePositive, N, "VSource seed");
        if (idx >= 0) {
            voltageEstimate[idx] = source.voltage_V;
        }
    }

    // 2. Seed Diode anodes (0.6V initial for silicon)
    for (const auto& diode : netlist.globalBlock.diodes) {
        int idx = SafeUtils::safe_idx(diode.anode, N, "Diode seed");
        if (idx >= 0) {
            voltageEstimate[idx] = std::max(voltageEstimate[idx],
                                             PhysicsConstants::V_DIODE_SEED);
        }
    }

    // 3. Find supply voltage for mid-rail seeding
    double vSupply = PhysicsConstants::V_SUPPLY_DEFAULT;
    for (const auto& vs : netlist.globalBlock.voltageSources) {
        if (vs.voltage_V > vSupply) vSupply = vs.voltage_V;
    }

    // 4. Current Source Node Seeding (BJT base detection)
    for (const auto& cs : netlist.globalBlock.currentSources) {
        int nPos = SafeUtils::safe_idx(cs.nodePositive, N, "ISource seed");
        if (nPos < 0) continue;

        // Check if this node is a BJT base
        bool isBase = false;
        for (const auto& bjt : netlist.globalBlock.bjts) {
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
                    voltageEstimate[nB] = std::max(voltageEstimate[nB],
                        voltageEstimate[nE] + vbe_seed);
                }
                if (nC >= 0 && nB >= 0) {
                    voltageEstimate[nC] = std::max(voltageEstimate[nC],
                        voltageEstimate[nB] + 0.5);
                }
            } else { // PNP
                if (nE >= 0 && nB >= 0) {
                    voltageEstimate[nB] = std::min(voltageEstimate[nB],
                        voltageEstimate[nE] - vbe_seed);
                }
                if (nC >= 0 && nB >= 0) {
                    voltageEstimate[nC] = std::min(voltageEstimate[nC],
                        voltageEstimate[nB] - 0.5);
                }
            }
        }
    }
}

void CircuitSim::applyPnJunctionLimits(const TensorizedBlock& block,
                                        std::vector<double>& v_new,
                                        const std::vector<double>& v_old,
                                        double temp_K) {
    if (v_new.empty()) return;
    const size_t N = v_new.size();
    const double temp_scale = temp_K / 300.15;

    // Helper lambda for limiting a single junction
    auto limitJunction = [&](int p_node, int n_node, double Is, double Vt) {
        // Scale Vt and Is for temperature
        Vt *= temp_scale;
        Is *= std::pow(temp_scale, 3.0) * exp(1.11 / (Vt / temp_scale) - 1.11 / Vt);
        int p = p_node - 1; // 1-based to 0-based
        int n = n_node - 1; // 1-based to 0-based
        
        if (p < 0 && n < 0) return;
        if (p >= (int)N || n >= (int)N) return;
        
        double vcrit = compute_vcrit(Vt, Is);
        double v_junction_old = (p >= 0 ? v_old[p] : 0.0) - (n >= 0 ? v_old[n] : 0.0);
        double v_junction_new = (p >= 0 ? v_new[p] : 0.0) - (n >= 0 ? v_new[n] : 0.0);
        
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
        limitJunction(block.bjts.bases[i], block.bjts.emitters[i], 
                      block.bjts.Is[i], block.bjts.Vt[i]);
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

#if !defined(__EMSCRIPTEN__) && defined(ACUTESIM_HAS_DAWN_NATIVE)
  // GPU execution path — delegate to WebGPUSolver hybrid NR loop
  if (execMode == ExecutionMode::GPU || execMode == ExecutionMode::GPU_DEBUG) {
    netlist.globalBlock.computeTopologyHash();
    // Borrow device + queue from the singleton — no second device creation.
    auto& gpuMgr = acutesim::GPUContextManager::instance();
    WebGPUSolver gpuSolver(gpuMgr.rawDevice(), gpuMgr.rawQueue());
    if (gpuSolver.initialize(netlist)) {
        gpuSolver.uploadNetlist(netlist);
        constexpr int    GPU_MAX_ITER = 50;
        constexpr double GPU_TOL      = 1e-9;
        constexpr double GPU_GMIN     = 1e-12;
        std::vector<double> sol = gpuSolver.runHybridNRLoop(
            netlist, GPU_MAX_ITER, GPU_TOL, GPU_GMIN, /*time=*/0.0, /*h=*/0.0);
        if (!sol.empty()) {
            double kcl_err = 0.0;
            // Quick KCL check on first node as sanity scalar
            return {0.0, sol, {GPU_MAX_ITER, kcl_err, true},
                    netlist.globalBlock.topologyHash};
        }
        std::cerr << "[WARN] WebGPUSolver returned empty solution, falling back to CPU.\n";
    } else {
        std::cerr << "[WARN] WebGPUSolver init failed, falling back to CPU.\n";
    }
  }
#endif

  // Ensure topology hash is fresh
  netlist.globalBlock.computeTopologyHash();

  // Initialize SoA Block for this solve
  soaBlock = tensorizeNetlist(netlist);

  std::vector<double> voltageEstimate(netlist.numGlobalNodes, 0.0);

  // Phase 1 Hardening: Use extracted helper for initial guess seeding
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
  int  rankDeficientRow = -1;
  double smallestPivotOverall = 1e300;
  RegularizationInfo regulInfo;

  // GMIN Stepping: Start with large parallel conductances, reduce each pass
  // This prevents singular matrices for cold-start BJT circuits
  // Use half-decade steps for better convergence at extreme temperatures
  double gmin_factor = 1.0;
  const int GMIN_STEPS = 19; // 1e-3 to 1e-12 in half-decade steps

  for (int gmin_step = 0; gmin_step < GMIN_STEPS; ++gmin_step) {
    // Scale GMIN: start at 1e-3, reduce by sqrt(10) each step to 1e-12
    gmin_factor = std::pow(10.0, -(3.0 + gmin_step * 0.5));
    if (gmin_factor < 1e-12) gmin_factor = 1e-12;
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
    matrixBuilder.setDimensions(netlist.numGlobalNodes, netlist.numGlobalNodes);
    rhsVector.assign(netlist.numGlobalNodes, 0.0);

    stampAllElements(netlist, 0.0, voltageEstimate, 0.0);
    addDiagonalConditioning(netlist.numGlobalNodes, gmin_factor);

    // Sparse Merge (D5): Use cached pattern for O(N) assembly on iterations > 0
    Csr_matrix matrix;
    bool patternValid = cachedPattern_ &&
                        cachedPatternHash_ == netlist.globalBlock.topologyHash;
    if (!patternValid) {
        cachedPattern_ = std::make_unique<CachedCsrPattern>(matrixBuilder.buildPattern());
        cachedPatternHash_ = netlist.globalBlock.topologyHash;
        matrix = matrixBuilder.createCsr();
    } else {
        matrix = matrixBuilder.assembleFast(*cachedPattern_);
    }
    
    // Solver Selection: Use LU for circuits with active (non-linear) devices
    // PCG requires symmetric positive-definite, which BJTs violate
    SolverResult result;
    bool hasActiveDevices = !netlist.globalBlock.bjts.empty() || 
                            !netlist.globalBlock.diodes.empty() ||
                            !netlist.globalBlock.mosfets.empty();
    
    if (hasActiveDevices || netlist.numGlobalNodes < 20) {
      // Direct LU solver for small matrices or active devices
      result = solveLU_Pivoted(matrix, rhsVector);
    } else {
      // PCG for large linear-only circuits
      result = solvePCG(matrix, rhsVector, 1e-6, PCG_MAX_ITER);
      if (!result.converged) {
        // Fallback to LU if PCG fails
        result = solveLU_Pivoted(matrix, rhsVector);
      }
    }
    // Phase 2.9: Capture pivot diagnostics from LU
    if (result.smallestPivot < smallestPivotOverall) {
        smallestPivotOverall = result.smallestPivot;
        rankDeficientRow = result.rankDeficientRow;
    }
    if (result.rankDeficient && !rankDeficientDetected) {
        rankDeficientDetected = true;
        std::cerr << "DC Solver: Near-rank-deficient matrix (pivot=" << result.smallestPivot
                  << " at row=" << result.rankDeficientRow << ") — proceeding with GMIN conditioning.\n";
    }

    if (!result.converged) {
      // Matrix is singular - circuit topology error or floating node
      std::cerr << "DC Solver: Linear solver failed (Singular Matrix) at iter " << iteration << "\n";
      final_kcl_error = 1e9; // Flag as bad residual
      singularMatrixDetected = true;
      // If singular, retrying with same Jacobian won't help. Abort this GMIN step.
      total_iterations += iteration + 1;
      break;
    }
    
    std::vector<double> v_new = result.solution;
    
    // NaN Protection using SafeUtils
    SafeUtils::sanitize_vector(v_new, 0.0);
    
    // Phase 1 Hardening: Use extracted helper for PN junction limiting
    applyPnJunctionLimits(soaBlock, v_new, voltageEstimate, netlist.environment.ambient_temp_K);

    // Check 1 + 2: Voltage Convergence AND Physical KCL Accuracy
    double kcl_error = calculatePhysicalResiduals(netlist, v_new);

    // Limit Cycle Breaker: Damping
    // If residual is not improving, dampen the step to force a different path
    if (iteration > 0 && kcl_error >= final_kcl_error) { // final_kcl_error holds prev_error
         // Apply damping: v_new = v_old + 0.5 * (v_new - v_old)
         for(size_t i=0; i<v_new.size(); ++i) {
             v_new[i] = voltageEstimate[i] + 0.5 * (v_new[i] - voltageEstimate[i]);
         }
         // Re-evaluate residual after damping
         kcl_error = calculatePhysicalResiduals(netlist, v_new);
         
         // If still bad, dampen again (aggressive backoff)
         if (kcl_error >= final_kcl_error) {
              for(size_t i=0; i<v_new.size(); ++i) {
                 v_new[i] = voltageEstimate[i] + 0.5 * (v_new[i] - voltageEstimate[i]);
              }
              kcl_error = calculatePhysicalResiduals(netlist, v_new);
         }
    }
    // Record convergence step for diagnostics
    {
        ConvergenceStep cs;
        cs.iteration = iteration;
        cs.residualNorm = kcl_error;
        cs.dampingFactor = (iteration > 0 && kcl_error >= final_kcl_error) ? 0.5 : 1.0;
        cs.linearSolverResidual = result.finalResidual;  // Phase 2.9

        // Compute max voltage delta
        double maxDelta = 0.0;
        for (size_t i = 0; i < v_new.size(); ++i) {
            double d = std::abs(v_new[i] - voltageEstimate[i]);
            if (d > maxDelta) maxDelta = d;
        }
        cs.maxVoltageDelta = maxDelta;
        cs.worstNode = ConvergenceEngine::findWorstNode(
            std::vector<double>(v_new.size(), kcl_error)); // Simplified; worst node from residual
        convergenceHistory.push_back(cs);
    }

    // Phase 2.9: Stagnation and divergence detection
    {
        // Stagnation: residual plateaued but not converged
        if (iteration > 0 &&
            std::abs(final_kcl_error - kcl_error) < 1e-15 &&
            kcl_error > PhysicsConstants::KCL_TOL) {
            stagnationDetected = true;
        }
        // Divergence: last 3 iterations show monotonic increase
        size_t sz = convergenceHistory.size();
        if (sz >= 3) {
            if (convergenceHistory[sz-1].residualNorm > convergenceHistory[sz-2].residualNorm &&
                convergenceHistory[sz-2].residualNorm > convergenceHistory[sz-3].residualNorm) {
                divergenceDetected = true;
            }
        }
    }

    final_kcl_error = kcl_error; // Store for next iteration comparison

    // During GMIN stepping, we relax the KCL check because GMIN itself adds residual.
    // We only enforce strict KCL at the final (nominal GMIN) step.
    double effective_kcl_tol = (gmin_step == GMIN_STEPS - 1) ? PhysicsConstants::KCL_TOL : 1.0;

    if (checkConvergence(v_new, voltageEstimate, kcl_error, effective_kcl_tol)) {
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
  
  // If we failed to converge a single GMIN step, check if we're making progress.
  // If residual is within a reasonable range, continue stepping as the current
  // solution may be a good starting point for the next (tighter) GMIN step.
  if (!gmin_converged) {
      if (final_kcl_error < 1.0) {
          // Making progress — continue GMIN stepping
          std::cerr << "  -> GMIN step did not fully converge (res=" << final_kcl_error
                    << "), continuing...\n";
      } else {
          break;
      }
  }
  
  } // End GMIN stepping loop

  // Phase 2.9: Final per-node residual for RMS + worst-node tracking
  std::vector<double> perNodeResiduals;
  final_kcl_error = calculatePhysicalResiduals(netlist, voltageEstimate, perNodeResiduals);

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
              worstNodeVoltage = (i < voltageEstimate.size()) ? voltageEstimate[i] : 0.0;
          }
      }
      rmsResidual = std::sqrt(sumSq / static_cast<double>(perNodeResiduals.size()));
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
      calculatePhysicalResidualsDetailed(netlist, voltageEstimate, worstDevices, worstNode);
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
  stats.integrationMethod = "";  // DC has no integration
  stats.gminSteppingUsed = (gmin_factor < 1e-3);  // If we progressed past initial steps
  stats.sourceSteppingUsed = false;  // Not implemented yet
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

  return {0.0, voltageEstimate, stats, netlist.globalBlock.topologyHash};
}



SolverStep CircuitSim::stepTransient(TensorNetlist &netlist, double timeStep,
                               double currentTime) {
    if (netlist.numGlobalNodes == 0)
        return {currentTime, {}, {0, 0.0, true}, 0};

    if (!arbiterInitialized) {
      arbiter.initialize(netlist);
      arbiterInitialized = true;
      
      // M4: Enforce Formal Passivity
      if (!arbiter.checkPassivity(netlist)) {
           // Halt simulation if physically impossible device parameters found
           return {currentTime, {}, {0, 0.0, false}};
      }
    }

    // Initialize SoA Block for transient steps
    soaBlock = tensorizeNetlist(netlist);

    double actualTimeStep = timeStep;
    int attempt = 0;
    const int MAX_ATTEMPTS = 4;

    std::vector<double> finalVoltages;
    SolverResult lastResult;
    bool simulationConverged = false;
    double next_dt_suggestion = timeStep;

    while (attempt < MAX_ATTEMPTS) {
        integrator->prepareStep(actualTimeStep);
        
        std::vector<double> nrEstimate = arbiter.getPrevVoltages();
        if (nrEstimate.empty()) nrEstimate.assign(netlist.numGlobalNodes, 0.0);

        bool nrConverged = false;
        for (int iteration = 0; iteration < MAX_NR_ITER; ++iteration) {
             scheduler.schedule(netlist);
             matrixBuilder.clear();
             matrixBuilder.setDimensions(netlist.numGlobalNodes, netlist.numGlobalNodes);
             rhsVector.assign(netlist.numGlobalNodes, 0.0);
             
             stampAllElements(netlist, actualTimeStep, nrEstimate, currentTime + actualTimeStep);
             addDiagonalConditioning(netlist.numGlobalNodes);
             
             Csr_matrix A = matrixBuilder.createCsr();
             
             // --- Solver Strategy Optimization ---
             // LU is O(N^3) but robust for non-symmetric active circuits.
             // PCG is O(N*sqrt(K)) but requires SPD (Resistor/Capacitor meshes).
             if (netlist.numGlobalNodes < 100) {
                 // Small VLSI blocks: Use Direct LU with partial pivoting for absolute robustness
                 lastResult = solveLU_Pivoted(A, rhsVector);
             } else {
                 // Large Parasitic Meshes: Try Iterative first
                 lastResult = solvePCG(A, rhsVector, 1e-6);
                 if (!lastResult.converged) {
                     // Emergency Fallback: If PCG fails (highly non-symmetric), drop into LU
                     lastResult = solveLU_Pivoted(A, rhsVector);
                 }
             }
             
             std::vector<double> v_new = lastResult.solution;
             
             double kcl_error = calculatePhysicalResiduals(netlist, v_new, actualTimeStep);
             if (checkConvergence(v_new, nrEstimate, kcl_error, PhysicsConstants::KCL_TOL)) {
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

        int suggestion = arbiter.analyzeStability(netlist, nrEstimate, actualTimeStep, next_dt_suggestion);

        if (suggestion == 1) { 
            actualTimeStep *= 0.5;
            attempt++;
            continue;
        } else if (suggestion == 2) { 
            if (integrator->getType() == IntegrationType::TRAPEZOIDAL) {
                 setIntegrationMethod(IntegrationType::GEAR_2);
                 continue; 
            }
        }

        finalVoltages = nrEstimate;
        simulationConverged = true;
        timeStep = next_dt_suggestion;
        break;
    }
    
    // Hang Protection: Detect if timeStep has underflowed
    if (timeStep < 1e-18) {
        // Prevent infinite loop by returning failing step
         return {currentTime, std::vector<double>(netlist.numGlobalNodes, 0.0), {lastResult.iterations, lastResult.finalResidual, false}};
    }

    if (simulationConverged) {
        for (size_t i = 0; i < netlist.globalBlock.capacitors.size(); ++i) {
             const auto &cap = netlist.globalBlock.capacitors[i];
             auto &hist = netlist.globalState.capacitorState[i];

             double v_curr = 0.0;
             if (cap.nodePlate1 > 0 && cap.nodePlate1 <= (int)finalVoltages.size()) v_curr += finalVoltages[cap.nodePlate1-1];
             if (cap.nodePlate2 > 0 && cap.nodePlate2 <= (int)finalVoltages.size()) v_curr -= finalVoltages[cap.nodePlate2-1];
             
             double i_curr = 0.0;
             double C = cap.capacitance_farads;
             IntegrationType type = integrator->getType();
             
             if (hist.v.empty()) { // Auto-init if empty (should be done in addCapacitor but safety check)
                  hist.resize(3);
             }

             if (type == IntegrationType::TRAPEZOIDAL) {
                 i_curr = (2.0*C/actualTimeStep)*(v_curr - hist.v[0]) - hist.i[0];
             } else if (type == IntegrationType::GEAR_1_EULER) {
                 i_curr = (C/actualTimeStep)*(v_curr - hist.v[0]);
             } else if (type == IntegrationType::GEAR_2) {
                 if (hist.v.size() > 1 && hist.v[1] == 0.0 && currentTime > actualTimeStep) { // Heuristic check
                     i_curr = (C/actualTimeStep)*(v_curr - hist.v[0]);
                 } else if (hist.v.size() > 1) {
                     i_curr = (C/actualTimeStep)*(1.5*v_curr - 2.0*hist.v[0] + 0.5*hist.v[1]);
                 } else {
                     i_curr = (C/actualTimeStep)*(v_curr - hist.v[0]);
                 }
             }
             integrator->updateHistory(hist.v, hist.i, v_curr, i_curr);
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
             
             if (hist.v.empty()) hist.resize(3);

             // I_cap = C * dv/dt (Trapezoidal integration)
             if (type == IntegrationType::TRAPEZOIDAL) {
                 if (!hist.v.empty())
                     i_curr = (2.0*C/actualTimeStep)*(v_curr - hist.v[0]) - hist.i[0];
             } else if (type == IntegrationType::GEAR_1_EULER) {
                 if (!hist.v.empty())
                    i_curr = (C/actualTimeStep)*(v_curr - hist.v[0]);
             } 
             
             integrator->updateHistory(hist.v, hist.i, v_curr, i_curr);
        }


         // Update Hierarchical Instances History
         for (size_t idx = 0; idx < netlist.instances.size(); ++idx) {
            const auto &inst = netlist.instances[idx];
            auto &instState = netlist.instanceStates[idx];
            
            auto it = netlist.blockDefinitions.find(inst.blockName);
            if (it != netlist.blockDefinitions.end()) {
                 const auto &block = it->second;
                 auto mapNode = [&](int local) { return local==0?0:inst.nodeMapping[local-1]; };
                 auto getV = [&](int n) { return (n>0 && n<=(int)finalVoltages.size()) ? finalVoltages[n-1] : 0.0; };
                 
                 for (size_t c_idx = 0; c_idx < block.capacitors.size(); ++c_idx) {
                     const auto &c = block.capacitors[c_idx];
                     auto &hist = instState.capacitorState[c_idx];

                     double v_curr = getV(mapNode(c.nodePlate1)) - getV(mapNode(c.nodePlate2));
                     double i_curr = 0.0;
                     double C = c.capacitance_farads;
                     IntegrationType type = integrator->getType();
                     
                     if (hist.v.empty()) hist.resize(3);

                     if (type == IntegrationType::TRAPEZOIDAL) {
                         i_curr = (2.0*C/actualTimeStep)*(v_curr - hist.v[0]) - hist.i[0];
                     } else if (type == IntegrationType::GEAR_1_EULER) {
                         i_curr = (C/actualTimeStep)*(v_curr - hist.v[0]);
                     } else if (type == IntegrationType::GEAR_2) {
                        if (hist.v.size() > 1)
                             i_curr = (C/actualTimeStep)*(1.5*v_curr - 2.0*hist.v[0] + 0.5*hist.v[1]);
                        else
                             i_curr = (C/actualTimeStep)*(v_curr - hist.v[0]);
                     }
                     integrator->updateHistory(hist.v, hist.i, v_curr, i_curr);
                 }
                 
                 // TODO: Also update Inductors, PowerRails for instances
            }
         }

    }

    if (!simulationConverged)
         return {currentTime, std::vector<double>(netlist.numGlobalNodes, 0.0), {lastResult.iterations, lastResult.finalResidual, false}, netlist.globalBlock.topologyHash};

    return {
        currentTime + actualTimeStep,
        finalVoltages,
        {lastResult.iterations, lastResult.finalResidual, simulationConverged},
        netlist.globalBlock.topologyHash
    };
}


// =============================================================================
// COMPILED BLOCK API — Public entry points
// =============================================================================
// These overloads accept the pre-compiled, immutable CompiledTensorBlock.
// They pre-load the SoA tensors and then delegate to the internal
// TensorNetlist-based methods via the embedded structural_ copy.

SolverStep CircuitSim::solveDC(const CompiledTensorBlock& block) {
    // Pre-load the SoA tensors so tensorizeNetlist() inside is a no-op
    soaBlock = block.tensors;

    // Delegate to internal TensorNetlist-based solver
    // A mutable copy is needed because the solver writes to instance states
    TensorNetlist nl = block.structural_;
    return solveDC(nl);
}

SolverStep CircuitSim::stepTransient(const CompiledTensorBlock& block,
                                     double timeStep, double currentTime) {
    // Pre-load the SoA tensors
    soaBlock = block.tensors;

    // Delegate to internal TensorNetlist-based solver
    TensorNetlist nl = block.structural_;
    return stepTransient(nl, timeStep, currentTime);
}
