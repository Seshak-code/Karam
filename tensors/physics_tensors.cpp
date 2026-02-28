#include "../tensors/physics_tensors.h"
#include "kernel_dispatch.h"
#include "../infrastructure/model_registry.h"
#include "../netlist/circuit.h"
#include <iostream>
#include <cmath>
#include <algorithm>

/*
 * physics_tensors.cpp - Implementation of AoS->SoA conversion & Kernel Dispatch.
 */

using namespace acutesim::compute;

// ============================================================================
// KERNEL IMPLEMENTATIONS (GENERIC CPU)
// ============================================================================

void batchDiodePhysics_generic(DiodeTensor& tensor, const std::vector<double>& voltages) {
    size_t n = tensor.size();
    const double GMIN = 1e-12;
    
    for (size_t i = 0; i < n; ++i) {
        int nA = tensor.node_a[i];
        int nC = tensor.node_c[i];
        
        double v_a = (nA > 0 && nA <= (int)voltages.size()) ? voltages[nA - 1] : 0.0;
        double v_c = (nC > 0 && nC <= (int)voltages.size()) ? voltages[nC - 1] : 0.0;
        
        tensor.v_d[i] = v_a - v_c;
        
        double Is = tensor.Is[i];
        double N = tensor.N[i];
        double Vt = tensor.Vt[i];
        double n_vt = N * Vt;
        double Vcrit = 30.0 * n_vt;
        
        double arg = tensor.v_d[i] / n_vt;
        
        // Linear continuation (branchless-style using std::min/max)
        double arg_clamped = std::min(arg, 30.0);
        double exp_arg = std::exp(arg_clamped);
        
        double i_base = Is * (exp_arg - 1.0);
        double g_base = (Is / n_vt) * exp_arg;
        
        double delta = std::max(0.0, tensor.v_d[i] - Vcrit);
        tensor.i_d[i] = i_base + delta * g_base;
        tensor.g_d[i] = std::max(GMIN, g_base);
    }
}

void batchMosfetPhysics_generic(MosfetTensor& tensor, const std::vector<double>& voltages) {
    size_t n = tensor.size();
    
    for (size_t i = 0; i < n; ++i) {
        int nD = tensor.drains[i];
        int nG = tensor.gates[i];
        int nS = tensor.sources[i];
        
        double v_d_orig = (nD > 0 && nD <= (int)voltages.size()) ? voltages[nD - 1] : 0.0;
        double v_g_orig = (nG > 0 && nG <= (int)voltages.size()) ? voltages[nG - 1] : 0.0;
        double v_s_orig = (nS > 0 && nS <= (int)voltages.size()) ? voltages[nS - 1] : 0.0;
        
        bool isPMOS = tensor.isPMOS[i];
        double sign = isPMOS ? -1.0 : 1.0;
        
        // PMOS equivalent: flip all voltages relative to source?
        // Actually legacy flips: Vgs'=-Vgs, Vds'=-Vds, Vth'=-Vth
        double v_g = isPMOS ? v_g_orig : v_g_orig;
        double v_s = isPMOS ? v_s_orig : v_s_orig;
        double v_d = isPMOS ? v_d_orig : v_d_orig;

        tensor.vgs[i] = v_g - v_s;
        tensor.vds[i] = v_d - v_s;
        
        double local_vgs = sign * tensor.vgs[i];
        double local_vds = sign * tensor.vds[i];
        double local_vth = sign * tensor.Vth[i];
        
        double K = 0.5 * tensor.Kp[i] * (tensor.W[i] / tensor.L[i]);
        double vov = local_vgs - local_vth;
        
        if (vov <= 0.0) {
            // Cutoff
            tensor.ids[i] = 0.0;
            tensor.gm[i] = 0.0;
            tensor.gds[i] = 1e-12; // GMIN
        } else if (local_vds < vov) {
            // Linear region
            tensor.ids[i] = sign * K * (2.0 * vov * local_vds - local_vds * local_vds);
            tensor.gm[i] = 2.0 * K * local_vds;
            tensor.gds[i] = 2.0 * K * (vov - local_vds);
        } else {
            // Saturation
            double lam = tensor.lambda[i];
            tensor.ids[i] = sign * K * vov * vov * (1.0 + lam * local_vds);
            tensor.gm[i] = 2.0 * K * vov * (1.0 + lam * local_vds);
            tensor.gds[i] = K * vov * vov * lam;
        }
        
        if (tensor.gds[i] < 1e-12) tensor.gds[i] = 1e-12;
    }
}

void batchBJTPhysics_generic(BJTTensor& tensor, const std::vector<double>& voltages) {
    size_t n = tensor.size();
    const double GMIN = 1e-12;
    
    for (size_t i = 0; i < n; ++i) {
        int nC = tensor.collectors[i];
        int nB = tensor.bases[i];
        int nE = tensor.emitters[i];
        
        double v_c = (nC > 0 && nC <= (int)voltages.size()) ? voltages[nC - 1] : 0.0;
        double v_b = (nB > 0 && nB <= (int)voltages.size()) ? voltages[nB - 1] : 0.0;
        double v_e = (nE > 0 && nE <= (int)voltages.size()) ? voltages[nE - 1] : 0.0;
        
        double sign = tensor.isNPN[i] ? 1.0 : -1.0;
        double v_be = sign * (v_b - v_e);
        double v_bc = sign * (v_b - v_c);
        
        double Is = tensor.Is[i];
        double Vt = tensor.Vt[i];
        double BetaF = tensor.BetaF[i];
        double BetaR = tensor.BetaR[i];
        
        const double Vcrit = 30.0 * Vt;

        auto linear_continuation_diode = [&](double v_j) -> std::pair<double, double> {
            double arg = v_j / Vt;
            double arg_clamped = std::min(arg, 30.0);
            
            double exp_val = std::exp(arg_clamped);
            double i_base = Is * (exp_val - 1.0);
            double g_base = (Is / Vt) * exp_val;
            double g_clamped = std::max(GMIN, g_base);
            
            double delta = std::max(0.0, v_j - Vcrit);
            double i_ext = g_clamped * delta; // Use clamped slope
            
            return {i_base + i_ext, g_clamped};
        };
        
        auto [i_f, g_f] = linear_continuation_diode(v_be);
        auto [i_r, g_r] = linear_continuation_diode(v_bc);
        
        double invBetaF = 1.0 / BetaF;
        double invBetaR = 1.0 / BetaR;
        
        double i_ct = i_f - i_r;
        double ic_local = i_ct - i_r * invBetaR;
        double ib_local = i_f * invBetaF + i_r * invBetaR;
        double ie_local = -ic_local - ib_local;
        
        tensor.Ic[i] = sign * ic_local;
        tensor.Ib[i] = sign * ib_local;
        tensor.Ie[i] = sign * ie_local;
        
        // Conductances
        double dIc_dVbe = g_f;
        double dIc_dVbc = -g_r * (1.0 + invBetaR);
        double dIb_dVbe = g_f * invBetaF;
        double dIb_dVbc = g_r * invBetaR;
        double dIe_dVbe = -(dIc_dVbe + dIb_dVbe);
        double dIe_dVbc = -(dIc_dVbc + dIb_dVbc);
        
        // Terminal Conductances (Common for NPN and PNP)
        // Apply chain rule: dI/dVc = -dI/dVbc, dI/dVe = -dI/dVbe, dI/dVb = dI/dVbe + dI/dVbc
        tensor.g_cc[i] = -dIc_dVbc;
        tensor.g_cb[i] = dIc_dVbe + dIc_dVbc;
        tensor.g_ce[i] = -dIc_dVbe;
        tensor.g_bc[i] = -dIb_dVbc;
        tensor.g_bb[i] = dIb_dVbe + dIb_dVbc;
        tensor.g_be[i] = -dIb_dVbe;
        tensor.g_ec[i] = -dIe_dVbc;
        tensor.g_eb[i] = dIe_dVbe + dIe_dVbc;
        tensor.g_ee[i] = -dIe_dVbe;
        
        // CRITICAL: Add GMIN to diagonal terms to prevent singular Jacobian
        // This is essential for convergence when BJT is in cutoff/low current
        tensor.g_cc[i] = std::max(tensor.g_cc[i], GMIN);
        tensor.g_bb[i] = std::max(tensor.g_bb[i], GMIN);
        tensor.g_ee[i] = std::max(tensor.g_ee[i], GMIN);
    }
}

// ============================================================================
// PUBLIC DISPATCH WRAPPERS
// ============================================================================

void batchDiodePhysics(DiodeTensor& tensor, const std::vector<double>& voltages) {
    KernelDispatcher::get().batch_diode_physics(tensor, voltages);
}

void batchMosfetPhysics(MosfetTensor& tensor, const std::vector<double>& voltages) {
    KernelDispatcher::get().batch_mosfet_physics(tensor, voltages);
}

void batchBJTPhysics(BJTTensor& tensor, const std::vector<double>& voltages) {
    KernelDispatcher::get().batch_bjt_physics(tensor, voltages);
}

// ============================================================================
// AOS -> SOA CONVERSION
// ============================================================================

TensorizedBlock tensorizeBlock(const TensorBlock& block,
                               const std::map<std::string, ModelCard>* modelCards) {
    TensorizedBlock result;
    
    // Convert Resistors
    result.resistors.nodes1.reserve(block.resistors.size());
    result.resistors.nodes2.reserve(block.resistors.size());
    result.resistors.R.reserve(block.resistors.size());
    
    // Safety Clamps for Fixed-Point Solver (1e9 Scale)
    // Max Conductance G_max = 2.0 S => R_min = 0.5 Ohm
    // Min Conductance G_min = 1 nS => R_max = 1 GigaOhm
    const double R_MIN_SAFE = 0.5; 
    const double R_MAX_SAFE = 1.0e9;

    for (const auto& r : block.resistors) {
        double safeR = r.resistance_ohms;
        if (safeR < R_MIN_SAFE) {
            std::cerr << "[WARNING] Resistor clamped: " << safeR << " -> " << R_MIN_SAFE << " (Min Safe R)\n";
            safeR = R_MIN_SAFE;
        } else if (safeR > R_MAX_SAFE) {
             // Not strictly dangerous for crash, but prevents 0 conductance
            safeR = R_MAX_SAFE;
        }
        result.resistors.push(r.nodeTerminal1, r.nodeTerminal2, safeR);
    }
    
    // Convert Capacitors
    for (const auto& c : block.capacitors) {
        result.capacitors.push(c.nodePlate1, c.nodePlate2, c.capacitance_farads);
    }
    
    // Convert Diodes
    result.diodes.reserve(block.diodes.size());
    for (const auto& d : block.diodes) {
        // Rs default 0.0, Temp default passed
        result.diodes.push(d.anode, d.cathode, 
                           d.saturationCurrent_I_S_A, 
                           d.emissionCoefficient_N, 
                           d.thermalVoltage_V_T_V,
                           0.0, // Rs
                           d.temperature_K);
    }
    
    // Convert MOSFETs (with model card lookup)
    result.mosfets.reserve(block.mosfets.size());
    for (const auto& m : block.mosfets) {
        bool isPMOS = false;
        double kp = 200e-6, vth = 0.7, lam = 0.02;

        if (modelCards) {
            auto it = modelCards->find(m.modelName);
            if (it != modelCards->end()) {
                const auto& card = it->second;
                isPMOS = (card.type == "PMOS");
                kp  = card.get("KP", isPMOS ? 100e-6 : 200e-6);
                vth = card.get("VTO", isPMOS ? -0.7 : 0.7);
                lam = card.get("LAMBDA", 0.02);
            } else {
                isPMOS = (m.modelName.find("PMOS") != std::string::npos ||
                          m.modelName.find("pmos") != std::string::npos);
                if (isPMOS) { kp = 100e-6; vth = -0.7; }
            }
        } else {
            isPMOS = (m.modelName.find("PMOS") != std::string::npos ||
                      m.modelName.find("pmos") != std::string::npos);
            if (isPMOS) { kp = 100e-6; vth = -0.7; }
        }

        result.mosfets.push(m.drain, m.gate, m.source, m.body,
                           m.w, m.l,
                           kp, vth, lam,
                           isPMOS, m.temperature_K);
    }
    
    // Convert BJTs
    result.bjts.reserve(block.bjts.size());
    for (const auto& q : block.bjts) {
        result.bjts.push(q.nodeCollector, q.base, q.emitter,
                        q.saturationCurrent_I_S_A,
                        q.betaF, q.betaR,
                        q.thermalVoltage_V_T_V,
                        q.isNPN,
                        q.temperature_K);
    }

    // Convert Sources
    for (const auto& v : block.voltageSources) {
        result.voltageSources.push(v.nodePositive, v.nodeNegative, v.voltage_V);
    }
    for (const auto& i : block.currentSources) {
        result.currentSources.push(i.nodePositive, i.nodeNegative, i.current_A);
    }
    
    return result;
}

TensorizedBlock tensorizeNetlist(const TensorNetlist& netlist) {
    // 1. Start with the flattened global block (standard components)
    const auto* cards = netlist.modelCards.empty() ? nullptr : &netlist.modelCards;
    TensorizedBlock result = tensorizeBlock(netlist.globalBlock, cards);
    
    // -------------------------------------------------------------------------
    // BUCKETING STAGING AREA: Group devices by model to maximize SIMD efficiency
    // -------------------------------------------------------------------------
    struct DiodeJob { int a, c; double is, n, vt; std::string model; };
    struct MosfetJob { int d, g, s, b; double w, l, kp, vth, lam; bool pmos; std::string model; };
    struct BjtJob { int c, b, e; double is, bf, br, vt; bool npn; std::string model; };

    std::map<std::string, std::vector<DiodeJob>> diodeBuckets;
    std::map<std::string, std::vector<MosfetJob>> mosfetBuckets;
    std::map<std::string, std::vector<BjtJob>> bjtBuckets;

    std::map<std::string, size_t> typeCounts;
    auto& registry = ModelRegistry::instance();

    auto countBlock = [&](const TensorBlock& block) {
        for (const auto& gc : block.genericComponents) {
            const ModelInfo* info = registry.findModelByName(gc.modelName); // Helper needed or use getModel with known type?
            // Problem: We don't know the type from just modelName efficiently without scanning all types?
            // Optimization: ModelRegistry should map modelName -> DeviceType or ModelInfo directly.
            // For now, we iterate types to find it (slow but safe).
            // Better: Add lookup to registry.
        }
    };
    
    // We need a fast lookup for model name to info. 
    // Let's assume we can build a cache or modify registry. 
    // See below for Registry helper assumption.
    
    // Temporary lambda to find model info
    auto findInfo = [&](const std::string& name) -> const ModelInfo* {
        // Try to guess type or search all. 
        // This is inefficient. We should have ModelRegistry::findModel(name).
        // Let's implement a naive search here or rely on specific naming convention.
        // Actually, let's use the listDeviceTypes loop.
        auto types = registry.listDeviceTypes();
        for(const auto& t : types) {
             if(const ModelInfo* info = registry.getModel(t, name)) return info;
        }
        return nullptr;
    };

    // Count Global
    for (const auto& gc : netlist.globalBlock.genericComponents) {
        if (const ModelInfo* info = findInfo(gc.modelName)) {
            typeCounts[info->deviceType]++;
        }
    }
    
    // Count Instances
    for (const auto& instance : netlist.instances) {
        auto it = netlist.blockDefinitions.find(instance.blockName);
        if (it != netlist.blockDefinitions.end()) {
            for (const auto& gc : it->second.genericComponents) {
                 if (const ModelInfo* info = findInfo(gc.modelName)) {
                    typeCounts[info->deviceType]++;
                 }
            }
        }
    }

    // Allocate Tensors
    for (const auto& [type, count] : typeCounts) {
        if (count == 0) continue;
        // Get any model of this type to access create/resize functions
        auto models = registry.listModels(type);
        if (models.empty()) continue;
        const ModelInfo* factory = registry.getModel(type, models[0]);
        
        if (factory && factory->createTensor && factory->resizeTensor) {
            void* tensor = factory->createTensor();
            factory->resizeTensor(tensor, count);
            result.dynamicTensors[type] = {tensor, count};
        }
    }

    // -------------------------------------------------------------------------
    // DYNAMIC MODEL PASS 2: POPULATION
    // -------------------------------------------------------------------------
    std::map<std::string, size_t> typeIndices; // Current write index

    auto populateGeneric = [&](const TensorBlock& block, auto remap) {
        for (const auto& gc : block.genericComponents) {
            const ModelInfo* info = findInfo(gc.modelName);
            if (!info) {
                 std::cerr << "[Warning] Generic component '" << gc.instanceName 
                           << "' uses unknown model '" << gc.modelName << "'\n";
                 continue;
            }
            
            void* tensor = result.dynamicTensors[info->deviceType].tensor;
            if (!tensor) continue;
            
            size_t idx = typeIndices[info->deviceType]++;
            
            // Remap nodes
            std::vector<int> mappedNodes;
            mappedNodes.reserve(gc.nodes.size());
            for(int n : gc.nodes) mappedNodes.push_back(remap(n));
            
            // Call setInstance
            if (info->setInstance) {
                info->setInstance(tensor, idx, mappedNodes.data(), gc.params.data());
            }
        }
    };

    auto collectFromBlock = [&](const TensorBlock& block, const std::function<int(int)>& remap) {
        // Collect Diodes
        for (const auto& d : block.diodes) {
            double vts = d.thermalVoltage_V_T_V * (netlist.environment.ambient_temp_K / 300.15);
            diodeBuckets[d.modelName].push_back({remap(d.anode), remap(d.cathode), 
                                                d.saturationCurrent_I_S_A, d.emissionCoefficient_N, vts, d.modelName});
        }
        // Collect MOSFETs
        for (const auto& m : block.mosfets) {
            bool isPMOS = false;
            double kp = 200e-6, vth = 0.7, lam = 0.02;
            if (cards) {
                auto cit = cards->find(m.modelName);
                if (cit != cards->end()) {
                    isPMOS = (cit->second.type == "PMOS");
                    kp  = cit->second.get("KP", isPMOS ? 100e-6 : 200e-6);
                    vth = cit->second.get("VTO", isPMOS ? -0.7 : 0.7);
                    lam = cit->second.get("LAMBDA", 0.02);
                } else {
                    isPMOS = (m.modelName.find("PMOS") != std::string::npos || m.modelName.find("pmos") != std::string::npos);
                    if (isPMOS) { kp = 100e-6; vth = -0.7; }
                }
            }
            mosfetBuckets[m.modelName].push_back({remap(m.drain), remap(m.gate), remap(m.source), remap(m.body),
                                                 m.w, m.l, kp, vth, lam, isPMOS, m.modelName});
        }
        // Collect BJTs
        for (const auto& q : block.bjts) {
            double vts = q.thermalVoltage_V_T_V * (netlist.environment.ambient_temp_K / 300.15);
            bjtBuckets[q.instanceName].push_back({remap(q.nodeCollector), remap(q.base), remap(q.emitter),
                                                 q.saturationCurrent_I_S_A, q.betaF, q.betaR, vts,
                                                 q.isNPN, q.instanceName});
        }

        // Linear components (Resistors, Capacitors, Inductors)
        for (const auto& r : block.resistors) 
            result.resistors.push(remap(r.nodeTerminal1), remap(r.nodeTerminal2), r.resistance_ohms);
        for (const auto& c : block.capacitors)
            result.capacitors.push(remap(c.nodePlate1), remap(c.nodePlate2), c.capacitance_farads);
        for (const auto& ind : block.inductors)
            result.inductors.push(remap(ind.nodeCoil1), remap(ind.nodeCoil2), ind.inductance_henries);
        for (const auto& v : block.voltageSources)
            result.voltageSources.push(remap(v.nodePositive), remap(v.nodeNegative), v.voltage_V);
        for (const auto& i : block.currentSources)
            result.currentSources.push(remap(i.nodePositive), remap(i.nodeNegative), i.current_A);
    };

    // 2. Gather & Populate
    // Reset result to start clean (tensorizeBlock at 307 already pushed some things)
    result.clear(); 
    
    // Global
    auto identity = [](int n){ return n; };
    collectFromBlock(netlist.globalBlock, identity);
    populateGeneric(netlist.globalBlock, identity);

    // Instances
    for (const auto& inst : netlist.instances) {
        auto it = netlist.blockDefinitions.find(inst.blockName);
        if (it == netlist.blockDefinitions.end()) continue;
        auto remap = [&](int n) { return (n <= 0 || n > (int)inst.nodeMapping.size()) ? 0 : inst.nodeMapping[n-1]; };
        collectFromBlock(it->second, remap);
        populateGeneric(it->second, remap);
    }

    // 3. Flush Buckets (Deterministic sort by model name)
    for (auto& [model, bucket] : diodeBuckets) {
        for (auto& d : bucket) result.diodes.push(d.a, d.c, d.is, d.n, d.vt, 0.0, netlist.environment.ambient_temp_K);
    }
    for (auto& [model, bucket] : mosfetBuckets) {
        for (auto& m : bucket) result.mosfets.push(m.d, m.g, m.s, m.b, m.w, m.l, m.kp, m.vth, m.lam, m.pmos, netlist.environment.ambient_temp_K);
    }
    for (auto& [model, bucket] : bjtBuckets) {
        for (auto& q : bucket) result.bjts.push(q.c, q.b, q.e, q.is, q.bf, q.br, q.vt, q.npn, netlist.environment.ambient_temp_K);
    }
    
    return result;
}
