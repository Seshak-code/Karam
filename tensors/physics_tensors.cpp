#include "../tensors/physics_tensors.h"
#include "../physics/physics_constants.h"
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
    const double GMIN = PhysicsConstants::JUNCTION_GMIN;
    
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
        double Vcrit = PhysicsConstants::JUNCTION_EXP_ARG_MAX * n_vt;
        
        double arg = tensor.v_d[i] / n_vt;
        
        // Linear continuation (branchless-style using std::min/max)
        double arg_clamped = std::min(arg, PhysicsConstants::JUNCTION_EXP_ARG_MAX);
        double exp_arg = std::exp(arg_clamped);
        
        double i_base = Is * (exp_arg - 1.0);
        double g_base = (Is / n_vt) * exp_arg;
        
        double delta = std::max(0.0, tensor.v_d[i] - Vcrit);
        // ngspice GMIN: parallel junction conductance — mirrors diode_model_pair()
        tensor.i_d[i] = i_base + delta * g_base + GMIN * tensor.v_d[i];
        tensor.g_d[i] = g_base + GMIN;
    }
}

void batchBJTPhysics_generic(BJTTensor& tensor, const std::vector<double>& voltages) {
    size_t n = tensor.size();
    const double GMIN = PhysicsConstants::JUNCTION_GMIN;
    
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
        
        const double Vcrit = PhysicsConstants::JUNCTION_EXP_ARG_MAX * Vt;

        auto linear_continuation_diode = [&](double v_j) -> std::pair<double, double> {
            double arg = v_j / Vt;
            double arg_clamped = std::min(arg, PhysicsConstants::JUNCTION_EXP_ARG_MAX);
            
            double exp_val = std::exp(arg_clamped);
            double i_base = Is * (exp_val - 1.0);
            double g_base = (Is / Vt) * exp_val;
            // continuation with the junction's own slope, plus the parallel GMIN —
            // exactly diode_model_pair()
            double delta = std::max(0.0, v_j - Vcrit);
            return {i_base + g_base * delta + GMIN * v_j, g_base + GMIN};
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
        // (no max(g, GMIN) floors: the junction GMIN above keeps the matrix
        // regular, and a floor would be conductance with no matching current)
    }
}

// ============================================================================
// PUBLIC DISPATCH WRAPPERS
// ============================================================================

void batchDiodePhysics(DiodeTensor& tensor, const std::vector<double>& voltages) {
    KernelDispatcher::get().batch_diode_physics(tensor, voltages);
}


void batchBJTPhysics(BJTTensor& tensor, const std::vector<double>& voltages) {
    KernelDispatcher::get().batch_bjt_physics(tensor, voltages);
}

// ============================================================================
// AOS -> SOA CONVERSION
// ============================================================================

TensorizedBlock tensorizeBlock(const TensorBlock& block) {
    TensorizedBlock result;
    
    // Convert Resistors
    result.resistors.nodes1.reserve(block.resistors.size());
    result.resistors.nodes2.reserve(block.resistors.size());
    result.resistors.R.reserve(block.resistors.size());
    
    // Exact values: the 0.5 ohm .. 1 Gohm clamps (for a fixed-point solver that no
    // longer exists) doubled every sub-ohm power-grid resistor and disagreed with
    // the KCL audit. R <= 0 is rejected by the parser.
    for (const auto& r : block.resistors)
        result.resistors.push(r.nodeTerminal1, r.nodeTerminal2, r.resistance_ohms);
    
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
    TensorizedBlock result = tensorizeBlock(netlist.globalBlock);
    
    // -------------------------------------------------------------------------
    // BUCKETING STAGING AREA: Group devices by model to maximize SIMD efficiency
    // -------------------------------------------------------------------------
    struct DiodeJob { int a, c; double is, n, vt; std::string model; };
    struct BjtJob { int c, b, e; double is, bf, br, vt; bool npn; std::string model; };

    std::map<std::string, std::vector<DiodeJob>> diodeBuckets;
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
    for (auto& [model, bucket] : bjtBuckets) {
        for (auto& q : bucket) result.bjts.push(q.c, q.b, q.e, q.is, q.bf, q.br, q.vt, q.npn, netlist.environment.ambient_temp_K);
    }
    
    return result;
}
