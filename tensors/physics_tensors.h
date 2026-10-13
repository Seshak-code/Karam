#pragma once
#include <vector>
#include <cstddef>
#include <map>
#include <unordered_map>
#include "../infrastructure/model_registry.h"
#include "../netlist/circuit.h"
#include "../physics/device_physics.h"
#include <array>

/*
 * physics_tensors.h - Structure-of-Arrays (SoA) Tensor Data Structures
 *
 * These structures enable batch-parallel processing of device physics:
 * - SIMD vectorization on CPU (AVX2/AVX-512)
 * - GPU compute shader dispatch (WebGPU/CUDA)
 * - Superposition decomposition (multi-source analysis)
 *
 * Each tensor stores device parameters in contiguous arrays for cache-optimal access.
 */

// ============================================================================
// DIODE TENSOR
// ============================================================================

struct DiodeTensor {
    // Node indices (global node IDs)
    std::vector<int> node_a;
    std::vector<int> node_c;
    
    // Device parameters
    std::vector<double> Is;   // Saturation current [A]
    std::vector<double> N;    // Emission coefficient
    std::vector<double> Vt;   // Thermal voltage [V]
    std::vector<double> GMIN; // Minimum conductance [S]
    
    // Thermal parameters (static model)
    std::vector<double> Rs;   // Series resistance [Ω] (parasitic)
    std::vector<double> Temp; // Operating temperature [K]
    
    // Iteration state (updated each NR pass)
    std::vector<double> v_d;  // Voltage across diode [V]
    std::vector<double> i_d;  // Current through diode [A]
    std::vector<double> g_d;  // Linearized conductance [S]
    
    // Design Rule Validation Masks (GPU-compatible uint8)
    // 0 = PASS, bit flags indicate specific violations
    std::vector<uint8_t> dr_validity;
    static constexpr uint8_t DR_OK = 0;
    static constexpr uint8_t DR_THERMAL_VIOLATION = 1 << 0;
    static constexpr uint8_t DR_REVERSE_BIAS_LIMIT = 1 << 1;
    
    size_t size() const { return node_a.size(); }
    
    void reserve(size_t n) {
        node_a.reserve(n); node_c.reserve(n);
        Is.reserve(n); N.reserve(n); Vt.reserve(n); GMIN.reserve(n);
        Rs.reserve(n); Temp.reserve(n);
        v_d.reserve(n); i_d.reserve(n); g_d.reserve(n);
        dr_validity.reserve(n);
    }
    
    void clear() {
        node_a.clear(); node_c.clear();
        Is.clear(); N.clear(); Vt.clear(); GMIN.clear();
        Rs.clear(); Temp.clear();
        v_d.clear(); i_d.clear(); g_d.clear();
        dr_validity.clear();
    }
    
    void push(int anode, int cathode, double is, double n, double vt, 
              double rs = 0.0, double temp = 300.0, double gmin = 1e-12) {
        node_a.push_back(anode);
        node_c.push_back(cathode);
        Is.push_back(is);
        N.push_back(n);
        Vt.push_back(vt);
        GMIN.push_back(gmin);
        Rs.push_back(rs);
        Temp.push_back(temp);
        v_d.push_back(0.0);
        i_d.push_back(0.0);
        g_d.push_back(gmin); 
        dr_validity.push_back(DR_OK);
    }
};

// Level-1 parameters of a MOSFET instance from its parser-resolved .model card
// (Mosfet::card). An unresolved card (the GUI's Verilog-A path) keeps the
// engine's 200u/0.7 V/0.02 NMOS and 100u/-0.7 V/0.02 PMOS by model name, no caps.
inline Mos1Params<double> mos1Params(const Mosfet& m) {
    const auto& c = m.card;
    Mos1Params<double> p;
    const bool pmos = c.resolved ? c.pmos
                                 : m.modelName.find("PMOS") != std::string::npos || m.modelName.find("pmos") != std::string::npos;
    p.polarity = pmos ? -1.0 : 1.0;
    const double M = m.extra.mult, leff = m.l - 2.0 * c.ld;
    if (!c.resolved) {
        p.vto = pmos ? -0.7 : 0.7;
        p.lambda = 0.02;
        p.beta = (pmos ? 100e-6 : 200e-6) * m.w / m.l * M;
        p.isat = 1e-14 * M;
        return p;
    }
    p.vto = c.vto; p.lambda = c.lambda; p.gamma = c.gamma; p.phi = c.phi;
    p.beta = c.kp * m.w / leff * M;
    p.isat = c.is * M;
    p.cox = c.tox > 0.0 ? 3.9 * 8.854214871e-12 / c.tox * m.w * leff * M : 0.0;
    p.cgso = c.cgso * m.w * M; p.cgdo = c.cgdo * m.w * M; p.cgbo = c.cgbo * leff * M;
    // CBD/CBS win over CJ * area (ngspice mos1temp)
    p.czbd = (c.cbd > 0.0 ? c.cbd : c.cj * m.geo.ad) * M; p.czbdsw = c.cjsw * m.geo.pd * M;
    p.czbs = (c.cbs > 0.0 ? c.cbs : c.cj * m.geo.as) * M; p.czbssw = c.cjsw * m.geo.ps * M;
    p.mj = c.mj; p.mjsw = c.mjsw; p.pb = c.pb; p.fc = c.fc;
    p.caps = p.cox > 0.0 || p.cgso > 0.0 || p.cgdo > 0.0 || p.cgbo > 0.0 ||
             p.czbd > 0.0 || p.czbdsw > 0.0 || p.czbs > 0.0 || p.czbssw > 0.0;
    return p;
}

// ============================================================================
// BJT TENSOR
// ============================================================================

struct BJTTensor {
    // Node indices
    std::vector<int> collectors;
    std::vector<int> bases;
    std::vector<int> emitters;
    
    // Device parameters
    std::vector<double> Is;      // Saturation current [A]
    std::vector<double> BetaF;   // Forward current gain
    std::vector<double> BetaR;   // Reverse current gain
    std::vector<double> Vt;      // Thermal voltage [V]
    std::vector<bool> isNPN;     // true = NPN, false = PNP
    
    // Thermal parameters
    std::vector<double> Temp;    // Operating temperature [K]
    
    // Iteration state (terminal currents)
    std::vector<double> Ic;      // Collector current [A]
    std::vector<double> Ib;      // Base current [A]
    std::vector<double> Ie;      // Emitter current [A]
    
    // Conductance matrix (3x3 per device, flattened)
    std::vector<double> g_cc, g_cb, g_ce;
    std::vector<double> g_bc, g_bb, g_be;
    std::vector<double> g_ec, g_eb, g_ee;
    
    // Design Rule Validation Masks (GPU-compatible uint8)
    std::vector<uint8_t> dr_validity;
    static constexpr uint8_t DR_OK = 0;
    static constexpr uint8_t DR_FLOATING_BASE = 1 << 0;
    static constexpr uint8_t DR_THERMAL_VIOLATION = 1 << 1;
    
    size_t size() const { return collectors.size(); }
    
    void reserve(size_t n) {
        collectors.reserve(n); bases.reserve(n); emitters.reserve(n);
        Is.reserve(n); BetaF.reserve(n); BetaR.reserve(n); Vt.reserve(n); isNPN.reserve(n);
        Temp.reserve(n);
        Ic.reserve(n); Ib.reserve(n); Ie.reserve(n);
        g_cc.reserve(n); g_cb.reserve(n); g_ce.reserve(n);
        g_bc.reserve(n); g_bb.reserve(n); g_be.reserve(n);
        g_ec.reserve(n); g_eb.reserve(n); g_ee.reserve(n);
        dr_validity.reserve(n);
    }
    
    void clear() {
        collectors.clear(); bases.clear(); emitters.clear();
        Is.clear(); BetaF.clear(); BetaR.clear(); Vt.clear(); isNPN.clear();
        Temp.clear();
        Ic.clear(); Ib.clear(); Ie.clear();
        g_cc.clear(); g_cb.clear(); g_ce.clear();
        g_bc.clear(); g_bb.clear(); g_be.clear();
        g_ec.clear(); g_eb.clear(); g_ee.clear();
        dr_validity.clear();
    }
    
    void push(int c, int b, int e, double is, double betaF, double betaR, 
              double vt, bool npn, double temp = 300.0) {
        collectors.push_back(c);
        bases.push_back(b);
        emitters.push_back(e);
        Is.push_back(is);
        BetaF.push_back(betaF);
        BetaR.push_back(betaR);
        Vt.push_back(vt);
        isNPN.push_back(npn);
        Temp.push_back(temp);
        
        // Initialize currents to zero
        Ic.push_back(0.0);
        Ib.push_back(0.0);
        Ie.push_back(0.0);
        
        // Initialize conductances to GMIN
        double gmin = 1e-12;
        g_cc.push_back(gmin); g_cb.push_back(0.0); g_ce.push_back(0.0);
        g_bc.push_back(0.0); g_bb.push_back(gmin); g_be.push_back(0.0);
        g_ec.push_back(0.0); g_eb.push_back(0.0); g_ee.push_back(gmin);
        dr_validity.push_back(DR_OK);
    }
};

// ============================================================================
// RESISTOR TENSOR (Linear - for completeness)
// ============================================================================

struct ResistorTensor {
    std::vector<int> nodes1;
    std::vector<int> nodes2;
    std::vector<double> R;  // Resistance [Ω]
    
    size_t size() const { return nodes1.size(); }
    
    void push(int n1, int n2, double r) {
        nodes1.push_back(n1);
        nodes2.push_back(n2);
        R.push_back(r);
    }
    
    void clear() {
        nodes1.clear(); nodes2.clear(); R.clear();
    }
};

// ============================================================================
// CAPACITOR TENSOR (for transient)
// ============================================================================

struct CapacitorTensor {
    std::vector<int> plates1;
    std::vector<int> plates2;
    std::vector<double> C;         // Capacitance [F]
    std::vector<double> v_prev;    // Voltage at t-1
    std::vector<double> v_prev2;   // Voltage at t-2 (Gear2)
    std::vector<double> i_prev;    // Current at t-1
    
    size_t size() const { return plates1.size(); }
    
    void push(int p1, int p2, double c) {
        plates1.push_back(p1);
        plates2.push_back(p2);
        C.push_back(c);
        v_prev.push_back(0.0);
        v_prev2.push_back(0.0);
        i_prev.push_back(0.0);
    }
    
    void clear() {
        plates1.clear(); plates2.clear(); C.clear();
        v_prev.clear(); v_prev2.clear(); i_prev.clear();
    }
};

// ============================================================================
// INDUCTOR TENSOR (for transient)
// ============================================================================

struct InductorTensor {
    std::vector<int> coil1;
    std::vector<int> coil2;
    std::vector<double> L;         // Inductance [H]
    std::vector<double> i_prev;    // Current at t-1
    std::vector<double> v_prev;    // Voltage at t-1
    
    size_t size() const { return coil1.size(); }
    
    void push(int c1, int c2, double l) {
        coil1.push_back(c1);
        coil2.push_back(c2);
        L.push_back(l);
        i_prev.push_back(0.0);
        v_prev.push_back(0.0);
    }
    
    void clear() {
        coil1.clear(); coil2.clear(); L.clear();
        i_prev.clear(); v_prev.clear();
    }
};

// ============================================================================
// SOURCE TENSORS
// ============================================================================

struct VoltageSourceTensor {
    std::vector<int> nodesPos;
    std::vector<int> nodesNeg;
    std::vector<double> voltages; // Nominal DC value
    
    size_t size() const { return nodesPos.size(); }
    
    void push(int np, int nn, double v) {
        nodesPos.push_back(np);
        nodesNeg.push_back(nn);
        voltages.push_back(v);
    }
    
    void clear() {
        nodesPos.clear(); nodesNeg.clear(); voltages.clear();
    }
};

struct CurrentSourceTensor {
    std::vector<int> nodesPos;
    std::vector<int> nodesNeg;
    std::vector<double> currents;
    
    size_t size() const { return nodesPos.size(); }
    
    void push(int np, int nn, double i) {
        nodesPos.push_back(np);
        nodesNeg.push_back(nn);
        currents.push_back(i);
    }
    
    void clear() {
        nodesPos.clear(); nodesNeg.clear(); currents.clear();
    }
};

// ============================================================================
// UNIFIED TENSOR BLOCK
// ============================================================================

/**
 * TensorizedBlock - Unified container for all device tensors.
 * This replaces the AoS `TensorBlock` for solver-side operations.
 */
struct TensorizedBlock {
    ResistorTensor resistors;
    CapacitorTensor capacitors;
    InductorTensor inductors;
    VoltageSourceTensor voltageSources;
    CurrentSourceTensor currentSources;
    DiodeTensor diodes;
    BJTTensor bjts;
    
    // Dynamic Models (Type-Erased)
    // Key: DeviceType (e.g. "Diode", "MosfetLevel1") -> Value: tensor + instance count
    struct DynamicTensorEntry {
        void* tensor = nullptr;
        size_t count = 0;
    };
    std::unordered_map<std::string, DynamicTensorEntry> dynamicTensors;

    void clear() {
        resistors.clear();
        capacitors.clear();
        inductors.clear();
        voltageSources.clear();
        currentSources.clear();
        diodes.clear();
        bjts.clear();
        
        // Clear dynamic tensors
        auto& registry = ModelRegistry::instance();
        for (auto& [type, entry] : dynamicTensors) {
            if (!entry.tensor) continue;
            auto models = registry.listModels(type);
            if (!models.empty()) {
                const ModelInfo* info = registry.getModel(type, models[0]);
                if (info && info->destroyTensor) {
                    info->destroyTensor(entry.tensor);
                }
            }
        }
        dynamicTensors.clear();
    }
};

// ============================================================================
// CONVERSION FUNCTIONS (AoS -> SoA)
// ============================================================================

// Forward declaration: TensorBlock is defined in circuit.h
struct TensorBlock;

/**
 * Convert a TensorBlock (AoS) to TensorizedBlock (SoA).
 * This enables efficient batch processing on the solver side.
 */
TensorizedBlock tensorizeBlock(const struct TensorBlock& block);

/**
 * Convert a full TensorNetlist (Hierarchical) to a single TensorizedBlock (Flat SoA).
 * All hierarchy is flattened into global node indices.
 */
TensorizedBlock tensorizeNetlist(const class TensorNetlist& netlist);

// ============================================================================
// BATCH PHYSICS KERNELS (Vectorizable)
// ============================================================================

#include <cmath>

/**
 * Batch Diode Physics - Evaluates all diodes in a tensor simultaneously.
 * Updates: i_d, g_d arrays in place.
 */
void batchDiodePhysics(DiodeTensor& tensor, const std::vector<double>& voltages);

/**
 * Batch BJT Physics - Evaluates all BJTs in a tensor simultaneously.
 * Updates: Ic, Ib, Ie, and conductance matrix arrays in place.
 */
void batchBJTPhysics(BJTTensor& tensor, const std::vector<double>& voltages);

