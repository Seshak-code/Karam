#pragma once
#include <vector>
#include <cstddef>
#include <map>
#include <unordered_map>
#include <string>
#include "../model_registry.h"

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

// ============================================================================
// MOSFET TENSOR  
// ============================================================================

struct MosfetTensor {
    // Node indices
    std::vector<int> drains;
    std::vector<int> gates;
    std::vector<int> sources;
    std::vector<int> bodies;
    
    // Device parameters
    std::vector<double> W;       // Channel width [m]
    std::vector<double> L;       // Channel length [m]
    std::vector<double> Kp;      // Transconductance parameter [A/V²]
    std::vector<double> Vth;     // Threshold voltage [V]
    std::vector<double> lambda;  // Channel length modulation [1/V]
    std::vector<bool> isPMOS; 
    
    // Thermal parameters
    std::vector<double> Temp;    // Operating temperature [K]
    
    // Iteration state
    std::vector<double> vgs;     // Gate-source voltage [V]
    std::vector<double> vds;     // Drain-source voltage [V]
    std::vector<double> ids;     // Drain current [A]
    std::vector<double> gm;      // Transconductance [S] (dId/dVgs)
    std::vector<double> gmb;     // Body transconductance [S] (dId/dVbs)
    std::vector<double> gds;     // Output conductance [S] (dId/dVds)
    
    // Design Rule Validation Masks (GPU-compatible uint8)
    std::vector<uint8_t> dr_validity;
    static constexpr uint8_t DR_OK = 0;
    static constexpr uint8_t DR_FLOATING_GATE = 1 << 0;
    static constexpr uint8_t DR_OPEN_DRAIN = 1 << 1;
    static constexpr uint8_t DR_THERMAL_VIOLATION = 1 << 2;
    
    size_t size() const { return drains.size(); }
    
    void reserve(size_t n) {
        drains.reserve(n); gates.reserve(n); sources.reserve(n); bodies.reserve(n);
        W.reserve(n); L.reserve(n); Kp.reserve(n); Vth.reserve(n); lambda.reserve(n);
        isPMOS.reserve(n);
        Temp.reserve(n);
        vgs.reserve(n); vds.reserve(n); ids.reserve(n); gm.reserve(n); gmb.reserve(n); gds.reserve(n);
        dr_validity.reserve(n);
    }
    
    void clear() {
        drains.clear(); gates.clear(); sources.clear(); bodies.clear();
        W.clear(); L.clear(); Kp.clear(); Vth.clear(); lambda.clear();
        isPMOS.clear();
        Temp.clear();
        vgs.clear(); vds.clear(); ids.clear(); gm.clear(); gmb.clear(); gds.clear();
        dr_validity.clear();
    }
    
    void push(int d, int g, int s, int b, double w, double l, 
              double kp = 200e-6, double vth = 0.7, double lam = 0.02, 
              bool pmos = false, double temp = 300.0) {
        drains.push_back(d);
        gates.push_back(g);
        sources.push_back(s);
        bodies.push_back(b);
        W.push_back(w);
        L.push_back(l);
        Kp.push_back(kp);
        Vth.push_back(vth);
        lambda.push_back(lam);
        isPMOS.push_back(pmos);
        Temp.push_back(temp);
        vgs.push_back(0.0);
        vds.push_back(0.0);
        ids.push_back(0.0);
        gm.push_back(0.0);
        gmb.push_back(0.0);
        gds.push_back(1e-12); // MOSFET GMIN
        dr_validity.push_back(DR_OK);
    }
};

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
    std::vector<double> i_prev;    // Current at t-1
    
    size_t size() const { return plates1.size(); }
    
    void push(int p1, int p2, double c) {
        plates1.push_back(p1);
        plates2.push_back(p2);
        C.push_back(c);
        v_prev.push_back(0.0);
        i_prev.push_back(0.0);
    }
    
    void clear() {
        plates1.clear(); plates2.clear(); C.clear();
        v_prev.clear(); i_prev.clear();
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
    MosfetTensor mosfets;
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
        mosfets.clear();
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

// Forward declaration
struct ModelCard;

/**
 * Convert a TensorBlock (AoS) to TensorizedBlock (SoA).
 * This enables efficient batch processing on the solver side.
 * @param modelCards Optional pointer to model card library for per-device parameter lookup.
 */
TensorizedBlock tensorizeBlock(const struct TensorBlock& block,
                                const std::map<std::string, ModelCard>* modelCards = nullptr);

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
 * Batch MOSFET Physics - Evaluates all MOSFETs in a tensor simultaneously.
 * Updates: ids, gm, gds arrays in place.
 */
void batchMosfetPhysics(MosfetTensor& tensor, const std::vector<double>& voltages);

/**
 * Batch BJT Physics - Evaluates all BJTs in a tensor simultaneously.
 * Updates: Ic, Ib, Ie, and conductance matrix arrays in place.
 */
void batchBJTPhysics(BJTTensor& tensor, const std::vector<double>& voltages);

