#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <cassert>
#include <iostream>
#include "graph_canonical.h"

/*
 * circuit.h
 * Circuit Netlist - Component definitions and hierarchical storage.
 *
 * Variable Naming Convention (EE Standard):
 *   - Resistor: nodeTerminal1/2
 *   - Capacitor: nodePlate1/2
 *   - Inductor: nodeCoil1/2
 *   - Diode: nodeAnode, nodeCathode
 *   - Mosfet: drain, gate, source, body
 *   - BJT: collector, base, emitter
 *   - Controlled Sources: nodeOutputPos/Neg, nodeControlPos/Neg
 */


// ============================================================================
// SPICE MODEL CARD
// ============================================================================

// Stores device model parameters parsed from .MODEL directives.
struct ModelCard {
    std::string name;        // e.g. "nfet_01v8", "DMOD"
    std::string type;        // "NMOS", "PMOS", "D", "NPN", "PNP", "NJFET", "PJFET"
    int level = 1;           // Model level (1 = Shichman-Hodges, 4 = BSIM4, etc.)
    std::map<std::string, double> params;  // Key-value: "KP"->200e-6, "VTO"->0.7, etc.

    double get(const std::string& key, double defaultVal) const {
        auto it = params.find(key);
        return (it != params.end()) ? it->second : defaultVal;
    }
};

// ============================================================================
// PASSIVE COMPONENTS
// ============================================================================

/*
 * Resistor (R)
 * Ohm's Law: V = I * R
 * MNA Stamp (Conductance Form):
 *   G[n1][n1] += 1/R, G[n2][n2] += 1/R
 *   G[n1][n2] -= 1/R, G[n2][n1] -= 1/R
 */
struct Resistor {
    std::string name;             // e.g. "R1"
    int nodeTerminal1;            // First terminal (node index)
    int nodeTerminal2;            // Second terminal (node index)
    double resistance_ohms;       // Resistance in Ohms [Ω]
    
    // Generic PDK Support
    std::string model_name;       // e.g., "poly_res"
    double width_m = 0.0;
    double length_m = 0.0;
    double tc1 = 0.0;
    double tc2 = 0.0;

    double tolerance_percent = 0.0; // Monte Carlo Tolerance (e.g. 1.0 = 1%)
    std::string expression;         // Symbolic override (e.g. "Rload", "2*Vbias/Iref")
};

/*
 * Capacitor (C)
 * Physics: i_C = C * dv_C/dt
 * Trapezoidal Integration Model:
 *   G_eq = 2*C / dt
 *   I_eq = G_eq * v_prev + i_prev
 * The capacitor is replaced by a conductance in parallel with a history current source.
 */
struct Capacitor {
    std::string name;             // e.g. "C1"
    int nodePlate1;               // Positive plate (node index)
    int nodePlate2;               // Negative plate (node index)
    double capacitance_farads;    // Capacitance in Farads [F]
    
    // Generic PDK Support
    std::string model_name;       // e.g., "mim_cap"
    double width_m = 0.0;
    double length_m = 0.0;
    double m = 1.0;               // Multiplier
    
    double tolerance_percent = 0.0;
    std::string expression;         // Symbolic override for capacitance
};

/*
 * Inductor (L)
 * Physics: v_L = L * di_L/dt
 * Backward Euler Integration Model:
 *   R_eq = L / dt
 *   V_eq = R_eq * i_prev
 * Inductor adds an auxiliary current variable to the MNA system.
 */
struct Inductor {
    std::string name;             // e.g. "L1"
    int nodeCoil1;                // First coil terminal (node index)
    int nodeCoil2;                // Second coil terminal (node index)
    double inductance_henries;    // Inductance in Henries [H]
    
    double tolerance_percent = 0.0;

    // Generic PDK Support
    std::string model_name;       // e.g., "spiral_ind"
    double width_m = 0.0;         // Trace width
    double length_m = 0.0;        // Total path length
    double area_m2 = 0.0;         // Spiral area
    int turns = 0;                // Number of turns

    std::string expression;       // Symbolic override for inductance
};

/*
 * Mutual Inductor (K) - Transformer Coupling
 * Physics:
 *   v1 = L1 * di1/dt + M * di2/dt
 *   v2 = M * di1/dt + L2 * di2/dt
 * Where M = couplingCoefficient_k * sqrt(L1 * L2)
 */
struct MutualInductor {
    int inductor1_index;          // Index of first inductor in inductors vector
    int inductor2_index;          // Index of second inductor
    double couplingCoefficient_k; // Coupling factor (0.0 = none, 1.0 = ideal)
};

// ============================================================================
// INDEPENDENT SOURCES
// ============================================================================

/*
 * Independent Voltage Source (V)
 * Ideal: Forces v(+) - v(-) = V_dc
 * Adds an auxiliary branch current to MNA.
 */
struct VoltageSource {
    int nodePositive;
    int nodeNegative;
    double voltage_V;        // DC/Offset Voltage
    std::string type = "";   // "PULSE", "SINE", "DC", etc.

    // Waveform Parameters (SPICE-like)
    double ac_mag = 0.0;     // AC Magnitude (Small Signal)
    double ac_phase = 0.0;   // AC Phase
    
    // PULSE(v1 v2 td tr tf pw per)
    double pulse_v1 = 0.0;   // Initial value
    double pulse_v2 = 0.0;   // Pulsed value
    double pulse_td = 0.0;   // Delay time
    double pulse_tr = 0.0;   // Rise time
    double pulse_tf = 0.0;   // Fall time
    double pulse_pw = 0.0;   // Pulse width
    double pulse_per = 0.0;  // Period
    
    // SINE(vo va freq td theta)
    double sine_vo = 0.0;    // Offset (same as voltage_V for DC)
    double sine_va = 0.0;    // Amplitude
    double sine_freq = 0.0;  // Frequency [Hz]
    double sine_td = 0.0;    // Delay
    double sine_theta = 0.0; // Damping factor
    double sine_phase = 0.0; // Phase [deg] (Extension)

    std::string expression;  // Symbolic override for DC/main value
};

/*
 * Independent Current Source (I)
 * Ideal: Forces i = I_dc (from + to -)
 * Stamps directly into RHS vector.
 */
struct CurrentSource {
    int nodePositive;
    int nodeNegative;
    double current_A;        // DC Current
    std::string type = "";   // "PULSE", "SINE", "DC"

    // Waveform Parameters (Mirror of VoltageSource)
    double ac_mag = 0.0;
    double ac_phase = 0.0;

    double pulse_v1 = 0.0;
    double pulse_v2 = 0.0;
    double pulse_td = 0.0;
    double pulse_tr = 0.0;
    double pulse_tf = 0.0;
    double pulse_pw = 0.0;
    double pulse_per = 0.0;

    double sine_vo = 0.0;
    double sine_va = 0.0;
    double sine_freq = 0.0;
    double sine_td = 0.0;
    double sine_theta = 0.0;
    double sine_phase = 0.0;

    std::string expression;  // Symbolic override for DC/main value
};

// ============================================================================
// DEPENDENT SOURCES (Bench + Block internal modes)
// ============================================================================

/*
 * VCVS - Voltage-Controlled Voltage Source
 * Vout = gain * Vcontrol
 * MNA: Adds branch equation with controlled coefficient.
 */
struct VCVS {
    int outPos, outNeg;
    int ctrlPos, ctrlNeg;
    double gain;
    std::string role;
};

struct VCCS {
    int outPos, outNeg;
    int ctrlPos, ctrlNeg;
    double gm;
    std::string role;
};

struct CCVS {
    int outPos, outNeg;
    int ctrlPos, ctrlNeg;
    double rm;
    std::string role;
};

struct CCCS {
    int outPos, outNeg;
    int ctrlPos, ctrlNeg;
    double alpha;
    std::string role;
};

// ============================================================================
// SYSTEM SIMULATION SOURCES (Phase 2.9)
// ============================================================================

/*
 * Power Rail (Vdd/Vcc)
 * Models a realistic power supply network (PDN).
 * Model: V_ideal -> [ESL] -> [ESR] -> Node_Chip -> [C_decap] -> GND
 */
struct PowerRail {
    std::string name;              // "Vdd", "Vcc", "Vio"
    int nodeRail;                  // The chip-side node index
    double nominal_V;              // 1.8V, 3.3V, 5V, 12V
    double ripple_Vpp;             // Peak-to-peak ripple voltage
    double frequency_Hz;           // Ripple frequency (e.g., 1MHz switching)
    double ESR_ohms;               // Equivalent Series Resistance (package + regulator)
    double ESL_henries;            // Equivalent Series Inductance (bond wires)
    double capacitance_F;          // Decoupling capacitance (local)
    
    // Monte Carlo parameters
    double tolerance_percent;      // ±5% for typical regulators
    double temp_coefficient;       // Voltage drift with temperature [V/K]
    
};

/*
 * PDK Power Domain (Phase 1.3 Addition)
 * Represents a named voltage domain for PDK-aware design.
 * Multiple instances of the same domain share a common node.
 * Model: Ideal voltage source with optional internal resistance.
 */
struct PDKPowerDomain {
    std::string name;              // Instance name, e.g., "VDD_CORE1"
    std::string domain;            // Domain grouping, e.g., "CORE", "IO", "ANALOG"
    enum Type { VDD, VSS, VDDIO, VBB } type;
    
    int globalNode;                // Assigned node index in circuit matrix
    double nominalVoltage;         // 1.8V, 0V, 3.3V, -0.3V (for VBB)
    double maxCurrent;             // Maximum current capacity [A]
    double internalResistance;     // IR drop model: V = Vnom - I * Rint
    
    // Verilog-A Export Helper
    std::string getVerilogAModule() const {
        if (type == VSS) {
            return "// VSS connects to global ground (node 0)";
        }
        std::string code =
            "module " + name + "_supply(vout);\n"
            "    output vout;\n"
            "    electrical vout;\n"
            "    parameter real vnom = " + std::to_string(nominalVoltage) + ";\n"
            "    parameter real rint = " + std::to_string(internalResistance) + ";\n"
            "    analog begin\n"
            "        V(vout) <+ vnom - I(vout) * rint;\n"
            "    end\n"
            "endmodule\n";
        return code;
    }
};

/*
 * Signal Source (Waveform Generator)
 * Models complex signal inputs (Clock, Data, Noise).
 */
struct SignalSource {
    std::string name;              // "CLK", "DATA_IN"
    int nodePositive;
    int nodeNegative;
    
    enum WaveformType { DC, PULSE, SINE, PWL, NOISE } type;
    
    double amplitude_V;
    double frequency_Hz;
    double phase_deg;
    double offset_V;
    double duty_cycle;             // 0.0 to 1.0 (Pulse only)
    double rise_time_s;            // (Pulse only)
    double fall_time_s;            // (Pulse only)
    
    // For future PWL
    std::vector<std::pair<double, double>> points;
};

// ============================================================================
// DIGITAL VERIFICATION COMPONENTS (Phase 13)
// ============================================================================

enum class LogicState {
    LOW = 0,
    HIGH = 1,
    UNKNOWN = 2, // X
    IMPEDANCE = 3 // Z
};

struct BusProbe {
    std::string name;             // e.g., "DATA_BUS"
    int bit_width;                // 8, 16, 32
    std::vector<int> nodeIndices; // [LSB ... MSB]
    
    // Logic Level Config
    double v_th_low = 0.8;        // Below 0.8V = 0
    double v_th_high = 2.0;       // Above 2.0V = 1
    
    struct DigitalSample {
        double time;
        uint64_t value;
    };
    
};

/*
 * Simulation Environment
 * Global context for the simulation run.
 */
struct SimulationEnvironment {
    double ambient_temp_K = 300.15;   // 27°C
    double temp_ramp_K_per_s = 0.0;   // For thermal cycling
    
    // Process Corners (Monte Carlo)
    enum Corner { TT, FF, SS, SF, FS } process = TT;
    double vdd_variation_sigma = 0.0; // σ for MC
    double global_voltage_scale = 1.0; // PVT Voltage Scale
    int monte_carlo_seed = 0;          // Deterministic seed for MC runs
    
    // Noise Profile
    bool thermal_noise_enabled = false;
    bool flicker_noise_enabled = false;
};




// ============================================================================
// NON-LINEAR SEMICONDUCTOR DEVICES
// ============================================================================

/*
 * Diode (D) - PN Junction
 * Shockley Equation:
 *   I_D = I_S * (exp(V_D / (N * V_T)) - 1)
 * Where:
 *   I_S = Saturation current (typ. 1e-14 A)
 *   N   = Emission coefficient (1.0 for ideal, 1.5-2 for real)
 *   V_T = Thermal voltage (kT/q = 25.85 mV @ 300K)
 *
 * Newton-Raphson Linearization:
 *   g_eq = dI/dV = (I_S / (N*V_T)) * exp(V_k / (N*V_T))
 *   I_eq = I(V_k) - g_eq * V_k
 */
struct Diode {
    std::string instanceName;         // e.g. "D1"
    int anode;                        // P-type side (+)
    int cathode;                      // N-type side (-)
    double saturationCurrent_I_S_A;       // I_S [A]
    double emissionCoefficient_N;         // N (dimensionless)
    double thermalVoltage_V_T_V;          // V_T [V]
    
    // Thermal (Optional, default 300.15K)
    double temperature_K = 300.15;

    // Generic PDK Support
    std::string modelName = "diode_generic"; 
    double area_m2 = 1e-12;       // Junction Area
    double perimeter_m = 4e-6;    // Junction Perimeter

};

/*
 * Schottky Diode (D) - Metal-Semiconductor Junction
 * Lower forward voltage drop (~0.3V vs ~0.7V PN), faster switching.
 * SPICE Model: Same Shockley equation but different defaults:
 *   I_S = 100nA - 1µA (higher than PN)
 *   N = 1.05 (closer to ideal)
 *   Rs = Series resistance (significant for high current)
 *   Cj0 = Zero-bias junction capacitance
 *   Vj = Junction potential (~0.4V for Schottky)
 *   M = Grading coefficient (~0.5)
 */
struct SchottkyDiode {
    std::string instanceName;             // e.g., "D1"
    int anode = -1;                       // Metal side (+)
    int cathode = -1;                     // Semiconductor side (-)
    std::string modelName = "BAT54";      // SPICE model reference
    
    // DC Parameters
    double saturationCurrent_I_S_A = 200e-9;   // Higher than PN (~200nA typ)
    double emissionCoefficient_N = 1.05;       // Closer to ideal
    double seriesResistance_Rs_ohms = 1.0;     // Package/bond wire resistance
    double thermalVoltage_V_T_V = 0.02585;     // kT/q @ 300K
    
    // Junction Capacitance (AC/Transient)
    double junctionCapacitance_Cj0_F = 2e-12;  // Zero-bias capacitance
    double junctionPotential_Vj_V = 0.4;       // Lower than PN
    double gradingCoefficient_M = 0.5;         // ~0.5 for abrupt junction
    
    double temperature_K = 300.15;
};



/*
 * Zener Diode (D) - Breakdown Diode
 * Operates in reverse breakdown region to regulate voltage.
 * Model:
 *   Forward: Standard PN junction
 *   Reverse: Breakdown at BV with impedance Rz
 */
struct ZenerDiode {
    std::string instanceName;             // e.g., "D1"
    int anode = -1;                       // Anode (+)
    int cathode = -1;                     // Cathode (-)
    std::string modelName = "1N4733";     // default 5.1V
    
    // DC Parameters
    double breakdownVoltage_V = 5.1;      // BV
    double currentAtBreakdown_A = 0.049;  // Izt
    double seriesResistance_Rs_ohms = 7.0; // Rz
    double emissionCoefficient_N = 1.0;
    double saturationCurrent_I_S_A = 1e-14;
    double thermalVoltage_V_T_V = 0.02585;
    
    double temperature_K = 300.15;
};

struct Mosfet {
    std::string instanceName;             // e.g., "M1"
    int drain;                        // D terminal
    int gate;                         // G terminal
    int source;                       // S terminal
    int body;                         // B (Bulk/Substrate) terminal
    
    // Standard SPICE / PDK Parameters
    std::string modelName;                // .model reference (e.g., "nfet_01v8")
    
    // Dimensions (Standard SPICE)
    double w = 1e-6;                      // Channel Width [m]
    double l = 1e-6;                      // Channel Length [m]
    


    // Layout Parasitics (Geometry)
    struct Geometry {
        double ad = 0.0; // Area Drain
        double as = 0.0; // Area Source
        double pd = 0.0; // Perimeter Drain
        double ps = 0.0; // Perimeter Source
        double nrd = 0.0; // Number of squares Drain
        double nrs = 0.0; // Number of squares Source
    } geo;

    // Advanced Params
    struct Params {
        int nf = 1;      // Number of fingers
        int mult = 1;    // Multiplier
    } extra;

    // Noise Parameters
    double KF = 0.0;     // Flicker noise coefficient
    double AF = 1.0;     // Flicker noise exponent
    double C_ox = 0.0;   // Gate oxide capacitance [F/m^2]

    double temperature_K = 300.15;
};



struct BJT {
    std::string instanceName;
    int nodeCollector = -1;                    // C terminal
    int base = -1;                             // B terminal
    int emitter = -1;                          // E terminal
    std::string modelName = "default";
    bool isNPN = true;                           // true=NPN, false=PNP
    
    // Model Parameters (Per-Instance for Phase 1, ideally per-model lookup)
    double betaF = 100.0;                         // Forward Beta
    double betaR = 1.0;                         // Reverse Beta
    double saturationCurrent_I_S_A = 1e-14;       // Is [A]
    double thermalVoltage_V_T_V = 0.02585;          // Vt [V]

    // Noise Parameters
    double r_bb = 0.0;   // Base spreading resistance [Ohm]
    double KF = 0.0;     // Flicker noise coefficient
    double AF = 1.0;     // Flicker noise exponent

    double temperature_K = 300.15;
};

/*
 * JFET (J) - Junction Field Effect Transistor
 * Shichman-Hodges Model:
 *   Cutoff:     Vgs <= Vto (for N-ch) or Vgs >= -Vto (for P-ch)
 *   Linear:     Id = Beta * (2*(Vgs-Vto)*Vds - Vds^2) * (1 + lambda*Vds)
 *   Saturation: Id = Beta * (Vgs - Vto)^2 * (1 + lambda*Vds)
 * Where:
 *   Beta = IDSS / Vto^2 (typ. 1e-4 A/V^2)
 *   Vto  = Pinch-off voltage (typ. -2V for N-ch, +2V for P-ch)
 *   lambda = Channel-length modulation (typ. 0.01)
 */
struct JFET {
    std::string instanceName;             // e.g., "J1"
    int drain = -1;                       // D terminal
    int gate = -1;                        // G terminal
    int source = -1;                      // S terminal
    std::string modelName = "default";
    bool isNChannel = true;               // true=N-ch, false=P-ch
    
    // Model Parameters
    double beta = 1e-4;                   // Transconductance parameter [A/V^2]
    double Vto = -2.0;                    // Pinch-off voltage [V] (negative for N-ch)
    double lambda = 0.01;                 // Channel-length modulation [1/V]
    double thermalVoltage_V_T_V = 0.02585; // For gate-source diode (optional)
    
    double temperature_K = 300.15;
};

// ============================================================================
// INTERCONNECTS & TRANSMISSION LINES
// ============================================================================

/*
 * Lossless Transmission Line (T)
 * Telegrapher's Equations yield a time-delayed reflection model:
 *   v_1(t) + Z_0 * i_1(t) = v_2(t - T_d) + Z_0 * i_2(t - T_d)
 * Modeled as history-dependent current sources at each port.
 */
struct TransmissionLine {
    int nodePort1Pos, nodePort1Neg;       // Port 1 terminals
    int nodePort2Pos, nodePort2Neg;       // Port 2 terminals
    double characteristicImpedance_Z0_ohms; // Z_0 [Ω]
    double propagationDelay_Td_s;         // T_d [seconds]
};

/*
 * Probe (Measurement Tool)
 * Does not affect physics (infinite impedance).
 * Used to request V(pos) - V(neg) for plotting/GUI.
 */
struct Probe {
    std::string name;
    int nodePositive;
    int nodeNegative;
};

// ============================================================================
// TENSOR HIERARCHY (Hierarchical Netlist Structure)
// ============================================================================

/*
 * TensorBlock - A reusable sub-circuit definition (e.g., Inverter, SRAM cell).
 * Components use LOCAL node indices (0 = local ground, 1..N = internal nodes).
 */

struct TensorBlock {
    std::string name;
    int numInternalNodes = 0;
    std::vector<std::string> pinNames;      // Metadata for Verilog-A port naming
    std::vector<int> portNodeIndices;         // Pin index -> internal node index (for subcircuit blocks)

    // Hierarchical interface pin metadata (populated during subcircuit resolution)
    struct InterfacePin {
        std::string name;        // Pin/PowerPort component name
        std::string netUUIDStr;  // Serialized NetUUID (from NetUUID::toString())
        int nodeIndex = 0;       // Solver node index inside this block
        bool isPower = false;    // true for PowerPort-type boundary pins
        std::string direction;   // "Input", "Output", "BiDir", "Power"
    };
    std::vector<InterfacePin> interfacePins;
    
    // Passive Components
    std::vector<Resistor> resistors;
    std::vector<Capacitor> capacitors;
    std::vector<Inductor> inductors;
    std::vector<MutualInductor> mutualInductors;
    
    // Independent Sources
    std::vector<VoltageSource> voltageSources;
    std::vector<CurrentSource> currentSources;

    // System Simulation Sources (Phase 2.9)
    std::vector<PowerRail> powerRails;
    std::vector<SignalSource> signalSources;
    
    // PDK Power Domains (Phase 1.3)
    std::vector<PDKPowerDomain> powerDomains;
    
    // Controlled Sources
    std::vector<VCVS> voltageControlledVoltageSources;
    std::vector<VCCS> voltageControlledCurrentSources;
    std::vector<CCVS> currentControlledVoltageSources;
    std::vector<CCCS> currentControlledCurrentSources;
    
    // Semiconductors
    std::vector<Diode> diodes;
    std::vector<SchottkyDiode> schottkyDiodes;
    std::vector<Mosfet> mosfets;
    std::vector<BJT> bjts;
    std::vector<JFET> jfets;
    std::vector<ZenerDiode> zenerDiodes;
    
    // Interconnects
    std::vector<TransmissionLine> transmissionLines;
    
    // Generic Components (Dynamic Physics Models)
    struct GenericComponent {
        std::string instanceName;
        std::string modelName; // Reference to registered model
        std::vector<int> nodes;
        std::vector<double> params;
    };
    std::vector<GenericComponent> genericComponents;
    
    // Aux
    std::vector<Probe> probes;
    std::vector<BusProbe> busProbes;

    // --- Phase 1.5: Parasitic Extraction Support ---
    struct ParasiticResistor {
        std::string name;
        int node1, node2;
        double R;
    };
    struct ParasiticCapacitor {
        std::string name;
        int node1, node2;
        double C;
    };
    struct WireSegment {
        std::string layer;      // e.g. "M1", "M2"
        int node1, node2;
        double width_um;
        double length_um;
        double r_per_um = 0.0;
        double c_per_um = 0.0;
    };

    std::vector<ParasiticResistor> parasiticResistors;
    std::vector<ParasiticCapacitor> parasiticCapacitors;
    std::vector<WireSegment> wireSegments;

    bool isFrozen = false;       // Phase 1: Structural Immutability
    size_t topologyHash = 0;     // Phase 4: Symbolic Reuse Key

    // --- Helper Methods ---
    void freeze() {
        if (isFrozen) return;
        isFrozen = true;
        computeTopologyHash();
    }

    void computeTopologyHash() {
        // Phase 1.4: Canonical Graph Hashing
        // Uses orientation-invariant edges and BFS-based node relabeling.
        std::vector<GraphCanonical::CanonicalEdge> edges;
        
        // Type tags: 0=Resistor, 1=Capacitor, 2=Inductor, 3=Diode, 4=Mosfet, 5=BJT
        for (const auto& r : resistors) 
            edges.emplace_back(r.nodeTerminal1, r.nodeTerminal2, 0);
        for (const auto& c : capacitors) 
            edges.emplace_back(c.nodePlate1, c.nodePlate2, 1);
        for (const auto& l : inductors) 
            edges.emplace_back(l.nodeCoil1, l.nodeCoil2, 2);
        for (const auto& d : diodes) 
            edges.emplace_back(d.anode, d.cathode, 3);
        for (const auto& m : mosfets) {
            // MOSFET is a 4-terminal device; we represent as (D-S) and (G-S) edges
            edges.emplace_back(m.drain, m.source, 4);
            edges.emplace_back(m.gate, m.source, 4);
        }
        for (const auto& b : bjts) {
            // BJT: (C-E) and (B-E) edges
            edges.emplace_back(b.nodeCollector, b.emitter, 5);
            edges.emplace_back(b.base, b.emitter, 5);
        }
        
        // Ground is assumed to be node 0 for blocks
        topologyHash = GraphCanonical::computeTopologyHash(edges, 0);
    }

    void updateNodeCount(int node) {
        if (isFrozen) return; // Prevent node count updates on frozen blocks
        if (node > numInternalNodes) numInternalNodes = node;
    }

    void addResistor(const std::string& name, int n1, int n2, double R) {
        assert(n1 >= 0 && n2 >= 0 && "Node indices must be not negative");
        assert(R > 0 && "Resistance must be positive");
        if (isFrozen) return;
        resistors.push_back({name, n1, n2, R});
        updateNodeCount(n1); updateNodeCount(n2);
    }
    // Backward compatibility overload (generates name if missing, though typically netlist parser provides it)
    void addResistor(int n1, int n2, double R) {
        static int r_count = 0;
        addResistor("R_AUTO_" + std::to_string(++r_count), n1, n2, R);
    }
    void addCapacitor(const std::string& name, int n1, int n2, double C) {
        assert(n1 >= 0 && n2 >= 0 && "Node indices must be not negative");
        assert(C > 0 && "Capacitance must be positive");
        if (isFrozen) return;
        Capacitor cap = {name, n1, n2, C};
        capacitors.push_back(cap);
        updateNodeCount(n1); updateNodeCount(n2);
    }
    void addCapacitor(int n1, int n2, double C) {
        static int c_count = 0;
        addCapacitor("C_AUTO_" + std::to_string(++c_count), n1, n2, C);
    }
    void addInductor(const std::string& name, int n1, int n2, double L) {
        assert(n1 >= 0 && n2 >= 0 && "Node indices must be not negative");
        assert(L > 0 && "Inductance must be positive");
        if (isFrozen) return;
        Inductor ind = {name, n1, n2, L};
        ind.model_name = "ind_generic";

        inductors.push_back(ind);
        updateNodeCount(n1); updateNodeCount(n2);
    }
     void addInductor(int n1, int n2, double L) {
        static int l_count = 0;
        addInductor("L_AUTO_" + std::to_string(++l_count), n1, n2, L);
    }
    void addVoltageSource(int nPos, int nNeg, double V, std::string type = "") {
        assert(nPos >= 0 && nNeg >= 0 && "Node indices must be not negative");
        if (isFrozen) return;
        voltageSources.push_back({nPos, nNeg, V, type});
        updateNodeCount(nPos); updateNodeCount(nNeg);
    }
    void addCurrentSource(int nPos, int nNeg, double I) {
        if (isFrozen) return;
        currentSources.push_back({nPos, nNeg, I});
        updateNodeCount(nPos); updateNodeCount(nNeg);
    }
    
    // System Simulation Helpers
    void addPowerRail(const std::string& name, int node, double V, double ripple = 0.0) {
        if (isFrozen) return;
        PowerRail rail;
        rail.name = name;
        rail.nodeRail = node;
        rail.nominal_V = V;
        rail.ripple_Vpp = ripple;
        // Defaults for simple model
        rail.frequency_Hz = 1e6;
        rail.ESR_ohms = 0.01;
        rail.ESL_henries = 1e-9;
        rail.capacitance_F = 100e-9;
        rail.tolerance_percent = 5.0;
        rail.temp_coefficient = 0.0;
        
        powerRails.push_back(rail);
        updateNodeCount(node);
    }
    
    void addSignalSource(const std::string& name, int nPos, int nNeg, 
                         SignalSource::WaveformType type, double amp, double freq) {
        if (isFrozen) return;
        SignalSource src;
        src.name = name;
        src.nodePositive = nPos;
        src.nodeNegative = nNeg;
        src.type = type;
        src.amplitude_V = amp;
        src.frequency_Hz = freq;
        // Defaults
        src.phase_deg = 0.0;
        src.offset_V = 0.0;
        src.duty_cycle = 0.5;
        
        signalSources.push_back(src);
        updateNodeCount(nPos); updateNodeCount(nNeg);
    }
    void addDiode(const std::string& name, int anode, int cathode, double Is, double N = 1.0, double Vt = 0.02585) {
        if (isFrozen) return;
        Diode d;
        d.instanceName = name;
        d.anode = anode;
        d.cathode = cathode;
        d.saturationCurrent_I_S_A = Is;
        d.emissionCoefficient_N = N;
        d.thermalVoltage_V_T_V = Vt;
        d.modelName = "diode_generic"; // Default PDK model
        diodes.push_back(d);
        updateNodeCount(anode); updateNodeCount(cathode);
    }
    void addDiode(int anode, int cathode, double Is, double N = 1.0, double Vt = 0.02585) {
        static int d_count = 0;
        addDiode("D_AUTO_" + std::to_string(++d_count), anode, cathode, Is, N, Vt);
    }
    void addMosfet(const std::string& name, int D, int G, int S, int B, 
                   const std::string& model, double W, double L) {
        if (isFrozen) return;
        Mosfet m;
        m.instanceName = name;
        m.drain = D; m.gate = G; m.source = S; m.body = B;
        m.modelName = model;
        m.w = W; m.l = L;
        
        mosfets.push_back(m);
        updateNodeCount(D); updateNodeCount(G); updateNodeCount(S); updateNodeCount(B);
    }
    void addBJT(const std::string& name, int C, int B, int E, 
                const std::string& model, bool isNPN = true,
                double betaF = 100.0, double betaR = 1.0, 
                double Is = 1e-14, double Vt = 0.02585) {
        if (isFrozen) return;
        
        BJT newBjt;
        newBjt.instanceName = name;
        newBjt.nodeCollector = C;
        newBjt.base = B;
        newBjt.emitter = E;
        newBjt.modelName = model;
        newBjt.isNPN = isNPN;
        newBjt.betaF = betaF;
        newBjt.betaR = betaR;
        newBjt.saturationCurrent_I_S_A = Is;
        newBjt.thermalVoltage_V_T_V = Vt;
        
        // Defensive Fix: Prevent invalid Is
        if (newBjt.saturationCurrent_I_S_A <= 1e-20) {
             std::cerr << "[FIX] BJT " << name << " initialized with Is=" << newBjt.saturationCurrent_I_S_A 
                      << ". Forcing default 1e-14." << std::endl;
            newBjt.saturationCurrent_I_S_A = 1e-14;
        }

        bjts.push_back(newBjt);
        updateNodeCount(C); updateNodeCount(B); updateNodeCount(E);
    }
    void addJFET(const std::string& name, int D, int G, int S, 
                 const std::string& model, bool isNChannel = true,
                 double beta = 1e-4, double Vto = -2.0, double lambda = 0.01) {
        if (isFrozen) return;
        
        JFET newJfet;
        newJfet.instanceName = name;
        newJfet.drain = D;
        newJfet.gate = G;
        newJfet.source = S;
        newJfet.modelName = model;
        newJfet.isNChannel = isNChannel;
        newJfet.beta = beta;
        newJfet.Vto = Vto;
        newJfet.lambda = lambda;
        
        jfets.push_back(newJfet);
        updateNodeCount(D); updateNodeCount(G); updateNodeCount(S);
    }
    void addSchottkyDiode(const std::string& name, int anode, int cathode, 
                          const std::string& model = "BAT54",
                          double Is = 200e-9, double N = 1.05, double Rs = 1.0) {
        if (isFrozen) return;
        
        SchottkyDiode newD;
        newD.instanceName = name;
        newD.anode = anode;
        newD.cathode = cathode;
        newD.modelName = model;
        newD.saturationCurrent_I_S_A = Is;
        newD.emissionCoefficient_N = N;
        newD.seriesResistance_Rs_ohms = Rs;
        
        schottkyDiodes.push_back(newD);
        updateNodeCount(anode); updateNodeCount(cathode);
    }
    void addZenerDiode(const std::string& name, int anode, int cathode, 
                       double Vz = 5.1, double Rz = 7.0, double Izt = 0.049) {
        if (isFrozen) return;
        
        ZenerDiode newD;
        newD.instanceName = name;
        newD.anode = anode;
        newD.cathode = cathode;
        newD.breakdownVoltage_V = Vz;
        newD.seriesResistance_Rs_ohms = Rz;
        newD.currentAtBreakdown_A = Izt;
        
        zenerDiodes.push_back(newD);
        updateNodeCount(anode); updateNodeCount(cathode);
    }
    void addTransmissionLine(int p1p, int p1n, int p2p, int p2n, double Z0, double Td) {
        if (isFrozen) return;
        transmissionLines.push_back({p1p, p1n, p2p, p2n, Z0, Td});
        updateNodeCount(p1p); updateNodeCount(p1n); updateNodeCount(p2p); updateNodeCount(p2n);
    }
    void addProbe(const std::string& name, int nPos, int nNeg) {
        if (isFrozen) return;
        probes.push_back({name, nPos, nNeg});
        // NOTE: Probes do NOT call updateNodeCount because they are passive observers
        // and should not inflate the matrix dimensions.
    }

    // Dependent Sources
    void addVCVS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double gain, std::string role = "") {
        if (isFrozen) return;
        voltageControlledVoltageSources.push_back({outPos, outNeg, ctrlPos, ctrlNeg, gain, role});
        updateNodeCount(outPos); updateNodeCount(outNeg); updateNodeCount(ctrlPos); updateNodeCount(ctrlNeg);
    }
    void addVCCS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double gm, std::string role = "") {
        if (isFrozen) return;
        voltageControlledCurrentSources.push_back({outPos, outNeg, ctrlPos, ctrlNeg, gm, role});
        updateNodeCount(outPos); updateNodeCount(outNeg); updateNodeCount(ctrlPos); updateNodeCount(ctrlNeg);
    }
    void addCCVS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double rm, std::string role = "") {
        if (isFrozen) return;
        // Sensing via ctrlPos-ctrlNeg (small sense resistor model in stamps.h)
        currentControlledVoltageSources.push_back({outPos, outNeg, ctrlPos, ctrlNeg, rm, role});
        updateNodeCount(outPos); updateNodeCount(outNeg); updateNodeCount(ctrlPos); updateNodeCount(ctrlNeg);
    }
    void addCCCS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double alpha, std::string role = "") {
        if (isFrozen) return;
        currentControlledCurrentSources.push_back({outPos, outNeg, ctrlPos, ctrlNeg, alpha, role});
        updateNodeCount(outPos); updateNodeCount(outNeg); updateNodeCount(ctrlPos); updateNodeCount(ctrlNeg);
    }
};

// ============================================================================
// SUBCIRCUIT DEFINITIONS (Hierarchical IC Packages)
// ============================================================================

/*
 * SubCircuitDefinition - Represents the internal circuitry of an IC package.
 * Used for hierarchical simulation where ICs like ULN2003A expand to their
 * internal Darlington transistors, resistors, and protection diodes.
 *
 * Example: ULN2003A has 16 pins, 7 Darlington channels internally.
 * Pin mapping connects external package pins to internal node indices.
 */
struct SubCircuitDefinition {
    std::string name;                         // e.g., "ULN2003A"
    std::string description;                  // Human-readable description
    
    // Package Information
    int pinCount = 0;                         // Total pins (e.g., 16 for DIP-16)
    std::vector<std::string> pinNames;        // e.g., {"1B", "2B", ..., "GND", "VCC"}
    
    // Internal Circuitry (uses local node indices starting from 0)
    TensorBlock internalBlock;
    
    // Pin-to-Internal-Node Mapping
    // Key: pin index (0-based), Value: internal node index
    std::map<int, int> pinToInternalNode;

    // Pin-to-NetUUID Mapping (Persistence)
    // Stores the original NetUUID string of the net connected to this pin at creation time.
    std::vector<std::string> pinNetUUIDs;
    
    // Helper to get internal node for a given pin
    int getInternalNode(int pinIndex) const {
        auto it = pinToInternalNode.find(pinIndex);
        return (it != pinToInternalNode.end()) ? it->second : -1;
    }
};

/*
 * ICPackageInstance - An instance of a SubCircuitDefinition placed on the schematic.
 * Stores the mapping from package pins to global netlist nodes.
 */
struct ICPackageInstance {
    std::string instanceName;                 // e.g., "U1"
    std::string subcircuitName;               // Reference to SubCircuitDefinition
    
    // Pin-to-Global-Node Mapping (set during netlist generation)
    std::map<int, int> pinToGlobalNode;       // Key: pin index, Value: global node
    
    // Position (for GUI)
    double x = 0.0, y = 0.0;
    double rotation = 0.0;
};

// BlockInstance - A placed instance of a TensorBlock in the global netlist.

struct BlockInstance {
    std::string blockName;                // Name of the TensorBlock template
    std::vector<int> nodeMapping;         // Maps local node index -> global node index
    std::map<std::string, double> parameters; // Instance-specific parameters (e.g. "W", "L")
};

// BlockState - Dynamic state for a block instance (Solving the Shared History Bug)
struct BlockState {
    struct History {
        std::vector<double> v; // [t-1, t-2, ...]
        std::vector<double> i; // [t-1, t-2, ...]
        void resize(size_t n) { v.resize(n, 0.0); i.resize(n, 0.0); }
    };

    struct MosfetOP { double vgs=0, vds=0, vbs=0; };

    std::vector<History> capacitorState;
    std::vector<History> inductorState;
    std::vector<History> powerRailState;
    std::vector<MosfetOP> mosfetState;
    std::vector<std::vector<BusProbe::DigitalSample>> busProbeState; 
    // Future: BJT OP, etc.

    // Initialize state based on block definition
    void initialize(const TensorBlock& block) {
        capacitorState.resize(block.capacitors.size());
        for(auto& h : capacitorState) h.resize(3); // default 3 slots

        inductorState.resize(block.inductors.size());
        for(auto& h : inductorState) h.resize(3);

        powerRailState.resize(block.powerRails.size());
        for(auto& h : powerRailState) h.resize(3);

        mosfetState.resize(block.mosfets.size());
        busProbeState.resize(block.busProbes.size());
    }
};

// TensorNetlist - The top-level hierarchical circuit container.

class TensorNetlist {
public:
    std::map<std::string, TensorBlock> blockDefinitions; // Library of blocks
    TensorBlock globalBlock;                             // Top-level components
    
    // State Storage (Per-instance)
    BlockState globalState;
    std::vector<BlockState> instanceStates;              // Index matches instances vector

    std::vector<BlockInstance> instances;                // Placed block instances
    int numGlobalNodes = 0;
    
    // Hierarchical Support
    std::vector<std::pair<std::string, std::string>> probes; // Name, GlobalNodeString
    std::map<std::string, std::shared_ptr<TensorBlock>> subBlocks; // Resolved sub-circuits

    // Global Simulation Context (Phase 2.9)
    SimulationEnvironment environment;

    // SPICE .MODEL card library
    std::map<std::string, ModelCard> modelCards;

    void addModelCard(const ModelCard& card) {
        modelCards[card.name] = card;
    }

    const ModelCard* findModelCard(const std::string& name) const {
        auto it = modelCards.find(name);
        return (it != modelCards.end()) ? &it->second : nullptr;
    }

    TensorNetlist() { 
        globalBlock.name = "GLOBAL"; 
        globalState.initialize(globalBlock); // Init empty global state
    }

    void defineBlock(TensorBlock block) {
        block.freeze();
        blockDefinitions[block.name] = block;
    }
    
    void addInstance(const std::string& blockName, std::vector<int> nodeMap) {
        // Sync numGlobalNodes with user-supplied mapping to avoid collisions
        for (int n : nodeMap) if (n > numGlobalNodes) numGlobalNodes = n;

        // Validation and Auto-Expansion of Node Map
        auto it = blockDefinitions.find(blockName);
        if (it != blockDefinitions.end()) {
             const TensorBlock& block = it->second;
             // Ensure nodeMap defines at least the pins? 
             // Ideally we check against pinCount.
             
             // Expand nodeMap for internal nodes
             // Typically pin nodes are 1..pinCount.
             // Internal nodes are pinCount+1..numInternalNodes.
             // Map them to new global nodes.
             
             int requiredSize = block.numInternalNodes + 1; // 0-based indexing requires size N+1
             
             if ((int)nodeMap.size() < requiredSize) {
                 int currentSize = nodeMap.size();
                 for (int i = currentSize; i < requiredSize; ++i) {
                     // Allocate new global node
                     numGlobalNodes++;
                     nodeMap.push_back(numGlobalNodes);
                 }
             }
        }
        
        instances.push_back({blockName, nodeMap});
        
        // Initialize state for new instance
        BlockState newState;
        if (it != blockDefinitions.end()) {
             newState.initialize(it->second);
        }
        instanceStates.push_back(newState);


        for (int n : nodeMap) if (n > numGlobalNodes) numGlobalNodes = n;
    }
    
    // --- Convenience Methods for Global Block ---
    void addResistor(int n1, int n2, double R) {
        globalBlock.addResistor(n1, n2, R);
        if (n1 > numGlobalNodes) numGlobalNodes = n1;
        if (n2 > numGlobalNodes) numGlobalNodes = n2;
    }
    void addResistor(const std::string& name, int n1, int n2, double R) {
        globalBlock.addResistor(name, n1, n2, R);
        if (n1 > numGlobalNodes) numGlobalNodes = n1;
        if (n2 > numGlobalNodes) numGlobalNodes = n2;
    }
    void addCapacitor(int n1, int n2, double C) {
        addCapacitor("", n1, n2, C);
    }
    void addCapacitor(const std::string& name, int n1, int n2, double C) {
        globalBlock.addCapacitor(name, n1, n2, C);
        // Sync global state
        globalState.capacitorState.emplace_back();
        globalState.capacitorState.back().resize(3);

        if (n1 > numGlobalNodes) numGlobalNodes = n1;
        if (n2 > numGlobalNodes) numGlobalNodes = n2;
    }
    void addInductor(int n1, int n2, double L) {
        addInductor("", n1, n2, L);
    }
    void addInductor(const std::string& name, int n1, int n2, double L) {
        globalBlock.addInductor(name, n1, n2, L);
        globalState.inductorState.emplace_back();
        globalState.inductorState.back().resize(3);

        if (n1 > numGlobalNodes) numGlobalNodes = n1;
        if (n2 > numGlobalNodes) numGlobalNodes = n2;
    }
    void addVoltageSource(int nPos, int nNeg, double V, std::string type = "") {
        globalBlock.addVoltageSource(nPos, nNeg, V, type);
        if (nPos > numGlobalNodes) numGlobalNodes = nPos;
        if (nNeg > numGlobalNodes) numGlobalNodes = nNeg;
    }
    void addCurrentSource(int nPos, int nNeg, double I) {
        globalBlock.addCurrentSource(nPos, nNeg, I);
        if (nPos > numGlobalNodes) numGlobalNodes = nPos;
        if (nNeg > numGlobalNodes) numGlobalNodes = nNeg;
    }
    void addPowerRail(const std::string& name, int node, double V, double ripple = 0.0) {
        globalBlock.addPowerRail(name, node, V, ripple);
        globalState.powerRailState.emplace_back();
        globalState.powerRailState.back().resize(3);

        if (node > numGlobalNodes) numGlobalNodes = node;
    }
    void addSignalSource(const std::string& name, int nPos, int nNeg, 
                         SignalSource::WaveformType type, double amp, double freq) {
        globalBlock.addSignalSource(name, nPos, nNeg, type, amp, freq);
        if (nPos > numGlobalNodes) numGlobalNodes = nPos;
        if (nNeg > numGlobalNodes) numGlobalNodes = nNeg;
    }
    void addDiode(const std::string& name, int anode, int cathode, double Is) {
        globalBlock.addDiode(name, anode, cathode, Is);
        if (anode > numGlobalNodes) numGlobalNodes = anode;
        if (cathode > numGlobalNodes) numGlobalNodes = cathode;
    }
    void addDiode(int anode, int cathode, double Is) {
        globalBlock.addDiode(anode, cathode, Is);
        if (anode > numGlobalNodes) numGlobalNodes = anode;
        if (cathode > numGlobalNodes) numGlobalNodes = cathode;
    }
    void addSchottkyDiode(const std::string& name, int anode, int cathode,
                          const std::string& model = "BAT54",
                          double Is = 200e-9, double N = 1.05, double Rs = 1.0) {
        globalBlock.addSchottkyDiode(name, anode, cathode, model, Is, N, Rs);
        if (anode > numGlobalNodes) numGlobalNodes = anode;
        if (cathode > numGlobalNodes) numGlobalNodes = cathode;
    }
    void addMosfet(const std::string& name, int D, int G, int S, int B,
                   const std::string& model, double W, double L) {
        globalBlock.addMosfet(name, D, G, S, B, model, W, L);
        globalState.mosfetState.emplace_back();
        numGlobalNodes = std::max({numGlobalNodes, D, G, S, B});
    }
    void addBJT(const std::string& name, int C, int B, int E, 
                const std::string& model, bool isNPN = true,
                double betaF = 100.0, double betaR = 1.0, 
                double Is = 1e-14, double Vt = 0.02585) {
        globalBlock.addBJT(name, C, B, E, model, isNPN, betaF, betaR, Is, Vt);
        numGlobalNodes = std::max({numGlobalNodes, C, B, E});
    }
    void addJFET(const std::string& name, int D, int G, int S, 
                 const std::string& model, bool isNChannel = true,
                 double beta = 1e-4, double Vto = -2.0, double lambda = 0.01) {
        globalBlock.addJFET(name, D, G, S, model, isNChannel, beta, Vto, lambda);
        numGlobalNodes = std::max({numGlobalNodes, D, G, S});
    }
    void addZenerDiode(const std::string& name, int anode, int cathode, 
                       double Vz = 5.1, double Rz = 7.0, double Izt = 0.049) {
        globalBlock.addZenerDiode(name, anode, cathode, Vz, Rz, Izt);
        numGlobalNodes = std::max({numGlobalNodes, anode, cathode});
    }
    void addTransmissionLine(int p1p, int p1n, int p2p, int p2n, double Z0, double Td) {
        globalBlock.addTransmissionLine(p1p, p1n, p2p, p2n, Z0, Td);
        numGlobalNodes = std::max({numGlobalNodes, p1p, p1n, p2p, p2n});
    }
    void addProbe(const std::string& name, int nPos, int nNeg) {
        globalBlock.addProbe(name, nPos, nNeg);
        // NOTE: Probes do NOT inflate numGlobalNodes because they are passive observers.
    }
    void addVCVS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double gain, std::string role = "") {
        globalBlock.addVCVS(outPos, outNeg, ctrlPos, ctrlNeg, gain, role);
        numGlobalNodes = std::max({numGlobalNodes, outPos, outNeg, ctrlPos, ctrlNeg});
    }
    void addVCCS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double gm, std::string role = "") {
        globalBlock.addVCCS(outPos, outNeg, ctrlPos, ctrlNeg, gm, role);
        numGlobalNodes = std::max({numGlobalNodes, outPos, outNeg, ctrlPos, ctrlNeg});
    }
    void addCCVS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double rm, std::string role = "") {
        globalBlock.addCCVS(outPos, outNeg, ctrlPos, ctrlNeg, rm, role);
        numGlobalNodes = std::max({numGlobalNodes, outPos, outNeg, ctrlPos, ctrlNeg});
    }
    void addCCCS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double alpha, std::string role = "") {
        globalBlock.addCCCS(outPos, outNeg, ctrlPos, ctrlNeg, alpha, role);
        numGlobalNodes = std::max({numGlobalNodes, outPos, outNeg, ctrlPos, ctrlNeg});
    }
    void addBusProbe(const std::string& name, const std::vector<int>& nodes) {
        BusProbe bp;
        bp.name = name;
        bp.bit_width = nodes.size();
        bp.nodeIndices = nodes;
        globalBlock.busProbes.push_back(bp);
    }
};

// ============================================================================
// SUBCIRCUIT EXTRACTION (Phase 1.7 - Component Abstraction)
// ============================================================================

/**
 * PortInfo - Describes an external port for subcircuit extraction.
 * Used to specify which nodes in the source netlist become pins in the subcircuit.
 */
struct PortInfo {
    std::string pinName;      // Name for the pin (e.g., "VIN", "VOUT", "GND")
    int globalNodeId;         // Node ID in the source TensorNetlist
    bool isPowerPin = false;  // True for VDD/VSS pins (affects symbol rendering)
    std::string netUUIDStr;   // Serialized NetUUID for persistence
};

/**
 * extractSubCircuit
 * 
 * Converts a flat TensorNetlist into a reusable SubCircuitDefinition.
 * 
 * @param source      The TensorNetlist to extract from (typically a Bench/Block design)
 * @param name        Name for the new subcircuit (e.g., "MyInverter")
 * @param description Human-readable description
 * @param ports       List of PortInfo specifying which nodes become external pins
 * @return            A SubCircuitDefinition ready for library storage
 * 
 * The algorithm:
 * 1. Create a new TensorBlock from the source's globalBlock.
 * 2. Build a node remapping table (global node → local node).
 * 3. Port nodes are mapped to sequential local indices (1, 2, 3...).
 * 4. Internal-only nodes are mapped after port nodes.
 * 5. All component node references are rewritten using the remap table.
 * 6. The resulting SubCircuitDefinition has pinToInternalNode set correctly.
 */
inline SubCircuitDefinition extractSubCircuit(
    const TensorNetlist& source,
    const std::string& name,
    const std::string& description,
    const std::vector<PortInfo>& ports)
{
    SubCircuitDefinition result;
    result.name = name;
    result.description = description;
    result.pinCount = static_cast<int>(ports.size());
    
    // Build the pin names and port-to-node mapping
    // Local node 0 = internal ground (if present)
    // Local nodes 1..N = external pins
    // Local nodes N+1.. = internal nodes
    
    std::map<int, int> globalToLocal;  // Global node → Local node
    int nextLocalNode = 1;  // Start at 1 (0 is reserved for ground)
    
    // Step 1: Map port nodes first (in order specified)
    for (size_t i = 0; i < ports.size(); ++i) {
        const auto& port = ports[i];
        result.pinNames.push_back(port.pinName);
        result.pinNetUUIDs.push_back(port.netUUIDStr);
        
        if (globalToLocal.find(port.globalNodeId) == globalToLocal.end()) {
            globalToLocal[port.globalNodeId] = nextLocalNode;
            result.pinToInternalNode[static_cast<int>(i)] = nextLocalNode;
            nextLocalNode++;
        } else {
            // Port references an already-mapped node (shared pin)
            result.pinToInternalNode[static_cast<int>(i)] = globalToLocal[port.globalNodeId];
        }
    }
    
    // Step 2: Collect all nodes used in the source netlist
    std::set<int> allNodes;
    const auto& block = source.globalBlock;
    
    for (const auto& r : block.resistors) {
        allNodes.insert(r.nodeTerminal1);
        allNodes.insert(r.nodeTerminal2);
    }
    for (const auto& c : block.capacitors) {
        allNodes.insert(c.nodePlate1);
        allNodes.insert(c.nodePlate2);
    }
    for (const auto& l : block.inductors) {
        allNodes.insert(l.nodeCoil1);
        allNodes.insert(l.nodeCoil2);
    }
    for (const auto& vs : block.voltageSources) {
        allNodes.insert(vs.nodePositive);
        allNodes.insert(vs.nodeNegative);
    }
    for (const auto& cs : block.currentSources) {
        allNodes.insert(cs.nodePositive);
        allNodes.insert(cs.nodeNegative);
    }
    for (const auto& d : block.diodes) {
        allNodes.insert(d.anode);
        allNodes.insert(d.cathode);
    }
    for (const auto& m : block.mosfets) {
        allNodes.insert(m.drain);
        allNodes.insert(m.gate);
        allNodes.insert(m.source);
        allNodes.insert(m.body);
    }
    for (const auto& b : block.bjts) {
        allNodes.insert(b.nodeCollector);
        allNodes.insert(b.base);
        allNodes.insert(b.emitter);
    }
    
    // Step 3: Map internal-only nodes
    for (int node : allNodes) {
        if (node == 0) continue;  // Ground stays as 0
        if (globalToLocal.find(node) == globalToLocal.end()) {
            globalToLocal[node] = nextLocalNode++;
        }
    }
    globalToLocal[0] = 0;  // Ground is always 0
    
    // Helper lambda to remap a node
    auto remap = [&](int globalNode) -> int {
        auto it = globalToLocal.find(globalNode);
        return (it != globalToLocal.end()) ? it->second : 0;
    };
    
    // Step 4: Copy and remap all components
    TensorBlock& out = result.internalBlock;
    out.name = name + "_internal";
    out.numInternalNodes = nextLocalNode - 1;
    out.pinNames = block.pinNames; // Copy pin metadata
    
    // Resistors
    for (const auto& r : block.resistors) {
        Resistor newR = r;
        newR.nodeTerminal1 = remap(r.nodeTerminal1);
        newR.nodeTerminal2 = remap(r.nodeTerminal2);
        out.resistors.push_back(newR);
    }
    
    // Capacitors
    for (const auto& c : block.capacitors) {
        Capacitor newC = c;
        newC.nodePlate1 = remap(c.nodePlate1);
        newC.nodePlate2 = remap(c.nodePlate2);
        out.capacitors.push_back(newC);
    }
    
    // Inductors
    for (const auto& l : block.inductors) {
        Inductor newL = l;
        newL.nodeCoil1 = remap(l.nodeCoil1);
        newL.nodeCoil2 = remap(l.nodeCoil2);
        out.inductors.push_back(newL);
    }
    
    // Voltage Sources (Note: These become internal sources in the subcircuit)
    for (const auto& vs : block.voltageSources) {
        VoltageSource newVS = vs;
        newVS.nodePositive = remap(vs.nodePositive);
        newVS.nodeNegative = remap(vs.nodeNegative);
        out.voltageSources.push_back(newVS);
    }
    
    // Current Sources
    for (const auto& cs : block.currentSources) {
        CurrentSource newCS = cs;
        newCS.nodePositive = remap(cs.nodePositive);
        newCS.nodeNegative = remap(cs.nodeNegative);
        out.currentSources.push_back(newCS);
    }
    
    // Diodes
    for (const auto& d : block.diodes) {
        Diode newD = d;
        newD.anode = remap(d.anode);
        newD.cathode = remap(d.cathode);
        out.diodes.push_back(newD);
    }
    
    // MOSFETs
    for (const auto& m : block.mosfets) {
        Mosfet newM = m;
        newM.drain = remap(m.drain);
        newM.gate = remap(m.gate);
        newM.source = remap(m.source);
        newM.body = remap(m.body);
        out.mosfets.push_back(newM);
    }
    
    // BJTs
    for (const auto& b : block.bjts) {
        BJT newB = b;
        newB.nodeCollector = remap(b.nodeCollector);
        newB.base = remap(b.base);
        newB.emitter = remap(b.emitter);
        out.bjts.push_back(newB);
    }
    
    // Note: Probes, BusProbes, and other runtime-only structures are NOT copied
    // to the subcircuit definition (they are measurement tools, not circuit elements).
    
    return result;
}
