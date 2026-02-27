#pragma once
#include "linalg.h"
#include <vector>
#include <cmath>
#include <algorithm>

/*
 * stamps.h - Component Stamping Logic
 * This file contains the low-level physics-to-matrix mapping for each circuit element.
 */

inline void stampResistor(int nodeI, int nodeJ, double resistance, MatrixConstructor& matrixBuilder)
{
    double g = 1.0 / resistance;
    if (nodeI != 0) 
    {
        matrixBuilder.add(nodeI - 1, nodeI - 1, g);
        if (nodeJ != 0) 
            matrixBuilder.add(nodeI - 1, nodeJ - 1, -g);
    }
    if (nodeJ != 0) 
    {
        matrixBuilder.add(nodeJ - 1, nodeJ - 1, g);
        if (nodeI != 0)
            matrixBuilder.add(nodeJ - 1, nodeI - 1, -g);
    }
}

inline void stampVoltageSourceAsNorton(int nodePos, int nodeNeg, double voltage, 
                                        MatrixConstructor& matrixBuilder, std::vector<double>& rhsVector)
{
    // std::cout << "StampVSource: " << nodePos << " " << nodeNeg << " V=" << voltage << "\n";

    // We treat independent voltage sources as high-conductance Norton equivalents 
    // to keep the MNA matrix entirely in the voltage domain.
    double r_int = 1e-3;
    double g_int = 1.0 / r_int;
    double i_eq = voltage * g_int;
    
    if (nodePos != 0) {
        matrixBuilder.add(nodePos - 1, nodePos - 1, g_int);
        rhsVector[nodePos - 1] += i_eq;
    }
    if (nodeNeg != 0) {
        matrixBuilder.add(nodeNeg - 1, nodeNeg - 1, g_int);
        rhsVector[nodeNeg - 1] -= i_eq;
    }
    if (nodePos != 0 && nodeNeg != 0) {
        matrixBuilder.add(nodePos - 1, nodeNeg - 1, -g_int);
        matrixBuilder.add(nodeNeg - 1, nodePos - 1, -g_int);
    }
}

/**
 * Stamp Power Rail (PDN Model)
 * Models: V_ideal -> ESR -> Node -> C_decap -> GND
 * 
 * At Node:
 * 1. Conductance G_esr = 1/ESR to V_ideal
 * 2. Conductance G_cap = 2C/h to GND
 * 
 * Matrix[Node, Node] += G_esr + G_cap
 * RHS[Node] += V_ideal * G_esr + I_cap_eq
 */
inline void stampPowerRail(int node, double V_nom, double V_ripple, double freq,
                          double ESR, double C, double h, double time,
                          double v_prev, double i_prev_cap,
                          MatrixConstructor& matrixBuilder, std::vector<double>& rhsVector)
{
    if (node == 0) return; // Cannot drive Ground
    
    // 1. Calculate Source Voltage V(t)
    double V_source = V_nom;
    if (V_ripple > 1e-9) {
        V_source += (V_ripple / 2.0) * std::sin(2.0 * M_PI * freq * time);
    }
    
    // 2. Conductances
    double G_esr = 1.0 / ESR;
    double G_cap_eq = (h > 1e-15) ? (2.0 * C / h) : 0.0;
    
    // 3. Current Sources
    // I_from_source = (V_source - V_node) * G_esr => Stamped as Isrc = V_source * G_esr (Norton)
    double I_norton_src = V_source * G_esr;
    
    // I_cap_eq (Trapezoidal)
    // i_cap(t) = G_cap_eq * (v_node - 0) + I_eq_hist
    // I_eq_hist = -G_cap_eq * v_prev - i_prev_cap
    // MNA: G * v = I_injected
    // I_injected_from_cap_model = -I_eq_hist = G_cap_eq * v_prev + i_prev_cap
    double I_cap_inj = 0.0;
    if (h > 1e-15) {
        I_cap_inj = G_cap_eq * v_prev + i_prev_cap;
    }
    
    // 4. Stamp Matrix
    matrixBuilder.add(node - 1, node - 1, G_esr + G_cap_eq);
    
    // 5. Stamp RHS
    rhsVector[node - 1] += I_norton_src + I_cap_inj;
}

inline void stampCapacitorTrapezoidal(int nodeI, int nodeJ, double C, double h, 
                                       double v_prev, double i_prev, 
                                       MatrixConstructor& matrixBuilder, std::vector<double>& rhsVector)
{
    // Trapezoidal Companion Model (Thevenin/Norton equivalent)
    double G_eq = (2.0 * C) / h;
    double I_eq = -G_eq * v_prev - i_prev;

    if (nodeI != 0) 
    {
        matrixBuilder.add(nodeI - 1, nodeI - 1, G_eq);
        if (nodeJ != 0) 
            matrixBuilder.add(nodeI - 1, nodeJ - 1, -G_eq);
    }
    if (nodeJ != 0) 
    {
        matrixBuilder.add(nodeJ - 1, nodeJ - 1, G_eq);
        if (nodeI != 0) 
            matrixBuilder.add(nodeJ - 1, nodeI - 1, -G_eq);
    }

    if (nodeI != 0) 
        rhsVector[nodeI - 1] -= I_eq;
    if (nodeJ != 0) 
        rhsVector[nodeJ - 1] += I_eq;
}

/**
 * Inductor Trapezoidal Companion Model
 * Physics: v_L = L * di/dt
 * Companion Model (Norton equivalent):
 *   R_eq = 2*L / h (equivalent resistance)
 *   I_eq = i_prev + (v_prev / R_eq)
 * Stamps R_eq in parallel with I_eq current source.
 */
inline void stampInductorTrapezoidal(int nodeI, int nodeJ, double L, double h, 
                                      double i_prev, double v_prev, 
                                      MatrixConstructor& matrixBuilder, std::vector<double>& rhsVector)
{
    // R_eq = 2*L / h => G_eq = h / (2*L)
    double G_eq = h / (2.0 * L);
    double I_eq = i_prev + G_eq * v_prev;

    // Stamp conductance G_eq (like resistor)
    if (nodeI != 0) 
    {
        matrixBuilder.add(nodeI - 1, nodeI - 1, G_eq);
        if (nodeJ != 0) 
            matrixBuilder.add(nodeI - 1, nodeJ - 1, -G_eq);
    }
    if (nodeJ != 0) 
    {
        matrixBuilder.add(nodeJ - 1, nodeJ - 1, G_eq);
        if (nodeI != 0) 
            matrixBuilder.add(nodeJ - 1, nodeI - 1, -G_eq);
    }

    // Stamp equivalent current source
    if (nodeI != 0) 
        rhsVector[nodeI - 1] += I_eq;
    if (nodeJ != 0) 
        rhsVector[nodeJ - 1] -= I_eq;
}



/**
 * stampMosfet - 4-Terminal PDK-Ready MOSFET Stamp
 * Supports Body Effect (Bulk terminal) and NR Linearization.
 * 
 * i_d = I_eq + gm*vgs + gmb*vbs + gds*vds
 * I_eq = I_drain_actual - (gm*vgs_op + gmb*vbs_op + gds*vds_op)
 */
inline void stampMosfet(int d, int g, int s, int b, 
                        double gm, double gmb, double gds, double i_eq, 
                        MatrixConstructor& mat, std::vector<double>& rhs) 
{
    auto add = [&](int r, int c, double val) 
    {
        if (r != 0 && c != 0) 
            mat.add(r-1, c-1, val);
    };
    
    // 1. Conductance Matrix (G)
    
    // gm (VGS control)
    add(d, g, gm);   add(d, s, -gm);
    add(s, g, -gm);  add(s, s, gm);
    
    // gmb (VBS control)
    add(d, b, gmb);  add(d, s, -gmb);
    add(s, b, -gmb); add(s, s, gmb);
    
    // gds (VDS control)
    add(d, d, gds);  add(d, s, -gds);
    add(s, d, -gds); add(s, s, gds);
    
    // 2. RHS Vector (Equivalent Current)
    if (d != 0) rhs[d-1] -= i_eq; // current leaving drain
    if (s != 0) rhs[s-1] += i_eq; // current entering source
}

inline void stampDiode(int anode, int cathode, double g_d, double i_d, double v_d, 
                       MatrixConstructor& mat, std::vector<double>& rhs) {
    // Linearization: I = I_eq + G_eq * V
    // where I_eq = i_d - g_d * v_d
    double i_eq = i_d - g_d * v_d;
    
    // Stamp Conductance (G)
    // +Node gets +G, -Node gets +G, Mutual gets -G
    if (anode != 0) {
        mat.add(anode-1, anode-1, g_d);
        if (cathode != 0) mat.add(anode-1, cathode-1, -g_d);
        rhs[anode-1] -= i_eq; // Current LEAVING anode
    }
    if (cathode != 0) {
        mat.add(cathode-1, cathode-1, g_d);
        if (anode != 0) mat.add(cathode-1, anode-1, -g_d);
        rhs[cathode-1] += i_eq; // Current ENTERING cathode
    }
}

inline void stampBJT(int c, int b, int e, double g_cc, double g_cb, double g_ce,
                     double g_bc, double g_bb, double g_be,
                     double g_ec, double g_eb, double g_ee,
                     double ic, double ib, double ie,
                     double vc, double vb, double ve,
                     MatrixConstructor& mat, std::vector<double>& rhs) {

    // Stamps for 3-terminal device are 3x3 block
    // RHS = I_equiv = I_k - (G_k * V_k) ... but vector form

    // I_eq_c = ic - (g_cc*vc + g_cb*vb + g_ce*ve)
    // I_eq_b = ib - (g_bc*vc + g_bb*vb + g_be*ve)
    // I_eq_e = ie - (g_ec*vc + g_eb*vb + g_ee*ve)

    double i_eq_c = ic - (g_cc*vc + g_cb*vb + g_ce*ve);
    double i_eq_b = ib - (g_bc*vc + g_bb*vb + g_be*ve);
    double i_eq_e = ie - (g_ec*vc + g_eb*vb + g_ee*ve);

    auto addG = [&](int r, int c, double val) {
        if (r!=0 && c!=0) mat.add(r-1, c-1, val);
    };

    auto addRHS = [&](int r, double val) {
        if (r!=0) rhs[r-1] -= val; // RHS is -I_eq (current entering node)
    };

    // Collector Row
    addG(c, c, g_cc); addG(c, b, g_cb); addG(c, e, g_ce);
    addRHS(c, i_eq_c);

    // Base Row
    addG(b, c, g_bc); addG(b, b, g_bb); addG(b, e, g_be);
    addRHS(b, i_eq_b);

    // Emitter Row
    addG(e, c, g_ec); addG(e, b, g_eb); addG(e, e, g_ee);
    addRHS(e, i_eq_e);
}

/**
 * VCVS - Voltage-Controlled Voltage Source
 * Vout = gain * Vcontrol
 *
 * Circuit model: High-impedance input (control), low-impedance output
 * ctrlPos/ctrlNeg: Control voltage terminals (sense)
 * outPos/outNeg: Output voltage terminals (drive)
 * gain: Voltage gain (dimensionless)
 *
 * Implementation: Norton equivalent with high-conductance voltage source
 * V_out = gain * (V_ctrl+ - V_ctrl-)
 */
inline void stampVCVS(int ctrlPos, int ctrlNeg, int outPos, int outNeg,
                      double gain, MatrixConstructor& mat, std::vector<double>& /*rhs*/)
{
    // Use Norton equivalent with low internal resistance
    double r_int = 1e-4; // 100 uOhm internal resistance
    double g_int = 1.0 / r_int;
    
    // Output resistance stamp
    if (outPos != 0) {
        mat.add(outPos - 1, outPos - 1, g_int);
        if (outNeg != 0) {
            mat.add(outPos - 1, outNeg - 1, -g_int);
            mat.add(outNeg - 1, outPos - 1, -g_int);
        }
    }
    if (outNeg != 0) mat.add(outNeg - 1, outNeg - 1, g_int);

    // Transconductance stamp: I_out = G * gain * (V_ctrl+ - V_ctrl-)
    // This allows single-iteration convergence for linear circuits.
    double gm = gain * g_int;
    if (outPos != 0) {
        if (ctrlPos != 0) mat.add(outPos - 1, ctrlPos - 1, -gm);
        if (ctrlNeg != 0) mat.add(outPos - 1, ctrlNeg - 1, gm);
    }
    if (outNeg != 0) {
        if (ctrlPos != 0) mat.add(outNeg - 1, ctrlPos - 1, gm);
        if (ctrlNeg != 0) mat.add(outNeg - 1, ctrlNeg - 1, -gm);
    }
}

/**
 * VCCS - Voltage-Controlled Current Source
 * Iout = gm * Vcontrol
 *
 * ctrlPos/ctrlNeg: Control voltage terminals (sense)
 * outPos/outNeg: Output current terminals (current flows from outPos to outNeg)
 * gm: Transconductance (Siemens)
 */
inline void stampVCCS(int ctrlPos, int ctrlNeg, int outPos, int outNeg,
                      double gm, MatrixConstructor& mat, std::vector<double>& /*rhs*/)
{
    // VCCS stamps transconductance directly into the matrix
    // I_out = gm * (V_ctrl+ - V_ctrl-)
    // Current enters outPos, leaves outNeg

    if (outPos != 0 && ctrlPos != 0) mat.add(outPos - 1, ctrlPos - 1, gm);
    if (outPos != 0 && ctrlNeg != 0) mat.add(outPos - 1, ctrlNeg - 1, -gm);
    if (outNeg != 0 && ctrlPos != 0) mat.add(outNeg - 1, ctrlPos - 1, -gm);
    if (outNeg != 0 && ctrlNeg != 0) mat.add(outNeg - 1, ctrlNeg - 1, gm);
}

/**
 * CCVS - Current-Controlled Voltage Source
 * Vout = rm * Icontrol
 *
 * This requires sensing current through a branch, which needs MNA extension.
 * For simplicity, we model it as sensing voltage across a small resistor.
 *
 * ctrlPos/ctrlNeg: Control current sense terminals (current enters ctrlPos)
 * outPos/outNeg: Output voltage terminals
 * rm: Transresistance (Ohms)
 * r_sense: Sense resistor value (default 1 mOhm)
 */
inline void stampCCVS(int ctrlPos, int ctrlNeg, int outPos, int outNeg,
                      double rm, MatrixConstructor& mat, std::vector<double>& /*rhs*/,
                      double r_sense = 1.0)
{
    // Sense current via voltage across sense resistor
    double g_sense = 1.0 / r_sense;
    if (ctrlPos != 0) {
        mat.add(ctrlPos - 1, ctrlPos - 1, g_sense);
        if (ctrlNeg != 0) {
            mat.add(ctrlPos - 1, ctrlNeg - 1, -g_sense);
            mat.add(ctrlNeg - 1, ctrlPos - 1, -g_sense);
        }
    }
    if (ctrlNeg != 0) mat.add(ctrlNeg - 1, ctrlNeg - 1, g_sense);

    // CCVS is basically a VCVS where V_ctrl is actually V_sense (= I * R_sense)
    // V_out = rm * I = rm * (V_sense / R_sense) = (rm / R_sense) * V_sense
    // So current gain A = rm / R_sense
    double gain = rm / r_sense;
    
    double r_out = 1e-4;
    double g_out = 1.0 / r_out;
    
    if (outPos != 0) mat.add(outPos - 1, outPos - 1, g_out);
    if (outNeg != 0) mat.add(outNeg - 1, outNeg - 1, g_out);
    if (outPos != 0 && outNeg != 0) {
        mat.add(outPos - 1, outNeg - 1, -g_out);
        mat.add(outNeg - 1, outPos - 1, -g_out);
    }

    double gm = gain * g_out;
    if (outPos != 0) {
        if (ctrlPos != 0) mat.add(outPos - 1, ctrlPos - 1, -gm);
        if (ctrlNeg != 0) mat.add(outPos - 1, ctrlNeg - 1, gm);
    }
    if (outNeg != 0) {
        if (ctrlPos != 0) mat.add(outNeg - 1, ctrlPos - 1, gm);
        if (ctrlNeg != 0) mat.add(outNeg - 1, ctrlNeg - 1, -gm);
    }
}

/**
 * CCCS - Current-Controlled Current Source
 * Iout = alpha * Icontrol
 *
 * ctrlPos/ctrlNeg: Control current sense terminals
 * outPos/outNeg: Output current terminals
 * alpha: Current gain (dimensionless)
 * r_sense: Sense resistor value (default 1 mOhm)
 */
inline void stampCCCS(int ctrlPos, int ctrlNeg, int outPos, int outNeg,
                      double alpha, MatrixConstructor& mat, std::vector<double>& /*rhs*/,
                      double r_sense = 1.0)
{
    // Sense current
    double g_sense = 1.0 / r_sense;
    if (ctrlPos != 0) {
        mat.add(ctrlPos - 1, ctrlPos - 1, g_sense);
        if (ctrlNeg != 0) {
            mat.add(ctrlPos - 1, ctrlNeg - 1, -g_sense);
            mat.add(ctrlNeg - 1, ctrlPos - 1, -g_sense);
        }
    }
    if (ctrlNeg != 0) mat.add(ctrlNeg - 1, ctrlNeg - 1, g_sense);

    // CCCS: I_out = alpha * I_sense = alpha * (V_sense / R_sense) = (alpha / R_sense) * V_sense
    // This is just a VCCS with gm = alpha / R_sense
    double gm = alpha / r_sense;
    
    if (outPos != 0) {
        if (ctrlPos != 0) mat.add(outPos - 1, ctrlPos - 1, gm);
        if (ctrlNeg != 0) mat.add(outPos - 1, ctrlNeg - 1, -gm);
    }
    if (outNeg != 0) {
        if (ctrlPos != 0) mat.add(outNeg - 1, ctrlPos - 1, -gm);
        if (ctrlNeg != 0) mat.add(outNeg - 1, ctrlNeg - 1, gm);
    }
}

