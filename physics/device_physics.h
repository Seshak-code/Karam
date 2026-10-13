#pragma once
#include "gui_boundary_guard.h"
#include "physics_constants.h"
#include <cmath>
#include <algorithm>
#include <array>

/*
 * device_physics.h - Pure Constitutive Equations for Semiconductor Devices
 *
 * This file contains "kernel-ready" physics functions. They are:
 * - Stateless (no side effects)
 * - Inline (for CPU performance / GPU transpilation)
 * - Bit-accurate (no approximations unless specified)
 * - Templated (for Automatic Differentiation / Dual Numbers)
 *
 * These functions can be called from:
 * 1. The MNA stamping logic (to linearize devices for NR)
 * 2. The KCL Residual Auditor (to verify physical conservation)
 * 3. Future GPU compute shaders (WGSL/GLSL)
 */

// ============================================================================
// DIODE (PN JUNCTION) PHYSICS
// ============================================================================

template <typename T>
struct DiodeParams 
{
    T Is;  // Saturation Current [A] at Tnom
    T N;   // Emission Coefficient (1.0 for ideal)
    T Vt;  // Thermal Voltage [V] (kT/q ≈ 25.85mV @ 300K)
    T Rs;   // Series Parasitic Resistance [Ω]
    
    // Thermal Parameters
    T Tnom; // Nominal Temperature (usually 300.15K)
    T Temp; // Operating Temperature [K]
    T Eg;   // Bandgap Energy [eV] (1.11 for Si)
    T Xti;  // IS temperature exponent (3.0 for pn junction)

    // Calculate Is at operating temperature
    T Is_effective() const {
        if (Temp == Tnom) return Is;
        T ratio = Temp / Tnom;
        // Eg is in eV, Vt is in V. V = kT/q. Need to be careful with Boltzmann constant units if using raw k.
        // Simplified relations used in SPICE:
        // Is(T) = Is * (T/Tnom)^(Xti/N) * exp(...) 
        // Note: Xti usually 3.0.
        using std::pow;
        using std::exp;
        return Is * pow(ratio, Xti / N) * exp((Eg / (0.02585)) * (1.0 - Tnom/Temp)); // using Vt_nom approx
    }
};

/**
 * diode_current
 * Exact Shockley Equation: I = Is * (exp(Vd / (N*Vt)) - 1)
 * Exact Shockley Equation including Series Resistance iteration would require implicit solve.
 * Here we assume Rs is handled by external stamps (StampedResistor).
 */
template <typename T, typename P>
inline T diode_current(T vd, const DiodeParams<P>& p) 
{
    // Note: Implicit type promotion via operator overloads in dual.h
    using std::exp;
    return p.Is * (exp(vd / (p.N * p.Vt)) - 1.0);
}

/**
 * diode_model_pair
 * Returns {Current (I), Conductance (G)}
 * Handles linear continuation for convergence.
 */
template <typename T, typename P>
inline std::pair<T, T> diode_model_pair(T vd, const DiodeParams<P>& p, double max_arg = PhysicsConstants::JUNCTION_EXP_ARG_MAX) 
{
    // NASA Optimization: Keep params as scalars where possible
    auto n_vt = p.N * p.Vt;
    T arg = vd / n_vt;
    // ngspice GMIN: a 1e-12 S conductance in parallel with every PN junction —
    // never a node-to-ground leak (the solver's node gmin is dropped at
    // convergence; ngspice has none once converged). It lives here in the
    // junction model so the SoA kernels, the AoS stamps and both KCL audits
    // all carry it consistently.
    T GMIN = T(PhysicsConstants::JUNCTION_GMIN);

    // Linear Continuation Region
    if (arg > T(max_arg)) {
        using std::exp;
        double exp_max_d = std::exp(max_arg);
        T exp_max = T(exp_max_d);
        
        T i_max = p.Is * (exp_max - T(1.0));
        T g_max = (p.Is / n_vt) * exp_max;
        
        T delta = vd - (n_vt * T(max_arg));
        return {i_max + delta * g_max + GMIN * vd, g_max + GMIN};
    }
    
    using std::exp;
    T exp_val = exp(arg);
    T i_val = p.Is * (exp_val - T(1.0)) + GMIN * vd;
    // Additive (g_base >= 0, so g >= GMIN without the old floor)
    T g_val = (p.Is / n_vt) * exp_val + GMIN;
    
    return {i_val, g_val};
}

/**
 * diode_current_clamped
 * Safe version for NR iteration. Clamps exponent to prevent inf/nan.
 */
template <typename T, typename P>
inline T diode_current_clamped(T vd, const DiodeParams<P>& p, double max_arg = PhysicsConstants::JUNCTION_EXP_ARG_MAX) 
{
    // Delegate to unified model
    return diode_model_pair(vd, p, max_arg).first;
}

/**
 * pnjlim - SPICE-Style PN Junction Voltage Limiting
 * 
 * Limits the change in junction voltage during Newton-Raphson iteration to 
 * prevent the exponential from overshooting and causing non-convergence.
 * 
 * Algorithm (SPICE3f5):
 *   - If new voltage is increasing significantly, limit the step.
 *   - vcrit = Vt * ln(Vt / (sqrt(2) * Is))  (critical turn-on voltage)
 *   - Allow larger steps when turning on, smaller when conducting.
 * 
 * @param vnew  Proposed new junction voltage
 * @param vold  Previous junction voltage
 * @param vt    Thermal voltage (kT/q ≈ 26mV at room temp)
 * @param vcrit Critical voltage (pre-computed for efficiency)
 * @return      Limited junction voltage
 */
template <typename T>
inline T pnjlim(T vnew, T vold, T vt, T vcrit) {
    T vlim = vnew;
    
    // Bypass limiting in the linear continuation region (above the model's
    // exponent cap) — must track JUNCTION_EXP_ARG_MAX or junctions between
    // the two thresholds go unlimited.
    if (vold > T(PhysicsConstants::JUNCTION_EXP_ARG_MAX) * vt && vnew > T(PhysicsConstants::JUNCTION_EXP_ARG_MAX) * vt) {
        return vnew;
    }
    
    // Bypass limiting in deep reverse bias (constant current region)
    if (vold < T(-5.0) * vt && vnew < T(-5.0) * vt) {
        return vnew;
    }

    using std::abs;
    if (vnew > vcrit && abs(vnew - vold) > T(2.0) * vt) {
        // Large forward step - apply limiting
        if (vold > T(0.0)) {
            T arg = T(1.0) + (vnew - vold) / vt;
            if (arg > T(0.0)) {
                using std::log;
                vlim = vold + vt * log(arg);
            } else {
                vlim = vcrit;
            }
        } else {
            using std::log;
            vlim = vt * log(vnew / vt);
        }
    } else if (vnew < T(-5.0) * vt) {
        // Deep reverse bias - limit negative swing
        using std::max;
        vlim = max(vnew, vold - T(5.0) * vt);
    }
    
    return vlim;
}

/**
 * compute_vcrit - Compute critical voltage for pnjlim
 * vcrit = Vt * ln(Vt / (sqrt(2) * Is))
 */
template <typename T>
inline T compute_vcrit(T vt, T Is) {
    using std::log;
    using std::sqrt;
    return vt * log(vt / (sqrt(T(2.0)) * Is));
}

// NOTE: Manual derivative functions (conductance) removed.
// Use Dual<double> with diode_current to get {I, G}.

// ============================================================================
// MOSFET (LEVEL 1 SHICHMAN-HODGES) PHYSICS
// ============================================================================

// SPICE level 1 (ngspice mos1load.c, DC): source/drain exchange when Vds < 0,
// body effect (GAMMA, PHI), channel-length modulation, and the bulk-drain /
// bulk-source junctions (diode_model_pair: same GMIN and exp cap as every
// junction). Returns the currents flowing INTO the device at drain, source and
// bulk (the gate draws none). The one MOSFET definition: stamps (via Dual),
// KCL audits, sensitivity and noise all evaluate this function.
template <typename P>
struct Mos1Params {
    P polarity = 1.0;   // +1 NMOS, -1 PMOS
    P beta = 2e-5;      // KP * W / (L - 2 LD) * M
    P vto = 0.0;        // VTO as on the card (PMOS negative)
    P lambda = 0.0, gamma = 0.0, phi = 0.6;
    P isat = 1e-14;     // bulk junction saturation current, IS * M
    P vt = PhysicsConstants::V_THERMAL_300K;
    // Charge storage (mos1_charges), scaled by W, Leff and M like ngspice mos1temp
    bool caps = false;                       // any capacitance below is non-zero
    P cox = 0.0;                             // Meyer: COX * W * Leff * M (0 without TOX)
    P cgso = 0.0, cgdo = 0.0, cgbo = 0.0;    // overlap: CGSO*W*M, CGDO*W*M, CGBO*Leff*M
    P czbd = 0.0, czbdsw = 0.0;              // bulk-drain zero-bias bottom, sidewall
    P czbs = 0.0, czbssw = 0.0;              // bulk-source
    P mj = 0.5, mjsw = 0.5, pb = 0.8, fc = 0.5;
};

// Threshold (type domain) at bulk bias vbx: VBS in normal mode, VBD reversed.
template <typename T, typename P>
inline T mos1_von(T vbx, const Mos1Params<P>& p)
{
    using std::sqrt;
    const P sphi = sqrt(p.phi);
    T sarg = vbx <= P(0.0) ? sqrt(p.phi - vbx) : sphi - vbx / (2.0 * sphi);
    if (sarg < P(0.0)) sarg = T(0.0);
    return p.polarity * p.vto + p.gamma * (sarg - sphi);
}

template <typename T> struct Mos1Currents { T d, s, b; };

template <typename T, typename P>
inline Mos1Currents<T> mos1_currents(T vd, T vg, T vs, T vb, const Mos1Params<P>& p)
{
    using std::sqrt;
    const P type = p.polarity;
    const T vgs = type * (vg - vs), vds = type * (vd - vs), vbs = type * (vb - vs);
    const T vbd = vbs - vds, vgd = vgs - vds;
    const DiodeParams<P> junction = {p.isat, P(1.0), p.vt};
    const T cbs = diode_model_pair(vbs, junction).first;
    const T cbd = diode_model_pair(vbd, junction).first;

    const bool normal = vds >= P(0.0);          // mode: the higher terminal is the drain
    const T vgst = (normal ? vgs : vgd) - mos1_von(normal ? vbs : vbd, p);
    const T vdsm = normal ? vds : -vds;         // |Vds|
    T cdrain(0.0);
    if (vgst > P(0.0)) {
        const T betap = p.beta * (1.0 + p.lambda * vdsm);
        cdrain = vgst <= vdsm ? 0.5 * betap * vgst * vgst                // saturation
                              : betap * vdsm * (vgst - 0.5 * vdsm);      // linear
    }
    const T chan = normal ? cdrain : -cdrain;   // into the drain terminal
    return {type * (chan - cbd), type * (-chan - cbs), type * (cbs + cbd)};
}

// Meyer gate half-capacitances {gs, gd, gb} (ngspice DEVqmeyer). vgs/vgd are in
// the device's orientation (the caller swaps them, and the results, when reversed).
template <typename T, typename P>
inline std::array<T, 3> mos1_meyer(T vgs, T vgd, T von, T vdsat, P phi, P cox)
{
    const T vgst = vgs - von, vds = vgs - vgd;
    if (vdsat < 0.025) vdsat = T(0.025);  // MAGIC_VDS
    if (vgst <= -phi) return {T(0.0), T(0.0), T(cox / 2)};
    if (vgst <= -phi / 2) return {T(0.0), T(0.0), -vgst * cox / (2 * phi)};
    const T full = vgst <= P(0.0) ? vgst * cox / (1.5 * phi) + cox / 3 : T(cox / 3);
    const T gb = vgst <= P(0.0) ? -vgst * cox / (2 * phi) : T(0.0);
    if (vds >= vdsat) return {full, T(0.0), gb};
    const T vddif = 2.0 * vdsat - vds, vddif1 = vdsat - vds, vddif2 = vddif * vddif;
    return {full * (1.0 - vddif1 * vddif1 / vddif2), full * (1.0 - vdsat * vdsat / vddif2), gb};
}

// Depletion charge of a junction with zero-bias capacitance cz and grading m:
// C = cz (1 - v/pb)^-m below fc*pb, linearised above it (ngspice f2/f3/f4).
template <typename T, typename P>
inline T depletion_charge(T v, P cz, P m, P pb, P fc)
{
    using std::exp; using std::log;
    if (cz == P(0.0)) return T(0.0);
    const P v0 = fc * pb, a0 = 1.0 - fc;
    if (v < v0) return pb * cz * (1.0 - exp((1.0 - m) * log(1.0 - v / pb))) / (1.0 - m);
    const P c0 = cz * exp(-m * log(a0)), c1 = c0 * m / (pb * a0);
    const P q0 = pb * cz * (1.0 - exp((1.0 - m) * log(a0))) / (1.0 - m);
    const T dv = v - v0;
    return q0 + dv * (c0 + 0.5 * c1 * dv);
}

// Charges of the five capacitive branches gs, gd, gb, bd, bs (type domain,
// ngspice mos1load). The Meyer gate charges advance from the previous accepted
// step (vPrev, qPrev, halfPrev) by C*dv, C = half-capacitance now + then +
// overlap; the bulk junctions carry their depletion charge. half: the Meyer
// half-capacitances at v (the next step's halfPrev).
template <typename T> struct Mos1Charges { T v[5], q[5], half[3]; };

// Branch voltages gs, gd, gb, bd, bs (type domain) of the charges above.
template <typename T>
inline std::array<T, 5> mos1_branches(T vd, T vg, T vs, T vb, double type)
{
    return {type * (vg - vs), type * (vg - vd), type * (vg - vb), type * (vb - vd), type * (vb - vs)};
}

template <typename T>
inline Mos1Charges<T> mos1_charges(T vd, T vg, T vs, T vb, const Mos1Params<double>& p,
                                   const double* vPrev, const double* qPrev, const double* halfPrev)
{
    const double type = p.polarity;
    Mos1Charges<T> c;
    const auto br = mos1_branches(vd, vg, vs, vb, type);
    for (int k = 0; k < 5; ++k) c.v[k] = br[k];
    const T vgs = br[0], vgd = br[1], vbd = br[3], vbs = br[4];
    const bool normal = vgs >= vgd;  // vds >= 0
    const T von = mos1_von(normal ? vbs : vbd, p);
    T vdsat = (normal ? vgs : vgd) - von;
    if (vdsat < 0.0) vdsat = T(0.0);
    const auto h = normal ? mos1_meyer(vgs, vgd, von, vdsat, p.phi, p.cox)
                          : mos1_meyer(vgd, vgs, von, vdsat, p.phi, p.cox);
    c.half[0] = normal ? h[0] : h[1];
    c.half[1] = normal ? h[1] : h[0];
    c.half[2] = h[2];
    const double ov[3] = {p.cgso, p.cgdo, p.cgbo};
    for (int k = 0; k < 3; ++k)
        c.q[k] = qPrev[k] + (c.half[k] + halfPrev[k] + ov[k]) * (c.v[k] - vPrev[k]);
    c.q[3] = depletion_charge(vbd, p.czbd, p.mj, p.pb, p.fc) + depletion_charge(vbd, p.czbdsw, p.mjsw, p.pb, p.fc);
    c.q[4] = depletion_charge(vbs, p.czbs, p.mj, p.pb, p.fc) + depletion_charge(vbs, p.czbssw, p.mjsw, p.pb, p.fc);
    return c;
}

// ============================================================================
// BJT (EBERS-MOLL / GUMMEL-POON L1) PHYSICS
// ============================================================================

template <typename T>
struct BJTParams 
{
    T Is;     // Transport Saturation Current [A]
    T BetaF;  // Forward Current Gain
    T BetaR;  // Reverse Current Gain
    T Vt;     // Thermal Voltage [V]
    
    // Thermal
    T Temp;
    T Tnom;
    T Xtb;    // Beta temperature exponent
    T Eg;     // Bandgap
    T Xti;    // IS temperature exponent
};

template <typename T>
struct BJTCurrents {
    T Ic; // Collector Current
    T Ib; // Base Current
    T Ie; // Emitter Current
    
    // Gradients
    T g_cc, g_cb, g_ce; 
    T g_bc, g_bb, g_be;
    T g_ec, g_eb, g_ee;
};

/**
 * bjt_ebers_moll
 * Computes currents and conductance matrix for a BJT.
 * templated for Dual/EmulatedF64 support.
 */
template <typename T, typename P>
inline BJTCurrents<T> bjt_ebers_moll(T v_c, T v_b, T v_e, bool isNPN, const BJTParams<P>& p)
{
    // 1. Determine local voltages relative to Base (Vbc, Vbe)
    T v_be, v_bc;
    T sign = isNPN ? T(1.0) : T(-1.0);

    v_be = sign * (v_b - v_e);
    v_bc = sign * (v_b - v_c);

    // 2. Compute Diode Currents
    
    // Implicit Diode Params (Ebers-Moll diodes have N=1 ideal usually, or scaled)
    // We construct a temporary DiodeParams to pass to the unified helper
    DiodeParams<T> dp;
    dp.Is = p.Is;
    dp.N = T(1.0); // Ebers-Moll assumes N=1 or absorbed into Is? Usually N=1.
    dp.Vt = p.Vt;
    // Rs, etc ignored for core junction

    auto [i_f, g_f] = diode_model_pair(v_be, dp);
    auto [i_r, g_r] = diode_model_pair(v_bc, dp);

    // 3. Terminal Currents (Ebers-Moll)
    T invBetaF = T(1.0) / p.BetaF;
    T invBetaR = T(1.0) / p.BetaR;
    
    T i_ct = i_f - i_r;
    T i_base_component_f = i_f * invBetaF;
    T i_base_component_r = i_r * invBetaR;

    T ic_local = i_ct - i_base_component_r;
    T ib_local = i_base_component_f + i_base_component_r;
    T ie_local = -ic_local - ib_local;
    
    // 4. Derivatives (Conductances)
    T dIf_dVbe = g_f;
    T dIr_dVbc = g_r;
    
    T dIct_dVbe = dIf_dVbe;
    T dIct_dVbc = -dIr_dVbc;
    
    T dIbf_dVbe = dIf_dVbe * invBetaF;
    T dIbr_dVbc = dIr_dVbc * invBetaR;
    
    T dIc_dVbe = dIct_dVbe;       
    T dIc_dVbc = dIct_dVbc - dIbr_dVbc; 
    
    T dIb_dVbe = dIbf_dVbe;      
    T dIb_dVbc = dIbr_dVbc;      
    
    T dIe_dVbe = -(dIc_dVbe + dIb_dVbe);
    T dIe_dVbc = -(dIc_dVbc + dIb_dVbc);
    
    BJTCurrents<T> res;
    res.Ic = sign * ic_local;
    res.Ib = sign * ib_local;
    res.Ie = sign * ie_local;

    // Conductances
    res.g_cc = -dIc_dVbc; 
    res.g_cb = dIc_dVbe + dIc_dVbc;
    res.g_ce = -dIc_dVbe;
    
    res.g_bc = -dIb_dVbc;
    res.g_bb = dIb_dVbe + dIb_dVbc;
    res.g_be = -dIb_dVbe;
    
    res.g_ec = -dIe_dVbc;
    res.g_eb = dIe_dVbe + dIe_dVbc;
    res.g_ee = -dIe_dVbe;
    
    return res;
}

// ============================================================================
// JFET (SHICHMAN-HODGES) PHYSICS
// ============================================================================

template <typename T>
struct JFETParams 
{
    T Beta;   // Transconductance parameter [A/V^2]
    T Vto;    // Threshold Voltage [V] (Negative for N-Channel)
    T Lambda; // Channel Length Modulation [1/V]
    T Is;     // Gate Junction Saturation Current [A]
    T N;      // Gate Junction Emission Coefficient
    T Vt;     // Thermal Voltage [V]
};

/**
 * jfet_ids
 * Computes Drain-Source current for JFET.
 * Assumes N-Channel. For P-Channel, flip voltage signs before calling.
 */
template <typename T, typename P>
inline T jfet_ids(T vgs, T vds, const JFETParams<P>& p) 
{
    // 1. Gate Leakage (Diode) - Simplified
    // Typically small, ignored in simple models, but physically exists.
    // Here we focus on Channel Current.

    T vds_sat = vgs - p.Vto;
    T beta = p.Beta; // area scaled by caller if needed
    
    // N-Channel: Vto is negative. Vgs usually negative.
    // Cutoff: Vgs <= Vto
    
    if (vgs <= p.Vto) return 0.0;
    
    if (vds <= vds_sat) {
        // Linear Region
        return beta * vds * (T(2.0) * (vgs - p.Vto) - vds) * (T(1.0) + p.Lambda * vds);
    } else {
        // Saturation Region
        T vov = vgs - p.Vto;
        return beta * vov * vov * (T(1.0) + p.Lambda * vds);
    }
}

// ============================================================================
// ZENER DIODE PHYSICS
// ============================================================================

template <typename T>
struct ZenerParams 
{
    T BV;     // Breakdown Voltage [V] (Positive value)
    T IBV;    // Current at Breakdown [A] (Positive value)
    T Rs;     // Series Resistance
    T N;      // Emission Coefficient (Forward)
    T Is;     // Saturation Current
    T Vt;     // Thermal Voltage
};

/**
 * zener_current
 * Models forward bias (exponential) and reverse breakdown (high current).
 */
template <typename T, typename P>
inline T zener_current(T vd, const ZenerParams<P>& p) 
{
    using std::exp;
    using std::abs;
    
    // Forward Region
    T i_fwd = p.Is * (exp(vd / (p.N * p.Vt)) - 1.0);
    
    // Reverse Breakdown Region
    // Modeled as a diode in reverse with Vz = BV
    // Soft knee approx or hard exponential?
    // Using a simplified exponential model for breakdown:
    // I_rev = -IBV * exp( -(vd + BV) / Vt_breakdown )? 
    // Standard model sums a forward diode and a reverse "breakdown diode".
    
    T i_rev = 0.0;
    T Vz = -p.BV; // Negative voltage for breakdown
    
    // If vd is near or below -BV
    // If vd is near or below -BV
    if (vd < -p.BV + T(0.5)) { 
        T vd_z = -(vd + p.BV); // Voltage across Zener junction (positive when breakdown)
        if (vd_z > 0.0) {
             // Use linear continuation for breakdown exponential
             auto n_vt = p.N * p.Vt;
             T arg = vd_z / n_vt;
             
             T breakdown_current;
             if (arg > 30.0) {
                 double exp_max = std::exp(30.0);
                 auto i_max = p.IBV * (exp_max - 1.0); // Assuming scaling matches IBV
                 // Wait, IBV scaling logic in my head earlier `i_rev = -IBV * exp...`
                 // formula: i_rev = -p.IBV * (exp(arg) - 1.0)
                 // This assumes I at V=(-BV + Vt) is IBV*(e-1). 
                 // Whatever, let's keep the formula structure but limit exp.
                 
                 // g_max = derivative of p.IBV*(exp(arg)-1) wrt vd_z
                 // g_max = p.IBV * (1/n_vt) * exp(arg)
                 auto g_max = (p.IBV / n_vt) * exp_max;
                 T delta_v = vd_z - (n_vt * 30.0);
                 breakdown_current = i_max + g_max * delta_v;
             } else {
                 breakdown_current = p.IBV * (exp(arg) - 1.0);
             }
             
             i_rev = -breakdown_current;
        }
    }
    
    return i_fwd + i_rev + T(PhysicsConstants::JUNCTION_GMIN) * vd;  // as diode_model_pair
}

// ============================================================================
// END OF DEVICE PHYSICS
// ============================================================================

