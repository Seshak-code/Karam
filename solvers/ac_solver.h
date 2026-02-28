#pragma once

#include <complex>
#include <vector>
#include <string>
#include <cmath>
#include "../netlist/circuit.h"
#include "../physics/device_physics.h"
#include "../physics/noise_models.h"
#include "../autodiff/dual.h"
#include "../infrastructure/simtrace.h"
#include "../math/linalg.h"  // Phase 2.10: ComplexCsr, ComplexMatrixConstructor, solveComplexLU_Sparse

/**
 * ac_solver.h
 *
 * AC Small-Signal Analysis and Noise Analysis solver.
 *
 * Architecture:
 *   DC Solve -> linearized conductances (gm, gds, G_diode)
 *            -> AC admittance matrix Y(jw) = G + jwC
 *            -> For each noise source: solve Y*V = I_noise
 *            -> Output PSD: S_out(f) = sum |H_n(jw)|^2 * S_source(f)
 *            -> Integrate for RMS noise
 */

namespace ACSolver {

using Complex = std::complex<double>;

// ============================================================================
// RESULT TYPES
// ============================================================================

struct ACResult {
    std::vector<double> frequencies;
    // nodeVoltages[freq_idx][node_idx] = complex voltage
    std::vector<std::vector<Complex>> nodeVoltages;
};

struct NoiseResult {
    std::vector<double> frequencies;
    std::vector<double> output_noise_psd;  // S_out(f) [V^2/Hz] at each freq
    double integrated_noise_vrms = 0.0;     // sqrt(integral S_out df) [Vrms]
};

// ============================================================================
// COMPLEX DENSE LU SOLVER
// ============================================================================

/**
 * Solve A*x = b for complex dense matrix using LU with partial pivoting.
 * A is modified in-place.
 * Returns solution vector x.
 */
inline std::vector<Complex> solveComplexLU(
    std::vector<std::vector<Complex>>& A,
    std::vector<Complex>& b)
{
    int n = static_cast<int>(A.size());
    std::vector<int> pivot(n);
    for (int i = 0; i < n; ++i) pivot[i] = i;

    // Forward elimination with partial pivoting
    for (int k = 0; k < n; ++k) {
        // Find pivot
        double maxVal = 0.0;
        int maxRow = k;
        for (int i = k; i < n; ++i) {
            double absVal = std::abs(A[i][k]);
            if (absVal > maxVal) {
                maxVal = absVal;
                maxRow = i;
            }
        }

        if (maxVal < 1e-30) {
            // Singular or near-singular
            continue;
        }

        if (maxRow != k) {
            std::swap(A[k], A[maxRow]);
            std::swap(b[k], b[maxRow]);
            std::swap(pivot[k], pivot[maxRow]);
        }

        // Eliminate below
        for (int i = k + 1; i < n; ++i) {
            Complex factor = A[i][k] / A[k][k];
            A[i][k] = factor;
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    std::vector<Complex> x(n, Complex(0.0, 0.0));
    for (int i = n - 1; i >= 0; --i) {
        Complex sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }
        if (std::abs(A[i][i]) > 1e-30) {
            x[i] = sum / A[i][i];
        }
    }

    return x;
}

// ============================================================================
// AC ADMITTANCE MATRIX BUILDER
// ============================================================================

/**
 * Build the AC admittance matrix Y(jw) = G + jwC for a given frequency.
 * Uses DC operating point for linearized nonlinear device conductances.
 *
 * @param netlist       Circuit netlist
 * @param dcSolution    DC operating point voltages
 * @param omega         Angular frequency (2*pi*f)
 * @param Y             Output: admittance matrix (n x n complex)
 * @param rhs           Output: RHS vector (for AC source excitation)
 */
inline void buildAdmittanceMatrix(
    const TensorNetlist& netlist,
    const std::vector<double>& dcSolution,
    double omega,
    std::vector<std::vector<Complex>>& Y,
    std::vector<Complex>& rhs)
{
    int n = netlist.numGlobalNodes;
    Y.assign(n, std::vector<Complex>(n, Complex(0.0, 0.0)));
    rhs.assign(n, Complex(0.0, 0.0));

    auto getV = [&](int nodeIdx) -> double {
        if (nodeIdx <= 0 || nodeIdx > (int)dcSolution.size()) return 0.0;
        return dcSolution[nodeIdx - 1];
    };

    auto addY = [&](int r, int c, Complex val) {
        if (r > 0 && c > 0 && r <= n && c <= n)
            Y[r - 1][c - 1] += val;
    };

    // 1. Resistors: G = 1/R (real conductance)
    for (const auto& r : netlist.globalBlock.resistors) {
        double G = 1.0 / r.resistance_ohms;
        Complex Gc(G, 0.0);
        addY(r.nodeTerminal1, r.nodeTerminal1, Gc);
        addY(r.nodeTerminal1, r.nodeTerminal2, -Gc);
        addY(r.nodeTerminal2, r.nodeTerminal1, -Gc);
        addY(r.nodeTerminal2, r.nodeTerminal2, Gc);
    }

    // 2. Capacitors: Y = jwC
    for (const auto& c : netlist.globalBlock.capacitors) {
        Complex Yc(0.0, omega * c.capacitance_farads);
        addY(c.nodePlate1, c.nodePlate1, Yc);
        addY(c.nodePlate1, c.nodePlate2, -Yc);
        addY(c.nodePlate2, c.nodePlate1, -Yc);
        addY(c.nodePlate2, c.nodePlate2, Yc);
    }

    // 3. Inductors: Y = 1/(jwL)
    for (const auto& l : netlist.globalBlock.inductors) {
        if (l.inductance_henries > 0.0 && omega > 0.0) {
            Complex Yl = Complex(1.0, 0.0) / Complex(0.0, omega * l.inductance_henries);
            addY(l.nodeCoil1, l.nodeCoil1, Yl);
            addY(l.nodeCoil1, l.nodeCoil2, -Yl);
            addY(l.nodeCoil2, l.nodeCoil1, -Yl);
            addY(l.nodeCoil2, l.nodeCoil2, Yl);
        }
    }

    // 4. Voltage Sources: Norton equivalent with very small resistance
    for (const auto& vs : netlist.globalBlock.voltageSources) {
        double G = 1.0 / 1e-3; // R_int = 1e-3 ohm
        Complex Gc(G, 0.0);
        addY(vs.nodePositive, vs.nodePositive, Gc);
        addY(vs.nodePositive, vs.nodeNegative, -Gc);
        addY(vs.nodeNegative, vs.nodePositive, -Gc);
        addY(vs.nodeNegative, vs.nodeNegative, Gc);
        // AC voltage sources contribute to RHS (small-signal excitation = 0 for noise analysis)
    }

    // 5. Diodes: linearized conductance g_d = dI/dV at DC OP
    for (const auto& d : netlist.globalBlock.diodes) {
        double vd = getV(d.anode) - getV(d.cathode);
        DiodeParams<double> params = {d.saturationCurrent_I_S_A, d.emissionCoefficient_N,
                                      d.thermalVoltage_V_T_V};
        Dual<double> i_dual = diode_current_clamped(Dual<double>(vd, 1.0), params);
        double g_d = std::max(i_dual.grad, 1e-15);
        Complex Gc(g_d, 0.0);
        addY(d.anode, d.anode, Gc);
        addY(d.anode, d.cathode, -Gc);
        addY(d.cathode, d.anode, -Gc);
        addY(d.cathode, d.cathode, Gc);
    }

    // 6. Schottky Diodes
    for (const auto& sd : netlist.globalBlock.schottkyDiodes) {
        double vd = getV(sd.anode) - getV(sd.cathode);
        DiodeParams<double> params = {sd.saturationCurrent_I_S_A, sd.emissionCoefficient_N,
                                      sd.thermalVoltage_V_T_V};
        Dual<double> i_dual = diode_current_clamped(Dual<double>(vd, 1.0), params);
        double g_d = std::max(i_dual.grad, 1e-15);
        Complex Gc(g_d, 0.0);
        addY(sd.anode, sd.anode, Gc);
        addY(sd.anode, sd.cathode, -Gc);
        addY(sd.cathode, sd.anode, -Gc);
        addY(sd.cathode, sd.cathode, Gc);
    }

    // 7. MOSFETs: linearized small-signal model (gm, gds)
    for (const auto& m : netlist.globalBlock.mosfets) {
        double vgs = getV(m.gate) - getV(m.source);
        double vds = getV(m.drain) - getV(m.source);

        MosfetParams<double> p;
        p.Kp = 200e-6; p.Vth = 0.7; p.lambda = 0.02;
        p.W = m.w; p.L = m.l;

        Dual<double> ids_gm = mosfet_ids(Dual<double>(vgs, 1.0), Dual<double>(vds, 0.0), p);
        double gm = ids_gm.grad;

        Dual<double> ids_gds = mosfet_ids(Dual<double>(vgs, 0.0), Dual<double>(vds, 1.0), p);
        double gds = ids_gds.grad;

        // gm: VCCS from gate-source controlling drain-source current
        addY(m.drain, m.gate, Complex(gm, 0.0));
        addY(m.drain, m.source, Complex(-gm, 0.0));
        addY(m.source, m.gate, Complex(-gm, 0.0));
        addY(m.source, m.source, Complex(gm, 0.0));

        // gds: conductance drain-source
        addY(m.drain, m.drain, Complex(gds, 0.0));
        addY(m.drain, m.source, Complex(-gds, 0.0));
        addY(m.source, m.drain, Complex(-gds, 0.0));
        addY(m.source, m.source, Complex(gds, 0.0));
    }

    // 8. BJTs: linearized 3x3 conductance matrix
    for (const auto& q : netlist.globalBlock.bjts) {
        double vc = getV(q.nodeCollector);
        double vb = getV(q.base);
        double ve = getV(q.emitter);

        BJTParams<double> params;
        params.Is = q.saturationCurrent_I_S_A;
        params.BetaF = q.betaF;
        params.BetaR = q.betaR;
        params.Vt = q.thermalVoltage_V_T_V;

        BJTCurrents<double> res = bjt_ebers_moll(vc, vb, ve, q.isNPN, params);

        // Stamp 3x3 conductance matrix
        auto stampG = [&](int r, int c, double val) {
            addY(r, c, Complex(val, 0.0));
        };
        stampG(q.nodeCollector, q.nodeCollector, res.g_cc);
        stampG(q.nodeCollector, q.base, res.g_cb);
        stampG(q.nodeCollector, q.emitter, res.g_ce);
        stampG(q.base, q.nodeCollector, res.g_bc);
        stampG(q.base, q.base, res.g_bb);
        stampG(q.base, q.emitter, res.g_be);
        stampG(q.emitter, q.nodeCollector, res.g_ec);
        stampG(q.emitter, q.base, res.g_eb);
        stampG(q.emitter, q.emitter, res.g_ee);
    }

    // 9. JFETs
    for (const auto& j : netlist.globalBlock.jfets) {
        double vgs = getV(j.gate) - getV(j.source);
        double vds = getV(j.drain) - getV(j.source);

        double sign = j.isNChannel ? 1.0 : -1.0;
        JFETParams<double> p;
        p.Beta = j.beta; p.Vto = j.Vto; p.Lambda = j.lambda;

        Dual<double> ids_gm = jfet_ids(Dual<double>(sign * vgs, 1.0), Dual<double>(sign * vds, 0.0), p);
        double gm = ids_gm.grad;

        Dual<double> ids_gds = jfet_ids(Dual<double>(sign * vgs, 0.0), Dual<double>(sign * vds, 1.0), p);
        double gds = ids_gds.grad;

        addY(j.drain, j.gate, Complex(sign * gm, 0.0));
        addY(j.drain, j.source, Complex(-sign * gm, 0.0));
        addY(j.source, j.gate, Complex(-sign * gm, 0.0));
        addY(j.source, j.source, Complex(sign * gm, 0.0));

        addY(j.drain, j.drain, Complex(sign * gds, 0.0));
        addY(j.drain, j.source, Complex(-sign * gds, 0.0));
        addY(j.source, j.drain, Complex(-sign * gds, 0.0));
        addY(j.source, j.source, Complex(sign * gds, 0.0));
    }

    // 10. Zener Diodes
    for (const auto& z : netlist.globalBlock.zenerDiodes) {
        double vd = getV(z.anode) - getV(z.cathode);
        ZenerParams<double> zp;
        zp.BV = z.breakdownVoltage_V; zp.IBV = z.currentAtBreakdown_A;
        zp.Rs = z.seriesResistance_Rs_ohms; zp.N = z.emissionCoefficient_N;
        zp.Is = z.saturationCurrent_I_S_A; zp.Vt = z.thermalVoltage_V_T_V;

        Dual<double> i_dual = zener_current(Dual<double>(vd, 1.0), zp);
        double g_d = std::max(i_dual.grad, 1e-15);
        Complex Gc(g_d, 0.0);
        addY(z.anode, z.anode, Gc);
        addY(z.anode, z.cathode, -Gc);
        addY(z.cathode, z.anode, -Gc);
        addY(z.cathode, z.cathode, Gc);
    }

    // Diagonal conditioning (GMIN)
    for (int i = 0; i < n; ++i) {
        Y[i][i] += Complex(1e-12, 0.0);
    }
}

// ============================================================================
// AC SWEEP
// ============================================================================

/**
 * Perform AC small-signal frequency sweep.
 *
 * @param netlist       Circuit netlist
 * @param dcSolution    DC operating point voltages
 * @param fStart        Start frequency [Hz]
 * @param fStop         Stop frequency [Hz]
 * @param numPoints     Number of frequency points (log-spaced)
 * @param inputNode     Node for AC excitation (1-based), 0 = none
 * @param inputAmplitude AC excitation amplitude [V]
 */
inline ACResult solveAC(
    const TensorNetlist& netlist,
    const std::vector<double>& dcSolution,
    double fStart, double fStop, int numPoints,
    int inputNode = 0, double inputAmplitude = 1.0)
{
    ACResult result;
    if (netlist.numGlobalNodes == 0 || numPoints <= 0) return result;

    int n = netlist.numGlobalNodes;
    result.frequencies.resize(numPoints);
    result.nodeVoltages.resize(numPoints);

    // Generate log-spaced frequencies
    double logStart = std::log10(std::max(fStart, 1e-3));
    double logStop = std::log10(std::max(fStop, fStart + 1.0));
    for (int i = 0; i < numPoints; ++i) {
        double logF = logStart + (logStop - logStart) * i / std::max(numPoints - 1, 1);
        result.frequencies[i] = std::pow(10.0, logF);
    }

    for (int fi = 0; fi < numPoints; ++fi) {
        double freq = result.frequencies[fi];
        double omega = 2.0 * M_PI * freq;

        std::vector<std::vector<Complex>> Y;
        std::vector<Complex> rhs;
        buildAdmittanceMatrix(netlist, dcSolution, omega, Y, rhs);

        // Add AC excitation at input node
        if (inputNode > 0 && inputNode <= n) {
            // Inject 1A current at input node (Norton-equivalent of voltage source)
            // For true voltage excitation, we'd need a very large conductance.
            // Using G_large approach: stamp large G between inputNode and ground,
            // with I = G * V_ac
            double G_large = 1e6;
            Y[inputNode - 1][inputNode - 1] += Complex(G_large, 0.0);
            rhs[inputNode - 1] += Complex(G_large * inputAmplitude, 0.0);
        }

        result.nodeVoltages[fi] = solveComplexLU(Y, rhs);
    }

    return result;
}

// ============================================================================
// NOISE SOURCE ENUMERATION
// ============================================================================

struct NoiseSource {
    int nodePos;           // 1-based, positive injection node
    int nodeNeg;           // 1-based, negative injection node (0 = ground)
    double psd;            // Current PSD [A^2/Hz] at this frequency
    std::string sourceName;
};

/**
 * Enumerate all noise current sources in the circuit at a given frequency.
 * Returns a list of independent current noise sources with their PSD.
 */
inline std::vector<NoiseSource> computeNoiseSources(
    const TensorNetlist& netlist,
    const std::vector<double>& dcSolution,
    double freq,
    double temp_K = 300.15)
{
    std::vector<NoiseSource> sources;

    auto getV = [&](int nodeIdx) -> double {
        if (nodeIdx <= 0 || nodeIdx > (int)dcSolution.size()) return 0.0;
        return dcSolution[nodeIdx - 1];
    };

    // 1. Resistor thermal noise
    for (const auto& r : netlist.globalBlock.resistors) {
        double psd = NoiseModels::resistor_thermal_current_psd(r.resistance_ohms, temp_K);
        if (psd > 0.0) {
            sources.push_back({r.nodeTerminal1, r.nodeTerminal2, psd, r.name + "_thermal"});
        }
    }

    // 2. Diode shot noise
    for (const auto& d : netlist.globalBlock.diodes) {
        double vd = getV(d.anode) - getV(d.cathode);
        DiodeParams<double> params = {d.saturationCurrent_I_S_A, d.emissionCoefficient_N,
                                      d.thermalVoltage_V_T_V};
        double I_dc = diode_current_clamped(vd, params);
        double psd = NoiseModels::diode_shot_noise_psd(I_dc);
        if (psd > 0.0) {
            sources.push_back({d.anode, d.cathode, psd, d.modelName + "_shot"});
        }
    }

    // 3. BJT noise sources
    for (const auto& q : netlist.globalBlock.bjts) {
        double vc = getV(q.nodeCollector);
        double vb = getV(q.base);
        double ve = getV(q.emitter);

        BJTParams<double> params;
        params.Is = q.saturationCurrent_I_S_A;
        params.BetaF = q.betaF;
        params.BetaR = q.betaR;
        params.Vt = q.thermalVoltage_V_T_V;

        BJTCurrents<double> res = bjt_ebers_moll(vc, vb, ve, q.isNPN, params);

        // Collector shot noise
        double psd_ic = NoiseModels::bjt_collector_shot_psd(res.Ic);
        if (psd_ic > 0.0) {
            sources.push_back({q.nodeCollector, q.emitter, psd_ic,
                              q.instanceName + "_Ic_shot"});
        }

        // Base shot noise
        double psd_ib = NoiseModels::bjt_base_shot_psd(res.Ib);
        if (psd_ib > 0.0) {
            sources.push_back({q.base, q.emitter, psd_ib,
                              q.instanceName + "_Ib_shot"});
        }

        // Base spreading resistance thermal noise (voltage source => current source via g_bb)
        if (q.r_bb > 0.0) {
            double psd_rbb = NoiseModels::bjt_base_thermal_psd(q.r_bb, temp_K);
            // Convert V^2/Hz to A^2/Hz through r_bb: I_n^2 = V_n^2 / r_bb^2
            double psd_current = psd_rbb / (q.r_bb * q.r_bb);
            sources.push_back({q.base, 0, psd_current,
                              q.instanceName + "_rbb_thermal"});
        }
    }

    // 4. MOSFET noise sources
    for (const auto& m : netlist.globalBlock.mosfets) {
        double vgs = getV(m.gate) - getV(m.source);
        double vds = getV(m.drain) - getV(m.source);

        MosfetParams<double> p;
        p.Kp = 200e-6; p.Vth = 0.7; p.lambda = 0.02;
        p.W = m.w; p.L = m.l;

        Dual<double> ids_gm = mosfet_ids(Dual<double>(vgs, 1.0), Dual<double>(vds, 0.0), p);
        double gm = ids_gm.grad;
        double I_d = ids_gm.val;

        // Channel thermal noise
        double psd_thermal = NoiseModels::mosfet_channel_thermal_psd(gm, temp_K);
        if (psd_thermal > 0.0) {
            sources.push_back({m.drain, m.source, psd_thermal,
                              m.instanceName + "_channel_thermal"});
        }

        // Flicker noise
        if (m.KF > 0.0 && m.C_ox > 0.0) {
            double psd_flicker = NoiseModels::mosfet_flicker_psd(
                m.KF, I_d, m.AF, m.C_ox, m.w, m.l, freq);
            if (psd_flicker > 0.0) {
                sources.push_back({m.drain, m.source, psd_flicker,
                                  m.instanceName + "_flicker"});
            }
        }
    }

    return sources;
}

// ============================================================================
// NOISE ANALYSIS
// ============================================================================

/**
 * Perform noise analysis.
 *
 * Computes the output noise PSD and integrated RMS noise at a specified
 * output node, optionally referred to an input node.
 *
 * @param netlist       Circuit netlist
 * @param dcSolution    DC operating point voltages
 * @param outputNode    Output observation node (1-based)
 * @param inputNode     Input reference node (1-based), 0 = output-referred only
 * @param fStart        Start frequency [Hz]
 * @param fStop         Stop frequency [Hz]
 * @param numPoints     Number of frequency points (log-spaced)
 * @param temp_K        Temperature [K]
 */
inline NoiseResult solveNoise(
    const TensorNetlist& netlist,
    const std::vector<double>& dcSolution,
    int outputNode, int inputNode,
    double fStart, double fStop, int numPoints,
    double temp_K = 300.15)
{
    NoiseResult result;
    int n = netlist.numGlobalNodes;
    if (n == 0 || numPoints <= 0) return result;

    result.frequencies.resize(numPoints);
    result.output_noise_psd.resize(numPoints, 0.0);

    // Generate log-spaced frequencies
    double logStart = std::log10(std::max(fStart, 1e-3));
    double logStop = std::log10(std::max(fStop, fStart + 1.0));
    for (int i = 0; i < numPoints; ++i) {
        double logF = logStart + (logStop - logStart) * i / std::max(numPoints - 1, 1);
        result.frequencies[i] = std::pow(10.0, logF);
    }

    for (int fi = 0; fi < numPoints; ++fi) {
        double freq = result.frequencies[fi];
        double omega = 2.0 * M_PI * freq;

        // Build admittance matrix
        std::vector<std::vector<Complex>> Y_template;
        std::vector<Complex> rhs_unused;
        buildAdmittanceMatrix(netlist, dcSolution, omega, Y_template, rhs_unused);

        // Enumerate noise sources at this frequency
        auto noiseSources = computeNoiseSources(netlist, dcSolution, freq, temp_K);

        // For each noise source, solve Y*V = I_noise and accumulate
        // S_out(f) = sum_k |H_k(jw)|^2 * S_k(f)
        // where H_k is the transfer function from source k to output node
        double total_psd = 0.0;

        for (const auto& ns : noiseSources) {
            // Copy Y matrix (each solve modifies it)
            auto Y = Y_template;
            std::vector<Complex> rhs(n, Complex(0.0, 0.0));

            // Inject unit noise current at source location
            if (ns.nodePos > 0 && ns.nodePos <= n)
                rhs[ns.nodePos - 1] += Complex(1.0, 0.0);
            if (ns.nodeNeg > 0 && ns.nodeNeg <= n)
                rhs[ns.nodeNeg - 1] -= Complex(1.0, 0.0);

            auto V = solveComplexLU(Y, rhs);

            // Transfer function to output node
            Complex H_out(0.0, 0.0);
            if (outputNode > 0 && outputNode <= n) {
                H_out = V[outputNode - 1];
            }

            // Accumulate: |H|^2 * S_source
            total_psd += std::norm(H_out) * ns.psd;
        }

        result.output_noise_psd[fi] = total_psd;
    }

    // Integrate noise PSD using trapezoidal rule (log-spaced)
    double integrated = 0.0;
    for (int i = 1; i < numPoints; ++i) {
        double df = result.frequencies[i] - result.frequencies[i - 1];
        double avg_psd = 0.5 * (result.output_noise_psd[i] + result.output_noise_psd[i - 1]);
        integrated += avg_psd * df;
    }
    result.integrated_noise_vrms = std::sqrt(std::max(integrated, 0.0));

    return result;
}

// ============================================================================
// PHASE 2.10: SPARSE AC REFACTOR
// ============================================================================

/**
 * buildAdmittanceMatrixSparse — Phase 2.10
 * Builds the AC admittance matrix Y(jw) using the SoA ComplexMatrixConstructor.
 * If dcPattern is provided, uses O(N) assembleFast; otherwise full createCsr.
 *
 * @param netlist       Circuit netlist
 * @param dcSolution    DC operating point voltages
 * @param omega         Angular frequency (2*pi*f)
 * @param rhs_real      Output: RHS real part
 * @param rhs_imag      Output: RHS imaginary part
 * @param dcPattern     Optional cached DC sparsity pattern for fast assembly
 * @return              ComplexCsr admittance matrix
 */
inline ComplexCsr buildAdmittanceMatrixSparse(
    const TensorNetlist& netlist,
    const std::vector<double>& dcSolution,
    double omega,
    std::vector<double>& rhs_real,
    std::vector<double>& rhs_imag,
    const CachedCsrPattern* dcPattern = nullptr)
{
    int n = netlist.numGlobalNodes;
    ComplexMatrixConstructor builder;
    builder.setDimensions(n, n);
    rhs_real.assign(n, 0.0);
    rhs_imag.assign(n, 0.0);

    auto getV = [&](int idx) -> double {
        if (idx <= 0 || idx > (int)dcSolution.size()) return 0.0;
        return dcSolution[idx - 1];
    };

    // Helpers: add admittance stamp (pi-equivalent: +Y at (r,r), -Y at (r,c), etc.)
    auto addY_real = [&](int r, int c, double re) {
        if (r > 0 && r <= n && c > 0 && c <= n)
            builder.addEntry_real(r - 1, c - 1, re);
    };
    auto addY_imag = [&](int r, int c, double im) {
        if (r > 0 && r <= n && c > 0 && c <= n)
            builder.addEntry_imag(r - 1, c - 1, im);
    };
    auto addY = [&](int r, int c, double re, double im) {
        if (r > 0 && r <= n && c > 0 && c <= n)
            builder.addEntry(r - 1, c - 1, re, im);
    };
    auto stampY = [&](int p, int q, double re, double im) {
        addY(p, p,  re,  im); addY(p, q, -re, -im);
        addY(q, p, -re, -im); addY(q, q,  re,  im);
    };

    // 1. Resistors: Y = G (real)
    for (const auto& r : netlist.globalBlock.resistors)
        stampY(r.nodeTerminal1, r.nodeTerminal2, 1.0 / r.resistance_ohms, 0.0);

    // 2. Capacitors: Y = jwC (imaginary)
    for (const auto& c : netlist.globalBlock.capacitors)
        stampY(c.nodePlate1, c.nodePlate2, 0.0, omega * c.capacitance_farads);

    // 3. Inductors: Y = 1/(jwL) = -j/(wL) (imaginary, negative)
    for (const auto& l : netlist.globalBlock.inductors) {
        if (l.inductance_henries > 0.0 && omega > 0.0)
            stampY(l.nodeCoil1, l.nodeCoil2, 0.0, -1.0 / (omega * l.inductance_henries));
    }

    // 4. Voltage Sources: Norton equivalent (G_int = 1e3)
    for (const auto& vs : netlist.globalBlock.voltageSources)
        stampY(vs.nodePositive, vs.nodeNegative, 1.0 / 1e-3, 0.0);

    // 5. Diodes: linearized g_d at DC OP
    for (const auto& d : netlist.globalBlock.diodes) {
        DiodeParams<double> params = {d.saturationCurrent_I_S_A, d.emissionCoefficient_N, d.thermalVoltage_V_T_V};
        Dual<double> i_dual = diode_current_clamped(Dual<double>(getV(d.anode) - getV(d.cathode), 1.0), params);
        double g_d = std::max(i_dual.grad, 1e-15);
        stampY(d.anode, d.cathode, g_d, 0.0);
    }

    // 6. Schottky Diodes
    for (const auto& sd : netlist.globalBlock.schottkyDiodes) {
        DiodeParams<double> params = {sd.saturationCurrent_I_S_A, sd.emissionCoefficient_N, sd.thermalVoltage_V_T_V};
        Dual<double> i_dual = diode_current_clamped(Dual<double>(getV(sd.anode) - getV(sd.cathode), 1.0), params);
        double g_d = std::max(i_dual.grad, 1e-15);
        stampY(sd.anode, sd.cathode, g_d, 0.0);
    }

    // 7. MOSFETs: gm (VCCS) + gds (conductance)
    for (const auto& m : netlist.globalBlock.mosfets) {
        MosfetParams<double> p; p.Kp = 200e-6; p.Vth = 0.7; p.lambda = 0.02; p.W = m.w; p.L = m.l;
        double vgs = getV(m.gate) - getV(m.source), vds = getV(m.drain) - getV(m.source);
        double gm  = mosfet_ids(Dual<double>(vgs, 1.0), Dual<double>(vds, 0.0), p).grad;
        double gds = mosfet_ids(Dual<double>(vgs, 0.0), Dual<double>(vds, 1.0), p).grad;
        // gm: VCCS drain-source controlled by gate-source
        addY_real(m.drain,  m.gate,   gm); addY_real(m.drain,  m.source, -gm);
        addY_real(m.source, m.gate,  -gm); addY_real(m.source, m.source,  gm);
        // gds: conductance drain-source
        addY_real(m.drain,  m.drain,  gds); addY_real(m.drain,  m.source, -gds);
        addY_real(m.source, m.drain, -gds); addY_real(m.source, m.source,  gds);
    }

    // 8. BJTs: 3x3 linearized conductance matrix
    for (const auto& q : netlist.globalBlock.bjts) {
        BJTParams<double> params;
        params.Is = q.saturationCurrent_I_S_A; params.BetaF = q.betaF;
        params.BetaR = q.betaR; params.Vt = q.thermalVoltage_V_T_V;
        BJTCurrents<double> res = bjt_ebers_moll(getV(q.nodeCollector), getV(q.base), getV(q.emitter), q.isNPN, params);
        addY_real(q.nodeCollector, q.nodeCollector, res.g_cc); addY_real(q.nodeCollector, q.base, res.g_cb);
        addY_real(q.nodeCollector, q.emitter, res.g_ce);       addY_real(q.base, q.nodeCollector, res.g_bc);
        addY_real(q.base, q.base, res.g_bb);                   addY_real(q.base, q.emitter, res.g_be);
        addY_real(q.emitter, q.nodeCollector, res.g_ec);       addY_real(q.emitter, q.base, res.g_eb);
        addY_real(q.emitter, q.emitter, res.g_ee);
    }

    // 9. JFETs
    for (const auto& j : netlist.globalBlock.jfets) {
        double sign = j.isNChannel ? 1.0 : -1.0;
        JFETParams<double> p; p.Beta = j.beta; p.Vto = j.Vto; p.Lambda = j.lambda;
        double vgs = getV(j.gate) - getV(j.source), vds = getV(j.drain) - getV(j.source);
        double gm  = jfet_ids(Dual<double>(sign*vgs, 1.0), Dual<double>(sign*vds, 0.0), p).grad;
        double gds = jfet_ids(Dual<double>(sign*vgs, 0.0), Dual<double>(sign*vds, 1.0), p).grad;
        addY_real(j.drain,  j.gate,   sign*gm); addY_real(j.drain,  j.source, -sign*gm);
        addY_real(j.source, j.gate,  -sign*gm); addY_real(j.source, j.source,  sign*gm);
        addY_real(j.drain,  j.drain,  sign*gds); addY_real(j.drain,  j.source, -sign*gds);
        addY_real(j.source, j.drain, -sign*gds); addY_real(j.source, j.source,  sign*gds);
    }

    // 10. Zener Diodes
    for (const auto& z : netlist.globalBlock.zenerDiodes) {
        ZenerParams<double> zp;
        zp.BV = z.breakdownVoltage_V; zp.IBV = z.currentAtBreakdown_A;
        zp.Rs = z.seriesResistance_Rs_ohms; zp.N = z.emissionCoefficient_N;
        zp.Is = z.saturationCurrent_I_S_A; zp.Vt = z.thermalVoltage_V_T_V;
        double g_d = std::max(zener_current(Dual<double>(getV(z.anode)-getV(z.cathode), 1.0), zp).grad, 1e-15);
        stampY(z.anode, z.cathode, g_d, 0.0);
    }

    // GMIN diagonal conditioning
    for (int i = 0; i < n; ++i)
        builder.addEntry_real(i, i, 1e-12);

    return dcPattern ? builder.assembleFast(*dcPattern) : builder.createCsr();
}

/**
 * solveAC_Sparse — Phase 2.10
 * AC sweep using sparse ComplexCsr admittance matrix + solveComplexLU_Sparse.
 * Drop-in replacement for solveAC; same return type (ACResult).
 * Pass dcPattern for O(N) matrix assembly on each frequency point.
 */
inline ACResult solveAC_Sparse(
    const TensorNetlist& netlist,
    const std::vector<double>& dcSolution,
    double fStart, double fStop, int numPoints,
    int inputNode = 0, double inputAmplitude = 1.0,
    const CachedCsrPattern* dcPattern = nullptr)
{
    ACResult result;
    if (netlist.numGlobalNodes == 0 || numPoints <= 0) return result;

    int n = netlist.numGlobalNodes;
    result.frequencies.resize(numPoints);
    result.nodeVoltages.resize(numPoints);

    double logStart = std::log10(std::max(fStart, 1e-3));
    double logStop  = std::log10(std::max(fStop, fStart + 1.0));
    for (int i = 0; i < numPoints; ++i) {
        double logF = logStart + (logStop - logStart) * i / std::max(numPoints - 1, 1);
        result.frequencies[i] = std::pow(10.0, logF);
    }

    for (int fi = 0; fi < numPoints; ++fi) {
        double omega = 2.0 * M_PI * result.frequencies[fi];

        std::vector<double> rhs_real, rhs_imag;
        ComplexCsr Y = buildAdmittanceMatrixSparse(netlist, dcSolution, omega,
                                                    rhs_real, rhs_imag, dcPattern);

        // AC input excitation: large-G Norton equivalent at inputNode
        if (inputNode > 0 && inputNode <= n) {
            double G_large = 1e6;
            // Add G_large to (inputNode-1, inputNode-1) diagonal — rebuild needed
            // Simple approach: just add to rhs and diagonal via addEntry
            // Since Y is already built, we append and re-createCsr is not practical.
            // Instead: directly modify values_real at diagonal position.
            // Find diagonal in Y.row_pointer
            int row = inputNode - 1;
            for (int p = Y.row_pointer[row]; p < Y.row_pointer[row + 1]; ++p) {
                if (Y.col_indices[p] == row) {
                    Y.values_real[static_cast<size_t>(p)] += G_large;
                    break;
                }
            }
            rhs_real[inputNode - 1] += G_large * inputAmplitude;
        }

        ComplexSolverResult sol = solveComplexLU_Sparse(Y, rhs_real, rhs_imag);

        // Convert SoA to std::vector<Complex> for API compatibility
        result.nodeVoltages[fi].resize(n);
        for (int i = 0; i < n; ++i)
            result.nodeVoltages[fi][i] = Complex(sol.solution_real[i], sol.solution_imag[i]);
    }

    return result;
}

/**
 * solveNoise_Sparse — Phase 2.10
 * Noise analysis using sparse Y matrix. Pattern shared across noise sources.
 */
inline NoiseResult solveNoise_Sparse(
    const TensorNetlist& netlist,
    const std::vector<double>& dcSolution,
    int outputNode, int inputNode,
    double fStart, double fStop, int numPoints,
    double temp_K = 300.15,
    const CachedCsrPattern* dcPattern = nullptr)
{
    NoiseResult result;
    int n = netlist.numGlobalNodes;
    if (n == 0 || numPoints <= 0) return result;

    result.frequencies.resize(numPoints);
    result.output_noise_psd.resize(numPoints, 0.0);

    double logStart = std::log10(std::max(fStart, 1e-3));
    double logStop  = std::log10(std::max(fStop, fStart + 1.0));
    for (int i = 0; i < numPoints; ++i) {
        double logF = logStart + (logStop - logStart) * i / std::max(numPoints - 1, 1);
        result.frequencies[i] = std::pow(10.0, logF);
    }

    for (int fi = 0; fi < numPoints; ++fi) {
        double freq  = result.frequencies[fi];
        double omega = 2.0 * M_PI * freq;

        // Build Y template once per frequency
        std::vector<double> rhs_real_unused, rhs_imag_unused;
        ComplexCsr Y_template = buildAdmittanceMatrixSparse(
            netlist, dcSolution, omega, rhs_real_unused, rhs_imag_unused, dcPattern);

        auto noiseSources = computeNoiseSources(netlist, dcSolution, freq, temp_K);
        double total_psd = 0.0;

        for (const auto& ns : noiseSources) {
            // Copy template values (pattern is shared — only copy values)
            ComplexCsr Y = Y_template;  // shallow copy: all vectors copied

            std::vector<double> rhs_re(n, 0.0), rhs_im(n, 0.0);
            if (ns.nodePos > 0 && ns.nodePos <= n) rhs_re[ns.nodePos - 1] += 1.0;
            if (ns.nodeNeg > 0 && ns.nodeNeg <= n) rhs_re[ns.nodeNeg - 1] -= 1.0;

            ComplexSolverResult sol = solveComplexLU_Sparse(Y, rhs_re, rhs_im);

            Complex H_out(0.0, 0.0);
            if (outputNode > 0 && outputNode <= n) {
                H_out = Complex(sol.solution_real[outputNode - 1],
                                sol.solution_imag[outputNode - 1]);
            }
            total_psd += std::norm(H_out) * ns.psd;
        }

        result.output_noise_psd[fi] = total_psd;
    }

    // Integrate PSD (trapezoidal, log-spaced)
    double integrated = 0.0;
    for (int i = 1; i < numPoints; ++i) {
        double df = result.frequencies[i] - result.frequencies[i - 1];
        integrated += 0.5 * (result.output_noise_psd[i] + result.output_noise_psd[i - 1]) * df;
    }
    result.integrated_noise_vrms = std::sqrt(std::max(integrated, 0.0));

    return result;
}

} // namespace ACSolver
