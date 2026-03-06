#include "mpo_builder.h"
#include "../physics/device_physics.h"
#include "../autodiff/dual.h"
#include <algorithm>
#include <cmath>

// ============================================================================
// Helper: get voltage for a node (0-based index, ground=0 returns 0.0)
// ============================================================================
static double getVoltage(int node, const std::vector<double>& voltages) {
    if (node <= 0) return 0.0;
    size_t idx = static_cast<size_t>(node - 1);
    return (idx < voltages.size()) ? voltages[idx] : 0.0;
}

// ============================================================================
// Helper: collect non-ground terminal nodes (sorted, deduplicated)
// ============================================================================
static std::vector<uint32_t> collectTerminals(const std::vector<int>& nodes) {
    std::vector<uint32_t> terminals;
    for (int n : nodes) {
        if (n > 0)
            terminals.push_back(static_cast<uint32_t>(n));
    }
    std::sort(terminals.begin(), terminals.end());
    terminals.erase(std::unique(terminals.begin(), terminals.end()), terminals.end());
    return terminals;
}

// ============================================================================
// Helper: find index of node in sorted terminal list
// ============================================================================
static int terminalIndex(uint32_t node, const std::vector<uint32_t>& terminals) {
    auto it = std::lower_bound(terminals.begin(), terminals.end(), node);
    if (it != terminals.end() && *it == node)
        return static_cast<int>(it - terminals.begin());
    return -1;
}

// ============================================================================
// Helper: stamp a conductance into the local k×k matrix
// ============================================================================
static void localStamp(std::vector<double>& mat, uint32_t k,
                       int row, int col, double val) {
    if (row >= 0 && col >= 0 && row < static_cast<int>(k) && col < static_cast<int>(k))
        mat[row * k + col] += val;
}

// ============================================================================
// Helper: stamp a value into the local k-element RHS vector
// ============================================================================
static void localStampRHS(std::vector<double>& rhs, uint32_t k,
                          int idx, double val) {
    if (idx >= 0 && idx < static_cast<int>(k))
        rhs[idx] += val;
}

// ============================================================================
// MPOBuilder::buildMPOs
// ============================================================================

std::vector<DeviceMPO> MPOBuilder::buildMPOs(
    const TensorizedBlock& tensors,
    const std::vector<double>& voltages,
    double temp_K)
{
    std::vector<DeviceMPO> mpos;

    uint32_t offset = 0;

    // ── Diodes ────────────────────────────────────────────────────────────
    for (size_t i = 0; i < tensors.diodes.size(); ++i) {
        DeviceMPO mpo;
        mpo.deviceType = FactorDeviceType::Diode;
        mpo.deviceIndex = offset + static_cast<uint32_t>(i);

        int nA = tensors.diodes.node_a[i];
        int nC = tensors.diodes.node_c[i];
        mpo.terminalNodes = collectTerminals({nA, nC});
        mpo.rank = static_cast<uint32_t>(mpo.terminalNodes.size());

        if (mpo.rank == 0) { mpos.push_back(std::move(mpo)); continue; }

        mpo.localMatrix.assign(mpo.rank * mpo.rank, 0.0);
        mpo.localRHS.assign(mpo.rank, 0.0);

        // Evaluate linearized conductance via AD
        double vA = getVoltage(nA, voltages);
        double vC = getVoltage(nC, voltages);
        double v_diff = vA - vC;

        DiodeParams<double> params = {
            tensors.diodes.Is[i],
            tensors.diodes.N[i],
            tensors.diodes.Vt[i]
        };
        params.Vt *= (temp_K / 300.15);

        Dual<double> v_d(v_diff, 1.0);
        Dual<double> i_d = diode_current_clamped(v_d, params);
        double g_d = i_d.grad;
        double i_d0 = i_d.val;

        // Stamp [[g_d, -g_d], [-g_d, g_d]] into local matrix
        int iA = (nA > 0) ? terminalIndex(static_cast<uint32_t>(nA), mpo.terminalNodes) : -1;
        int iC = (nC > 0) ? terminalIndex(static_cast<uint32_t>(nC), mpo.terminalNodes) : -1;

        localStamp(mpo.localMatrix, mpo.rank, iA, iA,  g_d);
        localStamp(mpo.localMatrix, mpo.rank, iA, iC, -g_d);
        localStamp(mpo.localMatrix, mpo.rank, iC, iA, -g_d);
        localStamp(mpo.localMatrix, mpo.rank, iC, iC,  g_d);

        // Norton RHS: I_eq = i_d0 - g_d * v_diff
        // Current I_eq flows anode→cathode; rhs[A] -= I_eq, rhs[C] += I_eq
        double I_eq = i_d0 - g_d * v_diff;
        localStampRHS(mpo.localRHS, mpo.rank, iA, -I_eq);
        localStampRHS(mpo.localRHS, mpo.rank, iC,  I_eq);

        mpos.push_back(std::move(mpo));
    }
    offset += static_cast<uint32_t>(tensors.diodes.size());

    // ── MOSFETs ──────────────────────────────────────────────────────────
    for (size_t i = 0; i < tensors.mosfets.size(); ++i) {
        DeviceMPO mpo;
        mpo.deviceType = FactorDeviceType::Mosfet;
        mpo.deviceIndex = offset + static_cast<uint32_t>(i);

        int nD = tensors.mosfets.drains[i];
        int nG = tensors.mosfets.gates[i];
        int nS = tensors.mosfets.sources[i];
        int nB = tensors.mosfets.bodies[i];
        mpo.terminalNodes = collectTerminals({nD, nG, nS, nB});
        mpo.rank = static_cast<uint32_t>(mpo.terminalNodes.size());

        if (mpo.rank == 0) { mpos.push_back(std::move(mpo)); continue; }

        mpo.localMatrix.assign(mpo.rank * mpo.rank, 0.0);
        mpo.localRHS.assign(mpo.rank, 0.0);

        double vD = getVoltage(nD, voltages);
        double vG = getVoltage(nG, voltages);
        double vS = getVoltage(nS, voltages);

        double vgs = vG - vS;
        double vds = vD - vS;

        MosfetParams<double> p;
        p.Kp = tensors.mosfets.Kp[i];
        p.Vth = tensors.mosfets.Vth[i];
        p.lambda = tensors.mosfets.lambda[i];
        p.W = tensors.mosfets.W[i];
        p.L = tensors.mosfets.L[i];

        // gm = dId/dVgs
        Dual<double> ids_vgs = mosfet_ids(Dual<double>(vgs, 1.0), Dual<double>(vds, 0.0), p);
        double gm = ids_vgs.grad;
        double ids_val = ids_vgs.val;  // drain current at operating point

        // gds = dId/dVds
        Dual<double> ids_vds = mosfet_ids(Dual<double>(vgs, 0.0), Dual<double>(vds, 1.0), p);
        double gds = ids_vds.grad;

        double gmb = 0.0; // Body effect placeholder

        // MOSFET stamp pattern: I_ds enters drain, leaves source
        // Controlled by vgs (gm), vbs (gmb), vds (gds)
        int iD = (nD > 0) ? terminalIndex(static_cast<uint32_t>(nD), mpo.terminalNodes) : -1;
        int iG = (nG > 0) ? terminalIndex(static_cast<uint32_t>(nG), mpo.terminalNodes) : -1;
        int iS = (nS > 0) ? terminalIndex(static_cast<uint32_t>(nS), mpo.terminalNodes) : -1;
        int iB = (nB > 0) ? terminalIndex(static_cast<uint32_t>(nB), mpo.terminalNodes) : -1;

        // gm: VGS control
        localStamp(mpo.localMatrix, mpo.rank, iD, iG,  gm);
        localStamp(mpo.localMatrix, mpo.rank, iD, iS, -gm);
        localStamp(mpo.localMatrix, mpo.rank, iS, iG, -gm);
        localStamp(mpo.localMatrix, mpo.rank, iS, iS,  gm);

        // gmb: VBS control
        localStamp(mpo.localMatrix, mpo.rank, iD, iB,  gmb);
        localStamp(mpo.localMatrix, mpo.rank, iD, iS, -gmb);
        localStamp(mpo.localMatrix, mpo.rank, iS, iB, -gmb);
        localStamp(mpo.localMatrix, mpo.rank, iS, iS,  gmb);

        // gds: VDS control
        localStamp(mpo.localMatrix, mpo.rank, iD, iD,  gds);
        localStamp(mpo.localMatrix, mpo.rank, iD, iS, -gds);
        localStamp(mpo.localMatrix, mpo.rank, iS, iD, -gds);
        localStamp(mpo.localMatrix, mpo.rank, iS, iS,  gds);

        // Norton RHS: I_eq = ids - gm*vgs - gds*vds (- gmb*vbs)
        // I_ds enters drain, leaves source
        double I_eq = ids_val - gm * vgs - gds * vds;
        localStampRHS(mpo.localRHS, mpo.rank, iD, -I_eq);
        localStampRHS(mpo.localRHS, mpo.rank, iS,  I_eq);

        mpos.push_back(std::move(mpo));
    }
    offset += static_cast<uint32_t>(tensors.mosfets.size());

    // ── BJTs ─────────────────────────────────────────────────────────────
    for (size_t i = 0; i < tensors.bjts.size(); ++i) {
        DeviceMPO mpo;
        mpo.deviceType = FactorDeviceType::BJT;
        mpo.deviceIndex = offset + static_cast<uint32_t>(i);

        int nC = tensors.bjts.collectors[i];
        int nB = tensors.bjts.bases[i];
        int nE = tensors.bjts.emitters[i];
        mpo.terminalNodes = collectTerminals({nC, nB, nE});
        mpo.rank = static_cast<uint32_t>(mpo.terminalNodes.size());

        if (mpo.rank == 0) { mpos.push_back(std::move(mpo)); continue; }

        mpo.localMatrix.assign(mpo.rank * mpo.rank, 0.0);
        mpo.localRHS.assign(mpo.rank, 0.0);

        double vC = getVoltage(nC, voltages);
        double vB = getVoltage(nB, voltages);
        double vE = getVoltage(nE, voltages);

        BJTParams<double> p;
        p.Is = tensors.bjts.Is[i];
        p.BetaF = tensors.bjts.BetaF[i];
        p.BetaR = tensors.bjts.BetaR[i];
        p.Vt = tensors.bjts.Vt[i];

        double temp_scale = temp_K / 300.15;
        p.Vt *= temp_scale;
        p.Is *= std::pow(temp_scale, 3.0) * exp(1.11 / (tensors.bjts.Vt[i]) - 1.11 / p.Vt);

        bool isNPN = tensors.bjts.isNPN[i];
        BJTCurrents<double> res = bjt_ebers_moll(vC, vB, vE, isNPN, p);

        int iC = (nC > 0) ? terminalIndex(static_cast<uint32_t>(nC), mpo.terminalNodes) : -1;
        int iB = (nB > 0) ? terminalIndex(static_cast<uint32_t>(nB), mpo.terminalNodes) : -1;
        int iE = (nE > 0) ? terminalIndex(static_cast<uint32_t>(nE), mpo.terminalNodes) : -1;

        // Stamp the 3×3 conductance matrix
        localStamp(mpo.localMatrix, mpo.rank, iC, iC, res.g_cc);
        localStamp(mpo.localMatrix, mpo.rank, iC, iB, res.g_cb);
        localStamp(mpo.localMatrix, mpo.rank, iC, iE, res.g_ce);
        localStamp(mpo.localMatrix, mpo.rank, iB, iC, res.g_bc);
        localStamp(mpo.localMatrix, mpo.rank, iB, iB, res.g_bb);
        localStamp(mpo.localMatrix, mpo.rank, iB, iE, res.g_be);
        localStamp(mpo.localMatrix, mpo.rank, iE, iC, res.g_ec);
        localStamp(mpo.localMatrix, mpo.rank, iE, iB, res.g_eb);
        localStamp(mpo.localMatrix, mpo.rank, iE, iE, res.g_ee);

        // Norton RHS: I_eq = i_vec - J * v_vec
        // I_eq[C] = Ic - (g_cc*vC + g_cb*vB + g_ce*vE)
        // I_eq[B] = Ib - (g_bc*vC + g_bb*vB + g_be*vE)
        // I_eq[E] = Ie - (g_ec*vC + g_eb*vB + g_ee*vE)
        double I_eq_c = res.Ic - (res.g_cc * vC + res.g_cb * vB + res.g_ce * vE);
        double I_eq_b = res.Ib - (res.g_bc * vC + res.g_bb * vB + res.g_be * vE);
        double I_eq_e = res.Ie - (res.g_ec * vC + res.g_eb * vB + res.g_ee * vE);
        localStampRHS(mpo.localRHS, mpo.rank, iC, -I_eq_c);
        localStampRHS(mpo.localRHS, mpo.rank, iB, -I_eq_b);
        localStampRHS(mpo.localRHS, mpo.rank, iE, -I_eq_e);

        mpos.push_back(std::move(mpo));
    }
    offset += static_cast<uint32_t>(tensors.bjts.size());

    // ── Resistors ────────────────────────────────────────────────────────
    for (size_t i = 0; i < tensors.resistors.size(); ++i) {
        DeviceMPO mpo;
        mpo.deviceType = FactorDeviceType::Resistor;
        mpo.deviceIndex = offset + static_cast<uint32_t>(i);

        int n1 = tensors.resistors.nodes1[i];
        int n2 = tensors.resistors.nodes2[i];
        mpo.terminalNodes = collectTerminals({n1, n2});
        mpo.rank = static_cast<uint32_t>(mpo.terminalNodes.size());

        if (mpo.rank == 0) { mpos.push_back(std::move(mpo)); continue; }

        mpo.localMatrix.assign(mpo.rank * mpo.rank, 0.0);
        mpo.localRHS.assign(mpo.rank, 0.0);  // Linear: no Norton current

        double G = 1.0 / tensors.resistors.R[i];

        int i1 = (n1 > 0) ? terminalIndex(static_cast<uint32_t>(n1), mpo.terminalNodes) : -1;
        int i2 = (n2 > 0) ? terminalIndex(static_cast<uint32_t>(n2), mpo.terminalNodes) : -1;

        localStamp(mpo.localMatrix, mpo.rank, i1, i1,  G);
        localStamp(mpo.localMatrix, mpo.rank, i1, i2, -G);
        localStamp(mpo.localMatrix, mpo.rank, i2, i1, -G);
        localStamp(mpo.localMatrix, mpo.rank, i2, i2,  G);

        mpos.push_back(std::move(mpo));
    }
    offset += static_cast<uint32_t>(tensors.resistors.size());

    // ── Capacitors (DC: open circuit, stamp zero conductance) ─────────
    for (size_t i = 0; i < tensors.capacitors.size(); ++i) {
        DeviceMPO mpo;
        mpo.deviceType = FactorDeviceType::Capacitor;
        mpo.deviceIndex = offset + static_cast<uint32_t>(i);

        int p1 = tensors.capacitors.plates1[i];
        int p2 = tensors.capacitors.plates2[i];
        mpo.terminalNodes = collectTerminals({p1, p2});
        mpo.rank = static_cast<uint32_t>(mpo.terminalNodes.size());

        // DC analysis: capacitor is open circuit (zero conductance MPO)
        mpo.localMatrix.assign(mpo.rank * mpo.rank, 0.0);
        mpo.localRHS.assign(mpo.rank, 0.0);
        mpos.push_back(std::move(mpo));
    }
    offset += static_cast<uint32_t>(tensors.capacitors.size());

    // ── Inductors (DC: short circuit via large conductance) ───────────
    // Note: inductors introduce branch-current MNA variables, but for the
    // MPO representation we only track voltage-node contributions.
    // The branch-current coupling is handled via the contraction tree.
    for (size_t i = 0; i < tensors.inductors.size(); ++i) {
        DeviceMPO mpo;
        mpo.deviceType = FactorDeviceType::Inductor;
        mpo.deviceIndex = offset + static_cast<uint32_t>(i);

        int c1 = tensors.inductors.coil1[i];
        int c2 = tensors.inductors.coil2[i];
        mpo.terminalNodes = collectTerminals({c1, c2});
        mpo.rank = static_cast<uint32_t>(mpo.terminalNodes.size());
        mpo.localMatrix.assign(mpo.rank * mpo.rank, 0.0);
        mpo.localRHS.assign(mpo.rank, 0.0);  // Linear: no Norton current

        // DC: inductor is a short (very large conductance)
        double G = 1e6;
        int i1 = (c1 > 0) ? terminalIndex(static_cast<uint32_t>(c1), mpo.terminalNodes) : -1;
        int i2 = (c2 > 0) ? terminalIndex(static_cast<uint32_t>(c2), mpo.terminalNodes) : -1;

        localStamp(mpo.localMatrix, mpo.rank, i1, i1,  G);
        localStamp(mpo.localMatrix, mpo.rank, i1, i2, -G);
        localStamp(mpo.localMatrix, mpo.rank, i2, i1, -G);
        localStamp(mpo.localMatrix, mpo.rank, i2, i2,  G);

        mpos.push_back(std::move(mpo));
    }
    offset += static_cast<uint32_t>(tensors.inductors.size());

    // ── Voltage Sources (Norton equivalent) ──────────────────────────
    for (size_t i = 0; i < tensors.voltageSources.size(); ++i) {
        DeviceMPO mpo;
        mpo.deviceType = FactorDeviceType::VoltageSource;
        mpo.deviceIndex = offset + static_cast<uint32_t>(i);

        int np = tensors.voltageSources.nodesPos[i];
        int nn = tensors.voltageSources.nodesNeg[i];
        mpo.terminalNodes = collectTerminals({np, nn});
        mpo.rank = static_cast<uint32_t>(mpo.terminalNodes.size());
        mpo.localMatrix.assign(mpo.rank * mpo.rank, 0.0);
        mpo.localRHS.assign(mpo.rank, 0.0);

        // Norton equivalent: G_int = 1/r_int, r_int = 1e-3
        double G = 1e3;
        double V = tensors.voltageSources.voltages[i];
        int ip = (np > 0) ? terminalIndex(static_cast<uint32_t>(np), mpo.terminalNodes) : -1;
        int in = (nn > 0) ? terminalIndex(static_cast<uint32_t>(nn), mpo.terminalNodes) : -1;

        localStamp(mpo.localMatrix, mpo.rank, ip, ip,  G);
        localStamp(mpo.localMatrix, mpo.rank, ip, in, -G);
        localStamp(mpo.localMatrix, mpo.rank, in, ip, -G);
        localStamp(mpo.localMatrix, mpo.rank, in, in,  G);

        // Norton current: I_n = G * V, enters positive terminal
        localStampRHS(mpo.localRHS, mpo.rank, ip,  G * V);
        localStampRHS(mpo.localRHS, mpo.rank, in, -G * V);

        mpos.push_back(std::move(mpo));
    }
    offset += static_cast<uint32_t>(tensors.voltageSources.size());

    // ── Current Sources (Norton: no conductance stamp, only RHS) ─────
    for (size_t i = 0; i < tensors.currentSources.size(); ++i) {
        DeviceMPO mpo;
        mpo.deviceType = FactorDeviceType::CurrentSource;
        mpo.deviceIndex = offset + static_cast<uint32_t>(i);

        int np = tensors.currentSources.nodesPos[i];
        int nn = tensors.currentSources.nodesNeg[i];
        mpo.terminalNodes = collectTerminals({np, nn});
        mpo.rank = static_cast<uint32_t>(mpo.terminalNodes.size());

        // Current sources have zero conductance contribution to Jacobian
        mpo.localMatrix.assign(mpo.rank * mpo.rank, 0.0);
        mpo.localRHS.assign(mpo.rank, 0.0);

        // Current I enters positive terminal, leaves negative terminal
        double I_s = tensors.currentSources.currents[i];
        int ip = (np > 0) ? terminalIndex(static_cast<uint32_t>(np), mpo.terminalNodes) : -1;
        int in = (nn > 0) ? terminalIndex(static_cast<uint32_t>(nn), mpo.terminalNodes) : -1;
        localStampRHS(mpo.localRHS, mpo.rank, ip,  I_s);
        localStampRHS(mpo.localRHS, mpo.rank, in, -I_s);

        mpos.push_back(std::move(mpo));
    }

    return mpos;
}
