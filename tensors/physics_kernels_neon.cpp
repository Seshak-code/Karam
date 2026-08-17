// ARM NEON Physics Kernels — 2-wide double (128-bit)
// Layout matches physics_kernels_avx2.cpp conventions.
// AArch64 NEON has native float64x2_t support.

#include "../tensors/physics_tensors.h"
#include <vector>
#include <cmath>
#include <algorithm>

#if defined(__aarch64__)
#include <arm_neon.h>

// ── Helpers ──

// Lane-wise exp() fallback (no NEON exp intrinsic for f64)
inline float64x2_t vexp_f64(float64x2_t v) {
    alignas(16) double tmp[2];
    vst1q_f64(tmp, v);
    tmp[0] = std::exp(tmp[0]);
    tmp[1] = std::exp(tmp[1]);
    return vld1q_f64(tmp);
}

// Bitwise abs for f64 (clear sign bit)
inline float64x2_t vabs_f64(float64x2_t v) {
    return vabsq_f64(v);
}

// ============================================================================
// DIODE KERNEL — 2 devices per iteration
// ============================================================================

void batchDiodePhysics_neon(DiodeTensor& tensor, const std::vector<double>& voltages) {
    const size_t n = tensor.size();

    const float64x2_t vc_1  = vdupq_n_f64(1.0);
    const float64x2_t vc_30 = vdupq_n_f64(30.0);
    const float64x2_t vc_0  = vdupq_n_f64(0.0);

    const double* __restrict__ v_ptr = voltages.data();
    const int v_size = (int)voltages.size();

    size_t i = 0;
    // --- SIMD Loop: 2 devices per iteration ---
    for (; i + 1 < n; i += 2) {

        // Gather terminal voltages (manual — no NEON gather)
        alignas(16) double buf_a[2];
        for (int k = 0; k < 2; ++k) {
            int idx = tensor.node_a[i + k];
            buf_a[k] = (idx > 0 && idx <= v_size) ? v_ptr[idx - 1] : 0.0;
        }
        float64x2_t v_v_a = vld1q_f64(buf_a);

        alignas(16) double buf_c[2];
        for (int k = 0; k < 2; ++k) {
            int idx = tensor.node_c[i + k];
            buf_c[k] = (idx > 0 && idx <= v_size) ? v_ptr[idx - 1] : 0.0;
        }
        float64x2_t v_v_c = vld1q_f64(buf_c);

        // Load parameters
        float64x2_t v_Is   = vld1q_f64(&tensor.Is[i]);
        float64x2_t v_N    = vld1q_f64(&tensor.N[i]);
        float64x2_t v_Vt   = vld1q_f64(&tensor.Vt[i]);
        float64x2_t v_GMIN = vld1q_f64(&tensor.GMIN[i]);

        // v_d = v_a - v_c
        float64x2_t v_v_d = vsubq_f64(v_v_a, v_v_c);
        vst1q_f64(&tensor.v_d[i], v_v_d);

        // n_vt = N * Vt
        float64x2_t v_n_vt = vmulq_f64(v_N, v_Vt);

        // arg = v_d / n_vt
        float64x2_t v_arg = vdivq_f64(v_v_d, v_n_vt);

        // arg_clamped = min(arg, 30.0)
        float64x2_t v_arg_clamped = vminq_f64(v_arg, vc_30);

        // exp_val = exp(arg_clamped)
        float64x2_t v_exp_val = vexp_f64(v_arg_clamped);

        // i_base = Is * (exp_val - 1.0)
        float64x2_t v_i_base = vmulq_f64(v_Is, vsubq_f64(v_exp_val, vc_1));

        // g_base = (Is / n_vt) * exp_val
        float64x2_t v_g_base = vmulq_f64(vdivq_f64(v_Is, v_n_vt), v_exp_val);

        // delta = max(0.0, v_d - 30.0 * n_vt)
        float64x2_t v_vcrit = vmulq_f64(vc_30, v_n_vt);
        float64x2_t v_delta = vmaxq_f64(vc_0, vsubq_f64(v_v_d, v_vcrit));

        // i_d = i_base + delta * g_base
        float64x2_t v_i_d = vaddq_f64(v_i_base, vmulq_f64(v_delta, v_g_base));
        vst1q_f64(&tensor.i_d[i], v_i_d);

        // g_d = max(GMIN, g_base)
        float64x2_t v_g_d = vmaxq_f64(v_GMIN, v_g_base);
        vst1q_f64(&tensor.g_d[i], v_g_d);
    }

    // --- Scalar Tail ---
    for (; i < n; ++i) {
        int n_a = tensor.node_a[i];
        double v_a = (n_a > 0 && n_a <= v_size) ? v_ptr[n_a - 1] : 0.0;
        int n_c = tensor.node_c[i];
        double v_c = (n_c > 0 && n_c <= v_size) ? v_ptr[n_c - 1] : 0.0;

        tensor.v_d[i] = v_a - v_c;
        double n_vt = tensor.N[i] * tensor.Vt[i];
        double arg = tensor.v_d[i] / n_vt;
        double arg_clamped = std::min(arg, 30.0);
        double exp_val = std::exp(arg_clamped);
        double i_base = tensor.Is[i] * (exp_val - 1.0);
        double g_base = (tensor.Is[i] / n_vt) * exp_val;
        double delta = std::max(0.0, tensor.v_d[i] - 30.0 * n_vt);
        tensor.i_d[i] = i_base + delta * g_base;
        tensor.g_d[i] = std::max(tensor.GMIN[i], g_base);
    }
}

// ============================================================================
// MOSFET KERNEL — Shichman-Hodges Level 1, 2 devices per iteration
// ============================================================================

void batchMosfetPhysics_neon(MosfetTensor& tensor, const std::vector<double>& voltages) {
    const size_t n = tensor.size();
    const float64x2_t v_GMIN = vdupq_n_f64(1e-12);
    const float64x2_t v_half = vdupq_n_f64(0.5);
    const float64x2_t v_one  = vdupq_n_f64(1.0);
    const float64x2_t v_zero = vdupq_n_f64(0.0);

    const double* __restrict__ v_ptr = voltages.data();
    const int v_size = (int)voltages.size();

    size_t i = 0;
    // --- SIMD Loop: 2 devices per iteration ---
    for (; i + 1 < n; i += 2) {

        // Gather terminal voltages
        alignas(16) double buf_d[2], buf_g[2], buf_s[2];
        for (int k = 0; k < 2; ++k) {
            int idx_d = tensor.drains[i + k];
            buf_d[k] = (idx_d > 0 && idx_d <= v_size) ? v_ptr[idx_d - 1] : 0.0;
            int idx_g = tensor.gates[i + k];
            buf_g[k] = (idx_g > 0 && idx_g <= v_size) ? v_ptr[idx_g - 1] : 0.0;
            int idx_s = tensor.sources[i + k];
            buf_s[k] = (idx_s > 0 && idx_s <= v_size) ? v_ptr[idx_s - 1] : 0.0;
        }

        float64x2_t v_v_d = vld1q_f64(buf_d);
        float64x2_t v_v_g = vld1q_f64(buf_g);
        float64x2_t v_v_s = vld1q_f64(buf_s);

        // Load parameters
        float64x2_t v_Kp     = vld1q_f64(&tensor.Kp[i]);
        float64x2_t v_W      = vld1q_f64(&tensor.W[i]);
        float64x2_t v_L      = vld1q_f64(&tensor.L[i]);
        float64x2_t v_Vth    = vld1q_f64(&tensor.Vth[i]);
        float64x2_t v_lambda = vld1q_f64(&tensor.lambda[i]);

        // PMOS sign vector: +1 NMOS, -1 PMOS (was ignored before - critical fix)
        alignas(16) double buf_sign[2];
        for (int k = 0; k < 2; ++k)
            buf_sign[k] = tensor.isPMOS[i + k] ? -1.0 : 1.0;
        float64x2_t v_sign = vld1q_f64(buf_sign);

        // vgs = v_g - v_s, vds = v_d - v_s (raw, stored for stamping)
        float64x2_t vgs_val = vsubq_f64(v_v_g, v_v_s);
        float64x2_t vds_val = vsubq_f64(v_v_d, v_v_s);
        vst1q_f64(&tensor.vgs[i], vgs_val);
        vst1q_f64(&tensor.vds[i], vds_val);

        // NMOS-equivalent voltages: vgs' = sign*vgs, vds' = sign*vds, vth' = sign*vth
        float64x2_t vgs_eq = vmulq_f64(v_sign, vgs_val);
        float64x2_t vds_eq = vmulq_f64(v_sign, vds_val);
        float64x2_t vth_eq = vmulq_f64(v_sign, v_Vth);

        // beta = Kp * (W / L)
        float64x2_t v_beta = vmulq_f64(v_Kp, vdivq_f64(v_W, v_L));

        // vov = vgs' - vth'
        float64x2_t v_vov = vsubq_f64(vgs_eq, vth_eq);

        // Region select (NMOS-equivalent): cutoff / linear / saturation
        float64x2_t v_half_beta = vmulq_f64(v_half, v_beta);
        float64x2_t v_vov2 = vmulq_f64(v_vov, v_vov);
        float64x2_t v_clm = vaddq_f64(v_one, vmulq_f64(v_lambda, vds_eq));

        // Saturation branch
        float64x2_t v_ids_sat = vmulq_f64(vmulq_f64(v_half_beta, v_vov2), v_clm);
        float64x2_t v_gm_sat  = vmulq_f64(vmulq_f64(v_beta, v_vov), v_clm);
        float64x2_t v_gds_sat = vmulq_f64(vmulq_f64(v_half_beta, v_vov2), v_lambda);

        // Linear branch: ids = beta*(vov*vds' - 0.5*vds'^2)
        float64x2_t v_ids_lin = vmulq_f64(v_beta, vsubq_f64(vmulq_f64(v_vov, vds_eq), vmulq_f64(v_half, vmulq_f64(vds_eq, vds_eq))));
        float64x2_t v_gm_lin  = vmulq_f64(v_beta, vds_eq);
        float64x2_t v_gds_lin = vmulq_f64(v_beta, vsubq_f64(v_vov, vds_eq));

        // Per-lane region select
        uint64x2_t m_lin = vcltq_f64(vds_eq, v_vov);   // linear if vds' < vov
        uint64x2_t m_cut = vcleq_f64(v_vov, v_zero);   // cutoff if vov <= 0
        float64x2_t v_ids_sel = vbslq_f64(m_lin, v_ids_lin, v_ids_sat);
        float64x2_t v_gm_sel  = vbslq_f64(m_lin, v_gm_lin, v_gm_sat);
        float64x2_t v_gds_sel = vbslq_f64(m_lin, v_gds_lin, v_gds_sat);
        float64x2_t v_ids = vbslq_f64(m_cut, v_zero, v_ids_sel);
        float64x2_t v_gm  = vbslq_f64(m_cut, v_zero, v_gm_sel);
        float64x2_t v_gds = vbslq_f64(m_cut, v_GMIN, v_gds_sel);

        // Restore PMOS current sign: ids = sign * |ids|
        v_ids = vmulq_f64(v_sign, v_ids);

        // Floor conductances at GMIN
        v_gm  = vmaxq_f64(v_GMIN, v_gm);
        v_gds = vmaxq_f64(v_GMIN, v_gds);

        vst1q_f64(&tensor.ids[i], v_ids);
        vst1q_f64(&tensor.gm[i], v_gm);
        vst1q_f64(&tensor.gds[i], v_gds);
    }

    // --- Scalar Tail ---
    for (; i < n; ++i) {
        int n_d = tensor.drains[i];
        double v_d = (n_d > 0 && n_d <= v_size) ? v_ptr[n_d - 1] : 0.0;
        int n_g = tensor.gates[i];
        double v_g = (n_g > 0 && n_g <= v_size) ? v_ptr[n_g - 1] : 0.0;
        int n_s = tensor.sources[i];
        double v_s = (n_s > 0 && n_s <= v_size) ? v_ptr[n_s - 1] : 0.0;

        tensor.vgs[i] = v_g - v_s;
        tensor.vds[i] = v_d - v_s;

        // PMOS sign-flip (matches generic kernel)
        double sign = tensor.isPMOS[i] ? -1.0 : 1.0;
        double vgs_eq = sign * tensor.vgs[i];
        double vds_eq = sign * tensor.vds[i];
        double vth_eq = sign * tensor.Vth[i];

        double beta = tensor.Kp[i] * (tensor.W[i] / tensor.L[i]);
        double vov = vgs_eq - vth_eq;
        if (vov <= 0.0) {
            tensor.ids[i] = 0.0;
            tensor.gm[i] = 0.0;
            tensor.gds[i] = 1e-12;
        } else if (vds_eq < vov) {
            tensor.ids[i] = sign * beta * (vov * vds_eq - 0.5 * vds_eq * vds_eq);
            tensor.gm[i] = beta * vds_eq;
            tensor.gds[i] = beta * (vov - vds_eq);
        } else {
            double lam = tensor.lambda[i];
            tensor.ids[i] = sign * 0.5 * beta * vov * vov * (1.0 + lam * vds_eq);
            tensor.gm[i] = beta * vov * (1.0 + lam * vds_eq);
            tensor.gds[i] = 0.5 * beta * vov * vov * lam;
        }
        if (tensor.gm[i] < 1e-12) tensor.gm[i] = 1e-12;
        if (tensor.gds[i] < 1e-12) tensor.gds[i] = 1e-12;
    }
}

// ============================================================================
// BJT KERNEL — Ebers-Moll, 2 devices per iteration
// ============================================================================

void batchBJTPhysics_neon(BJTTensor& tensor, const std::vector<double>& voltages) {
    const size_t n = tensor.size();
    const double GMIN = 1e-12;
    const float64x2_t v_GMIN = vdupq_n_f64(GMIN);
    const float64x2_t v_zero = vdupq_n_f64(0.0);
    const float64x2_t v_one  = vdupq_n_f64(1.0);
    const float64x2_t v_30   = vdupq_n_f64(30.0);

    const double* __restrict__ v_ptr = voltages.data();
    const int v_size = (int)voltages.size();

    size_t i = 0;
    // --- SIMD Loop: 2 devices per iteration ---
    for (; i + 1 < n; i += 2) {

        // Gather terminal voltages
        alignas(16) double buf_c[2], buf_b[2], buf_e[2];
        for (int k = 0; k < 2; ++k) {
            int idx_c = tensor.collectors[i + k];
            buf_c[k] = (idx_c > 0 && idx_c <= v_size) ? v_ptr[idx_c - 1] : 0.0;
            int idx_b = tensor.bases[i + k];
            buf_b[k] = (idx_b > 0 && idx_b <= v_size) ? v_ptr[idx_b - 1] : 0.0;
            int idx_e = tensor.emitters[i + k];
            buf_e[k] = (idx_e > 0 && idx_e <= v_size) ? v_ptr[idx_e - 1] : 0.0;
        }

        float64x2_t v_v_c = vld1q_f64(buf_c);
        float64x2_t v_v_b = vld1q_f64(buf_b);
        float64x2_t v_v_e = vld1q_f64(buf_e);

        // Load NPN sign: +1 for NPN, -1 for PNP
        alignas(16) double sign_buf[2];
        sign_buf[0] = tensor.isNPN[i]     ? 1.0 : -1.0;
        sign_buf[1] = tensor.isNPN[i + 1] ? 1.0 : -1.0;
        float64x2_t v_sign = vld1q_f64(sign_buf);

        // v_be = sign * (v_b - v_e), v_bc = sign * (v_b - v_c)
        float64x2_t v_v_be = vmulq_f64(v_sign, vsubq_f64(v_v_b, v_v_e));
        float64x2_t v_v_bc = vmulq_f64(v_sign, vsubq_f64(v_v_b, v_v_c));

        // Load parameters
        float64x2_t v_Is    = vld1q_f64(&tensor.Is[i]);
        float64x2_t v_Vt    = vld1q_f64(&tensor.Vt[i]);
        float64x2_t v_BetaF = vld1q_f64(&tensor.BetaF[i]);
        float64x2_t v_BetaR = vld1q_f64(&tensor.BetaR[i]);

        // Vcrit = 30.0 * Vt
        float64x2_t v_vcrit = vmulq_f64(v_30, v_Vt);

        // --- Forward junction (BE) ---
        float64x2_t v_arg_f = vdivq_f64(v_v_be, v_Vt);
        float64x2_t v_arg_f_clamped = vminq_f64(v_arg_f, v_30);
        float64x2_t v_exp_f = vexp_f64(v_arg_f_clamped);
        float64x2_t v_if_base = vmulq_f64(v_Is, vsubq_f64(v_exp_f, v_one));
        float64x2_t v_gf_base = vmulq_f64(vdivq_f64(v_Is, v_Vt), v_exp_f);
        float64x2_t v_gf = vmaxq_f64(v_GMIN, v_gf_base);
        float64x2_t v_delta_f = vmaxq_f64(v_zero, vsubq_f64(v_v_be, v_vcrit));
        float64x2_t v_if = vaddq_f64(v_if_base, vmulq_f64(v_gf, v_delta_f));

        // --- Reverse junction (BC) ---
        float64x2_t v_arg_r = vdivq_f64(v_v_bc, v_Vt);
        float64x2_t v_arg_r_clamped = vminq_f64(v_arg_r, v_30);
        float64x2_t v_exp_r = vexp_f64(v_arg_r_clamped);
        float64x2_t v_ir_base = vmulq_f64(v_Is, vsubq_f64(v_exp_r, v_one));
        float64x2_t v_gr_base = vmulq_f64(vdivq_f64(v_Is, v_Vt), v_exp_r);
        float64x2_t v_gr = vmaxq_f64(v_GMIN, v_gr_base);
        float64x2_t v_delta_r = vmaxq_f64(v_zero, vsubq_f64(v_v_bc, v_vcrit));
        float64x2_t v_ir = vaddq_f64(v_ir_base, vmulq_f64(v_gr, v_delta_r));

        // Ebers-Moll currents
        float64x2_t v_invBF = vdivq_f64(v_one, v_BetaF);
        float64x2_t v_invBR = vdivq_f64(v_one, v_BetaR);

        float64x2_t v_ict = vsubq_f64(v_if, v_ir);
        float64x2_t v_ic_local = vsubq_f64(v_ict, vmulq_f64(v_ir, v_invBR));
        float64x2_t v_ib_local = vaddq_f64(vmulq_f64(v_if, v_invBF), vmulq_f64(v_ir, v_invBR));
        float64x2_t v_ie_local = vnegq_f64(vaddq_f64(v_ic_local, v_ib_local));

        // Apply sign for NPN/PNP
        vst1q_f64(&tensor.Ic[i], vmulq_f64(v_sign, v_ic_local));
        vst1q_f64(&tensor.Ib[i], vmulq_f64(v_sign, v_ib_local));
        vst1q_f64(&tensor.Ie[i], vmulq_f64(v_sign, v_ie_local));

        // Conductances: dI/dVbe, dI/dVbc chain rule
        // dIc/dVbe = gf, dIc/dVbc = -gr * (1 + 1/BetaR)
        float64x2_t v_dIc_dVbe = v_gf;
        float64x2_t v_dIc_dVbc = vnegq_f64(vmulq_f64(v_gr, vaddq_f64(v_one, v_invBR)));
        float64x2_t v_dIb_dVbe = vmulq_f64(v_gf, v_invBF);
        float64x2_t v_dIb_dVbc = vmulq_f64(v_gr, v_invBR);
        float64x2_t v_dIe_dVbe = vnegq_f64(vaddq_f64(v_dIc_dVbe, v_dIb_dVbe));
        float64x2_t v_dIe_dVbc = vnegq_f64(vaddq_f64(v_dIc_dVbc, v_dIb_dVbc));

        // Terminal conductances: dI/dVc = -dI/dVbc, dI/dVe = -dI/dVbe, dI/dVb = dI/dVbe + dI/dVbc
        vst1q_f64(&tensor.g_cc[i], vmaxq_f64(v_GMIN, vnegq_f64(v_dIc_dVbc)));
        vst1q_f64(&tensor.g_cb[i], vaddq_f64(v_dIc_dVbe, v_dIc_dVbc));
        vst1q_f64(&tensor.g_ce[i], vnegq_f64(v_dIc_dVbe));
        vst1q_f64(&tensor.g_bc[i], vnegq_f64(v_dIb_dVbc));
        vst1q_f64(&tensor.g_bb[i], vmaxq_f64(v_GMIN, vaddq_f64(v_dIb_dVbe, v_dIb_dVbc)));
        vst1q_f64(&tensor.g_be[i], vnegq_f64(v_dIb_dVbe));
        vst1q_f64(&tensor.g_ec[i], vnegq_f64(v_dIe_dVbc));
        vst1q_f64(&tensor.g_eb[i], vaddq_f64(v_dIe_dVbe, v_dIe_dVbc));
        vst1q_f64(&tensor.g_ee[i], vmaxq_f64(v_GMIN, vnegq_f64(v_dIe_dVbe)));
    }

    // --- Scalar Tail ---
    for (; i < n; ++i) {
        int nC = tensor.collectors[i];
        int nB = tensor.bases[i];
        int nE = tensor.emitters[i];

        double v_c = (nC > 0 && nC <= v_size) ? v_ptr[nC - 1] : 0.0;
        double v_b = (nB > 0 && nB <= v_size) ? v_ptr[nB - 1] : 0.0;
        double v_e = (nE > 0 && nE <= v_size) ? v_ptr[nE - 1] : 0.0;

        double sign = tensor.isNPN[i] ? 1.0 : -1.0;
        double v_be = sign * (v_b - v_e);
        double v_bc = sign * (v_b - v_c);

        double Is = tensor.Is[i];
        double Vt = tensor.Vt[i];
        double BetaF = tensor.BetaF[i];
        double BetaR = tensor.BetaR[i];
        double Vcrit = 30.0 * Vt;

        auto diode = [&](double vj) -> std::pair<double, double> {
            double arg = std::min(vj / Vt, 30.0);
            double ev = std::exp(arg);
            double ib = Is * (ev - 1.0);
            double gb = std::max(GMIN, (Is / Vt) * ev);
            double delta = std::max(0.0, vj - Vcrit);
            return {ib + gb * delta, gb};
        };

        auto [i_f, g_f] = diode(v_be);
        auto [i_r, g_r] = diode(v_bc);

        double invBF = 1.0 / BetaF;
        double invBR = 1.0 / BetaR;

        double ic_local = (i_f - i_r) - i_r * invBR;
        double ib_local = i_f * invBF + i_r * invBR;
        double ie_local = -ic_local - ib_local;

        tensor.Ic[i] = sign * ic_local;
        tensor.Ib[i] = sign * ib_local;
        tensor.Ie[i] = sign * ie_local;

        double dIc_dVbe = g_f;
        double dIc_dVbc = -g_r * (1.0 + invBR);
        double dIb_dVbe = g_f * invBF;
        double dIb_dVbc = g_r * invBR;
        double dIe_dVbe = -(dIc_dVbe + dIb_dVbe);
        double dIe_dVbc = -(dIc_dVbc + dIb_dVbc);

        tensor.g_cc[i] = std::max(GMIN, -dIc_dVbc);
        tensor.g_cb[i] = dIc_dVbe + dIc_dVbc;
        tensor.g_ce[i] = -dIc_dVbe;
        tensor.g_bc[i] = -dIb_dVbc;
        tensor.g_bb[i] = std::max(GMIN, dIb_dVbe + dIb_dVbc);
        tensor.g_be[i] = -dIb_dVbe;
        tensor.g_ec[i] = -dIe_dVbc;
        tensor.g_eb[i] = dIe_dVbe + dIe_dVbc;
        tensor.g_ee[i] = std::max(GMIN, -dIe_dVbe);
    }
}

#endif // __aarch64__
