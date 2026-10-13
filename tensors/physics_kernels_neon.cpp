// ARM NEON Physics Kernels — 2-wide double (128-bit)
// Layout matches physics_kernels_avx2.cpp conventions.
// AArch64 NEON has native float64x2_t support.

#include "../tensors/physics_tensors.h"
#include "../physics/physics_constants.h"
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
    const float64x2_t vc_argmax = vdupq_n_f64(PhysicsConstants::JUNCTION_EXP_ARG_MAX);
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

        // arg_clamped = min(arg, PhysicsConstants::JUNCTION_EXP_ARG_MAX)
        float64x2_t v_arg_clamped = vminq_f64(v_arg, vc_argmax);

        // exp_val = exp(arg_clamped)
        float64x2_t v_exp_val = vexp_f64(v_arg_clamped);

        // i_base = Is * (exp_val - 1.0)
        float64x2_t v_i_base = vmulq_f64(v_Is, vsubq_f64(v_exp_val, vc_1));

        // g_base = (Is / n_vt) * exp_val
        float64x2_t v_g_base = vmulq_f64(vdivq_f64(v_Is, v_n_vt), v_exp_val);

        // delta = max(0.0, v_d - PhysicsConstants::JUNCTION_EXP_ARG_MAX * n_vt)
        float64x2_t v_vcrit = vmulq_f64(vc_argmax, v_n_vt);
        float64x2_t v_delta = vmaxq_f64(vc_0, vsubq_f64(v_v_d, v_vcrit));

        // i_d = i_base + delta * g_base + GMIN * v_d
        // (ngspice GMIN: parallel junction conductance — mirrors diode_model_pair())
        float64x2_t v_i_d = vaddq_f64(v_i_base, vmulq_f64(v_delta, v_g_base));
        v_i_d = vaddq_f64(v_i_d, vmulq_f64(v_GMIN, v_v_d));
        vst1q_f64(&tensor.i_d[i], v_i_d);

        // g_d = g_base + GMIN  (additive; replaces the max() floor)
        float64x2_t v_g_d = vaddq_f64(v_g_base, v_GMIN);
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
        double arg_clamped = std::min(arg, PhysicsConstants::JUNCTION_EXP_ARG_MAX);
        double exp_val = std::exp(arg_clamped);
        double i_base = tensor.Is[i] * (exp_val - 1.0);
        double g_base = (tensor.Is[i] / n_vt) * exp_val;
        double delta = std::max(0.0, tensor.v_d[i] - PhysicsConstants::JUNCTION_EXP_ARG_MAX * n_vt);
        // ngspice GMIN: parallel junction conductance — mirrors diode_model_pair()
        tensor.i_d[i] = i_base + delta * g_base + tensor.GMIN[i] * tensor.v_d[i];
        tensor.g_d[i] = g_base + tensor.GMIN[i];
    }
}

// ============================================================================
// BJT KERNEL — Ebers-Moll, 2 devices per iteration
// ============================================================================

void batchBJTPhysics_neon(BJTTensor& tensor, const std::vector<double>& voltages) {
    const size_t n = tensor.size();
    const double GMIN = PhysicsConstants::JUNCTION_GMIN;
    const float64x2_t v_GMIN = vdupq_n_f64(GMIN);
    const float64x2_t v_zero = vdupq_n_f64(0.0);
    const float64x2_t v_one  = vdupq_n_f64(1.0);
    const float64x2_t v_argmax   = vdupq_n_f64(PhysicsConstants::JUNCTION_EXP_ARG_MAX);

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

        // Vcrit = PhysicsConstants::JUNCTION_EXP_ARG_MAX * Vt
        float64x2_t v_vcrit = vmulq_f64(v_argmax, v_Vt);

        // --- Forward junction (BE) ---
        float64x2_t v_arg_f = vdivq_f64(v_v_be, v_Vt);
        float64x2_t v_arg_f_clamped = vminq_f64(v_arg_f, v_argmax);
        float64x2_t v_exp_f = vexp_f64(v_arg_f_clamped);
        float64x2_t v_if_base = vmulq_f64(v_Is, vsubq_f64(v_exp_f, v_one));
        float64x2_t v_gf_base = vmulq_f64(vdivq_f64(v_Is, v_Vt), v_exp_f);
        // continuation with the junction's own slope plus the parallel GMIN (diode_model_pair)
        float64x2_t v_gf = vaddq_f64(v_gf_base, v_GMIN);
        float64x2_t v_delta_f = vmaxq_f64(v_zero, vsubq_f64(v_v_be, v_vcrit));
        float64x2_t v_if = vaddq_f64(v_if_base, vmulq_f64(v_gf_base, v_delta_f));
        v_if = vaddq_f64(v_if, vmulq_f64(v_GMIN, v_v_be));

        // --- Reverse junction (BC) ---
        float64x2_t v_arg_r = vdivq_f64(v_v_bc, v_Vt);
        float64x2_t v_arg_r_clamped = vminq_f64(v_arg_r, v_argmax);
        float64x2_t v_exp_r = vexp_f64(v_arg_r_clamped);
        float64x2_t v_ir_base = vmulq_f64(v_Is, vsubq_f64(v_exp_r, v_one));
        float64x2_t v_gr_base = vmulq_f64(vdivq_f64(v_Is, v_Vt), v_exp_r);
        float64x2_t v_gr = vaddq_f64(v_gr_base, v_GMIN);
        float64x2_t v_delta_r = vmaxq_f64(v_zero, vsubq_f64(v_v_bc, v_vcrit));
        float64x2_t v_ir = vaddq_f64(v_ir_base, vmulq_f64(v_gr_base, v_delta_r));
        v_ir = vaddq_f64(v_ir, vmulq_f64(v_GMIN, v_v_bc));

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
        vst1q_f64(&tensor.g_cc[i], vnegq_f64(v_dIc_dVbc));
        vst1q_f64(&tensor.g_cb[i], vaddq_f64(v_dIc_dVbe, v_dIc_dVbc));
        vst1q_f64(&tensor.g_ce[i], vnegq_f64(v_dIc_dVbe));
        vst1q_f64(&tensor.g_bc[i], vnegq_f64(v_dIb_dVbc));
        vst1q_f64(&tensor.g_bb[i], vaddq_f64(v_dIb_dVbe, v_dIb_dVbc));
        vst1q_f64(&tensor.g_be[i], vnegq_f64(v_dIb_dVbe));
        vst1q_f64(&tensor.g_ec[i], vnegq_f64(v_dIe_dVbc));
        vst1q_f64(&tensor.g_eb[i], vaddq_f64(v_dIe_dVbe, v_dIe_dVbc));
        vst1q_f64(&tensor.g_ee[i], vnegq_f64(v_dIe_dVbe));
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
        double Vcrit = PhysicsConstants::JUNCTION_EXP_ARG_MAX * Vt;

        auto diode = [&](double vj) -> std::pair<double, double> {
            double arg = std::min(vj / Vt, PhysicsConstants::JUNCTION_EXP_ARG_MAX);
            double ev = std::exp(arg);
            double ib = Is * (ev - 1.0);
            double g0 = (Is / Vt) * ev;  // continuation slope + parallel GMIN (diode_model_pair)
            double delta = std::max(0.0, vj - Vcrit);
            return {ib + g0 * delta + GMIN * vj, g0 + GMIN};
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

        tensor.g_cc[i] = -dIc_dVbc;
        tensor.g_cb[i] = dIc_dVbe + dIc_dVbc;
        tensor.g_ce[i] = -dIc_dVbe;
        tensor.g_bc[i] = -dIb_dVbc;
        tensor.g_bb[i] = dIb_dVbe + dIb_dVbc;
        tensor.g_be[i] = -dIb_dVbe;
        tensor.g_ec[i] = -dIe_dVbc;
        tensor.g_eb[i] = dIe_dVbe + dIe_dVbc;
        tensor.g_ee[i] = -dIe_dVbe;
    }
}

#endif // __aarch64__
