/*
 * gpu_nr_loop.wgsl
 * Persistent GPU-resident Newton-Raphson loop kernels.
 * Uses EmulatedF64 (hi-lo float32 pairs) for numerical authority.
 *
 * Phase D5a: Full f64_exp with range reduction + Horner polynomial,
 *             hi+lo Jacobian/RHS writes, BJT kernel, residual/convergence.
 */

struct f64 {
    hi: f32,
    lo: f32,
};

// --- EmulatedF64 Helpers (Double-Float Arithmetic) ---

fn f64_from_f32(a: f32) -> f64 {
    return f64(a, 0.0);
}

fn f64_add(a: f64, b: f64) -> f64 {
    let s = a.hi + b.hi;
    let v = s - a.hi;
    let e = (a.hi - (s - v)) + (b.hi - v) + a.lo + b.lo;
    let hi = s + e;
    let lo = e + (s - hi);
    return f64(hi, lo);
}

fn f64_sub(a: f64, b: f64) -> f64 {
    return f64_add(a, f64(-b.hi, -b.lo));
}

fn f64_mul(a: f64, b: f64) -> f64 {
    let hi = a.hi * b.hi;
    let lo = fma(a.hi, b.hi, -hi) + a.hi * b.lo + a.lo * b.hi;
    let s = hi + lo;
    let l = lo + (hi - s);
    return f64(s, l);
}

// Newton iteration for f64 division
fn f64_div(a: f64, b: f64) -> f64 {
    let x = a.hi / b.hi; // Float32 estimate
    let x64 = f64_from_f32(x);
    // One refinement: x = x * (2 - b*x)
    let b_x = f64_mul(b, x64);
    let error = f64_sub(f64_from_f32(2.0), b_x);
    let refined_recip = f64_mul(x64, error);
    return f64_mul(a, refined_recip);
}

// --- Proper f64_exp: range reduction + 6th-order Horner polynomial ---
// Algorithm:
//   1. Clamp large forward bias at MAX_ARG=30 (linear continuation)
//   2. Range reduce: k = round(x / ln2), r = x - k*ln2
//   3. exp(r) via Horner polynomial (degree 6)
//   4. Reconstruct: exp(x) = 2^k * exp(r)
fn f64_exp(x: f64) -> f64 {
    let MAX_ARG: f32 = 30.0;
    let EXP30: f32 = 1.06864745815e13; // exp(30) in f32

    // Underflow guard
    if (x.hi <= -87.0) {
        return f64(0.0, 0.0);
    }

    // Linear continuation for large forward bias (mirrors CPU pnjlim clamping)
    if (x.hi >= MAX_ARG) {
        // exp(x) ≈ exp(30) + exp(30)*(x - 30) = exp(30)*(1 + (x-30))
        let slope = f64(EXP30, 0.0);
        let dx = f64_sub(x, f64(MAX_ARG, 0.0));
        return f64_add(slope, f64_mul(slope, dx));
    }

    // Range reduction: k = round(x.hi / ln2), r = x - k*ln2
    // ln2 in double-float: hi=0.6931471805599453, lo=2.3190468138463e-17
    let INV_LN2: f32 = 1.4426950408889634;
    let LN2_HI: f32 = 0.6931471805599453;
    let LN2_LO: f32 = 2.3190468138463e-17;

    let k = round(x.hi * INV_LN2);
    let ln2_f64 = f64(LN2_HI, LN2_LO);
    let r = f64_sub(x, f64_mul(f64(k, 0.0), ln2_f64));

    // Horner: 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120 + r/720)))))
    // Innermost to outermost:
    var p = f64(1.38888888888889e-3, 0.0); // 1/720
    p = f64_add(f64(8.33333333333333e-3, 0.0), f64_mul(r, p)); // 1/120
    p = f64_add(f64(4.16666666666667e-2, 0.0), f64_mul(r, p)); // 1/24
    p = f64_add(f64(1.66666666666667e-1, 0.0), f64_mul(r, p)); // 1/6
    p = f64_add(f64(5.0e-1, 0.0), f64_mul(r, p));              // 1/2
    p = f64_add(f64(1.0, 0.0), f64_mul(r, p));                 // 1
    p = f64_add(f64(1.0, 0.0), f64_mul(r, p));                 // 1 (outermost)

    // Reconstruct: exp(x) = 2^k * exp(r)
    let scale = pow(2.0, k);
    return f64_mul(f64(scale, 0.0), p);
}

// --- Data Structures (AoS) ---

struct DiodeDevice {
    anode: i32,
    cathode: i32,
    Is_hi: f32, Is_lo: f32,
    N_hi: f32,  N_lo: f32,
    Vt_hi: f32, Vt_lo: f32,
    v_d_hi: f32, v_d_lo: f32,
    i_d_hi: f32, i_d_lo: f32,
    g_d_hi: f32, g_d_lo: f32,
};
struct DiodeSoA {
    devices: array<DiodeDevice>,
};

struct MosfetDevice {
    drain: i32,
    gate: i32,
    source: i32,
    body: i32,
    W_hi: f32, W_lo: f32,
    L_hi: f32, L_lo: f32,
    Kp_hi: f32, Kp_lo: f32,
    Vth_hi: f32, Vth_lo: f32,
    lambda_hi: f32, lambda_lo: f32,
    isPMOS: u32,
    vgs_hi: f32, vgs_lo: f32,
    vds_hi: f32, vds_lo: f32,
    vbs_hi: f32, vbs_lo: f32,
    ids_hi: f32, ids_lo: f32,
    gm_hi: f32,  gm_lo: f32,
    gmb_hi: f32, gmb_lo: f32,
    gds_hi: f32, gds_lo: f32,
};
struct MosfetSoA {
    devices: array<MosfetDevice>,
};

// BJT AoS device (input params + output results)
struct BJTDevice {
    collector: i32,
    base: i32,
    emitter: i32,
    isNPN: u32,
    Is_hi: f32, Is_lo: f32,
    betaF: f32,
    betaR: f32,
    Vt_hi: f32, Vt_lo: f32,
    // Output: terminal currents
    ic_hi: f32, ic_lo: f32,
    ib_hi: f32, ib_lo: f32,
    // Output: conductance matrix (3x3 = 9 terms, C/B/E rows and cols)
    g_cc_hi: f32, g_cc_lo: f32,
    g_cb_hi: f32, g_cb_lo: f32,
    g_ce_hi: f32, g_ce_lo: f32,
    g_bc_hi: f32, g_bc_lo: f32,
    g_bb_hi: f32, g_bb_lo: f32,
    g_be_hi: f32, g_be_lo: f32,
    g_ec_hi: f32, g_ec_lo: f32,
    g_eb_hi: f32, g_eb_lo: f32,
    g_ee_hi: f32, g_ee_lo: f32,
    // Stored voltages for i_eq computation in assembly
    vc_hi: f32, vc_lo: f32,
    vb_hi: f32, vb_lo: f32,
    ve_hi: f32, ve_lo: f32,
};
struct BJTSoA {
    devices: array<BJTDevice>,
};

struct GlobalState {
    time_hi: f32,
    time_lo: f32,
    h_hi: f32,
    h_lo: f32,
};

// --- Bindings ---
// Group 0: MNA state (devices + solution vectors)
@group(0) @binding(0) var<storage, read_write> diodes: DiodeSoA;
@group(0) @binding(1) var<storage, read_write> mosfets: MosfetSoA;
@group(0) @binding(2) var<storage, read_write> voltages_hi: array<f32>;
@group(0) @binding(3) var<storage, read_write> voltages_lo: array<f32>;
@group(0) @binding(4) var<storage, read_write> deltaV_hi: array<f32>;
@group(0) @binding(5) var<storage, read_write> deltaV_lo: array<f32>;
@group(0) @binding(6) var<storage, read_write> rhs_hi: array<f32>;
@group(0) @binding(7) var<storage, read_write> rhs_lo: array<f32>;
// Split Jacobian (replaces CsrMatrix struct — WGSL forbids two unsized arrays in one struct)
@group(0) @binding(8) var<storage, read_write> jacobian_hi: array<f32>;
@group(0) @binding(9) var<storage, read_write> jacobian_lo: array<f32>;

// Stamp-index maps for CSR positional writes
struct DiodeStampMap {
    aa: u32, cc: u32, ac: u32, ca: u32,
};
struct MosfetStampMap {
    dd: u32, dg: u32, ds: u32, db: u32,
    sd: u32, sg: u32, ss: u32, sb: u32,
};
// 9 Jacobian positions for BJT: (CC, CB, CE, BC, BB, BE, EC, EB, EE)
struct BJTStampMap {
    cc: u32, cb: u32, ce: u32,
    bc: u32, bb: u32, be: u32,
    ec: u32, eb: u32, ee: u32,
};

// Group 1: Topology (frozen per netlist) + simulation state
@group(1) @binding(0) var<storage, read>       diode_maps: array<DiodeStampMap>;
@group(1) @binding(1) var<storage, read>       mosfet_maps: array<MosfetStampMap>;
@group(1) @binding(2) var<storage, read>       global_state: GlobalState;
@group(1) @binding(3) var<storage, read_write> bjts: BJTSoA;           // BJTs (moved from g0)
@group(1) @binding(4) var<storage, read>       bjt_maps: array<BJTStampMap>;
@group(1) @binding(5) var<storage, read_write> residual_buffer: array<f32>; // Per-workgroup max residuals
@group(1) @binding(6) var<storage, read_write> convergence_flag: array<u32>; // [0]=1 if converged

// --- Stage 1a: Batch Diode Physics ---

@compute @workgroup_size(64)
fn batchDiodePhysics(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&diodes.devices)) { return; }

    let nA = diodes.devices[i].anode;
    let nC = diodes.devices[i].cathode;

    var v_a = f64(0.0, 0.0);
    if (nA > 0) { v_a = f64(voltages_hi[u32(nA-1)], voltages_lo[u32(nA-1)]); }
    var v_c = f64(0.0, 0.0);
    if (nC > 0) { v_c = f64(voltages_hi[u32(nC-1)], voltages_lo[u32(nC-1)]); }

    let v_d = f64_sub(v_a, v_c);
    diodes.devices[i].v_d_hi = v_d.hi;
    diodes.devices[i].v_d_lo = v_d.lo;

    let Is  = f64(diodes.devices[i].Is_hi, diodes.devices[i].Is_lo);
    let nvt = f64_mul(
        f64(diodes.devices[i].N_hi,  diodes.devices[i].N_lo),
        f64(diodes.devices[i].Vt_hi, diodes.devices[i].Vt_lo)
    );

    let arg     = f64_div(v_d, nvt);
    let exp_arg = f64_exp(arg);          // properly range-reduced

    let i_d = f64_mul(Is, f64_sub(exp_arg, f64(1.0, 0.0)));
    let g_d = f64_div(f64_mul(Is, exp_arg), nvt);

    diodes.devices[i].i_d_hi = i_d.hi; diodes.devices[i].i_d_lo = i_d.lo;
    diodes.devices[i].g_d_hi = g_d.hi; diodes.devices[i].g_d_lo = g_d.lo;
}

// --- Stage 1b: Batch MOSFET Physics ---

@compute @workgroup_size(64)
fn batchMosfetPhysics(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&mosfets.devices)) { return; }

    let nD = mosfets.devices[i].drain;
    let nG = mosfets.devices[i].gate;
    let nS = mosfets.devices[i].source;
    let nB = mosfets.devices[i].body;

    var v_d_orig = f64(0.0, 0.0);
    if (nD > 0) { v_d_orig = f64(voltages_hi[u32(nD-1)], voltages_lo[u32(nD-1)]); }
    var v_g_orig = f64(0.0, 0.0);
    if (nG > 0) { v_g_orig = f64(voltages_hi[u32(nG-1)], voltages_lo[u32(nG-1)]); }
    var v_s_orig = f64(0.0, 0.0);
    if (nS > 0) { v_s_orig = f64(voltages_hi[u32(nS-1)], voltages_lo[u32(nS-1)]); }
    var v_b_orig = f64(0.0, 0.0);
    if (nB > 0) { v_b_orig = f64(voltages_hi[u32(nB-1)], voltages_lo[u32(nB-1)]); }

    let pmos = mosfets.devices[i].isPMOS;
    var sign = 1.0f;
    if (pmos != 0u) { sign = -1.0f; }

    let vgs = f64_sub(v_g_orig, v_s_orig);
    let vds = f64_sub(v_d_orig, v_s_orig);
    let vbs = f64_sub(v_b_orig, v_s_orig);

    mosfets.devices[i].vgs_hi = vgs.hi; mosfets.devices[i].vgs_lo = vgs.lo;
    mosfets.devices[i].vds_hi = vds.hi; mosfets.devices[i].vds_lo = vds.lo;
    mosfets.devices[i].vbs_hi = vbs.hi; mosfets.devices[i].vbs_lo = vbs.lo;

    let local_vgs = f64_mul(f64(sign, 0.0), vgs);
    let local_vds = f64_mul(f64(sign, 0.0), vds);
    let local_vth = f64_mul(f64(sign, 0.0), f64(mosfets.devices[i].Vth_hi, mosfets.devices[i].Vth_lo));

    let K = f64_mul(
        f64(0.5, 0.0),
        f64_mul(
            f64(mosfets.devices[i].Kp_hi, mosfets.devices[i].Kp_lo),
            f64_div(
                f64(mosfets.devices[i].W_hi, mosfets.devices[i].W_lo),
                f64(mosfets.devices[i].L_hi, mosfets.devices[i].L_lo)
            )
        )
    );
    let vov = f64_sub(local_vgs, local_vth);

    if (vov.hi <= 0.0) {
        mosfets.devices[i].ids_hi = 0.0; mosfets.devices[i].ids_lo = 0.0;
        mosfets.devices[i].gm_hi  = 0.0; mosfets.devices[i].gm_lo  = 0.0;
        mosfets.devices[i].gmb_hi = 0.0; mosfets.devices[i].gmb_lo = 0.0;
        mosfets.devices[i].gds_hi = 1e-12; mosfets.devices[i].gds_lo = 0.0; // GMIN
    } else if (local_vds.hi < vov.hi) {
        // Linear region
        let ids = f64_mul(f64(sign, 0.0), f64_mul(K,
            f64_sub(
                f64_mul(f64(2.0, 0.0), f64_mul(vov, local_vds)),
                f64_mul(local_vds, local_vds)
            )
        ));
        let gm  = f64_mul(f64(2.0, 0.0), f64_mul(K, local_vds));
        let gds = f64_mul(f64(2.0, 0.0), f64_mul(K, f64_sub(vov, local_vds)));

        mosfets.devices[i].ids_hi = ids.hi; mosfets.devices[i].ids_lo = ids.lo;
        mosfets.devices[i].gm_hi  = gm.hi;  mosfets.devices[i].gm_lo  = gm.lo;
        mosfets.devices[i].gmb_hi = 0.0;    mosfets.devices[i].gmb_lo = 0.0;
        mosfets.devices[i].gds_hi = gds.hi; mosfets.devices[i].gds_lo = gds.lo;
    } else {
        // Saturation region
        let lam  = f64(mosfets.devices[i].lambda_hi, mosfets.devices[i].lambda_lo);
        let term = f64_add(f64(1.0, 0.0), f64_mul(lam, local_vds));
        let ids  = f64_mul(f64(sign, 0.0), f64_mul(K, f64_mul(f64_mul(vov, vov), term)));
        let gm   = f64_mul(f64(2.0, 0.0), f64_mul(K, f64_mul(vov, term)));
        let gds  = f64_mul(K, f64_mul(f64_mul(vov, vov), lam));

        mosfets.devices[i].ids_hi = ids.hi; mosfets.devices[i].ids_lo = ids.lo;
        mosfets.devices[i].gm_hi  = gm.hi;  mosfets.devices[i].gm_lo  = gm.lo;
        mosfets.devices[i].gmb_hi = 0.0;    mosfets.devices[i].gmb_lo = 0.0;
        mosfets.devices[i].gds_hi = gds.hi; mosfets.devices[i].gds_lo = gds.lo;
    }
}

// --- Stage 1c: Batch BJT Physics (Ebers-Moll) ---

@compute @workgroup_size(64)
fn batchBJTPhysics(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&bjts.devices)) { return; }

    let nC = bjts.devices[i].collector;
    let nB = bjts.devices[i].base;
    let nE = bjts.devices[i].emitter;

    // Read node voltages (1-based; 0 = ground)
    var v_c = f64(0.0, 0.0);
    if (nC > 0) { v_c = f64(voltages_hi[u32(nC-1)], voltages_lo[u32(nC-1)]); }
    var v_b = f64(0.0, 0.0);
    if (nB > 0) { v_b = f64(voltages_hi[u32(nB-1)], voltages_lo[u32(nB-1)]); }
    var v_e = f64(0.0, 0.0);
    if (nE > 0) { v_e = f64(voltages_hi[u32(nE-1)], voltages_lo[u32(nE-1)]); }

    // Store voltages for assembly (i_eq requires vc, vb, ve)
    bjts.devices[i].vc_hi = v_c.hi; bjts.devices[i].vc_lo = v_c.lo;
    bjts.devices[i].vb_hi = v_b.hi; bjts.devices[i].vb_lo = v_b.lo;
    bjts.devices[i].ve_hi = v_e.hi; bjts.devices[i].ve_lo = v_e.lo;

    let isNPN = bjts.devices[i].isNPN;
    var sign = 1.0f;
    if (isNPN == 0u) { sign = -1.0f; }
    let sign_f64 = f64(sign, 0.0);

    let Is   = f64(bjts.devices[i].Is_hi,  bjts.devices[i].Is_lo);
    let Vt   = f64(bjts.devices[i].Vt_hi,  bjts.devices[i].Vt_lo);
    let betaF = bjts.devices[i].betaF;
    let betaR = bjts.devices[i].betaR;
    let invBetaF = f64(1.0 / betaF, 0.0);
    let invBetaR = f64(1.0 / betaR, 0.0);

    // v_be = sign * (v_b - v_e),  v_bc = sign * (v_b - v_c)
    let v_be = f64_mul(sign_f64, f64_sub(v_b, v_e));
    let v_bc = f64_mul(sign_f64, f64_sub(v_b, v_c));

    // Forward junction: IF = Is*(exp(v_be/Vt) - 1),  gF = Is*exp(v_be/Vt)/Vt
    let arg_f  = f64_div(v_be, Vt);
    let exp_f  = f64_exp(arg_f);
    let i_f    = f64_mul(Is, f64_sub(exp_f, f64(1.0, 0.0)));
    let g_f    = f64_div(f64_mul(Is, exp_f), Vt);

    // Reverse junction: IR = Is*(exp(v_bc/Vt) - 1),  gR = Is*exp(v_bc/Vt)/Vt
    let arg_r  = f64_div(v_bc, Vt);
    let exp_r  = f64_exp(arg_r);
    let i_r    = f64_mul(Is, f64_sub(exp_r, f64(1.0, 0.0)));
    let g_r    = f64_div(f64_mul(Is, exp_r), Vt);

    // Terminal currents (Ebers-Moll)
    //   ic_local = (i_f - i_r) - i_r/betaR
    //   ib_local = i_f/betaF  + i_r/betaR
    let i_ct      = f64_sub(i_f, i_r);
    let ib_comp_f = f64_mul(i_f, invBetaF);
    let ib_comp_r = f64_mul(i_r, invBetaR);
    let ic_local  = f64_sub(i_ct, ib_comp_r);
    let ib_local  = f64_add(ib_comp_f, ib_comp_r);
    // ie_local = -ic_local - ib_local  (not stored separately)

    // Apply polarity sign
    let ic = f64_mul(sign_f64, ic_local);
    let ib = f64_mul(sign_f64, ib_local);

    bjts.devices[i].ic_hi = ic.hi; bjts.devices[i].ic_lo = ic.lo;
    bjts.devices[i].ib_hi = ib.hi; bjts.devices[i].ib_lo = ib.lo;

    // Conductances: dIc/dVbe, dIc/dVbc, dIb/dVbe, dIb/dVbc
    // (see device_physics.h bjt_ebers_moll for derivation)
    let dIc_dVbe = g_f;                                              // dIct/dVbe
    let dIc_dVbc = f64_sub(f64_mul(f64(-1.0, 0.0), g_r),
                           f64_mul(g_r, invBetaR));                  // -g_r - g_r/betaR
    let dIb_dVbe = f64_mul(g_f, invBetaF);
    let dIb_dVbc = f64_mul(g_r, invBetaR);
    let dIe_dVbe = f64_mul(f64(-1.0, 0.0), f64_add(dIc_dVbe, dIb_dVbe));
    let dIe_dVbc = f64_mul(f64(-1.0, 0.0), f64_add(dIc_dVbc, dIb_dVbc));

    // Conductance matrix (node-referenced: Vbe=Vb-Ve, Vbc=Vb-Vc)
    //   g_cc = -dIc/dVbc  (dIc/dVc since Vbc=Vb-Vc → dIc/dVc = -dIc/dVbc)
    //   g_cb =  dIc/dVbe + dIc/dVbc  (dIc/dVb)
    //   g_ce = -dIc/dVbe             (dIc/dVe = -dIc/dVbe)
    let g_cc = f64_mul(f64(-1.0, 0.0), dIc_dVbc);
    let g_cb = f64_add(dIc_dVbe, dIc_dVbc);
    let g_ce = f64_mul(f64(-1.0, 0.0), dIc_dVbe);
    let g_bc = f64_mul(f64(-1.0, 0.0), dIb_dVbc);
    let g_bb = f64_add(dIb_dVbe, dIb_dVbc);
    let g_be = f64_mul(f64(-1.0, 0.0), dIb_dVbe);
    let g_ec = f64_mul(f64(-1.0, 0.0), dIe_dVbc);
    let g_eb = f64_add(dIe_dVbe, dIe_dVbc);
    let g_ee = f64_mul(f64(-1.0, 0.0), dIe_dVbe);

    bjts.devices[i].g_cc_hi = g_cc.hi; bjts.devices[i].g_cc_lo = g_cc.lo;
    bjts.devices[i].g_cb_hi = g_cb.hi; bjts.devices[i].g_cb_lo = g_cb.lo;
    bjts.devices[i].g_ce_hi = g_ce.hi; bjts.devices[i].g_ce_lo = g_ce.lo;
    bjts.devices[i].g_bc_hi = g_bc.hi; bjts.devices[i].g_bc_lo = g_bc.lo;
    bjts.devices[i].g_bb_hi = g_bb.hi; bjts.devices[i].g_bb_lo = g_bb.lo;
    bjts.devices[i].g_be_hi = g_be.hi; bjts.devices[i].g_be_lo = g_be.lo;
    bjts.devices[i].g_ec_hi = g_ec.hi; bjts.devices[i].g_ec_lo = g_ec.lo;
    bjts.devices[i].g_eb_hi = g_eb.hi; bjts.devices[i].g_eb_lo = g_eb.lo;
    bjts.devices[i].g_ee_hi = g_ee.hi; bjts.devices[i].g_ee_lo = g_ee.lo;
}

// --- Stage 2: Assemble Jacobian (write both hi AND lo) ---
// Note: WGSL lacks f64 atomics, so concurrent writes to the same CSR cell
// from different threads may race. In the pipeline-overlap design, physics
// runs in the previous pass and all stamp writes per device are distinct CSR
// entries (guaranteed by the topology CSR mapping). For safety, this kernel
// is dispatched with max(numDiodes, numMosfets, numBJTs) invocations and
// each thread handles at most one device of each type independently.

@compute @workgroup_size(64)
fn assembleJacobian(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    // --- Diode Assembly ---
    if (i < arrayLength(&diodes.devices)) {
        let g   = f64(diodes.devices[i].g_d_hi, diodes.devices[i].g_d_lo);
        let map = diode_maps[i];

        // Write BOTH hi and lo to preserve full f64 precision
        if (map.aa != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.aa], jacobian_lo[map.aa]);
            let upd = f64_add(cur, g);
            jacobian_hi[map.aa] = upd.hi;
            jacobian_lo[map.aa] = upd.lo;
        }
        if (map.cc != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.cc], jacobian_lo[map.cc]);
            let upd = f64_add(cur, g);
            jacobian_hi[map.cc] = upd.hi;
            jacobian_lo[map.cc] = upd.lo;
        }
        if (map.ac != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.ac], jacobian_lo[map.ac]);
            let upd = f64_sub(cur, g);
            jacobian_hi[map.ac] = upd.hi;
            jacobian_lo[map.ac] = upd.lo;
        }
        if (map.ca != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.ca], jacobian_lo[map.ca]);
            let upd = f64_sub(cur, g);
            jacobian_hi[map.ca] = upd.hi;
            jacobian_lo[map.ca] = upd.lo;
        }

        // RHS equivalent current: i_eq = i_d - g_d * v_d
        let v_d = f64(diodes.devices[i].v_d_hi, diodes.devices[i].v_d_lo);
        let i_d = f64(diodes.devices[i].i_d_hi, diodes.devices[i].i_d_lo);
        let i_eq = f64_sub(i_d, f64_mul(g, v_d));
        let nA = diodes.devices[i].anode;
        let nC = diodes.devices[i].cathode;
        if (nA > 0) {
            let cur = f64(rhs_hi[u32(nA-1)], rhs_lo[u32(nA-1)]);
            let upd = f64_sub(cur, i_eq);
            rhs_hi[u32(nA-1)] = upd.hi;
            rhs_lo[u32(nA-1)] = upd.lo;
        }
        if (nC > 0) {
            let cur = f64(rhs_hi[u32(nC-1)], rhs_lo[u32(nC-1)]);
            let upd = f64_add(cur, i_eq);
            rhs_hi[u32(nC-1)] = upd.hi;
            rhs_lo[u32(nC-1)] = upd.lo;
        }
    }

    // --- MOSFET Assembly ---
    if (i < arrayLength(&mosfets.devices)) {
        let dev = mosfets.devices[i];
        let gm  = f64(dev.gm_hi,  dev.gm_lo);
        let gmb = f64(dev.gmb_hi, dev.gmb_lo);
        let gds = f64(dev.gds_hi, dev.gds_lo);
        let ids = f64(dev.ids_hi, dev.ids_lo);
        let vgs = f64(dev.vgs_hi, dev.vgs_lo);
        let vds = f64(dev.vds_hi, dev.vds_lo);
        let vbs = f64(dev.vbs_hi, dev.vbs_lo);
        let map = mosfet_maps[i];

        // Helper: accumulate into CSR entry (both hi and lo)
        // Row D
        if (map.dd != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.dd], jacobian_lo[map.dd]);
            let upd = f64_add(cur, gds);
            jacobian_hi[map.dd] = upd.hi; jacobian_lo[map.dd] = upd.lo;
        }
        if (map.dg != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.dg], jacobian_lo[map.dg]);
            let upd = f64_add(cur, gm);
            jacobian_hi[map.dg] = upd.hi; jacobian_lo[map.dg] = upd.lo;
        }
        if (map.ds != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.ds], jacobian_lo[map.ds]);
            let upd = f64_sub(cur, f64_add(gds, f64_add(gm, gmb)));
            jacobian_hi[map.ds] = upd.hi; jacobian_lo[map.ds] = upd.lo;
        }
        if (map.db != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.db], jacobian_lo[map.db]);
            let upd = f64_add(cur, gmb);
            jacobian_hi[map.db] = upd.hi; jacobian_lo[map.db] = upd.lo;
        }
        // Row S
        if (map.sd != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.sd], jacobian_lo[map.sd]);
            let upd = f64_sub(cur, gds);
            jacobian_hi[map.sd] = upd.hi; jacobian_lo[map.sd] = upd.lo;
        }
        if (map.sg != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.sg], jacobian_lo[map.sg]);
            let upd = f64_sub(cur, gm);
            jacobian_hi[map.sg] = upd.hi; jacobian_lo[map.sg] = upd.lo;
        }
        if (map.ss != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.ss], jacobian_lo[map.ss]);
            let upd = f64_add(cur, f64_add(gds, f64_add(gm, gmb)));
            jacobian_hi[map.ss] = upd.hi; jacobian_lo[map.ss] = upd.lo;
        }
        if (map.sb != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.sb], jacobian_lo[map.sb]);
            let upd = f64_sub(cur, gmb);
            jacobian_hi[map.sb] = upd.hi; jacobian_lo[map.sb] = upd.lo;
        }

        // RHS: i_eq = ids - (gm*vgs + gmb*vbs + gds*vds)
        let i_eq = f64_sub(ids,
            f64_add(f64_mul(gm, vgs),
            f64_add(f64_mul(gmb, vbs), f64_mul(gds, vds))));
        let nD = dev.drain;
        let nS = dev.source;
        if (nD > 0) {
            let cur = f64(rhs_hi[u32(nD-1)], rhs_lo[u32(nD-1)]);
            let upd = f64_sub(cur, i_eq);
            rhs_hi[u32(nD-1)] = upd.hi; rhs_lo[u32(nD-1)] = upd.lo;
        }
        if (nS > 0) {
            let cur = f64(rhs_hi[u32(nS-1)], rhs_lo[u32(nS-1)]);
            let upd = f64_add(cur, i_eq);
            rhs_hi[u32(nS-1)] = upd.hi; rhs_lo[u32(nS-1)] = upd.lo;
        }
    }

    // --- BJT Assembly ---
    if (i < arrayLength(&bjts.devices)) {
        let dev = bjts.devices[i];
        let nC  = dev.collector;
        let nB  = dev.base;
        let nE  = dev.emitter;
        let map = bjt_maps[i];

        let g_cc = f64(dev.g_cc_hi, dev.g_cc_lo);
        let g_cb = f64(dev.g_cb_hi, dev.g_cb_lo);
        let g_ce = f64(dev.g_ce_hi, dev.g_ce_lo);
        let g_bc = f64(dev.g_bc_hi, dev.g_bc_lo);
        let g_bb = f64(dev.g_bb_hi, dev.g_bb_lo);
        let g_be = f64(dev.g_be_hi, dev.g_be_lo);
        let g_ec = f64(dev.g_ec_hi, dev.g_ec_lo);
        let g_eb = f64(dev.g_eb_hi, dev.g_eb_lo);
        let g_ee = f64(dev.g_ee_hi, dev.g_ee_lo);
        let ic   = f64(dev.ic_hi,   dev.ic_lo);
        let ib   = f64(dev.ib_hi,   dev.ib_lo);
        let ie   = f64_mul(f64(-1.0, 0.0), f64_add(ic, ib));
        let vc   = f64(dev.vc_hi, dev.vc_lo);
        let vb   = f64(dev.vb_hi, dev.vb_lo);
        let ve   = f64(dev.ve_hi, dev.ve_lo);

        // Collector row Jacobian
        if (map.cc != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.cc], jacobian_lo[map.cc]);
            let upd = f64_add(cur, g_cc);
            jacobian_hi[map.cc] = upd.hi; jacobian_lo[map.cc] = upd.lo;
        }
        if (map.cb != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.cb], jacobian_lo[map.cb]);
            let upd = f64_add(cur, g_cb);
            jacobian_hi[map.cb] = upd.hi; jacobian_lo[map.cb] = upd.lo;
        }
        if (map.ce != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.ce], jacobian_lo[map.ce]);
            let upd = f64_add(cur, g_ce);
            jacobian_hi[map.ce] = upd.hi; jacobian_lo[map.ce] = upd.lo;
        }
        // Base row Jacobian
        if (map.bc != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.bc], jacobian_lo[map.bc]);
            let upd = f64_add(cur, g_bc);
            jacobian_hi[map.bc] = upd.hi; jacobian_lo[map.bc] = upd.lo;
        }
        if (map.bb != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.bb], jacobian_lo[map.bb]);
            let upd = f64_add(cur, g_bb);
            jacobian_hi[map.bb] = upd.hi; jacobian_lo[map.bb] = upd.lo;
        }
        if (map.be != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.be], jacobian_lo[map.be]);
            let upd = f64_add(cur, g_be);
            jacobian_hi[map.be] = upd.hi; jacobian_lo[map.be] = upd.lo;
        }
        // Emitter row Jacobian
        if (map.ec != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.ec], jacobian_lo[map.ec]);
            let upd = f64_add(cur, g_ec);
            jacobian_hi[map.ec] = upd.hi; jacobian_lo[map.ec] = upd.lo;
        }
        if (map.eb != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.eb], jacobian_lo[map.eb]);
            let upd = f64_add(cur, g_eb);
            jacobian_hi[map.eb] = upd.hi; jacobian_lo[map.eb] = upd.lo;
        }
        if (map.ee != 0xffffffffu) {
            let cur = f64(jacobian_hi[map.ee], jacobian_lo[map.ee]);
            let upd = f64_add(cur, g_ee);
            jacobian_hi[map.ee] = upd.hi; jacobian_lo[map.ee] = upd.lo;
        }

        // RHS equivalent currents:
        //   i_eq_c = ic - (g_cc*vc + g_cb*vb + g_ce*ve)
        //   i_eq_b = ib - (g_bc*vc + g_bb*vb + g_be*ve)
        //   i_eq_e = ie - (g_ec*vc + g_eb*vb + g_ee*ve)
        let i_eq_c = f64_sub(ic,  f64_add(f64_mul(g_cc, vc), f64_add(f64_mul(g_cb, vb), f64_mul(g_ce, ve))));
        let i_eq_b = f64_sub(ib,  f64_add(f64_mul(g_bc, vc), f64_add(f64_mul(g_bb, vb), f64_mul(g_be, ve))));
        let i_eq_e = f64_sub(ie,  f64_add(f64_mul(g_ec, vc), f64_add(f64_mul(g_eb, vb), f64_mul(g_ee, ve))));

        if (nC > 0) {
            let cur = f64(rhs_hi[u32(nC-1)], rhs_lo[u32(nC-1)]);
            let upd = f64_sub(cur, i_eq_c);
            rhs_hi[u32(nC-1)] = upd.hi; rhs_lo[u32(nC-1)] = upd.lo;
        }
        if (nB > 0) {
            let cur = f64(rhs_hi[u32(nB-1)], rhs_lo[u32(nB-1)]);
            let upd = f64_sub(cur, i_eq_b);
            rhs_hi[u32(nB-1)] = upd.hi; rhs_lo[u32(nB-1)] = upd.lo;
        }
        if (nE > 0) {
            let cur = f64(rhs_hi[u32(nE-1)], rhs_lo[u32(nE-1)]);
            let upd = f64_sub(cur, i_eq_e);
            rhs_hi[u32(nE-1)] = upd.hi; rhs_lo[u32(nE-1)] = upd.lo;
        }
    }
}

// --- Stage 3: Update Solution ---
@compute @workgroup_size(64)
fn updateSolution(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&voltages_hi)) { return; }

    let x  = f64(voltages_hi[i], voltages_lo[i]);
    let dx = f64(deltaV_hi[i], deltaV_lo[i]);
    let new_x = f64_add(x, dx);

    voltages_hi[i] = new_x.hi;
    voltages_lo[i] = new_x.lo;
}

// --- Stage 4: Compute Residual (max-norm, per-workgroup reduction) ---
// Each invocation handles one RHS entry; workgroup computes local max,
// writes to residual_buffer[workgroup_id.x].

var<workgroup> shared_max: array<f32, 64>;

@compute @workgroup_size(64)
fn computeResidual(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:     vec3<u32>
) {
    let i   = global_id.x;
    let lid = local_id.x;
    let n   = arrayLength(&rhs_hi);

    // Each thread reads its RHS magnitude (hi component dominates)
    var val = 0.0f;
    if (i < n) {
        val = abs(rhs_hi[i]); // hi dominates; lo contributes < 1 ULP
    }
    shared_max[lid] = val;
    workgroupBarrier();

    // Binary reduction within workgroup
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            if (shared_max[lid + stride] > shared_max[lid]) {
                shared_max[lid] = shared_max[lid + stride];
            }
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride >> 1u;
    }

    // Thread 0 writes per-workgroup max to residual_buffer
    if (lid == 0u) {
        let wg_idx = wg_id.x;
        if (wg_idx < arrayLength(&residual_buffer)) {
            residual_buffer[wg_idx] = shared_max[0];
        }
    }
}

// --- Stage 5: Convergence Check (single thread) ---
// Reads all per-workgroup residuals, finds max, compares to tolerance.
// Writes 1 to convergence_flag[0] if converged, 0 otherwise.

@compute @workgroup_size(1)
fn convergenceCheck() {
    let n_wg  = arrayLength(&residual_buffer);
    let tol   = 1e-6f; // Must match CPU KCL_TOL

    var max_res = 0.0f;
    for (var k = 0u; k < n_wg; k++) {
        if (residual_buffer[k] > max_res) {
            max_res = residual_buffer[k];
        }
    }

    if (arrayLength(&convergence_flag) > 0u) {
        convergence_flag[0] = select(0u, 1u, max_res < tol);
    }
}

// --- Phase 2.2: GPU-resident Waveform Buffering ---

struct WaveformConfig {
    decimation_ratio: u32,
    buffer_capacity: u32,
    node_count: u32,
};

struct WaveformState {
    write_ptr: u32,
    decimation_counter: u32,
};

@group(2) @binding(0) var<storage, read>       wf_config: WaveformConfig;
@group(2) @binding(1) var<storage, read_write> wf_state: WaveformState;
@group(2) @binding(2) var<storage, read_write> wf_data: array<f32>;

@compute @workgroup_size(1)
fn recordWaveform(@builtin(global_invocation_id) global_id: vec3<u32>) {
    wf_state.decimation_counter += 1u;
    if (wf_state.decimation_counter < wf_config.decimation_ratio) {
        return;
    }
    wf_state.decimation_counter = 0u;

    let capacity    = wf_config.buffer_capacity;
    let nodes       = wf_config.node_count;
    let entry_stride = 2u + (nodes * 2u);

    let ptr    = wf_state.write_ptr;
    let offset = ptr * entry_stride;

    if (offset + entry_stride > arrayLength(&wf_data)) {
        return;
    }

    wf_data[offset]     = global_state.time_hi;
    wf_data[offset + 1u] = global_state.time_lo;

    for (var n = 0u; n < nodes; n++) {
        let v_offset = offset + 2u + (n * 2u);
        wf_data[v_offset]     = voltages_hi[n];
        wf_data[v_offset + 1u] = voltages_lo[n];
    }

    wf_state.write_ptr = (ptr + 1u) % capacity;
}
