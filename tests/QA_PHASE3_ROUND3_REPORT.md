# QA Phase 3 round 3 (branch qa/phase3-round3, phase3/pdk 64266a3)

Baseline: ctest 2/3 (test_engine_fixes: MOS1 `M=3`, already fixed in b8161f2, not re-reported); ngspice diff 382 PASS / 2 FAIL / 20 ENGINE-UNSUPPORTED (before qa3 files; FAIL = qa2_osdi_fewterm cpu+gpu, BUG-7); Xyce 110 PASS / 0 FAIL / 1 TIMEOUT (200 unsupported, 129 parse-error, 20 XYCE-DIALECT, 533 skipped); ibmpg2 PASS (worst 5.15e-6 V, 2.1 s); char_cells all PASS, worst 0.087 %.
Reproducers: `tests/ngspice_diff/netlists/qa3_*.cir` (`run_diff.py -k qa3_`); FAIL by design are those named below. `tests/qa3_ihp_cells.py` runs all IHP std cells in tran. Non-netlist probes were scratch scripts.

| ID | Sev | Reproducer | Engine vs ngspice | Hypothesis |
|----|-----|-----------|-------------------|-----------|
| BUG-1 | wrong answer | `qa3_bsim4_k2_nowk2.cir` (.op), `qa3_sky130_narrow.cir` | level=54 card with `k2` given but no `wk2`: V(3) 1.529 vs 1.868 (Id 3.5x); value of k2 irrelevant; adding `wk2=0` (even 1e-12) fixes. sky130 nfet l=0.5 w=1 Id 4.9x. Whole sky130 sweep (nfet 210 W/L edge points, pfet, pfet_hvt): nfet 111 bad before, 0 bad after patching `wk2=0` into every bin; every sky130 bin of L>=0.18 with W<1.26 (L-only bins) hits it, pmos too | bsim4.osdi binning/given logic for K2 vs WK2 (or engine given-flag passing); only k2 found by bisecting all 288 params of a bin |
| BUG-2 | wrong answer | `qa3_bsim4_tnom_default.cir` | level=54 card without TNOM behaves as TNOM=25, ngspice 27: 4.2 mV (0.26 % Id) at aligned temp; explicit tnom=27 matches | bsim4va default TNOM; set 27 when not given |
| BUG-3 | wrong answer / failure | `qa3_sky130_pfet_w042.cir`, `qa3_bin_edge_top.cir`, `qa3_bin_edge_fuzz.cir` | pfet_01v8 W=0.42 (any L): "no bin covers" (0.42*1e-6 is 1 ulp below wmin=4.2e-7). W=wmax or L=lmax of the last bin: engine error, ngspice uses that bin. ngspice treats edges with ~1e-3 relative tolerance (W=4.9999u picks the next bin; engine the lower, 15 % off). pfet W=100 L=0.15 5x off (BUG-1) | `spice_parser.cpp` ~L548-560 exact `<=`/`<`; use tolerance and inclusive top edge; message also shows doubled model name SKY130_..__SKY130_..__MODEL |
| BUG-4 | wrong answer (low) | scratch: `XM ... sky130_fd_pr__nfet_01v8 l=0.15 w=1.26 nf=2` (also w=1.0, 2.0, 3.0) | nf=2 exactly on a bin edge: engine 2.31e-5 vs 3.33e-5 A; nf=1 and off-edge nf=2 agree | edge handling as BUG-3 |
| BUG-5 | wrong answer (DC) | `qa3_dc_latch_asym.cir` (.op), `qa3_dc_latch_isrc.cir`, `qa3_dc_latch_vsweep.cir` (.dc); IHP `sg13g2_sighold` tran initial state | bistable circuit: engine returns the unstable mid equilibrium (q=0.900, qb=0.862) while ngspice gives a stable state (1.80/0); .dc sweeps of I1/VF follow the metastable branch at all 41/73 points. IHP sighold SH 0.61 V vs 6e-7 V; dfrbp Q at t=0 0 vs 1.2 (legal, arbitrary). Pre-existing (identical on 3e5dee4) | initial seed at VDD/2 converges to the symmetric saddle; ngspice's gmin/source stepping ends on a stable state |
| BUG-6 | wrong answer | `qa3_tran_alias_det.cir`, `qa3_tran_stall_lc.cir` | step is capped only by tstep and there is no source-edge/LTE refinement: 100 MHz sine into diode detector with tstep=1u samples sin()=0 every step, output 0 vs ngspice 1.29 V; 10 MHz LC tank IC=1, tstep=10u gives 4e-81 vs a 0.5 V cosine. (tstep >= 2 points/period works: `qa3_tran_coarse_det_*` pass) | LTE limiter was removed (handoff); tstep is a print interval in SPICE; ngspice caps at tstep and refines |
| BUG-7 | regression | `qa2_osdi_fewterm.cir` (.op) | 3-terminal bsim4va: V(3) 0.5091 vs ngspice 0.4940; 3e5dee4 gave 0.49398 | OSDI internal-node/branch regularisation added in 64266a3 (osdi_device junction gmin / 1 uOhm) changes the floating bulk |
| BUG-8 | unsupported (message ok) | `qa3_r_keyword.cir`, `qa3_c_model.cir`, `qa3_temper_param.cir`, `qa3_subckt_dotparams.cir`, `qa3_params_mid.cir` | `R n1 n2 r=val`; `C n1 n2 model l= w=`; `temper` in .param; `.params` card; `r=3k PARAMS: q=2` all rejected, ngspice accepts. These block IHP `cap_cmim`, HBT (also needs VBIC OSDI) and `schottky_nbl1`. IHP rsil/rppd/rhigh need r3_cmc OSDI (clear error); `R tc1/tc2` unsupported (clear) | |
| BUG-9 | wrong answer (.param) | scratch expression probes | `2**3**2` engine 512, ngspice 64 (left assoc; `^` too); `nint(2.5)`,`nint(-2.5)` 3/-3 vs 2/-2 (half-even); `pwr(-2,3)` -8 vs 8 (|x|^y); `5mil` 25.4u vs 5m (ngspice reads m); `min(3,1,2)` 1 vs 2 (ng two-arg). Unsupported: asinh acosh atanh sqr `<>`. `1/0`, `ln(0)`, `1e400` give silent inf (ng errors); `sqrt(-1)`,`log(-1)`,`10%0` NaN reported as "Singular matrix: check node 2" | numparam.cpp |
| BUG-10 | message | `/private/tmp` probes e11,e13 | `.param a=` and `.param a=1 b` report "line 0"; wrong-arity X reports internal name `S__1`, OSDI model `RSIL__RMOD_RSIL`; errors inside .lib files carry the .lib card's line only; `X ... m=2` on a subckt: "has no parameter M" (ngspice multiplies); `R=0` from a param: "Singular matrix: check branch 2 (loop of voltage sources)" (ngspice sets 1e-12); second `.else` accepted silently; untaken .if branch with bad expression accepted (superset) | numparam.cpp / parser |

Also: engine bins MOS1 (level 1) model families; ngspice only bins BSIM models (superset, harmless).

## Passed
- .param: precedence, unary minus, ternary, && || !, comparisons, SI suffixes, floor/ceil/int/sgn/exp/ln/log10/sqrt/abs/pow/hypot/atan2, forward refs, redefinition, cycle and unknown-parameter errors; quotes/braces/spaces, params in PULSE/SIN/IC/W/L.
- Subckts: 4-level nesting passing params down, body .param overridden from X, X override naming callee's own param, defaults referencing earlier header params, equal/different sets (625-leaf parameter tree), params in tran/ac, .if/.elseif/.elif/.else nested at top level and in subckts, .model inside .if, local .model with params, binned BSIM4 families global and in a subckt, .option scale (MOS1 caps, params).
- IHP: lv corners tt/ff/ss/fs/sf, lv ac/noise (m, ng), hv nmos/pmos (default, ng=2/3 branches, as/ad given, m) dc/ac/noise/tran: all within 0.1 %. All 84 std cells in tran (`tests/qa3_ihp_cells.py`) PASS except BUG-5 initial states.
- sky130 nfet/pfet/pfet_hvt full W/L edge sweep: only BUG-1/3/4.
- DC: Schmitt trigger sweeps up/down, hard start/stop, I-source sweeps, diode/BJT stacks, 500 V diode, SCR, E/G op-amps, 2-stage CMOS op-amp unity feedback, Widlar, diff pair, 5-stage CMOS amplifier chain: Newton without damping converged everywhere ngspice did.
- `.options scale`, m on OSDI, error messages for bad .param/.subckt/.if were specific and non-empty.

## Resolution (2026-10-15, phase3/pdk)
- BUG-1: `bsim4.va` (Cogenda's port) never defaults `wk2` (the `wk1 = 0` line is duplicated where it belongs) or
  `wgamma1`, so they binned at the -12345789 "not given" marker; `tests/build_osdi.sh` patches the copy it
  compiles. Fixed (`qa3_bsim4_k2_nowk2`, `qa3_sky130_narrow` 2e-8 rel).
- BUG-2: same build patch, `DEFAULT_TNOM` 25 -> 27 (ngspice). Also found on the way: the engine's built-in Vt
  was 25.85 mV (299.98 K) and OSDI devices were run at 299.98 K to match it, while the harness ran ngspice at
  26.83 C to hide it. Now Vt = kT/q at 300.15 K, OSDI at the ambient temperature, harness at 27 C; the engine
  and ngspice agree to 7 digits on the same OSDI card.
- BUG-3/4: ngspice's binning rule, probed: edges inclusive within 1e-9 m, total W (not W/NF), and the card
  defined last wins on a shared edge. Fixed (`qa3_bin_edge_*`, `qa3_sky130_pfet_w042`, `qa3_sky130_nf2_edge`).
  `qa3_bin_edge_top` keeps 5e-4: its card says version=4.5, the VA is BSIM4.8.
- BUG-5: the cold-start heuristic seed (source values on their nodes, mid-rail current-source nodes, diode and
  Vbe ladders) put latches on their metastable midpoint. Deleted: DC starts from zeros like ngspice. All
  latch reproducers pass; `qa3_ihp_cells.py` 67 -> 74 of 77 (the 3 left: two flip-flops power up in the other
  legal state, one resolves a setup-marginal edge 4.6 ps differently).
- BUG-6: ngspice's step control (first step min(tstep, tstop/100)/100, cap min(tstep, tstop/50)). The
  detector passes; the LC tank over 1000 periods is out of phase in both simulators (ngspice too: 1.84 V
  max error vs exact), so its netlist is replaced by an analytic envelope check in `test_engine_fixes`.
- BUG-7: the internal-node leak to ground added in 64266a3 is removed (only branch-current unknowns keep
  the 1 uOhm series term). Fixed.
- BUG-8: `R a b r=..`, R TC1/TC2 (exact no-ops at TNOM), semiconductor `C a b model l= w=` (ngspice formula),
  `temper`, `.params`, `PARAMS:` between pairs: supported. Also found: `C ... m=` was parsed and never applied.
  r3_cmc / VBIC OSDI libraries remain out of scope.
- BUG-9: all items match ngspice except min/max with three arguments, now an explicit error.
- BUG-10: generated names (specialized subcircuits, hoisted models) never reach messages; global `.param`
  errors carry their line; a second `.else` and `m=` on an X are explicit errors; the singular-branch hint
  names 0-ohm resistors. Errors inside `.lib` files still carry the `.lib` card's line (open).
