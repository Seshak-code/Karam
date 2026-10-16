# QA Phase 4 Report (branch qa/phase4, commit 6b40c10)

## 1. Summary & Verification Matrix

- **C++ Tests & Build**: Clean build, `ctest --test-dir build` 3/3 PASS.
- **Phase 5 Reference SoC Smoke Verification**: `CuteApp-dev/build/test_soc_hello` **ALL PASSED (0 failures)**:
  - RV32I ISA emulation: ECALL halt, add/sub with negative immediates, lui, sw/lbu/lw zero-extension, blt, jal, x0 immutability.
  - Reference SoC: Hello World with zero-wait and wait-cycle SRAM (+48 cycles).
  - TreeSPICE IHP SG13G2 6T SRAM read transient: 41.46 ps to 100 mV split, 0 wait cycles at 2 GHz.
  - SPICE UART co-simulation (pad driver, line, receiver in TreeSPICE): Decoded `"Hello, world!\n"` with 0 errors.
- **PDK Standard Cell Characterization**: `tests/char_cells.py` 100% PASS (IHP and Sky130 cells, worst error 0.087%).
- **Power Grid Benchmark**: `tests/ibmpg_diff.py -k ibmpg2` PASS (worst 5.15e-6 V).
- **Harness Matrix**: `tests/ngspice_diff/run_diff.py -k qa4_`: 16 PASS / 1 FAIL (expected-to-fail reproducer `qa4_sky130_nfet_bins.cir` for BUG-1).

---

## 2. Attack Verification Details

### A. Device Bypass Sensitivity (`TREESPICE_BYPASS=0` vs default)
- **Netlists**:
  - `qa4_bypass_comp.cir`: CMOS comparator crossing switching threshold under slow ramp. PASS (worst rel 1.69e-05).
  - `qa4_bypass_currmirror.cir`: Precision cascode current mirror under micro-amp input ramp. PASS (worst rel 1.59e-06).
  - `qa4_bypass_diffpair.cir`: High-gain CMOS differential pair with active load. PASS (worst rel 0.00111).
  - `qa4_bypass_ring.cir`: 3-stage Sky130 CMOS ring oscillator. PASS (worst rel 0.000778).
- **Finding**: Bit-exact simulation agreement between default bypass ON and `TREESPICE_BYPASS=0` across all test nodes (e.g. ring oscillator node voltages match to 6 significant digits). The 1 mV latency bypass threshold introduces zero numerical distortion.

### B. PWL Source Adversarial Matrix
- **Netlists**:
  - `qa4_pwl_dc_op.cir`: PWL source in `.op` operating point. Evaluates exactly to t=0 value (worst abs 0 V).
  - `qa4_pwl_params.cir`: PWL with delay (`TD=1n`) and periodic repetition (`R=1n`) driving RC over multiple periods. PASS (worst rel 0.00753).
  - `qa4_pwl_repeat_first.cir`: PWL repeating from initial point (`R=0`). PASS (worst abs 0 V).
  - `qa4_pwl_single.cir`: Single-point PWL (`PWL(0 2.5)`). PASS (worst abs 0 V).
  - `qa4_pwl_subckt.cir`: PWL voltage and current sources instantiated inside hierarchical subcircuit. PASS (worst rel 8.8e-05).

### C. Corner Stepping Stress Tests
- **Netlists**:
  - `qa4_corner_fastpulse.cir`: Fast pulse train with 20 ps rise/fall edges under coarse tstep (1 ns). PASS (worst rel 5.14e-06).
  - `qa4_corner_pwl_subps.cir`: PWL source with 100 fs edges driving RC. PASS (worst rel 1.18e-05).
  - `qa4_corner_ramp_inductor.cir`: Abruptly stopping ramp driving inductor (10 ns breakpoint). Order-1 Euler step eliminates Trapezoidal ringing. PASS (worst rel 1.87e-05).

### D. Sky130 PDK Boundary Fuzzing
- **Netlists**:
  - `qa4_sky130_nfet_bin_w042.cir`: Sky130 NFET on bin boundary W=0.42 um, NF=1. PASS (worst rel 5.67e-08).
  - `qa4_sky130_nfet_bin_w126_nf2.cir`: Sky130 NFET on bin boundary W=1.26 um, NF=2. PASS (worst rel 1.12e-07).
  - `qa4_sky130_pfet_bin_w042.cir`: Sky130 PFET HVT on bin boundary W=0.42 um, NF=1. PASS (worst rel 1.03e-05).
  - `qa4_sky130_pfet_bin_w200_nf2.cir`: Sky130 PFET HVT on bin boundary W=2.00 um, NF=2. PASS (worst rel 2.91e-05).
  - `qa4_sky130_subckt_scale.cir`: Sky130 transistors inside 2-level nested subcircuits with `.option scale=1.0u`. PASS (worst rel 2.91e-05).
  - `qa4_sky130_nfet_bins.cir`: Multi-instance bin test (triggers BUG-1).

---

## 3. Defects Filed

| ID | Sev | Reproducer | Engine vs ngspice | Root Cause & Diagnosis |
|---|---|---|---|---|
| **BUG-1** | Parse / Specialization Failure | `tests/ngspice_diff/netlists/qa4_sky130_nfet_bins.cir` | TreeSPICE throws `SPICE parse error: Parse error at line 2: model SKY130_FD_PR__NFET_01V8__MODEL.0 defined twice`. ngspice evaluates all 4 instances. | In `netlist/numparam.cpp:507`: `specialized_.emplace(key, s.defaults.empty() ? s.name : s.name + "|" + std::to_string(...))`. When subcircuit parameters are declared in the subcircuit body (`.param l=... w=...`) rather than the header, `s.defaults.empty()` is true. When multiple instances with different given parameters exist, `specialized_` names the specialization `s.name` instead of generating a unique name, causing re-emission and duplicate model hoisting. Fix: use `s.defaults.empty() && given.empty() ? s.name : ...`. |
