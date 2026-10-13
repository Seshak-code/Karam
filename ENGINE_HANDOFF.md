# Engine Handoff — Karam (acutesim_engine)

*Since 2026-10-13 Karam is a standalone repository. CuteApp consumes it as the submodule `acutesim_engine/`.*

*Written 2026-10-12 at the end of the engine-review session. Read this first, then `CLAUDE.md`,
then the two Roadmap sections it points to. Everything below was verified by running it.*

## 0. Where things stand

- **Branch:** `main` (PR #1 merged 2026-10-12 by Cline: `engine/spice-correctness` → `main`,
  merge commit `43003d5`; the branch was deleted after merge). New engine work starts on
  `main` or a fresh branch off it. Stale leftover: `engine/tensor-mna` (8 months old, superseded).
- **Update 2026-10-12 (third session):** Task 1 and **all of Phase 2** are done; the legacy
  solver stack is deleted (~40k lines). **Next:** (a) junction gmin (see §6, fixes `aac_10_bridge`),
  (b) golden datasets: IBM power-grid benchmarks (solver scale) + Xyce regression subset (devices),
  (c) GUI Verilog-A transport (PULSE + AC magnitude), (d) Roadmap Phase 3.
- **Update 2026-10-12 (fourth session, committed by Cline):** the third session's uncommitted stretch is now
  committed and verified: stricter parser (unknown dot-cards/model params/behavioural sources are hard
  errors, `.global`, `R=0`, auto `.tran 0 <stop>` step, first-order step after waveform corners), Verilog-A
  transport round-trips PULSE/SIN/AC mag+phase, IBM power-grid datasets downloaded to `datasets/`
  (git-ignored by design) with new harnesses `tests/ibmpg_diff.py` + `tests/xyce_diff.py`.
  Verified: clean build; ngspice harness 82/83 (only the known `aac_10_bridge`); full ctest 28/32 —
  `test_gpu_parity`, `test_kernel_dispatch`, `test_numerical_intelligence`, `test_grpc_orchestrator`
  fail **identically at the committed baseline** (pre-existing, not from this stretch).
  Trap: `tests/` is in `.gitignore` ("Local Project Exclusions") — **new** test files are silently
  untracked; `git add -f` them. Cross-agent protocol: `.clinerules/01-collab-handoff.md`.
- **Update 2026-10-12 (fifth session, Cline):** **junction gmin DONE** — the engine now puts
  the 1e-12 S gmin across each PN junction (diode/BJT/zener models + SoA kernels, both audits)
  instead of node-to-ground, matching ngspice; the node gmin is dropped at convergence.
  **`aac_10_bridge` passes** — ngspice harness 84 PASS / 0 FAIL (cpu+gpu). Full ctest 28/32,
  the 4 pre-existing failures unchanged. **Next:** (a) wire the golden datasets
  (`tests/ibmpg_diff.py`, `tests/xyce_diff.py`) into regular verification, (b) Roadmap Phase 3.
  (The "two queued GUI issues" turned out to be already fixed: the Verilog-A transport gap by
  `c7faeb0`, the two failing GUI tests by `b1d9fd2` — `test_hierarchy_gui` and
  `test_virtuoso_parity` both verified passing 2026-10-12, offscreen Qt env; Roadmap row updated.)
- **Update 2026-10-12 (sixth session, Claude Code — review of `7332f62`):** junction gmin kept;
  review fixes: (1) BJT SoA kernels (generic + NEON vector/tail) counted GMIN twice in the
  linear-continuation region and kept `max(g, GMIN)` floors — now exactly `diode_model_pair`
  (slope g_base, + GMIN·v; no floors); one `PhysicsConstants::JUNCTION_GMIN`. (2) The contraction
  solver left a root front's pivot-less rows at x = 0 and reported success, so floating nodes
  "converged" (series caps → V=0, undriven gate → −5 V): now singular, and DC fails with
  "Singular matrix: check node N" exactly like ngspice. A node no element touches (X pin to an
  unused port) is not a circuit node and is pinned to 0 V (ngspice drops it). (3) Ladder end by
  step index (no float compare); `gminSteppingUsed` true only if the ladder ran; sensitivity Jacobian
  without node gmin; `addDiagonalConditioning` has no default gmin. (4) `test_std_cells` Darlington
  check was vacuous (read the internal node; VCC on node 10 left nodes 3–9 floating) — now β = 10 200.
  Noise shot current keeps the junction-gmin term (ngspice's DIOcurrent includes it).
  Verified: all 19 suites; ngspice 84/84; Xyce 0 FAIL / 78 PASS (V–L loops at DC are XYCE-DIALECT:
  ngspice `.op` is singular too); IBM 6/6 (ibmpg6 37 s). **Next:** golden datasets into regular
  verification, Roadmap Phase 3.
- **Update 2026-10-12 (seventh session, Claude Code):** Phase 3 started under the dev/QA split
  (`.clinerules/06-dev-qa-split.md`). PR #2 `phase3/mos1`: complete level-1 MOSFET DC (S/D swap, body
  effect, bulk junctions, M, LD, SPICE defaults) as one `mos1_currents()` + Dual-Jacobian `stampMos1()`;
  SIMD MOSFET kernels deleted; awaiting Cline QA. Next increments: MOS1 capacitances → diode RS/CJO/TT/BV
  → Gummel–Poon BJT + caps → `.lib`/`.include` + corners → BSIM4 (the Verilog-A compiler the Roadmap
  planned to use was deleted as unused; BSIM4 needs its own route, decide at that increment).
- **Update 2026-10-12 (eighth session, Claude Code):** PR #3 is on `phase3/mos1-caps` (`70318fd`, stacked on
  PR #2). MOS1 charges follow ngspice:
  - Meyer gate (TOX), overlap, and bulk-junction depletion caps.
  - One `mos1_charges()` feeds the stamp, audit, history, LTE and AC.
  - Each card is resolved once by the parser (`Mosfet::card`); `MosfetTensor` is deleted.
- **Update 2026-10-12 (QA session, Cline):** Phase 3.1 **PASSED** and Phase 3.2 **PASSED with BUG-1
  filed** — full verdicts, the three BUG entries (BUG-1 uic/MOS-charge init from 3.2; BUG-2 depth-2
  `.subckt` wrong DC, pre-existing; BUG-3 coarse-tstep LTE, pre-existing) and the 25 committed
  adversarial/reproducer netlists are in `memory-bank/qa-queue.md`. Process repair: PR #3's stack was
  merged into its base before the base merged, orphaning 3.2 from main — corrective PR #4 opened by QA
  and merged (`6bdf3ee`); stacked PRs must merge bottom-up from now on. Full matrix on `70318fd`:
  ngspice 144 PASS (only the 5 expected-to-fail reproducers fail), Xyce 78/0, IBM 6/6, ctest 28/32
  (pre-existing only), GUI path unchanged. **Next (dev):** BUG-1 fix → FIXED–RETEST; the ≥100-stage
  `.op` singular issue; then BUG-2/BUG-3 (both pre-existing, in the queue).

  Verified: 19 suites; ngspice 104/104 (AC caps rel 3e-9, ring oscillator 1.4 %); Xyce 0 FAIL; ibmpg2 PASS.
  READY FOR QA. **Found (pre-existing on main):** `.op` on an inverter chain of 100 or more stages fails with
  "Singular matrix"; ngspice converges with dynamic gmin stepping.

  **Next:** DC homotopy for long digital chains, then diode RS/CJO/TT/BV.
- **Update 2026-10-13 (Claude Code, took over Cline's unfinished `fix/qa-bugs`):** BUG-1/2/3 were
  reference-resolution artifacts of the ngspice harness, not engine bugs (BUG-3: the unchanged engine is
  within 4.6 mV of a dense ngspice run; the `tmax = tstep/10` reference was 114 mV off). Harness:
  reference `tmax` also capped at 1/20 of the shortest source edge, and an engine value at a switching
  jump is judged against the reference cell's bracket. Engine: ngspice **dynamic gmin stepping** replaces
  the fixed ladder, so 10–500-stage inverter chains `.op` like ngspice. Cline's engine LTE limiter and
  root-pivot fallback were dropped (no wrong answer fixed; 180× steps). Verified: 17 suites, ngspice
  160/0, Xyce 78/0 (1 pre-existing TIMEOUT), ibmpg2. Also: `acutesim_engine/README.md` (PR #5) is the
  Karam README via the weekly subtree sync. **Next:** QA retest (`qa-queue.md`), then diode RS/CJO/TT/BV.
- **Status of the plan:** Roadmap Phase 1 (Foundations) is done. See `Roadmap.md` →
  "⚙️ ENGINE FEATURE LIST & VERIFIED SUPPORT STATUS" (what works, by device/analysis, with ✅/⚠️/❌/🚫/❓)
  and "🎯 END-TO-END GOAL & ENGINE SCALE PLAN" (phases and gates).

```
0bce71c perf(engine): run independent contraction fronts on worker threads
a90e4af fix(engine): apply MOSFET channel-length modulation in the linear region
0b90505 feat(engine): MNA branch currents for V sources, inductors, E/F/H and K
a200bd7 feat(engine): tensor-contraction direct solver as the default linear solver
04d4b95 test: ngspice differential harness and engine waveform dump tool
1b7b760 fix(engine): working subcircuits — local node scoping and instance-aware solver
2d858fa docs(roadmap): add end-to-end SoC goal and phased engine scale plan
cada846 docs(roadmap): add verified engine feature list and support status
4e16c99 fix(engine): SPICE correctness — fail-closed convergence, parser, waveforms, E/G/J
fe2c077 chore: ignore graphify-out code-graph artifacts
```
`git log 97a0735..HEAD` has the full messages.

## 1. The owner's direction (do not drift from this)

- **Goal:** one application to design an SoC, connect RAM, run firmware on it and see the program's
  output, covering every step from schematic/RTL to sign-off.
- **Core bet: tensor contraction.** Simulating up to 10⁹ transistors on a local GPU via contraction
  and the parallelism superposition gives. Superposition holds *inside each Newton step* (the
  linearised system): child fronts contract independently and their Schur complements **add** into the
  parent. It does **not** parallelise across Newton iterations or timesteps.
- **Physics reality (measured):** M4 Pro, 26 GB unified memory. Flat SPICE-accurate 10⁹ devices does
  not fit (>150 GB). Billion-scale needs FastSPICE techniques on top of contraction: identical cells
  share tensors (isomorphism reuse), activity bypass, GPU-batched fronts. Firmware runs on RTL/gate
  models; SPICE is for the blocks that need it, co-simulated.
- The owner likes delegating focused, well-specified tasks to **Sonnet subagents**, and wants
  verification by **running** things (the ngspice harness), not by reading checklists.

## 2. Engine architecture as of now

```
SPICE text ──► SPICEParser (netlist/spice_parser.*) ──► TensorNetlist (netlist/circuit.h)
                 • local node scoping in .subckt, X expanded after parse (recursive)
                 • .model/.param/F-H control names resolved after the whole file
                 • unknown/unsupported cards = hard parse error (never silently dropped)
TensorNetlist ─► CircuitSim::solveDC / runTransient / solveAC / solveNoise (solvers/circuitsim_solver.cpp)
                 unknowns x = [N node voltages | B branch currents]   (systemSize())
                 per Newton iteration:
                   stampAllElements (physics/circuitsim.h)
                     ├─ SoA path (default): stampSoABlock = R, I, C, D, M, Q tensors
                     │    + stampUncovered = G (VCCS), J (JFET) from the netlist
                     ├─ AoS path (LEGACY_AOS, used by ai_api speculative runner)
                     └─ stampBranchSystem = V, L(+K), E, F, H  ◄─ ONE function for both paths
                   solveLinear(A, b) ─► contraction solver (math/contraction_solver.*)
                   kclExcess(...)   ─► audit residual beyond KCL_RELTOL·(|A||x|) (scale-aware)
                   checkConvergence (fail-closed on NaN)
Session API: engine_impl.cpp (ISimulationEngine): .op/.dc/.tran/.ac/.noise from the netlist cards
            (or DCConfigDTO / ACConfigDTO / TransientConfigDTO::uic)
GUI path:   compute/orchestration/driver/local_worker_driver.cpp → the same CircuitSim entry points
            (string parameters: stopTime, stepHint, uic, fStart/fStop/points, inputSource, outputNode)
AC/noise:   Y(jw) = G + jwC; G = the DC Jacobian at the OP without node gmin, C = capacitors + L/M on
            the branch rows; real 2N system [[G,-wC],[wC,G]] through the contraction solver; noise uses
            one adjoint solve per frequency (R^T, own plan) + ngspice Nintegrate for totals
```

Key invariants (each was a real bug when broken):
- **Stamps and the KCL audit must use the same formulas.** Branch elements guarantee it by construction
  (`buildBranchSystem` feeds both). Other devices: if you change a stamp, change
  `calculatePhysicalResiduals` (global AND instance sections) too.
- **Branch order:** all V sources, then inductors, then E, then H; each over the global block, then
  instances in `tensorizeNetlist` order (`branchSlots()`). History updates walk the same order.
- **Outputs are node voltages only:** `SolverStep::nodeVoltages` is truncated to N. The full x lives in
  `CircuitSim::lastSolution_` (seeds transient Newton and sensitivity).
- **Waveforms:** a single `waveformValue()` (netlist/circuit.h) is used by parser, stamps and audit.
- **Junction exponent cap:** `PhysicsConstants::JUNCTION_EXP_ARG_MAX = 50`, shared by every diode/BJT
  kernel and `pnjlim` (keep them in sync).
- **Device parameters:** `mosfetModel()` (circuit.h) and `mosfetKernelParams()/bjtKernelParams()`
  (circuitsim.h) are the one definition for tensorize, both KCL audits and noise.
- **Responses fail closed:** `SimulationResponseDTO::success/converged` default to false.
- **Contraction solver:** plan once per sparsity pattern (MC64-style matching → hubs to root → nested
  dissection → fronts grouped in levels); factor every iteration (assemble from CSR, add children,
  threshold partial pivoting u = 0.01, delayed pivots to parent); solve + iterative refinement. Levels run
  on `std::thread` workers (deterministic). It is the only linear solver.

## 3. How to build and verify (run all of these after any engine change)

Karam is its own repository (Seshak-code/Karam). AcuteSim/CuteApp mounts it as the git submodule
`acutesim_engine/`. Engine work happens here: commit and push in Karam, then bump the submodule
pointer in CuteApp and run CuteApp's integration suites there.

```bash
cmake -S . -B build && cmake --build build -j10
ctest --test-dir build      # test_engine_fixes, test_contraction_solver, test_convergence_engine
TMPDIR=/private/tmp/claude-502 python3 tests/ngspice_diff/run_diff.py   # ~2 min; needs ngspice (brew install ngspice)
python3 tests/xyce_diff.py              # 0 FAIL expected; needs datasets/xyce
python3 tests/ibmpg_diff.py -k ibmpg2   # needs datasets/ibmpg
```
Integration suites that need the app layers stay in CuteApp: run them there with the submodule at the
new commit. They include test_regression (the CI gate, run from the CuteApp root),
test_engine_parity, test_hierarchy_headless, test_std_cells, test_soa_bridge, test_physics_kernels,
test_headless_orchestrator, test_analysis_registry, test_determinism_xclient, test_veriloga_roundtrip,
test_parallel_monte_carlo, test_driver_swap, test_flow and test_simulation.

- `tests/test_engine_fixes.cpp` = one regression check per fix (analytic/hand-derived references,
  CPU and GPU modes). Add a check there for every engine fix.
- `build/acutesim_engine_dump <netlist> <cpu|gpu> <dc|tran|ac|noise> [step stop]` prints
  machine-readable waveforms (node order = first appearance, subckt bodies excluded).
- **Golden DB updates:** never accept `--update` wholesale. Run it in a scratch copy, splice only the
  entries whose change you have derived physically, and say why in the commit (see commits
  `4e16c99`-series diode entry and `0b90505`).

## 4. Task 1 — GUI large-design performance (`NetlistCompiler::compile`)

**What is known (verified):**
- `NetlistCompiler::compile()` (`acutesim_engine/infrastructure/netlist_compiler.cpp`, ctx overload in
  `netlist_compiler_ctx.cpp`) runs: tensorize → GraphPartitioner → FactorGraph → **TreewidthAnalyzer
  (Min-Fill, hash-set elimination, ~quadratic: 100+ s at 10⁴ nodes)** → TNCompiler. The WebGPU upload
  used to call it on every solve; that call was replaced by `tensorizeNetlist()` (commit `a200bd7`).
- Since `0b90505` the solver no longer reads `tnAnalysis` / `tnProgram` / `partitions`. The only
  remaining reader is `WebGPUSolver`'s legacy contraction path (`renderGraph`), itself unused by `solveDC`.
- Callers of `compile()`: `compute/orchestration/batch_runner.cpp` (once per sweep point),
  `tests/test_hierarchy_headless.cpp`, `tests/test_hierarchy_gui.cpp`, `tests/test_virtuoso_parity.cpp`.
- **Correction to the previous session's summary:** the GUI's normal path
  (`LocalWorkerDriver` → `BatchExecutionEngine` → `CircuitSim::solveDC/stepTransient`) does **not**
  call `compile()`. `gui/analysis/parametric_sweep_dialog.h` includes `batch_runner.h` but no GUI code
  was found instantiating `BatchRunner`. So the "GUI compile slowdown" may not be what users hit.

**Do this:**
1. Measure first. Drive the real app path headlessly with a generated 10⁴–10⁵-node netlist (grids:
   see `Roadmap` measurements; generator = a few lines of Python) through `LocalWorkerDriver` and profile
   (`sample <pid>` on macOS). Find where time actually goes (schematic→netlist conversion, design DB,
   `BatchExecutionEngine`, Dawn/GPU context start-up ~0.5–1 s even in CPU mode, …). Fix what dominates.
2. Make `compile()` cheap regardless: drop or lazily compute the stages nobody consumes
   (factor graph, Min-Fill, TN compile, partitioning). Check each consumer with grep before removing;
   keep `tensors`, metadata, `structural_`, interface pins. Expected: `test_hierarchy_headless` (97) still
   passes; compile of a 10⁴-node netlist drops from minutes to ~ms.
3. Decide (ask the owner) whether to delete the now-dead legacy stack: `sparse_lu.cpp` (dense),
   `contraction_executor`, `schur_solver`, `mpo_builder`, `tn_compiler`, `treewidth_analyzer`,
   `factor_graph`, the WebGPU hybrid NR loop. They have tests of their own; deletion is a scope call.
- **Gate:** a 10⁴-node design runs DC end-to-end through the app path in < 1 s; batch sweeps do not
  re-run quadratic analysis per point.
- **Done (measured, 100×100 / 316×316 resistor grids through `LocalWorkerDriver`):** the GUI path never
  called `compile()`; the real cost was `TensorBlock::computeTopologyHash` (std::map BFS, run in the
  parser AND in `solveDC`: ~1.5 s of a 2 s solve at 10⁵ nodes). `canonicalRelabel` is now linear and
  bit-identical (20k random graphs + pinned value in `test_engine_fixes`). `compile()` no longer runs
  partitioner / FactorGraph / Min-Fill / TN compile (no readers): minutes → 0.5 ms at 10⁴, 7 ms at 10⁵.
  10⁴ DC job 419 → 213 ms; 10⁵ 4.95 → 2.1 s (generate 1.0 s + parse 1.4 s remain; `build/` is a **Debug**
  build, so orchestration/compute code is -O0 — only the engine forces -O3). The Verilog-A preview dock
  (highlighted, GUI thread, 0.6 s per 2.5 MB) now shows the first 256 KB only.
  Found + fixed on the way: the CSR pattern cache was keyed on the canonical hash (ignores sources and
  node numbering) and `assembleFast` silently dropped stamps outside the pattern — a reused `CircuitSim`
  returned 0 V for a moved source. The pattern now self-validates.
  Step 3 done (owner: "no tech debt"): deleted sparse_lu, Schur, MPO/TN compiler/executor, treewidth,
  factor graph, partitioner, contraction tree/render graph, bond-dimension profiler, Metal/WebGPU
  solvers + shaders, NetlistCompiler/CompiledTensorBlock/BatchRunner (GUI keeps `SweepParameter`),
  SuperpositionSolver, StagedDispatcher, QuickAnalysis, the unused engine Verilog-A compiler, orphan
  headers, stale generated protobuf copies, dead CircuitSim members, and their tests.

## 5. Task 2 — Roadmap Phase 2 (Analysis parity)

From `Roadmap.md` Phase 2, with pointers:
1. **DC operating point as transient initial state.** Today transient always starts from zero state even
   without `uic` (`engine_impl.cpp` solves DC only as a gate; `stepTransient` history starts empty).
   Seed capacitor voltages / inductor currents from the DC solution unless `uic`; honour `.ic`.
   Add `uic`/IC to the DTO (`engine_api/simulation_dto.h`) so the GUI path can request it.
2. **Breakpoints** at PULSE corners (and SIN start/delay): the step controller must land on them
   (`waveformValue` has a `ponytail:` note). Square-wave integrator (#17) converges only first-order now.
3. **`.dc` sweeps** in the parser/session (the ngspice harness emulates them today).
4. **AC + noise through `ISimulationEngine`.** `SessionImpl::runAC` currently returns a DC result
   labelled AC 🚫; `runNoise` aliases it. The GUI path has a real `ACSolver::solveAC`
   (`solvers/ac_solver.h`, unverified). It is voltage-only (Norton sources): it must get the same MNA
   branch-current treatment, then be wired to the session and checked against ngspice (`.ac` netlists
   `aac_05..09` already in the harness, currently SKIPPED).
5. **One transient loop** shared by `SessionImpl::dispatch_tran_internal` and
   `LocalWorkerDriver` (the latter has fixed step, stops at first failure, no UIC).
- **Gate:** every analysis in the feature list ✅ against ngspice (extend `run_diff.py` to `.ac`).
- **Done:** items 1, 2, 5. `CircuitSim::runTransient` is the one loop (session, `LocalWorkerDriver`, Fourier):
  DC bias seeds C/L history (`seedReactiveHistory`) unless UIC; `.ic V(n)=v` is forced through 1e10 S
  during the bias (SPICE3) or is the initial state with UIC; C/L `IC=` honoured with UIC (inductor junk
  params are now an error); `TransientConfigDTO::uic`; GUI honours `stepHint`. Steps land on PULSE corners
  and SIN delay (`nextBreakpoint()` beside `waveformValue()`). Removed the automatic TRAP→GEAR_2 "ringing"
  switch: every GEAR_2 formula is constant-step BDF2 and the switch fired while h changed, injecting jumps
  (`ponytail:` note at the old site). Harness: x_rc_pulse 51.7 → 0.36 mV, hier_rc_ladder_pulse 82.8 → 1.3 mV,
  aac_17 square integrator 3.7 → 0.08 mV; new netlists x_rc_dcop_step, x_rl_dcop_step, x_elem_ic_uic,
  x_dotic_bias. Item 4 note: `ACSolver` reuses the DC pattern via `assembleFast` — pass the `complete`
  flag there too, or AC-only stamps are dropped silently. (Moot now: the old ACSolver was deleted.)
- **Done:** items 3 and 4. `.dc` native (parser + `DCConfigDTO`, harness no longer emulates it).
  AC/noise as in §2, `.ac`/`.noise` cards, `ACConfigDTO`, `NoiseResultDTO`; the GUI driver's AC (which
  never excited the circuit and ignored the corner) and NOISE (an AC alias) now use the same solver.
  The old ACSolver (its own device physics, hard-coded MOSFET params, Norton sources, no E/F/G/H/K or
  subcircuits) and PZ (fabricated poles) were deleted; TF rebuilt on `solveAC`. Harness gained 5 AC +
  4 noise comparisons (aac_05..09 had a SPICE2 stray `sin`, removed).
- **Also fixed on the way:** fail-open response defaults; LocalWorkerDriver simulating an empty netlist
  for non-Verilog-A sources; GUI Monte Carlo always running 10 samples (`numRuns` vs `sampleCount`);
  UIC unreachable on batch runs; instance KCL audit using hard-coded MOSFET params / unclamped diodes;
  three tests that passed vacuously (JSON netlists, overwritten payload, a race).

## 6. Known issues / backlog (not yet fixed)

- ~~Junction gmin~~ **FIXED 2026-10-12 (Cline):** gmin now sits across each junction
  (device models + SoA kernels, both audits) instead of node-to-ground; the node gmin
  is dropped at convergence. `aac_10_bridge` passes. Details: `ENGINE_HANDOFF.md` §0.
- Temperature: devices use a fixed Vt = 25.85 mV (≈ 299.98 K) while `ambient_temp_K` defaults to
  300.15 K (noise uses the latter: 0.03 % vs the harness's 26.83 °C). Derive Vt from the temperature.
- Session Monte Carlo: `mcSeed` never applied (every run identical) 🚫; the driver path works. `test_headless_orchestrator` has 3
  pre-existing failures (DTO default `success`, two MC result counts) — verified identical at baseline.
- Hard nonlinear DC: plain-Newton-first costs extra iterations (Darlington 35 → ~115); consider a lower
  iteration cap on the direct attempt.
- Instance KCL audit: diodes use the unclamped equation and MOSFETs hard-coded params (global audit uses
  model cards) — inconsistent for model-card MOSFETs inside subcircuits.
- MOSFET level 1: no source/drain swap for Vds < 0, no body effect. Diode: no RS/CJO/TT. BJT: no VAF, RB/RC/RE,
  junction caps. AREA, EXP/PWL/SFFM/AM, T lines: explicit errors.
- Subcircuit parameters (`params:`) unsupported. `.include/.lib/.options/.ic/.print` silently ignored.
- Floating high-impedance nodes: the "KCL < 1e-9 A ⇒ converged" shortcut can accept stale voltages.
- Junction-cap shortcut and other simplifications are marked `ponytail:` in code (grep for it).
- AVX2/AVX-512 kernels are x86-only and were not built on this Mac; they mirror the NEON code.
- Engine CLI (`acutesim_engine_cli`) parses `.tran 10u` with `istringstream` (no SI suffixes).

## 7. Gotchas learned the hard way

- **Qt macros:** never name engine identifiers `slots`, `signals`, `emit`, `foreach` — engine headers are
  compiled into Qt targets (`test_std_cells` broke on a local `slots`).
- **Stale binaries:** `cmake --build build --target <x>` before trusting a result; a stale CLI once
  showed the pre-fix dense solver. New targets need `cmake -S . -B build -DFETCHCONTENT_UPDATES_DISCONNECTED=ON`.
- **macOS `sed` has no `\b`:** use `perl -pi -e`.
- **SPICE line 1 is the title** — test netlists need a title line or the first element vanishes.
- **No `-ffast-math`** (it deletes NaN checks; CLAUDE.md requires `-fno-fast-math`).
- **Absolute tolerances lie at stiff nodes:** use the scale-aware metrics (`kclExcess`, backward error),
  and justify any tolerance change in the commit.
- `graphify-out/` (code graph: `graphify update .`) is git-ignored; regenerate as needed.
- Commits: conventional prefix, explain *why*, end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
