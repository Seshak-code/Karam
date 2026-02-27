# Tensor MNA Engine Library

The `acutesim_engine/` directory is the **Standalone Simulation Library** — the core scientific kernel of AcuteSim, extracted and explicitly separated from all GUI and client code.

## Architectural Status: Complete (Post-Migration)
All engine source code has been migrated here from its former scattered locations. AcuteSim now **depends** on this engine as a library rather than owning it inline.

## Directory Structure

```
acutesim_engine/
├── autodiff/          ← Dual number AD foundation (zero deps)
├── math/              ← LinAlg, EmulatedF64, integration methods
├── models/            ← Device model YAML definitions
├── netlist/           ← circuit.h, graph_canonical, netlist_export
├── tensors/           ← SoA layout, SIMD kernel dispatch
├── physics/           ← Device models, MNA stampers, statistical engine
│   └── veriloga/      ← Full Verilog-A lexer → AST → assembler
├── infrastructure/    ← Netlist compiler, scheduler, trace, Monte Carlo
├── solvers/           ← DC/AC/Tran/PZ/TF/Fourier/WebGPU solvers
├── shaders/           ← WGSL GPU kernels (persistent NR loop)
├── internal/          ← engine_pch.h, precision.h (never visible outside)
├── tools/physics_compiler/ ← Python model compiler toolchain
├── CMakeLists.txt     ← OBJECT → STATIC library target
├── engine_export.h    ← ENGINE_API / ENGINE_INTERNAL visibility macros
├── engine_impl.h      ← EngineImpl + SessionImpl (non-virtual hot path)
├── gpu_context_manager.h/cpp ← Dawn device pool singleton
└── flow.md            ← This file
```

## Key Properties

| Property | Value |
|---|---|
| GPU Acceleration | WebGPU/Dawn, persistent 6-stage NR loop |
| Precision | f64 (CPU), EmulatedF64 hi-lo (GPU) |
| Sparsity | CSR matrix with PCG or pivoted LU |
| Parallelism | SoA layout + AVX-512/NEON + OpenMP batch |
| Visibility | `-fvisibility=hidden` — only ENGINE_API escapes |
| Side-effects | Zero Qt, zero GUI dependencies |

## 🤖 SME Validation Checklist
- [ ] **No Qt**: Does any header in this directory include a Qt type? (ABSOLUTELY FORBIDDEN)
- [ ] **No virtual in hot path**: Is there a virtual call inside the Newton-Raphson loop? (Forbidden)
- [ ] **GPU pool**: Is a new Dawn device being created per job? (Forbidden — use `GPUContextManager::instance()`)
- [ ] **Precision macro**: Are IEEE-critical functions (pnjlim, LTE) wrapped in `ACUTESIM_PRECISE_FP_BEGIN/END`?
- [ ] **Symbol visibility**: Are new public types declared with `ENGINE_API`? Internal types with `ENGINE_INTERNAL`?
- [ ] **API purity**: Does `engine_api/` include any header from `acutesim_engine/internal/`? (Forbidden — reverse firewall)
