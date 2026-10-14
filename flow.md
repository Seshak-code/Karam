# Tensor MNA Engine Library

The `acutesim_engine/` directory is the **Standalone Simulation Library** — the core scientific kernel of AcuteSim, extracted and explicitly separated from all GUI and client code.

## Architectural Status: Complete (Post-Migration)
All engine source code has been migrated here from its former scattered locations. AcuteSim now **depends** on this engine as a library rather than owning it inline.

## Directory Structure

```
acutesim_engine/
├── autodiff/          ← Dual number AD foundation (zero deps)
├── math/              ← LinAlg, EmulatedF64, integration methods
├── netlist/           ← circuit.h, graph_canonical, netlist_export
├── tensors/           ← SoA layout of the linear elements
├── physics/           ← Device models, MNA stampers, statistical engine
├── infrastructure/    ← Netlist compiler, scheduler, trace, Monte Carlo
├── solvers/           ← DC/AC/Tran/PZ/TF/Fourier/WebGPU solvers
├── internal/          ← engine_pch.h, precision.h (never visible outside)
├── osdi/              ← OSDI loader: compiled Verilog-A devices (OpenVAF)
├── CMakeLists.txt     ← OBJECT → STATIC library target
├── engine_export.h    ← ENGINE_API / ENGINE_INTERNAL visibility macros
├── engine_impl.h      ← EngineImpl + SessionImpl (non-virtual hot path)
├── gpu_context_manager.h/cpp ← Dawn device pool singleton
└── flow.md            ← This file
```

## 🏛️ The Architecture Doctrine (Next-Gen SPICE)

To transition AcuteSim into a **next-generation mixed-signal simulation architecture**, all future work on the solvers, WebGPU dispatch, and netlist compiler must strictly adhere to the following principles.

### The Unbreakable SPICE Contract
The numerical system solved at every Newton iteration must be identically equivalent to canonical SPICE MNA. **You may change how equations are computed, but never what equations exist.**
1. **Assembly Equivalence**: The final assembled Jacobian $J(x)$ and Residual $F(x)$ matrix must be identical to canonical SPICE.
2. **Newton State Preservation**: All device evaluations in an iteration must see exactly the same state vector snapshot. No partial model updates via async execution.
3. **Time Integration Ownership**: Transient history (capacitors, inductors, Verilog-A states) must remain strictly owned by the solver timeline, not the scheduler.

### The 3-Layer Execution Model
1. **Layer 1: SPICE Semantic Engine (Stable Contract)**
   - Netlist parsing, device model semantics, and numeric control. **Never GPU-specific.**
2. **Layer 2: Compilation Layer (Tensor Network Builder)**
   - Transforms the semantic circuit graph into tensor blocks, factor graphs, and contraction DAGs.
   - **Factor Graphs & MPOs**: Converts circuits into semantic variable/factor nodes; devices become device-local Rank-$k$ dense Matrix Product Operators instead of scalar sparse matrix stamps.
   - **Contraction Scheduling**: Uses a cuTensorNet-style hyper-optimizer to find Contraction Tree DAGs for optimal dense evaluation.
   - **Bond Cutting**: Slices high-connectivity tensors to enable distributed multi-GPU boundaries.
   - **Virtual VLIW Scheduling**: Computes behavioral `if/else` gaps (Verilog-A) via predicated tensor math (`mask*A + (1-mask)*B`) to prevent GPU warp divergence.
3. **Layer 3: Execution Backend (AI Tensor Engine)**
   - **Mixed-Precision Iterative Refinement (IR)**: Maintains exact Emulated-F64 accuracy for the Residual $F(x)$ check and the final state updates, while delegating the dense Tensor operations to ultra-fast, lower-precision F16/TF32 hardware Tensor Cores.
   - Executes dense tensor contraction schedules (GEMM) rather than sparse matrix LU factorization.
   - Uses non-symmetric Krylov solvers (e.g., GMRES) for Schur Complement boundary assembly.

## Key Properties

| Property | Value |
|---|---|
| GPU | Dawn/WebGPU adapter probe only (`gpu_context_manager`); `gpu` mode solves on the CPU path (Phase 4) |
| Precision | f64 throughout |
| Linear solver | tensor-contraction direct solver (`math/contraction_solver`), fronts on worker threads |
| Devices | one definition per device kind, Jacobian by Dual numbers or OSDI |
| Visibility | `-fvisibility=hidden` — only ENGINE_API escapes |
| Side-effects | Zero Qt, zero GUI dependencies |

## Implementation status

The WebGPU Newton loop, its WGSL shaders, the SIMD device kernels and the graph partitioner described in
earlier versions of this file were deleted (they solved a voltage-only system that is not SPICE MNA, or
were unused). GPU work restarts in Phase 4 from the design principles above; see `ENGINE_HANDOFF.md` §0.

## 🤖 SME Validation Checklist
- [ ] **No Qt**: Does any header in this directory include a Qt type? (ABSOLUTELY FORBIDDEN)
- [ ] **No virtual in hot path**: Is there a virtual call inside the Newton-Raphson loop? (Forbidden)
- [ ] **GPU pool**: Is a new Dawn device being created per job? (Forbidden — use `GPUContextManager::instance()`)
- [ ] **Precision macro**: Are IEEE-critical functions (pnjlim, LTE) wrapped in `ACUTESIM_PRECISE_FP_BEGIN/END`?
- [ ] **Symbol visibility**: Are new public types declared with `ENGINE_API`? Internal types with `ENGINE_INTERNAL`?
- [ ] **API purity**: Does `engine_api/` include any header from `acutesim_engine/internal/`? (Forbidden — reverse firewall)
