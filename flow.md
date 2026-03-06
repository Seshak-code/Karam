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
| GPU Acceleration | WebGPU/Dawn, persistent 6-stage NR loop |
| Precision | f64 (CPU), EmulatedF64 hi-lo (GPU) |
| Sparsity | CSR matrix with PCG or pivoted LU |
| Parallelism | SoA layout + AVX-512/NEON + OpenMP batch |
| Visibility | `-fvisibility=hidden` — only ENGINE_API escapes |
| Side-effects | Zero Qt, zero GUI dependencies |

## Implementation Status: Mixed-Signal Acceleration Pillars

| Pillar | Status | Files |
|--------|--------|-------|
| **Pillar 1: SPICE Contract** (14/14 parity) | ✅ Complete | `solvers/webgpu_solver.cpp`, `shaders/gpu_nr_loop.wgsl` |
| **Pillar 2: Virtual VLIW** (predicated WGSL) | ✅ Complete (Phase A) | `shaders/gpu_nr_loop.wgsl` — `f64_select()`, predicated `f64_exp`, MOSFET region select, voltage gathers |
| **Pillar 4: Static Routing** (workgroup cache) | ✅ Complete (Phase B) | `solvers/webgpu_solver.h/cpp` — `VoltageRouteBuffer`; `shaders/gpu_nr_loop.wgsl` — `var<workgroup>` cache + `workgroupBarrier()` |
| **Pillar 3: Graph Partitioning** (SM-local subgraphs) | ✅ Complete (Phase C) | `infrastructure/graph_partitioner.h/cpp` — BFS partitioner; `infrastructure/compiled_block.h` — `partitions` field; `infrastructure/netlist_compiler.cpp` — post-tensorize call |

### Phase A — Virtual VLIW Changes (gpu_nr_loop.wgsl)
- Added `f64_select(false_val, true_val, cond)` — component-wise `select()` for EmulatedF64 structs.
- Rewrote `f64_exp` to be fully predicated: all three paths (underflow / linear continuation / Horner polynomial) are always computed; `f64_select()` picks the correct result. No `return` inside `if`.
- Replaced voltage-gather `if (n > 0)` branches in Diode/MOSFET/BJT kernels with `select(0.0, voltages_hi[max(0,n-1)], n>0)`.
- Replaced `if (pmos != 0u)` / `if (isNPN == 0u)` sign selection with `select()`.
- Replaced MOSFET `if/else if/else` region selection with eager evaluation of all three regions + two-stage `f64_select(f64_select(sat, lin, in_linear), cut, in_cutoff)`.

### Phase B — Static Routing Changes
- `webgpu_solver.h`: Added `diodeVoltageRouteBuffer`, `mosfetVoltageRouteBuffer`, `bjtVoltageRouteBuffer`.
- `webgpu_solver.cpp`: Allocated route buffers in `setupResources()`; populated in `buildStampMaps()` alongside stamp maps.
- `gpu_nr_loop.wgsl`: Added `@group(1) @binding(7..9)` for route arrays; `var<workgroup>` caches (128/256/192 f32 slots); each physics kernel now preloads terminal voltages into shared memory then reads from cache.

### Phase C — Graph Partitioning Changes
- `infrastructure/graph_partitioner.h/cpp`: `GraphPartitioner::partition()` builds node-to-device adjacency, device-neighbour graph, BFS-seeds partitions from highest-degree device, greedy-expands to `devicesPerWorkgroup` limit.
- `infrastructure/compiled_block.h`: `std::vector<TensorPartition> partitions` field.
- `infrastructure/netlist_compiler.cpp`: Calls `GraphPartitioner::partition()` after `tensorizeNetlist()`.
- `tests/test_graph_partitioner.cpp`: 6-test coverage (empty, chain, 200 independent, mixed types, exact-64, nodeSet correctness).

## 🤖 SME Validation Checklist
- [ ] **No Qt**: Does any header in this directory include a Qt type? (ABSOLUTELY FORBIDDEN)
- [ ] **No virtual in hot path**: Is there a virtual call inside the Newton-Raphson loop? (Forbidden)
- [ ] **GPU pool**: Is a new Dawn device being created per job? (Forbidden — use `GPUContextManager::instance()`)
- [ ] **Precision macro**: Are IEEE-critical functions (pnjlim, LTE) wrapped in `ACUTESIM_PRECISE_FP_BEGIN/END`?
- [ ] **Symbol visibility**: Are new public types declared with `ENGINE_API`? Internal types with `ENGINE_INTERNAL`?
- [ ] **API purity**: Does `engine_api/` include any header from `acutesim_engine/internal/`? (Forbidden — reverse firewall)
