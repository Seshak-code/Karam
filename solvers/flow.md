# Solvers Flow

The `compute/solvers/` directory contains the **High-Level Algorithms** that drive the simulation.
While `compute/math` provides the tools (matrices, integrators), `compute/solvers` provides the *strategy* for using them to solve physics problems.

## Key Files & Responsibilities

### 1. `circuitsim_solver.cpp`: The Master Solver
Implementing `solveDC` and `stepTransient`, this file orchestrates the entire non-linear solution loop.
*   **DC Operating Point**:
    *   **Seeding**: Intelligent initial guesses (diode drops, BJT heuristics) to help convergence.
    *   **GMIN Stepping**: If Newton-Raphson fails, it adds large parallel conductances (GMIN) and slowly removes them to guide the solver to a solution.
    *   **Backend Selection**: Automatically chooses `solveLU_Pivoted` (Robust, O(N^3)) for active circuits or `solvePCG` (Fast, O(N)) for large passive meshes.
    *   **Convergence Checks**: Verifies both voltage stability (`delta_v`) and physical truth (`KCL Residuals`).
*   **Transient Analysis**:
    *   **Adaptive Stepping**: Uses `analyzeStability` (from `infrastructure`) to cut time steps if LTE (Local Truncation Error) is high.
    *   **Integration Arbiter**: Switches between Trapezoidal and Gear2 methods based on ringing detection.

### 2. `superposition.h`: The Linear Decomposer
Implements the **Superposition Principle** for linear circuits.
*   **Usage**: Used for educational visualizations ("Show contribution of Source A") or small-signal analysis.
*   **Method**: Solves $N$ sub-circuits where only one source is active at a time, then sums the results.

### 3. `webgpu_solver.h`: The Dawn Hardware Accelerator
A high-performance backend using **Google Dawn** to run the Newton-Raphson loop entirely on the GPU via WGSL.
*   **Architecture**: Uses a persistent **6-stage NR kernel** (Batch Physics, Assemble Jacobian, Linear Solve, Update Solution, Residual Norm, Convergence Check).
*   **Precision**: Implements **EmulatedF64** (hi-lo `f32` pairs) to maintain SPICE-level precision (15-17 digits) without native `f64` support.
*   **Integration**: Fully integrated with the **Structure-of-Arrays (SoA)** physics tensors for coalesced memory access.

## Core Logic & Snippets

### Initial Guess Seeding (`circuitsim_solver.cpp`)
We don't start at 0V. We guess standard pn-junction drops to save iterations.

```cpp
void CircuitSim::seedInitialGuess(..., vector<double>& estimate) {
    // Seed Diode Anodes
    for (const auto& diode : diodes) {
        if (diode.anode >= 0) estimate[diode.anode] = 0.6; // V_DIODE_SEED
    }
    // Seed BJT Bases (Propagate from Emitter)
    for (const auto& bjt : bjts) {
        // ... logic to set Base = Emitter + 0.7V ...
    }
}
```

### Automatic Backend Selection (`circuitsim_solver.cpp`)
The solver is smart enough to know that BJTs make the matrix non-symmetric, ruling out PCG.

```cpp
bool hasActiveDevices = !bjts.empty() || !diodes.empty() || !mosfets.empty();

if (hasActiveDevices || numNodes < 20) {
    // Direct LU: Robust logic for non-linear/small circuits
    result = solveLU_Pivoted(matrix, rhs);
} else {
    // PCG: Fast logic for large parasitic grids
    result = solvePCG(matrix, rhs);
    if (!result.converged) result = solveLU_Pivoted(matrix, rhs); // Fallback
}
```

## Development Rules

1.  **Convergence is Physical**: Never trust `delta_v < tolerance` alone. Always check `KCL Residual < 1e-12` (1pA). A solver can "converge" to a non-physical state if the matrix is ill-conditioned.
2.  **GMIN is Mandatory**: All non-linear DC solves must rely on GMIN stepping as a fallback. Cold-starting a BJT feedback loop without it is mathematically impossible often.
3.  **State Hygiene**: The solver must be stateless between calls (except for `prev_voltage` in transient). Do not store "solver state" in the `TensorNetlist`.
4.  **GPU Abstraction**: If modifying the solver loop, ensure `webgpu_solver.h` logic (Dawn/WGSL) is updated or at least not broken.

## 🤖 SME Validation Checklist
*(Consult this list before modifying `compute/solvers/`)*

- [ ] **Physical Convergence**: Are you checking `KCL Residuals` (Amps) and not just `Delta V`?
- [ ] **Robustness Fallback**: If using PCG/Iterative methods, is there a `GMIN` or `LU` fallback path?
- [ ] **Statelessness**: Are you storing simulation state (e.g. `prev_voltage`) inside the Solver class instead of the `BlockState`? (Forbidden).
- [ ] **GPU Parity**: If you changed the solver algorithm, did you update/verify `webgpu_solver.h`?
