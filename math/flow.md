# Compute Math Flow

The `compute/math/` directory contains the **Numerical Kernel** of AcuteSim. It provides the fundamental algebra and calculus operations required to solve circuit equations.

It is designed to be:
1.  **Sparse-Optimized**: Using Compressed Sparse Row (CSR) storage for memory efficiency.
2.  **Parallel-Ready**: Leveraging OpenMP for vector operations and Matrix-Vector Multiplication (SpMV).
3.  **Method-Agnostic**: Using Strategy Patterns to swap integration methods (Trapezoidal vs Gear) dynamically.

## Key Files & Responsibilities

### 1. `linalg.h`: The Linear Algebra Engines
Implements the matrix storage and solvers for `Ax = b`.
*   **Storage**: `Csr_matrix` (Compressed Sparse Row) and `MatrixConstructor` (Triplet-based assembly).
*   **Lock-Free Assembly**: Uses **Thread-Local Storage (TLS)** buffers to allow multiple threads to stamp simultaneously without mutex contention.
*   **Batch Interface**: `addBatch` allows efficiently stamping vector results from physics kernels directly.
*   **Iterative Solver**: `solvePCG` (Preconditioned Conjugate Gradient). Best for large, symmetric, diagonally dominant matrices (e.g., pure resistor/mesh grids). uses Jacobi Preconditioning.
*   **Direct Solver**: `solveLU_Pivoted` (Dense LU with Partial Pivoting). The fallback for "nasty" matrices (non-symmetric, active components, singular diagonals).
*   **Parallelism**: Uses `#pragma omp parallel for` on SpMV, vector adds, and matrix assembly merging.

### 2. `integration_method.h`: The Time-Stepping Strategy
Defines *how* differential equations (C dv/dt = I) are discretized into algebraic equations.
*   **Interface**: `IIntegrationMethod` defines `stampCapacitor` and `predict`.
*   **Trapezoidal**: Default for accuracy. Inherently stable (A-stable) but can oscillate on stiff ringing.
*   **Gear-1 (Backward Euler)**: Maximum stability, introduces artificial damping. Used for initial convergence.
*   **Gear-2 (BDF2)**: Good compromise between accuracy and stability for transient simulations.

### 3. `emulated_f64.h`: Precision Verification
Defines `struct EmulatedF64` (float pair) to mock GPU double-double arithmetic.
*   **Purpose**: Validates that physics kernels are strictly templated and ADL-compliant.
*   **Usage**: Used in `test_emulated_f64` and `test_bjt_stability` to prove architecture independence.

### 4. `fpu_control.h`: Determinism Enforcer
Ensures cross-platform identical results by locking down the Floating Point Unit.
*   **Rounding**: Enforces `FE_TONEAREST` (Round to Nearest Even).
*   **Denormals**: Disables "Flush to Zero" (DAZ/FTZ) to preserve subnormal precision (~1e-308).
*   **Safety**: Prevents x87 precision changes or "Fast Math" optimizations from breaking reproducibility.

## Core Logic & Snippets

### Matrix Assembly (`linalg.h`)
Circuit stamps are collected into thread-local buffers (arrays of Triplets). `createCsr` merges these buffers in parallel to form the final compressed matrix.

```cpp
Csr_matrix createCsr() {
    // 1. Parallel Merge of Thread Buffers
    // Calculate offsets and copy to a single large buffer
    #pragma omp parallel for
    for (int t = 0; t < num_threads; ++t) {
        std::copy(thread_buffers[t].begin(), thread_buffers[t].end(), 
                  merged.begin() + offset[t]);
    }
    
    // 2. Sort Triplets
    // 3. Deduplicate (Sum KCL)
    // 4. Generate Row Pointers
}

// Lock-Free Stamping
void add(int row, int col, double value) {
    int tid = omp_get_thread_num();
    thread_buffers[tid].push_back({row, col, value});
}
```

### Integration Strategy (`integration_method.h`)
The solver doesn't know *which* method is active; it just calls `stampCapacitor`.

```cpp
void TrapezoidalIntegration::stampCapacitor(..., MatrixConstructor& mat, std::vector<double>& rhs) {
    // I = C * (Vn - Vn-1)/dt  ->  Discretized via Trapezoidal Rule
    double G_eq = (2.0 * C) / timeStep;
    double I_eq = -G_eq * v_prev - i_prev; // Companion Model Source
    
    // Stamp Companion Conducance (G_eq)
    mat.add(nodeI, nodeI, G_eq); 
    
    // Stamp Companion Current Source (I_eq)
    rhs[nodeI - 1] -= I_eq;
}
```

### Solver Selection (`linalg.h`)
The system tries PCG first for speed, then falls back to LU if convergence fails.

```cpp
// PCG relies on M^-1 = 1/diag(A) preconditioner
std::vector<double> getDiagonalInverse(const Csr_matrix &matrix) {
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        // ... find diagonal element ...
        if (abs(val) > 1e-18) diagInv[i] = 1.0 / val;
    }
}
```

## Development Rules

1.  **Thread Safety**: Any function in `linalg.h` marked with OMP must operate on disjoint data or use reductions.
2.  **Singularity Checks**: Solvers must return `result.converged = false` rather than crashing on singular matrices (floating nodes).
3.  **Dense Fallback**: Always maintain `convertCsrToDense` and `solveLU_Pivoted` as a gold-standard reference, even if PCG is preferred.

## 🤖 SME Validation Checklist
*(Consult this list before modifying `compute/math/`)*

- [ ] **Zero Allocations**: Are you allocating memory (std::vector, new) inside a loop? (Forbidden: Use `reserve()` or pre-allocated buffers).
- [ ] **Thread Safety**: Is any shared state modified inside an OMP region without efficient locking or reduction?
- [ ] **Precision**: Did you respect `emulated_f64` rules? (No fast-math assumptions).
- [ ] **Singularity Safety**: Does the solver code handle singular matrices/NaNs gracefully via return status (not crash)?
- [ ] **Dense Fallback**: If modifying iterative solvers, did you ensure `solveLU` remains the gold-standard fallback?
