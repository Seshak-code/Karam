# Physics Module Context & Flow

## Overview
The `physics/` directory contains the core numerical engine of CuteApp. It is designed as a "Dual Kernel" architecture where **Physical Truth** (Exact constitutive equations) is separated from **Numerical Stability** (Linearization/Clamping). The engine uses a **Structure-of-Arrays (SoA)** data layout to enable high-performance **GPU acceleration via Google Dawn**, utilizing a persistent 6-stage NR iteration loop.

## Key Files & Responsibilities

### 1. Constitutive Equations (`device_physics.h`)
**Role**: The "Source of Truth". Contains pure, stateless C++ functions defining low-level device behavior.
-   **Design**: Templated for `double` (fast scalar) and `Dual<double>` (Automatic Differentiation).
-   **Key Concepts**:
    -   **Unified Diode Model**: `diode_model_pair` unifies logic for discrete Diodes and BJT junctions, returning `{I, G}` pairs.
    
    ```cpp
    // physics/device_physics.h
    template <typename T, typename P>
    inline std::pair<T, T> diode_model_pair(T vd, const DiodeParams<P>& p, double max_arg = 30.0);

    // Legacy Helpers (Wrappers around diode_model_pair)
    template <typename T, typename P>
    inline T diode_current_clamped(T vd, const DiodeParams<P>& p, double max_arg = 30.0);
    ```

    -   **PNJLIM**: SPICE-style voltage limiting logic (`pnjlim`) to dampen Newton-Raphson oscillations.
    
    ```cpp
    // physics/device_physics.h
    inline double pnjlim(double vnew, double vold, double vt, double vcrit) {
        if (vnew > vcrit && std::abs(vnew - vold) > 2.0 * vt) {
            // ... strict limiting logic ...
            vlim = vold + vt * std::log(arg);
        }
        return vlim;
    }
    ```

### 2. Matrix Assembly (`circuitsim.h` Stampers)
**Role**: The "Mapper". Translates physical quantities (Current, Conductance) into Linear Algebra terms (Matrix entries, RHS vector).
-   **Pattern**: MNA (Modified Nodal Analysis) stamping via specialized helpers.
-   **Design**: Reusable stamper functions (`stampResistor`, `stampDiode`, `stampBJT`) encapsulate the physics mapping, allowing the main solver loop to remain clean and extensible.
    
    ```cpp
    // physics/circuitsim.h
    inline void stampResistor(...) {
        double g = 1.0 / r_val;
        matrixBuilder.add(n1, n1, g);
        matrixBuilder.add(n2, n2, g);
        matrixBuilder.add(n1, n2, -g);
        matrixBuilder.add(n2, n1, -g);
    }
    ```

    - **Generic Models**: Dispatches to `ModelRegistry::stampJacobian`, which invokes the compiled model's `stamp` function (loaded from shared library).

### 3. Solver Engine (`circuitsim.h`)
**Role**: The "Orchestrator". Manages the time-step evolution and non-linear solution search.
-   **IntegratorArbiter**: Monitors stability via LTE and Lyapunov Energy checks.

    ```cpp
    // physics/circuitsim.h :: IntegratorArbiter
    int analyzeStability(..., double dt, double &next_dt_suggestion) {
        double lte = calculateLTE(netlist, current_voltages, dt);
        
        // M4: Lyapunov Energy Guard
        bool energy_exploding = (current_energy > prev_energy + energy_tolerance) ...;
        
        if (high_lte || energy_exploding) {
             next_dt_suggestion = dt * 0.5; // Cut step
             return 1; // Retry
        }
        return 0; // Accept
    }
    ```

-   **Audit**: `calculatePhysicalResiduals` checks KCL conservation using EXACT physics (no clamping).

    ```cpp
    // physics/circuitsim.h
    double calculatePhysicalResiduals(...) {
        for (const auto &d : netlist.globalBlock.diodes) {
            // Sum currents entering/leaving nodes
            addRes(d.anode, -i_d);
            addRes(d.cathode, i_d);
        }
        // Returns max absolute KCL violation (Amps)
    }
    ```

### 4. Configuration (`simulation_profile.h`)
**Role**: The "Contract". Defines what a simulation run looks like.

    ```cpp
    // physics/simulation_profile.h
    struct SimulationProfile {
        AnalysisType analysis = AnalysisType::DC;
        size_t topologyHash = 0; // Links to unique netlist state
        
        // Explicit constraints for valid run
        bool requiresDC = false;
        bool requiresERC = true; 
    };
    ```

## Data Flow
1.  **Input**: `TensorNetlist` (State + Definition). `BlockState` initialized per instance.
2.  **Loop**:
    -   `CircuitSim::stepTransient` calls `IntegratorArbiter`.
    -   If stable, `matrixBuilder` assembles Jacobian via `stamps.h` (Parallel/Batch).
        -   **Global Block**: Stamped via `stampSoABlock` (Optimized AVX-512).
        -   **Hierarchical Instances**: Stamped via `stampBlockLinear`/`stampBlockNonLinear` loop (Scalar).
    -   Solver computes $\Delta V$.
    -   **State Update**: `BlockState` histories (inductors/capacitors) updated.
    -   `calculatePhysicalResiduals` validates the result against physics.
3.  **Output**: `SimulationTrace` containing time-series voltages.

## Development Rules
1.  **Stateless Physics**: Never introduce state variables (like `history`) into `device_physics.h` kernels.
2.  **Explicit Clamping**: All exponentials MUST have a `_clamped` variant.
3.  **Generic Math (ADL)**: Do NOT use `std::exp` or `std::sin` directly in device kernels. Use unqualified `exp()` to allow Argument Dependent Lookup (ADL) to find `acutesim::exp(Dual)` or defaults.

## 🤖 SME Validation Checklist
*(Consult this list before modifying `physics/`)*

- [ ] **Stateless Physics**: Did you introduce any state variables into `device_physics.h`? (Forbidden: Must remains pure functional).
- [ ] **Explicit Clamping**: Do all new exponentials/logarithms have a `_clamped` variant or safety guard?
- [ ] **ADL Compatibility**: Are you using `std::exp` instead of `exp` in templated kernels? (Forbidden: Breaks Dual number support).
- [ ] **Conservation of Charge**: Have you verified that `calculatePhysicalResiduals` accounts for the new physics terms?
- [ ] **Tensor Parity**: If you modified `TensorBlock` (AoS), did you also update `TensorizedBlock` (SoA) in `physics_tensors.h`?
