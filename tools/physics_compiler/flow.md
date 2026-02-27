# Physics Compiler Architecture

The `scripts/physics_compiler` directory contains the **Cognitive Alchemist** — a Python-based compiler that transforms high-level Verilog-A models into GPU-optimised Tensor MNA shaders.

## Data Flow: Model Transmutation
1.  **Frontend**: The Verilog-A parser (Python) generates an Intermediate Representation (IR).
2.  **Symbolic Math**: `jacobian.py` calculates symbolic derivatives for the Jacobian matrix.
3.  **IR Optimization**: The IR is simplified for parallel execution.
4.  **Backend**: Code generators transform the IR into GLSL, C++, or OpenCL.

## Logic: Symbolic Differentiation
The compiler performs automatic differentiation on the model equations, ensuring that the engine always has exact Jacobians for Newton-Raphson convergence.

## 🤖 SME Validation Checklist
- [ ] **IR Invariance**: Does a change in a backend affect the IR structure? (Forbidden: Backends should be consumers only).
- [ ] **Symbolic Correctness**: Does the Jacobian calculation handle non-linearities (e.g., `exp()`, `log()`) correctly?
- [ ] **Performance**: Do the backends generate branchless code where possible?
