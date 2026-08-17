# Physics Compiler Backends

The `scripts/physics_compiler/backends` directory contains **Code Generators** that transform the Intermediate Representation (IR) into target-specific executable code.

## Logic: Target Optimization
Backends optimize the mathematical equations for specific execution environments:
-   **GLSL/SPIR-V**: For GPU acceleration.
-   **C++**: For CPU-based multi-threaded simulation.
-   **OpenCL**: For heterogeneous platform support.

## 🤖 SME Validation Checklist
- [ ] **Numerical Parity**: Do different backends produce the same numerical result for the same IR?
- [ ] **Register Efficiency**: Is the generated code optimized for the target's register/cache architecture?
