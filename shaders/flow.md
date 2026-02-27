# Physics Shader Architecture

The `compute/shaders` directory contains the **GPU Execution Kernels**.

## Logic: Parallel Physics
These GLSL files implement the actual MNA solver steps (Matrix Assembly, Pre-conditioners, Solvers) optimized for SIMT execution.

## 🤖 SME Validation Checklist
- [ ] **SIMD Optimization**: Does the shader avoid divergence?
- [ ] **Numeric Precision**: Is `highp` precision used for sensitive calculations?
