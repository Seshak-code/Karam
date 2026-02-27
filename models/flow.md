# Models Directory Flow

The `models/` directory serves as the **Source of Truth** for user-defined and experimental physics models. Unlike the hardcoded standard cells in `components/`, these models are **Dynamic**, meaning they are compiled, loaded, and executed at runtime without recompiling the main application.

## 1. Supported Formats

We support two distinct formats for defining physics, both of which compile down to the same internal representation (IR).

### AcuteSim YAML (`.yaml`) — Canonical Format
The canonical, declarative model format. All models checked into `models/` use YAML. Verilog-A sources are converted to YAML during initial development and the `.va` originals are not retained.

### Verilog-A (`.va`) — Import Format
The industry-standard language for compact modeling. The Verilog-A frontend parses `.va` files into the same IR as YAML. Use this for importing third-party models, then maintain them as YAML going forward.

```verilog
module diode(p, n);
    inout p, n;
    electrical p, n;
    parameter real Is = 1e-14;

    analog begin
        I(p, n) <+ Is * (limexp(V(p, n) / $vt) - 1);
    end
endmodule
```

### YAML Example

```yaml
name: Diode
params:
  Is: 1e-14
equations:
  - "I_pn = Is * (exp(V_pn / 0.025) - 1)"
```

## 2. The Compilation Pipeline

Models in this directory are transformed into high-performance executable code by the **Physics Compiler** (`scripts/physics_compiler/`).

| Stage | Input | Tool | Output |
| :--- | :--- | :--- | :--- |
| **1. Parse** | `.va` / `.yaml` | `frontend.py` | **Intermediate Representation (AST)** |
| **2. Optimize** | AST | `ir.py` | **Optimized AST** (SSA form, Constant Folding) |
| **3. Codegen** | AST | `backends/scalar_cpp.py` | **C++ Source** (`diode.cpp`) |
| **4. Build** | C++ | `CMake` / `Clang` | **Shared Library** (`diode.dylib` / `.so`) |

## 3. Runtime Integration

Once compiled, the model interacts with the rest of the codebase as follows:

### Loading (`gui/widgets/component_library.cpp`)
*   The GUI scans the `models/` build artifacts.
*   It uses `dlopen` (via `ModelRegistry`) to load the shared library.
*   The `REGISTER_PHYSICS_MODEL` macro inside the generated C++ code self-registers the model with the singleton `ModelRegistry`.

### Instantiation (`gui/editors/schematic_editor.cpp`)
*   Users select the model from the **"Verilog-A"** tab in the library.
*   The `Generic` component type is used to represent it on the canvas.
*   The `subCircuitName` property stores the model name (e.g., "diode").
*   Pins are dynamically rendered based on the `ModelInfo` registered by the loaded library.

### Simulation (`compute/tensors/physics_tensors.cpp`)
*   During `tensorizeBlock`, these components are packed into `genericComponents` tensors.
*   In the solver loop (`physics/circuitsim.h`), `ModelRegistry::stampJacobian` is called.
*   This dispatches execution to the compiled C++ function in the shared library, which computes currents and Jacobian entries (Gmin, Gds, Gm) and stamps them into the system matrix.

## Development Rules

1.  **Statelessness**: Models defined here effectively describe *constitutive equations*. They should not attempt to store internal state variables (like "previous voltage") directly; instead, use time-derivative operators `ddt()` which the solver manages.
2.  **Continuity**: Functions used (exp, log, sqrt) must be continuous and differentiable. The compiler will automatically substitute safe variants (`limexp`, `soft_sqrt`) to prevent convergence failures.
3.  **Portability**: The generated C++ code is standard C++17 and should compile on macOS, Linux, and Windows without modification.
