# Autodiff Module Context & Flow

## Overview
The `autodiff/` directory provides the **Mathematical Foundation** for the simulator's ability to compute Jacobians and Parameter Sensitivities mechanically, without symbolic math engines or manual derivative derivation. It implements **Forward-Mode Automatic Differentiation** using Dual Numbers.

## Key Files & Responsibilities

### 1. Dual Numbers (`dual.h`)
**Role**: The "Fundamental Atom". A lightweight value type that carries both a value ($f(x)$) and a gradient ($f'(x)$) through every calculation.
-   **Design**: Header-only, templated `Dual<T>` struct.
-   **Forward Mode**: Ideal for functions with few inputs and many outputs (or dense Jacobians where we seed one variable at a time).
-   **NASA Principle**: Control flow (`<`, `>`, `==`) is determined *solely* by the Primal Value. The gradient is a passenger.

## Data Flow
1.  **Seeding**: To find $\frac{\partial f}{\partial x}$, you instantiate `Dual<double> x(value, 1.0)`. The `1.0` seeds the derivative.
2.  **Propagation**: As `x` passes through operators (`+`, `*`, `sin`, `exp`), the chain rule is applied automatically to the `.grad` member.
3.  **Extraction**: The final result `y` contains `y.val` (the result of $f(x)$) and `y.grad` (the value of $f'(x)$).

## Guards & Patterns

### 1. Operator Overloading
We overload standard operators to enforce the Chain Rule.
```cpp
// Product Rule: (uv)' = u'v + uv'
constexpr Dual& operator*=(const Dual& rhs) {
    grad = grad * rhs.val + val * rhs.grad;
    val *= rhs.val;
    return *this;
}
```

### 2. Standard Library Specialization
We specialize `std::` math functions to allow `device_physics.h` to stay generic.
```cpp
namespace std {
    template <typename T>
    inline Dual<T> exp(const Dual<T>& x) {
        T ex = std::exp(x.val);
        // d/dx exp(u) = exp(u) * u'
        return { ex, ex * x.grad };
    }
}
```

### 3. Control Flow Grounding
Branching logic matches the physical simulation path.
```cpp
// The simulator will take the same branch whether calculating value or derivative
template <typename T>
constexpr bool operator<(const Dual<T>& lhs, const Dual<T>& rhs) { 
    return lhs.val < rhs.val; 
}
```

## Usage in Physics
*   **Scalar Computation**: `physics/device_physics.h` uses `T` templates.
*   **Jacobian Assembly**: `physics/circuitsim.h` instantiates these templates with `T = Dual<double>`.
*   **Performance**: Since `Dual` is a simple struct of two doubles, compilers can aggressively inline and vectorize these operations to SIMD registers (AVX-512 `vmovapd`).

## Development Rules
1.  **No Exceptions**: Derivatives must be safe. `std::sqrt` handles derivatives, but callers must ensure inputs are valid (handled by `physics` clamping).
2.  **Generic Kernels**: Code in `device_physics` must never explicitly cast to `double` if it breaks `Dual` propagation. Use simple literals (`1.0`, not `(double)1.0`) or `T(1.0)`.

## 🤖 SME Validation Checklist
*(Consult this list before modifying `autodiff/`)*

- [ ] **Dependency Check**: Does this change introduce any dependency on `physics/`? (Forbidden: Autodiff is a foundational dependency).
- [ ] **Dual Propagation**: Does the change strictly maintain the chain rule (`.grad` propagation)?
- [ ] **Control Flow**: Did you ensure that all `if/else` logic depends **only** on `.val` and never on `.grad`?
- [ ] **Std Namespace**: If specializing a new math function, is it inside `namespace std` to allow Argument-Dependent Lookup (ADL)?
- [ ] **Test Coverage**: Does `test_dual.cpp` cover the new operator/function?
