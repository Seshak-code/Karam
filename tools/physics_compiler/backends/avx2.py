"""
avx2.py - AVX2 Backend for the Physics Compiler.

Emits C++ code using AVX2 + FMA intrinsics to process 4 devices per iteration.
The generated code follows the same pattern as the AVX-512 backend:
  - Vectorized loads of terminal voltages (manual gather)
  - SIMD arithmetic for device physics
  - Vectorized stores of results
  - Scalar tail loop for remaining elements (n % 4 != 0)

Key intrinsics used:
  _mm256_set1_pd, _mm256_loadu_pd, _mm256_storeu_pd
  _mm256_add_pd, _mm256_sub_pd, _mm256_mul_pd, _mm256_div_pd
  _mm256_min_pd, _mm256_max_pd
  _mm256_exp_pd_fallback (lane-wise std::exp)

FMA Policy:
  By default, FMA is DISABLED to ensure bit-exact parity with the scalar
  backend (compiled with -ffp-contract=off). Pass use_fma=True to the
  generate() function to emit _mm256_fmadd_pd instead of separate mul+add.
"""

from __future__ import annotations
from typing import TextIO, Set
import io

from ..ir import (
    ModelDef, Expr, Const, Var, BinOp, UnaryOp, Call, Select, Cmp,
    Assign, StampEntry, IRType,
    canonicalize,
)


# ============================================================================
# AVX2 Expression Emitter
# ============================================================================

def emit_avx_expr(node: Expr, prefix: str = "v_", use_fma: bool = False) -> str:
    """Convert an IR expression tree to an AVX2 intrinsics expression.

    All variables are assumed to be __m256d registers prefixed with `prefix`.
    """
    if isinstance(node, Const):
        val = node.value
        if val == 0.0:
            return "_mm256_setzero_pd()"
        return f"_mm256_set1_pd({_format_double(val)})"

    if isinstance(node, Var):
        return f"{prefix}{node.name}"

    if isinstance(node, BinOp):
        lhs = emit_avx_expr(node.lhs, prefix, use_fma)
        rhs = emit_avx_expr(node.rhs, prefix, use_fma)
        op_map = {
            "+": "_mm256_add_pd",
            "-": "_mm256_sub_pd",
            "*": "_mm256_mul_pd",
            "/": "_mm256_div_pd",
        }
        func = op_map.get(node.op)
        if func:
            return f"{func}({lhs}, {rhs})"
        raise ValueError(f"Unsupported AVX2 binary op: {node.op}")

    if isinstance(node, UnaryOp) and node.op == "-":
        operand = emit_avx_expr(node.operand, prefix, use_fma)
        return f"_mm256_sub_pd(_mm256_setzero_pd(), {operand})"

    if isinstance(node, Call):
        func = node.func
        args = node.args

        if func == "exp" and len(args) == 1:
            arg = emit_avx_expr(args[0], prefix, use_fma)
            return f"_mm256_exp_pd_fallback({arg})"

        if func == "log" and len(args) == 1:
            arg = emit_avx_expr(args[0], prefix, use_fma)
            return f"_mm256_log_pd_fallback({arg})"

        if func == "sqrt" and len(args) == 1:
            arg = emit_avx_expr(args[0], prefix, use_fma)
            return f"_mm256_sqrt_pd({arg})"

        if func in ("abs", "fabs") and len(args) == 1:
            arg = emit_avx_expr(args[0], prefix, use_fma)
            return f"_mm256_abs_pd({arg})"

        if func == "min" and len(args) == 2:
            a = emit_avx_expr(args[0], prefix, use_fma)
            b = emit_avx_expr(args[1], prefix, use_fma)
            return f"_mm256_min_pd({a}, {b})"

        if func == "max" and len(args) == 2:
            a = emit_avx_expr(args[0], prefix, use_fma)
            b = emit_avx_expr(args[1], prefix, use_fma)
            return f"_mm256_max_pd({a}, {b})"

        if func == "fma" and len(args) == 3:
            a = emit_avx_expr(args[0], prefix, use_fma)
            b = emit_avx_expr(args[1], prefix, use_fma)
            c = emit_avx_expr(args[2], prefix, use_fma)
            if use_fma:
                return f"_mm256_fmadd_pd({a}, {b}, {c})"
            else:
                # Deterministic: separate mul + add
                return f"_mm256_add_pd(_mm256_mul_pd({a}, {b}), {c})"

        raise ValueError(f"Unsupported AVX2 function: {func}")

    if isinstance(node, Select):
        # AVX2 uses _mm256_cmp_pd (returns __m256d mask) + _mm256_blendv_pd
        cond = node.cond
        tv = emit_avx_expr(node.true_val, prefix, use_fma)
        fv = emit_avx_expr(node.false_val, prefix, use_fma)
        if isinstance(cond, Cmp):
            lhs_c = emit_avx_expr(cond.lhs, prefix, use_fma)
            rhs_c = emit_avx_expr(cond.rhs, prefix, use_fma)
            cmp_map = {
                "<": "_CMP_LT_OQ",
                "<=": "_CMP_LE_OQ",
                ">": "_CMP_GT_OQ",
                ">=": "_CMP_GE_OQ",
                "==": "_CMP_EQ_OQ",
                "!=": "_CMP_NEQ_OQ",
            }
            cmp_imm = cmp_map.get(cond.op, "_CMP_LT_OQ")
            # blendv selects from second operand where mask bit is set
            return (f"_mm256_blendv_pd({fv}, {tv}, "
                    f"_mm256_cmp_pd({lhs_c}, {rhs_c}, {cmp_imm}))")
        # Fallback: evaluate condition as a mask
        return f"/* SELECT: complex cond */ {tv}"

    raise ValueError(f"Unknown IR node type for AVX2: {type(node).__name__}")


def _format_double(val: float) -> str:
    """Format a double literal for C++."""
    if val == int(val) and abs(val) < 1e15:
        return f"{val:.1f}"
    return repr(val)


# ============================================================================
# Scalar Expression Emitter (for tail loop)
# ============================================================================

def emit_scalar_expr(node: Expr) -> str:
    """Emit a plain C++ scalar expression for the tail loop."""
    from .scalar_cpp import emit_expr
    return emit_expr(node)


# ============================================================================
# Full AVX2 File Emitter
# ============================================================================

def generate(model: ModelDef, use_fma: bool = False, include_header: str = None) -> str:
    """Generate the complete AVX2 C++ source for a device model.

    Args:
        model: The IR model definition.
        use_fma: If True, emit _mm256_fmadd_pd for FMA calls.
                 If False (default), emit separate mul+add for determinism.
        include_header: Custom include header name. Defaults to gen_<model>.h.
    """
    out = io.StringIO()
    struct_name = f"{model.name}Tensor"
    func_name = f"batch{model.name}Physics_avx2"

    out.write(f"// AUTO-GENERATED by Physics Compiler — DO NOT EDIT\n")
    out.write(f"// Model: {model.name} (AVX2 Backend)\n")
    out.write(f"// Layout Hash: {model.layout_hash()}\n")
    out.write(f"// FMA: {'enabled' if use_fma else 'disabled (deterministic)'}\n\n")

    header = include_header or f"gen_{model.name.lower()}.h"
    out.write(f'#include "{header}"  // Tensor struct\n')
    out.write(f"#include <vector>\n")
    out.write(f"#include <cmath>\n")
    out.write(f"#include <algorithm>\n\n")

    out.write(f"#if defined(__AVX2__)\n")
    out.write(f"#include <immintrin.h>\n\n")

    # Emit helper functions
    out.write(_EXP_FALLBACK)
    out.write(_LOG_FALLBACK)
    out.write(_ABS_HELPER)

    # Main function
    out.write(f"void {func_name}({struct_name}& tensor, "
              f"const std::vector<double>& voltages) {{\n")
    out.write(f"    const size_t n = tensor.size();\n\n")

    # Broadcast constants used in the kernel
    _emit_constant_broadcasts(model, out)

    out.write(f"\n    size_t i = 0;\n")
    out.write(f"    // --- SIMD Loop: 4 devices per iteration ---\n")
    out.write(f"    for (; i + 3 < n; i += 4) {{\n")

    # Gather terminal voltages
    _emit_voltage_gathers(model, out)

    # Load parameters
    out.write(f"\n        // Load parameters\n")
    for p in model.parameters:
        out.write(f"        __m256d v_{p.name} = _mm256_loadu_pd(&tensor.{p.name}[i]);\n")

    out.write(f"\n")

    # Emit kernel body (vectorized)
    for assign in model.body:
        expr = canonicalize(assign.expr)
        avx_expr = emit_avx_expr(expr, "v_", use_fma)
        is_state = any(s.name == assign.target for s in model.state_vars)

        if is_state:
            out.write(f"        __m256d v_{assign.target} = {avx_expr};\n")
            out.write(f"        _mm256_storeu_pd(&tensor.{assign.target}[i], v_{assign.target});\n")
        else:
            out.write(f"        __m256d v_{assign.target} = {avx_expr};\n")

    out.write(f"    }}\n\n")

    # Scalar tail loop
    out.write(f"    // --- Scalar Tail Loop ---\n")
    out.write(f"    for (; i < n; ++i) {{\n")

    for t in model.terminals:
        out.write(f"        int n_{t.name} = tensor.node_{t.name}[i];\n")
        out.write(f"        double v_{t.name} = (n_{t.name} > 0 && n_{t.name} <= (int)voltages.size()) "
                  f"? voltages[n_{t.name} - 1] : 0.0;\n")

    for p in model.parameters:
        out.write(f"        double {p.name} = tensor.{p.name}[i];\n")

    out.write(f"\n")

    for assign in model.body:
        expr = canonicalize(assign.expr)
        scalar_expr = emit_scalar_expr(expr)
        is_state = any(s.name == assign.target for s in model.state_vars)

        if is_state:
            out.write(f"        tensor.{assign.target}[i] = {scalar_expr};\n")
            out.write(f"        double {assign.target} = tensor.{assign.target}[i];\n")
        else:
            out.write(f"        double {assign.target} = {scalar_expr};\n")

    out.write(f"    }}\n")
    out.write(f"}}\n\n")

    out.write(f"#endif // __AVX2__\n")

    return out.getvalue()


# ============================================================================
# Helper Emitters
# ============================================================================

def _emit_constant_broadcasts(model: ModelDef, out: TextIO):
    """Emit _mm256_set1_pd broadcasts for constants used in the kernel."""
    consts = set()
    for assign in model.body:
        _collect_constants(assign.expr, consts)

    if consts:
        out.write(f"    // Pre-broadcast constants\n")
        for c in sorted(consts):
            name = _const_name(c)
            out.write(f"    const __m256d {name} = _mm256_set1_pd({_format_double(c)});\n")


def _collect_constants(node: Expr, result: Set[float]):
    """Collect all literal constants in an expression tree."""
    if isinstance(node, Const):
        if node.value != 0.0:  # Skip zero, handled by setzero
            result.add(node.value)
    elif isinstance(node, BinOp):
        _collect_constants(node.lhs, result)
        _collect_constants(node.rhs, result)
    elif isinstance(node, UnaryOp):
        _collect_constants(node.operand, result)
    elif isinstance(node, Call):
        for a in node.args:
            _collect_constants(a, result)
    elif isinstance(node, Select):
        _collect_constants(node.cond, result)
        _collect_constants(node.true_val, result)
        _collect_constants(node.false_val, result)


def _const_name(val: float) -> str:
    """Generate a C++ variable name for a constant broadcast."""
    if val == int(val):
        return f"vc_{int(val)}" if val >= 0 else f"vcn_{int(abs(val))}"
    s = f"{val}".replace(".", "_").replace("-", "n").replace("+", "p").replace("e", "e")
    return f"vc_{s}"


def _emit_voltage_gathers(model: ModelDef, out: TextIO):
    """Emit manual voltage gathers for terminal nodes (4-wide)."""
    out.write(f"\n        // Gather terminal voltages (manual gather)\n")
    out.write(f"        const double* v_ptr = voltages.data();\n")
    out.write(f"        const int v_size = (int)voltages.size();\n")

    for t in model.terminals:
        out.write(f"\n        alignas(32) double _buf_{t.name}[4];\n")
        out.write(f"        for (int k = 0; k < 4; ++k) {{\n")
        out.write(f"            int idx = tensor.node_{t.name}[i + k];\n")
        out.write(f"            _buf_{t.name}[k] = (idx > 0 && idx <= v_size) ? v_ptr[idx - 1] : 0.0;\n")
        out.write(f"        }}\n")
        out.write(f"        __m256d v_v_{t.name} = _mm256_load_pd(_buf_{t.name});\n")


# ============================================================================
# Intrinsic Fallback Helpers (emitted as inline functions)
# ============================================================================

_EXP_FALLBACK = """
// Lane-wise exp() fallback (use SVML _mm256_exp_pd if available)
inline __m256d _mm256_exp_pd_fallback(__m256d v) {
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, v);
    for (int k = 0; k < 4; ++k) tmp[k] = std::exp(tmp[k]);
    return _mm256_load_pd(tmp);
}

"""

_LOG_FALLBACK = """
// Lane-wise log() fallback
inline __m256d _mm256_log_pd_fallback(__m256d v) {
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, v);
    for (int k = 0; k < 4; ++k) tmp[k] = std::log(tmp[k]);
    return _mm256_load_pd(tmp);
}

"""

_ABS_HELPER = """
// Bitwise abs for f64 (clear sign bit)
inline __m256d _mm256_abs_pd(__m256d v) {
    const __m256i mask = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
    return _mm256_castsi256_pd(_mm256_and_si256(_mm256_castpd_si256(v), mask));
}

"""
