"""
avx512.py - AVX-512 Backend for the Physics Compiler.

Emits C++ code using AVX-512 intrinsics to process 8 devices per iteration.
The generated code follows the same pattern as the hand-written
`physics_kernels_avx512.cpp`:
  - Vectorized loads of terminal voltages (manual gather)
  - SIMD arithmetic for device physics
  - Vectorized stores of results
  - Scalar tail loop for remaining elements (n % 8 != 0)

Key intrinsics used:
  _mm512_set1_pd, _mm512_load_pd, _mm512_store_pd
  _mm512_add_pd, _mm512_sub_pd, _mm512_mul_pd, _mm512_div_pd
  _mm512_min_pd, _mm512_max_pd
  _mm512_exp_pd_fallback (lane-wise std::exp)
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
# AVX-512 Expression Emitter
# ============================================================================

def emit_avx_expr(node: Expr, prefix: str = "v_") -> str:
    """Convert an IR expression tree to an AVX-512 intrinsics expression.

    All variables are assumed to be __m512d registers prefixed with `prefix`.
    """
    if isinstance(node, Const):
        val = node.value
        if val == 0.0:
            return "_mm512_setzero_pd()"
        return f"_mm512_set1_pd({_format_double(val)})"

    if isinstance(node, Var):
        return f"{prefix}{node.name}"

    if isinstance(node, BinOp):
        lhs = emit_avx_expr(node.lhs, prefix)
        rhs = emit_avx_expr(node.rhs, prefix)
        op_map = {
            "+": "_mm512_add_pd",
            "-": "_mm512_sub_pd",
            "*": "_mm512_mul_pd",
            "/": "_mm512_div_pd",
        }
        func = op_map.get(node.op)
        if func:
            return f"{func}({lhs}, {rhs})"
        raise ValueError(f"Unsupported AVX binary op: {node.op}")

    if isinstance(node, UnaryOp) and node.op == "-":
        operand = emit_avx_expr(node.operand, prefix)
        return f"_mm512_sub_pd(_mm512_setzero_pd(), {operand})"

    if isinstance(node, Call):
        func = node.func
        args = node.args

        if func == "exp" and len(args) == 1:
            arg = emit_avx_expr(args[0], prefix)
            return f"_mm512_exp_pd_fallback({arg})"

        if func == "log" and len(args) == 1:
            arg = emit_avx_expr(args[0], prefix)
            return f"_mm512_log_pd_fallback({arg})"

        if func == "sqrt" and len(args) == 1:
            arg = emit_avx_expr(args[0], prefix)
            return f"_mm512_sqrt_pd({arg})"

        if func in ("abs", "fabs") and len(args) == 1:
            arg = emit_avx_expr(args[0], prefix)
            # abs via AND with sign-mask (clear sign bit)
            return f"_mm512_abs_pd({arg})"

        if func == "min" and len(args) == 2:
            a = emit_avx_expr(args[0], prefix)
            b = emit_avx_expr(args[1], prefix)
            return f"_mm512_min_pd({a}, {b})"

        if func == "max" and len(args) == 2:
            a = emit_avx_expr(args[0], prefix)
            b = emit_avx_expr(args[1], prefix)
            return f"_mm512_max_pd({a}, {b})"

        if func == "fma" and len(args) == 3:
            a = emit_avx_expr(args[0], prefix)
            b = emit_avx_expr(args[1], prefix)
            c = emit_avx_expr(args[2], prefix)
            return f"_mm512_fmadd_pd({a}, {b}, {c})"

        raise ValueError(f"Unsupported AVX function: {func}")

    if isinstance(node, Select):
        # Use mask-based select: _mm512_mask_blend_pd
        cond = node.cond
        tv = emit_avx_expr(node.true_val, prefix)
        fv = emit_avx_expr(node.false_val, prefix)
        if isinstance(cond, Cmp):
            lhs_c = emit_avx_expr(cond.lhs, prefix)
            rhs_c = emit_avx_expr(cond.rhs, prefix)
            cmp_map = {
                "<": "_CMP_LT_OQ",
                "<=": "_CMP_LE_OQ",
                ">": "_CMP_GT_OQ",
                ">=": "_CMP_GE_OQ",
                "==": "_CMP_EQ_OQ",
                "!=": "_CMP_NEQ_OQ",
            }
            cmp_imm = cmp_map.get(cond.op, "_CMP_LT_OQ")
            return f"_mm512_mask_blend_pd(_mm512_cmp_pd_mask({lhs_c}, {rhs_c}, {cmp_imm}), {fv}, {tv})"
        # Fallback: evaluate condition as a mask
        return f"/* SELECT: complex cond */ {tv}"

    raise ValueError(f"Unknown IR node type for AVX: {type(node).__name__}")


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
# Full AVX-512 File Emitter
# ============================================================================

def generate(model: ModelDef) -> str:
    """Generate the complete AVX-512 C++ source for a device model."""
    out = io.StringIO()
    struct_name = f"{model.name}Tensor"
    func_name = f"batch{model.name}Physics_avx512"

    out.write(f"// AUTO-GENERATED by Physics Compiler — DO NOT EDIT\n")
    out.write(f"// Model: {model.name} (AVX-512 Backend)\n")
    out.write(f"// Layout Hash: {model.layout_hash()}\n\n")

    out.write(f'#include "gen_{model.name.lower()}.h"  // Scalar tensor struct\n')
    out.write(f"#include <vector>\n")
    out.write(f"#include <cmath>\n")
    out.write(f"#include <algorithm>\n\n")

    out.write(f"#if defined(__AVX512F__)\n")
    out.write(f"#include <immintrin.h>\n\n")

    # Emit exp fallback helper
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
    out.write(f"    // --- SIMD Loop: 8 devices per iteration ---\n")
    out.write(f"    for (; i + 7 < n; i += 8) {{\n")

    # Gather terminal voltages
    _emit_voltage_gathers(model, out)

    # Load parameters
    out.write(f"\n        // Load parameters\n")
    for p in model.parameters:
        out.write(f"        __m512d v_{p.name} = _mm512_loadu_pd(&tensor.{p.name}[i]);\n")

    out.write(f"\n")

    # Emit kernel body (vectorized)
    for assign in model.body:
        expr = canonicalize(assign.expr)
        avx_expr = emit_avx_expr(expr, "v_")
        is_state = any(s.name == assign.target for s in model.state_vars)

        if is_state:
            out.write(f"        __m512d v_{assign.target} = {avx_expr};\n")
            out.write(f"        _mm512_storeu_pd(&tensor.{assign.target}[i], v_{assign.target});\n")
        else:
            out.write(f"        __m512d v_{assign.target} = {avx_expr};\n")

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

    out.write(f"#endif // __AVX512F__\n")

    return out.getvalue()


# ============================================================================
# Helper Emitters
# ============================================================================

def _emit_constant_broadcasts(model: ModelDef, out: TextIO):
    """Emit _mm512_set1_pd broadcasts for constants used in the kernel."""
    # Scan the body for literal constants that appear frequently
    consts = set()
    for assign in model.body:
        _collect_constants(assign.expr, consts)

    if consts:
        out.write(f"    // Pre-broadcast constants\n")
        for c in sorted(consts):
            name = _const_name(c)
            out.write(f"    const __m512d {name} = _mm512_set1_pd({_format_double(c)});\n")


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
    """Emit manual voltage gathers for terminal nodes."""
    out.write(f"\n        // Gather terminal voltages (manual gather for portability)\n")
    out.write(f"        const double* v_ptr = voltages.data();\n")
    out.write(f"        const int v_size = (int)voltages.size();\n")

    for t in model.terminals:
        out.write(f"\n        alignas(64) double _buf_{t.name}[8];\n")
        out.write(f"        for (int k = 0; k < 8; ++k) {{\n")
        out.write(f"            int idx = tensor.node_{t.name}[i + k];\n")
        out.write(f"            _buf_{t.name}[k] = (idx > 0 && idx <= v_size) ? v_ptr[idx - 1] : 0.0;\n")
        out.write(f"        }}\n")
        out.write(f"        __m512d v_v_{t.name} = _mm512_load_pd(_buf_{t.name});\n")


# ============================================================================
# Intrinsic Fallback Helpers (emitted as inline functions)
# ============================================================================

_EXP_FALLBACK = """
// Lane-wise exp() fallback (use SVML _mm512_exp_pd if available)
inline __m512d _mm512_exp_pd_fallback(__m512d v) {
    alignas(64) double tmp[8];
    _mm512_store_pd(tmp, v);
    for (int k = 0; k < 8; ++k) tmp[k] = std::exp(tmp[k]);
    return _mm512_load_pd(tmp);
}

"""

_LOG_FALLBACK = """
// Lane-wise log() fallback
inline __m512d _mm512_log_pd_fallback(__m512d v) {
    alignas(64) double tmp[8];
    _mm512_store_pd(tmp, v);
    for (int k = 0; k < 8; ++k) tmp[k] = std::log(tmp[k]);
    return _mm512_load_pd(tmp);
}

"""

_ABS_HELPER = """
// Bitwise abs for f64 (clear sign bit)
inline __m512d _mm512_abs_pd(__m512d v) {
    const __m512i mask = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
    return _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(v), mask));
}

"""
