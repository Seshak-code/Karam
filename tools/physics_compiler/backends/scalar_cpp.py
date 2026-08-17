"""
scalar_cpp.py - Scalar C++ Backend for the Physics Compiler.

Walks the IR tree and emits plain C++ code:
  1. A SoA Tensor struct (e.g. `DiodeTensor`)
  2. A batch physics function (e.g. `batchDiodePhysics`)
  3. Jacobian stamp helper code
  4. Layout hash constant for runtime verification

This backend produces portable, debuggable code with no SIMD dependencies.
It serves as the reference implementation for validation against AVX/GPU.
"""

from __future__ import annotations
from typing import TextIO
import io

from ..ir import (
    ModelDef, Expr, Const, Var, BinOp, UnaryOp, Call, Select, Cmp, IfElse,
    Assign, StampEntry, IRType,
    canonicalize,
)


# ============================================================================
# Expression Emitter
# ============================================================================

def emit_expr(node: Expr) -> str:
    """Convert an IR expression tree to a C++ expression string."""
    if isinstance(node, Const):
        # Emit doubles with enough precision
        val = node.value
        if val == int(val) and abs(val) < 1e15:
            return f"{val:.1f}"
        return repr(val)

    if isinstance(node, Var):
        return node.name

    if isinstance(node, BinOp):
        lhs = emit_expr(node.lhs)
        rhs = emit_expr(node.rhs)
        return f"({lhs} {node.op} {rhs})"

    if isinstance(node, UnaryOp):
        operand = emit_expr(node.operand)
        return f"(-{operand})"

    if isinstance(node, Call):
        args_str = ", ".join(emit_expr(a) for a in node.args)
        # Map intrinsic names to std:: equivalents
        func_map = {
            "exp": "std::exp",
            "log": "std::log",
            "sqrt": "std::sqrt",
            "abs": "std::abs",
            "fabs": "std::fabs",
            "min": "std::min",
            "max": "std::max",
            "pow": "std::pow",
            "fma": "std::fma",
        }
        cpp_func = func_map.get(node.func, node.func)
        return f"{cpp_func}({args_str})"

    if isinstance(node, Cmp):
        lhs = emit_expr(node.lhs)
        rhs = emit_expr(node.rhs)
        return f"({lhs} {node.op} {rhs})"

    if isinstance(node, Select):
        cond = emit_expr(node.cond)
        tv = emit_expr(node.true_val)
        fv = emit_expr(node.false_val)
        return f"({cond} ? {tv} : {fv})"

    raise ValueError(f"Unknown IR node type: {type(node).__name__}")


# ============================================================================
# Tensor Struct Emitter
# ============================================================================

def emit_tensor_struct(model: ModelDef, out: TextIO):
    """Emit the SoA tensor struct for a device model."""
    struct_name = f"{model.name}Tensor"

    out.write(f"// Layout Hash: {model.layout_hash()}\n")
    out.write(f"struct {struct_name} {{\n")

    # Terminal node indices
    out.write("    // Node indices (global node IDs)\n")
    for t in model.terminals:
        out.write(f"    std::vector<int> node_{t.name};\n")
    out.write("\n")

    # Device parameters
    out.write("    // Device parameters\n")
    for p in model.parameters:
        cpp_type = "double" if p.dtype == IRType.F64 else "float"
        comment = f"  // {p.description}" if p.description else ""
        unit = f" [{p.unit}]" if p.unit else ""
        out.write(f"    std::vector<{cpp_type}> {p.name};{comment}{unit}\n")
    out.write("\n")

    # State variables
    out.write("    // Iteration state (updated each NR pass)\n")
    for s in model.state_vars:
        cpp_type = "double" if s.dtype == IRType.F64 else "float"
        comment = f"  // {s.description}" if s.description else ""
        out.write(f"    std::vector<{cpp_type}> {s.name};{comment}\n")
    out.write("\n")

    # size()
    first_terminal = model.terminals[0].name if model.terminals else "node_a"
    out.write(f"    size_t size() const {{ return node_{first_terminal}.size(); }}\n\n")

    # clear()
    out.write("    void clear() {\n")
    for t in model.terminals:
        out.write(f"        node_{t.name}.clear();\n")
    for p in model.parameters:
        out.write(f"        {p.name}.clear();\n")
    for s in model.state_vars:
        out.write(f"        {s.name}.clear();\n")
    out.write("    }\n\n")

    out.write("    void reserve(size_t n) {\n")
    for t in model.terminals:
        out.write(f"        node_{t.name}.reserve(n);\n")
    for p in model.parameters:
        out.write(f"        {p.name}.reserve(n);\n")
    for s in model.state_vars:
        out.write(f"        {s.name}.reserve(n);\n")
    out.write("    }\n\n")

    # resize()
    out.write("    void resize(size_t n) {\n")
    for t in model.terminals:
        out.write(f"        node_{t.name}.resize(n, 0);\n")
    for p in model.parameters:
        out.write(f"        {p.name}.resize(n, {p.default});\n")
    for s in model.state_vars:
        out.write(f"        {s.name}.resize(n, 0.0);\n")
    out.write("    }\n\n")

    # push()
    param_args = ", ".join(
        f"double {p.name}_val = {p.default}" for p in model.parameters
    )
    terminal_args = ", ".join(f"int {t.name}" for t in model.terminals)
    out.write(f"    void push({terminal_args}, {param_args}) {{\n")
    for t in model.terminals:
        out.write(f"        node_{t.name}.push_back({t.name});\n")
    for p in model.parameters:
        out.write(f"        {p.name}.push_back({p.name}_val);\n")
    for s in model.state_vars:
        out.write(f"        {s.name}.push_back(0.0);\n")
    out.write("    }\n")

    # Layout hash constant
    out.write(f"\n    static constexpr const char* LAYOUT_HASH = \"{model.layout_hash()}\";\n")

    out.write(f"}};\n\n")


# ============================================================================
# Batch Physics Function Emitter
# ============================================================================

def emit_batch_physics(model: ModelDef, out: TextIO):
    """Emit the scalar batch physics evaluation function."""
    struct_name = f"{model.name}Tensor"
    func_name = f"batch{model.name}Physics"

    out.write(f"/**\n")
    out.write(f" * {func_name} - Evaluates all {model.name}s in a tensor simultaneously.\n")
    out.write(f" * Generated by Physics Compiler. Layout Hash: {model.layout_hash()}\n")
    out.write(f" */\n")
    out.write(f"inline void {func_name}({struct_name}& tensor, "
              f"const std::vector<double>& voltages) {{\n")
    out.write(f"    const size_t n = tensor.size();\n")
    out.write(f"\n")
    out.write(f"    for (size_t i = 0; i < n; ++i) {{\n")

    # Load terminal voltages
    for t in model.terminals:
        out.write(f"        int n_{t.name} = tensor.node_{t.name}[i];\n")
        out.write(f"        double v_{t.name} = (n_{t.name} > 0 && n_{t.name} <= (int)voltages.size()) "
                  f"? voltages[n_{t.name} - 1] : 0.0;\n")

    # Load parameters
    for p in model.parameters:
        out.write(f"        double {p.name} = tensor.{p.name}[i];\n")

    out.write("\n")

    # Emit kernel body (track declared locals to avoid redefinition)
    declared_locals = set()
    for assign in model.body:
        expr = canonicalize(assign.expr)
        cpp_expr = emit_expr(expr)

        # Check if this is a state variable (store back to tensor)
        is_state = any(s.name == assign.target for s in model.state_vars)
        if is_state:
            out.write(f"        tensor.{assign.target}[i] = {cpp_expr};\n")
            if assign.target not in declared_locals:
                out.write(f"        double {assign.target} = tensor.{assign.target}[i];\n")
                declared_locals.add(assign.target)
            else:
                out.write(f"        {assign.target} = tensor.{assign.target}[i];\n")
        else:
            if assign.target not in declared_locals:
                out.write(f"        double {assign.target} = {cpp_expr};\n")
                declared_locals.add(assign.target)
            else:
                out.write(f"        {assign.target} = {cpp_expr};\n")

    out.write(f"    }}\n")
    out.write(f"}}\n\n")


# ============================================================================
# Stamp Helper Emitter
# ============================================================================

def emit_stamp_helper(model: ModelDef, out: TextIO):
    """Emit a function that stamps the Jacobian and RHS for this device type."""
    struct_name = f"{model.name}Tensor"
    func_name = f"stamp{model.name}"

    out.write(f"/**\n")
    out.write(f" * {func_name} - Stamps Jacobian and RHS for all {model.name}s.\n")
    out.write(f" * Generated by Physics Compiler.\n")
    out.write(f" */\n")
    out.write(f"inline void {func_name}(const {struct_name}& tensor, "
              f"SparseStampTarget* target) {{\n")
    out.write(f"    const size_t n = tensor.size();\n\n")
    out.write(f"    for (size_t i = 0; i < n; ++i) {{\n")

    # Load node indices
    for t in model.terminals:
        out.write(f"        int n_{t.name} = tensor.node_{t.name}[i];\n")

    # Load state variables needed by stamps
    state_names = {s.name for s in model.state_vars}
    needed_vars = set()
    for stamp in model.stamps + model.rhs_stamps:
        _collect_vars(stamp.expr, needed_vars)
    needed_state = needed_vars & state_names
    for var_name in sorted(needed_state):
        out.write(f"        double {var_name} = tensor.{var_name}[i];\n")

    out.write("\n")

    # Jacobian stamps
    if model.stamps:
        out.write("        // Jacobian stamps\n")
        for stamp in model.stamps:
            row = f"n_{stamp.row_terminal}"
            col = f"n_{stamp.col_terminal}"
            expr = emit_expr(canonicalize(stamp.expr))
            # Only stamp if both nodes are non-ground
            out.write(f"        if ({row} > 0 && {col} > 0) "
                      f"target->add({row} - 1, {col} - 1, {expr});\n")

    # RHS stamps
    if model.rhs_stamps:
        out.write("\n        // RHS stamps\n")
        for stamp in model.rhs_stamps:
            node = f"n_{stamp.row_terminal}"
            expr = emit_expr(canonicalize(stamp.expr))
            out.write(f"        if ({node} > 0) "
                      f"target->addRhs({node} - 1, {expr});\n")

    out.write(f"    }}\n")
    out.write(f"}}\n\n")


def _collect_vars(node: Expr, result: set):
    """Recursively collect all Var names referenced in an expression."""
    if isinstance(node, Var):
        result.add(node.name)
    elif isinstance(node, BinOp):
        _collect_vars(node.lhs, result)
        _collect_vars(node.rhs, result)
    elif isinstance(node, UnaryOp):
        _collect_vars(node.operand, result)
    elif isinstance(node, Call):
        for a in node.args:
            _collect_vars(a, result)
    elif isinstance(node, Select):
        _collect_vars(node.cond, result)
        _collect_vars(node.true_val, result)
        _collect_vars(node.false_val, result)


def emit_type_erased_wrappers(model: ModelDef, out: TextIO):
    """Emit type-erased C wrapper functions for the ModelRegistry."""
    struct_name = f"{model.name}Tensor"
    prefix = model.name.lower()

    out.write(f"// ============================================================\n")
    out.write(f"// Type-Erased Wrappers for ModelRegistry\n")
    out.write(f"// ============================================================\n\n")

    # create
    out.write(f"inline void* {prefix}_create() {{\n")
    out.write(f"    return new {struct_name}();\n")
    out.write(f"}}\n\n")

    # destroy
    out.write(f"inline void {prefix}_destroy(void* tensor) {{\n")
    out.write(f"    delete static_cast<{struct_name}*>(tensor);\n")
    out.write(f"}}\n\n")

    # resize
    out.write(f"inline void {prefix}_resize(void* tensor, size_t count) {{\n")
    out.write(f"    static_cast<{struct_name}*>(tensor)->resize(count);\n")
    out.write(f"}}\n\n")

    # setInstance
    out.write(f"inline void {prefix}_set_instance(void* tensor, size_t index, "
              f"const int* nodes, const double* params) {{\n")
    out.write(f"    auto* t = static_cast<{struct_name}*>(tensor);\n")
    # Terminals (nodes)
    for i, term in enumerate(model.terminals):
        out.write(f"    t->node_{term.name}[index] = nodes[{i}];\n")
    # Parameters
    for i, param in enumerate(model.parameters):
        out.write(f"    t->{param.name}[index] = params[{i}];\n")
    out.write(f"}}\n\n")

    # batchPhysics
    out.write(f"inline void {prefix}_batch_physics(void* tensor, const std::vector<double>& voltages) {{\n")
    out.write(f"    {struct_name}* t = static_cast<{struct_name}*>(tensor);\n")
    out.write(f"    batch{model.name}Physics(*t, voltages);\n")
    out.write(f"}}\n\n")

    # stampJacobian
    if model.stamps or model.rhs_stamps:
        out.write(f"inline void {prefix}_stamp_jacobian(const void* tensor, size_t count, SparseStampTarget* target) {{\n")
        out.write(f"    const {struct_name}* t = static_cast<const {struct_name}*>(tensor);\n")
        out.write(f"    stamp{model.name}(*t, target);\n")
        out.write(f"}}\n\n")
    else:
        out.write(f"// No stamps for {model.name}\n")
        out.write(f"inline void {prefix}_stamp_jacobian(const void* tensor, size_t count, SparseStampTarget* target) {{}}\n\n")


def generate(model: ModelDef) -> str:
    """Generate the complete C++ header for a device model.

    Returns the generated code as a string.
    """
    out = io.StringIO()

    out.write(f"#pragma once\n")
    out.write(f"// AUTO-GENERATED by Physics Compiler — DO NOT EDIT\n")
    out.write(f"// Model: {model.name}\n")
    out.write(f"// Layout Hash: {model.layout_hash()}\n")
    out.write(f"// Description: {model.description}\n")
    out.write(f"// Compiler Constraints: F64-only SSA, -fno-fast-math, -ffp-contract=off\n\n")

    out.write(f"#include <vector>\n")
    out.write(f"#include <cmath>\n")
    out.write(f"#include <algorithm>\n")
    out.write(f"#include <cstddef>\n")
    out.write(f'#include "model_registry.h"\n\n')

    emit_tensor_struct(model, out)
    emit_batch_physics(model, out)

    # Only emit stamp helper if stamps are defined
    if model.stamps or model.rhs_stamps:
        out.write(f"// Stamp helpers — require linalg.h to be included BEFORE this header.\n")
        # We don't need linalg.h check strictly if we use SparseStampTarget interface.
        # But for ABI safety let's assume ModelRegistry is present.
        emit_stamp_helper(model, out)
        out.write(f"\n")

    # Type-erased wrappers
    emit_type_erased_wrappers(model, out)

    # Registration
    prefix = model.name.lower()
    stamp_fn = f"{prefix}_stamp_jacobian"
    
    # Terminal Names Array
    term_names_str = ", ".join(f'"{t.name}"' for t in model.terminals)
    term_array_name = f"{prefix}_terminals"
    out.write(f"static const char* {term_array_name}[] = {{ {term_names_str}, nullptr }};\n\n")

    out.write(f"REGISTER_PHYSICS_MODEL_EX(\n")
    out.write(f"    {model.name},\n")
    out.write(f"    {model.name},  // deviceType\n")
    out.write(f'    "{model.description}",\n')
    out.write(f'    "{model.layout_hash()}",\n')
    out.write(f"    {len(model.terminals)},\n")
    out.write(f"    {len(model.parameters)},\n")
    out.write(f"    {prefix}_create,\n")
    out.write(f"    {prefix}_destroy,\n")
    out.write(f"    {prefix}_resize,\n")
    out.write(f"    {prefix}_set_instance,\n")
    out.write(f"    {prefix}_batch_physics,\n")
    out.write(f"    {stamp_fn},\n")
    out.write(f"    {term_array_name}\n")
    out.write(f")\n")

    return out.getvalue()

