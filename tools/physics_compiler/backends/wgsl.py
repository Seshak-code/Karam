"""
wgsl.py - WebGPU Shader Language (WGSL) Backend for the Physics Compiler.

Emits compute shaders that run device physics on the GPU using
EmulatedF64 (hi-lo f32 pairs) or native f32 depending on precision mode.

Key features:
  - Specialization constants for workgroup sizes
  - SoA buffer bindings matching the C++ tensor layout
  - Compute shader entry points for batch physics evaluation
"""

from __future__ import annotations
from typing import TextIO
import io

from ..ir import (
    ModelDef, Expr, Const, Var, BinOp, UnaryOp, Call, Select,
    Assign, StampEntry, IRType,
    canonicalize,
)


# ============================================================================
# WGSL Expression Emitter
# ============================================================================

def emit_expr_wgsl(node: Expr) -> str:
    """Convert an IR expression tree to a WGSL expression string."""
    if isinstance(node, Const):
        val = node.value
        # WGSL requires explicit float literals
        if val == int(val) and abs(val) < 1e15:
            return f"{val:.1f}"
        # Scientific notation: WGSL doesn't support 1e-14, use explicit
        return f"f32({val})"

    if isinstance(node, Var):
        return node.name

    if isinstance(node, BinOp):
        lhs = emit_expr_wgsl(node.lhs)
        rhs = emit_expr_wgsl(node.rhs)
        return f"({lhs} {node.op} {rhs})"

    if isinstance(node, UnaryOp):
        operand = emit_expr_wgsl(node.operand)
        return f"(-{operand})"

    if isinstance(node, Call):
        args_str = ", ".join(emit_expr_wgsl(a) for a in node.args)
        # Map intrinsic names to WGSL equivalents
        func_map = {
            "exp": "exp",
            "log": "log",
            "sqrt": "sqrt",
            "abs": "abs",
            "fabs": "abs",
            "min": "min",
            "max": "max",
            "pow": "pow",
            "fma": "fma",
        }
        wgsl_func = func_map.get(node.func, node.func)
        return f"{wgsl_func}({args_str})"

    if isinstance(node, Select):
        cond = emit_expr_wgsl(node.cond)
        tv = emit_expr_wgsl(node.true_val)
        fv = emit_expr_wgsl(node.false_val)
        return f"select({fv}, {tv}, {cond})"

    raise ValueError(f"Unknown IR node type for WGSL: {type(node).__name__}")


# ============================================================================
# Buffer Struct Emitter
# ============================================================================

def emit_buffer_struct(model: ModelDef, out: TextIO):
    """Emit WGSL struct definition for the SoA device buffer."""
    struct_name = f"{model.name}Data"

    out.write(f"// SoA Buffer for {model.name}\n")
    out.write(f"struct {struct_name} {{\n")

    # Terminal node indices
    for t in model.terminals:
        out.write(f"    node_{t.name}: array<i32>,\n")

    # Parameters
    for p in model.parameters:
        wgsl_type = "f32"  # GPU uses f32 for performance
        out.write(f"    {p.name}: array<{wgsl_type}>,\n")

    # State variables
    for s in model.state_vars:
        out.write(f"    {s.name}: array<f32>,\n")

    out.write(f"}};\n\n")


# ============================================================================
# Compute Shader Emitter
# ============================================================================

def emit_compute_shader(model: ModelDef, out: TextIO):
    """Emit the compute shader for batch physics evaluation."""
    struct_name = f"{model.name}Data"
    entry_name = f"batch_{model.name.lower()}_physics"

    # Specialization constants
    out.write(f"// Specialization constants\n")
    out.write(f"override WORKGROUP_SIZE: u32 = 64u;\n\n")

    # Bindings
    out.write(f"@group(0) @binding(0) var<storage, read_write> devices: {struct_name};\n")
    out.write(f"@group(0) @binding(1) var<storage, read> voltages: array<f32>;\n")
    out.write(f"@group(0) @binding(2) var<uniform> num_devices: u32;\n\n")

    # Compute entry point
    out.write(f"@compute @workgroup_size(WORKGROUP_SIZE)\n")
    out.write(f"fn {entry_name}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n")
    out.write(f"    let i = gid.x;\n")
    out.write(f"    if (i >= num_devices) {{ return; }}\n\n")

    # Load terminal voltages
    for t in model.terminals:
        node_var = f"n_{t.name}"
        out.write(f"    let {node_var} = devices.node_{t.name}[i];\n")
        out.write(f"    let v_{t.name}: f32 = select(0.0, voltages[{node_var} - 1], {node_var} > 0);\n")

    out.write("\n")

    # Load parameters
    for p in model.parameters:
        out.write(f"    let {p.name}: f32 = devices.{p.name}[i];\n")

    out.write("\n")

    # Emit kernel body
    for assign in model.body:
        expr = canonicalize(assign.expr)
        wgsl_expr = emit_expr_wgsl(expr)

        is_state = any(s.name == assign.target for s in model.state_vars)
        if is_state:
            out.write(f"    devices.{assign.target}[i] = {wgsl_expr};\n")
            out.write(f"    let {assign.target}: f32 = devices.{assign.target}[i];\n")
        else:
            out.write(f"    let {assign.target}: f32 = {wgsl_expr};\n")

    out.write(f"}}\n")


# ============================================================================
# Full File Emitter
# ============================================================================

def generate(model: ModelDef) -> str:
    """Generate the complete WGSL shader for a device model.

    Returns the generated WGSL code as a string.
    """
    out = io.StringIO()

    out.write(f"// AUTO-GENERATED by Physics Compiler — DO NOT EDIT\n")
    out.write(f"// Model: {model.name}\n")
    out.write(f"// Layout Hash: {model.layout_hash()}\n")
    out.write(f"// Description: {model.description}\n\n")

    emit_buffer_struct(model, out)
    emit_compute_shader(model, out)

    return out.getvalue()
