"""
ir.py - Intermediate Representation (AST) for the Physics Compiler.

This module defines the node types that form the IR. All frontends (YAML,
Verilog-A) produce this IR, and all backends (Scalar C++, AVX-512, AVX2, WGSL)
consume it.

Design Principles:
  1. Immutable: Nodes are frozen dataclasses. Optimization passes produce
     new trees rather than mutating in place.
  2. Typed: Every expression node carries a resolved type (f64, i32, bool).
  3. Backend-agnostic: No C++ or WGSL syntax leaks into the IR.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict
import hashlib
import json


# ============================================================================
# Types
# ============================================================================

class IRType(Enum):
    """Scalar types supported by the IR."""
    F64  = auto()  # double / EmulatedF64
    F32  = auto()  # float (GPU fast-path)
    I32  = auto()  # int32
    BOOL = auto()  # boolean (control flow)


# ============================================================================
# Expression Nodes
# ============================================================================

@dataclass(frozen=True)
class Expr:
    """Base class for all expression nodes."""
    dtype: IRType = IRType.F64


@dataclass(frozen=True)
class Const(Expr):
    """Literal constant value."""
    value: float = 0.0


@dataclass(frozen=True)
class Var(Expr):
    """Reference to a named variable (parameter, terminal voltage, state)."""
    name: str = ""


@dataclass(frozen=True)
class BinOp(Expr):
    """Binary arithmetic operation."""
    op: str = "+"   # One of: +, -, *, /
    lhs: Expr = field(default_factory=Expr)
    rhs: Expr = field(default_factory=Expr)


@dataclass(frozen=True)
class UnaryOp(Expr):
    """Unary operation (negation)."""
    op: str = "-"
    operand: Expr = field(default_factory=Expr)


@dataclass(frozen=True)
class Call(Expr):
    """Intrinsic function call (exp, log, sqrt, abs, min, max, pow, fma)."""
    func: str = ""
    args: tuple = ()  # Tuple of Expr


@dataclass(frozen=True)
class Cmp(Expr):
    """Comparison producing a boolean."""
    op: str = "<"    # One of: <, <=, >, >=, ==, !=
    lhs: Expr = field(default_factory=Expr)
    rhs: Expr = field(default_factory=Expr)
    dtype: IRType = IRType.BOOL


@dataclass(frozen=True)
class Select(Expr):
    """Ternary select: cond ? true_val : false_val.

    This is the ONLY conditional form in the IR. Backends may lower it to
    if/else or branchless select depending on target.
    """
    cond: Expr = field(default_factory=Expr)
    true_val: Expr = field(default_factory=Expr)
    false_val: Expr = field(default_factory=Expr)


@dataclass(frozen=True)
class IfElse(Expr):
    """Multi-branch conditional for region-based models (cutoff/linear/sat).

    Each branch is (condition, body_assignments). The last branch may have
    condition=None (else clause).
    """
    branches: tuple = ()  # Tuple of (Expr|None, dict[str, Expr])


# ============================================================================
# Statement Nodes
# ============================================================================

@dataclass(frozen=True)
class Assign:
    """Variable assignment: target = expr."""
    target: str = ""
    expr: Expr = field(default_factory=Expr)


@dataclass(frozen=True)
class StampEntry:
    """A single Jacobian stamp directive.

    Tells the stamp code generator to add `expr` to J[row_node, col_node].
    """
    row_terminal: str = ""  # e.g. "d" for drain
    col_terminal: str = ""  # e.g. "g" for gate
    expr: Expr = field(default_factory=Expr)
    is_rhs: bool = False    # True if this is an RHS contribution, not Jacobian


# ============================================================================
# Top-Level Model Definition
# ============================================================================

@dataclass(frozen=True)
class Terminal:
    """A device terminal (pin)."""
    name: str = ""
    description: str = ""


@dataclass(frozen=True)
class Parameter:
    """A device model parameter."""
    name: str = ""
    dtype: IRType = IRType.F64
    default: float = 0.0
    description: str = ""
    unit: str = ""


@dataclass(frozen=True)
class StateVar:
    """An iteration-state variable (updated each NR pass)."""
    name: str = ""
    dtype: IRType = IRType.F64
    description: str = ""


@dataclass
class ModelDef:
    """Complete device model definition — the root of the IR tree.

    This is mutable (not frozen) because it is built incrementally by the
    frontend, then frozen via `layout_hash()` before being handed to backends.
    """
    name: str = ""
    description: str = ""
    terminals: List[Terminal] = field(default_factory=list)
    parameters: List[Parameter] = field(default_factory=list)
    state_vars: List[StateVar] = field(default_factory=list)

    # The physics kernel body: a list of Assign statements
    body: List[Assign] = field(default_factory=list)

    # Jacobian stamp directives (derived from body by middle-end)
    stamps: List[StampEntry] = field(default_factory=list)

    # RHS contribution directives
    rhs_stamps: List[StampEntry] = field(default_factory=list)

    # --- Layout Versioning ---
    _layout_hash: Optional[str] = field(default=None, repr=False)

    def layout_hash(self) -> str:
        """Compute a deterministic hash of the model's data layout.

        This hash is embedded in the generated C++ header and verified at
        runtime to prevent binary mismatches between the model definition
        and compiled code.
        """
        if self._layout_hash is not None:
            return self._layout_hash

        # Hash is over: name, terminal names (sorted), parameter names+defaults
        # (sorted), state_var names (sorted).
        canonical = {
            "name": self.name,
            "terminals": sorted(t.name for t in self.terminals),
            "parameters": sorted(
                [{"name": p.name, "default": p.default, "dtype": p.dtype.name}
                 for p in self.parameters],
                key=lambda x: x["name"]
            ),
            "state_vars": sorted(s.name for s in self.state_vars),
        }
        raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        self._layout_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self._layout_hash


# ============================================================================
# Deterministic Ordering Utilities
# ============================================================================

def canonical_sort_key(node: Expr) -> str:
    """Produce a deterministic string key for an expression tree.

    Used to sort commutative operands (addition, multiplication) to ensure
    bit-exact reproducibility across builds and platforms.
    """
    if isinstance(node, Const):
        return f"C:{node.value}"
    if isinstance(node, Var):
        return f"V:{node.name}"
    if isinstance(node, BinOp):
        return f"B:{node.op}:{canonical_sort_key(node.lhs)}:{canonical_sort_key(node.rhs)}"
    if isinstance(node, UnaryOp):
        return f"U:{node.op}:{canonical_sort_key(node.operand)}"
    if isinstance(node, Call):
        args_key = ",".join(canonical_sort_key(a) for a in node.args)
        return f"F:{node.func}:{args_key}"
    return f"?:{type(node).__name__}"


def canonicalize(node: Expr) -> Expr:
    """Sort commutative operands for deterministic code generation."""
    if isinstance(node, BinOp) and node.op in ("+", "*"):
        lhs = canonicalize(node.lhs)
        rhs = canonicalize(node.rhs)
        # Canonical order: sort by string key
        if canonical_sort_key(lhs) > canonical_sort_key(rhs):
            lhs, rhs = rhs, lhs
        return BinOp(op=node.op, lhs=lhs, rhs=rhs, dtype=node.dtype)
    if isinstance(node, BinOp):
        return BinOp(
            op=node.op,
            lhs=canonicalize(node.lhs),
            rhs=canonicalize(node.rhs),
            dtype=node.dtype,
        )
    if isinstance(node, UnaryOp):
        return UnaryOp(op=node.op, operand=canonicalize(node.operand), dtype=node.dtype)
    if isinstance(node, Call):
        return Call(
            func=node.func,
            args=tuple(canonicalize(a) for a in node.args),
            dtype=node.dtype,
        )
    if isinstance(node, Select):
        return Select(
            cond=canonicalize(node.cond),
            true_val=canonicalize(node.true_val),
            false_val=canonicalize(node.false_val),
            dtype=node.dtype,
        )
    return node  # Const, Var — leaves


# ============================================================================
# IR Hardening Passes
# ============================================================================

def _rename_var_in_expr(expr: Expr, old_name: str, new_name: str) -> Expr:
    """Recursively rename a variable reference in an expression tree."""
    if isinstance(expr, Var) and expr.name == old_name:
        return Var(name=new_name, dtype=expr.dtype)
    if isinstance(expr, BinOp):
        return BinOp(
            op=expr.op,
            lhs=_rename_var_in_expr(expr.lhs, old_name, new_name),
            rhs=_rename_var_in_expr(expr.rhs, old_name, new_name),
            dtype=expr.dtype,
        )
    if isinstance(expr, UnaryOp):
        return UnaryOp(
            op=expr.op,
            operand=_rename_var_in_expr(expr.operand, old_name, new_name),
            dtype=expr.dtype,
        )
    if isinstance(expr, Call):
        return Call(
            func=expr.func,
            args=tuple(_rename_var_in_expr(a, old_name, new_name) for a in expr.args),
            dtype=expr.dtype,
        )
    if isinstance(expr, Cmp):
        return Cmp(
            op=expr.op,
            lhs=_rename_var_in_expr(expr.lhs, old_name, new_name),
            rhs=_rename_var_in_expr(expr.rhs, old_name, new_name),
            dtype=expr.dtype,
        )
    if isinstance(expr, Select):
        return Select(
            cond=_rename_var_in_expr(expr.cond, old_name, new_name),
            true_val=_rename_var_in_expr(expr.true_val, old_name, new_name),
            false_val=_rename_var_in_expr(expr.false_val, old_name, new_name),
            dtype=expr.dtype,
        )
    return expr  # Const — leaf


def enforce_ssa(model: ModelDef) -> ModelDef:
    """Transform a model's body into SSA form.

    If a variable is assigned more than once, subsequent assignments get
    suffixed (e.g., arg -> arg_1, arg_2). All references in later statements
    are updated to use the latest version.

    State variables are exempt from renaming because they represent the
    final output value written back to the tensor.
    """
    state_var_names = {s.name for s in model.state_vars}
    seen: Dict[str, int] = {}  # name -> version count
    rename_map: Dict[str, str] = {}  # original -> current SSA name
    new_body: List[Assign] = []

    for assign in model.body:
        # Apply current renames to the expression
        expr = assign.expr
        for old, new in rename_map.items():
            expr = _rename_var_in_expr(expr, old, new)

        target = assign.target

        if target in state_var_names:
            # State vars keep their name (they are tensor outputs)
            new_body.append(Assign(target=target, expr=expr))
        elif target in seen:
            # Duplicate assignment — create SSA version
            seen[target] += 1
            new_name = f"{target}_{seen[target]}"
            rename_map[target] = new_name
            new_body.append(Assign(target=new_name, expr=expr))
        else:
            seen[target] = 0
            new_body.append(Assign(target=target, expr=expr))

    model.body = new_body
    return model


class IRValidationError(Exception):
    """Raised when the IR violates purity constraints."""
    pass


def validate_types(model: ModelDef):
    """Verify that all expression nodes use F64 (except Cmp → BOOL).

    Raises IRValidationError if any node uses F32 or I32 in the physics body.
    """
    def _check(expr: Expr, context: str):
        if isinstance(expr, Cmp):
            # Cmp produces BOOL, that's fine
            _check(expr.lhs, context)
            _check(expr.rhs, context)
            return
        if expr.dtype not in (IRType.F64, IRType.BOOL):
            raise IRValidationError(
                f"Type violation in {context}: {type(expr).__name__} "
                f"has dtype={expr.dtype.name}, expected F64. "
                f"Implicit casts are not allowed."
            )
        if isinstance(expr, BinOp):
            _check(expr.lhs, context)
            _check(expr.rhs, context)
        elif isinstance(expr, UnaryOp):
            _check(expr.operand, context)
        elif isinstance(expr, Call):
            for a in expr.args:
                _check(a, context)
        elif isinstance(expr, Select):
            _check(expr.cond, context)
            _check(expr.true_val, context)
            _check(expr.false_val, context)

    for assign in model.body:
        _check(assign.expr, f"assign '{assign.target}'")
    for stamp in model.stamps:
        _check(stamp.expr, f"stamp [{stamp.row_terminal},{stamp.col_terminal}]")
    for stamp in model.rhs_stamps:
        _check(stamp.expr, f"rhs_stamp [{stamp.row_terminal}]")


def validate_ssa(model: ModelDef):
    """Verify that no variable is assigned more than once (SSA invariant).

    State variables may appear at most once as a final write.
    Raises IRValidationError on duplicate assignments.
    """
    state_var_names = {s.name for s in model.state_vars}
    seen = set()
    for assign in model.body:
        if assign.target in seen and assign.target not in state_var_names:
            raise IRValidationError(
                f"SSA violation: '{assign.target}' assigned more than once. "
                f"Run enforce_ssa() first."
            )
        seen.add(assign.target)


def stable_rename_temps(model: ModelDef) -> ModelDef:
    """Rename all non-parameter, non-state-var, non-terminal locals to
    deterministic sequential names: _t0, _t1, _t2, ...

    This ensures identical code generation regardless of the naming
    conventions used in the model source file.
    """
    # Identify names that must NOT be renamed
    reserved = set()
    for t in model.terminals:
        reserved.add(f"v_{t.name}")
        reserved.add(f"n_{t.name}")
    for p in model.parameters:
        reserved.add(p.name)
    for s in model.state_vars:
        reserved.add(s.name)

    # Build rename map for locals
    counter = 0
    rename_map: Dict[str, str] = {}
    for assign in model.body:
        if assign.target not in reserved and assign.target not in rename_map:
            new_name = f"_t{counter}"
            rename_map[assign.target] = new_name
            counter += 1

    if not rename_map:
        return model

    # Apply renames to all body statements
    new_body = []
    for assign in model.body:
        expr = assign.expr
        for old, new in rename_map.items():
            expr = _rename_var_in_expr(expr, old, new)
        new_target = rename_map.get(assign.target, assign.target)
        new_body.append(Assign(target=new_target, expr=expr))

    # Apply renames to stamp expressions
    new_stamps = []
    for stamp in model.stamps:
        expr = stamp.expr
        for old, new in rename_map.items():
            expr = _rename_var_in_expr(expr, old, new)
        new_stamps.append(StampEntry(
            row_terminal=stamp.row_terminal,
            col_terminal=stamp.col_terminal,
            expr=expr, is_rhs=stamp.is_rhs,
        ))

    new_rhs = []
    for stamp in model.rhs_stamps:
        expr = stamp.expr
        for old, new in rename_map.items():
            expr = _rename_var_in_expr(expr, old, new)
        new_rhs.append(StampEntry(
            row_terminal=stamp.row_terminal,
            col_terminal=stamp.col_terminal,
            expr=expr, is_rhs=stamp.is_rhs,
        ))

    model.body = new_body
    model.stamps = new_stamps
    model.rhs_stamps = new_rhs
    return model
