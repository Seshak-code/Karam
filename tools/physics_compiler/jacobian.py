"""
jacobian.py - Automatic Jacobian Derivation for the Physics Compiler.

Performs symbolic differentiation on IR expression trees to automatically
compute ∂I/∂V for Jacobian matrix stamping. This eliminates the need to
manually specify stamp entries in YAML model definitions.

Approach:
  - Works directly on the IR AST (no SymPy dependency for core logic).
  - Implements forward-mode symbolic differentiation via recursive tree walking.
  - Applies algebraic simplification passes to reduce generated expression size.
  - Produces StampEntry directives for the backends.

Supported derivative rules:
  - d/dx(const) = 0
  - d/dx(x) = 1
  - d/dx(f + g) = df + dg
  - d/dx(f * g) = f*dg + g*df       (product rule)
  - d/dx(f / g) = (g*df - f*dg)/g²  (quotient rule)
  - d/dx(-f) = -df
  - d/dx(exp(f)) = exp(f) * df       (chain rule)
  - d/dx(log(f)) = df / f
  - d/dx(sqrt(f)) = df / (2*sqrt(f))
  - d/dx(pow(f,g)) = pow(f,g) * (g*df/f + dg*log(f))
  - d/dx(min(f,g)) = select(f<g, df, dg)
  - d/dx(max(f,g)) = select(f>g, df, dg)
  - d/dx(abs(f)) = select(f>0, df, -df)
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional
from .ir import (
    ModelDef, Expr, Const, Var, BinOp, UnaryOp, Call, Select, Cmp,
    Assign, StampEntry, Terminal,
)


# ============================================================================
# Symbolic Differentiation
# ============================================================================

ZERO = Const(value=0.0)
ONE = Const(value=1.0)
TWO = Const(value=2.0)


def differentiate(expr: Expr, var: str) -> Expr:
    """Compute the symbolic derivative d(expr)/d(var).

    Args:
        expr: The expression tree to differentiate.
        var: The variable name to differentiate with respect to.

    Returns:
        A new expression tree representing the derivative.
    """
    if isinstance(expr, Const):
        return ZERO

    if isinstance(expr, Var):
        return ONE if expr.name == var else ZERO

    if isinstance(expr, UnaryOp) and expr.op == "-":
        # d/dx(-f) = -d/dx(f)
        df = differentiate(expr.operand, var)
        return simplify(UnaryOp(op="-", operand=df))

    if isinstance(expr, BinOp):
        df = differentiate(expr.lhs, var)
        dg = differentiate(expr.rhs, var)

        if expr.op == "+":
            # d/dx(f + g) = df + dg
            return simplify(BinOp(op="+", lhs=df, rhs=dg))

        if expr.op == "-":
            # d/dx(f - g) = df - dg
            return simplify(BinOp(op="-", lhs=df, rhs=dg))

        if expr.op == "*":
            # Product rule: d/dx(f*g) = f*dg + g*df
            term1 = BinOp(op="*", lhs=expr.lhs, rhs=dg)
            term2 = BinOp(op="*", lhs=expr.rhs, rhs=df)
            return simplify(BinOp(op="+", lhs=term1, rhs=term2))

        if expr.op == "/":
            # Quotient rule: d/dx(f/g) = (g*df - f*dg) / g²
            numer = BinOp(op="-",
                          lhs=BinOp(op="*", lhs=expr.rhs, rhs=df),
                          rhs=BinOp(op="*", lhs=expr.lhs, rhs=dg))
            denom = BinOp(op="*", lhs=expr.rhs, rhs=expr.rhs)
            return simplify(BinOp(op="/", lhs=numer, rhs=denom))

    if isinstance(expr, Call):
        func = expr.func
        args = expr.args

        if func == "exp" and len(args) == 1:
            # d/dx(exp(f)) = exp(f) * df
            df = differentiate(args[0], var)
            return simplify(BinOp(op="*", lhs=expr, rhs=df))

        if func == "log" and len(args) == 1:
            # d/dx(log(f)) = df / f
            df = differentiate(args[0], var)
            return simplify(BinOp(op="/", lhs=df, rhs=args[0]))

        if func == "sqrt" and len(args) == 1:
            # d/dx(sqrt(f)) = df / (2 * sqrt(f))
            df = differentiate(args[0], var)
            denom = BinOp(op="*", lhs=TWO, rhs=expr)
            return simplify(BinOp(op="/", lhs=df, rhs=denom))

        if func in ("abs", "fabs") and len(args) == 1:
            # d/dx(|f|) = sign(f) * df  →  select(f > 0, df, -df)
            df = differentiate(args[0], var)
            cond = Cmp(op=">", lhs=args[0], rhs=ZERO)
            return simplify(Select(cond=cond, true_val=df,
                                   false_val=UnaryOp(op="-", operand=df)))

        if func == "min" and len(args) == 2:
            # d/dx(min(f,g)) = select(f < g, df, dg)
            df = differentiate(args[0], var)
            dg = differentiate(args[1], var)
            cond = Cmp(op="<", lhs=args[0], rhs=args[1])
            return simplify(Select(cond=cond, true_val=df, false_val=dg))

        if func == "max" and len(args) == 2:
            # d/dx(max(f,g)) = select(f > g, df, dg)
            df = differentiate(args[0], var)
            dg = differentiate(args[1], var)
            cond = Cmp(op=">", lhs=args[0], rhs=args[1])
            return simplify(Select(cond=cond, true_val=df, false_val=dg))

        if func == "pow" and len(args) == 2:
            # d/dx(f^g) = f^g * (g * df/f + dg * log(f))
            f, g = args
            df = differentiate(f, var)
            dg = differentiate(g, var)
            term1 = BinOp(op="*", lhs=g,
                          rhs=BinOp(op="/", lhs=df, rhs=f))
            term2 = BinOp(op="*", lhs=dg,
                          rhs=Call(func="log", args=(f,)))
            inner = BinOp(op="+", lhs=term1, rhs=term2)
            return simplify(BinOp(op="*", lhs=expr, rhs=inner))

        if func == "fma" and len(args) == 3:
            # d/dx(fma(a,b,c)) = d/dx(a*b + c) = a*db + b*da + dc
            a, b, c = args
            da = differentiate(a, var)
            db = differentiate(b, var)
            dc = differentiate(c, var)
            return simplify(BinOp(op="+",
                                  lhs=BinOp(op="+",
                                            lhs=BinOp(op="*", lhs=a, rhs=db),
                                            rhs=BinOp(op="*", lhs=b, rhs=da)),
                                  rhs=dc))

        raise ValueError(f"Cannot differentiate function '{func}'")

    if isinstance(expr, Select):
        # d/dx(select(c, t, f)) = select(c, dt, df)
        # Condition is not differentiable; derive branches independently
        dt = differentiate(expr.true_val, var)
        df = differentiate(expr.false_val, var)
        return simplify(Select(cond=expr.cond, true_val=dt, false_val=df))

    raise ValueError(f"Cannot differentiate node type: {type(expr).__name__}")


# ============================================================================
# Algebraic Simplification
# ============================================================================

def _is_zero(node: Expr) -> bool:
    return isinstance(node, Const) and node.value == 0.0

def _is_one(node: Expr) -> bool:
    return isinstance(node, Const) and node.value == 1.0


def simplify(expr: Expr) -> Expr:
    """Apply algebraic simplification rules to reduce expression size.

    Recursively simplifies sub-expressions and eliminates trivial operations:
      0 + x → x,  x + 0 → x
      0 * x → 0,  x * 0 → 0
      1 * x → x,  x * 1 → x
      x / 1 → x
      0 / x → 0
      -0 → 0
      Const op Const → Const  (constant folding)
    """
    if isinstance(expr, Const) or isinstance(expr, Var):
        return expr

    if isinstance(expr, UnaryOp) and expr.op == "-":
        inner = simplify(expr.operand)
        if _is_zero(inner):
            return ZERO
        if isinstance(inner, Const):
            return Const(value=-inner.value)
        if isinstance(inner, UnaryOp) and inner.op == "-":
            return inner.operand  # -(-x) = x
        return UnaryOp(op="-", operand=inner)

    if isinstance(expr, BinOp):
        lhs = simplify(expr.lhs)
        rhs = simplify(expr.rhs)

        # Constant folding
        if isinstance(lhs, Const) and isinstance(rhs, Const):
            try:
                if expr.op == "+": return Const(value=lhs.value + rhs.value)
                if expr.op == "-": return Const(value=lhs.value - rhs.value)
                if expr.op == "*": return Const(value=lhs.value * rhs.value)
                if expr.op == "/" and rhs.value != 0:
                    return Const(value=lhs.value / rhs.value)
            except (OverflowError, ZeroDivisionError):
                pass

        if expr.op == "+":
            if _is_zero(lhs): return rhs
            if _is_zero(rhs): return lhs
        elif expr.op == "-":
            if _is_zero(rhs): return lhs
            if _is_zero(lhs): return simplify(UnaryOp(op="-", operand=rhs))
        elif expr.op == "*":
            if _is_zero(lhs) or _is_zero(rhs): return ZERO
            if _is_one(lhs): return rhs
            if _is_one(rhs): return lhs
        elif expr.op == "/":
            if _is_zero(lhs): return ZERO
            if _is_one(rhs): return lhs

        return BinOp(op=expr.op, lhs=lhs, rhs=rhs, dtype=expr.dtype)

    if isinstance(expr, Call):
        args = tuple(simplify(a) for a in expr.args)
        return Call(func=expr.func, args=args, dtype=expr.dtype)

    if isinstance(expr, Select):
        cond = simplify(expr.cond)
        tv = simplify(expr.true_val)
        fv = simplify(expr.false_val)
        # If both branches are equal, eliminate select
        if isinstance(tv, Const) and isinstance(fv, Const) and tv.value == fv.value:
            return tv
        return Select(cond=cond, true_val=tv, false_val=fv, dtype=expr.dtype)

    return expr


# ============================================================================
# Jacobian Stamp Generation
# ============================================================================

def resolve_body_expr(model: ModelDef, var_name: str) -> Optional[Expr]:
    """Look up the assignment expression for a variable in the model body.

    This is used to inline intermediate variables before differentiation.
    """
    for assign in model.body:
        if assign.target == var_name:
            return assign.expr
    return None


def inline_intermediates(expr: Expr, model: ModelDef,
                         visited: Optional[Set[str]] = None) -> Expr:
    """Recursively inline intermediate variable assignments.

    Replaces Var references to local assignments with their defining
    expressions, producing a single self-contained expression tree suitable
    for differentiation.
    """
    if visited is None:
        visited = set()

    if isinstance(expr, Const):
        return expr

    if isinstance(expr, Var):
        name = expr.name
        # Don't inline terminal voltages or parameters — they are leaf vars
        param_names = {p.name for p in model.parameters}
        terminal_volts = {f"v_{t.name}" for t in model.terminals}

        if name in param_names or name in terminal_volts:
            return expr

        # Inline intermediate assignments
        if name not in visited:
            body_expr = resolve_body_expr(model, name)
            if body_expr is not None:
                visited.add(name)
                return inline_intermediates(body_expr, model, visited)

        return expr

    if isinstance(expr, BinOp):
        return BinOp(op=expr.op,
                     lhs=inline_intermediates(expr.lhs, model, visited.copy()),
                     rhs=inline_intermediates(expr.rhs, model, visited.copy()),
                     dtype=expr.dtype)

    if isinstance(expr, UnaryOp):
        return UnaryOp(op=expr.op,
                       operand=inline_intermediates(expr.operand, model, visited.copy()),
                       dtype=expr.dtype)

    if isinstance(expr, Call):
        return Call(func=expr.func,
                    args=tuple(inline_intermediates(a, model, visited.copy())
                               for a in expr.args),
                    dtype=expr.dtype)

    if isinstance(expr, Select):
        return Select(
            cond=inline_intermediates(expr.cond, model, visited.copy()),
            true_val=inline_intermediates(expr.true_val, model, visited.copy()),
            false_val=inline_intermediates(expr.false_val, model, visited.copy()),
            dtype=expr.dtype,
        )

    return expr


def derive_jacobian_stamps(model: ModelDef) -> List[StampEntry]:
    """Automatically derive Jacobian stamp entries from the model's current
    contributions.

    For each current-contributing state variable (i_d, ids, etc.) and each
    terminal voltage, computes ∂I/∂V symbolically.

    Returns a list of StampEntry directives.
    """
    stamps: List[StampEntry] = []

    # Identify current-contributing state variables
    # Convention: variables named 'i_*' or 'ids' are branch currents
    current_vars = []
    for sv in model.state_vars:
        if sv.name.startswith("i_") or sv.name == "ids":
            current_vars.append(sv.name)

    if not current_vars:
        return stamps

    # Terminal voltage variable names
    terminal_volts = {f"v_{t.name}": t.name for t in model.terminals}

    for curr_var in current_vars:
        # Get the expression for the current
        curr_expr = resolve_body_expr(model, curr_var)
        if curr_expr is None:
            continue

        # Inline all intermediates to get a single expression tree
        full_expr = inline_intermediates(curr_expr, model)

        for volt_var, terminal_name in terminal_volts.items():
            # Compute ∂I/∂V
            deriv = differentiate(full_expr, volt_var)
            deriv = simplify(deriv)

            # Skip zero derivatives (no dependency on this terminal)
            if _is_zero(deriv):
                continue

            # Determine the row terminal from the current variable
            # Convention: i_d → row is 'd', ids → for MOSFET, row is 'd'
            if curr_var.startswith("i_"):
                row_terminal = curr_var[2:]  # i_d → 'd'
            else:
                # Default: first terminal for the positive contribution
                row_terminal = model.terminals[0].name

            stamps.append(StampEntry(
                row_terminal=row_terminal,
                col_terminal=terminal_name,
                expr=deriv,
                is_rhs=False,
            ))

    return stamps


def derive_rhs_stamps(model: ModelDef) -> List[StampEntry]:
    """Derive RHS contribution stamps from current state variables.

    The RHS contribution for node n from current I is:
      rhs[n] += -(I - G*V)  (companion model)

    where G = ∂I/∂V is the linearized conductance.
    """
    rhs: List[StampEntry] = []

    current_vars = []
    for sv in model.state_vars:
        if sv.name.startswith("i_") or sv.name == "ids":
            current_vars.append(sv.name)

    if not current_vars:
        return rhs

    # For each current-contributing variable, compute the RHS
    for curr_var in current_vars:
        if curr_var.startswith("i_"):
            terminal = curr_var[2:]
        else:
            terminal = model.terminals[0].name

        # Check terminal exists
        if not any(t.name == terminal for t in model.terminals):
            continue

        # RHS = -(i - g*v) for the positive terminal
        # We use the current var and conductance var as references
        curr_ref = Var(name=curr_var)
        rhs.append(StampEntry(
            row_terminal=terminal,
            col_terminal="",
            expr=curr_ref,
            is_rhs=True,
        ))

    return rhs


def print_derivative_report(model: ModelDef):
    """Print a human-readable report of the derived Jacobian entries."""
    stamps = derive_jacobian_stamps(model)

    print(f"=== Jacobian Derivatives for {model.name} ===")
    print(f"  Terminals: {[t.name for t in model.terminals]}")

    for stamp in stamps:
        from .backends.scalar_cpp import emit_expr
        expr_str = emit_expr(stamp.expr)
        print(f"  J[{stamp.row_terminal}, {stamp.col_terminal}] = {expr_str}")

    print(f"  Total stamps: {len(stamps)}")
    return stamps
