"""
frontend.py - YAML Frontend for the Physics Compiler.

Parses a YAML model definition file and produces a ModelDef IR tree.

Usage:
    from physics_compiler.frontend import parse_model
    model = parse_model("models/diode.yaml")
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml

from .ir import (
    ModelDef, Terminal, Parameter, StateVar, IRType,
    Expr, Const, Var, BinOp, UnaryOp, Call, Select,
    Assign, StampEntry,
)


# ============================================================================
# Expression Parser (Recursive Descent)
# ============================================================================

class ExprParser:
    """Parses a simple math expression string into an IR Expr tree.

    Supported syntax:
        expr     → additive
        additive → multiplicative (('+' | '-') multiplicative)*
        multiplicative → unary (('*' | '/') unary)*
        unary    → '-' unary | primary
        primary  → NUMBER | IDENT | IDENT '(' args ')' | '(' expr ')'
        args     → expr (',' expr)*

    Examples:
        "Is * (exp(v_d / (N * Vt)) - 1.0)"
        "max(GMIN, g_base)"
        "-g_d"
    """

    def __init__(self, text: str, known_vars: set):
        self.text = text.strip()
        self.pos = 0
        self.known_vars = known_vars

    def parse(self) -> Expr:
        result = self._additive()
        self._skip_whitespace()
        if self.pos < len(self.text):
            raise SyntaxError(
                f"Unexpected character at position {self.pos}: "
                f"'{self.text[self.pos:]}'"
            )
        return result

    # --- Recursive descent ---

    def _additive(self) -> Expr:
        left = self._multiplicative()
        while True:
            self._skip_whitespace()
            if self._match('+'):
                right = self._multiplicative()
                left = BinOp(op='+', lhs=left, rhs=right)
            elif self._match('-'):
                right = self._multiplicative()
                left = BinOp(op='-', lhs=left, rhs=right)
            else:
                break
        return left

    def _multiplicative(self) -> Expr:
        left = self._unary()
        while True:
            self._skip_whitespace()
            if self._match('*'):
                right = self._unary()
                left = BinOp(op='*', lhs=left, rhs=right)
            elif self._match('/'):
                right = self._unary()
                left = BinOp(op='/', lhs=left, rhs=right)
            else:
                break
        return left

    def _unary(self) -> Expr:
        self._skip_whitespace()
        if self._match('-'):
            operand = self._unary()
            # Optimize: -Const(x) -> Const(-x)
            if isinstance(operand, Const):
                return Const(value=-operand.value)
            return UnaryOp(op='-', operand=operand)
        return self._primary()

    def _primary(self) -> Expr:
        self._skip_whitespace()

        # Parenthesized expression
        if self._match('('):
            expr = self._additive()
            self._skip_whitespace()
            if not self._match(')'):
                raise SyntaxError(f"Expected ')' at position {self.pos}")
            return expr

        # Number literal
        num = self._try_number()
        if num is not None:
            return Const(value=num)

        # Identifier or function call
        ident = self._try_ident()
        if ident is not None:
            self._skip_whitespace()
            if self._match('('):
                # Function call
                args = self._parse_args()
                self._skip_whitespace()
                if not self._match(')'):
                    raise SyntaxError(
                        f"Expected ')' after function args at position {self.pos}"
                    )
                return Call(func=ident, args=tuple(args))
            else:
                # Variable reference
                return Var(name=ident)

        raise SyntaxError(
            f"Unexpected token at position {self.pos}: '{self.text[self.pos:self.pos+10]}'"
        )

    def _parse_args(self) -> List[Expr]:
        args = []
        self._skip_whitespace()
        if self.pos < len(self.text) and self.text[self.pos] == ')':
            return args  # Empty args
        args.append(self._additive())
        while self._match(','):
            args.append(self._additive())
        return args

    # --- Lexer helpers ---

    def _skip_whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos] in ' \t\n\r':
            self.pos += 1

    def _match(self, ch: str) -> bool:
        self._skip_whitespace()
        if self.pos < len(self.text) and self.text[self.pos] == ch:
            self.pos += 1
            return True
        return False

    def _try_number(self) -> Optional[float]:
        self._skip_whitespace()
        start = self.pos
        # Handle optional leading sign (only for standalone literals, not sub-expressions)
        if self.pos < len(self.text) and self.text[self.pos] in '0123456789.':
            while self.pos < len(self.text) and self.text[self.pos] in '0123456789.':
                self.pos += 1
            # Scientific notation: e.g. 1.0e-14
            if self.pos < len(self.text) and self.text[self.pos] in 'eE':
                self.pos += 1
                if self.pos < len(self.text) and self.text[self.pos] in '+-':
                    self.pos += 1
                while self.pos < len(self.text) and self.text[self.pos] in '0123456789':
                    self.pos += 1
            try:
                return float(self.text[start:self.pos])
            except ValueError:
                self.pos = start
                return None
        return None

    def _try_ident(self) -> Optional[str]:
        self._skip_whitespace()
        start = self.pos
        if self.pos < len(self.text) and (self.text[self.pos].isalpha() or self.text[self.pos] == '_'):
            while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
                self.pos += 1
            return self.text[start:self.pos]
        return None


# ============================================================================
# YAML Frontend
# ============================================================================

def _collect_known_vars(data: dict) -> set:
    """Collect all variable names available in the model scope."""
    names = set()

    # Terminal voltages: v_<terminal_name>
    for t in data.get("terminals", []):
        name = t if isinstance(t, str) else t.get("name", "")
        names.add(f"v_{name}")

    # Parameters
    for p in data.get("parameters", []):
        name = p if isinstance(p, str) else p.get("name", "")
        names.add(name)

    # State vars
    for s in data.get("state_vars", []):
        name = s if isinstance(s, str) else s.get("name", "")
        names.add(name)

    # Kernel intermediates (assigned variables)
    for step in data.get("kernel", []):
        if "assign" in step:
            names.add(step["assign"])

    return names


def _parse_expr(text: str, known_vars: set) -> Expr:
    """Parse an expression string into an IR Expr."""
    parser = ExprParser(text, known_vars)
    return parser.parse()


def parse_model(path: Union[str, Path]) -> ModelDef:
    """Parse a YAML model definition file into a ModelDef IR tree.

    Args:
        path: Path to the YAML file.

    Returns:
        A fully populated ModelDef ready for backend consumption.
    """
    path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    known_vars = _collect_known_vars(data)

    model = ModelDef(
        name=data.get("name", path.stem),
        description=data.get("description", ""),
    )

    # --- Terminals ---
    for t in data.get("terminals", []):
        if isinstance(t, str):
            model.terminals.append(Terminal(name=t))
        else:
            model.terminals.append(Terminal(
                name=t["name"],
                description=t.get("description", ""),
            ))

    # --- Parameters ---
    for p in data.get("parameters", []):
        if isinstance(p, str):
            model.parameters.append(Parameter(name=p))
        else:
            model.parameters.append(Parameter(
                name=p["name"],
                dtype=IRType.F64,
                default=float(p.get("default", 0.0)),
                description=p.get("description", ""),
                unit=p.get("unit", ""),
            ))

    # --- State Variables ---
    for s in data.get("state_vars", []):
        if isinstance(s, str):
            model.state_vars.append(StateVar(name=s))
        else:
            model.state_vars.append(StateVar(
                name=s["name"],
                description=s.get("description", ""),
            ))

    # --- Kernel Body ---
    for step in data.get("kernel", []):
        target = step["assign"]
        expr = _parse_expr(step["expr"], known_vars)
        model.body.append(Assign(target=target, expr=expr))

    # --- Jacobian Stamps ---
    stamps_data = data.get("stamps", {})
    for entry in stamps_data.get("jacobian", []):
        expr = _parse_expr(entry["expr"], known_vars)
        model.stamps.append(StampEntry(
            row_terminal=entry["row"],
            col_terminal=entry["col"],
            expr=expr,
            is_rhs=False,
        ))

    for entry in stamps_data.get("rhs", []):
        expr = _parse_expr(entry["expr"], known_vars)
        model.rhs_stamps.append(StampEntry(
            row_terminal=entry["node"],
            col_terminal="",
            expr=expr,
            is_rhs=True,
        ))

    return model
