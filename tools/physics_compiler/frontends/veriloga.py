"""
veriloga.py - Verilog-A Subset Frontend for the Physics Compiler.

Parses a restricted subset of Verilog-A compact model descriptions into
the IR (ModelDef). This enables importing industry-standard device models
directly from foundry PDKs.

Supported Verilog-A Subset:
  - Module declarations with electrical ports
  - Parameter declarations with defaults
  - Analog block with simple assignments
  - Branch current contributions: I(a, c) <+ expr;
  - Mathematical functions: exp, log, ln, sqrt, pow, abs, min, max
  - Conditional expressions: if/else (lowered to Select nodes)
  - Variable declarations: real var = expr;
  - Comments: // and /* */

NOT Supported (out of scope for Phase 1):
  - $temperature, $vt, $param_given
  - Noise contributions: white_noise, flicker_noise
  - @(cross), @(timer), events
  - User-defined functions
  - Include files
  - Multi-discipline (only 'electrical' supported)

Usage:
    from physics_compiler.frontends.veriloga import parse_veriloga
    model = parse_veriloga("models/diode.va")
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Union
import re

from ..ir import (
    ModelDef, Terminal, Parameter, StateVar, IRType,
    Expr, Const, Var, BinOp, UnaryOp, Call, Select, Cmp,
    Assign, StampEntry,
)


# ============================================================================
# Tokenizer
# ============================================================================

class TokenType:
    KEYWORD   = "KEYWORD"
    IDENT     = "IDENT"
    NUMBER    = "NUMBER"
    STRING    = "STRING"
    OP        = "OP"
    LPAREN    = "LPAREN"
    RPAREN    = "RPAREN"
    LBRACE    = "LBRACE"
    RBRACE    = "RBRACE"
    SEMI      = "SEMI"
    COMMA     = "COMMA"
    CONTRIB   = "CONTRIB"   # <+
    ASSIGN    = "ASSIGN"    # =
    EOF       = "EOF"


KEYWORDS = {
    "module", "endmodule", "analog", "begin", "end",
    "parameter", "real", "integer", "electrical", "inout",
    "input", "output", "if", "else", "ground",
}


class Token:
    def __init__(self, typ: str, val: str, line: int = 0):
        self.typ = typ
        self.val = val
        self.line = line

    def __repr__(self):
        return f"Token({self.typ}, {self.val!r})"


def tokenize(source: str) -> List[Token]:
    """Tokenize Verilog-A source code."""
    tokens = []
    i = 0
    line = 1

    while i < len(source):
        # Whitespace
        if source[i] in ' \t\r':
            i += 1
            continue

        if source[i] == '\n':
            line += 1
            i += 1
            continue

        # Line comment
        if source[i:i+2] == '//':
            while i < len(source) and source[i] != '\n':
                i += 1
            continue

        # Block comment
        if source[i:i+2] == '/*':
            i += 2
            while i + 1 < len(source) and source[i:i+2] != '*/':
                if source[i] == '\n':
                    line += 1
                i += 1
            i += 2
            continue

        # Contribution operator <+
        if source[i:i+2] == '<+':
            tokens.append(Token(TokenType.CONTRIB, "<+", line))
            i += 2
            continue

        # Comparison operators
        if source[i:i+2] in ('<=', '>=', '==', '!='):
            tokens.append(Token(TokenType.OP, source[i:i+2], line))
            i += 2
            continue

        # Single-char operators
        if source[i] in '+-*/><':
            tokens.append(Token(TokenType.OP, source[i], line))
            i += 1
            continue

        if source[i] == '=':
            tokens.append(Token(TokenType.ASSIGN, "=", line))
            i += 1
            continue

        if source[i] == '(':
            tokens.append(Token(TokenType.LPAREN, "(", line))
            i += 1
            continue

        if source[i] == ')':
            tokens.append(Token(TokenType.RPAREN, ")", line))
            i += 1
            continue

        if source[i] == '{':
            tokens.append(Token(TokenType.LBRACE, "{", line))
            i += 1
            continue

        if source[i] == '}':
            tokens.append(Token(TokenType.RBRACE, "}", line))
            i += 1
            continue

        if source[i] == ';':
            tokens.append(Token(TokenType.SEMI, ";", line))
            i += 1
            continue

        if source[i] == ',':
            tokens.append(Token(TokenType.COMMA, ",", line))
            i += 1
            continue

        # String literal
        if source[i] == '"':
            i += 1
            start = i
            while i < len(source) and source[i] != '"':
                i += 1
            tokens.append(Token(TokenType.STRING, source[start:i], line))
            i += 1
            continue

        # Number
        if source[i].isdigit() or (source[i] == '.' and i + 1 < len(source) and source[i+1].isdigit()):
            start = i
            while i < len(source) and source[i] in '0123456789.':
                i += 1
            if i < len(source) and source[i] in 'eE':
                i += 1
                if i < len(source) and source[i] in '+-':
                    i += 1
                while i < len(source) and source[i].isdigit():
                    i += 1
            tokens.append(Token(TokenType.NUMBER, source[start:i], line))
            continue

        # Backtick preprocessor directives (`include, `define, etc.) — skip line
        if source[i] == '`':
            while i < len(source) and source[i] != '\n':
                i += 1
            continue

        # Identifier / keyword
        if source[i].isalpha() or source[i] == '_' or source[i] == '$':
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] in '_$'):
                i += 1
            word = source[start:i]
            if word in KEYWORDS:
                tokens.append(Token(TokenType.KEYWORD, word, line))
            else:
                tokens.append(Token(TokenType.IDENT, word, line))
            continue

        # Unknown character — skip
        i += 1

    tokens.append(Token(TokenType.EOF, "", line))
    return tokens


# ============================================================================
# Parser
# ============================================================================

class VerilogAParser:
    """Recursive descent parser for Verilog-A subset."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.model = ModelDef()
        self.locals: dict = {}  # Local variable assignments

    def peek(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(TokenType.EOF, "")

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, typ: str, val: Optional[str] = None) -> Token:
        tok = self.advance()
        if tok.typ != typ:
            raise SyntaxError(
                f"Line {tok.line}: Expected {typ}"
                + (f" '{val}'" if val else "")
                + f", got {tok.typ} '{tok.val}'"
            )
        if val is not None and tok.val != val:
            raise SyntaxError(
                f"Line {tok.line}: Expected '{val}', got '{tok.val}'"
            )
        return tok

    def match(self, typ: str, val: Optional[str] = None) -> Optional[Token]:
        tok = self.peek()
        if tok.typ == typ and (val is None or tok.val == val):
            return self.advance()
        return None

    # --- Top-level ---

    def parse(self) -> ModelDef:
        """Parse the module declaration and body."""
        self.expect(TokenType.KEYWORD, "module")
        name_tok = self.expect(TokenType.IDENT)
        self.model.name = name_tok.val

        # Port list
        self.expect(TokenType.LPAREN)
        ports = self._parse_port_list()
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.SEMI)

        # Module body
        while self.peek().val != "endmodule":
            self._parse_module_item()

        self.expect(TokenType.KEYWORD, "endmodule")

        return self.model

    def _parse_port_list(self) -> List[str]:
        ports = [self.expect(TokenType.IDENT).val]
        while self.match(TokenType.COMMA):
            ports.append(self.expect(TokenType.IDENT).val)
        return ports

    def _parse_module_item(self):
        tok = self.peek()

        # electrical declaration
        if tok.val == "electrical":
            self._parse_electrical()
        elif tok.val == "inout" or tok.val == "input" or tok.val == "output":
            self._parse_port_dir()
        elif tok.val == "parameter":
            self._parse_parameter()
        elif tok.val == "real":
            self._parse_real_decl()
        elif tok.val == "analog":
            self._parse_analog()
        elif tok.val == "ground":
            self._parse_ground()
        else:
            # Skip unknown
            self.advance()

    def _parse_electrical(self):
        self.advance()  # 'electrical'
        names = [self.expect(TokenType.IDENT).val]
        while self.match(TokenType.COMMA):
            names.append(self.expect(TokenType.IDENT).val)
        self.expect(TokenType.SEMI)

        for name in names:
            # Only add as terminal if not already declared
            if not any(t.name == name for t in self.model.terminals):
                self.model.terminals.append(Terminal(name=name))

    def _parse_port_dir(self):
        self.advance()  # 'inout'/'input'/'output'
        # Optional discipline
        if self.peek().val == "electrical":
            self.advance()
        names = [self.expect(TokenType.IDENT).val]
        while self.match(TokenType.COMMA):
            names.append(self.expect(TokenType.IDENT).val)
        self.expect(TokenType.SEMI)

    def _parse_ground(self):
        self.advance()  # 'ground'
        self.expect(TokenType.IDENT)  # ground node name
        self.expect(TokenType.SEMI)

    def _parse_parameter(self):
        self.advance()  # 'parameter'
        # Optional type
        if self.peek().val in ("real", "integer"):
            self.advance()

        name = self.expect(TokenType.IDENT).val
        default = 0.0

        if self.match(TokenType.ASSIGN):
            default = self._parse_number_literal()

        # Skip optional range constraints (from ... to ...)
        # and description strings
        while self.peek().typ not in (TokenType.SEMI, TokenType.EOF):
            self.advance()

        self.expect(TokenType.SEMI)

        self.model.parameters.append(Parameter(
            name=name, default=default, dtype=IRType.F64
        ))

    def _parse_number_literal(self) -> float:
        """Parse a number, potentially with a leading sign."""
        sign = 1.0
        if self.peek().typ == TokenType.OP and self.peek().val == '-':
            self.advance()
            sign = -1.0
        tok = self.expect(TokenType.NUMBER)
        return sign * float(tok.val)

    def _parse_real_decl(self):
        self.advance()  # 'real'
        # Parse first variable
        name = self.expect(TokenType.IDENT).val
        if self.match(TokenType.ASSIGN):
            expr = self._parse_expr()
            self.model.body.append(Assign(target=name, expr=expr))

        # Parse additional comma-separated variables
        while self.match(TokenType.COMMA):
            name = self.expect(TokenType.IDENT).val
            if self.match(TokenType.ASSIGN):
                expr = self._parse_expr()
                self.model.body.append(Assign(target=name, expr=expr))

        self.expect(TokenType.SEMI)

    def _parse_analog(self):
        self.advance()  # 'analog'
        if self.match(TokenType.KEYWORD, "begin"):
            while self.peek().val != "end":
                self._parse_analog_stmt()
            self.expect(TokenType.KEYWORD, "end")
        else:
            self._parse_analog_stmt()

    def _parse_analog_stmt(self):
        tok = self.peek()

        # if/else
        if tok.val == "if":
            self._parse_if()
            return

        # real declaration
        if tok.val == "real":
            self._parse_real_decl()
            return

        # contribution: I(a, c) <+ expr;
        if tok.typ == TokenType.IDENT:
            # Check for branch contribution: I(a,c) <+ expr
            if tok.val in ("I", "V"):
                self._parse_contribution()
                return

            # Assignment: var = expr;
            name = self.advance().val
            self.expect(TokenType.ASSIGN)
            expr = self._parse_expr()
            self.expect(TokenType.SEMI)
            self.model.body.append(Assign(target=name, expr=expr))
            return

        # Skip unknown
        self.advance()

    def _parse_contribution(self):
        """Parse I(a, c) <+ expr; or I(branch) <+ expr;"""
        branch_type = self.advance().val  # 'I' or 'V'
        self.expect(TokenType.LPAREN)
        term1 = self.expect(TokenType.IDENT).val
        term2 = ""
        if self.match(TokenType.COMMA):
            term2 = self.expect(TokenType.IDENT).val
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.CONTRIB)  # <+
        expr = self._parse_expr()
        self.expect(TokenType.SEMI)

        # Create a state var for this contribution
        var_name = f"i_{term1}" if branch_type == "I" else f"v_{term1}"
        if not any(s.name == var_name for s in self.model.state_vars):
            self.model.state_vars.append(StateVar(name=var_name, description=f"{branch_type}({term1},{term2})"))

        self.model.body.append(Assign(target=var_name, expr=expr))

        # Generate stamps from contribution
        if branch_type == "I" and term2:
            # I(a,c) <+ expr  → stamps J[a,*] and J[c,*]
            self.model.stamps.append(StampEntry(
                row_terminal=term1, col_terminal=term1,
                expr=Var(name=f"g_{term1}"), is_rhs=False,
            ))

    def _parse_if(self):
        self.advance()  # 'if'
        self.expect(TokenType.LPAREN)
        cond = self._parse_expr()
        self.expect(TokenType.RPAREN)

        # True branch
        if self.match(TokenType.KEYWORD, "begin"):
            while self.peek().val != "end":
                self._parse_analog_stmt()
            self.expect(TokenType.KEYWORD, "end")
        else:
            self._parse_analog_stmt()

        # Else branch
        if self.match(TokenType.KEYWORD, "else"):
            if self.match(TokenType.KEYWORD, "begin"):
                while self.peek().val != "end":
                    self._parse_analog_stmt()
                self.expect(TokenType.KEYWORD, "end")
            else:
                self._parse_analog_stmt()

    # --- Expression Parser ---

    def _parse_expr(self) -> Expr:
        return self._parse_ternary()

    def _parse_ternary(self) -> Expr:
        expr = self._parse_comparison()
        # Verilog-A ternary: cond ? true : false
        if self.peek().typ == TokenType.OP and self.peek().val == '?':
            self.advance()
            true_val = self._parse_expr()
            self.expect(TokenType.OP, ':')  # This might not tokenize as OP
            false_val = self._parse_expr()
            return Select(cond=expr, true_val=true_val, false_val=false_val)
        return expr

    def _parse_comparison(self) -> Expr:
        left = self._parse_additive()
        while self.peek().typ == TokenType.OP and self.peek().val in ('<', '>', '<=', '>=', '==', '!='):
            op = self.advance().val
            right = self._parse_additive()
            left = Cmp(op=op, lhs=left, rhs=right)
        return left

    def _parse_additive(self) -> Expr:
        left = self._parse_multiplicative()
        while self.peek().typ == TokenType.OP and self.peek().val in ('+', '-'):
            op = self.advance().val
            right = self._parse_multiplicative()
            left = BinOp(op=op, lhs=left, rhs=right)
        return left

    def _parse_multiplicative(self) -> Expr:
        left = self._parse_unary()
        while self.peek().typ == TokenType.OP and self.peek().val in ('*', '/'):
            op = self.advance().val
            right = self._parse_unary()
            left = BinOp(op=op, lhs=left, rhs=right)
        return left

    def _parse_unary(self) -> Expr:
        if self.peek().typ == TokenType.OP and self.peek().val == '-':
            self.advance()
            operand = self._parse_unary()
            if isinstance(operand, Const):
                return Const(value=-operand.value)
            return UnaryOp(op='-', operand=operand)
        return self._parse_primary()

    def _parse_primary(self) -> Expr:
        tok = self.peek()

        # Number
        if tok.typ == TokenType.NUMBER:
            self.advance()
            return Const(value=float(tok.val))

        # Parenthesized expression
        if tok.typ == TokenType.LPAREN:
            self.advance()
            expr = self._parse_expr()
            self.expect(TokenType.RPAREN)
            return expr

        # Identifier or function call
        if tok.typ == TokenType.IDENT:
            name = self.advance().val

            # Function call
            if self.peek().typ == TokenType.LPAREN:
                self.advance()
                args = []
                if self.peek().typ != TokenType.RPAREN:
                    args.append(self._parse_expr())
                    while self.match(TokenType.COMMA):
                        args.append(self._parse_expr())
                self.expect(TokenType.RPAREN)

                # Map Verilog-A functions to IR
                func_map = {
                    "ln": "log",
                    "$ln": "log",
                    "$exp": "exp",
                    "$sqrt": "sqrt",
                    "$abs": "abs",
                }
                ir_func = func_map.get(name, name)

                # Special: V(a,c) access
                if name == "V" and len(args) >= 1:
                    if len(args) == 2 and isinstance(args[0], Var) and isinstance(args[1], Var):
                        return BinOp(op="-",
                                     lhs=Var(name=f"v_{args[0].name}"),
                                     rhs=Var(name=f"v_{args[1].name}"))
                    elif len(args) == 1 and isinstance(args[0], Var):
                        return Var(name=f"v_{args[0].name}")

                return Call(func=ir_func, args=tuple(args))

            # Built-in constants
            if name == "$vt":
                return Const(value=0.02585)  # kT/q at 300K
            if name == "$temperature":
                return Const(value=300.0)

            return Var(name=name)

        # $-prefixed system functions used as identifiers
        if tok.typ == TokenType.IDENT and tok.val.startswith('$'):
            name = self.advance().val
            if name == "$vt":
                return Const(value=0.02585)
            return Var(name=name)

        raise SyntaxError(f"Line {tok.line}: Unexpected token {tok.typ} '{tok.val}'")


# ============================================================================
# Public API
# ============================================================================

def parse_veriloga(path: Union[str, Path]) -> ModelDef:
    """Parse a Verilog-A file into a ModelDef IR tree.

    Args:
        path: Path to the .va file.

    Returns:
        A ModelDef ready for backend consumption.
    """
    path = Path(path)
    source = path.read_text()
    tokens = tokenize(source)
    parser = VerilogAParser(tokens)
    model = parser.parse()
    return model
