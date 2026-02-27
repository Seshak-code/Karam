# Verilog-A Parser Architecture (C++)

The `physics/veriloga` directory contains the **Lexical Analyzer** and **AST Generator** for Verilog-A models.

## Data Flow: Source to AST
1.  **Lexer**: Breaks `.va` files into tokens.
2.  **Parser**: Builds an Abstract Syntax Tree (AST) representing the circuit equations.
3.  **Analyzer**: Performs semantic checks (unit consistency, pin mapping).
4.  **Assembler**: Prepares the AST for consumption by the `physics_compiler`.

## Logic: Grammar Compliance
The parser implements a subset of LRM 2.4 for analog modeling, focusing on `analog` blocks and transition/contribution statements.

## 🤖 SME Validation Checklist
- [ ] **Memory Management**: Are AST nodes leaking? (Prefer `std::unique_ptr` for node trees).
- [ ] **Error Reporting**: Does a syntax error provide a line number and meaningful message?
