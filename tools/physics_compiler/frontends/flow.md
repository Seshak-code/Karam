# Physics Compiler Frontends

The `scripts/physics_compiler/frontends` directory contains **Parsers** that convert source modeling languages into the compiler's Intermediate Representation (IR).

## Logic: Language Translation
Each frontend is specialized for a specific syntax (e.g., `veriloga_frontend.py`). It handles lexical analysis and semantic mapping to the unified IR defined in `ir.py`.

## 🤖 SME Validation Checklist
- [ ] **IR Compliance**: Does the parser output valid objects from `ir.py`?
- [ ] **Error Handling**: Are syntax errors in the source language mapped to helpful compiler messages?
