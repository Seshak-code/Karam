"""
__main__.py - CLI entry point for the Physics Compiler.

Usage:
    python -m physics_compiler models/diode.yaml --backend scalar_cpp --output compute/generated/
    python -m physics_compiler models/diode.va --backend scalar_cpp --output compute/generated/

Generates C++ headers from YAML or Verilog-A model definitions.
"""

import argparse
import sys
from pathlib import Path

from .frontend import parse_model
from .frontends.veriloga import parse_veriloga
from .ir import (
    enforce_ssa, validate_types, validate_ssa,
    stable_rename_temps, IRValidationError,
)
from .backends.scalar_cpp import generate as generate_scalar_cpp
from .backends.wgsl import generate as generate_wgsl
from .backends.avx512 import generate as generate_avx512
from .backends.avx2 import generate as generate_avx2


def main():
    parser = argparse.ArgumentParser(
        prog="physics_compiler",
        description="Generate C++/WGSL simulation kernels from model definitions.",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to model definition (.yaml, .yml, or .va file).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["scalar_cpp", "avx512", "avx2", "wgsl"],
        default="scalar_cpp",
        help="Code generation backend (default: scalar_cpp).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory or file path. Defaults to stdout.",
    )
    parser.add_argument(
        "--dump-ir",
        action="store_true",
        help="Dump the parsed IR tree to stderr for debugging.",
    )
    parser.add_argument(
        "--auto-jacobian",
        action="store_true",
        help="Auto-derive Jacobian stamps using symbolic differentiation.",
    )
    parser.add_argument(
        "--stable-temp-naming",
        action="store_true",
        help="Rename all local temporaries to deterministic _t0, _t1, ... names.",
    )
    parser.add_argument(
        "--fma",
        action="store_true",
        help="Enable FMA intrinsics in AVX2/AVX-512 backends. Disabling (default) ensures bit-exact parity with scalar.",
    )
    parser.add_argument(
        "--include-header",
        type=str,
        default=None,
        help="Override the generated #include header name (e.g. 'physics_tensors.h').",
    )

    args = parser.parse_args()

    # 1. Parse the model (auto-detect by extension)
    model_path = Path(args.model)
    if model_path.suffix in (".va",):
        model = parse_veriloga(args.model)
    else:
        model = parse_model(args.model)

    # 1b. Optional auto-Jacobian
    if args.auto_jacobian:
        from .jacobian import derive_jacobian_stamps, derive_rhs_stamps
        auto_stamps = derive_jacobian_stamps(model)
        auto_rhs = derive_rhs_stamps(model)
        if auto_stamps:
            model.stamps = auto_stamps
            print(f"[Jacobian] Auto-derived {len(auto_stamps)} Jacobian stamps", file=sys.stderr)
        if auto_rhs:
            model.rhs_stamps = auto_rhs
            print(f"[Jacobian] Auto-derived {len(auto_rhs)} RHS stamps", file=sys.stderr)

    # 1c. IR Hardening Passes
    try:
        model = enforce_ssa(model)
        validate_types(model)
        validate_ssa(model)
        if args.stable_temp_naming:
            model = stable_rename_temps(model)
            print("[SSA] Applied stable temp naming", file=sys.stderr)
        print(f"[IR] Validated: F64-only SSA, {len(model.body)} statements", file=sys.stderr)
    except IRValidationError as e:
        print(f"[IR ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Optional IR dump
    if args.dump_ir:
        print(f"--- IR Dump: {model.name} ---", file=sys.stderr)
        print(f"  Layout Hash: {model.layout_hash()}", file=sys.stderr)
        print(f"  Terminals: {[t.name for t in model.terminals]}", file=sys.stderr)
        print(f"  Parameters: {[p.name for p in model.parameters]}", file=sys.stderr)
        print(f"  State Vars: {[s.name for s in model.state_vars]}", file=sys.stderr)
        print(f"  Body ({len(model.body)} statements):", file=sys.stderr)
        for a in model.body:
            print(f"    {a.target} = ..", file=sys.stderr)
        print(f"  Jacobian Stamps: {len(model.stamps)}", file=sys.stderr)
        print(f"  RHS Stamps: {len(model.rhs_stamps)}", file=sys.stderr)
        print(f"--- End IR Dump ---", file=sys.stderr)

    # 3. Generate code
    if args.backend == "scalar_cpp":
        code = generate_scalar_cpp(model)
    elif args.backend == "avx512":
        code = generate_avx512(model)
    elif args.backend == "avx2":
        code = generate_avx2(model, use_fma=args.fma, include_header=args.include_header)
    elif args.backend == "wgsl":
        code = generate_wgsl(model)

    # 4. Output
    if args.output:
        out_path = Path(args.output)
        if out_path.is_dir():
            out_path = out_path / f"gen_{model.name.lower()}.h"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(code)
        print(f"[OK] Generated: {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(code)


if __name__ == "__main__":
    main()

