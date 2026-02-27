# AcuteSim Engine Project Shell

The `acutesim_engine` directory is the **Future Repository Root** for the standalone Tensor MNA Engine.

## Architectural Goal: Repo-Split Readiness
This folder is currently a shell, designed to become the top-level directory when the engine is split into its own repository via `git filter-repo`.

## Logic: Pure Physics
Once populated, this directory will host the `main.cpp` for the headless CLI simulator and the CMake build system for `libacutesim_engine.a`.

## 🤖 SME Validation Checklist
- [ ] **Engine-Only**: Are any GUI or Connectivity headers present? (Forbidden: This folder must remain "Engine-Pure").
- [ ] **Standalone Build**: Does the code in this folder build without any parent context?
