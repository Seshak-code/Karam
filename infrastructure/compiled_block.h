#pragma once

#include "../tensors/physics_tensors.h"
#include "../../components/circuit.h"   // TensorNetlist (embedded structural copy)
#include "../../components/design_db.h" // NetUUID
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

/**
 * compiled_block.h
 *
 * CompiledTensorBlock — The immutable, solver-ready artifact produced by
 * NetlistCompiler.  The solver (CircuitSim) depends ONLY on this struct;
 * it never receives a raw TensorNetlist or any GUI types.
 *
 * Design Rationale:
 *   SchematicEditor → TensorNetlist → NetlistCompiler → CompiledTensorBlock
 *   CompiledTensorBlock is cached by SimulationWorkspace and invalidated
 *   only when the topology hash changes.
 *
 * Migration Note:
 *   The structural TensorNetlist is embedded inside this block so that
 *   internal solver paths (stampAllElements, calculatePhysicalResiduals)
 *   can still access it.  These will be incrementally migrated to use
 *   only `tensors`.  External callers MUST NOT access `structural_`.
 */

// ─── Interface Pin Metadata ──────────────────────────────────────────────────
// Populated during subcircuit resolution; describes each boundary pin of a
// compiled hierarchical block. Used by the hierarchy resolver and future
// HierarchyResolver class to route signals across hierarchy boundaries.
struct CompiledInterfacePin {
    acutesim::components::NetUUID netId;   // Net UUID inside the child scene
    int nodeIndex = 0;                     // Solver node index in the child's system
    acutesim::components::PinDirection direction = acutesim::components::PinDirection::Input;
    bool isPower = false;                  // true for PowerPort-type interface pins
    std::string name;                      // Pin name (from Pin/PowerPort component)
    uint32_t stableIndex = 0;             // Original pin order in .sch (deterministic solver binding)
};

struct CompiledTensorBlock {
    // ─── Computational Data (SoA) ───────────────────────────────────────
    TensorizedBlock tensors;

    // ─── Topology Metadata ──────────────────────────────────────────────
    size_t topologyHash = 0;   // Structural fingerprint for cache checks
    size_t nodeCount    = 0;   // Total nodes in the flattened system

    // ─── Simulation Environment Snapshot ─────────────────────────────────
    double ambientTempK         = 300.15;
    double globalVoltageScale   = 1.0;
    int    monteCarloSeed       = 0;
    bool   thermalNoiseEnabled  = false;
    bool   flickerNoiseEnabled  = false;

    // ─── Interface Pin Metadata ──────────────────────────────────────────
    // Populated by the subcircuit resolver for hierarchical blocks.
    std::vector<CompiledInterfacePin> interfacePins;

    // ─── Structural Netlist (Internal — migration shim) ──────────────────
    // TODO: Remove once all solver internals operate on tensors only.
    TensorNetlist structural_;

    // ─── Future GPU Fields (Reserved) ────────────────────────────────────
    // size_t stampOffsets = 0;   // Pre-computed CSR column offsets
    // void*  gpuHandle = nullptr;   // Opaque GPU buffer handle
};
