#pragma once

#include "compiled_block.h"
#include "engine_api/simulation_context.h"
#include <memory>

// Forward declaration — full definition in circuit.h
class TensorNetlist;

/**
 * netlist_compiler.h
 *
 * NetlistCompiler converts a structural TensorNetlist into an immutable
 * CompiledTensorBlock ready for the solver.
 *
 * This is the single point where structural → computational conversion
 * happens.  The result is cached by the orchestration layer
 * (SimulationWorkspace) and shared with the solver via shared_ptr.
 */

class NetlistCompiler {
public:
    /**
     * Compile a structural TensorNetlist into a solver-ready block.
     *
     * Thread Safety: This function is stateless and re-entrant.
     * The returned shared_ptr is const so the solver cannot mutate
     * the compiled topology.
     */
    static std::shared_ptr<const CompiledTensorBlock>
    compile(const TensorNetlist& netlist);

    /**
     * Variable-Aware Compilation.
     * 
     * Resolves all symbolic parameters in the TensorNetlist (e.g. R="Rload")
     * using the values provided in the SimulationContext.
     * 
     * This ensures the resulting CompiledTensorBlock contains only raw doubles,
     * keeping the solver fast and numeric-only.
     */
    static std::shared_ptr<const CompiledTensorBlock>
    compile(const TensorNetlist& netlist, const acutesim::orchestration::SimulationContext& ctx);
};
