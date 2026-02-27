#include "netlist_compiler.h"

/**
 * netlist_compiler_ctx.cpp
 *
 * Variable-aware compilation — kept separate so that targets which don't
 * use parametric sweeps (e.g. unit tests) don't pull in simulation_context.cpp.
 */

std::shared_ptr<const CompiledTensorBlock>
NetlistCompiler::compile(const TensorNetlist& netlist, const acutesim::orchestration::SimulationContext& ctx)
{
    // Helper lambda to resolve a single double via the context evaluator
    auto resolve = [&](double& val, const std::string& expr) {
        if (expr.empty()) return;
        val = ctx.evaluate(expr);
    };

    // Make a mutable copy so we don't modify the caller's netlist
    auto resolved = netlist;

    for (auto& r : resolved.globalBlock.resistors)
        resolve(r.resistance_ohms, r.expression);
    for (auto& c : resolved.globalBlock.capacitors)
        resolve(c.capacitance_farads, c.expression);
    for (auto& l : resolved.globalBlock.inductors)
        resolve(l.inductance_henries, l.expression);
    for (auto& v : resolved.globalBlock.voltageSources)
        resolve(v.voltage_V, v.expression);
    for (auto& i : resolved.globalBlock.currentSources)
        resolve(i.current_A, i.expression);

    return compile(resolved);
}
