#pragma once
#include "../tensors/physics_tensors.h"
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * factor_graph.h — Factor Graph IR (Phase A Analytical Topology Engine)
 *
 * Models the MNA linear system as a bipartite graph:
 *   variable nodes (MNA unknowns) ↔ factor nodes (circuit devices)
 *
 * MNA variables are:
 *   • Circuit-node voltages  (IDs 1 .. numCircuitNodes-1, ground=0 excluded)
 *   • Branch-current unknowns for voltage sources and inductors
 *     (IDs starting at numCircuitNodes, assigned sequentially)
 *
 * Including branch-current variables gives the mathematically correct
 * treewidth of the full MNA Jacobian, not just the voltage sub-system.
 *
 * Adjacency is stored lazily via variableToFactors: two variables are
 * adjacent in the primal graph iff they appear in at least one common factor.
 * This avoids materialising O(k²) clique edges upfront for high-degree
 * devices (MOSFETs, PDN macro models).
 *
 * Flat device encoding in FactorNode::deviceIndex:
 *   [0,                nd)             → Diodes
 *   [nd,               nd+nm)          → MOSFETs
 *   [nd+nm,            nd+nm+nb)       → BJTs
 *   [nd+nm+nb,         ...+nr)         → Resistors
 *   [nd+nm+nb+nr,      ...+nc)         → Capacitors
 *   [nd+nm+nb+nr+nc,   ...+nil)        → Inductors
 *   [nd+nm+nb+nr+nc+nil, ...+nvs)      → Voltage Sources
 *   [nd+nm+nb+nr+nc+nil+nvs, ...+ncs)  → Current Sources
 */

enum class FactorDeviceType {
    Resistor, Capacitor, Inductor, VoltageSource,
    CurrentSource, Diode, Mosfet, BJT
};

// MNA variable types: either a circuit-node voltage or an explicit branch
// current introduced by a voltage-source or inductor MNA stamp.
enum class VariableType { NodeVoltage, BranchCurrent };

struct VariableNode {
    uint32_t   id;    // Unique variable ID
    VariableType type;
};

struct FactorNode {
    FactorDeviceType       type;
    uint32_t               deviceIndex;        // flat encoding (see above)
    std::vector<uint32_t>  connectedVariables; // variable IDs (voltages + branch currents)
};

// Bipartite factor graph for the MNA topology.
// Build via fromTensorizedBlock(); then pass to TreewidthAnalyzer::analyze().
class FactorGraph {
public:
    std::vector<VariableNode>  variables;           // all MNA unknowns (sorted by id)
    std::vector<FactorNode>    factors;
    uint32_t                   numCircuitNodes   = 0;
    uint32_t                   numBranchCurrents = 0;

    // Lazy adjacency index: variable id → indices into factors[].
    // Enables O(k) neighbor lookup without pre-materialising the primal graph.
    std::unordered_map<uint32_t, std::vector<uint32_t>> variableToFactors;

    // Primary factory: reads all device arrays from TensorizedBlock.
    // numCircuitNodes must equal CompiledTensorBlock::nodeCount (used for
    // branch-current ID assignment starting above the voltage variable space).
    static FactorGraph fromTensorizedBlock(const TensorizedBlock& tensors,
                                            uint32_t numCircuitNodes);

    // Returns co-factor neighbours of varId, excluding any variable in
    // `eliminated`. Result is sorted and deduplicated.
    std::vector<uint32_t> neighborsOf(
        uint32_t varId,
        const std::unordered_set<uint32_t>& eliminated) const;

    // Build the full explicit primal graph (for diagnostics / unit tests).
    // For performance-critical code, use variableToFactors + neighborsOf().
    std::unordered_map<uint32_t, std::vector<uint32_t>> buildPrimalGraph() const;

    bool   empty()         const { return factors.empty(); }
    size_t variableCount() const { return variables.size(); }
    size_t factorCount()   const { return factors.size(); }
};
