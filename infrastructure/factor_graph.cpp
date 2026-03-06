#include "factor_graph.h"
#include <algorithm>
#include <set>

// ============================================================================
// FactorGraph::fromTensorizedBlock
// ============================================================================

FactorGraph FactorGraph::fromTensorizedBlock(const TensorizedBlock& tensors,
                                              uint32_t numCircuitNodes)
{
    FactorGraph fg;
    fg.numCircuitNodes = numCircuitNodes;

    // Collect all unique non-ground voltage-node IDs (sorted by std::set).
    std::set<uint32_t> voltageVarSet;
    auto trackVoltage = [&](int node) {
        if (node > 0) voltageVarSet.insert(static_cast<uint32_t>(node));
    };

    // Branch-current IDs start immediately above the circuit-node voltage space.
    // Inductors and voltage sources each introduce one explicit MNA current row.
    uint32_t branchId    = numCircuitNodes;
    uint32_t branchCount = 0;

    const uint32_t nd  = static_cast<uint32_t>(tensors.diodes.size());
    const uint32_t nm  = static_cast<uint32_t>(tensors.mosfets.size());
    const uint32_t nb  = static_cast<uint32_t>(tensors.bjts.size());
    const uint32_t nr  = static_cast<uint32_t>(tensors.resistors.size());
    const uint32_t nc  = static_cast<uint32_t>(tensors.capacitors.size());
    const uint32_t nil = static_cast<uint32_t>(tensors.inductors.size());
    const uint32_t nvs = static_cast<uint32_t>(tensors.voltageSources.size());
    const uint32_t ncs = static_cast<uint32_t>(tensors.currentSources.size());

    // Register factor and update variableToFactors index atomically.
    auto addFactor = [&](FactorNode fn) {
        auto fi = static_cast<uint32_t>(fg.factors.size());
        for (uint32_t vid : fn.connectedVariables)
            fg.variableToFactors[vid].push_back(fi);
        fg.factors.push_back(std::move(fn));
    };

    uint32_t offset = 0;

    // ── Diodes ───────────────────────────────────────────────────────────────
    for (uint32_t i = 0; i < nd; ++i) {
        FactorNode fn;
        fn.type        = FactorDeviceType::Diode;
        fn.deviceIndex = offset + i;
        int a = tensors.diodes.node_a[i], c = tensors.diodes.node_c[i];
        if (a > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(a)); trackVoltage(a); }
        if (c > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(c)); trackVoltage(c); }
        addFactor(std::move(fn));
    }
    offset += nd;

    // ── MOSFETs ───────────────────────────────────────────────────────────────
    for (uint32_t i = 0; i < nm; ++i) {
        FactorNode fn;
        fn.type        = FactorDeviceType::Mosfet;
        fn.deviceIndex = offset + i;
        int d = tensors.mosfets.drains[i],  g = tensors.mosfets.gates[i];
        int s = tensors.mosfets.sources[i], b = tensors.mosfets.bodies[i];
        if (d > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(d)); trackVoltage(d); }
        if (g > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(g)); trackVoltage(g); }
        if (s > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(s)); trackVoltage(s); }
        if (b > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(b)); trackVoltage(b); }
        addFactor(std::move(fn));
    }
    offset += nm;

    // ── BJTs ──────────────────────────────────────────────────────────────────
    for (uint32_t i = 0; i < nb; ++i) {
        FactorNode fn;
        fn.type        = FactorDeviceType::BJT;
        fn.deviceIndex = offset + i;
        int c  = tensors.bjts.collectors[i];
        int bv = tensors.bjts.bases[i];
        int e  = tensors.bjts.emitters[i];
        if (c  > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(c));  trackVoltage(c);  }
        if (bv > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(bv)); trackVoltage(bv); }
        if (e  > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(e));  trackVoltage(e);  }
        addFactor(std::move(fn));
    }
    offset += nb;

    // ── Resistors ─────────────────────────────────────────────────────────────
    for (uint32_t i = 0; i < nr; ++i) {
        FactorNode fn;
        fn.type        = FactorDeviceType::Resistor;
        fn.deviceIndex = offset + i;
        int n1 = tensors.resistors.nodes1[i], n2 = tensors.resistors.nodes2[i];
        if (n1 > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(n1)); trackVoltage(n1); }
        if (n2 > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(n2)); trackVoltage(n2); }
        addFactor(std::move(fn));
    }
    offset += nr;

    // ── Capacitors ────────────────────────────────────────────────────────────
    for (uint32_t i = 0; i < nc; ++i) {
        FactorNode fn;
        fn.type        = FactorDeviceType::Capacitor;
        fn.deviceIndex = offset + i;
        int p1 = tensors.capacitors.plates1[i], p2 = tensors.capacitors.plates2[i];
        if (p1 > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(p1)); trackVoltage(p1); }
        if (p2 > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(p2)); trackVoltage(p2); }
        addFactor(std::move(fn));
    }
    offset += nc;

    // ── Inductors (introduce one branch-current MNA variable each) ────────────
    for (uint32_t i = 0; i < nil; ++i) {
        FactorNode fn;
        fn.type        = FactorDeviceType::Inductor;
        fn.deviceIndex = offset + i;
        int c1 = tensors.inductors.coil1[i], c2 = tensors.inductors.coil2[i];
        if (c1 > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(c1)); trackVoltage(c1); }
        if (c2 > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(c2)); trackVoltage(c2); }
        uint32_t iCurrId = branchId++;
        ++branchCount;
        fn.connectedVariables.push_back(iCurrId);
        addFactor(std::move(fn));
    }
    offset += nil;

    // ── Voltage Sources (introduce one branch-current MNA variable each) ──────
    for (uint32_t i = 0; i < nvs; ++i) {
        FactorNode fn;
        fn.type        = FactorDeviceType::VoltageSource;
        fn.deviceIndex = offset + i;
        int np = tensors.voltageSources.nodesPos[i], nn = tensors.voltageSources.nodesNeg[i];
        if (np > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(np)); trackVoltage(np); }
        if (nn > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(nn)); trackVoltage(nn); }
        uint32_t iCurrId = branchId++;
        ++branchCount;
        fn.connectedVariables.push_back(iCurrId);
        addFactor(std::move(fn));
    }
    offset += nvs;

    // ── Current Sources (Norton stamps — no explicit branch-current row) ───────
    for (uint32_t i = 0; i < ncs; ++i) {
        FactorNode fn;
        fn.type        = FactorDeviceType::CurrentSource;
        fn.deviceIndex = offset + i;
        int np = tensors.currentSources.nodesPos[i], nn = tensors.currentSources.nodesNeg[i];
        if (np > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(np)); trackVoltage(np); }
        if (nn > 0) { fn.connectedVariables.push_back(static_cast<uint32_t>(nn)); trackVoltage(nn); }
        addFactor(std::move(fn));
    }

    fg.numBranchCurrents = branchCount;

    // Build sorted variable list: voltage nodes first (naturally sorted from
    // std::set), then branch-current variables in ID order.
    for (uint32_t vid : voltageVarSet)
        fg.variables.push_back({vid, VariableType::NodeVoltage});
    for (uint32_t k = 0; k < branchCount; ++k)
        fg.variables.push_back({numCircuitNodes + k, VariableType::BranchCurrent});

    return fg;
}

// ============================================================================
// FactorGraph::neighborsOf
// ============================================================================

std::vector<uint32_t> FactorGraph::neighborsOf(
    uint32_t varId,
    const std::unordered_set<uint32_t>& eliminated) const
{
    std::set<uint32_t> nbrs;
    auto it = variableToFactors.find(varId);
    if (it == variableToFactors.end()) return {};
    for (uint32_t fi : it->second) {
        for (uint32_t vid : factors[fi].connectedVariables) {
            if (vid != varId && !eliminated.count(vid))
                nbrs.insert(vid);
        }
    }
    return std::vector<uint32_t>(nbrs.begin(), nbrs.end());
}

// ============================================================================
// FactorGraph::buildPrimalGraph  (diagnostic / unit-test helper)
// ============================================================================

std::unordered_map<uint32_t, std::vector<uint32_t>> FactorGraph::buildPrimalGraph() const
{
    std::unordered_map<uint32_t, std::vector<uint32_t>> adj;
    for (const auto& v : variables)
        adj[v.id]; // initialise all variable nodes (even isolated)

    std::set<uint64_t> edgeSet;
    const std::unordered_set<uint32_t> noElim;
    for (const auto& v : variables) {
        for (uint32_t w : neighborsOf(v.id, noElim)) {
            uint32_t lo = std::min(v.id, w), hi = std::max(v.id, w);
            uint64_t key = (static_cast<uint64_t>(lo) << 32) | hi;
            if (edgeSet.insert(key).second) {
                adj[lo].push_back(hi);
                adj[hi].push_back(lo);
            }
        }
    }
    return adj;
}
