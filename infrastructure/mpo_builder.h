#pragma once

#include "factor_graph.h"
#include "../tensors/physics_tensors.h"
#include <cstdint>
#include <vector>

/**
 * mpo_builder.h — Device-Local Jacobian Tensors (Phase 5.3.1)
 *
 * A Matrix Product Operator (MPO) encodes a single device's local Jacobian
 * contribution as a dense k×k matrix where k = number of non-ground terminals.
 *
 * For a resistor (2-terminal): 2×2 conductance matrix [[G,-G],[-G,G]]
 * For a diode (2-terminal):    2×2 linearized matrix [[g_d,-g_d],[-g_d,g_d]]
 * For a MOSFET (4-terminal):   4×4 with gm, gds, gmb entries
 * For a BJT (3-terminal):      3×3 with Ebers-Moll conductances
 *
 * The MPO is the tensor-network analogue of a sparse stamp: instead of
 * scattering entries into a global CSR matrix, we store them as dense local
 * blocks. The contraction tree then specifies the order in which these blocks
 * are combined.
 */

struct DeviceMPO {
    FactorDeviceType deviceType;
    uint32_t deviceIndex;                // flat encoding from FactorGraph
    std::vector<uint32_t> terminalNodes; // circuit node IDs (sorted, ground excluded)
    std::vector<double> localMatrix;     // row-major k×k dense Jacobian contribution
    std::vector<double> localRHS;        // k-element Norton equivalent current vector
    uint32_t rank = 0;                   // k = terminalNodes.size()
};

class MPOBuilder {
public:
    /**
     * Build MPOs for all devices in a TensorizedBlock at a given voltage state.
     *
     * Each MPO captures the linearized device contribution (same math as the
     * stamp functions in stamps.h, but stored as dense blocks).
     *
     * @param tensors   SoA tensor block
     * @param voltages  Current node voltages (0-based, size = nodeCount)
     * @param temp_K    Temperature in Kelvin
     * @return Vector of DeviceMPOs, one per device
     */
    static std::vector<DeviceMPO> buildMPOs(
        const TensorizedBlock& tensors,
        const std::vector<double>& voltages,
        double temp_K);
};
