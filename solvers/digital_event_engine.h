#pragma once

#include <cstdint>
#include <vector>
#include <queue>
#include <string>
#include <cmath>
#include <algorithm>
#include <functional>
#include "../math/linalg.h"

/**
 * digital_event_engine.h — Mixed-Signal Event-Driven Digital Engine (Gap 2)
 *
 * Implements a zero-delay discrete event-driven digital solver for
 * mixed-signal (AMS) co-simulation. The engine decouples digital logic
 * propagation from the analog Newton-Raphson loop:
 *
 *   1. A2D: Analog node voltages feed threshold comparators to produce
 *           digital logic levels (0, 1, X, Z).
 *   2. Digital Propagation: Logic events propagate through a gate netlist
 *           using zero-delay event scheduling.
 *   3. D2A: Digital output states stamp dynamic voltage/resistance sources
 *           into the analog MNA matrix for the next NR iteration.
 *
 * Gate Types: INV, AND2, OR2, NAND2, NOR2, XOR2, BUF, TRISTATE
 *
 * Boundary Model (D2A):
 *   Logic 1 → Thévenin source: V = VDD, R = R_on (typ. 100Ω)
 *   Logic 0 → Thévenin source: V = 0V,  R = R_on
 *   Logic Z → Open circuit (no stamp)
 *   Logic X → Midpoint: V = VDD/2, R = R_on * 10 (high impedance)
 *
 * Boundary Model (A2D):
 *   V > V_th_high → Logic 1
 *   V < V_th_low  → Logic 0
 *   V_th_low ≤ V ≤ V_th_high → Logic X (undefined)
 */

enum class LogicLevel : uint8_t {
    L0 = 0,   // Logic low
    L1 = 1,   // Logic high
    LX = 2,   // Unknown / metastable
    LZ = 3    // High impedance (tri-state)
};

enum class GateType : uint8_t {
    BUF,       // Buffer
    INV,       // Inverter
    AND2,      // 2-input AND
    OR2,       // 2-input OR
    NAND2,     // 2-input NAND
    NOR2,      // 2-input NOR
    XOR2,      // 2-input XOR
    TRISTATE   // Tri-state buffer (input + enable)
};

struct DigitalPort {
    uint32_t portId;           // unique port identifier
    uint32_t analogNodeId;     // MNA node this port interfaces with (1-based)
    double vThresholdHigh;     // voltage above which → logic 1
    double vThresholdLow;      // voltage below which → logic 0
    double vdd;                // supply voltage for D2A stamp (default 1.8V)
    double rOn;                // Thévenin output resistance (default 100Ω)
    double riseTime;           // rise/fall time for slew-rate modeling [s]
    bool isInput;              // true = A2D port, false = D2A port
    LogicLevel currentLevel;   // current logic state
};

struct DigitalGate {
    uint32_t gateId;
    GateType type;
    std::vector<uint32_t> inputPortIds;  // indices into ports_ array
    uint32_t outputPortId;               // index into ports_ array
    std::string name;
};

struct DigitalEvent {
    uint32_t portId;           // port index in ports_ array
    LogicLevel newLevel;
    double scheduledTime;

    // Priority queue: earlier events first
    bool operator>(const DigitalEvent& other) const {
        return scheduledTime > other.scheduledTime;
    }
};

class DigitalEventEngine {
public:
    DigitalEventEngine() = default;

    // ── Port Management ─────────────────────────────────────────────────

    /**
     * Add an analog-digital boundary port.
     * @return Port index in internal array
     */
    uint32_t addPort(const DigitalPort& port) {
        uint32_t idx = static_cast<uint32_t>(ports_.size());
        DigitalPort p = port;
        p.portId = idx;
        p.currentLevel = LogicLevel::LX;
        ports_.push_back(p);
        return idx;
    }

    /**
     * Add a digital gate connecting input ports to an output port.
     */
    void addGate(GateType type, const std::vector<uint32_t>& inputs,
                 uint32_t output, const std::string& name = "") {
        DigitalGate gate;
        gate.gateId = static_cast<uint32_t>(gates_.size());
        gate.type = type;
        gate.inputPortIds = inputs;
        gate.outputPortId = output;
        gate.name = name;
        gates_.push_back(gate);
    }

    // ── A2D: Analog → Digital ───────────────────────────────────────────

    /**
     * Sample analog node voltages and convert to digital levels at input ports.
     * Schedules events for any ports whose level changed.
     *
     * @param voltages   Current analog node voltages (0-indexed, MNA-sized)
     * @param time       Current simulation time
     */
    void sampleAnalog(const std::vector<double>& voltages, double time) {
        for (auto& port : ports_) {
            if (!port.isInput) continue;

            // Get analog voltage at this port's node
            double v = 0.0;
            if (port.analogNodeId > 0 &&
                (port.analogNodeId - 1) < voltages.size()) {
                v = voltages[port.analogNodeId - 1];
            }

            // Threshold comparison with hysteresis
            LogicLevel newLevel = port.currentLevel;
            if (v > port.vThresholdHigh) {
                newLevel = LogicLevel::L1;
            } else if (v < port.vThresholdLow) {
                newLevel = LogicLevel::L0;
            } else {
                // In the undefined band — keep current level (hysteresis)
                // Unless it was X (first sample), resolve to nearest
                if (port.currentLevel == LogicLevel::LX) {
                    double mid = (port.vThresholdHigh + port.vThresholdLow) / 2.0;
                    newLevel = (v >= mid) ? LogicLevel::L1 : LogicLevel::L0;
                }
            }

            if (newLevel != port.currentLevel) {
                port.currentLevel = newLevel;
                eventQueue_.push({port.portId, newLevel, time});
            }
        }
    }

    // ── Digital Event Propagation ────────────────────────────────────────

    /**
     * Propagate all pending digital events through the gate network.
     * Uses zero-delay semantics: all combinational paths settle instantly.
     * Returns the number of events propagated.
     */
    int propagateEvents() {
        int totalEvents = 0;
        int maxIterations = 1000;  // prevent infinite loops in feedback

        while (!eventQueue_.empty() && totalEvents < maxIterations) {
            DigitalEvent ev = eventQueue_.top();
            eventQueue_.pop();

            if (ev.portId >= ports_.size()) continue;
            ports_[ev.portId].currentLevel = ev.newLevel;
            totalEvents++;

            // Find all gates driven by this port
            for (const auto& gate : gates_) {
                bool isInput = false;
                for (uint32_t inp : gate.inputPortIds) {
                    if (inp == ev.portId) { isInput = true; break; }
                }
                if (!isInput) continue;

                // Evaluate gate
                LogicLevel output = evaluateGate(gate);

                // Schedule output event if level changed
                if (gate.outputPortId < ports_.size() &&
                    output != ports_[gate.outputPortId].currentLevel) {
                    eventQueue_.push({gate.outputPortId, output, ev.scheduledTime});
                }
            }
        }

        return totalEvents;
    }

    // ── D2A: Digital → Analog MNA Stamps ─────────────────────────────────

    /**
     * Stamp D2A boundary conditions into the analog MNA matrix.
     * For each output port, stamps a Thévenin equivalent source.
     *
     * @param builder   MNA matrix constructor
     * @param rhs       RHS current vector
     * @param numNodes  Total MNA node count
     * @param dt        Current timestep (for slew-rate modeling)
     */
    void stampD2A(MatrixConstructor& builder, std::vector<double>& rhs,
                  int numNodes, double dt) const {
        for (const auto& port : ports_) {
            if (port.isInput) continue;  // only stamp D2A (output) ports

            int nodeIdx = static_cast<int>(port.analogNodeId);
            if (nodeIdx <= 0 || nodeIdx > numNodes) continue;
            int row = nodeIdx - 1;

            double G = 0.0;   // Thévenin conductance
            double Veq = 0.0; // Thévenin voltage

            switch (port.currentLevel) {
                case LogicLevel::L1:
                    G = 1.0 / port.rOn;
                    Veq = port.vdd;
                    break;
                case LogicLevel::L0:
                    G = 1.0 / port.rOn;
                    Veq = 0.0;
                    break;
                case LogicLevel::LX:
                    // Unknown: weak pull to midpoint
                    G = 1.0 / (port.rOn * 10.0);
                    Veq = port.vdd / 2.0;
                    break;
                case LogicLevel::LZ:
                    // High impedance: no stamp (open circuit)
                    continue;
            }

            // Stamp Norton equivalent: I = G·Veq at node, G to diagonal
            // MNA stamp: G·(V_node - Veq) = 0
            //   → G on diagonal, -G·Veq in RHS
            builder.add(row, row, G);
            if (row < static_cast<int>(rhs.size())) {
                rhs[row] += G * Veq;
            }
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────

    size_t portCount() const { return ports_.size(); }
    size_t gateCount() const { return gates_.size(); }

    LogicLevel getPortLevel(uint32_t portIdx) const {
        if (portIdx < ports_.size()) return ports_[portIdx].currentLevel;
        return LogicLevel::LX;
    }

    void setPortLevel(uint32_t portIdx, LogicLevel level) {
        if (portIdx < ports_.size()) ports_[portIdx].currentLevel = level;
    }

    bool hasOutputPorts() const {
        for (const auto& p : ports_) {
            if (!p.isInput) return true;
        }
        return false;
    }

    /**
     * Reset all port states to X and clear the event queue.
     */
    void reset() {
        for (auto& port : ports_) {
            port.currentLevel = LogicLevel::LX;
        }
        while (!eventQueue_.empty()) eventQueue_.pop();
    }

    /**
     * Clear all ports and gates (topology change).
     */
    void clear() {
        ports_.clear();
        gates_.clear();
        reset();
    }

private:
    std::vector<DigitalPort> ports_;
    std::vector<DigitalGate> gates_;
    std::priority_queue<DigitalEvent, std::vector<DigitalEvent>,
                        std::greater<DigitalEvent>> eventQueue_;

    /**
     * Evaluate a digital gate's output from its current input levels.
     */
    LogicLevel evaluateGate(const DigitalGate& gate) const {
        // Collect input levels
        std::vector<LogicLevel> inputs;
        inputs.reserve(gate.inputPortIds.size());
        for (uint32_t idx : gate.inputPortIds) {
            if (idx < ports_.size()) {
                inputs.push_back(ports_[idx].currentLevel);
            } else {
                inputs.push_back(LogicLevel::LX);
            }
        }

        if (inputs.empty()) return LogicLevel::LX;

        // Check if any input is X → output is X (unless gate can resolve)
        bool hasX = false;
        for (auto l : inputs) {
            if (l == LogicLevel::LX || l == LogicLevel::LZ) hasX = true;
        }

        switch (gate.type) {
            case GateType::BUF:
                return inputs[0];

            case GateType::INV:
                if (inputs[0] == LogicLevel::L0) return LogicLevel::L1;
                if (inputs[0] == LogicLevel::L1) return LogicLevel::L0;
                return LogicLevel::LX;

            case GateType::AND2: {
                if (inputs.size() < 2) return LogicLevel::LX;
                // AND: any 0 → output 0 (dominant)
                for (auto l : inputs) {
                    if (l == LogicLevel::L0) return LogicLevel::L0;
                }
                if (hasX) return LogicLevel::LX;
                return LogicLevel::L1;
            }

            case GateType::OR2: {
                if (inputs.size() < 2) return LogicLevel::LX;
                // OR: any 1 → output 1 (dominant)
                for (auto l : inputs) {
                    if (l == LogicLevel::L1) return LogicLevel::L1;
                }
                if (hasX) return LogicLevel::LX;
                return LogicLevel::L0;
            }

            case GateType::NAND2: {
                if (inputs.size() < 2) return LogicLevel::LX;
                for (auto l : inputs) {
                    if (l == LogicLevel::L0) return LogicLevel::L1;
                }
                if (hasX) return LogicLevel::LX;
                return LogicLevel::L0;
            }

            case GateType::NOR2: {
                if (inputs.size() < 2) return LogicLevel::LX;
                for (auto l : inputs) {
                    if (l == LogicLevel::L1) return LogicLevel::L0;
                }
                if (hasX) return LogicLevel::LX;
                return LogicLevel::L1;
            }

            case GateType::XOR2: {
                if (inputs.size() < 2 || hasX) return LogicLevel::LX;
                bool a = (inputs[0] == LogicLevel::L1);
                bool b = (inputs[1] == LogicLevel::L1);
                return (a != b) ? LogicLevel::L1 : LogicLevel::L0;
            }

            case GateType::TRISTATE: {
                // inputs[0] = data, inputs[1] = enable
                if (inputs.size() < 2) return LogicLevel::LZ;
                if (inputs[1] == LogicLevel::L1) return inputs[0];  // enabled
                if (inputs[1] == LogicLevel::L0) return LogicLevel::LZ;  // disabled
                return LogicLevel::LX;  // enable unknown
            }

            default:
                return LogicLevel::LX;
        }
    }
};
