#include "netlist_export.h"
#include <sstream>
#include <set>
#include <vector>
#include <algorithm>
#include <map>

namespace NetlistExporter {

    // Helper: Canonical Net Name
    static std::string getNetName(int nodeID, const NetlistExporter::ExportOptions& options) {
        if (nodeID == 0) return "VSS"; // Map Ground to physical VSS
        return "n" + std::to_string(nodeID);
    }

    static void exportBlock(std::stringstream& ss, const TensorBlock& block, const std::string& moduleName, const ExportOptions& options) {
        // 1. Module Header
        ss << "module " << moduleName << " (";
        for (size_t i = 0; i < block.pinNames.size(); ++i) {
            ss << block.pinNames[i];
            if (i < block.pinNames.size() - 1) ss << ", ";
        }
        ss << ");\n\n";

        // 2. Wire Declarations (exclude pins which are ports)
        std::set<int> allNodes;
        auto collect = [&](int n) { if (n >= 0) allNodes.insert(n); };

        for (const auto& r : block.resistors) { collect(r.nodeTerminal1); collect(r.nodeTerminal2); }
        for (const auto& c : block.capacitors) { collect(c.nodePlate1); collect(c.nodePlate2); }
        for (const auto& l : block.inductors) { collect(l.nodeCoil1); collect(l.nodeCoil2); }
        for (const auto& d : block.diodes) { collect(d.anode); collect(d.cathode); }
        for (const auto& m : block.mosfets) { collect(m.drain); collect(m.gate); collect(m.source); collect(m.body); }
        for (const auto& b : block.bjts) { collect(b.nodeCollector); collect(b.base); collect(b.emitter); }

        std::set<int> pinNodes;
        for (size_t i = 0; i < block.pinNames.size(); ++i) pinNodes.insert((int)i + 1);

        ss << "  // Internal Nets\n";
        for (int n : allNodes) {
             if (n == 0 || pinNodes.count(n) > 0) continue; // Skip VSS and Pins
             ss << "  wire " << getNetName(n, options) << ";\n";
        }
        ss << "\n";

        // 3. Component Instantiations
        // Resistors
        for (size_t i = 0; i < block.resistors.size(); ++i) {
            const auto& comp = block.resistors[i];
            ss << "  resistor #(.r(" << comp.resistance_ohms << ")) " << (comp.name.empty() ? ("R" + std::to_string(i)) : comp.name) 
               << " (.p(" << getNetName(comp.nodeTerminal1, options) << "), .n(" << getNetName(comp.nodeTerminal2, options) << "));\n";
        }

        // MOSFETs
        for (size_t i = 0; i < block.mosfets.size(); ++i) {
            const auto& comp = block.mosfets[i];
            std::string name = comp.instanceName.empty() ? ("M" + std::to_string(i)) : comp.instanceName;
            std::string type = comp.modelName.find("PMOS") != std::string::npos || comp.modelName.find("pfet") != std::string::npos ? "pfet_01v8" : "nfet_01v8";
            
            ss << "  " << type << " " << name 
               << " (.D(" << getNetName(comp.drain, options) << "), .G(" << getNetName(comp.gate, options) << "), "
               << ".S(" << getNetName(comp.source, options) << "), .B(" << getNetName(comp.body, options) << "));\n";
        }

        // BJTs
        for (size_t i = 0; i < block.bjts.size(); ++i) {
            const auto& comp = block.bjts[i];
            std::string name = comp.instanceName.empty() ? ("Q" + std::to_string(i)) : comp.instanceName;
            std::string type = comp.isNPN ? "npn" : "pnp";
            ss << "  " << type << " #(.bf(" << comp.betaF << ")) " << name 
               << " (.C(" << getNetName(comp.nodeCollector, options) << "), .B(" << getNetName(comp.base, options) << "), "
               << ".E(" << getNetName(comp.emitter, options) << "));\n";
        }

        ss << "\nendmodule\n\n";
    }

    std::string toVerilog(const acutesim::compute::orchestration::HierarchicalSchematicDTO& hierDto, const ExportOptions& options) {
        // Simple conversion from DTO to TensorNetlist for reuse of existing export logic
        TensorNetlist tempNl;
        auto& top = hierDto.topGraph;
        
        auto getNetIdx = [&](const std::string& name) -> int {
            if (name == "0" || name == "GND" || name == "gnd" || name == "VSS") return 0;
            for (size_t i = 0; i < top.nets.size(); ++i) {
                if (top.nets[i].name == name) return (int)i + 1;
            }
            return (int)top.nets.size() + 1; // fallback
        };

        auto getVal = [](const std::map<std::string, std::string>& params, const std::string& key, double def = 0.0) {
            auto it = params.find(key);
            if (it == params.end()) return def;
            try { return std::stod(it->second); } catch(...) { return def; }
        };

        for (const auto& inst : top.instances) {
            if (inst.type == "Resistor") {
                Resistor r;
                r.name = inst.instanceName;
                r.nodeTerminal1 = inst.connectedNets.size() > 0 ? getNetIdx(inst.connectedNets[0]) : 0;
                r.nodeTerminal2 = inst.connectedNets.size() > 1 ? getNetIdx(inst.connectedNets[1]) : 0;
                r.resistance_ohms = getVal(inst.parameters, "R", getVal(inst.parameters, "val", 1000.0));
                tempNl.globalBlock.resistors.push_back(r);
            } else if (inst.type == "Mosfet") {
                Mosfet m;
                m.instanceName = inst.instanceName;
                m.drain = inst.connectedNets.size() > 0 ? getNetIdx(inst.connectedNets[0]) : 0;
                m.gate  = inst.connectedNets.size() > 1 ? getNetIdx(inst.connectedNets[1]) : 0;
                m.source= inst.connectedNets.size() > 2 ? getNetIdx(inst.connectedNets[2]) : 0;
                m.body  = inst.connectedNets.size() > 3 ? getNetIdx(inst.connectedNets[3]) : 0;
                auto itModel = inst.parameters.find("model");
                if (itModel != inst.parameters.end()) m.modelName = itModel->second;
                tempNl.globalBlock.mosfets.push_back(m);
            }
            // ... more types can be added as needed ...
        }

        return toVerilog(tempNl, options);
    }

    std::string toVerilog(const TensorNetlist& netlist, const ExportOptions& options) {
        std::stringstream ss;
        
        ss << "// Auto-Generated from Schematic via NetlistExporter\n";
        ss << "// Target Flavor: " << (options.flavor == ExportFlavor::AnalogMixedSignal ? "Verilog-AMS" : "Structural Verilog") << "\n\n";

        // 1. Export Sub-Module Definitions
        std::set<std::string> exportedBlocks;
        for (const auto& inst : netlist.instances) {
            if (exportedBlocks.count(inst.blockName) == 0) {
                auto it = netlist.blockDefinitions.find(inst.blockName);
                if (it != netlist.blockDefinitions.end()) {
                    exportBlock(ss, it->second, inst.blockName, options);
                    exportedBlocks.insert(inst.blockName);
                }
            }
        }

        // 2. Export Top Module
        // Include power ports if explicit power is enabled
        if (options.useExplicitPower) {
            ss << "module " << options.topModuleName << " (";
            bool first = true;
            for (const auto& [key, name] : options.powerNetNames) {
                if (!first) ss << ", ";
                ss << "inout " << name;
                first = false;
            }
            ss << ");\n";
        } else {
            ss << "module " << options.topModuleName << " ();\n";
        }

        // Collect all global nets
        std::set<int> allNodes;
        auto collect = [&](int n) { if (n >= 0) allNodes.insert(n); };

        const auto& block = netlist.globalBlock;
        for (const auto& r : block.resistors) { collect(r.nodeTerminal1); collect(r.nodeTerminal2); }
        for (const auto& m : block.mosfets) { collect(m.drain); collect(m.gate); collect(m.source); collect(m.body); }
        for (const auto& b : block.bjts) { collect(b.nodeCollector); collect(b.base); collect(b.emitter); }
        for (const auto& inst : netlist.instances) {
            for (int n : inst.nodeMapping) collect(n);
        }

        ss << "  // Global Nets\n";
        for (int n : allNodes) {
             if (n == 0) continue; // Skip physical ground
             ss << "  wire " << getNetName(n, options) << ";\n";
        }
        ss << "\n";

        // 3. Top-Level Components
        // Resistors
        for (const auto& r : block.resistors) {
            ss << "  resistor #(.r(" << r.resistance_ohms << ")) " << (r.name.empty() ? "R_top" : r.name) 
               << " (.p(" << getNetName(r.nodeTerminal1, options) << "), .n(" << getNetName(r.nodeTerminal2, options) << "));\n";
        }
        
        // Voltage Sources
        for (size_t i = 0; i < block.voltageSources.size(); ++i) {
            const auto& v = block.voltageSources[i];
            ss << "  vsource #(.v(" << v.voltage_V << ")) V" << i 
               << " (.p(" << getNetName(v.nodePositive, options) << "), .n(" << getNetName(v.nodeNegative, options) << "));\n";
        }

        for (const auto& m : block.mosfets) {
            std::string type = m.modelName.find("pfet") != std::string::npos ? "pfet_01v8" : "nfet_01v8";
            ss << "  " << type << " " << m.instanceName 
               << " (.D(" << getNetName(m.drain, options) << "), .G(" << getNetName(m.gate, options) << "), "
               << ".S(" << getNetName(m.source, options) << "), .B(" << getNetName(m.body, options) << "));\n";
        }

        // BJTs
        for (const auto& b : block.bjts) {
            std::string type = b.isNPN ? "npn" : "pnp";
            ss << "  " << type << " " << b.instanceName 
               << " (.C(" << getNetName(b.nodeCollector, options) << "), .B(" << getNetName(b.base, options) << "), "
               << ".E(" << getNetName(b.emitter, options) << "));\n";
        }

        // Hierarchical Instances
        for (size_t i = 0; i < netlist.instances.size(); ++i) {
            const auto& inst = netlist.instances[i];
            auto it = netlist.blockDefinitions.find(inst.blockName);
            ss << "  " << inst.blockName << " U" << i << " (";
            if (it != netlist.blockDefinitions.end()) {
                for (size_t j = 0; j < it->second.pinNames.size(); ++j) {
                    int globalNode = (j < inst.nodeMapping.size()) ? inst.nodeMapping[j] : 0;
                    ss << "." << it->second.pinNames[j] << "(" << getNetName(globalNode, options) << ")";
                    if (j < it->second.pinNames.size() - 1) ss << ", ";
                }
            }
            ss << ");\n";
        }

        ss << "\nendmodule\n";
        return ss.str();
    }

    std::string toSPEF(const TensorNetlist& netlist, const ExportOptions& options) {
        std::stringstream ss;
        const auto& block = netlist.globalBlock;

        // Header
        ss << "*SPEF \"IEEE 1481-1998\"\n";
        ss << "*DESIGN \"" << options.topModuleName << "\"\n";
        ss << "*DATE \"Tue Jan 28 14:00:00 2026\"\n";
        ss << "*VENDOR \"AcuteSim\"\n";
        ss << "*PROGRAM \"NetlistExporter\"\n";
        ss << "*VERSION \"0.1\"\n";
        ss << "*DESIGN_FLOW \"ANALOG_MIXED_SIGNAL\"\n";
        ss << "*DIVIDER /\n";
        ss << "*DELIMITER :\n";
        ss << "*BUS_BIT []\n";
        ss << "*T_UNIT 1.00000 NS\n";
        ss << "*C_UNIT 1.00000 FF\n";
        ss << "*R_UNIT 1.00000 OHM\n";
        ss << "*L_UNIT 1.00000 HENRY\n\n";

        // Ports Section (Match Verilog power ports)
        ss << "*PORTS\n";
        for (const auto& [key, name] : options.powerNetNames) {
            ss << name << " B\n";
        }
        ss << "*END\n\n";

        // Parasitics (D_NET)
        for(const auto& seg : block.wireSegments) {
            int netID = seg.node1; 
             ss << "*D_NET *" << getNetName(netID, options) << " 1.5\n"; 
             ss << "*CONN\n";
             ss << "*CAP\n";
             ss << "1 *" << getNetName(netID, options) << ":1 " << (seg.c_per_um * seg.length_um) << "\n";
             ss << "*RES\n";
             ss << "1 *" << getNetName(seg.node1, options) << ":1 *" << getNetName(seg.node2, options) << ":1 " 
                << (seg.r_per_um * seg.length_um) << "\n";
             ss << "*END\n\n";
        }
        
        return ss.str();
    }

} // namespace NetlistExporter
