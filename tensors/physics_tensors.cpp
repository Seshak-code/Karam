#include "../tensors/physics_tensors.h"
#include "../physics/physics_constants.h"
#include "../infrastructure/model_registry.h"
#include "../netlist/circuit.h"
#include <iostream>
#include <cmath>
#include <algorithm>

/*
 * physics_tensors.cpp - AoS -> SoA conversion of the linear elements and generic models.
 */


// ============================================================================
// AOS -> SOA CONVERSION
// ============================================================================

TensorizedBlock tensorizeBlock(const TensorBlock& block) {
    TensorizedBlock result;
    
    // Convert Resistors
    result.resistors.nodes1.reserve(block.resistors.size());
    result.resistors.nodes2.reserve(block.resistors.size());
    result.resistors.R.reserve(block.resistors.size());
    
    // Exact values: the 0.5 ohm .. 1 Gohm clamps (for a fixed-point solver that no
    // longer exists) doubled every sub-ohm power-grid resistor and disagreed with
    // the KCL audit. R <= 0 is rejected by the parser.
    for (const auto& r : block.resistors)
        result.resistors.push(r.nodeTerminal1, r.nodeTerminal2, r.resistance_ohms);
    
    // Convert Capacitors
    for (const auto& c : block.capacitors) {
        result.capacitors.push(c.nodePlate1, c.nodePlate2, c.capacitance_farads);
    }
    
    // Convert Sources
    for (const auto& v : block.voltageSources) {
        result.voltageSources.push(v.nodePositive, v.nodeNegative, v.voltage_V);
    }
    for (const auto& i : block.currentSources) {
        result.currentSources.push(i.nodePositive, i.nodeNegative, i.current_A);
    }
    
    return result;
}

TensorizedBlock tensorizeNetlist(const TensorNetlist& netlist) {
    // 1. Start with the flattened global block (standard components)
    TensorizedBlock result = tensorizeBlock(netlist.globalBlock);
    
    // -------------------------------------------------------------------------
    // BUCKETING STAGING AREA: Group devices by model to maximize SIMD efficiency
    // -------------------------------------------------------------------------
    std::map<std::string, size_t> typeCounts;
    auto& registry = ModelRegistry::instance();

    auto countBlock = [&](const TensorBlock& block) {
        for (const auto& gc : block.genericComponents) {
            const ModelInfo* info = registry.findModelByName(gc.modelName); // Helper needed or use getModel with known type?
            // Problem: We don't know the type from just modelName efficiently without scanning all types?
            // Optimization: ModelRegistry should map modelName -> DeviceType or ModelInfo directly.
            // For now, we iterate types to find it (slow but safe).
            // Better: Add lookup to registry.
        }
    };
    
    // We need a fast lookup for model name to info. 
    // Let's assume we can build a cache or modify registry. 
    // See below for Registry helper assumption.
    
    // Temporary lambda to find model info
    auto findInfo = [&](const std::string& name) -> const ModelInfo* {
        // Try to guess type or search all. 
        // This is inefficient. We should have ModelRegistry::findModel(name).
        // Let's implement a naive search here or rely on specific naming convention.
        // Actually, let's use the listDeviceTypes loop.
        auto types = registry.listDeviceTypes();
        for(const auto& t : types) {
             if(const ModelInfo* info = registry.getModel(t, name)) return info;
        }
        return nullptr;
    };

    // Count Global
    for (const auto& gc : netlist.globalBlock.genericComponents) {
        if (const ModelInfo* info = findInfo(gc.modelName)) {
            typeCounts[info->deviceType]++;
        }
    }
    
    // Count Instances
    for (const auto& instance : netlist.instances) {
        auto it = netlist.blockDefinitions.find(instance.blockName);
        if (it != netlist.blockDefinitions.end()) {
            for (const auto& gc : it->second.genericComponents) {
                 if (const ModelInfo* info = findInfo(gc.modelName)) {
                    typeCounts[info->deviceType]++;
                 }
            }
        }
    }

    // Allocate Tensors
    for (const auto& [type, count] : typeCounts) {
        if (count == 0) continue;
        // Get any model of this type to access create/resize functions
        auto models = registry.listModels(type);
        if (models.empty()) continue;
        const ModelInfo* factory = registry.getModel(type, models[0]);
        
        if (factory && factory->createTensor && factory->resizeTensor) {
            void* tensor = factory->createTensor();
            factory->resizeTensor(tensor, count);
            result.dynamicTensors[type] = {tensor, count};
        }
    }

    // -------------------------------------------------------------------------
    // DYNAMIC MODEL PASS 2: POPULATION
    // -------------------------------------------------------------------------
    std::map<std::string, size_t> typeIndices; // Current write index

    auto populateGeneric = [&](const TensorBlock& block, auto remap) {
        for (const auto& gc : block.genericComponents) {
            const ModelInfo* info = findInfo(gc.modelName);
            if (!info) {
                 std::cerr << "[Warning] Generic component '" << gc.instanceName 
                           << "' uses unknown model '" << gc.modelName << "'\n";
                 continue;
            }
            
            void* tensor = result.dynamicTensors[info->deviceType].tensor;
            if (!tensor) continue;
            
            size_t idx = typeIndices[info->deviceType]++;
            
            // Remap nodes
            std::vector<int> mappedNodes;
            mappedNodes.reserve(gc.nodes.size());
            for(int n : gc.nodes) mappedNodes.push_back(remap(n));
            
            // Call setInstance
            if (info->setInstance) {
                info->setInstance(tensor, idx, mappedNodes.data(), gc.params.data());
            }
        }
    };

    auto collectFromBlock = [&](const TensorBlock& block, const std::function<int(int)>& remap) {
        // Linear components (Resistors, Capacitors, Inductors)
        for (const auto& r : block.resistors) 
            result.resistors.push(remap(r.nodeTerminal1), remap(r.nodeTerminal2), r.resistance_ohms);
        for (const auto& c : block.capacitors)
            result.capacitors.push(remap(c.nodePlate1), remap(c.nodePlate2), c.capacitance_farads);
        for (const auto& ind : block.inductors)
            result.inductors.push(remap(ind.nodeCoil1), remap(ind.nodeCoil2), ind.inductance_henries);
        for (const auto& v : block.voltageSources)
            result.voltageSources.push(remap(v.nodePositive), remap(v.nodeNegative), v.voltage_V);
        for (const auto& i : block.currentSources)
            result.currentSources.push(remap(i.nodePositive), remap(i.nodeNegative), i.current_A);
    };

    // 2. Gather & Populate
    // Reset result to start clean (tensorizeBlock at 307 already pushed some things)
    result.clear(); 
    
    // Global
    auto identity = [](int n){ return n; };
    collectFromBlock(netlist.globalBlock, identity);
    populateGeneric(netlist.globalBlock, identity);

    // Instances
    for (const auto& inst : netlist.instances) {
        auto it = netlist.blockDefinitions.find(inst.blockName);
        if (it == netlist.blockDefinitions.end()) continue;
        auto remap = [&](int n) { return (n <= 0 || n > (int)inst.nodeMapping.size()) ? 0 : inst.nodeMapping[n-1]; };
        collectFromBlock(it->second, remap);
        populateGeneric(it->second, remap);
    }

    return result;
}
