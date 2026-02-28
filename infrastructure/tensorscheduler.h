#pragma once
#include "../netlist/circuit.h"
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <cassert>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

/*
 * tensorscheduler.h
 * Tensor Scheduler - Manages batching, structural reuse, and hardware selection.
 */


// Hardware Backend Types
enum class SolverBackend {
    CPU_SEQUENTIAL,  // Baseline (for debugging)
    CPU_OPENMP,      // Multi-threaded CPU
    WEBGPU_COMPUTE   // Future: GLSL/WGSL Compute Shaders
};

// --- SOLID: Dependency Inversion - Abstract Interface ---

// Interface for hardware detection (DIP - Dependency Inversion Principle).
// Allows mocking in unit tests and swapping detection strategies.
class IHardwareDetector {
public:
    virtual ~IHardwareDetector() = default;
    virtual SolverBackend detectBestBackend() const = 0;
};

// Default implementation of IHardwareDetector.
class HardwareManager : public IHardwareDetector {
public:
    SolverBackend detectBestBackend() const override {
        #ifdef __EMSCRIPTEN__
            // Check for WebGPU availability via JS interop
            // This requires -s USE_WEBGPU=1 during compilation
            // For now, we assume if we are in Wasm with the feature flag, we use it.
            return SolverBackend::WEBGPU_COMPUTE;
        #else
            return SolverBackend::CPU_OPENMP;
        #endif
    }
};

// SolverJob - Represents a batch of identical-topology block instances to solve together.
struct SolverJob {
    const TensorBlock* blockTmpl;
    std::vector<int> instanceIds;

    double calculateCost() const {
        assert(blockTmpl != nullptr && "SolverJob must have a valid block template");
        double componentWeight = blockTmpl->resistors.size() + 
                                blockTmpl->mosfets.size() * 5.0 + 
                                blockTmpl->capacitors.size() * 2.0;
        double nodeWeight = blockTmpl->numInternalNodes * 10.0;
        return (componentWeight + nodeWeight) * instanceIds.size();
    }
};

// TensorScheduler - Groups instances by topology for efficient batched solving.
// Uses IHardwareDetector for Dependency Inversion.
class TensorScheduler 
{

public:
    std::vector<SolverJob> jobs;
    SolverBackend activeBackend;
    
    TensorScheduler() {
        HardwareManager defaultDetector;
        activeBackend = defaultDetector.detectBestBackend();
    }
    
    // SOLID: Dependency Injection constructor
    explicit TensorScheduler(const IHardwareDetector& detector) {
        activeBackend = detector.detectBestBackend();
    }
    
    /*
     * Analyzes the netlist and groups identical blocks into jobs.
     *
     * Optimizations:
     * 1. Groups by Topology Hash (Research-Grade Reuse).
     * 2. Calculates expected Job Cost for backend load balancing.
     */

    void schedule(const TensorNetlist& netlist) {
        jobs.clear();
        
        // Phase 4 research: Group by structural topology, not just block names.
        // This allows instances of DIFFERENT blocks to be batched if their graphs match.
        std::map<size_t, std::vector<int>> groupedByHash;
        
        for (int i = 0; i < (int)netlist.instances.size(); ++i) {
            const auto& inst = netlist.instances[i];
            auto it = netlist.blockDefinitions.find(inst.blockName);
            if (it != netlist.blockDefinitions.end()) {
                groupedByHash[it->second.topologyHash].push_back(i);
            }
        }
        
        for (auto const& [hash, instances] : groupedByHash) {
            // Pick any instance from the group to get the template
            const auto& firstInst = netlist.instances[instances[0]];
            auto it = netlist.blockDefinitions.find(firstInst.blockName);
            if (it != netlist.blockDefinitions.end()) {
                jobs.push_back({ &(it->second), instances });
            }
        }
    }
};
