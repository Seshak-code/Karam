#pragma once
// Physics Compiler – Runtime Model Registry with Type-Erased Tensors.
//
// Provides dynamic model discovery, selection, and generic solver integration.
// Generated models register themselves at static-init time via REGISTER_PHYSICS_MODEL.
//
// The MNA solver iterates over registered models calling stored function
// pointers (batchPhysics, stamp) on void-casted tensors. This enables
// parallel Jacobian assembly without hardcoding device types.

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <cassert>
#include <cstddef>

// Abstract interface for matrix stamping
struct SparseStampTarget {
    virtual ~SparseStampTarget() = default;

    // Add value to Jacobian matrix at (row, col). 0-indexed.
    virtual void add(int row, int col, double val) = 0;

    // Add value to RHS vector at row. 0-indexed.
    virtual void addRhs(int row, double val) = 0;
};

// ============================================================================
// Model Info & Function Pointer Types
// ============================================================================

/// Function pointer: create a new tensor struct, returns void*.
using TensorCreateFn   = void* (*)();

/// Function pointer: destroy a tensor struct previously created.
using TensorDestroyFn  = void (*)(void* tensor);

/// Function pointer: resize tensor arrays to `count` devices.
using TensorResizeFn   = void (*)(void* tensor, size_t count);

/// Function pointer: evaluate device physics for all instances.
///   tensor:    the type-erased SoA tensor struct
///   voltages:  current global voltage vector
using BatchPhysicsFn   = void (*)(void* tensor, const std::vector<double>& voltages);

/// Function pointer: stamp Jacobian entries into sparse matrix.
/// Function pointer: stamp Jacobian entries into sparse matrix.
///   tensor:     the type-erased SoA tensor struct
///   count:      number of device instances
///   target:     abstract interface for matrix stamping
using StampFn          = void (*)(const void* tensor, size_t count,
                                  SparseStampTarget* target);

/// Function pointer: set instance data (nodes and parameters).
///   tensor:     the type-erased SoA tensor struct
///   index:      index of the instance to set (0 to count-1)
///   nodes:      array of global node indices (size = terminalCount)
///   params:     array of parameter values (size = parameterCount)
using SetInstanceFn    = void (*)(void* tensor, size_t index,
                                  const int* nodes, const double* params);

struct ModelInfo {
    std::string name;           // e.g. "Shockley"
    std::string deviceType;     // e.g. "Diode", "Mosfet"
    std::string description;
    std::string layoutHash;     // For binary compatibility checks
    int terminalCount;
    int parameterCount;

    // Optional: Terminal Names for GUI
    const char** terminalNames = nullptr;

    // --- Type-Erased Tensor Interface ---
    TensorCreateFn   createTensor   = nullptr;
    TensorDestroyFn  destroyTensor  = nullptr;
    TensorResizeFn   resizeTensor   = nullptr;
    SetInstanceFn    setInstance    = nullptr;
    BatchPhysicsFn   batchPhysics   = nullptr;
    StampFn          stampJacobian  = nullptr;
};


// ============================================================================
// Model Registry (Singleton)
// ============================================================================

class ModelRegistry {
public:
    static ModelRegistry& instance() {
        static ModelRegistry registry;
        return registry;
    }

    // Register a model (called from static initializers)
    void registerModel(const ModelInfo& info) {
        // Guard against duplicate registration (e.g. from hot-reload)
        auto& models = models_[info.deviceType];
        for (auto it = models.begin(); it != models.end(); ++it) {
            if (it->name == info.name) {
                // Replace existing registration (hot-reload case)
                *it = info;
                return;
            }
        }
        models.push_back(info);
    }

    // Unregister a model by name (for hot-reload cleanup)
    bool unregisterModel(const std::string& deviceType, const std::string& name) {
        auto it = models_.find(deviceType);
        if (it == models_.end()) return false;
        auto& models = it->second;
        for (auto mit = models.begin(); mit != models.end(); ++mit) {
            if (mit->name == name) {
                models.erase(mit);
                return true;
            }
        }
        return false;
    }

    // List all models for a device type
    std::vector<std::string> listModels(const std::string& deviceType) const {
        std::vector<std::string> names;
        auto it = models_.find(deviceType);
        if (it != models_.end()) {
            for (const auto& m : it->second) {
                names.push_back(m.name);
            }
        }
        return names;
    }

    // Get model info by name
    const ModelInfo* getModel(const std::string& deviceType, const std::string& name) const {
        auto it = models_.find(deviceType);
        if (it != models_.end()) {
            for (const auto& m : it->second) {
                if (m.name == name) return &m;
            }
        }
        return nullptr;
    }

    // List all device types
    std::vector<std::string> listDeviceTypes() const {
        std::vector<std::string> types;
        for (const auto& [type, _] : models_) {
            types.push_back(type);
        }
        return types;
    }

    // -----------------------------------------------------------------------
    // Runtime Hash Verification
    // -----------------------------------------------------------------------

    /// Verify layout hash at runtime. Returns true if hash matches.
    /// Call this before simulation to prevent silent corruption.
    bool verifyHash(const std::string& deviceType, const std::string& name,
                    const std::string& expectedHash) const {
        const ModelInfo* info = getModel(deviceType, name);
        if (!info) return false;
        return info->layoutHash == expectedHash;
    }

    /// Assert layout hash matches or abort simulation.
    /// Use this in the solver hot path to guarantee binary compatibility.
    void assertHash(const std::string& deviceType, const std::string& name,
                    const std::string& expectedHash) const {
        const ModelInfo* info = getModel(deviceType, name);
        if (!info) {
            std::cerr << "[ModelRegistry] FATAL: Model '" << deviceType
                      << "/" << name << "' not found.\n";
            assert(false && "Model not registered");
            return;
        }
        if (info->layoutHash != expectedHash) {
            std::cerr << "[ModelRegistry] FATAL: Layout hash mismatch for '"
                      << deviceType << "/" << name << "'.\n"
                      << "  Expected: " << expectedHash << "\n"
                      << "  Got:      " << info->layoutHash << "\n"
                      << "  The YAML/VA model has changed since last compile.\n"
                      << "  Re-run the Physics Compiler to regenerate.\n";
            assert(false && "Layout hash mismatch — refusing simulation");
        }
    }

    // Find model by name across all device types
    const ModelInfo* findModelByName(const std::string& name) const {
        for (const auto& [type, models] : models_) {
            for (const auto& m : models) {
                if (m.name == name) return &m;
            }
        }
        return nullptr;
    }

    // Print all registered models (for diagnostics)
    void dump() const {
        for (const auto& [type, models] : models_) {
            std::cout << "[" << type << "]\n";
            for (const auto& m : models) {
                std::cout << "  " << m.name
                          << " (hash=" << m.layoutHash
                          << ", terminals=" << m.terminalCount
                          << ", params=" << m.parameterCount
                          << ", physics=" << (m.batchPhysics ? "yes" : "no")
                          << ", stamp=" << (m.stampJacobian ? "yes" : "no")
                          << ")\n";
            }
        }
    }

private:
    ModelRegistry() = default;
    std::unordered_map<std::string, std::vector<ModelInfo>> models_;
};


// ============================================================================
// Registration Macros (placed at bottom of generated headers)
// ============================================================================

#define REGISTER_PHYSICS_MODEL(NAME, DEVICE_TYPE, DESC, HASH, TERMINALS, PARAMS) \
    namespace { \
        struct _Register_##NAME { \
            _Register_##NAME() { \
                ModelInfo info; \
                info.name = #NAME; \
                info.deviceType = #DEVICE_TYPE; \
                info.description = DESC; \
                info.layoutHash = HASH; \
                info.terminalCount = TERMINALS; \
                info.parameterCount = PARAMS; \
                ModelRegistry::instance().registerModel(info); \
            } \
        } _register_##NAME##_instance; \
    }

// Extended macro that registers tensor function pointers including setInstance and terminal names.
#define REGISTER_PHYSICS_MODEL_EX(NAME, DEVICE_TYPE, DESC, HASH, TERMINALS, PARAMS, \
                                   CREATE_FN, DESTROY_FN, RESIZE_FN, SET_FN, BATCH_FN, STAMP_FN, TERM_NAMES) \
    namespace { \
        struct _Register_##NAME { \
            _Register_##NAME() { \
                ModelInfo info; \
                info.name = #NAME; \
                info.deviceType = #DEVICE_TYPE; \
                info.description = DESC; \
                info.layoutHash = HASH; \
                info.terminalCount = TERMINALS; \
                info.parameterCount = PARAMS; \
                info.createTensor = CREATE_FN; \
                info.destroyTensor = DESTROY_FN; \
                info.resizeTensor = RESIZE_FN; \
                info.setInstance = SET_FN; \
                info.batchPhysics = BATCH_FN; \
                info.stampJacobian = STAMP_FN; \
                info.terminalNames = TERM_NAMES; \
                ModelRegistry::instance().registerModel(info); \
            } \
        } _register_##NAME##_instance; \
    }
