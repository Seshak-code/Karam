#pragma once
// ============================================================================
// gpu_context_manager.h — GPU Device Pool (Internal)
// ============================================================================
// Prevents Dawn initialization from dominating per-job runtime.
//
// RULES:
//   ✅ Engine owns GPU device lifetime exclusively
//   ❌ GUI never touches WGPUDevice or WGPUAdapter objects
//   ❌ Per-job device creation is FORBIDDEN
//
// Pattern: Acquire → Use → Release back to pool.
// Multiple concurrent sessions share the same GPU device safely
// because WebGPU command encoding is CPU-side and thread-safe.
// ============================================================================

#include "acutesim_engine/engine_export.h"
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <cstddef>

#ifdef ACUTESIM_GPU_ENABLED
#include <webgpu/webgpu_cpp.h>
using GPUDevice = wgpu::Device;
using GPUQueue  = wgpu::Queue;
#else
// Stub types for CPU-only builds
struct GPUDevice { bool valid() const { return false; } };
struct GPUQueue  { bool valid() const { return false; } };
#endif

namespace acutesim {

struct ENGINE_INTERNAL GPUAdapterInfo {
    std::string name;
    std::string driverDescription;
    bool        isDiscrete = false;
    size_t      dedicatedVRAMBytes = 0;
};

class ENGINE_INTERNAL GPUContextManager {
public:
    // Singleton — engine owns the GPU context lifetime
    static GPUContextManager& instance();

    // Pool operations
    // acquire() returns an existing device if available, initializes lazily
    GPUDevice acquire();
    void      release(GPUDevice device);

    // Capability queries (called once, cached)
    bool            isAvailable() const;
    GPUAdapterInfo  adapterInfo() const;
    GPUQueue        sharedQueue() const;

    // Called during engine shutdown — release all Dawn resources
    void shutdown();

    // Non-copyable singleton
    GPUContextManager(const GPUContextManager&) = delete;
    GPUContextManager& operator=(const GPUContextManager&) = delete;

private:
    GPUContextManager();
    ~GPUContextManager();

    void initialize();

    bool              initialized_  = false;
    bool              available_    = false;
    GPUAdapterInfo    adapterInfo_;

#ifdef ACUTESIM_GPU_ENABLED
    wgpu::Instance    instance_;
    wgpu::Adapter     adapter_;
    GPUDevice         primaryDevice_;
    GPUQueue          queue_;
    std::vector<GPUDevice> pool_;
    std::mutex        poolMutex_;
#endif
};

} // namespace acutesim
