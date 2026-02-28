// ============================================================================
// gpu_context_manager.cpp — GPU Device Pool Implementation
// ============================================================================
#include "acutesim_engine/internal/engine_pch.h"
#include "acutesim_engine/gpu_context_manager.h"

#ifdef ACUTESIM_GPU_ENABLED
#ifndef __EMSCRIPTEN__
// Required to register the Dawn proc table before any wgpu:: calls
#include <dawn/native/DawnNative.h>
#include <dawn/dawn_proc.h>
#endif
#endif

namespace acutesim {

GPUContextManager& GPUContextManager::instance() {
    static GPUContextManager instance;
    return instance;
}

#ifdef ACUTESIM_GPU_ENABLED

GPUContextManager::GPUContextManager() {
    initialize();
}

GPUContextManager::~GPUContextManager() {
    shutdown();
}

void GPUContextManager::initialize() {
    if (initialized_) return;
    initialized_ = true;

#ifndef __EMSCRIPTEN__
    // Register the native Dawn proc table (required before any wgpu:: or wgpu C calls)
    dawnProcSetProcs(&dawn::native::GetProcs());

    // Enumerate adapters synchronously via the native API — avoids async callback issues.
    // nativeInstance_ is a member so it is destroyed by shutdown(), not by function-local
    // static lifetime (which would outlive any re-initialization attempt).
    nativeInstance_ = std::make_unique<dawn::native::Instance>();
    auto adapters = nativeInstance_->EnumerateAdapters();
    if (adapters.empty()) { available_ = false; return; }

    // Select the first discrete or integrated GPU adapter
    dawn::native::Adapter selected;
    for (const auto& a : adapters) {
        WGPUAdapterInfo info{};
        wgpuAdapterGetInfo(a.Get(), &info);
        const bool usable = (info.adapterType == WGPUAdapterType_DiscreteGPU ||
                             info.adapterType == WGPUAdapterType_IntegratedGPU);
        if (usable) {
            if (info.device.length > 0)
                adapterInfo_.name = std::string(info.device.data, info.device.length);
            else
                adapterInfo_.name = "Unknown GPU";
            if (info.description.length > 0)
                adapterInfo_.driverDescription = std::string(info.description.data,
                                                              info.description.length);
            adapterInfo_.isDiscrete = (info.adapterType == WGPUAdapterType_DiscreteGPU);
            wgpuAdapterInfoFreeMembers(info);
            selected = a;
            break;
        }
        wgpuAdapterInfoFreeMembers(info);
    }

    if (!selected) { available_ = false; return; }

    // Create device directly from the native adapter (synchronous, no callbacks)
    WGPUDevice rawDev = selected.CreateDevice(static_cast<const WGPUDeviceDescriptor*>(nullptr));
    if (!rawDev) { available_ = false; return; }

    primaryDevice_ = wgpu::Device::Acquire(rawDev);
    queue_ = primaryDevice_.GetQueue();
    pool_.push_back(primaryDevice_);
    available_ = true;
#else
    available_ = false; // WASM: GPU init is handled asynchronously elsewhere
#endif
}

GPUDevice GPUContextManager::acquire() {
    std::lock_guard<std::mutex> lk(poolMutex_);
    if (!available_) return GPUDevice{};
    // For now return shared primary device (command encoding is CPU-side thread-safe)
    return primaryDevice_;
}

void GPUContextManager::release(GPUDevice /*device*/) {
    // No-op while using shared primary device model
}

bool GPUContextManager::isAvailable() const { return available_; }

GPUAdapterInfo GPUContextManager::adapterInfo() const { return adapterInfo_; }

GPUQueue GPUContextManager::sharedQueue() const {
    std::lock_guard<std::mutex> lk(poolMutex_);
    return queue_;
}

WGPUDevice GPUContextManager::rawDevice() const {
    std::lock_guard<std::mutex> lk(poolMutex_);
    return primaryDevice_.Get();
}

WGPUQueue GPUContextManager::rawQueue() const {
    std::lock_guard<std::mutex> lk(poolMutex_);
    return queue_.Get();
}

void GPUContextManager::shutdown() {
    pool_.clear();
    primaryDevice_ = {};
    queue_         = {};
    adapter_       = {};
    instance_      = {};
#ifndef __EMSCRIPTEN__
    nativeInstance_.reset();
#endif
    available_     = false;
}

#else
// CPU-only (WASM or Dawn-disabled) stubs

GPUContextManager::GPUContextManager()  {}
GPUContextManager::~GPUContextManager() {}
void           GPUContextManager::initialize()     {}
GPUDevice      GPUContextManager::acquire()         { return {}; }
void           GPUContextManager::release(GPUDevice) {}
bool           GPUContextManager::isAvailable() const { return false; }
GPUAdapterInfo GPUContextManager::adapterInfo() const { return {}; }
GPUQueue       GPUContextManager::sharedQueue() const  { return {}; }
void           GPUContextManager::shutdown()       {}

#endif

} // namespace acutesim
