// ============================================================================
// gpu_context_manager.cpp — GPU Device Pool Implementation
// ============================================================================
#include "acutesim_engine/internal/engine_pch.h"
#include "acutesim_engine/gpu_context_manager.h"

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

    // Create Dawn instance
    wgpu::InstanceDescriptor desc{};
    instance_ = wgpu::CreateInstance(&desc);
    if (!instance_) {
        available_ = false;
        return;
    }

    // Request adapter (blocking for initialization — called once at startup)
    wgpu::RequestAdapterOptions opts{};
    opts.powerPreference = wgpu::PowerPreference::HighPerformance;

    instance_.RequestAdapter(&opts, wgpu::CallbackMode::WaitAnyOnly,
        [this](wgpu::RequestAdapterStatus status,
               wgpu::Adapter adapter,
               const char* msg) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                available_ = false;
                return;
            }
            adapter_ = std::move(adapter);

            wgpu::AdapterInfo info{};
            adapter_.GetInfo(&info);
            adapterInfo_.name              = info.device ? info.device : "Unknown";
            adapterInfo_.driverDescription = info.description ? info.description : "";
            adapterInfo_.isDiscrete        = (info.adapterType == wgpu::AdapterType::DiscreteGPU);
            available_ = true;
        });

    if (!available_) return;

    // Create the primary device (shared across all sessions)
    wgpu::DeviceDescriptor devDesc{};
    devDesc.uncapturedErrorCallbackInfo = {
        nullptr, [](const wgpu::Device&, wgpu::ErrorType type, const char* msg) {
            // TODO: route through EngineCallbacks once session context available
        }
    };

    adapter_.RequestDevice(&devDesc, wgpu::CallbackMode::WaitAnyOnly,
        [this](wgpu::RequestDeviceStatus status,
               wgpu::Device device,
               const char* msg) {
            if (status == wgpu::RequestDeviceStatus::Success) {
                primaryDevice_ = std::move(device);
                queue_  = primaryDevice_.GetQueue();
                pool_.push_back(primaryDevice_);
            } else {
                available_ = false;
            }
        });
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

GPUQueue GPUContextManager::sharedQueue() const { return queue_; }

void GPUContextManager::shutdown() {
    pool_.clear();
    primaryDevice_ = {};
    queue_         = {};
    adapter_       = {};
    instance_      = {};
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
