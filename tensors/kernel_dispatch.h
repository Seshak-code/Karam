#pragma once

#include "../tensors/physics_tensors.h"
#include <functional>
#include <iostream>
#include <vector>

namespace acutesim {
namespace compute {

enum class SimdArch {
    Generic,         // Scalar C++ fallback
    Neon,            // ARM NEON
    AVX2,            // x86 AVX2 (4-wide double)
    AVX512,          // x86 AVX-512 (8-wide double)
    AVX10_256,       // AVX10 256-bit
    AVX10_512,       // AVX10 512-bit
    GPU_Integrated,  // iGPU (Metal/Vulkan, shared memory — no PCIe cost)
    GPU_Discrete     // dGPU (PCIe-attached — pipeline overlap needed)
};

class KernelDispatcher {
public:
    using DiodeKernel = void(*)(DiodeTensor&, const std::vector<double>&);
    using MosfetKernel = void(*)(MosfetTensor&, const std::vector<double>&);
    using BJTKernel = void(*)(BJTTensor&, const std::vector<double>&);

    SimdArch architecture;  // Best CPU SIMD tier
    DiodeKernel batch_diode_physics;
    MosfetKernel batch_mosfet_physics;
    BJTKernel batch_bjt_physics;

    // GPU capability (probed from Dawn adapters at startup)
    bool hasIntegratedGPU = false;
    bool hasDiscreteGPU   = false;
    SimdArch bestCpuArch  = SimdArch::Generic;
    SimdArch bestGpuArch  = SimdArch::Generic; // Generic = no GPU

    static KernelDispatcher& get();

private:
    KernelDispatcher(); // Private CTR (Singleton)
    void detectCpuAndBind();
    void probeGpuAdapters(); // Dawn adapter enumeration (non-EMSCRIPTEN only)
};

} // namespace compute
} // namespace acutesim
