#pragma once

#include "../infrastructure/tn_compiler.h"
#include "../math/linalg.h"
#include <vector>
#include <string>
#include <cstdint>

/**
 * metal_contraction_backend.h — Apple Metal Tensor Contraction Backend (Gap 3)
 *
 * Provides a native Metal Performance Shaders (MPS) backend for tensor
 * contraction on Apple Silicon (M1/M2/M3/M4). Leverages:
 *
 *   1. Unified Memory Architecture (UMA): Zero-copy buffer sharing between
 *      CPU and GPU — eliminates all staging buffer overhead.
 *   2. Metal 3 Matrix Multiply: Hardware-accelerated small matrix operations
 *      via MTLMatrixMultiplication for contraction tile operations.
 *   3. Shared memory atomics: MTLComputePipelineState with threadgroup
 *      memory for tile-local Schur elimination (same algorithm as WGSL).
 *
 * Architecture:
 *   The Metal backend mirrors the WebGPU shader structure but uses MSL
 *   (Metal Shading Language) compute kernels compiled at runtime. On
 *   Apple Silicon, the UMA eliminates the CPU-GPU transfer bottleneck
 *   entirely — the contraction tree walk becomes a sequence of GPU
 *   dispatches with no buffer copies.
 *
 * Compile guard: Only available on Apple Silicon (aarch64 macOS).
 * Falls back to WebGPU path on Intel Macs and other platforms.
 */

#if defined(__APPLE__) && defined(__aarch64__) && !defined(__EMSCRIPTEN__)
#define ACUTESIM_METAL_BACKEND 1
#else
#define ACUTESIM_METAL_BACKEND 0
#endif

class MetalContractionBackend {
public:
    MetalContractionBackend() = default;
    ~MetalContractionBackend() = default;

    /**
     * Check if the Metal backend is available at runtime.
     * Returns true only on Apple Silicon with Metal 3+ support.
     */
    static bool isAvailable() {
#if ACUTESIM_METAL_BACKEND
        return checkMetalSupport();
#else
        return false;
#endif
    }

    /**
     * Initialize the Metal compute pipeline.
     * Creates MTLDevice, compiles MSL shaders, allocates shared buffers.
     *
     * @return true if initialization succeeded
     */
    bool initialize() {
#if ACUTESIM_METAL_BACKEND
        return initializeImpl();
#else
        return false;
#endif
    }

    /**
     * Execute the full contraction tree on Metal, returning the solution vector.
     *
     * Algorithm:
     *   1. Upload MPO data to shared (UMA) buffers
     *   2. Bottom-up tree walk: dispatch seed → merge → schur_elim_record
     *   3. Root solve (CPU dense LU on the small root tile)
     *   4. Top-down: dispatch back_substitute for all records
     *   5. Read solution directly from shared buffer (zero-copy on UMA)
     *
     * @param prog       Compiled TN program with contraction tree + MPOs
     * @param nodeCount  Total MNA node count
     * @param gmin       Minimum conductance for pivot conditioning
     * @return Solution vector (node voltages), empty on failure
     */
    std::vector<double> executeContraction(
        const TNCompiledProgram& prog,
        uint32_t nodeCount,
        double gmin)
    {
#if ACUTESIM_METAL_BACKEND
        return executeContractionImpl(prog, nodeCount, gmin);
#else
        (void)prog; (void)nodeCount; (void)gmin;
        return {};
#endif
    }

    /**
     * Release all Metal resources.
     */
    void shutdown() {
#if ACUTESIM_METAL_BACKEND
        shutdownImpl();
#endif
    }

private:
#if ACUTESIM_METAL_BACKEND
    // Opaque Metal handles (avoid #import <Metal/Metal.h> in header)
    void* mtlDevice_   = nullptr;  // id<MTLDevice>
    void* mtlQueue_    = nullptr;  // id<MTLCommandQueue>
    void* mtlLibrary_  = nullptr;  // id<MTLLibrary>

    // Compute pipelines (id<MTLComputePipelineState>)
    void* pipelineSeedLeaf_      = nullptr;
    void* pipelineMergeAccum_    = nullptr;
    void* pipelineSchurRecord_   = nullptr;
    void* pipelineBackSub_       = nullptr;

    // Shared (UMA) buffers (id<MTLBuffer>)
    void* bufWorkspaceHi_ = nullptr;
    void* bufWorkspaceLo_ = nullptr;
    void* bufRhsHi_       = nullptr;
    void* bufRhsLo_       = nullptr;
    void* bufElimPivotHi_ = nullptr;
    void* bufElimPivotLo_ = nullptr;
    void* bufElimRhsHi_   = nullptr;
    void* bufElimRhsLo_   = nullptr;
    void* bufElimRowHi_   = nullptr;
    void* bufElimRowLo_   = nullptr;
    void* bufElimVarIds_  = nullptr;
    void* bufElimNeighborIds_    = nullptr;
    void* bufElimNeighborCount_  = nullptr;
    void* bufSolutionHi_  = nullptr;
    void* bufSolutionLo_  = nullptr;

    uint32_t numSlots_ = 0;
    uint32_t maxElimRecords_ = 0;

    static bool checkMetalSupport();
    bool initializeImpl();
    void shutdownImpl();
    std::vector<double> executeContractionImpl(
        const TNCompiledProgram& prog, uint32_t nodeCount, double gmin);

    // MSL shader source (equivalent to tn_contraction.wgsl)
    static const char* getMSLShaderSource();
#endif
};

/**
 * cuda_contraction_backend.h — NVIDIA cuTensorNet Backend Stub (Gap 3)
 *
 * Compile-guarded stub for NVIDIA CUDA/cuTensorNet integration.
 * Interface defined here; implementation requires CUDA SDK.
 */
class CudaContractionBackend {
public:
    static bool isAvailable() {
#if defined(ACUTESIM_CUDA_ENABLED)
        return true;
#else
        return false;
#endif
    }

    bool initialize() { return false; }

    std::vector<double> executeContraction(
        const TNCompiledProgram& prog,
        uint32_t nodeCount,
        double gmin)
    {
        (void)prog; (void)nodeCount; (void)gmin;
        return {};  // Stub — requires cuTensorNet SDK
    }

    void shutdown() {}
};
