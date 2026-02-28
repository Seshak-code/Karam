#pragma once

#include "../tensors/physics_tensors.h"
#include "../netlist/circuit.h"
#include <vector>
#include <string>
#include <functional>
#include <mutex>

#ifdef ACUTESIM_GPU_ENABLED
#ifdef __EMSCRIPTEN__
#include <webgpu/webgpu.h>
#else
#include <webgpu/webgpu.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#endif
#endif

// Forward declarations from linalg.h / circuitsim.h
struct Csr_matrix;
struct SolverStep;

/**
 * WebGPUSolver
 * Implements GPU-resident Newton-Raphson simulation using EmulatedF64 logic.
 */
class WebGPUSolver {
public:
    WebGPUSolver();
    ~WebGPUSolver();

#ifdef ACUTESIM_GPU_ENABLED
    // External-device constructor: borrows an already-created device and queue
    // from GPUContextManager. Does NOT call initWebGPU(), does NOT call
    // dawnProcSetProcs(), and does NOT release the handles on destruction.
    explicit WebGPUSolver(WGPUDevice externalDevice, WGPUQueue externalQueue);
#endif

#ifdef ACUTESIM_GPU_ENABLED
    bool initialize(const TensorNetlist& netlist);

    // Core netlist upload (freezes topology, builds CSR stamp maps)
    void uploadNetlist(const TensorNetlist& netlist);

    // Pipeline-overlapped NR loop (primary solver entry point)
    std::vector<double> runHybridNRLoop(
        const TensorNetlist& netlist,
        int   maxIter,
        double tol,
        double gmin,
        double time = 0.0,
        double h    = 0.0);

    // Download final solution from GPU → CPU
    std::vector<double> downloadSolution();
#endif

private:
#ifdef ACUTESIM_GPU_ENABLED
    // Core WebGPU handles
    WGPUDevice   device   = nullptr;
    WGPUQueue    queue    = nullptr;
    WGPUInstance instance = nullptr;

    // SoA device buffers (GPU-resident physics state)
    WGPUBuffer diodeSoABuffer  = nullptr;
    WGPUBuffer mosfetSoABuffer = nullptr;
    WGPUBuffer bjtSoABuffer    = nullptr;

    // MNA voltage / delta-V (EmulatedF64 hi/lo pairs)
    WGPUBuffer voltageBufferHi  = nullptr;
    WGPUBuffer voltageBufferLo  = nullptr;
    WGPUBuffer deltaVBufferHi   = nullptr;
    WGPUBuffer deltaVBufferLo   = nullptr;

    // Jacobian / RHS (CSR, EmulatedF64)
    WGPUBuffer rhsBufferHi      = nullptr;
    WGPUBuffer rhsBufferLo      = nullptr;
    WGPUBuffer jacobianBufferHi = nullptr;
    WGPUBuffer jacobianBufferLo = nullptr;

    // CSR topology
    WGPUBuffer csrRowPtrBuffer = nullptr;
    WGPUBuffer csrColIdxBuffer = nullptr;

    // Device stamp position maps
    WGPUBuffer diodeMapBuffer  = nullptr;
    WGPUBuffer mosfetMapBuffer = nullptr;
    WGPUBuffer bjtMapBuffer    = nullptr;

    // Global time/step state
    WGPUBuffer globalStateBuffer = nullptr;

    // Convergence
    WGPUBuffer residualBuffer     = nullptr;
    WGPUBuffer convergenceFlagBuf = nullptr;

    // Waveform ring buffer
    WGPUBuffer waveformConfigBuffer = nullptr;
    WGPUBuffer waveformStateBuffer  = nullptr;
    WGPUBuffer waveformDataBuffer   = nullptr;

    // CPU-readable staging buffers
    WGPUBuffer stagingJacobianHi = nullptr;
    WGPUBuffer stagingJacobianLo = nullptr;
    WGPUBuffer stagingRhsHi      = nullptr;
    WGPUBuffer stagingRhsLo      = nullptr;
    WGPUBuffer stagingVoltageHi  = nullptr;
    WGPUBuffer stagingVoltageLo  = nullptr;

    // Bind groups
    WGPUBindGroup bindGroup0 = nullptr;
    WGPUBindGroup bindGroup1 = nullptr;
    WGPUBindGroup bindGroup2 = nullptr;

    // Compute pipelines
    WGPUComputePipeline physicsPipelineDiodes  = nullptr;
    WGPUComputePipeline physicsPipelineMosfets = nullptr;
    WGPUComputePipeline physicsPipelineBJTs    = nullptr;
    WGPUComputePipeline assemblyPipeline       = nullptr;
    WGPUComputePipeline solutionUpdatePipeline = nullptr;
    WGPUComputePipeline residualPipeline       = nullptr;
    WGPUComputePipeline convergencePipeline    = nullptr;
    WGPUComputePipeline recordWaveformPipeline = nullptr;

    // Topology/sizing
    size_t numNodes_   = 0;
    size_t numDiodes_  = 0;
    size_t numMosfets_ = 0;
    size_t numBJTs_    = 0;
    int    csrNnz_     = 0;

    // Pipeline-overlap readback state
    struct NRIterationState {
        std::vector<double> jacobianHi;
        std::vector<double> jacobianLo;
        std::vector<double> rhsHi;
        std::vector<double> rhsLo;
    };
    NRIterationState readbackBuffers_[2];
    int  activeReadback_  = 0;
    bool readbackPending_ = false;

    bool ownsDevice_ = true;   // false when device/queue provided externally

    // Private implementation methods
    bool              initWebGPU();
    WGPUShaderModule  loadShader(const std::string& source);
    void              setupResources(const TensorNetlist& netlist);
    void              createPipelines(WGPUShaderModule sm);
    void              buildStampMaps(const TensorNetlist& netlist);
    void              uploadDeltaV(const std::vector<double>& deltaV);
    void              dispatchPhysicsAsync(double time, double h);
    void              initiateReadback();
    bool              isReadbackReady();
    NRIterationState& getReadbackData();
    void              runNRStep(double time, double h);
    bool              checkConvergence();
    std::vector<std::vector<double>> downloadWaveform();
    WGPUBuffer        createStagingBuffer(size_t byteSize, const char* label = nullptr);
    void              recordCopyToStaging(WGPUCommandEncoder enc,
                                          WGPUBuffer src, WGPUBuffer dst, size_t sz);
    void              createBindGroups();
#endif
};
