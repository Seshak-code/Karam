#pragma once

#include "physics_tensors.h"
#include "../../components/circuit.h"
#include <vector>
#include <string>
#include <functional>

#ifdef __EMSCRIPTEN__
#include <webgpu/webgpu.h>
#else
#include <webgpu/webgpu.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#endif

// Forward declarations from linalg.h / circuitsim.h
struct Csr_matrix;
struct SolverStep;

/**
 * WebGPUSolver
 * Implements GPU-resident Newton-Raphson simulation using EmulatedF64 logic.
 *
 * Features:
 * - Persistent GPU Residency: All MNA buffers stay on GPU.
 * - EmulatedF64 Architecture: hi-lo float32 pairs for double-precision parity.
 * - Pipeline-Overlapped Hybrid NR Loop: GPU physics N+1 overlaps CPU LU solve N.
 * - CSR pattern caching for deterministic O(N) sparse assembly.
 */
class WebGPUSolver {
public:
    WebGPUSolver();
    ~WebGPUSolver();

    bool initialize(const TensorNetlist& netlist);

    // Core netlist upload (freezes topology, builds CSR stamp maps)
    void uploadNetlist(const TensorNetlist& netlist);

    // Pipeline-overlapped NR loop (primary solver entry point)
    // Returns node voltages after convergence (or maxIter exhaustion).
    std::vector<double> runHybridNRLoop(
        const TensorNetlist& netlist,
        int   maxIter,
        double tol,
        double gmin,
        double time = 0.0,
        double h    = 0.0);

    // Legacy single-step interface (kept for compatibility)
    void runNRStep(double time, double h);
    bool checkConvergence();

    // Download final solution from GPU → CPU
    std::vector<double> downloadSolution();

    // Waveform ring-buffer readback (Phase 2.2)
    std::vector<std::vector<double>> downloadWaveform();

    // Upload CPU-computed deltaV back to GPU (pipeline overlap step)
    void uploadDeltaV(const std::vector<double>& deltaV);

    // Async physics dispatch: kicks off GPU kernels, returns immediately
    void dispatchPhysicsAsync(double time, double h);

    // Initiate async Jacobian/RHS readback from GPU → staging buffers
    void initiateReadback();

    // Poll whether last readback is complete (returns true when data is ready)
    bool isReadbackReady();

    // --- Double-buffered NR iteration state (ping-pong for pipeline overlap) ---
    struct NRIterationState {
        std::vector<float> jacobian_hi, jacobian_lo;
        std::vector<float> rhs_hi, rhs_lo;
        bool ready = false;
    };
    NRIterationState& getReadbackData();

private:
    WGPUDevice   device   = nullptr;
    WGPUQueue    queue    = nullptr;
    WGPUInstance instance = nullptr;

    // --- Persistent MNA Buffers (EmulatedF64 split) ---
    WGPUBuffer voltageBufferHi  = nullptr;
    WGPUBuffer voltageBufferLo  = nullptr;
    WGPUBuffer deltaVBufferHi   = nullptr;
    WGPUBuffer deltaVBufferLo   = nullptr;
    WGPUBuffer rhsBufferHi      = nullptr;
    WGPUBuffer rhsBufferLo      = nullptr;
    WGPUBuffer jacobianBufferHi = nullptr;  // CSR hi — replaces CsrMatrix struct
    WGPUBuffer jacobianBufferLo = nullptr;  // CSR lo

    // --- Topology Buffers (Frozen per netlist upload) ---
    WGPUBuffer csrRowPtrBuffer  = nullptr;
    WGPUBuffer csrColIdxBuffer  = nullptr;
    WGPUBuffer diodeMapBuffer   = nullptr;
    WGPUBuffer mosfetMapBuffer  = nullptr;
    WGPUBuffer bjtMapBuffer     = nullptr;  // BJT CSR stamp maps (new)
    WGPUBuffer globalStateBuffer = nullptr;

    // --- Component SoA Buffers ---
    WGPUBuffer diodeSoABuffer  = nullptr;
    WGPUBuffer mosfetSoABuffer = nullptr;
    WGPUBuffer bjtSoABuffer    = nullptr;  // BJTs (new)

    // --- Convergence / Residual Buffers ---
    WGPUBuffer residualBuffer     = nullptr;  // Per-workgroup max norms
    WGPUBuffer convergenceFlagBuf = nullptr;  // [0] = 1 if converged

    // --- Waveform Buffering (Phase 2.2) ---
    WGPUBuffer waveformConfigBuffer = nullptr;
    WGPUBuffer waveformStateBuffer  = nullptr;
    WGPUBuffer waveformDataBuffer   = nullptr;

    // --- Staging Buffers for Async Readback ---
    WGPUBuffer stagingJacobianHi  = nullptr;
    WGPUBuffer stagingJacobianLo  = nullptr;
    WGPUBuffer stagingRhsHi       = nullptr;
    WGPUBuffer stagingRhsLo       = nullptr;
    WGPUBuffer stagingVoltageHi   = nullptr;
    WGPUBuffer stagingVoltageLo   = nullptr;

    // --- Persistent Bind Groups ---
    WGPUBindGroup bindGroup0 = nullptr;  // MNA state
    WGPUBindGroup bindGroup1 = nullptr;  // Topology + simulation state
    WGPUBindGroup bindGroup2 = nullptr;  // Waveform

    // --- Compute Pipelines ---
    WGPUComputePipeline physicsPipelineDiodes   = nullptr;
    WGPUComputePipeline physicsPipelineMosfets  = nullptr;
    WGPUComputePipeline physicsPipelineBJTs     = nullptr;  // new
    WGPUComputePipeline assemblyPipeline        = nullptr;
    WGPUComputePipeline solutionUpdatePipeline  = nullptr;  // renamed from solverPipeline
    WGPUComputePipeline residualPipeline        = nullptr;  // new
    WGPUComputePipeline convergencePipeline     = nullptr;
    WGPUComputePipeline recordWaveformPipeline  = nullptr;

    // --- Topology State (frozen per uploadNetlist) ---
    size_t numNodes_   = 0;
    size_t numDiodes_  = 0;
    size_t numMosfets_ = 0;
    size_t numBJTs_    = 0;
    int    csrNnz_     = 0;

    // --- Pipeline-Overlap Double Buffer ---
    NRIterationState readbackBuffers_[2];
    int activeReadback_ = 0;
    bool readbackPending_ = false;

    // --- Internal Helpers ---
    bool initWebGPU();
    void setupResources(const TensorNetlist& netlist);
    void createPipelines(WGPUShaderModule shaderModule);
    void createBindGroups();

    WGPUShaderModule loadShader(const std::string& source);

    // Create a MAP_READ staging buffer of the given byte size
    WGPUBuffer createStagingBuffer(size_t byteSize, const char* label);

    // Record a CopyBufferToBuffer into the given encoder (GPU→staging)
    void recordCopyToStaging(WGPUCommandEncoder enc,
                             WGPUBuffer src, WGPUBuffer dst, size_t byteSize);

    // Build CSR position maps for diodes/mosfets/BJTs from the pattern
    void buildStampMaps(const TensorNetlist& netlist);
};
