#pragma once

#include "../tensors/physics_tensors.h"
#include "../netlist/circuit.h"
#include "../infrastructure/tn_compiler.h"  // TNCompiledProgram, ContractionTree
#include "../infrastructure/contraction_render_graph.h"  // RenderGraphSchedule
#include "tensor_workspace.h"               // EliminationRecord (for back-substitution)
#include "apple_uma.h"                      // Phase 5.7.2: UMA zero-copy
#include "thermal_monitor.h"                // Phase 5.7.2: Thermal budget
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

    // Pipeline-overlapped NR loop (primary solver entry point — legacy CSR path)
    std::vector<double> runHybridNRLoop(
        const TensorNetlist& netlist,
        int   maxIter,
        double tol,
        double gmin,
        double time = 0.0,
        double h    = 0.0);

    // ── Gap 1: TN Contraction Backend ────────────────────────────────────────
    // Upload a compiled TN program's MPOs and ContractionTree to GPU buffers.
    // Must be called once after uploadNetlist() whenever topology changes.
    // Idempotent — safe to call again when parameters change (MPO values update).
    void uploadContractionProgram(const TNCompiledProgram& prog, uint32_t nodeCount);

    // Execute the contraction tree on GPU via tn_contraction.wgsl.
    // Walks the ContractionTree bottom-up, dispatching seed/merge/elim kernels.
    // On completion, the solution vector is written to voltageBufferHi/Lo.
    // Returns the recovered node voltages (downloaded from voltageBufferHi/Lo).
    //
    // Invariants:
    //   - uploadContractionProgram() must have been called first.
    //   - KCL residual check is performed by the caller (circuitsim_solver.cpp).
    std::vector<double> executeContractionSweep(uint32_t nodeCount, double gmin);

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

    // Voltage route buffers — per-device terminal index lists.
    // Enables cooperative workgroup preload into var<workgroup> cache (Phase B).
    // Diode:  2 x u32 per device (anode-1, cathode-1; 0xFFFFFFFF = ground).
    // MOSFET: 4 x u32 per device (drain, gate, source, body terminal indices).
    // BJT:    3 x u32 per device (collector, base, emitter terminal indices).
    WGPUBuffer diodeVoltageRouteBuffer  = nullptr;
    WGPUBuffer mosfetVoltageRouteBuffer = nullptr;
    WGPUBuffer bjtVoltageRouteBuffer    = nullptr;

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

    // ── Gap 1: TN Contraction GPU Buffers ────────────────────────────────────
    // Workspace hi/lo buffer pair: flat storage for all intermediate tensor slots.
    // Layout: slot s at byte offset s * maxRank * maxRank * sizeof(float).
    WGPUBuffer tnWorkspaceHi = nullptr;
    WGPUBuffer tnWorkspaceLo = nullptr;
    // Per-slot RHS vectors: slot s at byte offset s * maxRank * sizeof(float).
    WGPUBuffer tnRhsHi = nullptr;
    WGPUBuffer tnRhsLo = nullptr;
    // Contraction uniforms buffer (ContractionUniforms struct — matches WGSL layout)
    WGPUBuffer tnUniformBuffer = nullptr;
    // Index-mapping tables for current contraction step (left + right open indices)
    WGPUBuffer tnIndexMapLeft  = nullptr;
    WGPUBuffer tnIndexMapRight = nullptr;

    // Elimination records (Group 2) for zero-sync Schur elimination
    WGPUBuffer tnElimPivotHi = nullptr;
    WGPUBuffer tnElimPivotLo = nullptr;
    WGPUBuffer tnElimRhsHi = nullptr;
    WGPUBuffer tnElimRhsLo = nullptr;
    WGPUBuffer tnElimRowHi = nullptr;
    WGPUBuffer tnElimRowLo = nullptr;
    WGPUBuffer tnElimVarIds = nullptr;
    WGPUBuffer tnElimNeighborIds = nullptr;
    WGPUBuffer tnElimNeighborCount = nullptr;
    WGPUBuffer tnElimUniformBuffer = nullptr;
    
    // Final solution vectors (Group 2)
    WGPUBuffer tnSolutionHi = nullptr;
    WGPUBuffer tnSolutionLo = nullptr;

    // Pipelines for tn_contraction.wgsl entry points
    WGPUComputePipeline tnSeedLeafPipeline  = nullptr;
    WGPUComputePipeline tnMergeAccumPipeline = nullptr;
    WGPUComputePipeline tnSchurElimPipeline  = nullptr;
    WGPUComputePipeline tnSchurElimRecordPipeline = nullptr;
    WGPUComputePipeline tnBackSubstitutePipeline = nullptr;

    // Bind groups for TN contraction
    // group 0: workspace
    // group 1: index maps
    // group 2: elimination records & solution
    WGPUBindGroup tnBindGroup0 = nullptr;
    WGPUBindGroup tnBindGroup1 = nullptr;
    WGPUBindGroup tnBindGroup2 = nullptr;

    // Current TN workspace metadata (set by uploadContractionProgram)
    uint32_t tnMaxRank_   = 0;   // padded slot stride (max device rank + margin)
    uint32_t tnNumSlots_  = 0;   // total slots = number of tree nodes

    // Gap 1 Phase 2: tree walk state (set by uploadContractionProgram)
    const ContractionTree* lastTree_ = nullptr;
    std::vector<std::vector<uint32_t>> leafTerminalNodes_;

    // Staging buffers for per-node slot readback during tree walk
    WGPUBuffer tnStagingMatHi = nullptr;
    WGPUBuffer tnStagingMatLo = nullptr;
    WGPUBuffer tnStagingRhsHi = nullptr;
    WGPUBuffer tnStagingRhsLo = nullptr;

    // Phase 5.7.4: Render graph schedule (slot aliasing + wavefronts)
    RenderGraphSchedule renderGraph_;

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

    // ── Gap 1: TN contraction private helpers ────────────────────────────────
    // Load and compile tn_contraction.wgsl, create the three TN pipelines.
    void createTNPipelines();
    // Create tnBindGroup0/1 from auto-derived pipeline layouts.
    void createTNBindGroups();
    // Upload ContractionUniforms struct for one tree node.
    void uploadTNUniforms(uint32_t slotLeft, uint32_t slotRight, uint32_t slotResult,
                          uint32_t rankLeft, uint32_t rankRight, uint32_t rankResult,
                          uint32_t elimRow,  uint32_t doSchur);
    // Upload the open-index mapping vectors for left and right children.
    void uploadIndexMaps(const std::vector<uint32_t>& mapLeft,
                         const std::vector<uint32_t>& mapRight);
    // Upload ElimRecordUniforms for zero-sync schur record / back sub
    void uploadTNElimUniforms(uint32_t record_offset, uint32_t elim_var_id,
                              uint32_t num_neighbors, uint32_t backsub_count,
                              uint32_t slot_result, uint32_t rank_result,
                              uint32_t elim_row, float gmin);
    // Dispatch one of the three TN kernels on `numElems` elements.
    void dispatchTNKernel(WGPUComputePipeline pl, uint32_t numElems,
                          WGPUCommandEncoder enc, WGPUComputePassEncoder pass);
#endif
};
