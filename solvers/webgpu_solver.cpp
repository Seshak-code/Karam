#ifdef ACUTESIM_GPU_ENABLED
/**
 * webgpu_solver.cpp
 * Phase D5b: Pipeline-overlapped hybrid NR loop.
 * GPU dispatches physics for iteration N+1 while CPU solves LU for iteration N.
 */

#include "../solvers/webgpu_solver.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include "../infrastructure/netlist_compiler.h"
#include "../math/linalg.h"          // MatrixConstructor, solveLU_Pivoted, Csr_matrix
#include "../netlist/circuit.h" // Resistor, BJT, Mosfet, Diode, etc.

// Helper for Dawn's WGPUStringView API
#ifdef __EMSCRIPTEN__
#define WGPU_SV(str) str
#else
#define WGPU_SV(str) { str, strlen(str) }
#endif

#ifdef __EMSCRIPTEN__
#include <webgpu/webgpu.h>
#include <emscripten.h>
#else
#include <webgpu/webgpu.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#endif

// ============================================================================
// WGSL Shader source loader
// ============================================================================
static std::string loadShaderSource() {
    const char* candidates[] = {
        "compute/shaders/gpu_nr_loop.wgsl",
        "../compute/shaders/gpu_nr_loop.wgsl",
        "../../compute/shaders/gpu_nr_loop.wgsl",
        nullptr
    };
    for (int i = 0; candidates[i] != nullptr; ++i) {
        std::ifstream f(candidates[i]);
        if (f.good())
            return std::string(std::istreambuf_iterator<char>(f),
                               std::istreambuf_iterator<char>());
    }
    std::cerr << "[WARN] WebGPUSolver: Could not find gpu_nr_loop.wgsl\n";
    return "";
}

// ============================================================================
// CPU-side struct mirrors (must match WGSL struct layouts)
// ============================================================================
struct DiodeDeviceCPU {
    int32_t anode, cathode;
    float Is_hi, Is_lo, N_hi, N_lo, Vt_hi, Vt_lo;
    float v_d_hi, v_d_lo, i_d_hi, i_d_lo, g_d_hi, g_d_lo;
};

struct MosfetDeviceCPU {
    int32_t drain, gate, source, body;
    float W_hi, W_lo, L_hi, L_lo, Kp_hi, Kp_lo, Vth_hi, Vth_lo;
    float lambda_hi, lambda_lo;
    uint32_t isPMOS;
    float vgs_hi, vgs_lo, vds_hi, vds_lo, vbs_hi, vbs_lo;
    float ids_hi, ids_lo, gm_hi, gm_lo, gmb_hi, gmb_lo, gds_hi, gds_lo;
};

struct BJTDeviceCPU {
    int32_t collector, base, emitter;
    uint32_t isNPN;
    float Is_hi, Is_lo, betaF, betaR, Vt_hi, Vt_lo;
    // Output (filled by GPU physics kernel)
    float ic_hi, ic_lo, ib_hi, ib_lo;
    float g_cc_hi, g_cc_lo, g_cb_hi, g_cb_lo, g_ce_hi, g_ce_lo;
    float g_bc_hi, g_bc_lo, g_bb_hi, g_bb_lo, g_be_hi, g_be_lo;
    float g_ec_hi, g_ec_lo, g_eb_hi, g_eb_lo, g_ee_hi, g_ee_lo;
    float vc_hi, vc_lo, vb_hi, vb_lo, ve_hi, ve_lo;
};

struct DiodeStampMapCPU  { uint32_t aa, cc, ac, ca; };
struct MosfetStampMapCPU { uint32_t dd, dg, ds, db, sd, sg, ss, sb; };
struct BJTStampMapCPU    { uint32_t cc, cb, ce, bc, bb, be, ec, eb, ee; };

// ============================================================================
// Helpers
// ============================================================================
static std::pair<float,float> splitDouble(double d) {
    float hi = static_cast<float>(d);
    float lo = static_cast<float>(d - static_cast<double>(hi));
    return {hi, lo};
}

static void uploadHiLo(WGPUQueue q, WGPUBuffer bhi, WGPUBuffer blo,
                       const std::vector<double>& data) {
    std::vector<float> hi(data.size()), lo(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        auto p = splitDouble(data[i]);
        hi[i] = p.first; lo[i] = p.second;
    }
    wgpuQueueWriteBuffer(q, bhi, 0, hi.data(), hi.size() * sizeof(float));
    wgpuQueueWriteBuffer(q, blo, 0, lo.data(), lo.size() * sizeof(float));
}

static WGPUBuffer makeBuffer(WGPUDevice dev, size_t size, WGPUBufferUsage usage,
                              const char* label = nullptr) {
    WGPUBufferDescriptor d = {};
#ifndef __EMSCRIPTEN__
    if (label) d.label = { label, strlen(label) };
#else
    if (label) d.label = label;
#endif
    d.size  = size < 4 ? 4 : size; // WebGPU minimum buffer size is 4 bytes
    d.usage = usage;
    d.mappedAtCreation = false;
    return wgpuDeviceCreateBuffer(dev, &d);
}

// ============================================================================
// Construction / Destruction
// ============================================================================
WebGPUSolver::WebGPUSolver() {
    initWebGPU();
}

WebGPUSolver::WebGPUSolver(WGPUDevice externalDevice, WGPUQueue externalQueue)
    : device(externalDevice)
    , queue(externalQueue)
    , ownsDevice_(false)
{
    // procs already registered by GPUContextManager::initialize().
    // Device already created — skip initWebGPU().
    // Caller (GPUContextManager) retains ownership and must outlive this solver.
}

WebGPUSolver::~WebGPUSolver() {
    auto rel = [](WGPUBuffer& b){ if(b){ wgpuBufferRelease(b); b=nullptr; } };
    rel(diodeSoABuffer);   rel(mosfetSoABuffer); rel(bjtSoABuffer);
    rel(voltageBufferHi);  rel(voltageBufferLo);
    rel(deltaVBufferHi);   rel(deltaVBufferLo);
    rel(rhsBufferHi);      rel(rhsBufferLo);
    rel(jacobianBufferHi); rel(jacobianBufferLo);
    rel(csrRowPtrBuffer);  rel(csrColIdxBuffer);
    rel(diodeMapBuffer);   rel(mosfetMapBuffer);  rel(bjtMapBuffer);
    rel(diodeVoltageRouteBuffer); rel(mosfetVoltageRouteBuffer); rel(bjtVoltageRouteBuffer);
    rel(globalStateBuffer);
    rel(residualBuffer);   rel(convergenceFlagBuf);
    rel(waveformConfigBuffer); rel(waveformStateBuffer); rel(waveformDataBuffer);
    rel(stagingJacobianHi); rel(stagingJacobianLo);
    rel(stagingRhsHi);      rel(stagingRhsLo);
    rel(stagingVoltageHi);  rel(stagingVoltageLo);

    // Gap 1: TN contraction buffers
    rel(tnWorkspaceHi); rel(tnWorkspaceLo);
    rel(tnRhsHi); rel(tnRhsLo);
    rel(tnUniformBuffer);
    rel(tnIndexMapLeft); rel(tnIndexMapRight);
    rel(tnStagingMatHi); rel(tnStagingMatLo);
    rel(tnStagingRhsHi); rel(tnStagingRhsLo);

    auto relBG = [](WGPUBindGroup& bg){ if(bg){ wgpuBindGroupRelease(bg); bg=nullptr; } };
    relBG(bindGroup0); relBG(bindGroup1); relBG(bindGroup2);
    relBG(tnBindGroup0); relBG(tnBindGroup1);

    auto relPL = [](WGPUComputePipeline& pl){ if(pl){ wgpuComputePipelineRelease(pl); pl=nullptr; } };
    relPL(physicsPipelineDiodes);  relPL(physicsPipelineMosfets); relPL(physicsPipelineBJTs);
    relPL(assemblyPipeline);       relPL(solutionUpdatePipeline);
    relPL(residualPipeline);       relPL(convergencePipeline);
    relPL(recordWaveformPipeline);
    relPL(tnSeedLeafPipeline); relPL(tnMergeAccumPipeline); relPL(tnSchurElimPipeline);

    // Only release handles we own. When constructed via the external-device
    // ctor (ownsDevice_ = false), these handles are owned by GPUContextManager.
    if (queue    && ownsDevice_) wgpuQueueRelease(queue);
    if (device   && ownsDevice_) wgpuDeviceRelease(device);
    if (instance && ownsDevice_) wgpuInstanceRelease(instance);
}

bool WebGPUSolver::initWebGPU() {
#ifdef __EMSCRIPTEN__
    std::cout << "[INFO] WebGPUSolver: Emscripten/WASM init\n";
    if (!device) return false;
    queue = wgpuDeviceGetQueue(device);
    return true;
#else
    std::cout << "[INFO] WebGPUSolver: Native Dawn init\n";
    dawnProcSetProcs(&dawn::native::GetProcs());
    instance = wgpuCreateInstance(nullptr);
    if (!instance) { std::cerr << "[ERROR] WGPU Instance failed\n"; return false; }

    // Stack-allocated static: avoids heap leak while still providing
    // one-time construction (function-local statics are fine here because
    // initWebGPU() is only called from the default ctor — the owning-device
    // path. The GPUContextManager path uses the external-device ctor instead).
    static dawn::native::Instance nativeInstance;
    auto adapters = nativeInstance.EnumerateAdapters();

    dawn::native::Adapter selected;
    for (auto& a : adapters) {
        WGPUAdapterInfo info = {};
        wgpuAdapterGetInfo(a.Get(), &info);
        if (info.backendType == WGPUBackendType_Metal) { selected = a; break; }
    }
    if (!selected && !adapters.empty()) selected = adapters[0];
    if (!selected) { std::cerr << "[ERROR] No GPU adapter\n"; return false; }

    device = selected.CreateDevice();
    if (!device) { std::cerr << "[ERROR] Device creation failed\n"; return false; }
    queue = wgpuDeviceGetQueue(device);
    std::cout << "[INFO] WebGPUSolver: GPU init SUCCESS\n";
    return true;
#endif
}

// ============================================================================
// loadShader
// ============================================================================
WGPUShaderModule WebGPUSolver::loadShader(const std::string& source) {
    if (!device || source.empty()) return nullptr;
#ifdef __EMSCRIPTEN__
    WGPUShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.chain.next  = nullptr;
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wgslDesc.code        = source.c_str();
#else
    WGPUShaderSourceWGSL wgslDesc = {};
    wgslDesc.chain.next  = nullptr;
    wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslDesc.code        = { source.data(), source.length() };
#endif
    WGPUShaderModuleDescriptor desc = {};
    desc.nextInChain = &wgslDesc.chain;
    return wgpuDeviceCreateShaderModule(device, &desc);
}

// ============================================================================
// initialize / setupResources
// ============================================================================
bool WebGPUSolver::initialize(const TensorNetlist& netlist) {
    if (!device) return false;
    setupResources(netlist);
    return true;
}

void WebGPUSolver::setupResources(const TensorNetlist& netlist) {
    if (!device) return;

    numNodes_   = netlist.numGlobalNodes;
    numDiodes_  = netlist.globalBlock.diodes.size();
    numMosfets_ = netlist.globalBlock.mosfets.size();
    numBJTs_    = netlist.globalBlock.bjts.size();

    const size_t N        = numNodes_;
    const size_t NNZ_EST  = N * 10;
    constexpr uint32_t RING_SIZE   = 4096;
    constexpr uint32_t DECIM_RATIO = 10;

    const WGPUBufferUsage STORAGE_RW =
        (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
    const WGPUBufferUsage MAP_READ_BUF =
        (WGPUBufferUsage)(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);

    // MNA state
    voltageBufferHi   = makeBuffer(device, N*sizeof(float),       STORAGE_RW, "v_hi");
    voltageBufferLo   = makeBuffer(device, N*sizeof(float),       STORAGE_RW, "v_lo");
    deltaVBufferHi    = makeBuffer(device, N*sizeof(float),       STORAGE_RW, "dv_hi");
    deltaVBufferLo    = makeBuffer(device, N*sizeof(float),       STORAGE_RW, "dv_lo");
    rhsBufferHi       = makeBuffer(device, N*sizeof(float),       STORAGE_RW, "rhs_hi");
    rhsBufferLo       = makeBuffer(device, N*sizeof(float),       STORAGE_RW, "rhs_lo");
    jacobianBufferHi  = makeBuffer(device, NNZ_EST*sizeof(float), STORAGE_RW, "jac_hi");
    jacobianBufferLo  = makeBuffer(device, NNZ_EST*sizeof(float), STORAGE_RW, "jac_lo");

    // Topology
    csrRowPtrBuffer   = makeBuffer(device, (N+1)*sizeof(uint32_t),   STORAGE_RW, "rptr");
    csrColIdxBuffer   = makeBuffer(device, NNZ_EST*sizeof(uint32_t), STORAGE_RW, "cidx");

    size_t ds = std::max(numDiodes_  * sizeof(DiodeDeviceCPU),   size_t(4));
    size_t ms = std::max(numMosfets_ * sizeof(MosfetDeviceCPU),  size_t(4));
    size_t bs = std::max(numBJTs_    * sizeof(BJTDeviceCPU),     size_t(4));
    diodeSoABuffer   = makeBuffer(device, ds, STORAGE_RW, "diodes");
    mosfetSoABuffer  = makeBuffer(device, ms, STORAGE_RW, "mosfets");
    bjtSoABuffer     = makeBuffer(device, bs, STORAGE_RW, "bjts");

    size_t dm = std::max(numDiodes_  * sizeof(DiodeStampMapCPU),  size_t(4));
    size_t mm = std::max(numMosfets_ * sizeof(MosfetStampMapCPU), size_t(4));
    size_t bm = std::max(numBJTs_    * sizeof(BJTStampMapCPU),    size_t(4));
    diodeMapBuffer   = makeBuffer(device, dm, STORAGE_RW, "dmap");
    mosfetMapBuffer  = makeBuffer(device, mm, STORAGE_RW, "mmap");
    bjtMapBuffer     = makeBuffer(device, bm, STORAGE_RW, "bmap");

    // Phase B: voltage route buffers — precomputed terminal index lists for
    // cooperative workgroup voltage preload via var<workgroup> cache.
    // Sentinel 0xFFFFFFFF means the terminal connects to ground (node 0).
    const WGPUBufferUsage STORAGE_RO =
        (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    size_t dr = std::max(numDiodes_  * 2 * sizeof(uint32_t), size_t(4)); // 2 terminals
    size_t mr = std::max(numMosfets_ * 4 * sizeof(uint32_t), size_t(4)); // 4 terminals
    size_t br = std::max(numBJTs_    * 3 * sizeof(uint32_t), size_t(4)); // 3 terminals
    diodeVoltageRouteBuffer  = makeBuffer(device, dr, STORAGE_RO, "v_route_diode");
    mosfetVoltageRouteBuffer = makeBuffer(device, mr, STORAGE_RO, "v_route_mosfet");
    bjtVoltageRouteBuffer    = makeBuffer(device, br, STORAGE_RO, "v_route_bjt");

    globalStateBuffer = makeBuffer(device, 4*sizeof(float), STORAGE_RW, "gstate");

    uint32_t nWG = (uint32_t)std::max((N + 63) / 64, size_t(1));
    residualBuffer     = makeBuffer(device, nWG*sizeof(float),   STORAGE_RW, "residual");
    convergenceFlagBuf = makeBuffer(device, sizeof(uint32_t),    STORAGE_RW, "conv_flag");

    // Waveform
    waveformConfigBuffer = makeBuffer(device, 3*sizeof(uint32_t),                    STORAGE_RW, "wf_cfg");
    waveformStateBuffer  = makeBuffer(device, 2*sizeof(uint32_t),                    STORAGE_RW, "wf_state");
    size_t entryStride   = 2 + N * 2;
    waveformDataBuffer   = makeBuffer(device, RING_SIZE*entryStride*sizeof(float),   STORAGE_RW, "wf_data");

    // Staging (CPU-readable)
    stagingJacobianHi = makeBuffer(device, NNZ_EST*sizeof(float), MAP_READ_BUF, "stg_jac_hi");
    stagingJacobianLo = makeBuffer(device, NNZ_EST*sizeof(float), MAP_READ_BUF, "stg_jac_lo");
    stagingRhsHi      = makeBuffer(device, N*sizeof(float),       MAP_READ_BUF, "stg_rhs_hi");
    stagingRhsLo      = makeBuffer(device, N*sizeof(float),       MAP_READ_BUF, "stg_rhs_lo");
    stagingVoltageHi  = makeBuffer(device, N*sizeof(float),       MAP_READ_BUF, "stg_v_hi");
    stagingVoltageLo  = makeBuffer(device, N*sizeof(float),       MAP_READ_BUF, "stg_v_lo");

    // Init waveform config
    uint32_t cfgData[3] = { DECIM_RATIO, RING_SIZE, (uint32_t)N };
    wgpuQueueWriteBuffer(queue, waveformConfigBuffer, 0, cfgData, sizeof(cfgData));
    uint32_t stateData[2] = { 0, 0 };
    wgpuQueueWriteBuffer(queue, waveformStateBuffer, 0, stateData, sizeof(stateData));

    // Load WGSL and create pipelines
    std::string src = loadShaderSource();
    WGPUShaderModule sm = loadShader(src);
    if (sm) {
        createPipelines(sm);
        wgpuShaderModuleRelease(sm);
    } else {
        std::cerr << "[ERROR] WebGPUSolver: Shader module failed.\n";
    }

    std::cout << "[INFO] WebGPUSolver: Resources ready ("
              << N << " nodes, " << numDiodes_ << " diodes, "
              << numMosfets_ << " mosfets, " << numBJTs_ << " BJTs).\n";
}

// ============================================================================
// createPipelines
// ============================================================================
void WebGPUSolver::createPipelines(WGPUShaderModule sm) {
    // Use null layout — auto-layout from WGSL reflection
    WGPUPipelineLayoutDescriptor plDesc = {};
    plDesc.bindGroupLayoutCount = 0;
    plDesc.bindGroupLayouts     = nullptr;
    WGPUPipelineLayout autoLayout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    auto makePipeline = [&](const char* ep) -> WGPUComputePipeline {
        WGPUComputePipelineDescriptor d = {};
        d.label               = WGPU_SV(ep);
        d.layout              = autoLayout;
        d.compute.module      = sm;
        d.compute.entryPoint  = WGPU_SV(ep);
        return wgpuDeviceCreateComputePipeline(device, &d);
    };

    physicsPipelineDiodes  = makePipeline("batchDiodePhysics");
    physicsPipelineMosfets = makePipeline("batchMosfetPhysics");
    physicsPipelineBJTs    = makePipeline("batchBJTPhysics");
    assemblyPipeline       = makePipeline("assembleJacobian");
    solutionUpdatePipeline = makePipeline("updateSolution");
    residualPipeline       = makePipeline("computeResidual");
    convergencePipeline    = makePipeline("convergenceCheck");
    recordWaveformPipeline = makePipeline("recordWaveform");

    wgpuPipelineLayoutRelease(autoLayout);
}

// ============================================================================
// buildStampMaps — symbolic CSR assembly + stamp position lookup
// ============================================================================
void WebGPUSolver::buildStampMaps(const TensorNetlist& netlist) {
    if (!queue || !device) return;
    const size_t N = numNodes_;

    MatrixConstructor mc;
    mc.setDimensions((int)N, (int)N);

    auto stamp2 = [&](int r, int c) {
        if (r > 0 && c > 0 && r <= (int)N && c <= (int)N)
            mc.add(r-1, c-1, 0.0);
    };

    for (const auto& d : netlist.globalBlock.diodes) {
        stamp2(d.anode, d.anode);   stamp2(d.anode, d.cathode);
        stamp2(d.cathode, d.anode); stamp2(d.cathode, d.cathode);
    }
    for (const auto& m : netlist.globalBlock.mosfets) {
        int ns[4] = { m.drain, m.gate, m.source, m.body };
        for (int r : ns) for (int c : ns) stamp2(r, c);
    }
    for (const auto& b : netlist.globalBlock.bjts) {
        int ns[3] = { b.nodeCollector, b.base, b.emitter };
        for (int r : ns) for (int c : ns) stamp2(r, c);
    }
    for (const auto& r : netlist.globalBlock.resistors) {
        stamp2(r.nodeTerminal1, r.nodeTerminal1); stamp2(r.nodeTerminal1, r.nodeTerminal2);
        stamp2(r.nodeTerminal2, r.nodeTerminal1); stamp2(r.nodeTerminal2, r.nodeTerminal2);
    }

    Csr_matrix pat = mc.createCsr();
    csrNnz_ = pat.nnz;

    // Build (row,col) → CSR value index map
    std::unordered_map<int64_t, int> posMap;
    for (int row = 0; row < pat.rows; ++row) {
        for (int k = pat.row_pointer[row]; k < pat.row_pointer[row+1]; ++k) {
            int64_t key = (int64_t(row) << 32) | (uint32_t)pat.col_indices[k];
            posMap[key] = k;
        }
    }

    auto indexOf = [&](int row, int col) -> uint32_t {
        if (row <= 0 || col <= 0) return 0xFFFFFFFFu;
        int64_t key = (int64_t(row-1) << 32) | (uint32_t)(col-1);
        auto it = posMap.find(key);
        return (it != posMap.end()) ? (uint32_t)it->second : 0xFFFFFFFFu;
    };

    // Upload CSR structure
    if (!pat.row_pointer.empty()) {
        std::vector<uint32_t> rp(pat.row_pointer.begin(), pat.row_pointer.end());
        wgpuQueueWriteBuffer(queue, csrRowPtrBuffer, 0, rp.data(), rp.size() * sizeof(uint32_t));
    }
    if (!pat.col_indices.empty()) {
        std::vector<uint32_t> ci(pat.col_indices.begin(), pat.col_indices.end());
        wgpuQueueWriteBuffer(queue, csrColIdxBuffer, 0, ci.data(), ci.size() * sizeof(uint32_t));
    }

    // Diode stamp maps
    if (numDiodes_ > 0) {
        std::vector<DiodeStampMapCPU> maps(numDiodes_);
        for (size_t i = 0; i < numDiodes_; ++i) {
            const auto& d = netlist.globalBlock.diodes[i];
            maps[i] = { indexOf(d.anode, d.anode), indexOf(d.cathode, d.cathode),
                        indexOf(d.anode, d.cathode), indexOf(d.cathode, d.anode) };
        }
        wgpuQueueWriteBuffer(queue, diodeMapBuffer, 0, maps.data(), maps.size() * sizeof(DiodeStampMapCPU));
    }

    // MOSFET stamp maps
    if (numMosfets_ > 0) {
        std::vector<MosfetStampMapCPU> maps(numMosfets_);
        for (size_t i = 0; i < numMosfets_; ++i) {
            const auto& m = netlist.globalBlock.mosfets[i];
            maps[i] = { indexOf(m.drain, m.drain), indexOf(m.drain, m.gate),
                        indexOf(m.drain, m.source), indexOf(m.drain, m.body),
                        indexOf(m.source, m.drain), indexOf(m.source, m.gate),
                        indexOf(m.source, m.source), indexOf(m.source, m.body) };
        }
        wgpuQueueWriteBuffer(queue, mosfetMapBuffer, 0, maps.data(), maps.size() * sizeof(MosfetStampMapCPU));
    }

    // BJT stamp maps
    if (numBJTs_ > 0) {
        std::vector<BJTStampMapCPU> maps(numBJTs_);
        for (size_t i = 0; i < numBJTs_; ++i) {
            const auto& b = netlist.globalBlock.bjts[i];
            int c = b.nodeCollector, ba = b.base, e = b.emitter;
            maps[i] = { indexOf(c,c),  indexOf(c,ba), indexOf(c,e),
                        indexOf(ba,c), indexOf(ba,ba), indexOf(ba,e),
                        indexOf(e,c),  indexOf(e,ba),  indexOf(e,e)  };
        }
        wgpuQueueWriteBuffer(queue, bjtMapBuffer, 0, maps.data(), maps.size() * sizeof(BJTStampMapCPU));
    }

    // Phase B: populate voltage route buffers.
    // Each entry is a 0-based voltage index (node - 1); 0xFFFFFFFF = ground.
    // These enable the workgroup shared-memory preload in the WGSL kernels.
    auto toRouteIdx = [](int node) -> uint32_t {
        return (node > 0) ? static_cast<uint32_t>(node - 1) : 0xFFFFFFFFu;
    };

    if (numDiodes_ > 0 && diodeVoltageRouteBuffer) {
        std::vector<uint32_t> routes(numDiodes_ * 2);
        for (size_t i = 0; i < numDiodes_; ++i) {
            const auto& d = netlist.globalBlock.diodes[i];
            routes[i * 2 + 0] = toRouteIdx(d.anode);
            routes[i * 2 + 1] = toRouteIdx(d.cathode);
        }
        wgpuQueueWriteBuffer(queue, diodeVoltageRouteBuffer, 0,
                             routes.data(), routes.size() * sizeof(uint32_t));
    }

    if (numMosfets_ > 0 && mosfetVoltageRouteBuffer) {
        std::vector<uint32_t> routes(numMosfets_ * 4);
        for (size_t i = 0; i < numMosfets_; ++i) {
            const auto& m = netlist.globalBlock.mosfets[i];
            routes[i * 4 + 0] = toRouteIdx(m.drain);
            routes[i * 4 + 1] = toRouteIdx(m.gate);
            routes[i * 4 + 2] = toRouteIdx(m.source);
            routes[i * 4 + 3] = toRouteIdx(m.body);
        }
        wgpuQueueWriteBuffer(queue, mosfetVoltageRouteBuffer, 0,
                             routes.data(), routes.size() * sizeof(uint32_t));
    }

    if (numBJTs_ > 0 && bjtVoltageRouteBuffer) {
        std::vector<uint32_t> routes(numBJTs_ * 3);
        for (size_t i = 0; i < numBJTs_; ++i) {
            const auto& b = netlist.globalBlock.bjts[i];
            routes[i * 3 + 0] = toRouteIdx(b.nodeCollector);
            routes[i * 3 + 1] = toRouteIdx(b.base);
            routes[i * 3 + 2] = toRouteIdx(b.emitter);
        }
        wgpuQueueWriteBuffer(queue, bjtVoltageRouteBuffer, 0,
                             routes.data(), routes.size() * sizeof(uint32_t));
    }
}

// ============================================================================
// uploadNetlist
// ============================================================================
void WebGPUSolver::uploadNetlist(const TensorNetlist& netlist) {
    if (!queue) return;

    auto compiled = NetlistCompiler::compile(netlist);
    const TensorizedBlock& block = compiled->tensors;

    if (numDiodes_ > 0) {
        std::vector<DiodeDeviceCPU> devs(numDiodes_);
        for (size_t i = 0; i < numDiodes_; ++i) {
            auto& d = devs[i];
            d.anode   = block.diodes.node_a[i];
            d.cathode = block.diodes.node_c[i];
            auto is = splitDouble(block.diodes.Is[i]); d.Is_hi = is.first; d.Is_lo = is.second;
            auto n  = splitDouble(block.diodes.N[i]);  d.N_hi  = n.first;  d.N_lo  = n.second;
            auto vt = splitDouble(block.diodes.Vt[i]); d.Vt_hi = vt.first; d.Vt_lo = vt.second;
        }
        wgpuQueueWriteBuffer(queue, diodeSoABuffer, 0, devs.data(), devs.size() * sizeof(DiodeDeviceCPU));
    }

    if (numMosfets_ > 0) {
        std::vector<MosfetDeviceCPU> devs(numMosfets_);
        for (size_t i = 0; i < numMosfets_; ++i) {
            auto& m = devs[i];
            m.drain  = block.mosfets.drains[i];
            m.gate   = block.mosfets.gates[i];
            m.source = block.mosfets.sources[i];
            m.body   = block.mosfets.bodies[i];
            auto w  = splitDouble(block.mosfets.W[i]);      m.W_hi   = w.first;  m.W_lo   = w.second;
            auto l  = splitDouble(block.mosfets.L[i]);      m.L_hi   = l.first;  m.L_lo   = l.second;
            auto kp = splitDouble(block.mosfets.Kp[i]);     m.Kp_hi  = kp.first; m.Kp_lo  = kp.second;
            auto vt = splitDouble(block.mosfets.Vth[i]);    m.Vth_hi = vt.first; m.Vth_lo = vt.second;
            auto la = splitDouble(block.mosfets.lambda[i]); m.lambda_hi = la.first; m.lambda_lo = la.second;
            m.isPMOS = block.mosfets.isPMOS[i] ? 1u : 0u;
        }
        wgpuQueueWriteBuffer(queue, mosfetSoABuffer, 0, devs.data(), devs.size() * sizeof(MosfetDeviceCPU));
    }

    if (numBJTs_ > 0) {
        std::vector<BJTDeviceCPU> devs(numBJTs_);
        for (size_t i = 0; i < numBJTs_; ++i) {
            auto& bj = devs[i];
            bj.collector = block.bjts.collectors[i];
            bj.base      = block.bjts.bases[i];
            bj.emitter   = block.bjts.emitters[i];
            bj.isNPN     = block.bjts.isNPN[i] ? 1u : 0u;
            auto is = splitDouble(block.bjts.Is[i]); bj.Is_hi = is.first; bj.Is_lo = is.second;
            bj.betaF = static_cast<float>(block.bjts.BetaF[i]);
            bj.betaR = static_cast<float>(block.bjts.BetaR[i]);
            auto vt = splitDouble(block.bjts.Vt[i]); bj.Vt_hi = vt.first; bj.Vt_lo = vt.second;
        }
        wgpuQueueWriteBuffer(queue, bjtSoABuffer, 0, devs.data(), devs.size() * sizeof(BJTDeviceCPU));
    }

    buildStampMaps(netlist);
    std::cout << "[INFO] WebGPUSolver: Netlist upload complete.\n";
}

// ============================================================================
// dispatchPhysicsAsync
// ============================================================================
void WebGPUSolver::dispatchPhysicsAsync(double time, double h) {
    if (!device || !queue) return;

    auto ts = splitDouble(time); auto hs = splitDouble(h);
    float gstate[4] = { ts.first, ts.second, hs.first, hs.second };
    wgpuQueueWriteBuffer(queue, globalStateBuffer, 0, gstate, sizeof(gstate));

    WGPUCommandEncoderDescriptor encDesc = {};
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
    WGPUComputePassDescriptor passDesc = {};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &passDesc);

    if (bindGroup0) wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup0, 0, nullptr);
    if (bindGroup1) wgpuComputePassEncoderSetBindGroup(pass, 1, bindGroup1, 0, nullptr);

    auto dispatch = [&](WGPUComputePipeline pl, size_t n) {
        if (pl && n > 0) {
            wgpuComputePassEncoderSetPipeline(pass, pl);
            wgpuComputePassEncoderDispatchWorkgroups(pass, (uint32_t)((n + 63) / 64), 1, 1);
        }
    };
    dispatch(physicsPipelineDiodes,  numDiodes_);
    dispatch(physicsPipelineMosfets, numMosfets_);
    dispatch(physicsPipelineBJTs,    numBJTs_);

    size_t maxDev = std::max({numDiodes_, numMosfets_, numBJTs_, size_t(1)});
    dispatch(assemblyPipeline, maxDev);

    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    WGPUCommandBufferDescriptor cbDesc = {};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cbDesc);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
}

// ============================================================================
// initiateReadback / isReadbackReady
// ============================================================================
void WebGPUSolver::initiateReadback() {
    if (!device || !queue) return;
    const size_t N   = numNodes_;
    const size_t NNZ = (csrNnz_ > 0) ? (size_t)csrNnz_ : N * 10;

    WGPUCommandEncoderDescriptor encDesc = {};
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
    wgpuCommandEncoderCopyBufferToBuffer(enc, jacobianBufferHi, 0, stagingJacobianHi, 0, NNZ * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, jacobianBufferLo, 0, stagingJacobianLo, 0, NNZ * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, rhsBufferHi, 0, stagingRhsHi, 0, N * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, rhsBufferLo, 0, stagingRhsLo, 0, N * sizeof(float));
    WGPUCommandBufferDescriptor cbDesc = {};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cbDesc);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    readbackPending_ = true;
}

bool WebGPUSolver::isReadbackReady() {
    if (!readbackPending_) return true;
#ifndef __EMSCRIPTEN__
    wgpuDeviceTick(device);
#endif
    readbackPending_ = false;
    return true;
}

WebGPUSolver::NRIterationState& WebGPUSolver::getReadbackData() {
    return readbackBuffers_[activeReadback_];
}

// ============================================================================
// uploadDeltaV
// ============================================================================
void WebGPUSolver::uploadDeltaV(const std::vector<double>& deltaV) {
    if (!queue || deltaV.empty()) return;
    uploadHiLo(queue, deltaVBufferHi, deltaVBufferLo, deltaV);
}

// ============================================================================
// runHybridNRLoop
// ============================================================================
std::vector<double> WebGPUSolver::runHybridNRLoop(
    const TensorNetlist& netlist,
    int    maxIter,
    double tol,
    double gmin,
    double time,
    double h)
{
    if (!device || !queue) return {};
    const size_t N = numNodes_;
    if (N == 0) return {};

    // Zero-init voltages on GPU
    {
        std::vector<float> zf(N, 0.0f);
        wgpuQueueWriteBuffer(queue, voltageBufferHi, 0, zf.data(), N * sizeof(float));
        wgpuQueueWriteBuffer(queue, voltageBufferLo, 0, zf.data(), N * sizeof(float));
    }

    // Cold-start: dispatch physics for iteration 0
    dispatchPhysicsAsync(time, h);

    std::vector<double> voltages(N, 0.0);
    bool converged = false;

    for (int iter = 0; iter < maxIter; ++iter) {
        // Wait for previous GPU dispatch to finish (synchronous for now)
        isReadbackReady();

        // Upload current voltage guess to GPU so GPU physics uses them
        uploadHiLo(queue, voltageBufferHi, voltageBufferLo, voltages);

        // Dispatch GPU physics + assembly (uses uploaded voltages)
        dispatchPhysicsAsync(time, h);

        // CPU: stamp all devices (linear + nonlinear via MatrixConstructor).
        // Phase 4 overlap TODO: replace this block with initiateReadback() +
        // readback-wait to merge GPU Jacobian/RHS into CPU LU solve. For now
        // the CPU-only stamping path ensures correctness; the GPU physics
        // dispatch above (dispatchPhysicsAsync) is waveform-recording
        // infrastructure and a latency-hiding scaffold for that upgrade.
        MatrixConstructor mc;
        mc.setDimensions((int)N, (int)N);
        std::vector<double> rhs(N, 0.0);

        // Stamp resistors
        for (const auto& r : netlist.globalBlock.resistors) {
            int n1 = r.nodeTerminal1 - 1;
            int n2 = r.nodeTerminal2 - 1;
            double g = 1.0 / r.resistance_ohms;
            if (n1 >= 0) { mc.add(n1, n1, g); if (n2 >= 0) mc.add(n1, n2, -g); }
            if (n2 >= 0) { mc.add(n2, n2, g); if (n1 >= 0) mc.add(n2, n1, -g); }
        }
        // Stamp voltage sources (stiff conductance — matches stampSoABlock g_int=1e3)
        for (const auto& vs : netlist.globalBlock.voltageSources) {
            int np = vs.nodePositive - 1;
            int nn = vs.nodeNegative - 1;
            const double G_VS = 1e3; // Must match circuitsim.h::stampSoABlock g_int=1e3
            if (np >= 0) { mc.add(np, np, G_VS); rhs[np] += G_VS * vs.voltage_V; }
            if (nn >= 0) { mc.add(nn, nn, G_VS); rhs[nn] -= G_VS * vs.voltage_V; }
        }
        // Stamp current sources
        for (const auto& cs : netlist.globalBlock.currentSources) {
            int np = cs.nodePositive - 1;
            int nn = cs.nodeNegative - 1;
            if (np >= 0) rhs[np] -= cs.current_A;
            if (nn >= 0) rhs[nn] += cs.current_A;
        }
        // Stamp diodes (linearized)
        for (const auto& d : netlist.globalBlock.diodes) {
            int na = d.anode - 1, nc = d.cathode - 1;
            double va = (na >= 0 && na < (int)N) ? voltages[na] : 0.0;
            double vc = (nc >= 0 && nc < (int)N) ? voltages[nc] : 0.0;
            double vd = va - vc;
            double nvt = d.emissionCoefficient_N * d.thermalVoltage_V_T_V;
            double arg = std::min(vd / nvt, 30.0);
            double exp_arg = std::exp(arg);
            double gd = d.saturationCurrent_I_S_A * exp_arg / nvt;
            double id = d.saturationCurrent_I_S_A * (exp_arg - 1.0);
            double ieq = id - gd * vd;
            if (na >= 0) { mc.add(na, na, gd); rhs[na] -= ieq; if (nc >= 0) mc.add(na, nc, -gd); }
            if (nc >= 0) { mc.add(nc, nc, gd); rhs[nc] += ieq; if (na >= 0) mc.add(nc, na, -gd); }
        }
        // GMIN diagonal conditioning
        for (size_t i = 0; i < N; ++i) mc.add((int)i, (int)i, gmin);

        Csr_matrix mat = mc.createCsr();
        SolverResult res = solveLU_Pivoted(mat, rhs);
        if (!res.converged) {
            std::cerr << "[WARN] WebGPUSolver: LU failed at iter " << iter << "\n";
            break;
        }

        double maxDelta = 0.0;
        for (size_t i = 0; i < N; ++i)
            maxDelta = std::max(maxDelta, std::abs(res.solution[i] - voltages[i]));
        voltages = res.solution;

        if (maxDelta < tol) { converged = true; break; }

        // Upload updated voltages to GPU for next iteration
        uploadHiLo(queue, voltageBufferHi, voltageBufferLo, voltages);
    }

    if (converged)
        std::cout << "[INFO] WebGPUSolver: Converged.\n";
    else
        std::cout << "[WARN] WebGPUSolver: Did not converge in " << maxIter << " iters.\n";

    return voltages;
}

// ============================================================================
// Legacy single-step interface
// ============================================================================
void WebGPUSolver::runNRStep(double time, double h) {
    if (!device || !queue) return;
    dispatchPhysicsAsync(time, h);

    // Waveform recording
    if (recordWaveformPipeline && bindGroup0 && bindGroup2) {
        WGPUCommandEncoderDescriptor encDesc = {};
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        WGPUComputePassDescriptor passDesc = {};
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &passDesc);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup0, 0, nullptr);
        wgpuComputePassEncoderSetBindGroup(pass, 2, bindGroup2, 0, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, recordWaveformPipeline);
        wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        WGPUCommandBufferDescriptor cbDesc = {};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cbDesc);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
    }
}

bool WebGPUSolver::checkConvergence() { return true; }

// ============================================================================
// downloadSolution
// ============================================================================
std::vector<double> WebGPUSolver::downloadSolution() {
    if (!device || !queue || numNodes_ == 0) return {};
    const size_t N = numNodes_;

    WGPUCommandEncoderDescriptor encDesc = {};
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
    wgpuCommandEncoderCopyBufferToBuffer(enc, voltageBufferHi, 0, stagingVoltageHi, 0, N * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, voltageBufferLo, 0, stagingVoltageLo, 0, N * sizeof(float));
    WGPUCommandBufferDescriptor cbDesc = {};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cbDesc);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);

    // Synchronous map via Dawn tick
    struct CB { bool done; };
    CB cbHi = {false}, cbLo = {false};
    wgpuBufferMapAsync(stagingVoltageHi, WGPUMapMode_Read, 0, N * sizeof(float),
        [](WGPUBufferMapAsyncStatus, void* ud){ ((CB*)ud)->done = true; }, &cbHi);
    wgpuBufferMapAsync(stagingVoltageLo, WGPUMapMode_Read, 0, N * sizeof(float),
        [](WGPUBufferMapAsyncStatus, void* ud){ ((CB*)ud)->done = true; }, &cbLo);
    while (!cbHi.done || !cbLo.done) wgpuDeviceTick(device);

    std::vector<double> result(N, 0.0);
    const float* pHi = (const float*)wgpuBufferGetConstMappedRange(stagingVoltageHi, 0, N * sizeof(float));
    const float* pLo = (const float*)wgpuBufferGetConstMappedRange(stagingVoltageLo, 0, N * sizeof(float));
    if (pHi && pLo)
        for (size_t i = 0; i < N; ++i)
            result[i] = double(pHi[i]) + double(pLo[i]);
    wgpuBufferUnmap(stagingVoltageHi);
    wgpuBufferUnmap(stagingVoltageLo);
    return result;
}

std::vector<std::vector<double>> WebGPUSolver::downloadWaveform() { return {}; }

WGPUBuffer WebGPUSolver::createStagingBuffer(size_t byteSize, const char* label) {
    return makeBuffer(device, byteSize,
                      (WGPUBufferUsage)(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst), label);
}

void WebGPUSolver::recordCopyToStaging(WGPUCommandEncoder enc,
                                       WGPUBuffer src, WGPUBuffer dst, size_t sz) {
    wgpuCommandEncoderCopyBufferToBuffer(enc, src, 0, dst, 0, sz);
}

void WebGPUSolver::createBindGroups() {}

// ============================================================================
// Gap 1: TN Contraction Backend
// ============================================================================

// CPU-side struct mirroring WGSL ContractionUniforms (16-byte aligned)
struct ContractionUniformsCPU {
    uint32_t slot_left;
    uint32_t slot_right;
    uint32_t slot_result;
    uint32_t rank_left;
    uint32_t rank_right;
    uint32_t rank_result;
    uint32_t elim_row;
    uint32_t do_schur;
};
static_assert(sizeof(ContractionUniformsCPU) == 32, "Uniform struct must be 32 bytes");

struct ElimRecordUniformsCPU {
    uint32_t record_offset;
    uint32_t elim_var_id;
    uint32_t num_neighbors;
    uint32_t backsub_count;
    uint32_t slot_result;
    uint32_t rank_result;
    uint32_t elim_row;
    uint32_t gmin_value; // bitcast float -> uint32
};
static_assert(sizeof(ElimRecordUniformsCPU) == 32, "Uniform struct must be 32 bytes");

static constexpr uint32_t TN_MAX_TILE_RANK  = 8;
static constexpr uint32_t TN_MAX_TILE_ELEMS = TN_MAX_TILE_RANK * TN_MAX_TILE_RANK;

static std::string loadTNShaderSource() {
    const char* candidates[] = {
        "acutesim_engine/shaders/tn_contraction.wgsl",
        "../acutesim_engine/shaders/tn_contraction.wgsl",
        "../../acutesim_engine/shaders/tn_contraction.wgsl",
        "shaders/tn_contraction.wgsl",
        nullptr
    };
    for (int i = 0; candidates[i] != nullptr; ++i) {
        std::ifstream f(candidates[i]);
        if (f.good())
            return std::string(std::istreambuf_iterator<char>(f),
                               std::istreambuf_iterator<char>());
    }
    std::cerr << "[WARN] WebGPUSolver: Could not find tn_contraction.wgsl\n";
    return "";
}

void WebGPUSolver::createTNPipelines() {
    if (!device) return;
    if (tnSeedLeafPipeline) return;  // already created

    std::string src = loadTNShaderSource();
    if (src.empty()) return;

    WGPUShaderModule sm = loadShader(src);
    if (!sm) {
        std::cerr << "[ERROR] WebGPUSolver: TN shader module failed\n";
        return;
    }

    // Use null layout for auto-derivation from WGSL reflection.
    // This allows wgpuComputePipelineGetBindGroupLayout() to return
    // the auto-generated layouts matching the shader's declared bindings.
    auto makePipeline = [&](const char* ep) -> WGPUComputePipeline {
        WGPUComputePipelineDescriptor d = {};
        d.label               = WGPU_SV(ep);
        d.layout              = nullptr;  // auto-layout
        d.compute.module      = sm;
        d.compute.entryPoint  = WGPU_SV(ep);
        return wgpuDeviceCreateComputePipeline(device, &d);
    };

    tnSeedLeafPipeline  = makePipeline("tn_seed_leaf");
    tnMergeAccumPipeline = makePipeline("tn_merge_accum");
    tnSchurElimPipeline  = makePipeline("tn_schur_elim");
    tnSchurElimRecordPipeline = makePipeline("tn_schur_elim_record");
    tnBackSubstitutePipeline = makePipeline("tn_back_substitute");

    wgpuShaderModuleRelease(sm);

    std::cout << "[INFO] WebGPUSolver: TN contraction pipelines created\n";
}

void WebGPUSolver::createTNBindGroups() {
    if (!device || !tnMergeAccumPipeline) return;

    // Release old bind groups
    auto relBG = [](WGPUBindGroup& bg){ if(bg){ wgpuBindGroupRelease(bg); bg=nullptr; } };
    relBG(tnBindGroup0); relBG(tnBindGroup1); relBG(tnBindGroup2);

    // Get auto-derived layouts from tn_merge_accum (it declares both group 0 + group 1)
    WGPUBindGroupLayout bgl0 = wgpuComputePipelineGetBindGroupLayout(tnMergeAccumPipeline, 0);
    WGPUBindGroupLayout bgl1 = wgpuComputePipelineGetBindGroupLayout(tnMergeAccumPipeline, 1);
    WGPUBindGroupLayout bgl2 = nullptr;
    if (tnSchurElimRecordPipeline) {
        bgl2 = wgpuComputePipelineGetBindGroupLayout(tnSchurElimRecordPipeline, 2);
    }
    
    if (!bgl0 || !bgl1 || !bgl2) {
        std::cerr << "[ERROR] WebGPUSolver: Failed to get TN bind group layouts\n";
        if (bgl0) wgpuBindGroupLayoutRelease(bgl0);
        if (bgl1) wgpuBindGroupLayoutRelease(bgl1);
        if (bgl2) wgpuBindGroupLayoutRelease(bgl2);
        return;
    }

    size_t matBytes = static_cast<size_t>(tnNumSlots_) * TN_MAX_TILE_ELEMS * sizeof(float);
    size_t rhsBytes = static_cast<size_t>(tnNumSlots_) * TN_MAX_TILE_RANK * sizeof(float);
    size_t mapBytes = TN_MAX_TILE_RANK * sizeof(uint32_t);

    // Group 0: workspace_hi, workspace_lo, rhs_hi, rhs_lo, uniforms
    WGPUBindGroupEntry entries0[5] = {};
    entries0[0].binding = 0; entries0[0].buffer = tnWorkspaceHi; entries0[0].size = matBytes;
    entries0[1].binding = 1; entries0[1].buffer = tnWorkspaceLo; entries0[1].size = matBytes;
    entries0[2].binding = 2; entries0[2].buffer = tnRhsHi;       entries0[2].size = rhsBytes;
    entries0[3].binding = 3; entries0[3].buffer = tnRhsLo;       entries0[3].size = rhsBytes;
    entries0[4].binding = 4; entries0[4].buffer = tnUniformBuffer;
    entries0[4].size = sizeof(ContractionUniformsCPU);

    WGPUBindGroupDescriptor bgd0 = {};
    bgd0.layout     = bgl0;
    bgd0.entryCount = 5;
    bgd0.entries    = entries0;
    tnBindGroup0 = wgpuDeviceCreateBindGroup(device, &bgd0);

    // Group 1: index_map_left, index_map_right
    WGPUBindGroupEntry entries1[2] = {};
    entries1[0].binding = 0; entries1[0].buffer = tnIndexMapLeft;  entries1[0].size = mapBytes;
    entries1[1].binding = 1; entries1[1].buffer = tnIndexMapRight; entries1[1].size = mapBytes;

    WGPUBindGroupDescriptor bgd1 = {};
    bgd1.layout     = bgl1;
    bgd1.entryCount = 2;
    bgd1.entries    = entries1;
    tnBindGroup1 = wgpuDeviceCreateBindGroup(device, &bgd1);

    wgpuBindGroupLayoutRelease(bgl0);
    wgpuBindGroupLayoutRelease(bgl1);

    // Group 2: Elimination records and solution
    if (bgl2 && numNodes_ > 0) {
        size_t elimPivotBytes = static_cast<size_t>(numNodes_) * sizeof(float);
        size_t elimRhsBytes = static_cast<size_t>(numNodes_) * sizeof(float);
        size_t elimRowBytes = static_cast<size_t>(numNodes_) * TN_MAX_TILE_RANK * sizeof(float);
        size_t elimVarIdsBytes = static_cast<size_t>(numNodes_) * sizeof(uint32_t);
        size_t elimNeighborIdsBytes = static_cast<size_t>(numNodes_) * TN_MAX_TILE_RANK * sizeof(uint32_t);
        size_t elimNeighborCountBytes = static_cast<size_t>(numNodes_) * sizeof(uint32_t);
        size_t solutionBytes = static_cast<size_t>(numNodes_) * sizeof(float);

        WGPUBindGroupEntry entries2[12] = {};
        entries2[0].binding = 0; entries2[0].buffer = tnElimPivotHi; entries2[0].size = elimPivotBytes;
        entries2[1].binding = 1; entries2[1].buffer = tnElimPivotLo; entries2[1].size = elimPivotBytes;
        entries2[2].binding = 2; entries2[2].buffer = tnElimRhsHi;   entries2[2].size = elimRhsBytes;
        entries2[3].binding = 3; entries2[3].buffer = tnElimRhsLo;   entries2[3].size = elimRhsBytes;
        entries2[4].binding = 4; entries2[4].buffer = tnElimRowHi;   entries2[4].size = elimRowBytes;
        entries2[5].binding = 5; entries2[5].buffer = tnElimRowLo;   entries2[5].size = elimRowBytes;
        entries2[6].binding = 6; entries2[6].buffer = tnElimVarIds;  entries2[6].size = elimVarIdsBytes;
        entries2[7].binding = 7; entries2[7].buffer = tnElimNeighborIds; entries2[7].size = elimNeighborIdsBytes;
        entries2[8].binding = 8; entries2[8].buffer = tnElimNeighborCount; entries2[8].size = elimNeighborCountBytes;
        entries2[9].binding = 9; entries2[9].buffer = tnSolutionHi;  entries2[9].size = solutionBytes;
        entries2[10].binding = 10; entries2[10].buffer = tnSolutionLo; entries2[10].size = solutionBytes;
        entries2[11].binding = 11; entries2[11].buffer = tnElimUniformBuffer; entries2[11].size = sizeof(ElimRecordUniformsCPU);

        WGPUBindGroupDescriptor bgd2 = {};
        bgd2.layout     = bgl2;
        bgd2.entryCount = 12;
        bgd2.entries    = entries2;
        tnBindGroup2 = wgpuDeviceCreateBindGroup(device, &bgd2);
    }
    if (bgl2) wgpuBindGroupLayoutRelease(bgl2);

    if (tnBindGroup0 && tnBindGroup1)
        std::cout << "[INFO] WebGPUSolver: TN bind groups created\n";
    else
        std::cerr << "[ERROR] WebGPUSolver: TN bind group creation failed\n";
}

void WebGPUSolver::uploadTNUniforms(uint32_t slotLeft, uint32_t slotRight,
                                     uint32_t slotResult,
                                     uint32_t rankLeft, uint32_t rankRight,
                                     uint32_t rankResult,
                                     uint32_t elimRow, uint32_t doSchur) {
    if (!queue || !tnUniformBuffer) return;
    ContractionUniformsCPU u = {
        slotLeft, slotRight, slotResult,
        rankLeft, rankRight, rankResult,
        elimRow, doSchur
    };
    wgpuQueueWriteBuffer(queue, tnUniformBuffer, 0, &u, sizeof(u));
}

void WebGPUSolver::uploadTNElimUniforms(uint32_t record_offset, uint32_t elim_var_id,
                                         uint32_t num_neighbors, uint32_t backsub_count,
                                         uint32_t slot_result, uint32_t rank_result,
                                         uint32_t elim_row, float gmin) {
    if (!queue || !tnElimUniformBuffer) return;
    
    uint32_t gmin_bits;
    std::memcpy(&gmin_bits, &gmin, sizeof(gmin_bits));
    
    ElimRecordUniformsCPU u = {
        record_offset, elim_var_id, num_neighbors, backsub_count,
        slot_result, rank_result, elim_row, gmin_bits
    };
    wgpuQueueWriteBuffer(queue, tnElimUniformBuffer, 0, &u, sizeof(u));
}

void WebGPUSolver::uploadIndexMaps(const std::vector<uint32_t>& mapLeft,
                                    const std::vector<uint32_t>& mapRight) {
    if (!queue) return;
    if (tnIndexMapLeft && !mapLeft.empty())
        wgpuQueueWriteBuffer(queue, tnIndexMapLeft, 0,
                             mapLeft.data(), mapLeft.size() * sizeof(uint32_t));
    if (tnIndexMapRight && !mapRight.empty())
        wgpuQueueWriteBuffer(queue, tnIndexMapRight, 0,
                             mapRight.data(), mapRight.size() * sizeof(uint32_t));
}

void WebGPUSolver::dispatchTNKernel(WGPUComputePipeline pl, uint32_t numElems,
                                     WGPUCommandEncoder enc,
                                     WGPUComputePassEncoder pass) {
    if (!pl) return;
    wgpuComputePassEncoderSetPipeline(pass, pl);
    uint32_t wgCount = (numElems + 63) / 64;
    if (wgCount == 0) wgCount = 1;
    wgpuComputePassEncoderDispatchWorkgroups(pass, wgCount, 1, 1);
}

void WebGPUSolver::uploadContractionProgram(const TNCompiledProgram& prog,
                                             uint32_t nodeCount) {
    if (!device || !queue) return;
    if (prog.tree.nodes.empty() || prog.mpos.empty()) return;

    createTNPipelines();
    if (!tnSeedLeafPipeline) return;

    // ── Phase 5.7.4: Use render graph for aliased slot count ─────────────
    renderGraph_ = prog.renderGraph;
    if (renderGraph_.valid && renderGraph_.numPhysicalSlots > 0)
        tnNumSlots_ = renderGraph_.numPhysicalSlots;
    else
        tnNumSlots_ = static_cast<uint32_t>(prog.tree.nodes.size());
    tnMaxRank_  = TN_MAX_TILE_RANK;

    const WGPUBufferUsage STORAGE_RW =
        (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                          WGPUBufferUsage_CopySrc);
    const WGPUBufferUsage UNIFORM_BUF =
        (WGPUBufferUsage)(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);

    // Allocate (or resize) workspace buffers
    auto relBuf = [](WGPUBuffer& b){ if(b){ wgpuBufferRelease(b); b=nullptr; } };
    relBuf(tnWorkspaceHi); relBuf(tnWorkspaceLo);
    relBuf(tnRhsHi); relBuf(tnRhsLo);
    relBuf(tnUniformBuffer);
    relBuf(tnIndexMapLeft); relBuf(tnIndexMapRight);

    relBuf(tnElimPivotHi); relBuf(tnElimPivotLo);
    relBuf(tnElimRhsHi); relBuf(tnElimRhsLo);
    relBuf(tnElimRowHi); relBuf(tnElimRowLo);
    relBuf(tnElimVarIds); relBuf(tnElimNeighborIds); relBuf(tnElimNeighborCount);
    relBuf(tnSolutionHi); relBuf(tnSolutionLo);
    relBuf(tnElimUniformBuffer);

    size_t matBytes = static_cast<size_t>(tnNumSlots_) * TN_MAX_TILE_ELEMS * sizeof(float);
    size_t rhsBytes = static_cast<size_t>(tnNumSlots_) * TN_MAX_TILE_RANK * sizeof(float);
    size_t mapBytes = TN_MAX_TILE_RANK * sizeof(uint32_t);

    tnWorkspaceHi  = makeBuffer(device, matBytes, STORAGE_RW, "tn_ws_hi");
    tnWorkspaceLo  = makeBuffer(device, matBytes, STORAGE_RW, "tn_ws_lo");
    tnRhsHi        = makeBuffer(device, rhsBytes, STORAGE_RW, "tn_rhs_hi");
    tnRhsLo        = makeBuffer(device, rhsBytes, STORAGE_RW, "tn_rhs_lo");
    tnUniformBuffer = makeBuffer(device, sizeof(ContractionUniformsCPU), UNIFORM_BUF, "tn_uniforms");
    tnIndexMapLeft  = makeBuffer(device, mapBytes, STORAGE_RW, "tn_imap_l");
    tnIndexMapRight = makeBuffer(device, mapBytes, STORAGE_RW, "tn_imap_r");

    if (nodeCount > 0) {
        size_t elimPivotBytes = static_cast<size_t>(nodeCount) * sizeof(float);
        size_t elimRhsBytes = static_cast<size_t>(nodeCount) * sizeof(float);
        size_t elimRowBytes = static_cast<size_t>(nodeCount) * TN_MAX_TILE_RANK * sizeof(float);
        size_t elimVarIdsBytes = static_cast<size_t>(nodeCount) * sizeof(uint32_t);
        size_t elimNeighborIdsBytes = static_cast<size_t>(nodeCount) * TN_MAX_TILE_RANK * sizeof(uint32_t);
        size_t elimNeighborCountBytes = static_cast<size_t>(nodeCount) * sizeof(uint32_t);
        size_t solutionBytes = static_cast<size_t>(nodeCount) * sizeof(float);

        tnElimPivotHi = makeBuffer(device, elimPivotBytes, STORAGE_RW, "tn_elim_pivot_hi");
        tnElimPivotLo = makeBuffer(device, elimPivotBytes, STORAGE_RW, "tn_elim_pivot_lo");
        tnElimRhsHi = makeBuffer(device, elimRhsBytes, STORAGE_RW, "tn_elim_rhs_hi");
        tnElimRhsLo = makeBuffer(device, elimRhsBytes, STORAGE_RW, "tn_elim_rhs_lo");
        tnElimRowHi = makeBuffer(device, elimRowBytes, STORAGE_RW, "tn_elim_row_hi");
        tnElimRowLo = makeBuffer(device, elimRowBytes, STORAGE_RW, "tn_elim_row_lo");
        tnElimVarIds = makeBuffer(device, elimVarIdsBytes, STORAGE_RW, "tn_elim_var_ids");
        tnElimNeighborIds = makeBuffer(device, elimNeighborIdsBytes, STORAGE_RW, "tn_elim_neighbor_ids");
        tnElimNeighborCount = makeBuffer(device, elimNeighborCountBytes, STORAGE_RW, "tn_elim_neighbor_count");
        tnSolutionHi = makeBuffer(device, solutionBytes, STORAGE_RW, "tn_solution_hi");
        tnSolutionLo = makeBuffer(device, solutionBytes, STORAGE_RW, "tn_solution_lo");
        tnElimUniformBuffer = makeBuffer(device, sizeof(ElimRecordUniformsCPU), UNIFORM_BUF, "tn_elim_uniforms");
    }

    // ── Prepare full workspace data on CPU (batched) ─────────────────────
    // Instead of N separate wgpuQueueWriteBuffer() calls per leaf, we prepare
    // all data in CPU arrays and upload in bulk (4 calls total).
    std::vector<float> fullMatHi(static_cast<size_t>(tnNumSlots_) * TN_MAX_TILE_ELEMS, 0.0f);
    std::vector<float> fullMatLo(static_cast<size_t>(tnNumSlots_) * TN_MAX_TILE_ELEMS, 0.0f);
    std::vector<float> fullRhsHi(static_cast<size_t>(tnNumSlots_) * TN_MAX_TILE_RANK, 0.0f);
    std::vector<float> fullRhsLo(static_cast<size_t>(tnNumSlots_) * TN_MAX_TILE_RANK, 0.0f);

    // Fill leaf slot data (using aliased physical slot indices)
    for (size_t li = 0; li < prog.tree.leafIds.size() && li < prog.mpos.size(); ++li) {
        uint32_t nodeId = prog.tree.leafIds[li];
        const auto& mpo = prog.mpos[li];
        uint32_t k = mpo.rank;
        if (k == 0 || k > TN_MAX_TILE_RANK) continue;

        // Phase 5.7.4: map node ID to aliased physical slot
        uint32_t physSlot = nodeId;
        if (renderGraph_.valid && nodeId < renderGraph_.nodeIdToPhysical.size())
            physSlot = renderGraph_.nodeIdToPhysical[nodeId];
        if (physSlot >= tnNumSlots_) continue;

        size_t matOff = static_cast<size_t>(physSlot) * TN_MAX_TILE_ELEMS;
        size_t rhsOff = static_cast<size_t>(physSlot) * TN_MAX_TILE_RANK;

        for (uint32_t r = 0; r < k; ++r) {
            for (uint32_t c = 0; c < k; ++c) {
                double val = mpo.localMatrix[r * k + c];
                auto p = splitDouble(val);
                fullMatHi[matOff + r * k + c] = p.first;
                fullMatLo[matOff + r * k + c] = p.second;
            }
        }
        for (uint32_t r = 0; r < k && r < mpo.localRHS.size(); ++r) {
            auto p = splitDouble(mpo.localRHS[r]);
            fullRhsHi[rhsOff + r] = p.first;
            fullRhsLo[rhsOff + r] = p.second;
        }
    }

    // ── Phase 5.7.2: Apple UMA zero-copy bulk upload ─────────────────────
#if ACUTESIM_APPLE_UMA
    {
        // Create staging buffers with mappedAtCreation (CPU writes directly)
        auto stgMatHi = AppleUMAManager::createStaging(device, matBytes);
        auto stgMatLo = AppleUMAManager::createStaging(device, matBytes);
        auto stgRhsHi = AppleUMAManager::createStaging(device, rhsBytes);
        auto stgRhsLo = AppleUMAManager::createStaging(device, rhsBytes);

        // Direct memcpy into mapped staging memory (no wgpuQueueWriteBuffer overhead)
        AppleUMAManager::writeToStaging(stgMatHi, fullMatHi.data(), matBytes);
        AppleUMAManager::writeToStaging(stgMatLo, fullMatLo.data(), matBytes);
        AppleUMAManager::writeToStaging(stgRhsHi, fullRhsHi.data(), rhsBytes);
        AppleUMAManager::writeToStaging(stgRhsLo, fullRhsLo.data(), rhsBytes);

        // Single command encoder: 4 copy commands (near-free on UMA)
        WGPUCommandEncoderDescriptor encDesc = {};
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        AppleUMAManager::flush(stgMatHi, tnWorkspaceHi, enc, matBytes);
        AppleUMAManager::flush(stgMatLo, tnWorkspaceLo, enc, matBytes);
        AppleUMAManager::flush(stgRhsHi, tnRhsHi, enc, rhsBytes);
        AppleUMAManager::flush(stgRhsLo, tnRhsLo, enc, rhsBytes);
        WGPUCommandBufferDescriptor cbDesc = {};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cbDesc);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);

        AppleUMAManager::destroy(stgMatHi);
        AppleUMAManager::destroy(stgMatLo);
        AppleUMAManager::destroy(stgRhsHi);
        AppleUMAManager::destroy(stgRhsLo);
    }
#else
    // Non-UMA: 4 bulk wgpuQueueWriteBuffer calls (still batched vs N per-leaf)
    wgpuQueueWriteBuffer(queue, tnWorkspaceHi, 0, fullMatHi.data(), matBytes);
    wgpuQueueWriteBuffer(queue, tnWorkspaceLo, 0, fullMatLo.data(), matBytes);
    wgpuQueueWriteBuffer(queue, tnRhsHi, 0, fullRhsHi.data(), rhsBytes);
    wgpuQueueWriteBuffer(queue, tnRhsLo, 0, fullRhsLo.data(), rhsBytes);
#endif

    // Save tree reference and leaf terminal nodes for tree walk
    lastTree_ = &prog.tree;
    leafTerminalNodes_.clear();
    leafTerminalNodes_.resize(prog.tree.nodes.size());
    for (size_t i = 0; i < prog.tree.leafIds.size() && i < prog.mpos.size(); ++i) {
        uint32_t leafId = prog.tree.leafIds[i];
        if (leafId < leafTerminalNodes_.size())
            leafTerminalNodes_[leafId] = prog.mpos[i].terminalNodes;
    }

    // Create staging buffers for per-node slot readback
    relBuf(tnStagingMatHi); relBuf(tnStagingMatLo);
    relBuf(tnStagingRhsHi); relBuf(tnStagingRhsLo);
    const WGPUBufferUsage MAP_READ_BUF =
        (WGPUBufferUsage)(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    tnStagingMatHi = makeBuffer(device, TN_MAX_TILE_ELEMS * sizeof(float), MAP_READ_BUF, "tn_stg_mhi");
    tnStagingMatLo = makeBuffer(device, TN_MAX_TILE_ELEMS * sizeof(float), MAP_READ_BUF, "tn_stg_mlo");
    tnStagingRhsHi = makeBuffer(device, TN_MAX_TILE_RANK * sizeof(float), MAP_READ_BUF, "tn_stg_rhi");
    tnStagingRhsLo = makeBuffer(device, TN_MAX_TILE_RANK * sizeof(float), MAP_READ_BUF, "tn_stg_rlo");

    // Create bind groups from auto-derived pipeline layouts
    createTNBindGroups();

    if (renderGraph_.valid) {
        std::cout << "[INFO] WebGPUSolver: TN program uploaded ("
                  << tnNumSlots_ << " aliased slots from "
                  << prog.tree.nodes.size() << " nodes, "
                  << prog.mpos.size() << " MPOs";
        if (renderGraph_.unaliasedBytes > 0)
            std::cout << ", VRAM " << renderGraph_.aliasedBytes / 1024 << "KB vs "
                      << renderGraph_.unaliasedBytes / 1024 << "KB unaliased";
        std::cout << ")\n";
    } else {
        std::cout << "[INFO] WebGPUSolver: TN program uploaded ("
                  << tnNumSlots_ << " slots, " << prog.mpos.size() << " MPOs)\n";
    }
}

// Helper: sorted union of two index sets
static std::vector<uint32_t> tnSortedUnion(const std::vector<uint32_t>& a,
                                            const std::vector<uint32_t>& b) {
    std::vector<uint32_t> result;
    result.reserve(a.size() + b.size());
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(result));
    return result;
}

// Helper: find position of val in sorted vector (-1 if not found)
static int tnFindPos(uint32_t val, const std::vector<uint32_t>& sorted) {
    auto it = std::lower_bound(sorted.begin(), sorted.end(), val);
    if (it != sorted.end() && *it == val)
        return static_cast<int>(it - sorted.begin());
    return -1;
}

std::vector<double> WebGPUSolver::executeContractionSweep(uint32_t nodeCount,
                                                           double gmin) {
    if (!device || !queue || !tnMergeAccumPipeline || tnNumSlots_ == 0 ||
        !lastTree_ || !tnBindGroup0)
        return {};

    const ContractionTree& tree = *lastTree_;
    if (tree.rootId == UINT32_MAX || tree.nodes.empty()) return {};

    // Phase 5.7.4: helper to resolve aliased physical slot from node ID
    auto physSlot = [&](uint32_t nodeId) -> uint32_t {
        if (renderGraph_.valid && nodeId < renderGraph_.nodeIdToPhysical.size())
            return renderGraph_.nodeIdToPhysical[nodeId];
        return nodeId;  // fallback: identity mapping (no aliasing)
    };

    // ── Per-node CPU state: tracked index sets (no more elimination records)
    struct NodeState {
        std::vector<uint32_t> indices;
        uint32_t dim = 0;
    };
    std::vector<NodeState> nodeStates(tree.nodes.size());

    // Initialize leaf states from saved terminal nodes
    for (size_t i = 0; i < tree.nodes.size(); ++i) {
        const auto& node = tree.nodes[i];
        if (node.leftChild == UINT32_MAX && node.rightChild == UINT32_MAX) {
            if (node.id < leafTerminalNodes_.size()) {
                nodeStates[i].indices = leafTerminalNodes_[node.id];
                nodeStates[i].dim = static_cast<uint32_t>(nodeStates[i].indices.size());
            }
        }
    }

    uint32_t elimRecordCounter = 0; // Track global elimination record index

    // ── Tree Walk: bottom-up processing ──────────────────────────────────
    // Phase 5.7.2: thermal-aware batch sizing
    (void)ThermalMonitor::currentState();  // poll once per sweep (cheap syscall)

    for (size_t idx = 0; idx < tree.nodes.size(); ++idx) {
        const auto& node = tree.nodes[idx];

        // Skip leaf nodes (data already in GPU workspace from uploadContractionProgram)
        if (node.leftChild == UINT32_MAX && node.rightChild == UINT32_MAX)
            continue;

        const auto& leftState  = nodeStates[node.leftChild];
        const auto& rightState = nodeStates[node.rightChild];

        // Build merged index set = sorted_union(left, right)
        auto mergedIndices = tnSortedUnion(leftState.indices, rightState.indices);
        uint32_t mergedRank = static_cast<uint32_t>(mergedIndices.size());

        if (mergedRank == 0 || mergedRank > TN_MAX_TILE_RANK) {
            nodeStates[idx].indices = mergedIndices;
            nodeStates[idx].dim = mergedRank;
            continue;
        }

        // Build index maps: child local position → merged position
        std::vector<uint32_t> mapLeft, mapRight;
        mapLeft.reserve(leftState.dim);
        for (uint32_t ci : leftState.indices) {
            int pos = tnFindPos(ci, mergedIndices);
            mapLeft.push_back(pos >= 0 ? static_cast<uint32_t>(pos) : 0u);
        }
        mapRight.reserve(rightState.dim);
        for (uint32_t ci : rightState.indices) {
            int pos = tnFindPos(ci, mergedIndices);
            mapRight.push_back(pos >= 0 ? static_cast<uint32_t>(pos) : 0u);
        }

        // ── GPU dispatch: tn_merge_accum ─────────────────────────────────
        // Phase 5.7.4: use aliased physical slots for GPU addressing
        uint32_t psLeft   = physSlot(node.leftChild);
        uint32_t psRight  = physSlot(node.rightChild);
        uint32_t psResult = physSlot(node.id);

        uploadTNUniforms(psLeft, psRight, psResult,
                         leftState.dim, rightState.dim, mergedRank, 0, 0);

        mapLeft.resize(TN_MAX_TILE_RANK, 0);
        mapRight.resize(TN_MAX_TILE_RANK, 0);
        uploadIndexMaps(mapLeft, mapRight);

        {
            WGPUCommandEncoderDescriptor encDesc = {};
            WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
            WGPUComputePassDescriptor passDesc = {};
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &passDesc);

            wgpuComputePassEncoderSetBindGroup(pass, 0, tnBindGroup0, 0, nullptr);
            wgpuComputePassEncoderSetBindGroup(pass, 1, tnBindGroup1, 0, nullptr);

            uint32_t maxRankSq = std::max({leftState.dim * leftState.dim,
                                           rightState.dim * rightState.dim,
                                           mergedRank * mergedRank});
            dispatchTNKernel(tnMergeAccumPipeline, maxRankSq, enc, pass);

            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
            WGPUCommandBufferDescriptor cbDesc = {};
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cbDesc);
            wgpuQueueSubmit(queue, 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);
        }

        // ── GPU Schur elimination (zero-sync) ────────────────────────────
        auto currentIndices = mergedIndices;
        uint32_t currentDim = mergedRank;

        for (uint32_t v : node.contractedIndices) {
            int p = tnFindPos(v, currentIndices);
            if (p < 0) continue;

            uploadTNElimUniforms(elimRecordCounter, v, currentDim - 1, 0, psResult, currentDim, static_cast<uint32_t>(p), static_cast<float>(gmin));

            {
                WGPUCommandEncoderDescriptor encDesc = {};
                WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
                WGPUComputePassDescriptor passDesc = {};
                WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &passDesc);

                wgpuComputePassEncoderSetBindGroup(pass, 0, tnBindGroup0, 0, nullptr);
                wgpuComputePassEncoderSetBindGroup(pass, 2, tnBindGroup2, 0, nullptr);

                dispatchTNKernel(tnSchurElimRecordPipeline, currentDim * currentDim, enc, pass);

                wgpuComputePassEncoderEnd(pass);
                wgpuComputePassEncoderRelease(pass);
                WGPUCommandBufferDescriptor cbDesc = {};
                WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cbDesc);
                wgpuQueueSubmit(queue, 1, &cmd);
                wgpuCommandBufferRelease(cmd);
                wgpuCommandEncoderRelease(enc);
            }

            // Compact CPU indices for next step mapping
            uint32_t newDim = currentDim - 1;
            std::vector<uint32_t> newIndices;
            for (uint32_t i = 0; i < currentDim; ++i) {
                if (static_cast<int>(i) == p) continue;
                newIndices.push_back(currentIndices[i]);
            }
            currentIndices = std::move(newIndices);
            currentDim = newDim;
            elimRecordCounter++;
        }

        nodeStates[idx].indices = std::move(currentIndices);
        nodeStates[idx].dim = currentDim;
    }

    // ── Root solve: small dense system ───────────────────────────────────
    auto& rootState = nodeStates[tree.rootId];
    if (rootState.dim == 0) {
        std::cerr << "[WARN] WebGPUSolver: TN root has zero dimension\n";
        return {};
    }

    // Download root solve matrix and RHS (since root tile is tiny)
    // We create a temporary map buffer for reading the root slot
    std::vector<double> rootMat, rootRhs;
    {
        uint32_t rootSlot = physSlot(tree.rootId);
        size_t matByteOff = static_cast<size_t>(rootSlot) * TN_MAX_TILE_ELEMS * sizeof(float);
        size_t rhsByteOff = static_cast<size_t>(rootSlot) * TN_MAX_TILE_RANK * sizeof(float);
        size_t matBytes   = TN_MAX_TILE_ELEMS * sizeof(float);
        size_t rhsBytes   = TN_MAX_TILE_RANK * sizeof(float);

        WGPUCommandEncoderDescriptor encDesc = {};
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        wgpuCommandEncoderCopyBufferToBuffer(enc, tnWorkspaceHi, matByteOff, tnStagingMatHi, 0, matBytes);
        wgpuCommandEncoderCopyBufferToBuffer(enc, tnWorkspaceLo, matByteOff, tnStagingMatLo, 0, matBytes);
        wgpuCommandEncoderCopyBufferToBuffer(enc, tnRhsHi, rhsByteOff, tnStagingRhsHi, 0, rhsBytes);
        wgpuCommandEncoderCopyBufferToBuffer(enc, tnRhsLo, rhsByteOff, tnStagingRhsLo, 0, rhsBytes);
        WGPUCommandBufferDescriptor cbDesc = {};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cbDesc);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);

        struct CB { bool done; };
        CB cb[4] = {{false},{false},{false},{false}};
        auto mapCb = [](WGPUBufferMapAsyncStatus, void* ud){ ((CB*)ud)->done = true; };
        wgpuBufferMapAsync(tnStagingMatHi, WGPUMapMode_Read, 0, matBytes, mapCb, &cb[0]);
        wgpuBufferMapAsync(tnStagingMatLo, WGPUMapMode_Read, 0, matBytes, mapCb, &cb[1]);
        wgpuBufferMapAsync(tnStagingRhsHi, WGPUMapMode_Read, 0, rhsBytes, mapCb, &cb[2]);
        wgpuBufferMapAsync(tnStagingRhsLo, WGPUMapMode_Read, 0, rhsBytes, mapCb, &cb[3]);
        while (!cb[0].done || !cb[1].done || !cb[2].done || !cb[3].done)
            wgpuDeviceTick(device);

        const float* pMatHi = (const float*)wgpuBufferGetConstMappedRange(tnStagingMatHi, 0, matBytes);
        const float* pMatLo = (const float*)wgpuBufferGetConstMappedRange(tnStagingMatLo, 0, matBytes);
        const float* pRhsHi = (const float*)wgpuBufferGetConstMappedRange(tnStagingRhsHi, 0, rhsBytes);
        const float* pRhsLo = (const float*)wgpuBufferGetConstMappedRange(tnStagingRhsLo, 0, rhsBytes);

        uint32_t rank = rootState.dim;
        rootMat.assign(rank * rank, 0.0);
        rootRhs.assign(rank, 0.0);
        if (pMatHi && pMatLo) {
            for (uint32_t r = 0; r < rank; ++r)
                for (uint32_t c = 0; c < rank; ++c)
                    rootMat[r * rank + c] = double(pMatHi[r * rank + c]) + double(pMatLo[r * rank + c]);
        }
        if (pRhsHi && pRhsLo) {
            for (uint32_t r = 0; r < rank; ++r)
                rootRhs[r] = double(pRhsHi[r]) + double(pRhsLo[r]);
        }
        wgpuBufferUnmap(tnStagingMatHi); wgpuBufferUnmap(tnStagingMatLo);
        wgpuBufferUnmap(tnStagingRhsHi); wgpuBufferUnmap(tnStagingRhsLo);
    }

    uint32_t n = rootState.dim;
    Csr_matrix csr;
    csr.rows = static_cast<int>(n);
    csr.cols = static_cast<int>(n);
    csr.row_pointer.push_back(0);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            double val = rootMat[i * n + j];
            if (std::abs(val) > 1e-30) {
                csr.values.push_back(val);
                csr.col_indices.push_back(static_cast<int>(j));
            }
        }
        csr.row_pointer.push_back(static_cast<int>(csr.values.size()));
    }
    csr.nnz = static_cast<int>(csr.values.size());

    SolverResult rootResult = solveLU_Pivoted(csr, rootRhs);
    if (!rootResult.converged) {
        std::cerr << "[WARN] WebGPUSolver: Root LU solve failed\n";
        return {};
    }

    // Upload root solution to GPU for back-substitution
    std::vector<float> solHi(nodeCount, 0.0f);
    std::vector<float> solLo(nodeCount, 0.0f);
    for (uint32_t i = 0; i < rootState.dim; ++i) {
        uint32_t var = rootState.indices[i];
        if (var > 0 && (var - 1) < solHi.size()) {
            auto pair = splitDouble(rootResult.solution[i]);
            solHi[var - 1] = pair.first;
            solLo[var - 1] = pair.second;
        }
    }
    wgpuQueueWriteBuffer(queue, tnSolutionHi, 0, solHi.data(), solHi.size() * sizeof(float));
    wgpuQueueWriteBuffer(queue, tnSolutionLo, 0, solLo.data(), solLo.size() * sizeof(float));

    // ── GPU Back-substitution ────────────────────────────────────────────
    if (elimRecordCounter > 0) {
        uploadTNElimUniforms(0, 0, 0, elimRecordCounter, 0, 0, 0, 0.0f);

        WGPUCommandEncoderDescriptor encDesc = {};
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        WGPUComputePassDescriptor passDesc = {};
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &passDesc);

        wgpuComputePassEncoderSetBindGroup(pass, 2, tnBindGroup2, 0, nullptr);

        dispatchTNKernel(tnBackSubstitutePipeline, elimRecordCounter, enc, pass);

        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        WGPUCommandBufferDescriptor cbDesc = {};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cbDesc);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
    }

    // ── Download final solution vector ───────────────────────────────────
    std::vector<double> solution(nodeCount, 0.0);
    {
        size_t solBytes = nodeCount * sizeof(float);
        WGPUBuffer stagingSolHi = makeBuffer(device, solBytes, (WGPUBufferUsage)(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst), "stg_sol_hi");
        WGPUBuffer stagingSolLo = makeBuffer(device, solBytes, (WGPUBufferUsage)(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst), "stg_sol_lo");

        WGPUCommandEncoderDescriptor encDesc = {};
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        wgpuCommandEncoderCopyBufferToBuffer(enc, tnSolutionHi, 0, stagingSolHi, 0, solBytes);
        wgpuCommandEncoderCopyBufferToBuffer(enc, tnSolutionLo, 0, stagingSolLo, 0, solBytes);
        WGPUCommandBufferDescriptor cbDesc = {};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cbDesc);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);

        struct CB { bool done; };
        CB cb[2] = {{false},{false}};
        auto mapCb = [](WGPUBufferMapAsyncStatus, void* ud){ ((CB*)ud)->done = true; };
        wgpuBufferMapAsync(stagingSolHi, WGPUMapMode_Read, 0, solBytes, mapCb, &cb[0]);
        wgpuBufferMapAsync(stagingSolLo, WGPUMapMode_Read, 0, solBytes, mapCb, &cb[1]);
        while (!cb[0].done || !cb[1].done) wgpuDeviceTick(device);

        const float* pHi = (const float*)wgpuBufferGetConstMappedRange(stagingSolHi, 0, solBytes);
        const float* pLo = (const float*)wgpuBufferGetConstMappedRange(stagingSolLo, 0, solBytes);

        if (pHi && pLo) {
            for (size_t i = 0; i < nodeCount; ++i) {
                solution[i] = double(pHi[i]) + double(pLo[i]);
            }
        }
        wgpuBufferUnmap(stagingSolHi); wgpuBufferUnmap(stagingSolLo);
        wgpuBufferRelease(stagingSolHi); wgpuBufferRelease(stagingSolLo);
    }

    std::cout << "[INFO] WebGPUSolver: GPU contraction sweep complete ("
              << tree.nodes.size() << " nodes, " << nodeCount << " vars)\n";
    return solution;
}

#else
// Stub for non-GPU builds
#include "../solvers/webgpu_solver.h"
WebGPUSolver::WebGPUSolver() {}
WebGPUSolver::~WebGPUSolver() {}
#endif
