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

    auto relBG = [](WGPUBindGroup& bg){ if(bg){ wgpuBindGroupRelease(bg); bg=nullptr; } };
    relBG(bindGroup0); relBG(bindGroup1); relBG(bindGroup2);

    auto relPL = [](WGPUComputePipeline& pl){ if(pl){ wgpuComputePipelineRelease(pl); pl=nullptr; } };
    relPL(physicsPipelineDiodes);  relPL(physicsPipelineMosfets); relPL(physicsPipelineBJTs);
    relPL(assemblyPipeline);       relPL(solutionUpdatePipeline);
    relPL(residualPipeline);       relPL(convergencePipeline);
    relPL(recordWaveformPipeline);

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

#else
// Stub for non-GPU builds
#include "../solvers/webgpu_solver.h"
WebGPUSolver::WebGPUSolver() {}
WebGPUSolver::~WebGPUSolver() {}
#endif
