#include "gpu_front_lu.h"

#ifdef ACUTESIM_GPU_ENABLED
#include "../gpu_context_manager.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

using acutesim::GPUContextManager;

namespace {

// One elimination step k = four dispatches; P.k arrives through a dynamic
// uniform offset, so a whole front is one command buffer with no CPU sync.
// piv[0..m): pivot row of each step; piv[m]: 0, or failing step + 1 (later
// steps then do nothing and the host falls back to the CPU).
const char *kWgsl = R"(
struct Params { m: u32, k: u32, full: u32, pad: u32, thr: f32, p0: f32, p1: f32, p2: f32 };
@group(0) @binding(0) var<storage, read_write> M: array<f32>;
@group(0) @binding(1) var<uniform> P: Params;
@group(0) @binding(2) var<storage, read_write> piv: array<i32>;

var<workgroup> bestV: array<f32, 256>;
var<workgroup> bestR: array<i32, 256>;
var<workgroup> colMax: array<f32, 256>;

@compute @workgroup_size(256)
fn pivot(@builtin(local_invocation_index) t: u32) {
  let m = P.m; let k = P.k;
  var bv = 0.0; var br = -1; var cm = 0.0;
  if (piv[m] == 0) {
    for (var r = k + t; r < m; r += 256u) {
      let a = abs(M[r * m + k]);
      cm = max(cm, a);
      if (r < P.full && a > bv) { bv = a; br = i32(r); }
    }
  }
  bestV[t] = bv; bestR[t] = br; colMax[t] = cm;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (t < s) {
      let o = t + s;
      if (bestV[o] > bestV[t] || (bestV[o] == bestV[t] && bestR[o] >= 0 && (bestR[t] < 0 || bestR[o] < bestR[t]))) {
        bestV[t] = bestV[o]; bestR[t] = bestR[o];
      }
      colMax[t] = max(colMax[t], colMax[o]);
    }
    workgroupBarrier();
  }
  if (t == 0u && piv[m] == 0) {
    if (bestR[0] < 0 || !(bestV[0] > 0.0) || bestV[0] < P.thr * colMax[0]) { piv[m] = i32(k) + 1; }
    else { piv[k] = bestR[0]; }
  }
}

@compute @workgroup_size(64)
fn swaprows(@builtin(global_invocation_id) g: vec3<u32>) {
  let m = P.m; let k = P.k; let c = g.x;
  if (c >= m || piv[m] != 0) { return; }
  let p = u32(piv[k]);
  if (p == k) { return; }
  let a = M[k * m + c]; M[k * m + c] = M[p * m + c]; M[p * m + c] = a;
}

@compute @workgroup_size(16, 16)
fn update(@builtin(global_invocation_id) g: vec3<u32>) {
  let m = P.m; let k = P.k;
  let r = k + 1u + g.y; let c = k + 1u + g.x;
  if (r >= m || c >= m || piv[m] != 0) { return; }
  M[r * m + c] = M[r * m + c] - (M[r * m + k] / M[k * m + k]) * M[k * m + c];
}

@compute @workgroup_size(64)
fn scale(@builtin(global_invocation_id) g: vec3<u32>) {
  let m = P.m; let k = P.k; let r = k + 1u + g.x;
  if (r >= m || piv[m] != 0) { return; }
  M[r * m + k] = M[r * m + k] / M[k * m + k];
}
)";

struct Params { uint32_t m, k, full, pad; float thr, p0, p1, p2; };
constexpr uint32_t kStride = 256;  // minUniformBufferOffsetAlignment

struct Kernels {
    bool ok = false;
    wgpu::BindGroupLayout layout;
    wgpu::ComputePipeline pivot, swap, update, scale;
};

Kernels &kernels(const wgpu::Device &dev) {
    static Kernels K;
    static std::once_flag once;
    std::call_once(once, [&] {
        wgpu::ShaderSourceWGSL wgsl;
        wgsl.code = kWgsl;
        wgpu::ShaderModuleDescriptor sd;
        sd.nextInChain = &wgsl;
        wgpu::ShaderModule mod = dev.CreateShaderModule(&sd);
        wgpu::BindGroupLayoutEntry e[3] = {};
        e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
        e[0].buffer.type = wgpu::BufferBindingType::Storage;
        e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
        e[1].buffer.type = wgpu::BufferBindingType::Uniform;
        e[1].buffer.hasDynamicOffset = true;
        e[1].buffer.minBindingSize = sizeof(Params);
        e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
        e[2].buffer.type = wgpu::BufferBindingType::Storage;
        wgpu::BindGroupLayoutDescriptor ld;
        ld.entryCount = 3; ld.entries = e;
        K.layout = dev.CreateBindGroupLayout(&ld);
        wgpu::PipelineLayoutDescriptor pd;
        pd.bindGroupLayoutCount = 1; pd.bindGroupLayouts = &K.layout;
        wgpu::PipelineLayout pl = dev.CreatePipelineLayout(&pd);
        auto make = [&](const char *entry) {
            wgpu::ComputePipelineDescriptor cd;
            cd.layout = pl;
            cd.compute.module = mod;
            cd.compute.entryPoint = entry;
            return dev.CreateComputePipeline(&cd);
        };
        K.pivot = make("pivot"); K.swap = make("swaprows"); K.update = make("update"); K.scale = make("scale");
        K.ok = K.pivot && K.swap && K.update && K.scale;
    });
    return K;
}

bool readBack(const wgpu::Buffer &b, size_t bytes, void *dst) {
    bool done = false, ok = false;
    b.MapAsync(wgpu::MapMode::Read, 0, bytes, wgpu::CallbackMode::AllowProcessEvents,
               [&](wgpu::MapAsyncStatus s, const char *) { done = true; ok = s == wgpu::MapAsyncStatus::Success; });
    while (!done) GPUContextManager::instance().processEvents();
    if (ok) { std::memcpy(dst, b.GetConstMappedRange(0, bytes), bytes); b.Unmap(); }
    return ok;
}

}  // namespace

bool gpuFrontLuAvailable() { return GPUContextManager::instance().isAvailable(); }

bool gpuFrontLu(double *M, int m, int full, double threshold, int *sw) {
    auto &gpu = GPUContextManager::instance();
    if (!gpu.isAvailable() || m <= 0 || full <= 0) return false;
    const uint64_t bytesM = (uint64_t)m * m * sizeof(float);
    if (bytesM > (128ull << 20)) return false;  // WebGPU's default storage-binding limit
    static std::mutex serial;  // fronts of one level arrive from several threads
    std::lock_guard<std::mutex> lock(serial);
    wgpu::Device dev = gpu.acquire();
    Kernels &K = kernels(dev);
    if (!K.ok) return false;
    wgpu::Queue q = dev.GetQueue();

    auto buffer = [&](uint64_t size, wgpu::BufferUsage usage) {
        wgpu::BufferDescriptor d;
        d.size = size; d.usage = usage;
        return dev.CreateBuffer(&d);
    };
    using U = wgpu::BufferUsage;
    const uint64_t bytesP = (uint64_t)(m + 1) * sizeof(int32_t);
    wgpu::Buffer bm = buffer(bytesM, U::Storage | U::CopyDst | U::CopySrc);
    wgpu::Buffer bp = buffer(bytesP, U::Storage | U::CopyDst | U::CopySrc);
    wgpu::Buffer bu = buffer((uint64_t)full * kStride, U::Uniform | U::CopyDst);
    wgpu::Buffer rm = buffer(bytesM, U::MapRead | U::CopyDst);
    wgpu::Buffer rp = buffer(bytesP, U::MapRead | U::CopyDst);

    std::vector<float> mf((size_t)m * m);
    for (size_t i = 0; i < mf.size(); ++i) mf[i] = (float)M[i];
    q.WriteBuffer(bm, 0, mf.data(), bytesM);
    std::vector<int32_t> piv(m + 1, 0);
    q.WriteBuffer(bp, 0, piv.data(), bytesP);
    std::vector<uint8_t> params((size_t)full * kStride, 0);
    for (int k = 0; k < full; ++k) {
        Params p{(uint32_t)m, (uint32_t)k, (uint32_t)full, 0, (float)threshold, 0, 0, 0};
        std::memcpy(params.data() + (size_t)k * kStride, &p, sizeof p);
    }
    q.WriteBuffer(bu, 0, params.data(), params.size());

    wgpu::BindGroupEntry be[3] = {};
    be[0].binding = 0; be[0].buffer = bm; be[0].size = bytesM;
    be[1].binding = 1; be[1].buffer = bu; be[1].size = sizeof(Params);
    be[2].binding = 2; be[2].buffer = bp; be[2].size = bytesP;
    wgpu::BindGroupDescriptor bd;
    bd.layout = K.layout; bd.entryCount = 3; bd.entries = be;
    wgpu::BindGroup bg = dev.CreateBindGroup(&bd);

    wgpu::CommandEncoder enc = dev.CreateCommandEncoder();
    {
        wgpu::ComputePassEncoder pass = enc.BeginComputePass();
        auto groups = [](int n, int per) { return (uint32_t)((n + per - 1) / per); };
        for (int k = 0; k < full; ++k) {
            const uint32_t off = (uint32_t)k * kStride;
            pass.SetBindGroup(0, bg, 1, &off);
            pass.SetPipeline(K.pivot);  pass.DispatchWorkgroups(1);
            pass.SetPipeline(K.swap);   pass.DispatchWorkgroups(groups(m, 64));
            if (const int rest = m - k - 1; rest > 0) {
                pass.SetPipeline(K.update); pass.DispatchWorkgroups(groups(rest, 16), groups(rest, 16));
                pass.SetPipeline(K.scale);  pass.DispatchWorkgroups(groups(rest, 64));
            }
        }
        pass.End();
    }
    enc.CopyBufferToBuffer(bm, 0, rm, 0, bytesM);
    enc.CopyBufferToBuffer(bp, 0, rp, 0, bytesP);
    wgpu::CommandBuffer cmd = enc.Finish();
    q.Submit(1, &cmd);

    if (!readBack(rp, bytesP, piv.data()) || piv[m] != 0) return false;  // a pivot failed: CPU
    if (!readBack(rm, bytesM, mf.data())) return false;
    for (float v : mf) if (!std::isfinite(v)) return false;  // FP32 overflow: CPU in FP64
    for (size_t i = 0; i < mf.size(); ++i) M[i] = mf[i];
    for (int k = 0; k < full; ++k) sw[k] = piv[k];
    return true;
}

#else  // no GPU in this build

bool gpuFrontLuAvailable() { return false; }
bool gpuFrontLu(double *, int, int, double, int *) { return false; }

#endif
