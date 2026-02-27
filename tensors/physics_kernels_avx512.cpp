#include "physics_tensors.h"
#include <vector>
#include <cmath>
#include <algorithm>

// AVX-512 intrinsic support
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

/*
 * physics_kernels_avx512.cpp
 * 
 * Optimized AVX-512 implementations for device physics.
 * Included in build only if compiler supports AVX-512 or macro guarded.
 */

#if defined(__AVX512F__)

// Helper to compute exp() on __m512d using std::exp lane-wise (Fallback if no SVML)
// In a real high-perf scenario, use Intel SVML (_mm512_exp_pd) or SLEEF.
inline __m512d _mm512_exp_pd_fallback(__m512d v) {
    alignas(64) double tmp[8];
    _mm512_store_pd(tmp, v);
    for(int i=0; i<8; ++i) tmp[i] = std::exp(tmp[i]);
    return _mm512_load_pd(tmp);
}

void batchDiodePhysics_avx512(DiodeTensor& tensor, const std::vector<double>& voltages) {
    size_t n = tensor.size();
    const double GMIN = 1e-12;
    
    const __m512d v_gmin = _mm512_set1_pd(GMIN);
    const __m512d v_zero = _mm512_setzero_pd();
    const __m512d v_one  = _mm512_set1_pd(1.0);
    const __m512d v_30   = _mm512_set1_pd(30.0);
    
    size_t i = 0;
    // Process 8 diodes at a time
    for (; i + 7 < n; i += 8) {
        // 1. Gather Voltages (Scatter/Gather is slow pre-IceLake, but necessary for indirect access)
        // Load Anode/Cathode Indices
        // tensor.anodes is int (32-bit). We need to load them into __m256i.
        __m256i idx_a = _mm256_loadu_si256((__m256i*)&tensor.anodes[i]);
        __m256i idx_c = _mm256_loadu_si256((__m256i*)&tensor.cathodes[i]);
        
        // Manual Gather (Portable) vs _mm512_i32gather_pd
        // We do manual gather because voltages is std::vector checking bounds in loop might be needed
        // But for kernel speed we assume valid indices or 0.
        // To be safe and fast:
        alignas(64) double va_vals[8];
        alignas(64) double vc_vals[8];
        
        // Scalar fallback for gathering (often faster/safer than tricky gather instructions on some archs)
        const double* v_ptr = voltages.data();
        size_t v_size = voltages.size();
        
        for(int k=0; k<8; ++k) {
            int ia = tensor.anodes[i+k];
            int ic = tensor.cathodes[i+k];
            va_vals[k] = (ia > 0 && ia <= (int)v_size) ? v_ptr[ia-1] : 0.0;
            vc_vals[k] = (ic > 0 && ic <= (int)v_size) ? v_ptr[ic-1] : 0.0;
        }
        
        __m512d v_a = _mm512_load_pd(va_vals);
        __m512d v_c = _mm512_load_pd(vc_vals);
        
        // v_d = v_a - v_c
        __m512d v_d = _mm512_sub_pd(v_a, v_c);
        _mm512_storeu_pd(&tensor.v_d[i], v_d);
        
        // Parameters
        __m512d v_Is = _mm512_loadu_pd(&tensor.Is[i]);
        __m512d v_N  = _mm512_loadu_pd(&tensor.N[i]);
        __m512d v_Vt = _mm512_loadu_pd(&tensor.Vt[i]);
        
        // n_vt = N * Vt
        __m512d v_nvt = _mm512_mul_pd(v_N, v_Vt);
        
        // Vcrit = 30.0 * n_vt
        __m512d v_vcrit = _mm512_mul_pd(v_30, v_nvt);
        
        // arg = v_d / n_vt
        __m512d v_arg = _mm512_div_pd(v_d, v_nvt);
        
        // arg_clamped = min(arg, 30.0)
        __m512d v_arg_clamped = _mm512_min_pd(v_arg, v_30);
        
        // exp_arg = exp(arg_clamped)
        __m512d v_exp = _mm512_exp_pd_fallback(v_arg_clamped);
        
        // i_base = Is * (exp - 1.0)
        __m512d v_ibase = _mm512_mul_pd(v_Is, _mm512_sub_pd(v_exp, v_one));
        
        // g_base = (Is / n_vt) * exp
        __m512d v_gbase = _mm512_mul_pd(_mm512_div_pd(v_Is, v_nvt), v_exp);
        
        // delta = max(0.0, v_d - Vcrit)
        __m512d v_delta = _mm512_max_pd(v_zero, _mm512_sub_pd(v_d, v_vcrit));
        
        // i_d = i_base + delta * g_base
        __m512d v_id = _mm512_add_pd(v_ibase, _mm512_mul_pd(v_delta, v_gbase));
        
        // g_d = max(GMIN, g_base)
        __m512d v_gd = _mm512_max_pd(v_gmin, v_gbase);
        
        // Store results
        _mm512_storeu_pd(&tensor.i_d[i], v_id);
        _mm512_storeu_pd(&tensor.g_d[i], v_gd);
    }
    
    // Process remaining elements (Tail)
    for (; i < n; ++i) {
        int nA = tensor.anodes[i];
        int nC = tensor.cathodes[i];
        
        double v_a = (nA > 0 && nA <= (int)voltages.size()) ? voltages[nA - 1] : 0.0;
        double v_c = (nC > 0 && nC <= (int)voltages.size()) ? voltages[nC - 1] : 0.0;
        
        tensor.v_d[i] = v_a - v_c;
        
        double Is = tensor.Is[i];
        double N = tensor.N[i];
        double Vt = tensor.Vt[i];
        double n_vt = N * Vt;
        double Vcrit = 30.0 * n_vt;
        
        double arg = tensor.v_d[i] / n_vt;
        double arg_clamped = std::min(arg, 30.0);
        double exp_arg = std::exp(arg_clamped);
        
        double i_base = Is * (exp_arg - 1.0);
        double g_base = (Is / n_vt) * exp_arg;
        
        double delta = std::max(0.0, tensor.v_d[i] - Vcrit);
        tensor.i_d[i] = i_base + delta * g_base;
        tensor.g_d[i] = std::max(GMIN, g_base);
    }
}

#endif // __AVX512F__
