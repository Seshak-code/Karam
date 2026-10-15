#pragma once
// Dense FP32 LU of one contraction front on the GPU (Dawn / WebGPU), Phase 4.
//
// The same elimination contraction_solver.cpp runs on the CPU in FP64, minus its
// column search: for k < full, row partial pivoting over the fully-summed rows
// (k..full-1), accepted when |pivot| >= threshold * max over the whole column
// (boundary rows only veto), then the rank-1 update of the trailing matrix, the
// boundary block included (that block is the update tensor for the parent).
// A pivot that fails the threshold returns false and the caller factors the
// front on the CPU. FP32 factors are corrected by the solver's FP64 iterative
// refinement against the original matrix.

// True when a GPU device is available for front factorization.
bool gpuFrontLuAvailable();

// M: row-major m x m (in: assembled front, out: L below / U on+above the
// diagonal of the first `full` columns, update tensor in the trailing block).
// sw[k]: row swapped into k at step k (k < full). Returns false when the GPU is
// unavailable or a pivot fails the threshold (M is then left unchanged).
bool gpuFrontLu(double *M, int m, int full, double threshold, int *sw);
