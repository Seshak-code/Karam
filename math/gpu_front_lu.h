#pragma once
// Dense FP32 LU of one contraction front on the GPU (Dawn / WebGPU), Phase 4.
//
// The same elimination contraction_solver.cpp runs on the CPU in FP64, minus its
// column search: for k < full, row partial pivoting over the fully-summed rows
// (k..full-1), accepted when |pivot| >= threshold * max over the whole column
// (boundary rows only veto), then the rank-1 update of the trailing matrix, the
// boundary block included (that block is the update tensor for the parent).
// It stops at the first pivot that fails the threshold; the caller continues
// from there on the CPU (its column search and delayed pivots). FP32 factors are
// corrected by the solver's FP64 iterative refinement against the original matrix.

// True when a GPU device is available for front factorization.
bool gpuFrontLuAvailable();

// M: row-major m x m (in: assembled front, out: after k steps, L below / U on+
// above the diagonal of the first k columns, the rest updated). sw[i]: row
// swapped into i at step i (i < k). Returns k, the steps eliminated (0..full);
// 0 leaves M unchanged (no GPU, a first pivot that fails, a front too large).
int gpuFrontLu(double *M, int m, int full, double threshold, int *sw);
