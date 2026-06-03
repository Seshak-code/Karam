/*
 * tn_contraction.wgsl
 * GPU Tensor Network Contraction Kernels (Gap 1)
 *
 * Three compute shader entry points implementing contraction tree execution:
 *   tn_seed_leaf    - Copy MPO dense matrix + RHS into workspace slot
 *   tn_merge_accum  - Expand two child tensors into merged result index space
 *   tn_schur_elim   - Schur-eliminate one variable from merged tensor
 *
 * Uses EmulatedF64 hi/lo pairs for SPICE-level precision (15-17 digits).
 * Tile size capped at MAX_TILE_RANK = 8 (8x8 = 64 f64 entries ~ 1KB),
 * comfortably within the 163KB SM shared memory budget.
 */

// --- EmulatedF64 type and arithmetic ---

struct f64 {
    hi: f32,
    lo: f32,
};

fn f64_from_f32(a: f32) -> f64 {
    return f64(a, 0.0);
}

fn f64_from_f64(hi: f32, lo: f32) -> f64 {
    return f64(hi, lo);
}

fn f64_add(a: f64, b: f64) -> f64 {
    let s = a.hi + b.hi;
    let v = s - a.hi;
    let e = (a.hi - (s - v)) + (b.hi - v) + a.lo + b.lo;
    let r_hi = s + e;
    let r_lo = e + (s - r_hi);
    return f64(r_hi, r_lo);
}

fn f64_sub(a: f64, b: f64) -> f64 {
    return f64_add(a, f64(-b.hi, -b.lo));
}

fn f64_mul(a: f64, b: f64) -> f64 {
    let hi = a.hi * b.hi;
    let lo = fma(a.hi, b.hi, -hi) + a.hi * b.lo + a.lo * b.hi;
    let s = hi + lo;
    let l = lo + (hi - s);
    return f64(s, l);
}

fn f64_div(a: f64, b: f64) -> f64 {
    let x = a.hi / b.hi;
    let x64 = f64_from_f32(x);
    let b_x = f64_mul(b, x64);
    let error = f64_sub(f64_from_f32(2.0), b_x);
    let refined_recip = f64_mul(x64, error);
    return f64_mul(a, refined_recip);
}

fn f64_select(false_val: f64, true_val: f64, cond: bool) -> f64 {
    return f64(
        select(false_val.hi, true_val.hi, cond),
        select(false_val.lo, true_val.lo, cond)
    );
}

// --- Constants ---
const MAX_TILE_RANK: u32 = 8u;
const MAX_TILE_ELEMS: u32 = 64u;  // MAX_TILE_RANK * MAX_TILE_RANK

// --- Uniforms (per-dispatch, uploaded by CPU) ---
struct ContractionUniforms {
    slot_left: u32,      // workspace slot index of left child
    slot_right: u32,     // workspace slot index of right child
    slot_result: u32,    // workspace slot index for result
    rank_left: u32,      // dimension of left child's index set
    rank_right: u32,     // dimension of right child's index set
    rank_result: u32,    // dimension of result's index set
    elim_row: u32,       // row index (within merged tensor) of variable to eliminate
    do_schur: u32,       // 1 = perform Schur elimination, 0 = skip
};

// --- Buffer bindings ---
// Group 0: workspace storage (persistent across tree walk)
@group(0) @binding(0) var<storage, read_write> workspace_hi: array<f32>;
@group(0) @binding(1) var<storage, read_write> workspace_lo: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhs_hi: array<f32>;
@group(0) @binding(3) var<storage, read_write> rhs_lo: array<f32>;
@group(0) @binding(4) var<uniform> uniforms: ContractionUniforms;

// Group 1: index mapping tables (updated per contraction step)
@group(1) @binding(0) var<storage, read> index_map_left: array<u32>;
@group(1) @binding(1) var<storage, read> index_map_right: array<u32>;

// --- Shared memory tile for Schur elimination (TBR pattern) ---
var<workgroup> tile_mat_hi: array<f32, MAX_TILE_ELEMS>;
var<workgroup> tile_mat_lo: array<f32, MAX_TILE_ELEMS>;
var<workgroup> tile_rhs_hi: array<f32, MAX_TILE_RANK>;
var<workgroup> tile_rhs_lo: array<f32, MAX_TILE_RANK>;

// --- Helper: compute flat offset into workspace for slot s, row r, col c ---
fn ws_mat_offset(slot: u32, rank: u32, row: u32, col: u32) -> u32 {
    // Each slot occupies MAX_TILE_RANK * MAX_TILE_RANK floats for matrix
    return slot * MAX_TILE_ELEMS + row * rank + col;
}

fn ws_rhs_offset(slot: u32, row: u32) -> u32 {
    // Each slot occupies MAX_TILE_RANK floats for RHS
    return slot * MAX_TILE_RANK + row;
}

// ============================================================================
// Entry Point 1: tn_seed_leaf
// Copies an MPO's dense matrix + RHS from source slot into result slot.
// Dispatch: ceil(rank_result * rank_result / 64) workgroups.
// ============================================================================
@compute @workgroup_size(64)
fn tn_seed_leaf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rank = uniforms.rank_result;
    let total = rank * rank;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let src_slot = uniforms.slot_left;   // source MPO data is in slot_left
    let dst_slot = uniforms.slot_result;

    // Copy matrix element
    let src_off = src_slot * MAX_TILE_ELEMS + idx;
    let dst_off = dst_slot * MAX_TILE_ELEMS + idx;
    workspace_hi[dst_off] = workspace_hi[src_off];
    workspace_lo[dst_off] = workspace_lo[src_off];

    // Copy RHS (only for first 'rank' threads)
    if (idx < rank) {
        let src_r = src_slot * MAX_TILE_RANK + idx;
        let dst_r = dst_slot * MAX_TILE_RANK + idx;
        rhs_hi[dst_r] = rhs_hi[src_r];
        rhs_lo[dst_r] = rhs_lo[src_r];
    }
}

// ============================================================================
// Entry Point 2: tn_merge_accum
// Expands two child tensors into merged result index space using index maps.
// index_map_left[i] = position in result's index set for left's i-th index.
// index_map_right[i] = position in result's index set for right's i-th index.
// Dispatch: ceil(max(rank_left, rank_right)^2 / 64) workgroups.
// ============================================================================
@compute @workgroup_size(64)
fn tn_merge_accum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rank_l = uniforms.rank_left;
    let rank_r = uniforms.rank_right;
    let rank_res = uniforms.rank_result;
    let idx = gid.x;

    // Zero the result slot first (only need rank_res^2 threads)
    let res_total = rank_res * rank_res;
    if (idx < res_total) {
        let off = uniforms.slot_result * MAX_TILE_ELEMS + idx;
        workspace_hi[off] = 0.0;
        workspace_lo[off] = 0.0;
    }
    if (idx < rank_res) {
        let off = uniforms.slot_result * MAX_TILE_RANK + idx;
        rhs_hi[off] = 0.0;
        rhs_lo[off] = 0.0;
    }

    workgroupBarrier();

    // Accumulate left child's matrix into result using index_map_left
    if (idx < rank_l * rank_l) {
        let row_l = idx / rank_l;
        let col_l = idx % rank_l;
        let row_res = index_map_left[row_l];
        let col_res = index_map_left[col_l];

        let src = uniforms.slot_left * MAX_TILE_ELEMS + row_l * rank_l + col_l;
        let dst = uniforms.slot_result * MAX_TILE_ELEMS + row_res * rank_res + col_res;

        // Atomic-free accumulation: each (row_res, col_res) pair from left is
        // unique per left child, so no race within this block.
        let val = f64_add(
            f64(workspace_hi[dst], workspace_lo[dst]),
            f64(workspace_hi[src], workspace_lo[src])
        );
        workspace_hi[dst] = val.hi;
        workspace_lo[dst] = val.lo;
    }

    // Accumulate left RHS
    if (idx < rank_l) {
        let row_res = index_map_left[idx];
        let src = uniforms.slot_left * MAX_TILE_RANK + idx;
        let dst = uniforms.slot_result * MAX_TILE_RANK + row_res;
        let val = f64_add(
            f64(rhs_hi[dst], rhs_lo[dst]),
            f64(rhs_hi[src], rhs_lo[src])
        );
        rhs_hi[dst] = val.hi;
        rhs_lo[dst] = val.lo;
    }

    workgroupBarrier();

    // Accumulate right child's matrix into result using index_map_right
    if (idx < rank_r * rank_r) {
        let row_r = idx / rank_r;
        let col_r = idx % rank_r;
        let row_res = index_map_right[row_r];
        let col_res = index_map_right[col_r];

        let src = uniforms.slot_right * MAX_TILE_ELEMS + row_r * rank_r + col_r;
        let dst = uniforms.slot_result * MAX_TILE_ELEMS + row_res * rank_res + col_res;

        let val = f64_add(
            f64(workspace_hi[dst], workspace_lo[dst]),
            f64(workspace_hi[src], workspace_lo[src])
        );
        workspace_hi[dst] = val.hi;
        workspace_lo[dst] = val.lo;
    }

    // Accumulate right RHS
    if (idx < rank_r) {
        let row_res = index_map_right[idx];
        let src = uniforms.slot_right * MAX_TILE_RANK + idx;
        let dst = uniforms.slot_result * MAX_TILE_RANK + row_res;
        let val = f64_add(
            f64(rhs_hi[dst], rhs_lo[dst]),
            f64(rhs_hi[src], rhs_lo[src])
        );
        rhs_hi[dst] = val.hi;
        rhs_lo[dst] = val.lo;
    }
}

// ============================================================================
// Entry Point 3: tn_schur_elim
// Schur-eliminates one variable (at elim_row) from the merged tensor in
// slot_result. Uses workgroup shared memory tile (TBR pattern).
//
// Schur complement:
//   A_NN -= A_Nv * (1/A_vv) * A_vN
//   rhs_N -= A_Nv * (1/A_vv) * rhs_v
//
// Dispatch: 1 workgroup (single workgroup performs entire elimination).
// ============================================================================
@compute @workgroup_size(64)
fn tn_schur_elim(@builtin(local_invocation_id) lid: vec3<u32>) {
    let rank = uniforms.rank_result;
    let v = uniforms.elim_row;
    let slot = uniforms.slot_result;
    let tid = lid.x;

    if (uniforms.do_schur == 0u) {
        return;
    }

    // Load tile into shared memory
    if (tid < rank * rank) {
        let off = slot * MAX_TILE_ELEMS + tid;
        tile_mat_hi[tid] = workspace_hi[off];
        tile_mat_lo[tid] = workspace_lo[off];
    }
    if (tid < rank) {
        let off = slot * MAX_TILE_RANK + tid;
        tile_rhs_hi[tid] = rhs_hi[off];
        tile_rhs_lo[tid] = rhs_lo[off];
    }
    workgroupBarrier();

    // Read pivot A[v][v]
    let pivot_idx = v * rank + v;
    let pivot = f64(tile_mat_hi[pivot_idx], tile_mat_lo[pivot_idx]);

    // Guard against zero pivot
    let abs_pivot = abs(pivot.hi) + abs(pivot.lo);
    if (abs_pivot < 1e-30) {
        return;
    }

    let inv_pivot = f64_div(f64_from_f32(1.0), pivot);

    // Read rhs[v]
    let rhs_v = f64(tile_rhs_hi[v], tile_rhs_lo[v]);

    // Each thread handles one (i) row where i != v and i < rank
    // Iterate over all valid rows, one per thread
    if (tid < rank && tid != v) {
        let i = tid;
        // A_iv = A[i][v]
        let a_iv = f64(tile_mat_hi[i * rank + v], tile_mat_lo[i * rank + v]);
        let factor = f64_mul(a_iv, inv_pivot);

        // Update row i: A[i][j] -= factor * A[v][j] for all j != v
        for (var j = 0u; j < rank; j = j + 1u) {
            if (j != v) {
                let a_vj = f64(tile_mat_hi[v * rank + j], tile_mat_lo[v * rank + j]);
                let update = f64_mul(factor, a_vj);
                let old_val = f64(tile_mat_hi[i * rank + j], tile_mat_lo[i * rank + j]);
                let new_val = f64_sub(old_val, update);
                tile_mat_hi[i * rank + j] = new_val.hi;
                tile_mat_lo[i * rank + j] = new_val.lo;
            }
        }

        // Update RHS: rhs[i] -= factor * rhs[v]
        let rhs_update = f64_mul(factor, rhs_v);
        let old_rhs = f64(tile_rhs_hi[i], tile_rhs_lo[i]);
        let new_rhs = f64_sub(old_rhs, rhs_update);
        tile_rhs_hi[i] = new_rhs.hi;
        tile_rhs_lo[i] = new_rhs.lo;
    }

    workgroupBarrier();

    // Write tile back to global memory
    if (tid < rank * rank) {
        let off = slot * MAX_TILE_ELEMS + tid;
        workspace_hi[off] = tile_mat_hi[tid];
        workspace_lo[off] = tile_mat_lo[tid];
    }
    if (tid < rank) {
        let off = slot * MAX_TILE_RANK + tid;
        rhs_hi[off] = tile_rhs_hi[tid];
        rhs_lo[off] = tile_rhs_lo[tid];
    }
}

// ============================================================================
// GPU-Resident Elimination Record Buffers (Gap 1 Phase 2)
//
// These buffers allow Schur elimination to record its state directly on
// the GPU, eliminating the CPU-GPU ping-pong during the contraction sweep.
//
// Layout per elimination record (at record_offset):
//   elim_pivot_hi/lo[record_offset]              — pivot value A[v][v]
//   elim_rhs_hi/lo[record_offset]                — rhs[v] at elimination time
//   elim_var_ids[record_offset]                   — MNA variable ID being eliminated
//   elim_neighbor_count[record_offset]            — number of neighbor variables
//   elim_row_hi/lo[record_offset * MAX_TILE_RANK + j] — A[v][neighbor_j]
//   elim_neighbor_ids[record_offset * MAX_TILE_RANK + j] — neighbor var IDs
// ============================================================================

// Extended uniforms for GPU-resident elimination
struct ElimRecordUniforms {
    record_offset: u32,      // index into elimination record arrays
    elim_var_id: u32,        // MNA variable ID of the eliminated variable
    num_neighbors: u32,      // number of neighbor variables (rank - 1)
    backsub_count: u32,      // total number of records for back-substitution
    slot_result: u32,        // workspace slot to operate on
    rank_result: u32,        // dimension of the tensor
    elim_row: u32,           // row/col index to eliminate
    gmin_value: u32,         // bit-cast f32 GMIN for pivot conditioning
};

// Group 2: elimination record storage (GPU-resident, persistent)
@group(2) @binding(0) var<storage, read_write> elim_pivot_hi: array<f32>;
@group(2) @binding(1) var<storage, read_write> elim_pivot_lo: array<f32>;
@group(2) @binding(2) var<storage, read_write> elim_rhs_hi: array<f32>;
@group(2) @binding(3) var<storage, read_write> elim_rhs_lo: array<f32>;
@group(2) @binding(4) var<storage, read_write> elim_row_hi: array<f32>;
@group(2) @binding(5) var<storage, read_write> elim_row_lo: array<f32>;
@group(2) @binding(6) var<storage, read_write> elim_var_ids: array<u32>;
@group(2) @binding(7) var<storage, read_write> elim_neighbor_ids: array<u32>;
@group(2) @binding(8) var<storage, read_write> elim_neighbor_count: array<u32>;
@group(2) @binding(9) var<storage, read_write> solution_hi: array<f32>;
@group(2) @binding(10) var<storage, read_write> solution_lo: array<f32>;
@group(2) @binding(11) var<uniform> elim_uniforms: ElimRecordUniforms;

// ============================================================================
// Entry Point 4: tn_schur_elim_record
// GPU-resident Schur elimination with inline record writing.
//
// Performs the same Schur complement as tn_schur_elim, but additionally:
//   1. Writes the pivot, elimination row, and RHS to GPU record buffers
//   2. Records neighbor variable IDs for back-substitution
//   3. Compacts the eliminated row/col in-place (zeroes out the eliminated
//      row and column so subsequent merges treat them as inactive)
//
// This eliminates the need to download/upload between GPU and CPU.
// Dispatch: 1 workgroup per elimination.
// ============================================================================
@compute @workgroup_size(64)
fn tn_schur_elim_record(@builtin(local_invocation_id) lid: vec3<u32>) {
    let rank = elim_uniforms.rank_result;
    let v = elim_uniforms.elim_row;
    let slot = elim_uniforms.slot_result;
    let rec_off = elim_uniforms.record_offset;
    let tid = lid.x;

    // Load tile into shared memory
    if (tid < rank * rank) {
        let off = slot * MAX_TILE_ELEMS + tid;
        tile_mat_hi[tid] = workspace_hi[off];
        tile_mat_lo[tid] = workspace_lo[off];
    }
    if (tid < rank) {
        let off = slot * MAX_TILE_RANK + tid;
        tile_rhs_hi[tid] = rhs_hi[off];
        tile_rhs_lo[tid] = rhs_lo[off];
    }
    workgroupBarrier();

    // Read pivot A[v][v] with GMIN conditioning
    let pivot_idx = v * rank + v;
    var pivot = f64(tile_mat_hi[pivot_idx], tile_mat_lo[pivot_idx]);
    let abs_pivot = abs(pivot.hi) + abs(pivot.lo);

    // GMIN conditioning: add small conductance if pivot is near-zero
    let gmin = bitcast<f32>(elim_uniforms.gmin_value);
    if (abs_pivot < 1e-20) {
        pivot = f64_add(pivot, f64_from_f32(gmin));
        tile_mat_hi[pivot_idx] = pivot.hi;
        tile_mat_lo[pivot_idx] = pivot.lo;
    }

    // Abort if still too small after conditioning
    let abs_pivot2 = abs(pivot.hi) + abs(pivot.lo);
    if (abs_pivot2 < 1e-30) {
        // Write zero pivot to mark invalid record
        if (tid == 0u) {
            elim_pivot_hi[rec_off] = 0.0;
            elim_pivot_lo[rec_off] = 0.0;
            elim_neighbor_count[rec_off] = 0u;
        }
        return;
    }

    let inv_pivot = f64_div(f64_from_f32(1.0), pivot);
    let rhs_v = f64(tile_rhs_hi[v], tile_rhs_lo[v]);

    // --- Thread 0: Write elimination record to GPU buffers ---
    if (tid == 0u) {
        elim_pivot_hi[rec_off] = pivot.hi;
        elim_pivot_lo[rec_off] = pivot.lo;
        elim_rhs_hi[rec_off] = rhs_v.hi;
        elim_rhs_lo[rec_off] = rhs_v.lo;
        elim_var_ids[rec_off] = elim_uniforms.elim_var_id;

        // Write elimination row (A[v, j] for j != v) and neighbor IDs
        var neighbor_idx = 0u;
        let row_base = rec_off * MAX_TILE_RANK;
        for (var j = 0u; j < rank; j = j + 1u) {
            if (j != v) {
                elim_row_hi[row_base + neighbor_idx] = tile_mat_hi[v * rank + j];
                elim_row_lo[row_base + neighbor_idx] = tile_mat_lo[v * rank + j];
                // Neighbor var ID is read from index_map_left which is reused
                // as the merged index set for this purpose
                elim_neighbor_ids[row_base + neighbor_idx] = index_map_left[j];
                neighbor_idx = neighbor_idx + 1u;
            }
        }
        elim_neighbor_count[rec_off] = neighbor_idx;
    }

    workgroupBarrier();

    // --- Schur complement (identical math to tn_schur_elim) ---
    if (tid < rank && tid != v) {
        let i = tid;
        let a_iv = f64(tile_mat_hi[i * rank + v], tile_mat_lo[i * rank + v]);
        let factor = f64_mul(a_iv, inv_pivot);

        for (var j = 0u; j < rank; j = j + 1u) {
            if (j != v) {
                let a_vj = f64(tile_mat_hi[v * rank + j], tile_mat_lo[v * rank + j]);
                let update = f64_mul(factor, a_vj);
                let old_val = f64(tile_mat_hi[i * rank + j], tile_mat_lo[i * rank + j]);
                let new_val = f64_sub(old_val, update);
                tile_mat_hi[i * rank + j] = new_val.hi;
                tile_mat_lo[i * rank + j] = new_val.lo;
            }
        }

        let rhs_update = f64_mul(factor, rhs_v);
        let old_rhs = f64(tile_rhs_hi[i], tile_rhs_lo[i]);
        let new_rhs = f64_sub(old_rhs, rhs_update);
        tile_rhs_hi[i] = new_rhs.hi;
        tile_rhs_lo[i] = new_rhs.lo;
    }

    workgroupBarrier();

    // --- Zero out eliminated row and column (in-place compaction marker) ---
    // This marks the eliminated variable as inactive so subsequent merges
    // that read this slot won't accumulate stale data.
    if (tid < rank) {
        tile_mat_hi[v * rank + tid] = 0.0;
        tile_mat_lo[v * rank + tid] = 0.0;
        tile_mat_hi[tid * rank + v] = 0.0;
        tile_mat_lo[tid * rank + v] = 0.0;
    }
    if (tid == 0u) {
        tile_rhs_hi[v] = 0.0;
        tile_rhs_lo[v] = 0.0;
    }

    workgroupBarrier();

    // Write tile back to global memory
    if (tid < rank * rank) {
        let off = slot * MAX_TILE_ELEMS + tid;
        workspace_hi[off] = tile_mat_hi[tid];
        workspace_lo[off] = tile_mat_lo[tid];
    }
    if (tid < rank) {
        let off = slot * MAX_TILE_RANK + tid;
        rhs_hi[off] = tile_rhs_hi[tid];
        rhs_lo[off] = tile_rhs_lo[tid];
    }
}

// ============================================================================
// Entry Point 5: tn_back_substitute
// GPU-resident back-substitution: recovers eliminated variables from the
// GPU elimination record buffers.
//
// For each record (processed in reverse order by the CPU dispatch loop):
//   x_v = (rhs_v − Σ_j A[v][j] · x_j) / pivot_v
//
// The CPU dispatches this kernel once per elimination record in reverse
// tree-walk order. Each dispatch is a single workgroup that:
//   1. Reads the record's neighbor IDs and elimination row from GPU buffers
//   2. Looks up solved neighbor voltages from the solution buffer
//   3. Computes x_v and writes it to the solution buffer
//
// Dispatch: 1 workgroup per back-substitution step.
// ============================================================================
@compute @workgroup_size(64)
fn tn_back_substitute(@builtin(local_invocation_id) lid: vec3<u32>) {
    let rec_off = elim_uniforms.record_offset;
    let tid = lid.x;

    // Read record metadata
    let pivot = f64(elim_pivot_hi[rec_off], elim_pivot_lo[rec_off]);
    let abs_pivot = abs(pivot.hi) + abs(pivot.lo);
    if (abs_pivot < 1e-30) {
        return;  // invalid record (zero pivot)
    }

    let rhs_v = f64(elim_rhs_hi[rec_off], elim_rhs_lo[rec_off]);
    let var_id = elim_var_ids[rec_off];
    let num_neighbors = elim_neighbor_count[rec_off];
    let row_base = rec_off * MAX_TILE_RANK;

    // Serial reduction in thread 0 (num_neighbors ≤ 7 for MAX_TILE_RANK=8)
    // For such small neighbor counts, parallel reduction overhead exceeds benefit.
    if (tid == 0u) {
        var total_sum = f64_from_f32(0.0);
        for (var k = 0u; k < num_neighbors; k = k + 1u) {
            let nvar = elim_neighbor_ids[row_base + k];
            let a_vk = f64(elim_row_hi[row_base + k], elim_row_lo[row_base + k]);
            var x_k = f64_from_f32(0.0);
            if (nvar > 0u) {
                let si = nvar - 1u;
                x_k = f64(solution_hi[si], solution_lo[si]);
            }
            total_sum = f64_add(total_sum, f64_mul(a_vk, x_k));
        }

        // x_v = (rhs_v - sum) / pivot
        let numerator = f64_sub(rhs_v, total_sum);
        let x_v = f64_div(numerator, pivot);

        // Write to solution buffer
        if (var_id > 0u) {
            let sol_idx = var_id - 1u;
            solution_hi[sol_idx] = x_v.hi;
            solution_lo[sol_idx] = x_v.lo;
        }
    }
}
