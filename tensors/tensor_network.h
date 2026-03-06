#pragma once

/**
 * tensor_network.h — Tensor Network Unified Façade
 *
 * Single include that re-exports all core Tensor Network types for the
 * acutesim_engine's Tensor-MNA architecture. Use this as the entry point
 * when working with the TN pipeline from outside the engine internals.
 *
 * Pipeline: FactorGraph → MPOBuilder → ContractionTree → TNCompiler → ContractionExecutor
 */

// Phase A: Factor Graph IR
#include "../infrastructure/factor_graph.h"             // FactorNode, VariableNode, FactorGraph
#include "../infrastructure/treewidth_analyzer.h"       // TreewidthAnalysis

// Phase 5.3.0: Bond Dimension Profiling
#include "../infrastructure/bond_dimension_profiler.h"  // BondDimensionProfile, BondDimensionProfiler

// Phase 5.3.1: TN Compilation
#include "../infrastructure/mpo_builder.h"              // DeviceMPO, MPOBuilder
#include "../infrastructure/contraction_tree.h"         // ContractionNode, ContractionTree
#include "../infrastructure/tn_compiler.h"              // TNCompiledProgram, TNCompiler

// Phase C: Graph Partitioning
#include "../infrastructure/graph_partitioner.h"        // TensorPartition, GraphPartitioner

// Phase 5.3.3: Contraction Execution
#include "../solvers/contraction_executor.h"            // ContractionExecutor
#include "../solvers/tensor_workspace.h"                // TensorWorkspace, IntermediateTensor
#include "../solvers/schur_solver.h"                    // SchurSolver, SchurContribution

// Phase 5.3.4: Graphics-Pipeline Optimizations
#include "../infrastructure/adaptive_precision.h"       // VRS: AdaptivePrecisionRouter, PrecisionTier
#include "../solvers/neural_seeder.h"                   // DLSS: NeuralSeeder
