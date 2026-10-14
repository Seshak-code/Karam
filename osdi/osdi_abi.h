#pragma once
// OSDI (Open Source Device Interface) binary interface, versions 0.3 and 0.4:
// the layout of the data a compiled Verilog-A model library (e.g. from
// OpenVAF) exports, as published in the OSDI specification. Written for
// TreeSPICE from that specification; only what the loader uses is named.
//
// A 0.4 library's descriptor extends the 0.3 one; the loader steps through
// OSDI_DESCRIPTORS by OSDI_DESCRIPTOR_SIZE and reads the 0.3 prefix below.
#include <cstddef>
#include <cstdint>

namespace osdi {

// Parameter / opvar flags
constexpr uint32_t PARA_TY_MASK = 3, PARA_TY_REAL = 0, PARA_TY_INT = 1, PARA_TY_STR = 2;
constexpr uint32_t PARA_KIND_MASK = 3u << 30, PARA_KIND_MODEL = 0u << 30, PARA_KIND_INST = 1u << 30,
                   PARA_KIND_OPVAR = 2u << 30;
// access() flags
constexpr uint32_t ACCESS_FLAG_READ = 0, ACCESS_FLAG_SET = 1, ACCESS_FLAG_INSTANCE = 4;
// Jacobian entry flags
constexpr uint32_t JACOBIAN_ENTRY_RESIST_CONST = 1, JACOBIAN_ENTRY_REACT_CONST = 2,
                   JACOBIAN_ENTRY_RESIST = 4, JACOBIAN_ENTRY_REACT = 8;
// eval() flags
constexpr uint32_t CALC_RESIST_RESIDUAL = 1, CALC_REACT_RESIDUAL = 2, CALC_RESIST_JACOBIAN = 4,
                   CALC_REACT_JACOBIAN = 8, CALC_NOISE = 16, CALC_OP = 32, CALC_RESIST_LIM_RHS = 64,
                   CALC_REACT_LIM_RHS = 128, ENABLE_LIM = 256, INIT_LIM = 512, ANALYSIS_NOISE = 1024,
                   ANALYSIS_DC = 2048, ANALYSIS_AC = 4096, ANALYSIS_TRAN = 8192, ANALYSIS_IC = 16384,
                   ANALYSIS_STATIC = 32768, ANALYSIS_NODESET = 65536;
// eval() return flags
constexpr uint32_t EVAL_RET_FLAG_LIM = 1, EVAL_RET_FLAG_FATAL = 2, EVAL_RET_FLAG_FINISH = 4,
                   EVAL_RET_FLAG_STOP = 8;
constexpr uint32_t LOG_LVL_MASK = 7, LOG_LVL_ERR = 4, LOG_LVL_FATAL = 5;
constexpr uint32_t NONE = UINT32_MAX;  // "no node" (ground) in node pairs, "no pointer" offsets

struct LimFunction { char *name; uint32_t num_args; void *func_ptr; };
struct SimParas { char **names; double *vals; char **names_str; char **vals_str; };
struct SimInfo {
    SimParas paras;
    double abstime;
    double *prev_solve;
    double *prev_state;
    double *next_state;
    uint32_t flags;
};
struct InitError { uint32_t code; uint32_t parameter_id; };
struct InitInfo { uint32_t flags; uint32_t num_errors; InitError *errors; };
struct NodePair { uint32_t node_1, node_2; };  // Jacobian: node_1 = row (equation), node_2 = column
struct JacobianEntry { NodePair nodes; uint32_t react_ptr_off; uint32_t flags; };
struct Node {
    char *name, *units, *residual_units;
    uint32_t resist_residual_off, react_residual_off, resist_limit_rhs_off, react_limit_rhs_off;
    bool is_flow;
};
struct ParamOpvar { char **name; uint32_t num_alias; char *description; char *units; uint32_t flags; uint32_t len; };
struct NoiseSource { char *name; NodePair nodes; };

// The OSDI 0.3 descriptor (the prefix of a 0.4 one).
struct Descriptor {
    char *name;
    uint32_t num_nodes, num_terminals;
    Node *nodes;
    uint32_t num_jacobian_entries;
    JacobianEntry *jacobian_entries;
    uint32_t num_collapsible;
    NodePair *collapsible;
    uint32_t collapsed_offset;
    NoiseSource *noise_sources;
    uint32_t num_noise_src;
    uint32_t num_params, num_instance_params, num_opvars;
    ParamOpvar *param_opvar;
    uint32_t node_mapping_offset, jacobian_ptr_resist_offset;
    uint32_t num_states, state_idx_off;
    uint32_t bound_step_offset;
    uint32_t instance_size, model_size;
    void *(*access)(void *inst, void *model, uint32_t id, uint32_t flags);
    void (*setup_model)(void *handle, void *model, SimParas *sim_params, InitInfo *res);
    void (*setup_instance)(void *handle, void *inst, void *model, double temperature,
                           uint32_t num_terminals, SimParas *sim_params, InitInfo *res);
    uint32_t (*eval)(void *handle, void *inst, void *model, SimInfo *info);
    void (*load_noise)(void *inst, void *model, double freq, double *noise_dens);
    void (*load_residual_resist)(void *inst, void *model, double *dst);
    void (*load_residual_react)(void *inst, void *model, double *dst);
    void (*load_limit_rhs_resist)(void *inst, void *model, double *dst);
    void (*load_limit_rhs_react)(void *inst, void *model, double *dst);
    void (*load_spice_rhs_dc)(void *inst, void *model, double *dst, double *prev_solve);
    void (*load_spice_rhs_tran)(void *inst, void *model, double *dst, double *prev_solve, double alpha);
    void (*load_jacobian_resist)(void *inst, void *model);
    void (*load_jacobian_react)(void *inst, void *model, double alpha);
    void (*load_jacobian_tran)(void *inst, void *model, double alpha);
};

}  // namespace osdi
