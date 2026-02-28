cmake_minimum_required(VERSION 3.20)

# ===========================================================================
# acutesim_engine — Tensor MNA Simulation Library
# ===========================================================================
# This is the standalone engine library. ALL simulation, GPU, and physics
# logic lives here. The rest of AcuteSim is a consumer via engine_api/.
#
# Build pattern: OBJECT → STATIC (future: SHARED/PLUGIN without recompilation)
# ===========================================================================

# ---------------------------------------------------------------------------
# OBJECT target — compiled once, wrapped in STATIC (or SHARED later)
# ---------------------------------------------------------------------------
add_library(acutesim_engine_objects OBJECT)

# === HARDENING: Symbol Visibility Discipline ===
target_compile_options(acutesim_engine_objects PRIVATE
    -fvisibility=hidden
    -fvisibility-inlines-hidden
)

if(NOT EMSCRIPTEN)
    target_compile_options(acutesim_engine_objects PRIVATE
        -ffast-math
        -march=native     # Enables AVX-512 / NEON auto-detection at runtime
    )
endif()

# Enable exceptions and RTTI for legacy solver logic
set_target_properties(acutesim_engine_objects PROPERTIES
    COMPILE_FLAGS "-O3 -march=native"
)

target_compile_definitions(acutesim_engine_objects PRIVATE
    ACUTESIM_ENGINE_INTERNAL=1
)

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
target_include_directories(acutesim_engine_objects
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/engine_api>
        $<INSTALL_INTERFACE:include/engine_api>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/acutesim_engine
        ${CMAKE_SOURCE_DIR}   # for transitional includes
)

# ---------------------------------------------------------------------------
# GPU: Dawn/WebGPU
# ---------------------------------------------------------------------------
if(NOT EMSCRIPTEN AND TARGET dawn_native)
    target_include_directories(acutesim_engine_objects PRIVATE
        ${DAWN_INTERNAL_INCLUDES}
    )
    target_link_libraries(acutesim_engine_objects PRIVATE
        dawn_native
        dawn_proc
        webgpu_cpp
    )
    target_compile_definitions(acutesim_engine_objects PRIVATE
        ACUTESIM_GPU_ENABLED=1
        ACUTESIM_HAS_DAWN_NATIVE=1
        ACUTESIM_GPU_PROBE_AVAILABLE=1
    )
endif()

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------

target_sources(acutesim_engine_objects PRIVATE
    # === Façade (engine-api boundary) ===
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_context_manager.cpp

    # === Infrastructure ===
    ${CMAKE_CURRENT_SOURCE_DIR}/infrastructure/netlist_compiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/infrastructure/netlist_compiler_ctx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/infrastructure/postproc_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/infrastructure/hardware_capabilities.cpp

    # === Solvers ===
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/circuitsim_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/ac_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/fourier_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/pz_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/tf_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/superposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/register_analyses.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/webgpu_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/analysis_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/staged_dispatcher.cpp

    # === Tensors (SIMD kernels) ===
    ${CMAKE_CURRENT_SOURCE_DIR}/tensors/kernel_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensors/physics_tensors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensors/physics_kernels_avx2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensors/physics_kernels_avx512.cpp

    # === Netlist ===
    ${CMAKE_CURRENT_SOURCE_DIR}/netlist/netlist_export.cpp

    # === Physics (Verilog-A model compiler pipeline) ===
    ${CMAKE_CURRENT_SOURCE_DIR}/physics/quick_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physics/veriloga/lexer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physics/veriloga/parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physics/veriloga/ast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physics/veriloga/analyzer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physics/veriloga/graph_assembler.cpp
)

# AVX-512 source flags
if(NOT EMSCRIPTEN)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)")
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/tensors/physics_kernels_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f"
        )
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/tensors/physics_kernels_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2"
        )
    endif()
endif()

# ---------------------------------------------------------------------------
# STATIC library wrapper
# ---------------------------------------------------------------------------
add_library(acutesim_engine STATIC
    $<TARGET_OBJECTS:acutesim_engine_objects>
)

target_link_libraries(acutesim_engine PUBLIC acutesim_engine_objects)

if(NOT EMSCRIPTEN AND TARGET dawn_native)
    target_link_libraries(acutesim_engine PUBLIC
        dawn_native
        dawn_proc
        webgpu_cpp
    )
endif()

# ---------------------------------------------------------------------------
# CLI — Engine Consumer
# ---------------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cli/engine_cli_main.cpp)
    add_executable(acutesim_engine_cli cli/engine_cli_main.cpp)
    target_link_libraries(acutesim_engine_cli PRIVATE acutesim_sim_layer acutesim_orchestration acutesim_engine)
endif()
