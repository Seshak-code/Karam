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
# Without -fvisibility=hidden:
#   - templates leak ABI across translation units
#   - internal solver symbols become link dependencies
#   - .so/.dylib transition breaks instantly
# Only ENGINE_API-annotated symbols escape into consumers.
target_compile_options(acutesim_engine_objects PRIVATE
    -fvisibility=hidden
    -fvisibility-inlines-hidden
)

# Performance flags — safe because EmulatedF64 covers IEEE-critical paths.
# Sections requiring strict IEEE compliance use ACUTESIM_PRECISE_FP_BEGIN/END.
if(NOT EMSCRIPTEN)
    target_compile_options(acutesim_engine_objects PRIVATE
        -ffast-math
        -march=native     # Enables AVX-512 / NEON auto-detection at runtime
    )
endif()

target_compile_options(acutesim_engine_objects PRIVATE
    -fno-rtti
    -fno-exceptions       # No exception overhead in hot path
)

target_compile_definitions(acutesim_engine_objects PRIVATE
    ACUTESIM_ENGINE_INTERNAL=1
)

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
# PUBLIC: only engine_api/ is visible to consumers — never acutesim_engine/ internals
# PRIVATE: internal directory is for engine code only
target_include_directories(acutesim_engine_objects
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/engine_api>
        $<INSTALL_INTERFACE:include/engine_api>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/acutesim_engine
        ${CMAKE_SOURCE_DIR}   # for transitional includes during migration
)

# ---------------------------------------------------------------------------
# GPU: Dawn/WebGPU — PRIVATE only, consumers never see GPU types
# ---------------------------------------------------------------------------
if(TARGET dawn::webgpu)
    target_link_libraries(acutesim_engine_objects PRIVATE
        dawn::webgpu
        dawn::native
    )
    target_compile_definitions(acutesim_engine_objects PRIVATE
        ACUTESIM_GPU_ENABLED=1
    )
else()
    message(STATUS "acutesim_engine: Dawn not found — GPU solver disabled")
endif()

# ---------------------------------------------------------------------------
# Source files (populated incrementally as migration phases complete)
# ---------------------------------------------------------------------------
# Phase A: infra headers (header-only, no sources yet)
# Phase B1+: source files added here per migration batch

# ---------------------------------------------------------------------------
# STATIC library wrapper
# ---------------------------------------------------------------------------
add_library(acutesim_engine STATIC
    $<TARGET_OBJECTS:acutesim_engine_objects>
)

# Re-export the PUBLIC include directories so linking targets pick them up
target_link_libraries(acutesim_engine PUBLIC acutesim_engine_objects)

# ---------------------------------------------------------------------------
# Headless CLI runner (Phase E verification)
# ---------------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cli/engine_cli_main.cpp)
    add_executable(acutesim_engine_cli cli/engine_cli_main.cpp)
    target_link_libraries(acutesim_engine_cli PRIVATE acutesim_engine)
endif()
