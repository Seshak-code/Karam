#pragma once
// ============================================================================
// gui_boundary_guard.h — Compile-Time Layering Enforcer
// ============================================================================
// This header is included by core physics headers (device_physics.h,
// circuitsim.h). When ACUTESIM_GUI_BUILD is defined (set on the main AcuteSim
// executable and GUI test targets), any gui/ source file that #includes these
// physics headers directly will get an immediate, actionable compile error.
//
// CORRECT pattern: GUI accesses simulation results via DTOs
//   #include "engine_api/simulation_dto.h"
//   #include "engine_api/isimulation_driver.h"
//
// INCORRECT pattern (caught by this guard):
//   #include "physics/circuitsim.h"   // ERROR in gui/ sources
// ============================================================================

#ifdef ACUTESIM_GUI_BUILD
#  error "GUI layer must not include physics headers directly. \
Use orchestration DTOs (compute/orchestration/dto/simulation_dto.h) \
and ISimulationDriver instead."
#endif
