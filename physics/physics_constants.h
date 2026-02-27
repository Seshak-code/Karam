#pragma once
/**
 * physics_constants.h
 * 
 * Centralized Physical Constants for SPICE-Class Simulation.
 * Eliminates magic numbers scattered throughout the solver.
 */

namespace PhysicsConstants {

// =============================================================================
// THERMAL & FUNDAMENTAL
// =============================================================================

/** Thermal voltage at 300K (room temperature): kT/q */
constexpr double V_THERMAL_300K = 0.02585; // Volts

/** Boltzmann constant */
constexpr double K_BOLTZMANN = 1.380649e-23; // J/K

/** Electron charge */
constexpr double Q_ELECTRON = 1.602176634e-19; // Coulombs

// =============================================================================
// DEVICE DEFAULTS (Silicon)
// =============================================================================

/** Default diode saturation current (Silicon PN Junction) */
constexpr double IS_DEFAULT_DIODE = 1e-14; // Amperes

/** Default emission coefficient (Ideal diode) */
constexpr double N_DEFAULT_DIODE = 1.0;

/** Default BJT saturation current */
constexpr double IS_DEFAULT_BJT = 1e-14; // Amperes

/** Default BJT forward beta */
constexpr double BETA_F_DEFAULT = 100.0;

/** Default BJT reverse beta */
constexpr double BETA_R_DEFAULT = 1.0;

/** Default Schottky diode Is (higher than PN) */
constexpr double IS_DEFAULT_SCHOTTKY = 200e-9; // Amperes

// =============================================================================
// SOLVER HEURISTICS
// =============================================================================

/** Diode forward voltage seed for initial guess */
constexpr double V_DIODE_SEED = 0.6; // Volts

/** BJT Vbe seed for initial guess */
constexpr double V_BE_SEED = 0.7; // Volts

/** Default supply voltage assumption (if none found) */
constexpr double V_SUPPLY_DEFAULT = 5.0; // Volts

/** Voltage source internal resistance (Norton equivalent) */
constexpr double R_VSOURCE_INTERNAL = 1e-3; // Ohms

// =============================================================================
// TOLERANCES
// =============================================================================

/** Relative tolerance for Newton-Raphson convergence */
constexpr double RELTOL = 1e-3;

/** Absolute voltage tolerance */
constexpr double VNTOL = 1e-6; // Volts

/** Absolute current tolerance (pico-Ampere resolution) */
constexpr double ABSTOL = 1e-12; // Amperes

/** KCL residual tolerance */
constexpr double KCL_TOL = 1e-12; // Amperes

/** PCG solver tolerance */
constexpr double PCG_TOL = 1e-10;

/** Minimum time step to prevent underflow */
constexpr double DT_MIN = 1e-18; // Seconds

/** Maximum NR iterations */
constexpr int MAX_NR_ITER = 50;

/** Maximum PCG iterations */
constexpr int MAX_PCG_ITER = 1000;

/** GMIN stepping stages */
constexpr int GMIN_STEPS = 10;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/** Compute thermal voltage at a given temperature */
inline double thermalVoltage(double tempK) {
    return (K_BOLTZMANN * tempK) / Q_ELECTRON;
}

} // namespace PhysicsConstants
