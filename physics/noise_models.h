#pragma once

#include <cmath>

/**
 * noise_models.h - Stateless Noise Power Spectral Density Functions
 *
 * All functions return noise PSD in appropriate units (A^2/Hz or V^2/Hz).
 * These are pure functions with no side effects, following flow.md rules.
 *
 * Physical constants:
 *   k_B = 1.380649e-23 J/K (Boltzmann constant)
 *   q   = 1.602176634e-19 C (electron charge)
 */

namespace NoiseModels {

constexpr double k_B = 1.380649e-23;   // Boltzmann constant [J/K]
constexpr double q_e = 1.602176634e-19; // Electron charge [C]

// ============================================================================
// RESISTOR NOISE
// ============================================================================

/**
 * Thermal (Johnson-Nyquist) noise current PSD through a resistor.
 * S_I(f) = 4kT/R [A^2/Hz]
 *
 * @param R       Resistance [Ohm]
 * @param temp_K  Temperature [K]
 * @return Current noise PSD [A^2/Hz]
 */
inline double resistor_thermal_current_psd(double R, double temp_K) {
    if (R <= 0.0) return 0.0;
    return 4.0 * k_B * temp_K / R;
}

/**
 * Thermal (Johnson-Nyquist) noise voltage PSD across a resistor.
 * S_V(f) = 4kTR [V^2/Hz]
 *
 * @param R       Resistance [Ohm]
 * @param temp_K  Temperature [K]
 * @return Voltage noise PSD [V^2/Hz]
 */
inline double resistor_thermal_voltage_psd(double R, double temp_K) {
    if (R <= 0.0) return 0.0;
    return 4.0 * k_B * temp_K * R;
}

// ============================================================================
// DIODE NOISE
// ============================================================================

/**
 * Shot noise PSD for a diode junction.
 * S_I(f) = 2qI [A^2/Hz]
 *
 * @param I_dc  DC bias current [A] (absolute value used)
 * @return Current noise PSD [A^2/Hz]
 */
inline double diode_shot_noise_psd(double I_dc) {
    return 2.0 * q_e * std::abs(I_dc);
}

// ============================================================================
// BJT NOISE
// ============================================================================

/**
 * Collector shot noise PSD.
 * S_Ic(f) = 2qI_c [A^2/Hz]
 */
inline double bjt_collector_shot_psd(double I_c) {
    return 2.0 * q_e * std::abs(I_c);
}

/**
 * Base shot noise PSD.
 * S_Ib(f) = 2qI_b [A^2/Hz]
 */
inline double bjt_base_shot_psd(double I_b) {
    return 2.0 * q_e * std::abs(I_b);
}

/**
 * Base spreading resistance thermal noise PSD.
 * S_V(f) = 4kT * r_bb [V^2/Hz]
 *
 * @param r_bb    Base spreading resistance [Ohm]
 * @param temp_K  Temperature [K]
 */
inline double bjt_base_thermal_psd(double r_bb, double temp_K) {
    if (r_bb <= 0.0) return 0.0;
    return 4.0 * k_B * temp_K * r_bb;
}

// ============================================================================
// MOSFET NOISE
// ============================================================================

/**
 * Channel thermal noise PSD (drain current noise).
 * S_Id(f) = 4kT * (2/3) * gm [A^2/Hz]
 *
 * Uses the long-channel thermal noise model with gamma = 2/3.
 *
 * @param gm      Transconductance [S]
 * @param temp_K  Temperature [K]
 */
inline double mosfet_channel_thermal_psd(double gm, double temp_K) {
    if (gm <= 0.0) return 0.0;
    return 4.0 * k_B * temp_K * (2.0 / 3.0) * gm;
}

/**
 * MOSFET flicker (1/f) noise PSD.
 * S_Id(f) = KF * I_d^AF / (C_ox * W * L * f) [A^2/Hz]
 *
 * @param KF    Flicker noise coefficient
 * @param I_d   Drain current [A] (absolute value used)
 * @param AF    Flicker noise exponent (typically 1.0)
 * @param C_ox  Gate oxide capacitance per unit area [F/m^2]
 * @param W     Channel width [m]
 * @param L     Channel length [m]
 * @param freq  Frequency [Hz]
 */
inline double mosfet_flicker_psd(double KF, double I_d, double AF,
                                  double C_ox, double W, double L,
                                  double freq) {
    if (KF <= 0.0 || C_ox <= 0.0 || W <= 0.0 || L <= 0.0 || freq <= 0.0)
        return 0.0;
    return KF * std::pow(std::abs(I_d), AF) / (C_ox * W * L * freq);
}

} // namespace NoiseModels
