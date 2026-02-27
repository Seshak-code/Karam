#pragma once
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>

namespace acutesim {
namespace math {

/**
 * PCG32 - A minimal, portable, deterministic random number generator.
 * Guaranteed to produce the same sequence on all platforms/compilers.
 * Based on pcg-random.org
 */
class PortableRNG {
public:
    using result_type = uint32_t;

    explicit PortableRNG(uint64_t seed, uint64_t seq = 0xda3e39cb94b95bdbULL) {
        state = 0U;
        inc = (seq << 1u) | 1u;
        next();
        state += seed;
        next();
    }

    uint32_t next() {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    // Standard matching for std::distribution compatibility (if needed)
    static constexpr uint32_t min() { return 0; }
    static constexpr uint32_t max() { return 0xffffffff; }
    uint32_t operator()() { return next(); }

private:
    uint64_t state;
    uint64_t inc;
};

/**
 * PortableUniformDist - Deterministic uniform distribution.
 */
class PortableUniformDist {
public:
    PortableUniformDist(double a, double b) : m_a(a), m_b(b) {}

    double operator()(PortableRNG& rng) {
        double u = static_cast<double>(rng.next()) / 4294967296.0;
        return m_a + u * (m_b - m_a);
    }

private:
    double m_a, m_b;
};

/**
 * PortableNormalDist - Deterministic normal distribution using Box-Muller.
 */
class PortableNormalDist {
public:
    PortableNormalDist(double mean, double stddev) : m_mean(mean), m_stddev(stddev), m_hasSpare(false) {}

    double operator()(PortableRNG& rng) {
        if (m_hasSpare) {
            m_hasSpare = false;
            return m_mean + m_stddev * m_spare;
        }

        double u1 = static_cast<double>(rng.next()) / 4294967296.0;
        double u2 = static_cast<double>(rng.next()) / 4294967296.0;

        // Ensure u1 is not zero to avoid log(0)
        if (u1 < 1e-15) u1 = 1e-15;

        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * M_PI * u2;

        m_spare = r * std::sin(theta);
        m_hasSpare = true;
        return m_mean + m_stddev * r * std::cos(theta);
    }

private:
    double m_mean, m_stddev;
    double m_spare;
    bool m_hasSpare;
};

} // namespace math
} // namespace acutesim
