#pragma once
/**
 * safe_utils.h
 * 
 * NASA/JPL-Style Safety Utilities for High-Reliability Code.
 * Provides bounds-checked accessors and defensive assertions.
 */

#include <vector>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <string>

namespace SafeUtils {

// =============================================================================
// BOUNDS-CHECKED ACCESSORS
// =============================================================================

/**
 * safe_at - Bounds-checked vector access with informative error.
 * 
 * In DEBUG builds: Asserts on out-of-bounds.
 * In RELEASE builds: Throws std::out_of_range.
 */
template<typename T>
inline T& safe_at(std::vector<T>& vec, size_t index, const char* context = "") {
#ifndef NDEBUG
    assert(index < vec.size() && "safe_at: Index out of bounds");
#endif
    if (index >= vec.size()) {
        throw std::out_of_range(
            std::string("safe_at: Index ") + std::to_string(index) + 
            " out of bounds (size=" + std::to_string(vec.size()) + ") " + context
        );
    }
    return vec[index];
}

template<typename T>
inline const T& safe_at(const std::vector<T>& vec, size_t index, const char* context = "") {
#ifndef NDEBUG
    assert(index < vec.size() && "safe_at: Index out of bounds");
#endif
    if (index >= vec.size()) {
        throw std::out_of_range(
            std::string("safe_at: Index ") + std::to_string(index) + 
            " out of bounds (size=" + std::to_string(vec.size()) + ") " + context
        );
    }
    return vec[index];
}

/**
 * safe_idx - Validate 1-based node index and convert to 0-based.
 * Returns -1 if index is ground (0) or invalid.
 * 
 * Convention: NodeIndex 0 = Ground (not in solution vector).
 *             NodeIndex 1..N = Solution vector indices 0..N-1.
 */
inline int safe_idx(int nodeIndex, size_t vectorSize, const char* context = "") {
    if (nodeIndex <= 0) return -1; // Ground or invalid
    size_t zeroBasedIdx = static_cast<size_t>(nodeIndex - 1);
    if (zeroBasedIdx >= vectorSize) {
#ifndef NDEBUG
        assert(false && "safe_idx: Node index exceeds vector size");
#endif
        throw std::out_of_range(
            std::string("safe_idx: Node ") + std::to_string(nodeIndex) + 
            " exceeds vector size " + std::to_string(vectorSize) + " " + context
        );
    }
    return static_cast<int>(zeroBasedIdx);
}

// =============================================================================
// NaN/INF GUARDS
// =============================================================================

/**
 * sanitize_value - Replace NaN/Inf with a safe fallback.
 */
inline double sanitize_value(double v, double fallback = 0.0) {
    if (std::isnan(v) || std::isinf(v)) return fallback;
    return v;
}

/**
 * sanitize_vector - Sanitize all elements in a vector.
 * Returns the count of sanitized elements.
 */
inline int sanitize_vector(std::vector<double>& vec, double fallback = 0.0) {
    int count = 0;
    for (double& v : vec) {
        if (std::isnan(v) || std::isinf(v)) {
            v = fallback;
            ++count;
        }
    }
    return count;
}

// =============================================================================
// DEFENSIVE MACROS
// =============================================================================

/**
 * SAFE_ASSERT - Like assert, but logs context before aborting.
 * Use for invariants that should NEVER be violated in correct code.
 */
#define SAFE_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "[FATAL] SAFE_ASSERT failed: " << (message) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            assert(false); \
        } \
    } while(0)

/**
 * SAFE_FAIL - Unconditional failure for explicitly unreachable code paths.
 * Use in default switch cases, after exhaustive enums, or where logic errors
 * indicate a programming bug rather than a runtime condition.
 */
#define SAFE_FAIL(message) \
    do { \
        std::cerr << "[FATAL] SAFE_FAIL: " << (message) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        assert(false && "Unreachable code reached"); \
    } while(0)

} // namespace SafeUtils
