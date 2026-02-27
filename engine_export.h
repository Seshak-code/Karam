#pragma once
// ============================================================================
// engine_export.h — Symbol Visibility Discipline
// ============================================================================
// Controls which symbols escape the engine library boundary.
//
// With -fvisibility=hidden (set in CMakeLists.txt):
//   - ALL symbols are hidden by default
//   - Only ENGINE_API-annotated symbols are visible to consumers
//   - Templates and internal solver symbols stay internal
//   - This is the prerequisite for future .so/.dylib extraction
//
// Usage:
//   class ENGINE_API ISimulationEngine { ... };          // Visible
//   class ENGINE_INTERNAL CircuitSim { ... };            // Hidden
//   ENGINE_API std::unique_ptr<ISimulationEngine> create_engine();
// ============================================================================

#ifdef ACUTESIM_ENGINE_INTERNAL
    // Building the engine itself
    #if defined(__GNUC__) || defined(__clang__)
        #define ENGINE_API      __attribute__((visibility("default")))
        #define ENGINE_INTERNAL __attribute__((visibility("hidden")))
    #elif defined(_MSC_VER)
        #define ENGINE_API      __declspec(dllexport)
        #define ENGINE_INTERNAL
    #else
        #define ENGINE_API
        #define ENGINE_INTERNAL
    #endif
#else
    // Consuming the engine from outside
    #if defined(_MSC_VER)
        #define ENGINE_API      __declspec(dllimport)
    #else
        #define ENGINE_API
    #endif
    #define ENGINE_INTERNAL   // should never be visible outside anyway
#endif
