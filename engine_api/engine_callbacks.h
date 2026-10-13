#pragma once
#include <functional>
#include <string>

namespace acutesim {

enum class LogLevel { DEBUG, INFO, WARN, ERROR };

struct EngineCallbacks {
    std::function<void(float)>                  progress        = [](float){};
    std::function<bool()>                       cancelRequested = []{ return false; };
    std::function<void(LogLevel, std::string)>  log             = [](LogLevel, std::string){};

    // Fine-grained simulation hooks
    std::function<void(int, double)>            onConvergenceStep = [](int, double){};
    std::function<void(double, double)>         onTimeStep        = [](double, double){};
    std::function<void(int, int)>               onProgress        = [](int, int){};
};

} // namespace acutesim
