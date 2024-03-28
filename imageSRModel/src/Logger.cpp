#include "Logger.h"
#include <iostream>

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}
