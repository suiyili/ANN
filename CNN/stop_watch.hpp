#pragma once

#include <chrono>

class stop_watch final {
public:
  void start();

  size_t end_in_milliseconds();
  size_t end_in_seconds();
  size_t end_in_minutes();

private:
  std::chrono::time_point<std::chrono::system_clock> start_;
};
