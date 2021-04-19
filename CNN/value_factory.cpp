#include "value_factory.hpp"

namespace cnn::net {
value_array value_factory::generate(size_t size) {
  const float mean = 1.0f / size;
  float interval[2U] = {mean - 0.5f * mean, mean + 0.5f * mean};
  value_array values(0.0f, size);

  std::mt19937 gen(seeds_());
  std::uniform_real_distribution<float> uniform(interval[0], interval[1]);

  for (auto &v : values)
    v = uniform(gen);
  return move(values);
}
} // namespace cnn::net