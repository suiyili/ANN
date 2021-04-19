#pragma once

#include "value_factory_i.hpp"
#include <random>

namespace cnn::net {
class value_factory final : public value_factory_i {
public:
  virtual value_array generate(size_t size) override;

private:
  static constexpr float total_ = 1.0f;
  std::random_device seeds_;
};
} // namespace cnn::net
