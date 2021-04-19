#pragma once

#include "dnn_type.hpp"

namespace cnn {
class value_factory_i {
public:
  virtual ~value_factory_i() = 0;
  virtual value_array generate(size_t size) = 0;
};

inline value_factory_i::~value_factory_i() = default;
} // namespace cnn