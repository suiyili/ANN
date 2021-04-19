#pragma once

#include "simple_neuron_i.hpp"
#include "tensor_shape.hpp"

namespace cnn {
class neuron_array_i {
public:
  virtual ~neuron_array_i() = 0;

  virtual simple_neuron_i &operator[](size_t ordinal) = 0;
  virtual size_t size() const = 0;
};

inline neuron_array_i::~neuron_array_i() = default;
} // namespace cnn
