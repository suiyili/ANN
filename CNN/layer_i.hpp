#pragma once

#include "neuron_array_i.hpp"

namespace cnn {
class layer_i : public neuron_array_i {
public:
  ~layer_i() override = default;

  virtual const tensor_shape &layer_shape() const = 0;
};
} // namespace cnn