#pragma once

namespace cnn {
class simple_neuron_i {
public:
  virtual ~simple_neuron_i() = 0;

  virtual float output() const = 0;
  virtual void propagate_back(float error_gradient) = 0;
};

inline simple_neuron_i::~simple_neuron_i() = default;
} // namespace cnn
