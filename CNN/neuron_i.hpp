#pragma once

#include "simple_neuron_i.hpp"

namespace cnn {
class neuron_i : public simple_neuron_i {
public:
  ~neuron_i() override = default;

  virtual void forward() = 0;
  virtual void train() = 0;
  virtual void learn(float learning_rate) = 0;
};
} // namespace cnn
