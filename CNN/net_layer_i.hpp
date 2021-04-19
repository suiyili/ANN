#pragma once

#include "layer_i.hpp"

namespace cnn {
class net_layer_i : public layer_i {
public:
  ~net_layer_i() override = default;

  virtual void forward() = 0;
  virtual void train() = 0;
  virtual void learn(float learning_rate) = 0;
};
} // namespace cnn
