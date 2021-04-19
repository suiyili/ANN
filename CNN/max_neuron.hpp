#pragma once

#include "neuron_array_i.hpp"
#include "neuron_i.hpp"
#include <memory>

namespace cnn::neuron {
class max_neuron final : public neuron_i {
public:
  explicit max_neuron(std::unique_ptr<neuron_array_i> inputs) noexcept;
  ~max_neuron() override = default;

  void forward() noexcept override;
  float output() const noexcept override;

  void propagate_back(float error_gradient) noexcept override;
  void train() noexcept override;
  void learn(float learning_rate) noexcept override;

private:
  std::unique_ptr<neuron_array_i> inputs_;
  simple_neuron_i *max_;
};
} // namespace cnn::neuron