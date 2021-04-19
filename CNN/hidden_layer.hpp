#pragma once

#include "net_layer_i.hpp"
#include "neuron_factory_i.hpp"

namespace cnn::layer {
class hidden_layer final : public net_layer_i {
public:
  hidden_layer(const tensor_shape &shape, neuron_factory_i &factory) noexcept;
  ~hidden_layer() override = default;

private:
  simple_neuron_i &operator[](size_t ordinal) noexcept override;
  size_t size() const noexcept override;
  const tensor_shape &layer_shape() const noexcept override;
  void forward() noexcept override;
  void train() noexcept override;
  void learn(float learning_rate) noexcept override;

  static std::valarray<std::unique_ptr<neuron_i>>
  create_neurons(size_t size, neuron_factory_i &factory) noexcept;

  const tensor_shape shape_;
  std::valarray<std::unique_ptr<neuron_i>> neurons_;
};
} // namespace cnn::layer