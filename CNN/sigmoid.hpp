#pragma once

#include "hidden_neuron.hpp"

namespace cnn::neuron {
class sigmoid final : public hidden_neuron {
public:
  sigmoid(std::unique_ptr<neuron_array_i> inputs, value_factory_i &factory) noexcept;
  ~sigmoid() override = default;

  sigmoid(const sigmoid &) = delete;
  sigmoid(sigmoid &&) = delete;

#ifndef TEST
private:
#endif // !TEST
  activation_result get_result(float z) const noexcept override;
};
} // namespace cnn::neuron