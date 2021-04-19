#pragma once

#include "hidden_neuron.hpp"

namespace cnn::neuron {
class hard_tanh final : public hidden_neuron {
public:
  hard_tanh(std::unique_ptr<neuron_array_i> inputs,
            value_factory_i &factory) noexcept;
  ~hard_tanh() override = default;

  hard_tanh(const hard_tanh &) = delete;
  hard_tanh(hard_tanh &&) = delete;

#ifndef TEST
private:
#endif // !TEST

  activation_result get_result(float z) const noexcept override;
};
} // namespace cnn::neuron