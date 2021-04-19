#pragma once

#include "hidden_neuron.hpp"

namespace cnn::neuron {
class identity final : public hidden_neuron {
public:
  identity(std::unique_ptr<neuron_array_i> inputs, value_factory_i &factory) noexcept;
  ~identity() override = default;

  identity(const identity &) = delete;
  identity(identity &&) = delete;

#ifndef TEST
private:
#endif // !TEST
  activation_result get_result(float z) const noexcept override;
};
} // namespace cnn::neuron