#pragma once

#include "hidden_neuron.hpp"

using namespace cnn;
using namespace cnn::neuron;

class hidden_neuron_mock : public hidden_neuron {
public:
  hidden_neuron_mock(std::unique_ptr<neuron_array_i> inputs,
                     value_factory_i &factory);
  ~hidden_neuron_mock() override = default;

  activation_result get_result(const float z) const override;

  void set_output(float output);
  void set_derivative(float derivative);
  float call_get_result_with() const;

private:
  mutable float z_;
  mutable float out_;
  float output_;
  float derivative_;
};