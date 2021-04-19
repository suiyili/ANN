#pragma once

#include "activation.hpp"

using namespace cnn;
using namespace cnn::neuron;

class activation_mock final : public activation {
public:
  activation_mock(std::unique_ptr<neuron_array_i> inputs,
                  value_factory_i &factory);
  ~activation_mock() override = default;

  activation_mock(const activation_mock &) = delete;
  activation_mock(activation_mock &&) = delete;

  float get_argument() const;
  float get_back_prop() const;

  float output() const override;
  void set_argument(float z) override;
  float get_error_gradient(float back_prop) override;

private:
  float z_;
  float prop_;
};
