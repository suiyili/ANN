#pragma once

#include "activation_type.hpp"
#include "neuron_factory.hpp"
#include "value_factory_i.hpp"

using namespace cnn::neuron;

namespace cnn::net {
class activation_factory final : public neuron_factory {
public:
  activation_factory(activation_type type, value_factory_i &factory);
  ~activation_factory() override = default;

private:
  std::unique_ptr<neuron_i> create() override;

  activation_type type_;
  value_factory_i &factory_;
};
} // namespace cnn::net