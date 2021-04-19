#pragma once

#include "neuron_factory.hpp"
#include "pooling_neuron_type.hpp"

using namespace cnn::neuron;

namespace cnn::net {
class pooling_neuron_factory final : public neuron_factory {
public:
  explicit pooling_neuron_factory(pooling_neuron_type type);
  ~pooling_neuron_factory() override = default;
  std::unique_ptr<neuron_i> create() override;

private:
  pooling_neuron_type type_;
};
} // namespace cnn::net