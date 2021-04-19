#pragma once

#include "layer_factory_i.hpp"
#include "scalar.hpp"

using namespace cnn::layer;

namespace cnn::net {
class dropout_layer_factory final : public layer_factory_i {
public:
  explicit dropout_layer_factory(const scalar &scale);
  ~dropout_layer_factory() override = default;

private:
  std::unique_ptr<net_layer_i> create(layer_i &prev_layer) override;
  const scalar &scale_;
};
} // namespace cnn::net