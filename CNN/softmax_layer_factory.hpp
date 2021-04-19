#pragma once

#include "layer_factory_i.hpp"

namespace cnn::net {
class softmax_layer_factory final : public layer_factory_i {
public:
  ~softmax_layer_factory() override = default;

private:
  std::unique_ptr<net_layer_i> create(layer_i &prev_layer) override;
};
} // namespace cnn::net