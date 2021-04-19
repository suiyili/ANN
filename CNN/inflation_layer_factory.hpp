#pragma once

#include "layer_factory_i.hpp"

namespace cnn::net {
class inflation_layer_factory final : public layer_factory_i {
public:
  explicit inflation_layer_factory(size_t to_inflate);
  ~inflation_layer_factory() override = default;
  virtual std::unique_ptr<net_layer_i> create(layer_i &prev_layer) override;

private:
  const size_t channel_;
};
} // namespace cnn::net