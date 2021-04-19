#pragma once

#include "layer_factory_i.hpp"

using namespace cnn;

class layer_factory_mock final : public layer_factory_i {
public:
  explicit layer_factory_mock(std::unique_ptr<net_layer_i> layer);
  ~layer_factory_mock() override = default;

private:
  std::unique_ptr<net_layer_i> create(layer_i &prev_layer) override;
  std::unique_ptr<net_layer_i> layer_;
};
