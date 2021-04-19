#include "layer_factory_mock.hpp"

layer_factory_mock::layer_factory_mock(std::unique_ptr<net_layer_i> layer)
    : layer_(move(layer)) {}

std::unique_ptr<net_layer_i> layer_factory_mock::create(layer_i &prev_layer) {
  return move(layer_);
}
