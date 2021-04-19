#pragma once

#include "net_layer_i.hpp"
#include <memory>

namespace cnn {
class layer_factory_i {
public:
  virtual ~layer_factory_i() = 0;

  virtual std::unique_ptr<net_layer_i> create(layer_i &prev_layer) = 0;
};

inline layer_factory_i::~layer_factory_i() = default;
} // namespace cnn