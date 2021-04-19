#pragma once

#include "neuron_array_i.hpp"
#include "neuron_i.hpp"
#include <memory>

namespace cnn {
class neuron_factory_i {
public:
  virtual ~neuron_factory_i() = 0;

  virtual std::unique_ptr<neuron_i> create() = 0;
};

inline neuron_factory_i::~neuron_factory_i() = default;
} // namespace cnn
