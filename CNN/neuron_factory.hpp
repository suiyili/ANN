#pragma once

#include "neuron_factory_i.hpp"
#include <forward_list>

namespace cnn::net {
class neuron_factory : public neuron_factory_i {
public:
  neuron_factory();
  virtual ~neuron_factory() = 0;

  void append(std::unique_ptr<neuron_array_i> proxy);

protected:
  std::forward_list<std::unique_ptr<neuron_array_i>> proxies_;

private:
  std::forward_list<std::unique_ptr<neuron_array_i>>::const_iterator cursor_;
};
} // namespace cnn::net