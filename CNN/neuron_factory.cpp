#include "neuron_factory.hpp"

namespace cnn::net {
neuron_factory::neuron_factory() : cursor_(proxies_.cbefore_begin()) {}

neuron_factory::~neuron_factory() {}

void neuron_factory::append(std::unique_ptr<neuron_array_i> proxy) {
  proxies_.insert_after(cursor_, std::move(proxy));
  ++cursor_;
}
} // namespace cnn::net
