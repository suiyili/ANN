#include "pooling_neuron_factory.hpp"
#include "average_neuron.hpp"
#include "max_neuron.hpp"
#include <stdexcept>

using namespace cnn::neuron;

namespace cnn::net {
pooling_neuron_factory::pooling_neuron_factory(pooling_neuron_type type)
    : type_(type) {}

std::unique_ptr<neuron_i> cnn::net::pooling_neuron_factory::create() {
  auto proxy = move(proxies_.front());
  proxies_.pop_front();
  switch (type_) {
  case cnn::neuron::pooling_neuron_type::max:
    return make_unique<max_neuron>(move(proxy));
  case cnn::neuron::pooling_neuron_type::average:
    return make_unique<average_neuron>(move(proxy));
  default:
    throw std::invalid_argument("pooling neuron type is not given");
  }
}
} // namespace cnn::net