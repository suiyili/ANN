#include "activation_factory.hpp"
#include "hard_tanh.hpp"
#include "identity.hpp"
#include "max_neuron.hpp"
#include "relu.hpp"
#include "sigmoid.hpp"
#include "tangent.hpp"
#include <stdexcept>

using namespace cnn::neuron;

namespace cnn::net {
activation_factory::activation_factory(activation_type type,
                                       value_factory_i &factory)
    : type_(type), factory_(factory) {}

std::unique_ptr<neuron_i> activation_factory::create() {
  auto proxy = move(proxies_.front());
  proxies_.pop_front();

  switch (type_) {
  case activation_type::sigmoid:
    return make_unique<sigmoid>(move(proxy), factory_);
  case activation_type::tangent:
    return make_unique<tangent>(move(proxy), factory_);
  case activation_type::relu:
    return make_unique<relu>(move(proxy), factory_);
  case activation_type::hardtanh:
    return make_unique<hard_tanh>(move(proxy), factory_);
  case activation_type::identity:
    return make_unique<identity>(move(proxy), factory_);
  default:
    throw std::invalid_argument("neuron type not found");
  }
}

} // namespace cnn::net
