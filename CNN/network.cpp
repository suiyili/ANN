#include "network.hpp"

namespace cnn::net {
network::network(const tensor_shape &shape) : source_(shape) {}

void network::create_layer(layer_factory_i &factory) {
  auto &prev_layer = layers_.empty() ? source_ : *layers_.back();
  layers_.push_back(factory.create(prev_layer));
}

size_t network::predict(const value_array &values) noexcept {
  source_.read(values);
  forward(layers_);

  decltype(auto) last_layer = *layers_.back();
  size_t highest = 0;
  for (size_t i = 1U; i < last_layer.size(); ++i)
    if (last_layer[i].output() > last_layer[highest].output())
      highest = i;
  return highest;
}

void network::train(const value_array &values, size_t label) noexcept {
  source_.read(values);
  forward(layers_);

  decltype(auto) last_layer = *layers_.back();
  for (size_t i = 0U; i < last_layer.size(); ++i)
    last_layer[i].propagate_back(label == i ? 1.0f : 0.0f);

  for (auto layer = layers_.rbegin(); layer != layers_.rend(); ++layer)
    (*layer)->train();
}
void network::learn(float learning_rate) noexcept {
  for (auto &layer : layers_)
    layer->learn(learning_rate);
}

void network::forward(std::list<std::unique_ptr<net_layer_i>> &layers) noexcept {
  for (auto &layer : layers)
    layer->forward();
}
} // namespace cnn::net