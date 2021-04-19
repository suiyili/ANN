#include "net_builder.hpp"
#include "activation_factory.hpp"
#include "convolution_layer_factory.hpp"
#include "dense_layer_factory.hpp"
#include "dropout_layer_factory.hpp"
#include "inflation_layer_factory.hpp"
#include "pooling_layer_factory.hpp"
#include "pooling_neuron_factory.hpp"
#include "softmax_layer_factory.hpp"

namespace cnn::net {
net_builder::net_builder(const tensor_shape &input_shape)
    : net_(input_shape) {}

void net_builder::add_dense_layer(const tensor_shape &layer_shape,
                                  activation_type type) {
  activation_factory neuron_factory(type, factory_);
  dense_layer_factory layer_factory(layer_shape, neuron_factory);
  net_.create_layer(layer_factory);
}

void net_builder::add_convolution_layer(const tensor_shape &feature_shape,
                                        activation_type type,
                                        unsigned short channel) {
  activation_factory neuron_factory(type, factory_);
  convolution_layer_factory layer_factory(feature_shape, neuron_factory);
  net_.create_layer(layer_factory);

  if (channel > 1U) {
    inflation_layer_factory channel_factory(channel);
    net_.create_layer(channel_factory);
  }
}

void net_builder::add_pooling_layer(const tensor_shape &feature_shape,
                                    pooling_neuron_type type) {
  pooling_neuron_factory neuron_factory(type);
  pooling_layer_factory layer_factory(feature_shape, neuron_factory);
  net_.create_layer(layer_factory);
}

void net_builder::add_dropout_layer(float dropout_rate) {
  scale_.dropout(dropout_rate);
  dropout_layer_factory factory(scale_);
  net_.create_layer(factory);
}

void net_builder::add_output_layer(size_t category) {
  const tensor_shape category_shape{category};

  activation_factory neuron_factory(activation_type::identity, factory_);
  dense_layer_factory layer_factory(category_shape, neuron_factory);
  net_.create_layer(layer_factory);

  softmax_layer_factory output_layer_factory;
  net_.create_layer(output_layer_factory);
}

void net_builder::train(const value_array &input, size_t category) {
  net_.train(input, category);
}

size_t net_builder::predict(const value_array &input) {
  return net_.predict(input);
}

void net_builder::learn(float learning_rate) { net_.learn(learning_rate); }

void net_builder::switch_off_dropout() {
  scale_.dropout(0.0f);
  net_.learn(.0f);
}

} // namespace cnn::net