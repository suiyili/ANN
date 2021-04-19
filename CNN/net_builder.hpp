#pragma once

#include "activation_type.hpp"
#include "network.hpp"
#include "pooling_neuron_type.hpp"
#include "scalar.hpp"
#include "value_factory.hpp"

using namespace cnn::neuron;
using namespace cnn::layer;

namespace cnn::net {
class net_builder {
public:
  explicit net_builder(const tensor_shape &input_shape);

  void add_dense_layer(const tensor_shape &layer_shape, activation_type type);
  void add_convolution_layer(const tensor_shape &feature_shape,
                             activation_type type, unsigned short channel = 1U);
  void add_pooling_layer(const tensor_shape &feature_shape,
                         pooling_neuron_type type);
  void add_dropout_layer(float dropout_rate);
  void add_output_layer(size_t category);

  void train(const value_array &input, size_t category);
  size_t predict(const value_array &input);
  void learn(float learning_rate);
  void switch_off_dropout();

private:
  value_factory factory_;

  scalar scale_;
  network net_;
};
} // namespace cnn::net