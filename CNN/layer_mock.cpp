#include "layer_mock.hpp"

#include <utility>
#include "vector_allocator.hpp"

namespace cnn {
  layer_mock::layer_mock(tensor_shape  layer_shape)
    : layer_shape_(std::move(layer_shape)),
      neurons_(get_allocator().allocate<neuron_mock>(layer_shape_.volume()))
  {
    for (auto i = 0; i < layer_shape_.volume(); ++i)
      neurons_.push_back(neuron_mock());
  }

  layer_mock::layer_mock(size_t size) : layer_mock(tensor_shape(size, 1U))
  {}

  neuron_mock& layer_mock::operator[](size_t ordinal)
  {
    return neurons_[ordinal];
  }

  size_t layer_mock::size() const
  { return neurons_.size(); }

  bool layer_mock::forward_called() const
  { return forward_called_; }

  bool layer_mock::train_called() const
  { return train_called_; }

  float layer_mock::learn_called_with() const
  { return learn_called_with_; }

  void layer_mock::train()
  { train_called_ = true; }

  void layer_mock::forward()
  { forward_called_ = true; }

  void layer_mock::learn(float learning_rate)
  {
    learn_called_with_ = learning_rate;
  }

  const tensor_shape& layer_mock::layer_shape() const
  { return layer_shape_; }
}