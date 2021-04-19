
#include "neuron_mock.hpp"

namespace cnn {
  neuron_mock::neuron_mock(float output)
    : output_(output)
  {}

  void neuron_mock::forward()
  {
    forward_called_ = true;
  }

  float neuron_mock::output() const
  {
    return output_;
  }

  void neuron_mock::propagate_back(float error_gradient)
  {
    propagate_back_called_with_ = error_gradient;
  }

  void neuron_mock::train()
  {
    train_called_ = true;
  }

  void neuron_mock::learn(float learning_rate)
  {
    learn_called_with_ = learning_rate;
  }

  void neuron_mock::set_output(float out)
  {
    output_ = out;
  }

  bool neuron_mock::forward_called() const
  {
    return forward_called_;
  }

  bool neuron_mock::train_called() const
  {
    return train_called_;
  }

  const float neuron_mock::propagate_back_called_with() const
  {
    return propagate_back_called_with_;
  }

  float neuron_mock::learn_called_with() const
  {
    return learn_called_with_;
  }
}