#pragma once

#include "neuron_i.hpp"

namespace cnn {

  class neuron_mock final : public neuron_i {
  public:
    explicit neuron_mock(float output = .0f);

    ~neuron_mock() override = default;

    void forward() override;

    float output() const override;

    void propagate_back(float error_gradient) override;

    void train() override;

    void learn(float learning_rate) override;

    void set_output(float out);

    bool forward_called() const;

    const float propagate_back_called_with() const;

    bool train_called() const;

    float learn_called_with() const;

  private:
    bool forward_called_ = false;
    bool train_called_ = false;
    float learn_called_with_;

    float propagate_back_called_with_;
    float output_;
  };
}
