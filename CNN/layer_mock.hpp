#pragma once

#include "net_layer_i.hpp"
#include "neuron_mock.hpp"
#include <vector>
#include <memory_resource>

namespace cnn {

  class layer_mock final : public net_layer_i {
  public:
    explicit layer_mock(tensor_shape  layer_shape);

    explicit layer_mock(size_t size);

    ~layer_mock() override = default;

    neuron_mock& operator[](size_t ordinal) override;

    [[nodiscard]] const tensor_shape& layer_shape() const override;

    [[nodiscard]] size_t size() const override;

    void forward() override;

    void train() override;

    void learn(float learning_rate) override;

    bool forward_called() const;

    bool train_called() const;

    float learn_called_with() const;

  private:
    const tensor_shape layer_shape_;
    std::pmr::vector<neuron_mock> neurons_;

    bool forward_called_ = false;
    bool train_called_ = false;
    float learn_called_with_ = .0f;
  };
}