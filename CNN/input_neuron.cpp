#include "input_neuron.hpp"

namespace cnn::neuron {
input_neuron::input_neuron(const value_array *&source, size_t ordinal) noexcept
    : source_(source), ordinal_(ordinal) {}

float input_neuron::output() const noexcept { return (*source_)[ordinal_]; }

void input_neuron::propagate_back(float error_gradient) noexcept {}
} // namespace cnn::neuron