#pragma once

#include <valarray>

namespace cnn {
using tensor_vector = std::valarray<size_t>;
using neuron_indices = std::valarray<size_t>;
using value_array = std::valarray<float>;
} // namespace cnn