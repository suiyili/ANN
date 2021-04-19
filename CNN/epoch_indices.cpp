#include "epoch_indices.hpp"
#include <random>
#include <numeric>
#include <stdexcept>

namespace cnn {
epoch_indices::epoch_indices(size_t epoch_size)
    : indices_(init_indices(epoch_size)) {}

batch_iterator epoch_indices::generator(size_t batch_size) {
  return batch_iterator{indices_, batch_size};
}

void epoch_indices::shuffle() {
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(begin(indices_), end(indices_), g);
}

std::valarray<size_t> epoch_indices::init_indices(size_t count) {
  if (count == 0)
    throw std::invalid_argument("no record");

  std::valarray<size_t> indices((size_t)0U, count);
  std::iota(begin(indices), end(indices), 0U);
  return indices;
}

} // namespace cnn