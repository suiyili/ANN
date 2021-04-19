#pragma once

#include "batch_iterator.hpp"

namespace cnn {
class epoch_indices final {
public:
  epoch_indices(size_t epoch_size);

  epoch_indices(const epoch_indices &) = delete;
  epoch_indices &operator=(const epoch_indices &) = delete;

  batch_iterator generator(size_t batch_size);
  void shuffle();

private:
  static std::valarray<size_t> init_indices(size_t count);
  static void shuffle_indices(std::valarray<size_t> &indices);

  std::valarray<size_t> indices_;
};
} // namespace cnn