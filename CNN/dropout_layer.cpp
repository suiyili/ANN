#include "dropout_layer.hpp"
#include "vector_allocator.hpp"

namespace cnn::layer {
dropout_layer::dropout_layer(layer_i &prev_layer, const scalar &scale) noexcept
    : prev_layer_(prev_layer), scale_(scale),
      scalars_(init_scalars(prev_layer_)), dist_(0U, scalars_.size() - 1U),
      scalar_table_(scalars_.size()) {}

size_t dropout_layer::size() const noexcept { return scalars_.size(); }

simple_neuron_i &dropout_layer::operator[](size_t ordinal) noexcept {
  return scalars_[ordinal];
}

const tensor_shape &dropout_layer::layer_shape() const noexcept {
  return prev_layer_.layer_shape();
}

void dropout_layer::forward() noexcept {}

void dropout_layer::train() noexcept {}

void dropout_layer::learn(float) noexcept {
  auto total = scalars_.size();
  const auto scalar = (float)scale_;
  set_scalars(scale_.scaled(total));
  for (size_t i = 0U; i < total; ++i)
    scalars_[i].scale(scalar_table_[i] ? scalar : 0.0f);
}

std::pmr::vector<scalar_neuron> dropout_layer::init_scalars(layer_i &prev) noexcept {
  auto &allocator = get_allocator();
  auto scalars = allocator.allocate<scalar_neuron>(prev.size());

  for (size_t i = 0; i < prev.size(); ++i)
    scalars.emplace_back(prev[i]);
  return scalars;
}

void dropout_layer::set_scalars(size_t number_of_scalars) noexcept {
  if (number_of_scalars == scalar_table_.size())
    scalar_table_ = true;
  else {
    scalar_table_ = false;
    std::mt19937 gen(rd_());
    while (number_of_scalars > 0U) {
      auto i = dist_(gen);
      while (scalar_table_[i])
        i = (++i) % scalar_table_.size();
      scalar_table_[i] = true;

      --number_of_scalars;
    }
  }
}

} // namespace cnn::layer