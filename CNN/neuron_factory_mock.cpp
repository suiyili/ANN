#include "neuron_factory_mock.hpp"
#include "vector_allocator.hpp"
#include "vector_allocator.hpp"

neuron_factory_mock::neuron_factory_mock() 
:neurons_(get_allocator().allocate<neuron_mock*>(2)){}

std::unique_ptr<neuron_i> neuron_factory_mock::create() {
  auto n = std::make_unique<neuron_mock>();
  neurons_.push_back(n.get());
  return move(n);
}

neuron_mock &neuron_factory_mock::get_neuron(size_t ordinal) {
  return *neurons_[ordinal];
}
