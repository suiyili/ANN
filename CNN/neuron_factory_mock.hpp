#pragma once
#include "neuron_factory_i.hpp"
#include "neuron_mock.hpp"
#include <vector>
#include <memory_resource>

using namespace cnn;

class neuron_factory_mock final : public neuron_factory_i {
public:
  neuron_factory_mock();
  virtual ~neuron_factory_mock() override = default;

  virtual std::unique_ptr<neuron_i> create() override;

  neuron_mock &get_neuron(size_t ordinal = 0);

private:
  std::pmr::vector<neuron_mock *> neurons_;
};