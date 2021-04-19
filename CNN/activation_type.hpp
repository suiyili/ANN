#pragma once

namespace cnn::neuron {
enum class activation_type : char {
  sigmoid = 's',
  tangent = 't',
  relu = 'r',
  hardtanh = 'h',
  identity = 'i'
};
}
