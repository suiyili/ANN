#include "end_io_net.hpp"

namespace rnn {
  end_io_net::end_io_net(size_t label_size)
  {
    labels_.reserve(label_size);
  }
  void end_io_net::train(const std::vector<size_t>& input, size_t label) {
  labels_.push_back(label);
}
size_t end_io_net::predict(const std::vector<size_t>& input) {
  return 0;
}
}  // namespace rnn