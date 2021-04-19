#pragma once

#include <vector>

namespace rnn {
class end_io_net final{

public:
  explicit end_io_net(size_t label_size = 1024);
  void train(const std::vector<size_t>& input, size_t label);
  size_t predict(const std::vector<size_t>& input);
private:
  std::vector<size_t> labels_;
};

}  // namespace rnn