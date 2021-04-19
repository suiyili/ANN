#ifdef TEST

#include "data_reader.hpp"
#include "epoch_indices.hpp"
#include "image_cache.hpp"
#include "label_cache.hpp"
#include "net_builder.hpp"
#include "stop_watch.hpp"

#include <iostream>

#include <catch2/catch_test_macros.hpp>

namespace cnn::net {

static void show_image(size_t label, const value_array &img) {
  std::cout << label << std::endl;
  const tensor_shape layer_shape{28, 28};
  for (size_t i = 0; i < 28; ++i) {
    for (size_t j = 0; j < 28; ++j)
      if (img[layer_shape.ordinal_of({j, i})] > 0)
        std::cout << '1';
      else
        std::cout << ' ';
    std::cout << std::endl;
  }
  std::cin.ignore();
}

static void make_net(net_builder &spider);
static void train_net(net_builder &spider);
static float verify_net(net_builder &spider);

SCENARIO("MNIST", "[acceptance]") {
  GIVEN("mnist data") {
    tensor_shape input_shape{28U, 28U};
    net_builder spider(input_shape);
    make_net(spider);
    WHEN("train the network") {
      train_net(spider);
      THEN("the accurate should be greater than .97") {
        auto ratio = verify_net(spider);
        REQUIRE(ratio > .97f);
      }
    }
  }
}

inline void make_net(net_builder &spider) {
  // input {28, 28}

  spider.add_convolution_layer({5U, 5U}, activation_type::tangent,
                               16U); //{24, 24, 16} out
  spider.add_pooling_layer({2U, 2U, 16U},
                           pooling_neuron_type::max); //{12, 12} out
  spider.add_dense_layer({20}, activation_type::tangent);
  spider.add_dropout_layer(.5f);
  spider.add_output_layer(10U);
}

inline void train_net(net_builder &spider) {
  const size_t training_batch = 16U;
  const size_t training_epoch = 59936U;
  const size_t training_round = 20U;

  const float learning_rate = .25f;

  // prepare training data
  std::cout << "reading training data..." << std::endl;
  data_reader training_label_reader("data/train-labels.idx1-ubyte");
  label_cache training_labels(training_label_reader, training_epoch);

  data_reader training_image_reader("data/train-images.idx3-ubyte");
  image_cache training_images(training_image_reader, training_epoch);

  std::cout << "start training ..." << std::endl;
  // training network
  epoch_indices epoch(training_epoch);

  stop_watch watch;
  watch.start();

  for (size_t i = 0; i < training_round; ++i) {
    stop_watch w;
    w.start();
    auto batch = epoch.generator(training_batch);
    do {
      for (size_t j = 0U; j < training_batch; ++j) {
        auto k = batch[j];
        auto label = training_labels.get_label(k);
        auto &image = training_images.get_image(k);
        //show_image(label, image);
        spider.train(image, label);
      }
      spider.learn(learning_rate);
    } while (batch.move_next());
    epoch.shuffle();

    std::cout << "training round " << i << " with " << w.end_in_seconds()
              << " seconds" << std::endl;
  }

  std::cout << "training network with " << watch.end_in_minutes() << " minutes"
            << std::endl;
}

inline float verify_net(net_builder &spider) {
  spider.switch_off_dropout();

  const size_t testing_epoch = 9000U;

  std::cout << "reading testing data..." << std::endl;
  data_reader testing_label_reader("data/t10k-labels.idx1-ubyte");
  label_cache testing_labels(testing_label_reader, testing_epoch);

  data_reader testing_image_reader("data/t10k-images.idx3-ubyte");
  image_cache testing_images(testing_image_reader, testing_epoch);

  stop_watch watch;
  // verify network
  std::cout << "start testing net..." << std::endl;
  watch.start();

  size_t correct = 0;

  for (size_t i = 0; i < testing_epoch; ++i) {
    auto label = testing_labels.get_label(i);
    auto &image = testing_images.get_image(i);
    // show_image(label, image);
    if (spider.predict(image) == label)
      ++correct;
  }
  std::cout << "testing network with " << watch.end_in_minutes() << " minutes"
            << std::endl;

  return (float)correct / testing_epoch;
}

} // namespace cnn::net

#endif // TEST