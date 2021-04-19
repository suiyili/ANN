cmake_minimum_required(VERSION 3.12)
project(CNN)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_PREFIX_PATH  ../Catch2)
add_compile_definitions(TEST)

find_package (Threads)
find_package(Catch2 REQUIRED)

set(CNN_SRC
  activation.cpp
  layer_factory_mock.cpp
  activation_factory.cpp
  activation_mock.cpp
  activation_test.cpp
  average_neuron.cpp
  average_neuron_test.cpp
  batch_iterator.cpp
  batch_iterator_test.cpp
  convolutional_filter.cpp
  convolutional_filter_test.cpp
  convolution_layer_factory.cpp
  data_cache.cpp
  data_reader.cpp
  dense_layer_factory.cpp
  dense_layer_proxy.cpp
  dropout_layer.cpp
  dropout_layer_factory.cpp
  dropout_layer_test.cpp
  epoch_indices.cpp
  epoch_indices_test.cpp
  feature_filter.cpp
  feature_layer_factory.cpp
  feature_layer_proxy.cpp
  hard_tanh.cpp
  hard_tanh_test.cpp
  hidden_layer.cpp
  hidden_layer_test.cpp
  hidden_neuron.cpp
  hidden_neuron_mock.cpp
  hidden_neuron_test.cpp
  identity.cpp
  identity_test.cpp
  image_cache.cpp
  inflation_layer.cpp
  inflation_layer_factory.cpp
  inflation_layer_test.cpp
  input_layer.cpp
  input_layer_test.cpp
  input_neuron.cpp
  input_neuron_test.cpp
  label_cache.cpp
  layer_mock.cpp
  max_neuron.cpp
  max_neuron_test.cpp
  mnist_test.cpp
  multi_channel_layer_factory.cpp
  multi_channel_layer_factory_test.cpp
  net_builder.cpp
  network.cpp
  network_test.cpp
  neuron_factory.cpp
  neuron_factory_mock.cpp
  neuron_mock.cpp
  ordinal_cursor.cpp
  ordinal_cursor_test.cpp
  pooling_filter.cpp
  pooling_filter_test.cpp
  pooling_layer_factory.cpp
  pooling_neuron_factory.cpp
  relu.cpp
  relu_test.cpp
  scalar.cpp
  scalar_neuron.cpp
  scalar_neuron_test.cpp
  scalar_test.cpp
  sigmoid.cpp
  sigmoid_test.cpp
  softmax.cpp
  softmax_layer.cpp
  softmax_layer_factory.cpp
  softmax_layer_test.cpp
  softmax_test.cpp
  stop_watch.cpp
  tangent.cpp
  tangent_test.cpp
  tensor_shape.cpp
  tensor_shape_test.cpp
  tensor_vector_matcher.cpp
  value_array_matcher.cpp
  value_factory.cpp
  value_factory_mock.cpp
  vector_allocator.cpp
  )

add_executable(CNN ${CNN_SRC})
target_link_libraries (CNN Catch2::Catch2WithMain ${CMAKE_THREAD_LIBS_INIT} tbb)
