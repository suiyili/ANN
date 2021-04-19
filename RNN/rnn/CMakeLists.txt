add_subdirectory(generic)

set(RNN_SRC end_io_net.cpp
    gru_cell.cpp
  predict_end_net.cpp
    predict_end_net.hpp
    predict_sequence_net.cpp
    predict_sequence_net.hpp
    )
add_library(RNNLib ${RNN_SRC})
target_link_libraries(RNNLib GENERICLib Eigen3::Eigen)

set(RNN_TEST_SRC
  predict_end_net_test.cpp
    predict_sequence_net_test.cpp)
add_library(RNNTest ${RNN_TEST_SRC})
target_link_libraries(RNNTest RNNLib Catch2::Catch2WithMain)