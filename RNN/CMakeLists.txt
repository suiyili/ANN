cmake_minimum_required(VERSION 3.12)
project(RNN)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_PREFIX_PATH  ../Catch2 ../Eigen)
add_compile_definitions(TEST)

find_package (Threads)
find_package(Catch2 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(rnn)

add_executable(RNN rnn_test.cpp)
target_link_libraries(RNN PRIVATE RNNLib RNNTest)