set(GENERIC_SRC
    rand.hpp
    rand.cpp
    activity.hpp
    utils.hpp
    rnn_type.hpp
    activity.cpp
    utils.cpp)
add_library(GENERICLib STATIC ${GENERIC_SRC})
target_link_libraries(GENERICLib PRIVATE Eigen3::Eigen)