#ifndef ACTIVITY_HPP
#define ACTIVITY_HPP

#include "rnn_type.hpp"

namespace rnn::generic {
  class activity {
  public:
    static void tanh(MatD& x);

    static void tanh(VecD& x);

    static MatD tanhPrime(const MatD& x);

    static VecD tanhPrime(const VecD& x);

    static real logistic(const real x);

    static void logistic(MatD& x);

    static void logistic(VecD& x);

    static MatD logisticPrime(const MatD& x);

    static VecD logisticPrime(const VecD& x);

    static void relu(VecD& x);

    static VecD reluPrime(const VecD& x);
  };

}

#endif
