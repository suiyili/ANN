#ifndef RAND_HPP
#define RAND_HPP

#include "rnn_type.hpp"
#include <vector>

namespace rnn::generic {
  class rand {
  public:
    static rand r_;

    rand(unsigned long w_ = 88675123) :
      x(123456789), y(362436069), z(521288629), w(w_)
    {};

    unsigned long next();

    real zero2one();

    void uniform(MatD& mat, const real scale = 1.0);

    void uniform(VecD& vec, const real scale = 1.0);

    real gauss(real sigma, real mu = 0.0);

    void gauss(MatD& mat, real sigma, real mu = 0.0);

    void setMask(VecD& mask, const real p = 0.5);

    template<typename T>
    void shuffle(std::vector<T>& data);

  private:
    unsigned long x;
    unsigned long y;
    unsigned long z;
    unsigned long w;
    unsigned long t; //tmp
  };


}

#endif