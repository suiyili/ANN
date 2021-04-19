#ifndef GRU_CELL_HPP
#define GRU_CELL_HPP

#include "generic/rnn_type.hpp"
#include "generic/rand.hpp"
#include <fstream>

using namespace rnn::generic;

namespace rnn {
  class gru_cell final {
  public:
    gru_cell()
    {};

    gru_cell(int inputDim, int hiddenDim);

    class State;

    class Grad;

    MatD Wxr, Whr;
    VecD br;
    MatD Wxz, Whz;
    VecD bz;
    MatD Wxu, Whu;
    VecD bu;

    void init(rnn::generic::rand& rnd, real scale = 1.0);

    void forward(const VecD& xt, const State* prev, State* cur);

    void backward(gru_cell::State* prev, gru_cell::State* cur, gru_cell::Grad& grad,
                          const VecD& xt);

    void sgd(const gru_cell::Grad& grad, const real learningRate);

    void save(std::ofstream& ofs);

    void load(std::ifstream& ifs);
  };

  class gru_cell::State {
  public:
    VecD h, u, r, z;
    VecD rh;

    VecD delh, delx; //for backprop

    void clear();
  };

  class gru_cell::Grad {
  public:
    Grad()
    {}

    Grad(const gru_cell& gru);

    MatD Wxr, Whr;
    VecD br;
    MatD Wxz, Whz;
    VecD bz;
    MatD Wxu, Whu;
    VecD bu;

    void init();

    real norm();

    void operator+=(const gru_cell::Grad& grad);
  };
}

#endif