#include "activity.hpp"

namespace rnn::generic {

//f(x) = tanh(x)
#ifdef USE_EIGEN_TANH
void activity::tanh(MatD& x){
    x = x.array().tanh();
  }
  inline void activity::tanh(VecD& x){
    x = x.array().tanh();
  }
#else

void activity::tanh(MatD& x)
{
  x = x.unaryExpr(std::ptr_fun(::tanh));
}

void activity::tanh(VecD& x)
{
  x = x.unaryExpr(std::ptr_fun(::tanh));
}

#endif

//f'(x) = 1-(f(x))^2
MatD activity::tanhPrime(const MatD& x)
{
  return 1.0 - x.array().square();
}

VecD activity::tanhPrime(const VecD& x)
{
  return 1.0 - x.array().square();
}

//f(x) = sigm(x)
real activity::logistic(const real x)
{
  return 1.0 / (1.0 + ::exp(-x));
}

void activity::logistic(MatD& x)
{
  x = x.unaryExpr(std::ptr_fun((real (*)(const real)) activity::logistic));
}

void activity::logistic(VecD& x)
{
  x = x.unaryExpr(std::ptr_fun((real (*)(const real)) activity::logistic));
}

//f'(x) = f(x)(1-f(x))
MatD activity::logisticPrime(const MatD& x)
{
  return x.array() * (1.0 - x.array());
}

VecD activity::logisticPrime(const VecD& x)
{
  return x.array() * (1.0 - x.array());
}

//ReLu
void activity::relu(VecD& x)
{
  for (unsigned int i = 0; i < x.rows(); ++i) {
    if (x.coeff(i, 0) <= 0.0) {
      x.coeffRef(i, 0) = 0.0;
    }
  }
}

VecD activity::reluPrime(const VecD& x)
{
  VecD res = x;
  for (unsigned int i = 0; i < res.rows(); ++i) {
    if (res.coeff(i, 0) <= 0.0) {
      res.coeffRef(i, 0) = 0.0;
    } else {
      res.coeffRef(i, 0) = 1.0;
    }
  }
  return res;
}

}

