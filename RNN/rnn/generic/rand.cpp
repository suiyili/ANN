#include "rand.hpp"

namespace rnn::generic {
unsigned long rand::next()
{
  this->t = (this->x ^ (this->x << 11));
  this->x = this->y;
  this->y = this->z;
  this->z = this->w;
  return (this->w = (this->w ^ (this->w >> 19)) ^ (this->t ^ (this->t >> 8)));
}

real rand::zero2one()
{
  return ((this->next() & 0xFFFF) + 1) / 65536.0;
}

void rand::uniform(MatD& mat, const real scale)
{
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) {
      mat.coeffRef(i, j) = 2.0 * this->zero2one() - 1.0;
    }
  }

  mat *= scale;
}

void rand::uniform(VecD& vec, const real scale)
{
  for (int i = 0; i < vec.rows(); ++i) {
    vec.coeffRef(i, 0) = 2.0 * this->zero2one() - 1.0;
  }

  vec *= scale;
}

real rand::gauss(real sigma, real mu)
{
  return
      mu +
          sigma *
              sqrt(-2.0 * log(this->zero2one())) *
              sin(2.0 * M_PI * this->zero2one());
}

void rand::gauss(MatD& mat, real sigma, real mu)
{
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) {
      mat.coeffRef(i, j) = this->gauss(sigma, mu);
    }
  }
}

void rand::setMask(VecD& mask, const real p)
{
  for (int i = 0; i < mask.rows(); ++i) {
    mask.coeffRef(i, 0) = (this->zero2one() < p ? 1.0 : 0.0);
  }
}

template<typename T>
void rand::shuffle(std::vector<T>& data)
{
  T tmp;

  for (int i = data.size(), a, b; i > 1; --i) {
    a = i - 1;
    b = this->next() % i;
    tmp = data[a];
    data[a] = data[b];
    data[b] = tmp;
  }
}
}