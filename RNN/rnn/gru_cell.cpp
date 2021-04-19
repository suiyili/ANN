#include "gru_cell.hpp"
#include "generic/activity.hpp"
#include "generic/utils.hpp"
#include <Eigen/SVD>

namespace rnn {
gru_cell::gru_cell(const int inputDim, const int hiddenDim) {
  this->Wxr = MatD(hiddenDim, inputDim);
  this->Whr = MatD(hiddenDim, hiddenDim);
  this->br = VecD::Zero(hiddenDim);

  this->Wxz = MatD(hiddenDim, inputDim);
  this->Whz = MatD(hiddenDim, hiddenDim);
  this->bz = VecD::Zero(hiddenDim);

  this->Wxu = MatD(hiddenDim, inputDim);
  this->Whu = MatD(hiddenDim, hiddenDim);
  this->bu = VecD::Zero(hiddenDim);
}

void gru_cell::init(rnn::generic::rand &rnd, const real scale) {
  rnd.uniform(this->Wxr, scale);
  rnd.uniform(this->Whr, scale);

  rnd.uniform(this->Wxz, scale);
  rnd.uniform(this->Whz, scale);

  rnd.uniform(this->Wxu, scale);
  rnd.uniform(this->Whu, scale);

  this->Whr = Eigen::JacobiSVD<MatD>(this->Whr, Eigen::ComputeFullV |
      Eigen::ComputeFullU).matrixU();
  this->Whz = Eigen::JacobiSVD<MatD>(this->Whz, Eigen::ComputeFullV |
      Eigen::ComputeFullU).matrixU();
  this->Whu = Eigen::JacobiSVD<MatD>(this->Whu, Eigen::ComputeFullV |
      Eigen::ComputeFullU).matrixU();
}
void gru_cell::forward(const VecD &xt, const gru_cell::State *prev, gru_cell::State *cur) {

  cur->r = this->br + this->Wxr * xt + this->Whr * prev->h;
  cur->z = this->bz + this->Wxz * xt + this->Whz * prev->h;

  activity::logistic(cur->r);
  activity::logistic(cur->z);

  cur->rh = cur->r.array() * prev->h.array();
  cur->u = this->bu + this->Wxu * xt + this->Whu * cur->rh;
  activity::tanh(cur->u);
  cur->h = (1.0 - cur->z.array()) * prev->h.array() +
      cur->z.array() * cur->u.array();
}

void gru_cell::backward(gru_cell::State *prev, gru_cell::State *cur, gru_cell::Grad &grad,
                        const VecD &xt) {
  VecD delr, delz, delu, delrh;

  delz = activity::logisticPrime(cur->z).array() * cur->delh.array() *
      (cur->u - prev->h).array();
  delu =
      activity::tanhPrime(cur->u).array() * cur->delh.array() * cur->z.array();
  delrh = this->Whu.transpose() * delu;
  delr =
      activity::logisticPrime(cur->r).array() * delrh.array() * prev->h.array();

  cur->delx =
      this->Wxr.transpose() * delr +
          this->Wxz.transpose() * delz +
          this->Wxu.transpose() * delu;

  prev->delh.noalias() +=
      this->Whr.transpose() * delr +
          this->Whz.transpose() * delz;
  prev->delh.array() +=
      delrh.array() * cur->r.array() +
          cur->delh.array() * (1.0 - cur->z.array());

  grad.Wxr.noalias() += delr * xt.transpose();
  grad.Whr.noalias() += delr * prev->h.transpose();

  grad.Wxz.noalias() += delz * xt.transpose();
  grad.Whz.noalias() += delz * prev->h.transpose();

  grad.Wxu.noalias() += delu * xt.transpose();
  grad.Whu.noalias() += delu * cur->rh.transpose();

  grad.br += delr;
  grad.bz += delz;
  grad.bu += delu;
}

void gru_cell::sgd(const gru_cell::Grad &grad, const real learningRate) {
  this->Wxr -= learningRate * grad.Wxr;
  this->Whr -= learningRate * grad.Whr;
  this->br -= learningRate * grad.br;

  this->Wxz -= learningRate * grad.Wxz;
  this->Whz -= learningRate * grad.Whz;
  this->bz -= learningRate * grad.bz;

  this->Wxu -= learningRate * grad.Wxu;
  this->Whu -= learningRate * grad.Whu;
  this->bu -= learningRate * grad.bu;
}

void gru_cell::save(std::ofstream &ofs) {
  rnn::generic::save(ofs, this->Wxr);
  rnn::generic::save(ofs, this->Whr);
  rnn::generic::save(ofs, this->br);
  rnn::generic::save(ofs, this->Wxz);
  rnn::generic::save(ofs, this->Whz);
  rnn::generic::save(ofs, this->bz);
  rnn::generic::save(ofs, this->Wxu);
  rnn::generic::save(ofs, this->Whu);
  rnn::generic::save(ofs, this->bu);
}

void gru_cell::load(std::ifstream &ifs) {
  rnn::generic::load(ifs, this->Wxr);
  rnn::generic::load(ifs, this->Whr);
  rnn::generic::load(ifs, this->br);
  rnn::generic::load(ifs, this->Wxz);
  rnn::generic::load(ifs, this->Whz);
  rnn::generic::load(ifs, this->bz);
  rnn::generic::load(ifs, this->Wxu);
  rnn::generic::load(ifs, this->Whu);
  rnn::generic::load(ifs, this->bu);
}

void gru_cell::State::clear() {
  this->h = VecD();
  this->u = VecD();
  this->r = VecD();
  this->z = VecD();
  this->rh = VecD();
  this->delh = VecD();
  this->delx = VecD();
}

gru_cell::Grad::Grad(const gru_cell &gru) {
  this->Wxr = MatD::Zero(gru.Wxr.rows(), gru.Wxr.cols());
  this->Whr = MatD::Zero(gru.Whr.rows(), gru.Whr.cols());
  this->br = VecD::Zero(gru.br.rows());

  this->Wxz = MatD::Zero(gru.Wxz.rows(), gru.Wxz.cols());
  this->Whz = MatD::Zero(gru.Whz.rows(), gru.Whz.cols());
  this->bz = VecD::Zero(gru.bz.rows());

  this->Wxu = MatD::Zero(gru.Wxu.rows(), gru.Wxu.cols());
  this->Whu = MatD::Zero(gru.Whu.rows(), gru.Whu.cols());
  this->bu = VecD::Zero(gru.bu.rows());
};

void gru_cell::Grad::init() {
  this->Wxr.setZero();
  this->Whr.setZero();
  this->br.setZero();
  this->Wxz.setZero();
  this->Whz.setZero();
  this->bz.setZero();
  this->Wxu.setZero();
  this->Whu.setZero();
  this->bu.setZero();
}

real gru_cell::Grad::norm() {
  return
      this->Wxr.squaredNorm() + this->Whr.squaredNorm() +
          this->br.squaredNorm() +
          this->Wxz.squaredNorm() + this->Whz.squaredNorm() +
          this->bz.squaredNorm() +
          this->Wxu.squaredNorm() + this->Whu.squaredNorm() +
          this->bu.squaredNorm();
}

void gru_cell::Grad::operator+=(const gru_cell::Grad &grad) {
  this->Wxr += grad.Wxr;
  this->Whr += grad.Whr;
  this->br += grad.br;
  this->Wxz += grad.Wxz;
  this->Whz += grad.Whz;
  this->bz += grad.bz;
  this->Wxu += grad.Wxu;
  this->Whu += grad.Whu;
  this->bu += grad.bu;
}
}