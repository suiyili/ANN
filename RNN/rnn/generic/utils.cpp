#include "utils.hpp"

namespace rnn::generic {
real max(const real &x, const real &y) {
  return x > y ? x : y;
}

real min(const real &x, const real &y) {
  return x > y ? y : x;
}

bool isSpace(const char &c) {
  return (c == ' ' || c == '\t');
}

void split(const std::string &str, std::vector<std::string> &res) {
  bool tok = false;
  int beg = 0;

  res.clear();

  for (int i = 0, len = str.length(); i < len; ++i) {
    if (!tok && !isSpace(str[i])) {
      beg = i;
      tok = true;
    }

    if (tok && (i == len - 1 || isSpace(str[i]))) {
      tok = false;
      res.push_back(
          isSpace(str[i]) ? str.substr(beg, i - beg) : str.substr(beg,
                                                                  i - beg + 1));
    }
  }
}

void
split(const std::string &str, std::vector<std::string> &res, const char sep) {
  bool tok = false;
  int beg = 0;

  res.clear();

  for (int i = 0, len = str.length(); i < len; ++i) {
    if (!tok && str[i] != sep) {
      beg = i;
      tok = true;
    }

    if (tok && (i == len - 1 || str[i] == sep)) {
      tok = false;
      res.push_back(
          (str[i] == sep) ? str.substr(beg, i - beg) : str.substr(beg,
                                                                  i - beg + 1));
    }
  }
}

real cosDis(const MatD &a, const MatD &b) {
  return (a.array() * b.array()).sum() / (a.norm() * b.norm());
  //return a.col(0).dot(b.col(0))/(a.norm()*b.norm());
}

void save(std::ofstream &ofs, const MatD &params) {
  real val = 0.0;

  for (int i = 0; i < params.cols(); ++i) {
    for (int j = 0; j < params.rows(); ++j) {
      val = params.coeff(j, i);
      ofs.write((char *) &val, sizeof(real));
    }
  }
}

void save(std::ofstream &ofs, const VecD &params) {
  real val = 0.0;

  for (int i = 0; i < params.rows(); ++i) {
    val = params.coeff(i, 0);
    ofs.write((char *) &val, sizeof(real));
  }
}

void load(std::ifstream &ifs, MatD &params) {
  real val = 0.0;

  for (int i = 0; i < params.cols(); ++i) {
    for (int j = 0; j < params.rows(); ++j) {
      ifs.read((char *) &val, sizeof(real));
      params.coeffRef(j, i) = val;
    }
  }
}

void load(std::ifstream &ifs, VecD &params) {
  real val = 0.0;

  for (int i = 0; i < params.rows(); ++i) {
    ifs.read((char *) &val, sizeof(real));
    params.coeffRef(i, 0) = val;
  }
}

real stdDev(const Eigen::MatrixXd &input) {
  return ::sqrt(
      ((Eigen::MatrixXd) ((input.array() - input.sum() / input.rows()).pow(
          2.0))).sum() / (input.rows() - 1));
}

}

