#ifndef UTILS_HPP
#define UTILS_HPP

#include "rnn_type.hpp"
#include <string>
#include <vector>
#include <fstream>

namespace rnn::generic {
real max(const real &x, const real &y);

real min(const real &x, const real &y);

bool isSpace(const char &c);

void split(const std::string &str, std::vector<std::string> &res);

void
split(const std::string &str, std::vector<std::string> &res, const char sep);


real cosDis(const MatD &a, const MatD &b);

void save(std::ofstream &ofs, const MatD &params);

void save(std::ofstream &ofs, const VecD &params);

void load(std::ifstream &ifs, MatD &params);

void load(std::ifstream &ifs, VecD &params);

real stdDev(const Eigen::MatrixXd &input);
}

#endif