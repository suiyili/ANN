#ifndef RNN_TYPE_HPP
#define RNN_TYPE_HPP

#include <Eigen/Core>

namespace rnn::generic {

#ifdef USE_FLOAT
typedef float real;
typedef Eigen::MatrixXf MatD;
typedef Eigen::VectorXf VecD;
#else
typedef double real;
typedef Eigen::MatrixXd MatD;
typedef Eigen::VectorXd VecD;
#endif

typedef Eigen::MatrixXi MatI;
typedef Eigen::VectorXi VecI;
#define REAL_MAX std::numeric_limits<real>::max()

} // namespace rnn

#endif