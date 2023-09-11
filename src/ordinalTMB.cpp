// Copyright (c) 2023 Kinh Nguyen

// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include <TMB.hpp>
#include <ordinal.hpp>

// Only for logit link
template <class Type>
Type objective_function<Type>::operator()() {
  Type target = 0.0;

  DATA_MATRIX(Y); // N_SAMPLES x N_QUESTIONS
  DATA_MATRIX(X); // N_SAMPLES x P
  DATA_SPARSE_MATRIX(R); // N_SAMPLES x n_id
  int N_SAMPLES = Y.rows(), N_QUESTIONS = Y.cols();

  // DATA_INTEGER(shrinkage);
  
  PARAMETER_VECTOR(pi_norm); // N_RESPONSES - 1 x N_QUESTIONS, 
  target -= dnorm(pi_norm, Type(0), Type(1), true).sum();
  
  PARAMETER_VECTOR(beta); // shared between outcomes
  target -= dnorm(beta, Type(0), Type(1), true).sum();

  PARAMETER_VECTOR(iid);
  PARAMETER(sd_log);
  Type sd = exp(sd_log);

  if (shrinkage)
    target -= dnorm(sd, Type(0), Type(1), true) + sd_log; // half normal
  else
    target -= dnorm(sd_log, Type(0), Type(1e6), true) + sd_log; // log normal - this equals clmm

  PARAMETER_VECTOR(rhos); // cov() / sig1 * sig2 = ((Q*Q) - Q)/2
  target -= dnorm(rhos, Type(0), Type(1), true).sum();
  vector<Type> rhos_scale(rhos.size());
  for (int i = 0; i < rhos.size(); ++i)
    rhos_scale[i] = (exp(2.0 * rhos[i]) - 1.0) / (exp(2.0 * rhos[i]) + 1.0);

  vector<Type> eta = X * beta;
  vector<Type> ide = R * iid;
  eta += ide;
  
  // Composite likelihood
  vector<Type> ll(N_SAMPLES);
  ll.setZero();
  for (int n=0; n < N_SAMPLES; n++) {
    int rho_id = 0; // unstructured correlation
    for (int q = 0; q < N_QUESTIONS - 1; ++q) { // double loop through outcomes
      for (int p = q+1; p < N_QUESTIONS; ++p) { // double loop through outcomes
        Type y1 = Y(n, q), y2 = Y(n, p); // current raw responses
        int shift_q = q * 4, shift_p = p * 4;
        Type c_rho = rhos_scale[rho_id]; // current correllation
        ++rho_id; // increase here in case of skipping
        ll[n] = katomic::BVN(y1, y2, // data
          pi_norm[shift_q], pi_norm[shift_q + 1], pi_norm[shift_q + 2], pi_norm[shift_q + 3], 
          pi_norm[shift_p], pi_norm[shift_p + 1], pi_norm[shift_p + 2], pi_norm[shift_p + 3],
          eta[n], c_rho);
        target -= log(ll[n] + FLT_MIN);
      } // end 2D
    } // end individual
  } // end data


  matrix<Type> cutpoints(4, N_QUESTIONS);
  for (int q = 0; q < N_QUESTIONS; ++q) {
    vector<Type> c_pi = pi_norm(Eigen::seqN(q * 4, 4)), // n_responses - 1
      pi = simplex_transform(c_pi); // n_responses
    cutpoints.col(q) = make_cutpoints(pi, 2); // n_responses - 1
  }
  REPORT(cutpoints)
  REPORT(ll);
  REPORT(rhos_scale);
  REPORT(target);

  return target;
}
