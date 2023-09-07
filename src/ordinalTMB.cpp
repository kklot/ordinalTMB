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

  DATA_IMATRIX(Y); // n_samples x n_questions
  DATA_MATRIX(X); // n_samples x P
  DATA_SPARSE_MATRIX(R); // n_samples x n_id
  int n_samples = Y.rows(), n_questions = Y.cols();

  DATA_INTEGER(link);
  DATA_INTEGER(shrinkage);
  
  PARAMETER_MATRIX(pi_norm); // n_responses - 1 x n_questions, length J-1, then transform to simplex J
  int n_responses = pi_norm.rows() + 1;
  matrix<Type> cutpoints(n_responses - 1, n_questions);
  for (int q = 0; q < n_questions; ++q) {
    vector<Type> 
      c_pi = pi_norm.col(q), // n_responses - 1
      pi = simplex_transform(c_pi); // n_responses
    cutpoints.col(q) = make_cutpoints(pi, link); // n_responses - 1
  }
  
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
  vector<Type> rhos_scale(rhos.size());
  for (int i = 0; i < rhos.size(); ++i) {
    rhos_scale[i] = theta_to_rho(rhos[i]);
    target -= beta_correlation_lpdf(rhos[i], Type(2.0), Type(2.0));
  }

  vector<Type> eta = X * beta;
  vector<Type> ide = R * iid;
  eta += ide;
  
  // Composite likelihood
  vector<Type> ll(n_samples);
  vector<Type> lower(2), upper(2); // change in loop
  vector<int> infin(2); // change in loop
  for (int n=0; n < n_samples; n++) {
    int rho_id = 0; // unstructured correlation
    for (int q = 0; q < n_questions - 1; ++q) { // double loop through outcomes
      for (int p = q+1; p < n_questions; ++p) { // double loop through outcomes
        int Yq = Y(n, q), Yp = Y(n, p); // current raw responses
        Type c_rho = rhos_scale[rho_id]; // current correllation
        ++rho_id; // increase here in case of skipping
        if (std::isnan(Yq) | std::isnan(Yp)) continue;
        if (Yq == 1) { // dimension 1 prep for bivariate
          lower(0) = 0; // actually infty - see infin
          upper(0) = cutpoints(Yq - 1, q) - eta[n];
          infin(0) = 0;
        } else if ((Yq > 1) & (Yq < n_responses)) {
          lower(0) = cutpoints(Yq - 1 - 1, q) - eta[n];
          upper(0) = cutpoints(Yq - 1, q) - eta[n];
          infin(0) = 2;              
        } else if (Yq == n_responses) {
          lower(0) = cutpoints(Yq - 1 - 1, q) - eta[n];
          upper(0) = 0; // actually infty - see infin
          infin(0) = 1;
        }        
        if (Yp == 1) { // dimension 2 prep for bivariate
          lower(1) = 0; // actually infty - see infin
          upper(1) = cutpoints(Yp - 1, p) - eta[n];
          infin(1) = 0;
        } else if ((Yp > 1) & (Yp < n_responses)) {
          lower(1) = cutpoints(Yp - 1 - 1, p) - eta[n];
          upper(1) = cutpoints(Yp - 1, p) - eta[n];
          infin(1) = 2;              
        } else if (Yp == n_responses) {
          lower(1) = cutpoints(Yp - 1 - 1, p) - eta[n];
          upper(1) = 0; // actually infty - see infin
          infin(1) = 1;
        }
        double lli = mvbvn_(&lower(0), &upper(0), &infin(0), &c_rho);
        ll[n] = log(lli);
      } // end 2D
    } // end individual
  } // end data

  target -= ll.sum();

  REPORT(cutpoints);
  REPORT(ll);
  REPORT(rhos_scale);

  return target;
}
