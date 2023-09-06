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

  DATA_IMATRIX(Y); // N x Q
  DATA_MATRIX(X); // N x P
  DATA_SPARSE_MATRIX(R); // N x n_id
  int N = Y.rows(), Q = Y.cols();

  DATA_INTEGER(link);
  DATA_INTEGER(shrinkage);
  
  PARAMETER_MATRIX(pi_norm); // J - 1 x Q, length J-1, then transform to simplex J
  int J = pi_norm.rows() + 1;
  matrix<Type> cutpoints(J - 1, Q);
  for (int q = 0; q < Q; ++q) {
    vector<Type> 
    c_pi = pi_norm.col(q), // J - 1
    pi = simplex_transform(c_pi); // J
    cutpoints.col(q) = make_cutpoints(pi, link); // J - 1
  }
  
  PARAMETER_VECTOR(beta); // shared between outcomes
  target -= dnorm(beta, Type(0), Type(1), true).sum();

  PARAMETER_VECTOR(iid);
  PARAMETER(sd_log);
  Type sd = exp(sd_log);

  if (shrinkage) { 
    target -= dnorm(sd, Type(0), Type(1), true) + sd_log; // half normal
    target -= dnorm(iid, Type(0), sd, true).sum();
  } else {
    target -= dnorm(sd_log, Type(0), Type(1e6), true) + sd_log; // log normal - this equals clmm
    target -= dnorm(iid, Type(0), sd, true).sum();
  }

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
  vector<Type> ll(N);
  vector<Type> lower(2), upper(2); // change in loop
  vector<int> infin(2); // change in loop
  for (int n=0; n < N; n++) {
    Type shift = eta(n);
    int rho_id = 0; // unstructured correlation
    for (int q = 0; q < Q - 1; ++q) { // double loop through outcomes
      for (int p = q+1; p < Q; ++p) { // double loop through outcomes
        int Yq = Y(n, q), Yp = Y(n, p); // current raw responses
        Type c_rho = rhos_scale[rho_id]; // current correllation
        ++rho_id; // increase here in case of skipping
        if (std::isnan(Yq) | std::isnan(Yp)) continue;
        if (Yq == 1) { // dimension 1 prep for bivariate
          lower(0) = 0; // actually infty - see infin
          upper(0) = cutpoints(Yq - 1, q) - shift;
          infin(0) = 0;
        } else if ((Yq > 1) & (Yq < Q)) {
          lower(0) = cutpoints(Yq - 1 - 1, q) - shift;
          upper(0) = cutpoints(Yq - 1, q) - shift;
          infin(0) = 2;              
        } else if (Yq == Q) {
          lower(0) = cutpoints(Yq - 1 - 1, q) - shift;
          upper(0) = 0; // actually infty - see infin
          infin(0) = 1;
        }        
        if (Yp == 1) { // dimension 2 prep for bivariate
          lower(1) = 0; // actually infty - see infin
          upper(1) = cutpoints(Yp - 1, p) - shift;
          infin(1) = 0;
        } else if ((Yp > 1) & (Yp < Q)) {
          lower(1) = cutpoints(Yp - 1 - 1, p) - shift;
          upper(1) = cutpoints(Yp - 1, p) - shift;
          infin(1) = 2;              
        } else if (Yp == Q) {
          lower(1) = cutpoints(Yp - 1 - 1, p) - shift;
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
