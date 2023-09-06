#include <TMB.hpp>
#include <ordinal.hpp>

extern "C" {
  double F77_NAME(mvbvn)(double *lower, double *upper, int *infin, double *correl);
  const static R_CallMethodDef R_CallDef[] = {
    TMB_CALLDEFS,
    {"mvbvn", (DL_FUNC) &F77_NAME(mvbvn), 4},
    {NULL, NULL, 0}
  };
  void R_init_ordinalTMB(DllInfo *dll) 
  {
    R_registerRoutines(dll, NULL, R_CallDef, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    TMB_CCALLABLES("ordinalTMB");
  }
}

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
        Type c_rho = rhos[rho_id]; // current correllation
        int Yq = Y(n, q), Yp = Y(n, p); // current raw responses
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
        ++rho_id;
      } // end 2D
    } // end individual
  } // end data

  target -= ll.sum();

  REPORT(cutpoints);
  REPORT(ll);

  return target;
}
