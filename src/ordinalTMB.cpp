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

  DATA_VECTOR(y);
  DATA_MATRIX(X);
  DATA_SPARSE_MATRIX(R);
  DATA_INTEGER(link);
  DATA_INTEGER(shrinkage);
  
  PARAMETER_VECTOR(pi_norm); // length J-1, then transform to simplex J
  vector<Type> 
    pi = simplex_transform(pi_norm), // length J
    cutpoints = make_cutpoints(pi, link);
  
  PARAMETER_VECTOR(beta);
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

  vector<Type> eta = X * beta;
  vector<Type> ide = R * iid;
  eta += ide;

  vector<Type> pwll = pw_polr(y, eta, cutpoints, link);

  target -= pwll.sum();

  REPORT(cutpoints);
  vector<Type> OR = exp(beta);
  REPORT(OR);

  return target;
}
