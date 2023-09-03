#include <TMB.hpp>
#include <ordinal.hpp>

// Only for logit link
template <class Type>
Type objective_function<Type>::operator()() {
  Type target = 0.0;

  DATA_VECTOR(y);
  DATA_MATRIX(X);
  DATA_SPARSE_MATRIX(R);
  DATA_INTEGER(shrinkage);
  
  PARAMETER_VECTOR(pi_norm); // length J-1, then transform to simplex J
  vector<Type> 
    pi = simplex_transform(pi_norm), // length J
    cutpoints = make_cutpoints(pi);
  
  PARAMETER_VECTOR(beta);
  target -= dnorm(beta, Type(0), Type(1), true).sum();

  PARAMETER_VECTOR(iid);
  PARAMETER(sd_log);

  if (shrinkage) { 
    Type sd = exp(sd_log);
    target -= dnorm(sd, Type(0), Type(1), true) + sd_log; // half normal
    target -= dnorm(iid, Type(0), sd, true).sum();
  } else {
    target -= dnorm(sd_log, Type(0), Type(1e6), true) + sd_log; // log normal - this equals clmm
  }

  vector<Type> eta = X * beta;
  vector<Type> ide = R * iid;
  eta += ide;
  vector<Type> pwll = pw_polr(y, eta, cutpoints);

  target -= pwll.sum();

  REPORT(cutpoints);
  vector<Type> OR = exp(beta);
  REPORT(OR);

  return target;
}
