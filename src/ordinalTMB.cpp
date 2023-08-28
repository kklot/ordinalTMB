#include <TMB.hpp>
#include <ordinal.hpp>

// Only for logit link
template <class Type>
Type objective_function<Type>::operator()() {
  Type target = 0.0;

  DATA_VECTOR(y);
  DATA_MATRIX(X);
  DATA_SPARSE_MATRIX(R);
  
  PARAMETER_VECTOR(pi_norm); // length J-1, then transform to simplex J
  vector<Type> 
    pi = simplex_transform(pi_norm), // length J
    cutpoints = make_cutpoints(pi);
  
  PARAMETER_VECTOR(beta);
  target -= dnorm(beta, Type(0), Type(1), true).sum();

  PARAMETER_VECTOR(iid);
  PARAMETER(sd_log);
  Type sd = exp(sd_log);
  target -= dnorm(sd, Type(0), Type(1), true) + sd_log; // half normal
  target -= dnorm(iid, Type(0), sd, true).sum();

  vector<Type> eta = X * beta;
  vector<Type> ide = R * iid;
  eta += ide;
  vector<Type> pwll = pw_polr(y, eta, cutpoints);

  target -= pwll.sum();

  REPORT(cutpoints);

  return target;
}
