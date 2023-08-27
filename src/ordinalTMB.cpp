#include <TMB.hpp>
#include <ordinal.hpp>

// Only for logit link
template <class Type>
Type objective_function<Type>::operator()() {
  Type target = 0.0;

  DATA_VECTOR(y);
  DATA_MATRIX(X);
  
  PARAMETER_VECTOR(pi_norm); // length J-1, then transform to simplex J
  vector<Type> 
    pi = simplex_transform(pi_norm), // length J
    cutpoints = make_cutpoints(pi);
  
  PARAMETER_VECTOR(beta);
  target -= dnorm(beta, Type(0), Type(1), true).sum();

  vector<Type> 
    eta = X * beta, 
    pwll = pw_polr(y, eta, cutpoints);

  target -= pwll.sum();

  REPORT(cutpoints);

  return target;
}
